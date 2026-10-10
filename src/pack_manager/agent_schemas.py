"""Pydantic data schemas for Pack Manager AI.

Includes:
- STAGE 1: Visual Observation Schemas (Grounded in image, free from order confirmation bias)
- Canonical output payload schemas (PackManagerAIResponse)
"""

from __future__ import annotations
from enum import Enum

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ==============================================================================
# STAGE 1: VISUAL OBSERVATION SCHEMAS
# ==============================================================================

class BoundingBox2D(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] relative to image dimensions.
    Deterministically converts Gemini's known 0-1000 coordinate format to normalized 0-1 scale.
    Strict safety: Rejects malformed bounding boxes if coordinates are inverted (does not guess,
    swap, or infer width, preserving genuine spatial evidence).
    """
    ymin: float = Field(ge=0.0, le=1.0, description="Top coordinate (0.0 to 1.0)")
    xmin: float = Field(ge=0.0, le=1.0, description="Left coordinate (0.0 to 1.0)")
    ymax: float = Field(ge=0.0, le=1.0, description="Bottom coordinate (0.0 to 1.0)")
    xmax: float = Field(ge=0.0, le=1.0, description="Right coordinate (0.0 to 1.0)")
    is_valid: bool = Field(default=True, description="Spatial evidence validity flag")

    @model_validator(mode="before")
    @classmethod
    def normalize_scale(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            keys = ["ymin", "xmin", "ymax", "xmax"]
            # Deterministic conversion from known 0-1000 coordinate format to 0.0-1.0
            if any(isinstance(data.get(k), (int, float)) and data.get(k) > 1.0 for k in keys):
                for k in keys:
                    if k in data and isinstance(data[k], (int, float)):
                        data[k] = round(min(1.0, max(0.0, float(data[k]) / 1000.0)), 4)
        return data

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox2D:
        if self.ymax < self.ymin:
            raise ValueError(f"Malformed bounding box: ymax ({self.ymax}) < ymin ({self.ymin}). Spatial evidence rejected.")
        if self.xmax < self.xmin:
            raise ValueError(f"Malformed bounding box: xmax ({self.xmax}) < xmin ({self.xmin}). Spatial evidence rejected.")
        return self


class OcclusionDetail(BaseModel):
    """Occlusion status and visibility assessment of a physical item."""
    is_occluded: bool = False
    occlusion_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated fraction obscured [0.0 - 1.0]")
    notes: Optional[str] = None


class ObservedPhysicalItem(BaseModel):
    """A single discrete physical product or object observed in the open package."""
    object_id: str = Field(description="Unique identifier for this physical observation instance, e.g. 'obj-1'")
    category: Literal["product", "foreign_object", "documentation", "unidentified"] = "product"
    label: str = Field(description="Descriptive visual label, e.g. 'Folded T-Shirt', 'Baseball Cap'")
    visual_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Observed visual traits, e.g. {'color': 'black', 'text': 'M 100% COTTON', 'variant': 'black'}"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Detection and classification confidence score [0.0 - 1.0]")
    bbox: BoundingBox2D = Field(description="Normalized bounding box enclosing this physical item")
    occlusion: OcclusionDetail = Field(default_factory=OcclusionDetail)
    ambiguity_notes: Optional[str] = None
    image_id: str = Field(default="img-0", description="Identifier of the source photograph")


class ContainerObservation(BaseModel):
    """Observation of the shipping container / carton itself."""
    container_type: str = Field(default="cardboard_box", description="e.g. 'cardboard_box', 'poly_mailer', 'tote'")
    visible_damage: bool = False
    damage_notes: Optional[str] = None
    bbox: Optional[BoundingBox2D] = None


class PackageVisualObservation(BaseModel):
    """STAGE 1 OUTPUT: Pure visual observations from package photograph(s).

    Strictly free from order confirmation bias: does NOT contain expected counts or shipping decisions.
    """
    image_id: str = Field(default="box-photo-0")
    image_quality: Literal["clear", "blurry", "low_light", "glare", "obstructed"] = "clear"
    clarity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Visual clarity [0.0 - 1.0]")
    container: ContainerObservation = Field(default_factory=ContainerObservation)
    observed_items: list[ObservedPhysicalItem] = Field(
        default_factory=list,
        description="One detection entry per visible physical unit"
    )
    ambiguity_flags: list[str] = Field(
        default_factory=list,
        description="Observations where visibility, lighting, or overlap prevents certainty"
    )

    def total_observed_products(self) -> int:
        return len([it for it in self.observed_items if it.category == "product"])


# ==============================================================================


# ==============================================================================
# STAGE 2: SKU / CATALOGUE RESOLUTION SCHEMAS
# ==============================================================================

class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class CandidateSKU(BaseModel):
    sku: str
    score: float = Field(ge=0.0, le=1.0, description="Match score [0.0 - 1.0]")
    product_name: Optional[str] = None
    asin: Optional[str] = None
    compatibility_reasons: list[str] = Field(
        default_factory=list,
        description="Specific attributes matching or confirmed (e.g. 'color: blue', 'category: cap')"
    )


class ResolvedObjectItem(BaseModel):
    """Result of SKU catalogue resolution for a single observed physical unit."""
    object_id: str = Field(description="Corresponding Stage 1 object_id, e.g. 'obj-1'")
    resolved_sku: Optional[str] = Field(
        default=None,
        description="Populated ONLY when status is RESOLVED. Must be None for AMBIGUOUS or UNRESOLVED."
    )
    status: ResolutionStatus
    semantic_type: Literal["product", "foreign_object", "documentation", "unidentified"] = Field(
        default="product",
        description="Preserved Stage 1 semantic category (distinguishes foreign object from unknown product)"
    )
    unresolved_reason: Optional[str] = Field(
        default=None,
        description="Categorization of unresolved state: 'FOREIGN_OBJECT', 'UNKNOWN_PRODUCT', 'AMBIGUOUS_COMPETITORS', 'LOW_PERCEPTION_CONFIDENCE', 'SEVERE_OCCLUSION'"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Match/resolution confidence score")
    candidate_skus: list[CandidateSKU] = Field(default_factory=list)
    matching_evidence: list[str] = Field(
        default_factory=list,
        description="Observable evidence used to match, e.g. ['Observed color is blue', 'Observed form is baseball cap']"
    )
    ambiguity_reason: Optional[str] = Field(
        default=None,
        description="Explanation if status is AMBIGUOUS or UNRESOLVED"
    )
    bbox: BoundingBox2D = Field(description="Bounding box from Stage 1 observation")
    image_id: str = Field(default="img-0")


class Stage2Config(BaseModel):
    """Configurable thresholds for Stage 2 SKU catalogue resolution.
    Initial heuristic values; tuned using evaluation benchmarks.
    """
    resolution_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Minimum score required to declare a SKU definitively RESOLVED"
    )
    ambiguity_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Minimum candidate score to be considered plausible; below this is UNRESOLVED"
    )
    min_resolution_margin: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Minimum score delta required between top candidate and runner-up to avoid AMBIGUOUS"
    )
    max_occlusion_for_resolution: float = Field(
        default=0.40, ge=0.0, le=1.0,
        description="Maximum occlusion ratio permitted before resolution is forced to AMBIGUOUS"
    )
    min_observation_confidence: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum Stage 1 perception confidence required to declare a SKU RESOLVED"
    )


class PackageSKUResolution(BaseModel):
    """STAGE 2 OUTPUT: Full catalogue resolution report for all observed units in an image.
    Strictly order-blind: Contains only visual observations resolved against the product catalogue.
    """
    image_id: str
    resolved_objects: list[ResolvedObjectItem] = Field(
        default_factory=list,
        description="One resolution entry per Stage 1 observed item"
    )

    @property
    def total_objects(self) -> int:
        return len(self.resolved_objects)

    @property
    def resolved_items(self) -> list[ResolvedObjectItem]:
        return [o for o in self.resolved_objects if o.status == ResolutionStatus.RESOLVED]

    @property
    def ambiguous_items(self) -> list[ResolvedObjectItem]:
        return [o for o in self.resolved_objects if o.status == ResolutionStatus.AMBIGUOUS]

    @property
    def unresolved_items(self) -> list[ResolvedObjectItem]:
        return [o for o in self.resolved_objects if o.status == ResolutionStatus.UNRESOLVED]

    @property
    def foreign_objects(self) -> list[ResolvedObjectItem]:
        return [o for o in self.resolved_objects if o.semantic_type == "foreign_object"]

    @property
    def unknown_products(self) -> list[ResolvedObjectItem]:
        return [o for o in self.resolved_objects if o.semantic_type == "product" and o.status == ResolutionStatus.UNRESOLVED]




# ==============================================================================
# STAGE 3: DETERMINISTIC MANIFEST RECONCILIATION SCHEMAS
# ==============================================================================

ReconciliationStatus = Literal["MATCHED", "MISSING", "EXTRA", "QUANTITY_MISMATCH"]


class ReconciledLine(BaseModel):
    """Reconciliation result for a specific SKU comparing expected vs observed count."""
    sku: str = Field(description="Product SKU identifier")
    product_name: Optional[str] = None
    expected_qty: int = Field(ge=0, description="Summed expected quantity from order manifest")
    observed_qty: int = Field(ge=0, description="Summed count of confidently resolved Stage 2 items")
    delta: int = Field(description="observed_qty - expected_qty (0=match, <0=missing/under-pack, >0=extra/over-pack)")
    status: ReconciliationStatus
    object_ids: list[str] = Field(
        default_factory=list,
        description="Stage 1/2 object_ids that correspond to this resolved SKU"
    )
    notes: Optional[str] = None


class UnverifiedLine(BaseModel):
    """An expected SKU whose full quantity cannot be verified due to ambiguous Stage 2 observations."""
    expected_sku: str
    product_name: Optional[str] = None
    expected_qty: int = Field(ge=0)
    observed_resolved_qty: int = Field(ge=0, description="Count of units confidently resolved for this SKU")
    unverified_qty: int = Field(ge=1, description="Quantity shortfall potentially covered by ambiguous objects")
    possible_object_ids: list[str] = Field(
        default_factory=list,
        description="Object IDs of ambiguous items having this SKU as a candidate"
    )
    candidate_skus: list[CandidateSKU] = Field(
        default_factory=list,
        description="Candidate SKU alternatives from the ambiguous observations"
    )
    reason: str
    evidence: list[str] = Field(default_factory=list)


class ManifestReconciliation(BaseModel):
    """STAGE 3 OUTPUT: Pure Python deterministic comparison between order manifest and Stage 2 resolution.
    
    Contains exact differences: matched, missing, extra, quantity mismatches, unverified items,
    plus preserved foreign objects and unresolved products.
    Strictly does NOT declare final packing decisions (SEAL / STOP & FIX / UNCERTAIN).
    """
    order_id: str
    matched: list[ReconciledLine] = Field(
        default_factory=list,
        description="SKUs where observed count exactly equals expected count (delta == 0)"
    )
    missing: list[ReconciledLine] = Field(
        default_factory=list,
        description="Expected SKUs with 0 observed units and no ambiguous candidates (delta < 0)"
    )
    extra: list[ReconciledLine] = Field(
        default_factory=list,
        description="Observed resolved SKUs not present in expected order manifest (expected_qty == 0)"
    )
    quantity_mismatches: list[ReconciledLine] = Field(
        default_factory=list,
        description="Expected SKUs observed with quantity deficit or surplus (delta != 0) without ambiguity"
    )
    unverified: list[UnverifiedLine] = Field(
        default_factory=list,
        description="Expected SKUs with unverified shortfall due to ambiguous Stage 2 observations"
    )
    unresolved_products: list[ResolvedObjectItem] = Field(
        default_factory=list,
        description="Visible products from Stage 2 that did not match any catalogue SKU"
    )
    foreign_objects: list[ResolvedObjectItem] = Field(
        default_factory=list,
        description="Non-inventory foreign objects (e.g. tools, phone, trash) detected in package"
    )

    @property
    def is_perfect_match(self) -> bool:
        """True if every expected item is matched, no missing/extra/mismatch/unverified, and no foreign items."""
        return (
            len(self.matched) > 0
            and len(self.missing) == 0
            and len(self.extra) == 0
            and len(self.quantity_mismatches) == 0
            and len(self.unverified) == 0
            and len(self.unresolved_products) == 0
            and len(self.foreign_objects) == 0
        )

    @property
    def has_unverified(self) -> bool:
        return len(self.unverified) > 0

    @property
    def has_foreign_objects(self) -> bool:
        return len(self.foreign_objects) > 0

    @property
    def total_expected_units(self) -> int:
        return (
            sum(m.expected_qty for m in self.matched)
            + sum(m.expected_qty for m in self.missing)
            + sum(m.expected_qty for m in self.quantity_mismatches)
            + sum(u.expected_qty for u in self.unverified)
        )

    @property
    def total_observed_resolved_units(self) -> int:
        return (
            sum(m.observed_qty for m in self.matched)
            + sum(e.observed_qty for e in self.extra)
            + sum(q.observed_qty for q in self.quantity_mismatches)
            + sum(u.observed_resolved_qty for u in self.unverified)
        )


# CANONICAL PACK MANAGER RESPONSE SCHEMAS (Stages 3-5 & API / UI compatibility)
# ==============================================================================

ConfidenceLevel = Literal["high", "medium", "low"]
DecisionType = Literal["SEAL", "STOP_FIX", "UNCERTAIN"]
MatchStatusType = Literal["PASS", "FAIL", "UNCERTAIN"]




# ==============================================================================
# STAGE 4: FINAL DECISION ENGINE SCHEMAS
# ==============================================================================

class DecisionResult(BaseModel):
    """STAGE 4 OUTPUT: Autonomous packing decision based on deterministic verification rules.
    Precedence: UNCERTAIN > STOP_FIX > SEAL.
    """
    decision: DecisionType
    reason: str
    confidence: ConfidenceLevel
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues causing STOP_FIX or UNCERTAIN"
    )
    evidence_object_ids: list[str] = Field(
        default_factory=list,
        description="Object IDs linked to blocking issues or discrepancies"
    )
    resolved_skus_count: int = Field(default=0)
    expected_skus_count: int = Field(default=0)




# ==============================================================================
# STAGE 5: GROUNDED EVIDENCE & AUDIT TRAIL SCHEMAS
# ==============================================================================

class ExpectedItemSummary(BaseModel):
    """Summary of an ordered SKU item from the manifest."""
    sku: str
    expected_qty: int
    product_name: str
    asin: Optional[str] = None


class GroundedEvidenceRecord(BaseModel):
    """Traceable evidence linking an observed physical object to SKU resolution and reconciliation."""
    image_id: str
    object_id: str
    bbox: BoundingBox2D
    observed_label: str
    visual_attributes: dict[str, Any] = Field(default_factory=dict)
    observation_confidence: float
    resolved_sku: Optional[str] = None
    sku_resolution_status: ResolutionStatus
    sku_match_confidence: float = 0.0
    candidate_skus: list[CandidateSKU] = Field(default_factory=list)
    matching_evidence: list[str] = Field(default_factory=list)
    related_expected_sku: Optional[str] = None
    reconciliation_status: Optional[str] = None
    grounded_summary: str = Field(
        description="Human and audit-readable evidence sentence strictly grounded in visual facts."
    )


class PackVerificationReport(BaseModel):
    """STAGE 5 OUTPUT: Complete grounded verification report and immutable audit trail.
    Links Stages 1-4 into a single verifiable, audit-grade verification artifact.
    """
    order_id: str
    decision: DecisionType
    decision_reason: str
    confidence: ConfidenceLevel

    # Itemized Manifest & Observation Breakdown
    expected_items: list[ExpectedItemSummary] = Field(default_factory=list)
    observed_items: list[ObservedPhysicalItem] = Field(default_factory=list)
    matched_items: list[ReconciledLine] = Field(default_factory=list)
    missing_items: list[ReconciledLine] = Field(default_factory=list)
    extra_items: list[ReconciledLine] = Field(default_factory=list)
    quantity_mismatches: list[ReconciledLine] = Field(default_factory=list)
    unverified_items: list[UnverifiedLine] = Field(default_factory=list)

    # Grounded Evidence & Operator Guidance
    evidence_records: list[GroundedEvidenceRecord] = Field(
        default_factory=list,
        description="Per-object grounded evidence tracing image crop to SKU resolution and reconciliation"
    )
    blocking_issues: list[str] = Field(default_factory=list)
    operator_action: str = Field(
        description="Clear, directive warehouse operator instruction based on decision"
    )

    # Audit Metadata
    pipeline_version: str = "1.0.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    image_ids: list[str] = Field(default_factory=list)
    image_sha256: Optional[str] = None
    model_name: Optional[str] = None

    @property
    def is_sealed(self) -> bool:
        return self.decision == "SEAL"

    @property
    def is_stopped(self) -> bool:
        return self.decision == "STOP_FIX"

    @property
    def is_uncertain(self) -> bool:
        return self.decision == "UNCERTAIN"


class OrderItemInput(BaseModel):
    name: str
    brand: Optional[str] = None
    expected_qty: int = Field(ge=0)
    variant: Optional[str] = None
    sku: Optional[str] = None


class DetectedItemAI(BaseModel):
    name: str
    brand: Optional[str] = None
    detected_qty: int = Field(ge=0)
    variant: Optional[str] = None
    confidence: ConfidenceLevel = "high"
    notes: Optional[str] = ""
    bbox: Optional[BoundingBox2D] = None
    object_id: Optional[str] = None


class MatchResultAI(BaseModel):
    item_name: str
    expected_qty: int = Field(ge=0)
    detected_qty: int = Field(ge=0)
    variant_match: bool = True
    brand_match: bool = True
    status: MatchStatusType = "PASS"


class MissingItemAI(BaseModel):
    item_name: str
    expected_qty: int = Field(ge=1)
    reason: str = "Not visible in box"


class ExtraItemAI(BaseModel):
    item_name: str
    qty: int = Field(ge=1)
    notes: str = "Not in order"


class ProductConditionAI(BaseModel):
    visible_damage: bool = False
    notes: str = "No visible damage to products or packaging"


class PackManagerAIResponse(BaseModel):
    """The canonical output payload emitted by the Pack Manager AI Agent."""

    order_id: str
    order_items: list[OrderItemInput] = Field(default_factory=list)
    detected_items: list[DetectedItemAI] = Field(default_factory=list)
    matches: list[MatchResultAI] = Field(default_factory=list)
    missing_items: list[MissingItemAI] = Field(default_factory=list)
    extra_items: list[ExtraItemAI] = Field(default_factory=list)
    product_condition: ProductConditionAI = Field(default_factory=ProductConditionAI)
    decision: DecisionType
    decision_reason: str
    confidence: ConfidenceLevel
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> PackManagerAIResponse:
        """Enforce strict safety rules: SEAL is prohibited if defects or ambiguities exist."""
        has_missing = len(self.missing_items) > 0
        has_extra = len(self.extra_items) > 0
        has_failed_match = any(m.status == "FAIL" for m in self.matches)
        has_uncertain_match = any(m.status == "UNCERTAIN" for m in self.matches)
        has_damage = self.product_condition.visible_damage

        if self.decision == "SEAL":
            if has_missing or has_extra or has_failed_match or has_damage:
                # Disallow SEAL when defects are detected
                self.decision = "STOP_FIX"
                if not self.decision_reason:
                    self.decision_reason = "Defects detected during package verification."
            elif self.confidence == "low" or has_uncertain_match:
                self.decision = "UNCERTAIN"
                if not self.decision_reason:
                    self.decision_reason = "Inspection confidence is low; manual review required."
        return self

    def to_summary_dict(self) -> dict[str, Any]:
        """Convenience dictionary representation."""
        return self.model_dump(mode="json")
