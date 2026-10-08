"""STAGE 2: SKU / CATALOGUE RESOLUTION ENGINE

Maps Stage 1 visual observations to catalogue SKUs without seeing customer orders.
Order-Blind: Strictly does NOT accept Order manifests or expected quantities.

Architecture:
1. Configurable thresholds (initial heuristics, tunable via benchmark eval).
2. Hard compatibility filtering (eliminates conflicting colors, categories, form factors).
3. Deterministic metadata & attribute scoring among compatible candidates.
4. Reference-image disambiguation for near-duplicate or tie cases (optional, non-crashing).
5. Explicit ambiguity detection (no guessing) and preservation of semantic types
   (foreign_object vs unknown_product).
"""

from __future__ import annotations

import logging
import re
import math
from pathlib import Path
from typing import Any, Optional
from PIL import Image

from pack_manager.agent_schemas import (
    BoundingBox2D,
    CandidateSKU,
    ObservedPhysicalItem,
    PackageSKUResolution,
    PackageVisualObservation,
    ResolutionStatus,
    ResolvedObjectItem,
    Stage2Config,
)
from pack_manager.models import CatalogItem

logger = logging.getLogger(__name__)

# Heuristic default thresholds (Configurable via Stage2Config)
DEFAULT_RESOLUTION_THRESHOLD: float = 0.85
DEFAULT_AMBIGUITY_THRESHOLD: float = 0.50
DEFAULT_MIN_RESOLUTION_MARGIN: float = 0.15
DEFAULT_MAX_OCCLUSION_FOR_RESOLUTION: float = 0.40
DEFAULT_MIN_OBSERVATION_CONFIDENCE: float = 0.60


# Normalized color synonyms for robust matching
COLOR_GROUPS: dict[str, set[str]] = {
    "blue": {"blue", "royal blue", "light blue", "sky blue", "cyan"},
    "navy": {"navy", "navy blue", "dark blue", "midnight blue"},
    "red": {"red", "crimson", "scarlet", "ruby"},
    "black": {"black", "charcoal", "pitch black"},
    "white": {"white", "off-white", "ivory", "cream"},
    "green": {"green", "forest green", "olive", "emerald", "lime"},
    "gray": {"gray", "grey", "silver", "heather gray"},
    "yellow": {"yellow", "gold", "mustard"},
    "orange": {"orange", "amber", "tangerine"},
    "brown": {"brown", "tan", "khaki", "beige"},
}


def _normalize_color(raw_color: Optional[str]) -> Optional[str]:
    if not raw_color:
        return None
    c = raw_color.strip().lower()
    for canonical, synonyms in COLOR_GROUPS.items():
        if c == canonical or c in synonyms:
            return canonical
    return c


def _are_colors_incompatible(observed_color: Optional[str], catalogue_color: Optional[str]) -> bool:
    """Returns True if both colors are specified and clearly conflict."""
    if not observed_color or not catalogue_color:
        return False
    norm_obs = _normalize_color(observed_color)
    norm_cat = _normalize_color(catalogue_color)
    if not norm_obs or not norm_cat:
        return False
    # If they are identical canonical colors, compatible
    if norm_obs == norm_cat:
        return False
    # Blue vs Navy is an edge case: visually similar, handled by soft scoring/ambiguity/reference images, not hard rejected!
    if {norm_obs, norm_cat} == {"blue", "navy"}:
        return False
    # Distinct colors (e.g. red vs blue, red vs black, white vs black) are hard conflicts
    return True


def _check_hard_compatibility(
    item: ObservedPhysicalItem,
    cat_item: CatalogItem,
) -> tuple[bool, list[str]]:
    """Determine if a catalogue item is fundamentally compatible with the observation.
    General-purpose rules based on semantic category, colorway, and visual attributes.
    Returns (is_compatible, reasons_or_conflicts).
    """
    reasons: list[str] = []
    
    # 1. Semantic Category Conflict
    if item.category == "documentation" and cat_item.category != "documentation":
        return False, [f"Category conflict: documentation observation cannot match non-documentation SKU {cat_item.sku}"]
    if item.category == "product" and cat_item.category == "documentation":
        return False, [f"Category conflict: product observation cannot match documentation SKU {cat_item.sku}"]
    
    # 2. Definite Colorway Conflict
    obs_color = item.visual_attributes.get("color") or item.visual_attributes.get("variant")
    cat_color = cat_item.attributes.get("color") or cat_item.attributes.get("variant")
    
    if obs_color and cat_color and item.confidence >= 0.70:
        if _are_colors_incompatible(str(obs_color), str(cat_color)):
            return False, [f"Hard color conflict: observed '{obs_color}' vs catalogue '{cat_color}'"]
        elif _normalize_color(str(obs_color)) == _normalize_color(str(cat_color)):
            reasons.append(f"Color match: {cat_color}")

    return True, reasons


def _compute_metadata_score(
    item: ObservedPhysicalItem,
    cat_item: CatalogItem,
    compatibility_reasons: list[str],
) -> tuple[float, list[str]]:
    """Compute deterministic match score in [0.0, 1.0] across visual attributes.
    Completely general-purpose: uses token overlap, category alignment, color matching,
    text markings, and structured attributes.
    """
    score = 0.0
    evidence = list(compatibility_reasons)
    
    obs_label = item.label.lower()
    cat_name = cat_item.product_name.lower()
    cat_category = cat_item.category.lower()
    obs_color = item.visual_attributes.get("color") or item.visual_attributes.get("variant")
    norm_obs_col = _normalize_color(str(obs_color)) if obs_color else None
    cat_color = cat_item.attributes.get("color") or cat_item.attributes.get("variant")
    norm_cat_col = _normalize_color(str(cat_color)) if cat_color else None

    STOPWORDS = {"the", "and", "with", "for", "item", "product", "of", "in", "by"}
    
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9\-]+", text.lower())
        return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

    obs_tokens = set(tokenize(obs_label))
    cat_name_tokens = set(tokenize(cat_name))
    
    # Check words in product title that match observed label tokens
    matched_title_words = [
        w for w in cat_name_tokens
        if w in obs_tokens
    ]
    
    # 1. Product Name & Category Overlap (Weight: 0.35 - 0.50)
    if len(matched_title_words) >= 2:
        score += 0.50
        evidence.append(f"Product title token overlap ({len(matched_title_words)}): '{' '.join(matched_title_words)}'")
    elif len(matched_title_words) == 1:
        score += 0.40
        evidence.append(f"Product title keyword match: '{matched_title_words[0]}'")
    elif cat_category in obs_label or any(t in cat_category for t in obs_tokens):
        score += 0.25
        evidence.append(f"Category alignment: '{cat_item.category}'")

    # 2. Color / Variant Alignment (Weight: 0.40 - 0.45)
    if norm_obs_col and norm_cat_col:
        if norm_obs_col == norm_cat_col:
            score += 0.40
            evidence.append(f"Observed color '{obs_color}' matches catalogue color '{cat_color}'")
        elif {norm_obs_col, norm_cat_col} == {"blue", "navy"}:
            score += 0.20
            evidence.append(f"Near color match: observed '{obs_color}' vs catalogue '{cat_color}'")
    elif cat_color and str(cat_color).lower() in obs_label:
        score += 0.35
        evidence.append(f"Color keyword '{cat_color}' matched in label")

    # 3. Structured Visual Attributes Match (Weight: 0.05 - 0.10)
    obs_attrs = item.visual_attributes
    obs_text_raw = str(obs_attrs.get("text") or obs_attrs.get("visible_text") or obs_attrs.get("markings") or obs_attrs.get("print") or "").lower()
    for k, v in cat_item.attributes.items():
        if k in ("color", "variant"):
            continue
        v_str = str(v).lower()
        obs_text_words = set(re.findall(r"[a-z0-9]+", obs_text_raw))
        if (
            v_str in obs_label
            or v_str == str(obs_attrs.get(k, "")).lower()
            or v_str in obs_text_words
        ):
            score += 0.05
            evidence.append(f"Attribute match: {k}={v}")

    # 4. Visible Text / Markings / Packaging Print (Weight: 0.05 - 0.15)
    if obs_text_raw:
        text_tokens = set(tokenize(obs_text_raw))
        if cat_item.sku.lower() in obs_text_raw:
            score += 0.15
            evidence.append(f"Observed printed text matches SKU: '{cat_item.sku}'")
        else:
            matched_print = cat_name_tokens.intersection(text_tokens)
            if len(matched_print) >= 2:
                score += 0.15
                evidence.append(f"Observed printed markings match multiple title words: '{' '.join(matched_print)}'")
            elif len(matched_print) == 1:
                score += 0.05
                evidence.append(f"Observed printed markings match product title: '{list(matched_print)[0]}'")
            elif norm_obs_col and norm_obs_col in text_tokens:
                score += 0.05
                evidence.append(f"Observed printed markings match color: '{norm_obs_col}'")

    return min(1.0, round(score, 4)), evidence


def _compare_reference_image(
    crop_img: Image.Image,
    ref_image_path: Path,
) -> float:
    """Compute visual similarity score [0.0, 1.0] between cropped box and reference photo.
    Uses normalized average RGB perceptual distance.
    """
    try:
        if not ref_image_path.is_file():
            return 0.50

        with Image.open(ref_image_path) as ref_img:
            size = (64, 64)
            c_thumb = crop_img.convert("RGB").resize(size)
            r_thumb = ref_img.convert("RGB").resize(size)

            # Compatible across modern and legacy Pillow versions
            c_pixels = list(c_thumb.get_flattened_data()) if hasattr(c_thumb, "get_flattened_data") else list(c_thumb.getdata())
            r_pixels = list(r_thumb.get_flattened_data()) if hasattr(r_thumb, "get_flattened_data") else list(r_thumb.getdata())

            total_dist = 0.0
            for (r1, g1, b1), (r2, g2, b2) in zip(c_pixels, r_pixels):
                dist = math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
                total_dist += dist

            max_dist = math.sqrt(255**2 * 3) * len(c_pixels)
            similarity = 1.0 - (total_dist / max_dist)
            return max(0.0, min(1.0, similarity))
    except Exception as exc:
        logger.debug("Reference image comparison skipped: %s", exc)
        return 0.50


def resolve_object_item(
    item: ObservedPhysicalItem,
    catalogue: list[CatalogItem],
    config: Stage2Config,
    package_image: Optional[Image.Image] = None,
) -> ResolvedObjectItem:
    """Resolve a single observed physical unit against the product catalogue."""

    # 1. Short-circuit: Foreign Object Preservation
    if item.category == "foreign_object" or any(kw in item.label.lower() for kw in ["scissor", "phone", "tool", "trash", "wrapper"]):
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.UNRESOLVED,
            semantic_type="foreign_object",
            unresolved_reason="FOREIGN_OBJECT",
            confidence=item.confidence,
            candidate_skus=[],
            matching_evidence=["Observed foreign non-inventory object in package (not a catalogue product)"],
            ambiguity_reason=f"Object '{item.label}' identified as foreign/non-inventory item",
            bbox=item.bbox,
            image_id=item.image_id,
        )

    # 2. Hard Compatibility Filtering
    candidates: list[tuple[CatalogItem, float, list[str]]] = []
    
    for cat_item in catalogue:
        is_compat, reasons = _check_hard_compatibility(item, cat_item)
        if not is_compat:
            continue
        
        score, evidence = _compute_metadata_score(item, cat_item, reasons)
        if score >= config.ambiguity_threshold:
            candidates.append((cat_item, score, evidence))

    # Sort candidates by score descending
    candidates.sort(key=lambda c: c[1], reverse=True)

    # 3. Two-Stage Disambiguation: Reference Image Comparison
    if package_image and len(candidates) >= 2:
        top_score = candidates[0][1]
        runner_up_score = candidates[1][1]
        score_delta = top_score - runner_up_score

        # Trigger reference image disambiguation if scores are close
        if score_delta < config.min_resolution_margin:
            w, h = package_image.size
            crop_box = (
                int(item.bbox.xmin * w),
                int(item.bbox.ymin * h),
                int(item.bbox.xmax * w),
                int(item.bbox.ymax * h),
            )
            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                crop_img = package_image.crop(crop_box)
                updated_candidates = []
                for cat_item, base_score, ev in candidates:
                    ref_sim = 0.50
                    if cat_item.reference_images:
                        ref_path = Path(cat_item.reference_images[0].uri)
                        ref_sim = _compare_reference_image(crop_img, ref_path)
                    
                    visual_boost = (ref_sim - 0.50) * 0.40
                    adjusted_score = min(1.0, max(0.0, round(base_score + visual_boost, 4)))
                    ev_copy = list(ev)
                    if cat_item.reference_images and ref_sim != 0.50:
                        ev_copy.append(f"Reference image visual comparison score: {ref_sim:.2f}")
                    updated_candidates.append((cat_item, adjusted_score, ev_copy))

                updated_candidates.sort(key=lambda c: c[1], reverse=True)
                candidates = updated_candidates

    # Convert to CandidateSKU list
    candidate_skus = [
        CandidateSKU(
            sku=cat.sku,
            score=sc,
            product_name=cat.product_name,
            asin=cat.asin,
            compatibility_reasons=ev,
        )
        for cat, sc, ev in candidates
    ]

    # 4. Check for UNRESOLVED (No plausible candidates)
    if not candidates or candidates[0][1] < config.ambiguity_threshold:
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.UNRESOLVED,
            semantic_type="product",
            unresolved_reason="UNKNOWN_PRODUCT",
            confidence=round(candidates[0][1], 2) if candidates else 0.0,
            candidate_skus=candidate_skus,
            matching_evidence=["No compatible catalogue SKU found matching observed attributes"],
            ambiguity_reason=f"Observed item '{item.label}' does not match any known SKU in the catalogue",
            bbox=item.bbox,
            image_id=item.image_id,
        )

    top_cat, top_score, top_evidence = candidates[0]
    runner_up_score = candidates[1][1] if len(candidates) > 1 else 0.0
    score_margin = round(top_score - runner_up_score, 4)

    # 5. Check for AMBIGUOUS due to low perception confidence
    if item.confidence < config.min_observation_confidence:
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.AMBIGUOUS,
            semantic_type="product",
            unresolved_reason="LOW_PERCEPTION_CONFIDENCE",
            confidence=top_score,
            candidate_skus=candidate_skus,
            matching_evidence=top_evidence,
            ambiguity_reason=f"Low Stage 1 perception confidence ({item.confidence:.2f} < {config.min_observation_confidence:.2f}) prevents certifying SKU",
            bbox=item.bbox,
            image_id=item.image_id,
        )

    # 6. Check for AMBIGUOUS due to severe occlusion
    if item.occlusion.is_occluded and item.occlusion.occlusion_ratio > config.max_occlusion_for_resolution:
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.AMBIGUOUS,
            semantic_type="product",
            unresolved_reason="SEVERE_OCCLUSION",
            confidence=top_score,
            candidate_skus=candidate_skus,
            matching_evidence=top_evidence,
            ambiguity_reason=f"Severe occlusion ({item.occlusion.occlusion_ratio:.0%} > {config.max_occlusion_for_resolution:.0%}) obscures verifying features",
            bbox=item.bbox,
            image_id=item.image_id,
        )

    # 7. Check for AMBIGUOUS due to close competitors (Score margin too narrow)
    if len(candidates) >= 2 and score_margin < config.min_resolution_margin:
        runner_sku = candidates[1][0].sku
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.AMBIGUOUS,
            semantic_type="product",
            unresolved_reason="AMBIGUOUS_COMPETITORS",
            confidence=top_score,
            candidate_skus=candidate_skus,
            matching_evidence=top_evidence,
            ambiguity_reason=(
                f"Multiple plausible catalogue SKUs ({top_cat.sku} vs {runner_sku}) "
                f"cannot be reliably distinguished visually (margin {score_margin:.2f} < {config.min_resolution_margin:.2f})"
            ),
            bbox=item.bbox,
            image_id=item.image_id,
        )

    # 8. Check for UNRESOLVED/AMBIGUOUS if top score does not reach resolution threshold
    if top_score < config.resolution_threshold:
        return ResolvedObjectItem(
            object_id=item.object_id,
            resolved_sku=None,
            status=ResolutionStatus.AMBIGUOUS,
            semantic_type="product",
            unresolved_reason="INSUFFICIENT_CONFIDENCE",
            confidence=top_score,
            candidate_skus=candidate_skus,
            matching_evidence=top_evidence,
            ambiguity_reason=f"Top candidate match score ({top_score:.2f}) below definitive resolution threshold ({config.resolution_threshold:.2f})",
            bbox=item.bbox,
            image_id=item.image_id,
        )

    # 9. Definitively RESOLVED
    return ResolvedObjectItem(
        object_id=item.object_id,
        resolved_sku=top_cat.sku,
        status=ResolutionStatus.RESOLVED,
        semantic_type="product",
        unresolved_reason=None,
        confidence=top_score,
        candidate_skus=candidate_skus,
        matching_evidence=top_evidence,
        ambiguity_reason=None,
        bbox=item.bbox,
        image_id=item.image_id,
    )


def resolve_catalogue_skus(
    observation: PackageVisualObservation,
    catalogue: list[CatalogItem],
    *,
    config: Optional[Stage2Config] = None,
    package_image: Optional[Path | str | bytes | Image.Image] = None,
    **kwargs: Any,
) -> PackageSKUResolution:
    """STAGE 2 GATEWAY: Resolve observed objects against catalogue SKUs.

    CRITICAL ARCHITECTURAL CONSTRAINTS:
    - Strictly Order-Blind: Must not receive Order, OrderLine, or expected quantities.
    - No Guessing: Ambiguous or uncatalogued items emit resolved_sku=None.
    - Preserves Semantic Types: Distinguishes foreign objects from unknown products.
    """
    if "order" in kwargs or "order_lines" in kwargs or "expected" in kwargs:
        raise TypeError("Stage 2 is strictly order-blind and must NOT receive order manifests or expected quantities.")

    for arg in [observation, catalogue]:
        arg_type = type(arg).__name__
        if "Order" in arg_type or "OrderLine" in arg_type:
            raise TypeError(f"Stage 2 cannot accept '{arg_type}'. Only PackageVisualObservation and list[CatalogItem] are permitted.")

    cfg = config or Stage2Config()

    loaded_img: Optional[Image.Image] = None
    if package_image is not None:
        try:
            if isinstance(package_image, Image.Image):
                loaded_img = package_image
            elif isinstance(package_image, (str, Path)):
                p = Path(package_image)
                if p.is_file():
                    loaded_img = Image.open(p)
            elif isinstance(package_image, bytes):
                import io
                loaded_img = Image.open(io.BytesIO(package_image))
        except Exception as exc:
            logger.warning("Could not load package image for Stage 2 reference disambiguation: %s", exc)
            loaded_img = None

    resolved_objects: list[ResolvedObjectItem] = []

    for item in observation.observed_items:
        resolved = resolve_object_item(
            item=item,
            catalogue=catalogue,
            config=cfg,
            package_image=loaded_img,
        )
        resolved_objects.append(resolved)

    return PackageSKUResolution(
        image_id=observation.image_id,
        resolved_objects=resolved_objects,
    )


# DEMO / TEST FIXTURE CATALOGUE
# Used as fallback fixture catalogue for offline tests and sample demonstrations.
# Production verification flows prefer a runtime-supplied catalogue via verify_full(catalogue=...).
DEMO_FIXTURE_CATALOGUE: list[CatalogItem] = [
    CatalogItem(sku="CAP-BLU-001", product_name="Blue Cap", category="apparel", attributes={"color": "blue", "style": "cap"}),
    CatalogItem(sku="CAP-RED-001", product_name="Red Cap", category="apparel", attributes={"color": "red", "style": "cap"}),
    CatalogItem(sku="CAP-NAVY-01", product_name="Navy Baseball Cap", category="apparel", attributes={"color": "navy", "style": "cap"}),
    CatalogItem(sku="TSHIRT-BLK-M", product_name="Black T-Shirt", category="apparel", attributes={"color": "black", "size": "M"}),
    CatalogItem(sku="MANUAL-001", product_name="User Manual / Guide", category="documentation", attributes={"type": "paper"}),
    CatalogItem(sku="SCARF-RED-001", product_name="Winter Wool Scarf", category="apparel", attributes={"color": "red"}),
    CatalogItem(sku="CAM-ACT-001", product_name="Action Camera", category="electronics", attributes={"color": "black"}),
]

DEFAULT_WAREHOUSE_CATALOGUE = DEMO_FIXTURE_CATALOGUE
