"""STAGE 5: GROUNDED EVIDENCE & AUDIT TRAIL ENGINE

Packages and links grounded evidence already produced by Stages 1-4.
NO re-running vision, NO re-running SKU resolution, NO changing reconciliation,
NO changing decision, NO LLMs, NO fabricating evidence.

Produces audit-grade, traceable PackVerificationReport.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pack_manager.agent_schemas import (
    DecisionResult,
    ExpectedItemSummary,
    GroundedEvidenceRecord,
    ManifestReconciliation,
    PackageSKUResolution,
    PackageVisualObservation,
    PackVerificationReport,
    ResolutionStatus,
)
from pack_manager.models import Order, OrderLine


OPERATOR_ACTIONS: dict[str, str] = {
    "SEAL": "Seal package",
    "STOP_FIX": "Correct package contents before sealing",
    "UNCERTAIN": "Capture another image or perform manual verification",
}


def _extract_expected_summaries(order: Order | dict[str, Any] | Path | str) -> list[ExpectedItemSummary]:
    """Extract itemized expected lines from order manifest."""
    raw_lines: list[dict[str, Any]] = []

    if isinstance(order, Order):
        for l in order.lines:
            raw_lines.append({
                "sku": l.sku.strip(),
                "expected_qty": l.expected_qty,
                "product_name": l.product_name,
                "asin": l.asin,
            })
    elif isinstance(order, (str, Path)):
        p = Path(order)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            raw_lines = data.get("lines") or data.get("items") or []
    elif isinstance(order, dict):
        raw_lines = order.get("lines") or order.get("items") or []

    # Aggregate by SKU to match reconciliation
    agg: dict[str, ExpectedItemSummary] = {}
    for r in raw_lines:
        sku = str(r.get("sku") or r.get("name", "")).strip()
        if not sku:
            continue
        qty = int(r.get("expected_qty", r.get("quantity", 1)))
        name = str(r.get("product_name") or r.get("name", sku))
        asin = r.get("asin")

        if sku in agg:
            agg[sku].expected_qty += qty
        else:
            agg[sku] = ExpectedItemSummary(
                sku=sku,
                expected_qty=qty,
                product_name=name,
                asin=asin,
            )

    return list(agg.values())


def build_verification_report(
    reconciliation: ManifestReconciliation,
    decision: DecisionResult,
    observation: PackageVisualObservation,
    resolution: PackageSKUResolution,
    order: Optional[Order | dict[str, Any] | Path | str] = None,
    *,
    image_bytes: Optional[bytes] = None,
    image_sha256: Optional[str] = None,
    model_name: Optional[str] = None,
) -> PackVerificationReport:
    """STAGE 5: Package Stages 1-4 into a grounded, audit-grade verification report.

    Strict Constraints:
    - Never modifies Stage 4 decision or confidence.
    - Never invents image evidence for missing items.
    - Preserves exact image_id, object_id, and bounding boxes for all observed items.
    """
    # 1. Decision Preservation Guard: Stage 5 MUST NOT modify Stage 4 decision
    final_decision = decision.decision
    final_reason = decision.reason
    final_confidence = decision.confidence

    # 2. Extract Expected Items
    expected_summaries = _extract_expected_summaries(order) if order is not None else []
    if not expected_summaries:
        # Reconstruct from reconciliation lines if order not passed directly
        for m in reconciliation.matched + reconciliation.missing + reconciliation.quantity_mismatches:
            expected_summaries.append(ExpectedItemSummary(
                sku=m.sku,
                expected_qty=m.expected_qty,
                product_name=m.product_name or m.sku,
            ))
        for u in reconciliation.unverified:
            expected_summaries.append(ExpectedItemSummary(
                sku=u.expected_sku,
                expected_qty=u.expected_qty,
                product_name=u.product_name or u.expected_sku,
            ))

    # 3. Index Stage 2 Resolutions by object_id
    resolutions_by_obj = {o.object_id: o for o in resolution.resolved_objects}

    # 4. Map Reconciliation States
    matched_skus = {m.sku for m in reconciliation.matched}
    extra_skus = {e.sku for e in reconciliation.extra}
    mismatch_skus = {q.sku for q in reconciliation.quantity_mismatches}
    unverified_map = {
        obj_id: u.expected_sku
        for u in reconciliation.unverified
        for obj_id in u.possible_object_ids
    }

    # 5. Build Grounded Evidence Records (Only for genuinely observed objects from Stage 1/2)
    evidence_records: list[GroundedEvidenceRecord] = []

    for item in observation.observed_items:
        res = resolutions_by_obj.get(item.object_id)
        if res is None:
            continue

        rel_expected_sku: Optional[str] = None
        rec_status: Optional[str] = None

        if res.status == ResolutionStatus.RESOLVED and res.resolved_sku:
            sku = res.resolved_sku
            if sku in matched_skus:
                rec_status = "MATCHED"
                rel_expected_sku = sku
            elif sku in extra_skus:
                rec_status = "EXTRA"
                rel_expected_sku = None
            elif sku in mismatch_skus:
                rec_status = "QUANTITY_MISMATCH"
                rel_expected_sku = sku
            else:
                rec_status = "RESOLVED"
                rel_expected_sku = sku
        elif res.status == ResolutionStatus.AMBIGUOUS:
            rec_status = "UNVERIFIED"
            rel_expected_sku = unverified_map.get(item.object_id)
        else:
            rec_status = "UNRESOLVED"
            rel_expected_sku = None

        # Build precise, grounded evidence sentence
        bbox_str = f"[ymin={item.bbox.ymin:.2f}, xmin={item.bbox.xmin:.2f}, ymax={item.bbox.ymax:.2f}, xmax={item.bbox.xmax:.2f}]"
        if res.status == ResolutionStatus.RESOLVED and res.resolved_sku:
            summary = (
                f"{item.object_id} in image {item.image_id}, bbox {bbox_str}, "
                f"observed '{item.label}', resolved as {res.resolved_sku} with confidence {res.confidence:.2f}."
            )
        elif res.status == ResolutionStatus.AMBIGUOUS:
            cands_str = ", ".join(c.sku for c in res.candidate_skus) or "multiple candidates"
            summary = (
                f"{item.object_id} in image {item.image_id}, bbox {bbox_str}, "
                f"observed '{item.label}', ambiguous between candidate SKUs ({cands_str}) (confidence: {res.confidence:.2f})."
            )
        else:
            summary = (
                f"{item.object_id} in image {item.image_id}, bbox {bbox_str}, "
                f"observed '{item.label}' (semantic type: {res.semantic_type}), unresolved against product catalogue."
            )

        evidence_records.append(GroundedEvidenceRecord(
            image_id=item.image_id,
            object_id=item.object_id,
            bbox=item.bbox,
            observed_label=item.label,
            visual_attributes=item.visual_attributes,
            observation_confidence=item.confidence,
            resolved_sku=res.resolved_sku,
            sku_resolution_status=res.status,
            sku_match_confidence=res.confidence,
            candidate_skus=res.candidate_skus,
            matching_evidence=res.matching_evidence,
            related_expected_sku=rel_expected_sku,
            reconciliation_status=rec_status,
            grounded_summary=summary,
        ))

    # 6. Compute SHA-256 for Audit Traceability if Image Bytes Provided
    computed_sha = image_sha256
    if not computed_sha and image_bytes:
        computed_sha = f"sha256:{hashlib.sha256(image_bytes).hexdigest()}"

    # 7. Collect Image Identifiers
    image_ids = list(dict.fromkeys(
        [observation.image_id] + [it.image_id for it in observation.observed_items]
    ))

    # 8. Operator Directive Action
    operator_action = OPERATOR_ACTIONS.get(final_decision, "Perform manual verification")

    return PackVerificationReport(
        order_id=reconciliation.order_id,
        decision=final_decision,
        decision_reason=final_reason,
        confidence=final_confidence,
        expected_items=expected_summaries,
        observed_items=observation.observed_items,
        matched_items=reconciliation.matched,
        missing_items=reconciliation.missing,
        extra_items=reconciliation.extra,
        quantity_mismatches=reconciliation.quantity_mismatches,
        unverified_items=reconciliation.unverified,
        evidence_records=evidence_records,
        blocking_issues=decision.blocking_issues,
        operator_action=operator_action,
        pipeline_version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        image_ids=image_ids,
        image_sha256=computed_sha,
        model_name=model_name or "simulation-vlm",
    )
