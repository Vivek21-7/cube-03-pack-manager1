"""STAGE 4: FINAL DECISION ENGINE

Pure Python deterministic decision logic for Pack Manager AI.
NO LLMs (NO Gemini, OpenAI, Claude).

Evaluates Stage 3 manifest reconciliation and Stage 1/2 perception signals to produce:
SEAL | STOP_FIX | UNCERTAIN

Strict Precedence:
UNCERTAIN > STOP_FIX > SEAL
"""

from __future__ import annotations

import logging
from typing import Optional

from pack_manager.agent_schemas import (
    DecisionResult,
    DecisionType,
    ManifestReconciliation,
    PackageVisualObservation,
)

logger = logging.getLogger(__name__)


def evaluate_decision(
    reconciliation: ManifestReconciliation,
    observation: Optional[PackageVisualObservation] = None,
) -> DecisionResult:
    """STAGE 4: Synthesize final autonomous packing decision.

    Strict Precedence:
    1. UNCERTAIN (highest priority): Triggered if evidence cannot reliably verify contents.
       - Blurry, low-light, or occluded photographs
       - Unverified expected SKUs (ambiguity shields)
       - Unresolved products that prevent confident verification
    2. STOP_FIX: Triggered if evidence is confident and confirmed discrepancies exist.
       - Confirmed missing items
       - Confirmed extra items / substitutions
       - Confirmed quantity mismatches (over-pack or under-pack)
       - Foreign non-inventory objects (tools, phone, trash)
    3. SEAL: Triggered ONLY if all expected SKUs and quantities are matched with confident evidence.
    """
    blocking_issues: list[str] = []
    evidence_object_ids: list[str] = []

    expected_count = reconciliation.total_expected_units
    resolved_count = reconciliation.total_observed_resolved_units

    # =========================================================================
    # 1. UNCERTAIN EVALUATION (Highest Precedence)
    # =========================================================================

    # 1A: Visual Perception Quality (Stage 1 signals)
    if observation is not None:
        if observation.image_quality in ("blurry", "low_light", "glare", "obstructed") or observation.clarity_score < 0.50:
            issue = (
                f"Package photograph quality is {observation.image_quality} "
                f"(clarity score: {observation.clarity_score:.2f}). "
                "Visual evidence is insufficient to verify carton contents with certainty."
            )
            blocking_issues.append(issue)
            return DecisionResult(
                decision="UNCERTAIN",
                reason=issue + " Please capture a clearer, well-lit photograph of the open carton.",
                confidence="low",
                blocking_issues=blocking_issues,
                evidence_object_ids=[it.object_id for it in observation.observed_items],
                resolved_skus_count=resolved_count,
                expected_skus_count=expected_count,
            )

    # 1B: Unverified Expected SKUs (Ambiguity Shields from Stage 2/3)
    if reconciliation.has_unverified:
        for unver in reconciliation.unverified:
            cand_names = [c.sku for c in unver.candidate_skus] or ["alternative SKUs"]
            issue = (
                f"Expected SKU '{unver.expected_sku}' shortfall of {unver.unverified_qty} unit(s) cannot be verified: "
                f"ambiguous observation(s) {unver.possible_object_ids} could match {', '.join(cand_names)}."
            )
            blocking_issues.append(issue)
            evidence_object_ids.extend(unver.possible_object_ids)

    # 1C: Unresolved Products Affecting Verification
    if len(reconciliation.unresolved_products) > 0:
        for unres in reconciliation.unresolved_products:
            issue = (
                f"Uncatalogued visible product '{unres.object_id}' in package cannot be resolved to any known catalogue SKU. "
                "Prevents confident carton verification."
            )
            blocking_issues.append(issue)
            evidence_object_ids.append(unres.object_id)

    # If ANY uncertainty conditions are active, UNCERTAIN dominates!
    if blocking_issues:
        # Precedence Rule: If there are also confirmed defects, note them in blocking issues
        # but return UNCERTAIN because ambiguous items could materially affect the carton state
        if len(reconciliation.missing) > 0:
            blocking_issues.append(f"Confirmed missing: {', '.join(m.sku for m in reconciliation.missing)}")
        if len(reconciliation.extra) > 0:
            blocking_issues.append(f"Confirmed extra: {', '.join(e.sku for e in reconciliation.extra)}")

        return DecisionResult(
            decision="UNCERTAIN",
            reason=(
                f"Verification uncertain ({len(blocking_issues)} issue(s)): "
                + "; ".join(blocking_issues)
                + ". Operator inspection or clearer camera angle required."
            ),
            confidence="medium",
            blocking_issues=blocking_issues,
            evidence_object_ids=list(dict.fromkeys(evidence_object_ids)),
            resolved_skus_count=resolved_count,
            expected_skus_count=expected_count,
        )

    # =========================================================================
    # 2. STOP_FIX EVALUATION (Confirmed Defects with High Confidence)
    # =========================================================================

    # 2A: Foreign Non-Inventory Objects
    if reconciliation.has_foreign_objects:
        for fo in reconciliation.foreign_objects:
            issue = f"Unauthorized foreign object detected in packaging area: {fo.object_id}"
            blocking_issues.append(issue)
            evidence_object_ids.append(fo.object_id)

    # 2B: Confirmed Missing Items
    if len(reconciliation.missing) > 0:
        for m in reconciliation.missing:
            issue = f"Missing ordered item: {m.sku} (expected {m.expected_qty}, observed 0)"
            blocking_issues.append(issue)

    # 2C: Confirmed Extra Items (Substitutions / Foreign SKUs)
    if len(reconciliation.extra) > 0:
        for e in reconciliation.extra:
            issue = f"Unexpected extra item: {e.sku} (expected 0, observed {e.observed_qty})"
            blocking_issues.append(issue)
            evidence_object_ids.extend(e.object_ids)

    # 2D: Confirmed Quantity Mismatches (Under-pack or Over-pack)
    if len(reconciliation.quantity_mismatches) > 0:
        for q in reconciliation.quantity_mismatches:
            direction = "Over-pack" if q.delta > 0 else "Under-pack"
            issue = f"{direction}: {q.sku} (expected {q.expected_qty}, observed {q.observed_qty})"
            blocking_issues.append(issue)
            evidence_object_ids.extend(q.object_ids)

    if blocking_issues:
        # Determine specific reason label (e.g. wrong item substitution vs simple shortage)
        reasons_summary = "; ".join(blocking_issues)
        if len(reconciliation.missing) > 0 and len(reconciliation.extra) > 0:
            main_reason = f"CRITICAL DEFECT: Product substitution / wrong item detected. {reasons_summary}. DO NOT SEAL."
        elif len(reconciliation.missing) > 0:
            main_reason = f"Missing item(s) detected. {reasons_summary}. DO NOT SEAL."
        elif len(reconciliation.extra) > 0 or reconciliation.has_foreign_objects:
            main_reason = f"Unauthorized / extra item(s) detected in package. {reasons_summary}. DO NOT SEAL."
        else:
            main_reason = f"Quantity mismatch detected. {reasons_summary}. DO NOT SEAL."

        return DecisionResult(
            decision="STOP_FIX",
            reason=main_reason,
            confidence="high",
            blocking_issues=blocking_issues,
            evidence_object_ids=list(dict.fromkeys(evidence_object_ids)),
            resolved_skus_count=resolved_count,
            expected_skus_count=expected_count,
        )

    # =========================================================================
    # 3. SEAL EVALUATION (Conservative Approval)
    # =========================================================================

    if reconciliation.is_perfect_match:
        for m in reconciliation.matched:
            evidence_object_ids.extend(m.object_ids)

        return DecisionResult(
            decision="SEAL",
            reason=(
                f"All {expected_count} expected product unit(s) verified across {len(reconciliation.matched)} SKU(s) "
                "with high visual confidence. Packaging contents match order manifest exactly. APPROVED TO SEAL."
            ),
            confidence="high",
            blocking_issues=[],
            evidence_object_ids=list(dict.fromkeys(evidence_object_ids)),
            resolved_skus_count=resolved_count,
            expected_skus_count=expected_count,
        )

    # Fallback conservative guard: if not perfect and no explicit error caught, default to UNCERTAIN
    return DecisionResult(
        decision="UNCERTAIN",
        reason="Order reconciliation could not definitively certify package contents. Manual review required.",
        confidence="low",
        blocking_issues=["Uncertified carton state"],
        evidence_object_ids=evidence_object_ids,
        resolved_skus_count=resolved_count,
        expected_skus_count=expected_count,
    )
