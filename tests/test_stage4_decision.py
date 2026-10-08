"""Tests for STAGE 4: FINAL DECISION ENGINE

Verifies:
1. exact match -> SEAL
2. missing item -> STOP_FIX
3. extra item -> STOP_FIX
4. quantity mismatch (under-pack) -> STOP_FIX
5. quantity mismatch (over-pack) -> STOP_FIX
6. wrong item (missing + extra) -> STOP_FIX
7. ambiguous expected SKU -> UNCERTAIN
8. unresolved product affecting verification -> UNCERTAIN
9. blurry / low-confidence verification -> UNCERTAIN
10. multiple defects -> STOP_FIX
11. uncertainty + confirmed mismatch -> UNCERTAIN takes precedence over STOP_FIX
12. never SEAL when unverified exists
13. never SEAL when unresolved product exists
14. SEAL only when everything confidently matches
15. foreign object -> STOP_FIX
16. agent.decide integration
"""

import pytest
from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import (
    BoundingBox2D,
    CandidateSKU,
    DecisionResult,
    ObservedPhysicalItem,
    ManifestReconciliation,
    PackageVisualObservation,
    ReconciledLine,
    ResolutionStatus,
    ResolvedObjectItem,
    UnverifiedLine,
)
from pack_manager.decision_engine import evaluate_decision
from pack_manager.vlm_client import SimulationVLMClient


def _make_reconciliation(
    matched: list[ReconciledLine] | None = None,
    missing: list[ReconciledLine] | None = None,
    extra: list[ReconciledLine] | None = None,
    quantity_mismatches: list[ReconciledLine] | None = None,
    unverified: list[UnverifiedLine] | None = None,
    unresolved_products: list[ResolvedObjectItem] | None = None,
    foreign_objects: list[ResolvedObjectItem] | None = None,
) -> ManifestReconciliation:
    return ManifestReconciliation(
        order_id="ord-test",
        matched=matched or [],
        missing=missing or [],
        extra=extra or [],
        quantity_mismatches=quantity_mismatches or [],
        unverified=unverified or [],
        unresolved_products=unresolved_products or [],
        foreign_objects=foreign_objects or [],
    )


def test_exact_match_seals() -> None:
    """1. Exact match with confident verification approves SEAL."""
    rec = _make_reconciliation(
        matched=[
            ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED", object_ids=["obj-1"]),
            ReconciledLine(sku="SHIRT-BLK-M", expected_qty=2, observed_qty=2, delta=0, status="MATCHED", object_ids=["obj-2", "obj-3"]),
        ]
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "SEAL"
    assert dec.confidence == "high"
    assert len(dec.blocking_issues) == 0
    assert "obj-1" in dec.evidence_object_ids
    assert "obj-2" in dec.evidence_object_ids
    assert "APPROVED TO SEAL" in dec.reason


def test_missing_item_stops_fix() -> None:
    """2. Confirmed missing item triggers STOP_FIX."""
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        missing=[ReconciledLine(sku="MANUAL-01", expected_qty=1, observed_qty=0, delta=-1, status="MISSING")],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert dec.confidence == "high"
    assert any("MANUAL-01" in issue for issue in dec.blocking_issues)
    assert "DO NOT SEAL" in dec.reason


def test_extra_item_stops_fix() -> None:
    """3. Confirmed extra item triggers STOP_FIX."""
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        extra=[ReconciledLine(sku="SCARF-RED-01", expected_qty=0, observed_qty=1, delta=1, status="EXTRA", object_ids=["obj-extra"])],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert dec.confidence == "high"
    assert "SCARF-RED-01" in dec.blocking_issues[0]
    assert "obj-extra" in dec.evidence_object_ids


def test_quantity_mismatch_underpack_stops_fix() -> None:
    """4. Quantity deficit (under-pack) triggers STOP_FIX."""
    rec = _make_reconciliation(
        quantity_mismatches=[
            ReconciledLine(sku="SHIRT-BLK-M", expected_qty=2, observed_qty=1, delta=-1, status="QUANTITY_MISMATCH", object_ids=["obj-1"]),
        ]
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert dec.confidence == "high"
    assert any("Under-pack" in issue for issue in dec.blocking_issues)


def test_quantity_mismatch_overpack_stops_fix() -> None:
    """5. Quantity surplus (over-pack) triggers STOP_FIX."""
    rec = _make_reconciliation(
        quantity_mismatches=[
            ReconciledLine(sku="SHIRT-BLK-M", expected_qty=1, observed_qty=2, delta=1, status="QUANTITY_MISMATCH", object_ids=["obj-1", "obj-2"]),
        ]
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert dec.confidence == "high"
    assert any("Over-pack" in issue for issue in dec.blocking_issues)


def test_wrong_item_representation_stops_fix() -> None:
    """6. Wrong item (missing expected + extra observed) triggers STOP_FIX with substitution reason."""
    rec = _make_reconciliation(
        missing=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=0, delta=-1, status="MISSING")],
        extra=[ReconciledLine(sku="CAP-RED-01", expected_qty=0, observed_qty=1, delta=1, status="EXTRA", object_ids=["obj-red"])],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert dec.confidence == "high"
    assert "substitution" in dec.reason.lower() or "wrong item" in dec.reason.lower()


def test_ambiguous_expected_sku_uncertain() -> None:
    """7. Ambiguous expected SKU forces UNCERTAIN."""
    rec = _make_reconciliation(
        unverified=[
            UnverifiedLine(
                expected_sku="CAP-BLU-01",
                expected_qty=1,
                observed_resolved_qty=0,
                unverified_qty=1,
                possible_object_ids=["obj-amb"],
                candidate_skus=[CandidateSKU(sku="CAP-BLU-01", score=0.7), CandidateSKU(sku="CAP-NAVY-01", score=0.7)],
                reason="Cannot distinguish Blue vs Navy cap",
            )
        ]
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "UNCERTAIN"
    assert dec.confidence == "medium"
    assert "obj-amb" in dec.evidence_object_ids
    assert any("CAP-BLU-01" in issue for issue in dec.blocking_issues)


def test_unresolved_product_forces_uncertain() -> None:
    """8. Unresolved product visible in package prevents confident verification (UNCERTAIN)."""
    unres_item = ResolvedObjectItem(
        object_id="obj-mystery",
        resolved_sku=None,
        status=ResolutionStatus.UNRESOLVED,
        semantic_type="product",
        unresolved_reason="UNKNOWN_PRODUCT",
        confidence=0.0,
        bbox=BoundingBox2D(ymin=0.2, xmin=0.2, ymax=0.5, xmax=0.5),
    )
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        unresolved_products=[unres_item],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "UNCERTAIN"
    assert "obj-mystery" in dec.evidence_object_ids
    assert any("Uncatalogued" in issue for issue in dec.blocking_issues)


def test_blurry_low_confidence_photo_uncertain() -> None:
    """9. Blurry or low-clarity photo forces UNCERTAIN regardless of reconciliation state."""
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")]
    )
    obs = PackageVisualObservation(
        image_id="img-blur",
        image_quality="blurry",
        clarity_score=0.35,
    )

    dec = evaluate_decision(rec, observation=obs)

    assert dec.decision == "UNCERTAIN"
    assert dec.confidence == "low"
    assert "blurry" in dec.reason.lower()


def test_multiple_defects_stops_fix() -> None:
    """10. Multiple defects (missing + mismatch) yield STOP_FIX."""
    rec = _make_reconciliation(
        missing=[ReconciledLine(sku="ITEM-A", expected_qty=1, observed_qty=0, delta=-1, status="MISSING")],
        quantity_mismatches=[ReconciledLine(sku="ITEM-B", expected_qty=2, observed_qty=1, delta=-1, status="QUANTITY_MISMATCH")],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert len(dec.blocking_issues) == 2


def test_precedence_uncertainty_over_confirmed_mismatch() -> None:
    """11. Precedence: UNCERTAIN > STOP_FIX when unresolved evidence co-occurs with confirmed shortage."""
    # Example: 1 item missing, BUT an ambiguous object exists that could be that item
    rec = _make_reconciliation(
        missing=[ReconciledLine(sku="MANUAL-01", expected_qty=1, observed_qty=0, delta=-1, status="MISSING")],
        unverified=[
            UnverifiedLine(
                expected_sku="CAP-BLU-01",
                expected_qty=1,
                observed_resolved_qty=0,
                unverified_qty=1,
                possible_object_ids=["obj-1"],
                reason="Ambiguous cap",
            )
        ],
    )

    dec = evaluate_decision(rec)

    # UNCERTAIN must take priority because the carton state cannot be reliably certified
    assert dec.decision == "UNCERTAIN"
    assert any("CAP-BLU-01" in issue for issue in dec.blocking_issues)
    assert any("MANUAL-01" in issue for issue in dec.blocking_issues)


def test_never_seal_when_unverified_exists() -> None:
    """12. Never SEAL when unverified items exist."""
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="SHIRT-BLK-M", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        unverified=[
            UnverifiedLine(
                expected_sku="CAP-BLU-01",
                expected_qty=1,
                observed_resolved_qty=0,
                unverified_qty=1,
                possible_object_ids=["obj-amb"],
                reason="Ambiguous shade",
            )
        ],
    )

    dec = evaluate_decision(rec)

    assert dec.decision != "SEAL"
    assert dec.decision == "UNCERTAIN"


def test_never_seal_when_unresolved_product_exists() -> None:
    """13. Never SEAL when an uncatalogued unresolved product exists in package."""
    unres_item = ResolvedObjectItem(
        object_id="obj-unknown",
        resolved_sku=None,
        status=ResolutionStatus.UNRESOLVED,
        semantic_type="product",
        unresolved_reason="UNKNOWN_PRODUCT",
        confidence=0.0,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.3, xmax=0.3),
    )
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        unresolved_products=[unres_item],
    )

    dec = evaluate_decision(rec)

    assert dec.decision != "SEAL"
    assert dec.decision == "UNCERTAIN"


def test_foreign_object_triggers_stop_fix() -> None:
    """14. Foreign object in package triggers STOP_FIX."""
    fo_item = ResolvedObjectItem(
        object_id="obj-phone",
        resolved_sku=None,
        status=ResolutionStatus.UNRESOLVED,
        semantic_type="foreign_object",
        unresolved_reason="FOREIGN_OBJECT",
        confidence=0.99,
        bbox=BoundingBox2D(ymin=0.6, xmin=0.6, ymax=0.8, xmax=0.8),
    )
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
        foreign_objects=[fo_item],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "STOP_FIX"
    assert "obj-phone" in dec.evidence_object_ids
    assert any("foreign object" in issue.lower() for issue in dec.blocking_issues)


def test_seal_only_when_everything_confidently_matches() -> None:
    """15. SEAL produces high confidence only when all manifest rows match perfectly."""
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="SKU-1", expected_qty=3, observed_qty=3, delta=0, status="MATCHED", object_ids=["o1", "o2", "o3"])],
    )

    dec = evaluate_decision(rec)

    assert dec.decision == "SEAL"
    assert dec.confidence == "high"
    assert dec.blocking_issues == []


def test_agent_decide_integration() -> None:
    """16. PackManagerAIAgent.decide integration method."""
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    rec = _make_reconciliation(
        matched=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED", object_ids=["obj-1"])],
    )

    dec = agent.decide(reconciliation=rec)

    assert dec.decision == "SEAL"
    assert isinstance(dec, DecisionResult)


def test_container_damage_does_not_block_seal() -> None:
    """Container damage must NOT trigger STOP_FIX if contents match exactly and photo is clear."""
    from pack_manager.agent_schemas import ContainerObservation
    rec = ManifestReconciliation(
        order_id="ORD-001",
        matched=[ReconciledLine(sku="SKU-1", expected_qty=1, observed_qty=1, delta=0, status="MATCHED")],
    )
    obs = PackageVisualObservation(
        image_id="img-0",
        image_quality="clear",
        clarity_score=1.0,
        container=ContainerObservation(container_type="cardboard_box", visible_damage=True, damage_notes="Crushed box flap"),
        observed_items=[ObservedPhysicalItem(object_id="obj-1", label="Item", confidence=0.95, bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5))],
    )
    result = evaluate_decision(reconciliation=rec, observation=obs)
    assert result.decision == "SEAL"
    assert result.confidence == "high"
