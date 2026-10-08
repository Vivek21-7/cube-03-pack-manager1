"""Tests for STAGE 5: GROUNDED EVIDENCE & AUDIT TRAIL

Verifies:
1. SEAL report contains grounded evidence for all matched objects
2. STOP_FIX report links missing/extra evidence correctly
3. quantity mismatch links all observed object IDs
4. ambiguous item preserves candidate SKUs and scores
5. UNCERTAIN never fabricates resolved SKU
6. missing item does not invent image evidence
7. every resolved observed object has image_id + object_id + bbox
8. decision in Stage 5 exactly equals Stage 4 decision
9. Stage 5 cannot modify decision
10. report serializes cleanly to JSON
11. agent.create_verification_report integration
"""

import json
import pytest
from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import (
    BoundingBox2D,
    CandidateSKU,
    DecisionResult,
    ManifestReconciliation,
    PackageSKUResolution,
    PackageVisualObservation,
    PackVerificationReport,
    ReconciledLine,
    ResolutionStatus,
    ResolvedObjectItem,
    UnverifiedLine,
)
from pack_manager.grounded_audit import build_verification_report
from pack_manager.vlm_client import SimulationVLMClient


def _make_sample_pipeline(
    scenario: str = "seal",
) -> tuple[dict, PackageVisualObservation, PackageSKUResolution, ManifestReconciliation, DecisionResult]:
    order = {
        "order_id": f"ord-{scenario}",
        "lines": [
            {"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"},
            {"sku": "SHIRT-BLK-M", "expected_qty": 2, "product_name": "Black Shirt"},
        ],
    }

    box1 = BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4)
    box2 = BoundingBox2D(ymin=0.2, xmin=0.2, ymax=0.6, xmax=0.6)
    box3 = BoundingBox2D(ymin=0.3, xmin=0.3, ymax=0.7, xmax=0.7)

    if scenario == "seal":
        obs = PackageVisualObservation(
            image_id="img-seal",
            observed_items=[
                PackageVisualObservation.__fields__["observed_items"].default_factory()[0]  # will populate below
            ] if False else [
                {"object_id": "obj-1", "label": "Blue Baseball Cap", "confidence": 0.95, "bbox": box1, "image_id": "img-seal"},
                {"object_id": "obj-2", "label": "Black Cotton Shirt", "confidence": 0.92, "bbox": box2, "image_id": "img-seal"},
                {"object_id": "obj-3", "label": "Black Cotton Shirt", "confidence": 0.91, "bbox": box3, "image_id": "img-seal"},
            ],  # type: ignore
        )
        res = PackageSKUResolution(
            image_id="img-seal",
            resolved_objects=[
                ResolvedObjectItem(
                    object_id="obj-1", resolved_sku="CAP-BLU-01", status=ResolutionStatus.RESOLVED, confidence=0.95,
                    candidate_skus=[CandidateSKU(sku="CAP-BLU-01", score=0.95, product_name="Blue Cap")],
                    matching_evidence=["Color blue confirmed"], bbox=box1, image_id="img-seal",
                ),
                ResolvedObjectItem(
                    object_id="obj-2", resolved_sku="TSHIRT-BLK-M", status=ResolutionStatus.RESOLVED, confidence=0.92,
                    candidate_skus=[CandidateSKU(sku="TSHIRT-BLK-M", score=0.92, product_name="Black Shirt")],
                    matching_evidence=["Form shirt confirmed"], bbox=box2, image_id="img-seal",
                ),
                ResolvedObjectItem(
                    object_id="obj-3", resolved_sku="TSHIRT-BLK-M", status=ResolutionStatus.RESOLVED, confidence=0.91,
                    candidate_skus=[CandidateSKU(sku="TSHIRT-BLK-M", score=0.91, product_name="Black Shirt")],
                    matching_evidence=["Form shirt confirmed"], bbox=box3, image_id="img-seal",
                ),
            ],
        )
        rec = ManifestReconciliation(
            order_id="ord-seal",
            matched=[
                ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=1, delta=0, status="MATCHED", object_ids=["obj-1"]),
                ReconciledLine(sku="TSHIRT-BLK-M", expected_qty=2, observed_qty=2, delta=0, status="MATCHED", object_ids=["obj-2", "obj-3"]),
            ],
        )
        dec = DecisionResult(
            decision="SEAL",
            reason="All 3 units verified across 2 SKUs. Packaging approved to seal.",
            confidence="high",
            resolved_skus_count=3,
            expected_skus_count=3,
        )
        return order, obs, res, rec, dec

    elif scenario == "wrong_item":
        obs = PackageVisualObservation(
            image_id="img-wrong",
            observed_items=[
                {"object_id": "obj-1", "label": "Red Baseball Cap", "confidence": 0.95, "bbox": box1, "image_id": "img-wrong"},
            ],  # type: ignore
        )
        res = PackageSKUResolution(
            image_id="img-wrong",
            resolved_objects=[
                ResolvedObjectItem(
                    object_id="obj-1", resolved_sku="CAP-RED-01", status=ResolutionStatus.RESOLVED, confidence=0.95,
                    candidate_skus=[CandidateSKU(sku="CAP-RED-01", score=0.95, product_name="Red Cap")],
                    matching_evidence=["Observed color is red"], bbox=box1, image_id="img-wrong",
                ),
            ],
        )
        rec = ManifestReconciliation(
            order_id="ord-wrong",
            missing=[ReconciledLine(sku="CAP-BLU-01", expected_qty=1, observed_qty=0, delta=-1, status="MISSING")],
            extra=[ReconciledLine(sku="CAP-RED-01", expected_qty=0, observed_qty=1, delta=1, status="EXTRA", object_ids=["obj-1"])],
        )
        dec = DecisionResult(
            decision="STOP_FIX",
            reason="CRITICAL DEFECT: Product substitution / wrong item detected. DO NOT SEAL.",
            confidence="high",
            blocking_issues=["Missing: CAP-BLU-01", "Extra: CAP-RED-01"],
            evidence_object_ids=["obj-1"],
            resolved_skus_count=1,
            expected_skus_count=1,
        )
        return order, obs, res, rec, dec

    elif scenario == "ambiguous":
        obs = PackageVisualObservation(
            image_id="img-amb",
            observed_items=[
                {"object_id": "obj-1", "label": "Cap", "confidence": 0.80, "bbox": box1, "image_id": "img-amb"},
            ],  # type: ignore
        )
        res = PackageSKUResolution(
            image_id="img-amb",
            resolved_objects=[
                ResolvedObjectItem(
                    object_id="obj-1", resolved_sku=None, status=ResolutionStatus.AMBIGUOUS, confidence=0.70,
                    candidate_skus=[CandidateSKU(sku="CAP-BLU-01", score=0.72), CandidateSKU(sku="CAP-NAVY-01", score=0.69)],
                    matching_evidence=["Dark blue cap"], ambiguity_reason="Color shade ambiguous", bbox=box1, image_id="img-amb",
                ),
            ],
        )
        rec = ManifestReconciliation(
            order_id="ord-amb",
            unverified=[
                UnverifiedLine(
                    expected_sku="CAP-BLU-01", expected_qty=1, observed_resolved_qty=0, unverified_qty=1,
                    possible_object_ids=["obj-1"],
                    candidate_skus=[CandidateSKU(sku="CAP-BLU-01", score=0.72), CandidateSKU(sku="CAP-NAVY-01", score=0.69)],
                    reason="Shortfall cannot be confirmed missing due to ambiguous obj-1",
                )
            ],
        )
        dec = DecisionResult(
            decision="UNCERTAIN",
            reason="Verification uncertain: Expected SKU CAP-BLU-01 ambiguous.",
            confidence="medium",
            blocking_issues=["Expected SKU CAP-BLU-01 ambiguous with CAP-NAVY-01"],
            evidence_object_ids=["obj-1"],
            resolved_skus_count=0,
            expected_skus_count=1,
        )
        return order, obs, res, rec, dec

    raise ValueError(f"Unknown scenario: {scenario}")


def test_seal_report_contains_evidence_for_all_matched_objects() -> None:
    """1. SEAL report contains grounded evidence for all matched objects."""
    order, obs, res, rec, dec = _make_sample_pipeline("seal")
    report = build_verification_report(
        reconciliation=rec,
        decision=dec,
        observation=obs,
        resolution=res,
        order=order,
        image_bytes=b"dummy-image-bytes",
    )

    assert report.decision == "SEAL"
    assert report.confidence == "high"
    assert report.operator_action == "Seal package"
    assert len(report.matched_items) == 2
    assert len(report.evidence_records) == 3

    # Check evidence traceability
    ev_obj_ids = [e.object_id for e in report.evidence_records]
    assert "obj-1" in ev_obj_ids
    assert "obj-2" in ev_obj_ids
    assert "obj-3" in ev_obj_ids

    # Check grounded summary
    e1 = next(e for e in report.evidence_records if e.object_id == "obj-1")
    assert e1.resolved_sku == "CAP-BLU-01"
    assert e1.reconciliation_status == "MATCHED"
    assert "resolved as CAP-BLU-01" in e1.grounded_summary
    assert report.image_sha256.startswith("sha256:")


def test_stop_fix_report_links_missing_extra_evidence() -> None:
    """2. STOP_FIX report links missing and extra evidence correctly."""
    order, obs, res, rec, dec = _make_sample_pipeline("wrong_item")
    report = build_verification_report(
        reconciliation=rec,
        decision=dec,
        observation=obs,
        resolution=res,
        order=order,
    )

    assert report.decision == "STOP_FIX"
    assert report.operator_action == "Correct package contents before sealing"
    assert len(report.missing_items) == 1
    assert report.missing_items[0].sku == "CAP-BLU-01"

    assert len(report.extra_items) == 1
    assert report.extra_items[0].sku == "CAP-RED-01"

    # Evidence record strictly exists for the genuinely observed object (obj-1 / Red Cap)
    assert len(report.evidence_records) == 1
    e = report.evidence_records[0]
    assert e.object_id == "obj-1"
    assert e.resolved_sku == "CAP-RED-01"
    assert e.reconciliation_status == "EXTRA"


def test_missing_item_does_not_invent_image_evidence() -> None:
    """3. Missing item must NOT invent or fabricate an image evidence record."""
    order, obs, res, rec, dec = _make_sample_pipeline("wrong_item")
    report = build_verification_report(
        reconciliation=rec,
        decision=dec,
        observation=obs,
        resolution=res,
        order=order,
    )

    # Expected Blue Cap is missing; there should be NO evidence record for it
    evidence_skus = [e.resolved_sku for e in report.evidence_records]
    assert "CAP-BLU-01" not in evidence_skus
    # Missing item record has empty object_ids
    assert report.missing_items[0].object_ids == []


def test_quantity_mismatch_links_all_observed_object_ids() -> None:
    """4. Quantity mismatch links all observed object IDs."""
    order = {"order_id": "ord-qm", "lines": [{"sku": "SHIRT-BLK-M", "expected_qty": 1}]}
    box1 = BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4)
    box2 = BoundingBox2D(ymin=0.5, xmin=0.5, ymax=0.8, xmax=0.8)

    obs = PackageVisualObservation(
        image_id="img-qm",
        observed_items=[
            {"object_id": "obj-1", "label": "Shirt", "confidence": 0.9, "bbox": box1, "image_id": "img-qm"},
            {"object_id": "obj-2", "label": "Shirt", "confidence": 0.9, "bbox": box2, "image_id": "img-qm"},
        ],  # type: ignore
    )
    res = PackageSKUResolution(
        image_id="img-qm",
        resolved_objects=[
            ResolvedObjectItem(object_id="obj-1", resolved_sku="SHIRT-BLK-M", status=ResolutionStatus.RESOLVED, confidence=0.9, bbox=box1, image_id="img-qm"),
            ResolvedObjectItem(object_id="obj-2", resolved_sku="SHIRT-BLK-M", status=ResolutionStatus.RESOLVED, confidence=0.9, bbox=box2, image_id="img-qm"),
        ],
    )
    rec = ManifestReconciliation(
        order_id="ord-qm",
        quantity_mismatches=[
            ReconciledLine(sku="SHIRT-BLK-M", expected_qty=1, observed_qty=2, delta=1, status="QUANTITY_MISMATCH", object_ids=["obj-1", "obj-2"]),
        ],
    )
    dec = DecisionResult(decision="STOP_FIX", reason="Over-pack", confidence="high", evidence_object_ids=["obj-1", "obj-2"])

    report = build_verification_report(reconciliation=rec, decision=dec, observation=obs, resolution=res, order=order)

    assert len(report.quantity_mismatches) == 1
    assert report.quantity_mismatches[0].object_ids == ["obj-1", "obj-2"]
    assert len(report.evidence_records) == 2
    assert {e.object_id for e in report.evidence_records} == {"obj-1", "obj-2"}


def test_ambiguous_item_preserves_candidate_skus() -> None:
    """5. Ambiguous item preserves candidate SKUs, scores, and does NOT fabricate resolved SKU."""
    order, obs, res, rec, dec = _make_sample_pipeline("ambiguous")
    report = build_verification_report(reconciliation=rec, decision=dec, observation=obs, resolution=res, order=order)

    assert report.decision == "UNCERTAIN"
    assert report.operator_action == "Capture another image or perform manual verification"
    assert len(report.evidence_records) == 1

    ev = report.evidence_records[0]
    assert ev.object_id == "obj-1"
    assert ev.resolved_sku is None  # Must NOT fabricate resolved SKU
    assert ev.sku_resolution_status == ResolutionStatus.AMBIGUOUS
    assert len(ev.candidate_skus) == 2
    assert {c.sku for c in ev.candidate_skus} == {"CAP-BLU-01", "CAP-NAVY-01"}
    assert "ambiguous between candidate SKUs" in ev.grounded_summary


def test_every_resolved_observed_object_has_image_id_object_id_bbox() -> None:
    """6. Every evidence record must include image_id, object_id, and non-empty valid bbox."""
    order, obs, res, rec, dec = _make_sample_pipeline("seal")
    report = build_verification_report(reconciliation=rec, decision=dec, observation=obs, resolution=res, order=order)

    for ev in report.evidence_records:
        assert ev.image_id != ""
        assert ev.object_id != ""
        assert isinstance(ev.bbox, BoundingBox2D)
        assert 0.0 <= ev.bbox.ymin <= ev.bbox.ymax <= 1.0
        assert 0.0 <= ev.bbox.xmin <= ev.bbox.xmax <= 1.0
        assert ev.bbox.is_valid is True


def test_decision_in_stage5_strictly_equals_stage4_decision() -> None:
    """7. Stage 5 must strictly preserve Stage 4 decision and cannot modify it."""
    for scen in ["seal", "wrong_item", "ambiguous"]:
        order, obs, res, rec, dec = _make_sample_pipeline(scen)
        report = build_verification_report(reconciliation=rec, decision=dec, observation=obs, resolution=res, order=order)
        assert report.decision == dec.decision
        assert report.confidence == dec.confidence
        assert report.decision_reason == dec.reason


def test_report_serializes_cleanly_to_json() -> None:
    """8. Final report serializes cleanly to JSON without error."""
    order, obs, res, rec, dec = _make_sample_pipeline("seal")
    report = build_verification_report(
        reconciliation=rec,
        decision=dec,
        observation=obs,
        resolution=res,
        order=order,
        image_bytes=b"sample-bytes",
    )

    json_str = report.model_dump_json(indent=2)
    assert json_str is not None
    data = json.loads(json_str)

    assert data["order_id"] == "ord-seal"
    assert data["decision"] == "SEAL"
    assert len(data["evidence_records"]) == 3
    assert data["operator_action"] == "Seal package"
    assert "image_sha256" in data


def test_agent_create_verification_report_integration() -> None:
    """9. PackManagerAIAgent.create_verification_report integration method."""
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    order, obs, res, rec, dec = _make_sample_pipeline("seal")

    report = agent.create_verification_report(
        reconciliation=rec,
        decision=dec,
        observation=obs,
        resolution=res,
        order=order,
        image_sha256="sha256:abc12345",
    )

    assert isinstance(report, PackVerificationReport)
    assert report.is_sealed is True
    assert report.image_sha256 == "sha256:abc12345"
