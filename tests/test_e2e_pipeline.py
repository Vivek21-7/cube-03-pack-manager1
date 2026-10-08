"""Dedicated End-to-End Verification Pipeline Tests (Stages 1-5 & /api/verify).

Validates canonical 5-stage pipeline:
1. observe_package(photo) -> Stage 1 PackageVisualObservation (order-blind)
2. resolve_skus(observation, catalogue) -> Stage 2 PackageSKUResolution (order-blind)
3. reconcile_manifest(order, resolution) -> Stage 3 ManifestReconciliation (pure Python)
4. decide(reconciliation, observation) -> Stage 4 DecisionResult (deterministic precedence)
5. create_verification_report(...) -> Stage 5 PackVerificationReport (traceable audit evidence)

Strictly asserts final PackVerificationReport, NOT legacy models.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import (
    PackVerificationReport,
    ResolutionStatus,
)
from pack_manager.catalogue_resolver import DEMO_FIXTURE_CATALOGUE
from pack_manager.fixture_scenarios import generate_all_scenarios
from pack_manager.models import CatalogItem
from pack_manager.server import PackManagerRequestHandler
from pack_manager.vlm_client import SimulationVLMClient

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "scenarios"
SERVER_PORT = 8993
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict[str, Path]]:
    return generate_all_scenarios(FIXTURES)


@pytest.fixture(scope="module")
def live_server():
    server = ThreadingHTTPServer(("127.0.0.1", SERVER_PORT), PackManagerRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield BASE_URL
    server.shutdown()
    server.server_close()


# ==============================================================================
# 1. CORRECT ORDER -> SEAL
# ==============================================================================

def test_e2e_correct_order_seals(scenarios: dict[str, dict[str, Path]], live_server: str) -> None:
    """Correct order with all items present and undamaged must SEAL with full audit trail."""
    scen = scenarios["example_1_correct_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    # Direct agent.verify_full()
    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert isinstance(report, PackVerificationReport)
    assert report.decision == "SEAL"
    assert report.confidence == "high"
    assert report.operator_action == "Seal package"
    assert len(report.missing_items) == 0
    assert len(report.extra_items) == 0
    assert len(report.quantity_mismatches) == 0
    assert len(report.unverified_items) == 0
    assert len(report.matched_items) == 2
    assert report.image_sha256.startswith("sha256:")

    # Verify every observed unit has a grounded evidence record with bbox
    assert len(report.evidence_records) == 3  # 2 shirts + 1 cap
    for ev in report.evidence_records:
        assert ev.image_id == "box-photo-0"
        assert ev.object_id.startswith("obj-")
        assert ev.bbox.is_valid
        assert ev.grounded_summary != ""

    # Test via POST /api/verify
    photo_bytes = Path(scen["photo"]).read_bytes()
    order_dict = json.loads(Path(scen["order"]).read_text(encoding="utf-8"))
    cat_payload = [it.model_dump() for it in DEMO_FIXTURE_CATALOGUE]

    resp = httpx.post(
        f"{live_server}/api/verify",
        json={
            "order": order_dict,
            "catalogue": cat_payload,
            "photo_base64": base64.b64encode(photo_bytes).decode("ascii"),
            "provider": "simulation",
        },
    )
    assert resp.status_code == 200
    api_res = resp.json()
    assert api_res["decision"] == "SEAL"
    assert api_res["confidence"] == "high"
    assert api_res["operator_action"] == "Seal package"
    assert len(api_res["matched_items"]) == 2
    assert len(api_res["evidence_records"]) == 3


# ==============================================================================
# 2. WRONG ITEM -> STOP_FIX
# ==============================================================================

def test_e2e_wrong_item_stops_fix(scenarios: dict[str, dict[str, Path]], live_server: str) -> None:
    """Wrong item variant (Red Cap instead of Blue Cap) must STOP_FIX."""
    scen = scenarios["example_2_wrong_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert isinstance(report, PackVerificationReport)
    assert report.decision == "STOP_FIX"
    assert report.confidence == "high"
    assert report.operator_action == "Correct package contents before sealing"
    
    # Missing the ordered Blue Cap
    assert len(report.missing_items) == 1
    assert report.missing_items[0].sku == "CAP-BLU-001"

    # Extra the unexpected Red Cap
    assert len(report.extra_items) == 1
    assert report.extra_items[0].sku == "CAP-RED-001"

    # Shirts are still matched
    matched_skus = [m.sku for m in report.matched_items]
    assert "TSHIRT-BLK-M" in matched_skus

    # Test via POST /api/verify
    photo_bytes = Path(scen["photo"]).read_bytes()
    order_dict = json.loads(Path(scen["order"]).read_text(encoding="utf-8"))
    resp = httpx.post(
        f"{live_server}/api/verify",
        json={
            "order": order_dict,
            "catalogue": [it.model_dump() for it in DEMO_FIXTURE_CATALOGUE],
            "photo_base64": base64.b64encode(photo_bytes).decode("ascii"),
            "provider": "simulation",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "STOP_FIX"


# ==============================================================================
# 3. MISSING ITEM -> STOP_FIX
# ==============================================================================

def test_e2e_missing_item_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    """Missing user manual documentation must STOP_FIX."""
    scen = scenarios["example_3_missing_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert report.decision == "STOP_FIX"
    assert report.operator_action == "Correct package contents before sealing"
    assert len(report.missing_items) == 1
    assert report.missing_items[0].sku == "DOC-MANUAL-01"
    assert len(report.extra_items) == 0


# ==============================================================================
# 4. EXTRA ITEM -> STOP_FIX
# ==============================================================================

def test_e2e_extra_item_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    """Unexpected extra red scarf in package must STOP_FIX."""
    scen = scenarios["example_4_extra_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert report.decision == "STOP_FIX"
    assert report.operator_action == "Correct package contents before sealing"
    assert len(report.extra_items) == 1
    assert report.extra_items[0].sku == "SCARF-RED-001"
    assert len(report.missing_items) == 0


# ==============================================================================
# 5. WRONG QUANTITY -> STOP_FIX
# ==============================================================================

def test_e2e_wrong_quantity_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    """Quantity mismatch (customer ordered 3 shirts, carton only contains 2) must STOP_FIX."""
    scen = scenarios["example_1_correct_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    # Alter order manifest to expect 3 shirts instead of 2
    order_data = {
        "order_id": "ORD-QTY-MISMATCH",
        "items": [
            {"sku": "TSHIRT-BLK-M", "name": "Black T-Shirt", "expected_qty": 3},
            {"sku": "CAP-BLU-001", "name": "Blue Cap", "expected_qty": 1},
        ],
    }

    report: PackVerificationReport = agent.verify_full(
        order=order_data,
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert report.decision == "STOP_FIX"
    assert report.operator_action == "Correct package contents before sealing"
    assert len(report.quantity_mismatches) == 1
    qm = report.quantity_mismatches[0]
    assert qm.sku == "TSHIRT-BLK-M"
    assert qm.expected_qty == 3
    assert qm.observed_qty == 2
    assert qm.delta == -1


# ==============================================================================
# 6. BLURRY / AMBIGUOUS IMAGE -> UNCERTAIN
# ==============================================================================

def test_e2e_blurry_ambiguous_uncertain(scenarios: dict[str, dict[str, Path]]) -> None:
    """Blurry photo failing clarity gating must return UNCERTAIN."""
    scen = scenarios["example_5_unclear_photo"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert report.decision == "UNCERTAIN"
    assert report.confidence == "low"
    assert report.operator_action == "Capture another image or perform manual verification"
    assert any("blurry" in issue.lower() or "clarity" in issue.lower() for issue in report.blocking_issues)


# ==============================================================================
# 7. MULTIPLE IDENTICAL PRODUCTS -> CORRECT AGGREGATED COUNT
# ==============================================================================

def test_e2e_multiple_identical_products_correct_count(scenarios: dict[str, dict[str, Path]]) -> None:
    """Multiple identical units (obj-1 and obj-2) must aggregate correctly into count 2."""
    scen = scenarios["example_1_correct_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    shirt_match = next(m for m in report.matched_items if m.sku == "TSHIRT-BLK-M")
    assert shirt_match.expected_qty == 2
    assert shirt_match.observed_qty == 2
    assert set(shirt_match.object_ids) == {"obj-1", "obj-2"}

    # Separate bounding boxes for both discrete physical instances
    shirt_evidence = [e for e in report.evidence_records if e.resolved_sku == "TSHIRT-BLK-M"]
    assert len(shirt_evidence) == 2
    assert {e.object_id for e in shirt_evidence} == {"obj-1", "obj-2"}


# ==============================================================================
# 8. UNKNOWN / UNCATALOGUED PRODUCT -> UNCERTAIN
# ==============================================================================

def test_e2e_unknown_uncatalogued_product_uncertain(scenarios: dict[str, dict[str, Path]]) -> None:
    """Visible product not resolving to any catalogue item must trigger UNCERTAIN."""
    scen = scenarios["example_7_electronics_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    # Provide catalogue where the item in box does not match
    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    assert report.decision == "UNCERTAIN"
    assert report.operator_action == "Capture another image or perform manual verification"
    assert any("uncatalogued" in b.lower() or "cannot be resolved" in b.lower() for b in report.blocking_issues)


# ==============================================================================
# 9. DAMAGED BOX (CONTENTS MATCH) -> SEAL
# ==============================================================================

def test_e2e_damaged_box_contents_match_seals(scenarios: dict[str, dict[str, Path]]) -> None:
    """If the carton has visible damage but contents match exactly and evidence is clear:
    Pack Manager decision remains SEAL while container damage is preserved in audit metadata.
    """
    scen = scenarios["example_6_damaged_goods"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    report: PackVerificationReport = agent.verify_full(
        order=scen["order"],
        catalogue=DEMO_FIXTURE_CATALOGUE,
        photo=scen["photo"],
    )

    # Contents match exactly -> SEAL
    assert report.decision == "SEAL"
    assert report.confidence == "high"
    assert report.operator_action == "Seal package"
    assert len(report.matched_items) == 2  # 2 shirts + 1 cap
    assert len(report.missing_items) == 0
    assert len(report.extra_items) == 0
    assert len(report.quantity_mismatches) == 0
    assert len(report.unverified_items) == 0

    # Container damage is still preserved in Stage 1 observation
    obs = agent.observe_package(scen["photo"])
    assert obs.container.visible_damage is True
    assert "crushed" in obs.container.damage_notes.lower() or "punctured" in obs.container.damage_notes.lower()
