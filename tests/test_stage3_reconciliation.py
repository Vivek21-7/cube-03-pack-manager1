"""Tests for STAGE 3: DETERMINISTIC MANIFEST RECONCILIATION

Verifies:
1. exact match
2. missing SKU
3. extra SKU
4. wrong item = missing expected + extra observed
5. quantity too low (under-pack)
6. quantity too high (over-pack)
7. multiple identical resolved objects aggregate correctly
8. ambiguous candidate prevents false missing
9. ambiguous candidate prevents false extra
10. unresolved product preserved
11. foreign object preserved
12. order with multiple SKUs
13. no observed products (empty observation)
14. duplicate order lines for same SKU aggregate correctly
15. agent.reconcile_manifest integration test
"""

import pytest
from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import (
    BoundingBox2D,
    CandidateSKU,
    PackageSKUResolution,
    ResolutionStatus,
    ResolvedObjectItem,
)
from pack_manager.manifest_reconciler import reconcile_manifest
from pack_manager.models import Order, OrderLine, OrderStatus
from pack_manager.vlm_client import SimulationVLMClient


def _make_resolved_item(obj_id: str, sku: str, name: str = "") -> ResolvedObjectItem:
    return ResolvedObjectItem(
        object_id=obj_id,
        resolved_sku=sku,
        status=ResolutionStatus.RESOLVED,
        confidence=0.95,
        candidate_skus=[CandidateSKU(sku=sku, score=0.95, product_name=name or sku)],
        matching_evidence=[f"Visual attributes match {sku}"],
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )


def _make_ambiguous_item(obj_id: str, candidate_skus: list[str]) -> ResolvedObjectItem:
    return ResolvedObjectItem(
        object_id=obj_id,
        resolved_sku=None,
        status=ResolutionStatus.AMBIGUOUS,
        confidence=0.70,
        candidate_skus=[CandidateSKU(sku=s, score=0.70) for s in candidate_skus],
        matching_evidence=["Ambiguous visual characteristics"],
        ambiguity_reason=f"Cannot distinguish {candidate_skus}",
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )


def test_exact_match() -> None:
    """1. Exact match: Every expected item count matches observed count exactly."""
    order = {
        "order_id": "ord-exact",
        "lines": [
            {"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"},
            {"sku": "TSHIRT-BLK-M", "expected_qty": 2, "product_name": "Black T-Shirt"},
        ],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "CAP-BLU-01"),
            _make_resolved_item("obj-2", "TSHIRT-BLK-M"),
            _make_resolved_item("obj-3", "TSHIRT-BLK-M"),
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert rec.is_perfect_match is True
    assert len(rec.matched) == 2
    assert len(rec.missing) == 0
    assert len(rec.extra) == 0
    assert len(rec.quantity_mismatches) == 0
    assert len(rec.unverified) == 0

    cap = next(m for m in rec.matched if m.sku == "CAP-BLU-01")
    assert cap.expected_qty == 1 and cap.observed_qty == 1 and cap.delta == 0
    assert cap.object_ids == ["obj-1"]

    shirt = next(m for m in rec.matched if m.sku == "TSHIRT-BLK-M")
    assert shirt.expected_qty == 2 and shirt.observed_qty == 2 and shirt.delta == 0
    assert shirt.object_ids == ["obj-2", "obj-3"]


def test_missing_sku() -> None:
    """2. Missing SKU: Expected SKU absent from package with no ambiguous candidates."""
    order = {
        "order_id": "ord-missing",
        "lines": [
            {"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"},
            {"sku": "MANUAL-01", "expected_qty": 1, "product_name": "User Guide"},
        ],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "CAP-BLU-01")],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert len(rec.missing) == 1
    assert rec.missing[0].sku == "MANUAL-01"
    assert rec.missing[0].expected_qty == 1
    assert rec.missing[0].observed_qty == 0
    assert rec.missing[0].delta == -1
    assert rec.is_perfect_match is False


def test_extra_sku() -> None:
    """3. Extra SKU: Observed resolved SKU not on the expected order manifest."""
    order = {
        "order_id": "ord-extra",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "CAP-BLU-01"),
            _make_resolved_item("obj-2", "SCARF-RED-01", "Red Wool Scarf"),
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert len(rec.extra) == 1
    assert rec.extra[0].sku == "SCARF-RED-01"
    assert rec.extra[0].observed_qty == 1
    assert rec.extra[0].expected_qty == 0
    assert rec.extra[0].delta == 1
    assert rec.is_perfect_match is False


def test_wrong_item_representation() -> None:
    """4. Wrong item: Represented deterministically as expected missing + unexpected extra."""
    order = {
        "order_id": "ord-wrong",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "CAP-RED-01", "Red Cap")],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 0
    # Expected Blue Cap is missing
    assert len(rec.missing) == 1
    assert rec.missing[0].sku == "CAP-BLU-01"
    # Unexpected Red Cap is extra
    assert len(rec.extra) == 1
    assert rec.extra[0].sku == "CAP-RED-01"


def test_quantity_too_low() -> None:
    """5. Quantity too low: Expected 2, observed 1 with no ambiguous candidates."""
    order = {
        "order_id": "ord-low",
        "lines": [{"sku": "SHIRT-BLK-M", "expected_qty": 2, "product_name": "T-Shirt"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "SHIRT-BLK-M")],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 0
    assert len(rec.missing) == 0
    assert len(rec.quantity_mismatches) == 1
    assert rec.quantity_mismatches[0].sku == "SHIRT-BLK-M"
    assert rec.quantity_mismatches[0].expected_qty == 2
    assert rec.quantity_mismatches[0].observed_qty == 1
    assert rec.quantity_mismatches[0].delta == -1


def test_quantity_too_high() -> None:
    """6. Quantity too high: Expected 1, observed 2 units."""
    order = {
        "order_id": "ord-high",
        "lines": [{"sku": "SHIRT-BLK-M", "expected_qty": 1, "product_name": "T-Shirt"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "SHIRT-BLK-M"),
            _make_resolved_item("obj-2", "SHIRT-BLK-M"),
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.quantity_mismatches) == 1
    assert rec.quantity_mismatches[0].sku == "SHIRT-BLK-M"
    assert rec.quantity_mismatches[0].expected_qty == 1
    assert rec.quantity_mismatches[0].observed_qty == 2
    assert rec.quantity_mismatches[0].delta == 1


def test_multiple_identical_resolved_objects_aggregate() -> None:
    """7. Multiple identical resolved objects aggregate correctly into observed_qty."""
    order = {
        "order_id": "ord-multi",
        "lines": [{"sku": "SHIRT-BLK-M", "expected_qty": 3, "product_name": "T-Shirt"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "SHIRT-BLK-M"),
            _make_resolved_item("obj-2", "SHIRT-BLK-M"),
            _make_resolved_item("obj-3", "SHIRT-BLK-M"),
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert rec.matched[0].observed_qty == 3
    assert rec.matched[0].object_ids == ["obj-1", "obj-2", "obj-3"]


def test_ambiguous_candidate_prevents_false_missing() -> None:
    """8. Ambiguous observation with expected candidate prevents false MISSING."""
    order = {
        "order_id": "ord-amb",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"}],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_ambiguous_item("obj-1", ["CAP-BLU-01", "CAP-NAVY-01"])],
    )

    rec = reconcile_manifest(order, resolution)

    # Must NOT declare CAP-BLU-01 as missing!
    assert len(rec.missing) == 0
    # Must declare as UNVERIFIED
    assert len(rec.unverified) == 1
    unver = rec.unverified[0]
    assert unver.expected_sku == "CAP-BLU-01"
    assert unver.possible_object_ids == ["obj-1"]
    assert any(c.sku == "CAP-BLU-01" for c in unver.candidate_skus)


def test_ambiguous_candidate_prevents_false_extra() -> None:
    """9. Ambiguous observation alternative does NOT get marked as EXTRA."""
    order = {
        "order_id": "ord-amb-extra",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1, "product_name": "Blue Cap"}],
    }
    # obj-1 is ambiguous between CAP-BLU-01 and CAP-NAVY-01
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_ambiguous_item("obj-1", ["CAP-BLU-01", "CAP-NAVY-01"])],
    )

    rec = reconcile_manifest(order, resolution)

    # CAP-NAVY-01 is an alternative candidate of an ambiguous item; it must NOT be marked EXTRA
    assert len(rec.extra) == 0


def test_unresolved_product_preserved() -> None:
    """10. Unresolved product from Stage 2 is preserved separately in unresolved_products."""
    order = {
        "order_id": "ord-unres",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1}],
    }
    unres_item = ResolvedObjectItem(
        object_id="obj-unknown",
        resolved_sku=None,
        status=ResolutionStatus.UNRESOLVED,
        semantic_type="product",
        unresolved_reason="UNKNOWN_PRODUCT",
        confidence=0.0,
        bbox=BoundingBox2D(ymin=0.5, xmin=0.5, ymax=0.8, xmax=0.8),
    )
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "CAP-BLU-01"), unres_item],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert len(rec.unresolved_products) == 1
    assert rec.unresolved_products[0].object_id == "obj-unknown"
    assert rec.unresolved_products[0].semantic_type == "product"


def test_foreign_object_preserved() -> None:
    """11. Foreign object from Stage 2 is preserved separately in foreign_objects."""
    order = {
        "order_id": "ord-fo",
        "lines": [{"sku": "CAP-BLU-01", "expected_qty": 1}],
    }
    fo_item = ResolvedObjectItem(
        object_id="obj-scissors",
        resolved_sku=None,
        status=ResolutionStatus.UNRESOLVED,
        semantic_type="foreign_object",
        unresolved_reason="FOREIGN_OBJECT",
        confidence=0.98,
        bbox=BoundingBox2D(ymin=0.7, xmin=0.7, ymax=0.9, xmax=0.9),
    )
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "CAP-BLU-01"), fo_item],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert len(rec.foreign_objects) == 1
    assert rec.foreign_objects[0].object_id == "obj-scissors"
    assert rec.has_foreign_objects is True
    assert rec.is_perfect_match is False


def test_order_with_multiple_skus() -> None:
    """12. Order with multiple SKUs across matched, missing, extra, and quantity mismatch."""
    order = {
        "order_id": "ord-mixed",
        "lines": [
            {"sku": "SKU-A", "expected_qty": 1},
            {"sku": "SKU-B", "expected_qty": 2},
            {"sku": "SKU-C", "expected_qty": 1},
        ],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "SKU-A"),
            _make_resolved_item("obj-2", "SKU-B"),  # only 1 of 2 SKU-B
            _make_resolved_item("obj-3", "SKU-D"),  # un-ordered extra
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1 and rec.matched[0].sku == "SKU-A"
    assert len(rec.quantity_mismatches) == 1 and rec.quantity_mismatches[0].sku == "SKU-B"
    assert len(rec.missing) == 1 and rec.missing[0].sku == "SKU-C"
    assert len(rec.extra) == 1 and rec.extra[0].sku == "SKU-D"


def test_no_observed_products() -> None:
    """13. No observed products (empty resolution) marks all expected lines missing."""
    order = {
        "order_id": "ord-empty",
        "lines": [
            {"sku": "SKU-1", "expected_qty": 1},
            {"sku": "SKU-2", "expected_qty": 3},
        ],
    }
    resolution = PackageSKUResolution(image_id="img-1", resolved_objects=[])

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 0
    assert len(rec.missing) == 2
    assert rec.total_expected_units == 4
    assert rec.total_observed_resolved_units == 0


def test_duplicate_order_lines_aggregate_correctly() -> None:
    """14. Duplicate order lines for the same SKU aggregate correctly."""
    order = {
        "order_id": "ord-dups",
        "lines": [
            {"sku": "SKU-A", "expected_qty": 1, "product_name": "Widget"},
            {"sku": "SKU-A", "expected_qty": 2, "product_name": "Widget"},
        ],
    }
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[
            _make_resolved_item("obj-1", "SKU-A"),
            _make_resolved_item("obj-2", "SKU-A"),
            _make_resolved_item("obj-3", "SKU-A"),
        ],
    )

    rec = reconcile_manifest(order, resolution)

    assert len(rec.matched) == 1
    assert rec.matched[0].expected_qty == 3
    assert rec.matched[0].observed_qty == 3
    assert rec.matched[0].delta == 0


def test_agent_reconcile_manifest_integration() -> None:
    """15. PackManagerAIAgent.reconcile_manifest pipeline integration."""
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    order = Order(
        order_id="ord-agent",
        organization_id="org-1",
        client_id="client-1",
        lines=[OrderLine(line_id="l1", sku="CAP-BLU-01", expected_qty=1, product_name="Blue Cap")],
        status=OrderStatus.READY_TO_PACK,
    )
    resolution = PackageSKUResolution(
        image_id="img-1",
        resolved_objects=[_make_resolved_item("obj-1", "CAP-BLU-01")],
    )

    rec = agent.reconcile_manifest(order, resolution)

    assert rec.order_id == "ord-agent"
    assert rec.is_perfect_match is True
    assert len(rec.matched) == 1
