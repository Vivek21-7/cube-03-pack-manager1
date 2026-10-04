"""Isolated check-module tests (injected detections, no vision)."""

from pack_manager.checks import (
    run_decision_synthesis,
    run_extra_item_detection,
    run_item_identity,
    run_item_presence,
    run_missing_item_detection,
    run_quantity_match,
)
from pack_manager.models import (
    CheckKey,
    Decision,
    DetectedItem,
    Order,
    OrderLine,
    OrderStatus,
    Verdict,
)


def _order(*skus_qty: tuple[str, int]) -> Order:
    lines = [
        OrderLine(line_id=f"l{i}", sku=sku, expected_qty=qty, product_name=sku)
        for i, (sku, qty) in enumerate(skus_qty)
    ]
    return Order(
        order_id="o1",
        organization_id="org",
        client_id="c",
        lines=lines,
        status=OrderStatus.READY_TO_PACK,
    )


def _det(sku: str | None, qty: int = 1, conf: float = 0.9, label: str | None = None) -> DetectedItem:
    return DetectedItem(
        detection_id="d1",
        sku=sku,
        label=label or (sku or "unknown"),
        quantity=qty,
        confidence=conf,
        image_id="img",
    )


def test_presence_empty_is_uncertain() -> None:
    result = run_item_presence([])
    assert result.verdict == Verdict.UNCERTAIN
    assert result.check_key == CheckKey.ITEM_PRESENCE


def test_quantity_wrong_is_fail() -> None:
    result = run_quantity_match(_order(("CAP-BLUE-001", 2)), [_det("CAP-BLUE-001", qty=1)])
    assert result.verdict == Verdict.FAIL


def test_identity_wrong_colorway_is_fail() -> None:
    result = run_item_identity(_order(("CAP-BLUE-001", 1)), [_det("CAP-RED-001", label="Red Cap")])
    assert result.verdict == Verdict.FAIL


def test_extra_unordered_sku_is_fail() -> None:
    result = run_extra_item_detection(
        _order(("CAP-BLUE-001", 1)),
        [_det("CAP-BLUE-001"), _det("SOCK-001", label="Socks")],
    )
    assert result.verdict == Verdict.FAIL


def test_missing_expected_sku_is_fail() -> None:
    result = run_missing_item_detection(_order(("CAP-BLUE-001", 1), ("SOCK-001", 1)), [_det("CAP-BLUE-001")])
    assert result.verdict == Verdict.FAIL


def test_uncertain_detection_does_not_pass_identity() -> None:
    result = run_item_identity(_order(("CAP-BLUE-001", 1)), [_det("__uncertain__", conf=0.3)])
    assert result.verdict == Verdict.UNCERTAIN


def test_synthesis_seal_only_if_all_pass() -> None:
    order = _order(("CAP-BLUE-001", 1))
    dets = [_det("CAP-BLUE-001")]
    prior = [
        run_item_presence(dets),
        run_quantity_match(order, dets),
        run_item_identity(order, dets),
        run_extra_item_detection(order, dets),
        run_missing_item_detection(order, dets),
    ]
    synth = run_decision_synthesis(prior)
    assert synth.verdict == Verdict.PASS
    from pack_manager.checks.synthesis import synthesize_decision

    assert synthesize_decision(prior) == Decision.SEAL
