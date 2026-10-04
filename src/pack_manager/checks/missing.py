"""Missing-item detection: an ordered SKU is not found in the photograph."""

from __future__ import annotations

import time

from pack_manager.checks._common import (
    RULES_MODEL,
    UNCERTAIN_SKU,
    expected_qty_by_sku,
    mean_confidence,
    observed_qty_by_sku,
    timed_ms,
)
from pack_manager.models import CheckKey, CheckResult, DetectedItem, Order, Verdict


def run_missing_item_detection(
    order: Order,
    detections: list[DetectedItem],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()
    expected = expected_qty_by_sku(order)
    observed = observed_qty_by_sku(detections)

    if not detections or any(
        d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45 for d in detections
    ):
        return CheckResult(
            check_key=CheckKey.MISSING_ITEM_DETECTION,
            verdict=Verdict.UNCERTAIN,
            confidence=mean_confidence(detections),
            detail=(
                "Missing items cannot be asserted: the photograph is empty of confident "
                "identifications. A missing SKU requires positive evidence that it is absent, "
                "or a complete, unambiguous view of the box contents."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    missing = [
        f"{sku} (expected {qty}, observed {observed.get(sku, 0)})"
        for sku, qty in expected.items()
        if observed.get(sku, 0) < qty
    ]
    if missing:
        return CheckResult(
            check_key=CheckKey.MISSING_ITEM_DETECTION,
            verdict=Verdict.FAIL,
            confidence=mean_confidence(detections),
            detail="Ordered item(s) not found at expected quantity: " + "; ".join(missing) + ".",
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    return CheckResult(
        check_key=CheckKey.MISSING_ITEM_DETECTION,
        verdict=Verdict.PASS,
        confidence=mean_confidence(detections),
        detail="Every ordered SKU is visible at least at the expected quantity.",
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
