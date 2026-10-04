"""Extra-item detection: something is in the box that is not on the order."""

from __future__ import annotations

import time

from pack_manager.checks._common import (
    RULES_MODEL,
    UNCERTAIN_SKU,
    UNKNOWN_SKU,
    expected_qty_by_sku,
    mean_confidence,
    timed_ms,
)
from pack_manager.models import CheckKey, CheckResult, DetectedItem, Order, Verdict


def run_extra_item_detection(
    order: Order,
    detections: list[DetectedItem],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()
    expected = expected_qty_by_sku(order)

    if not detections or any(
        d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45 for d in detections
    ):
        return CheckResult(
            check_key=CheckKey.EXTRA_ITEM_DETECTION,
            verdict=Verdict.UNCERTAIN,
            confidence=mean_confidence(detections),
            detail=(
                "Cannot rule extra items in or out: the photograph has no confident "
                "identifications, or at least one detection is ambiguous."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    extras: list[str] = []
    observed_counts: dict[str, int] = {}
    for det in detections:
        sku = det.sku or UNKNOWN_SKU
        observed_counts[sku] = observed_counts.get(sku, 0) + det.quantity

    for sku, qty in observed_counts.items():
        if sku not in expected:
            extras.append(f"{sku} x{qty}")
        elif qty > expected[sku]:
            extras.append(f"{sku} extra x{qty - expected[sku]}")

    if extras:
        return CheckResult(
            check_key=CheckKey.EXTRA_ITEM_DETECTION,
            verdict=Verdict.FAIL,
            confidence=mean_confidence(detections),
            detail="Unexpected item(s) visible in the pack: " + ", ".join(extras) + ".",
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    return CheckResult(
        check_key=CheckKey.EXTRA_ITEM_DETECTION,
        verdict=Verdict.PASS,
        confidence=max(mean_confidence(detections), 0.5),
        detail="No extra or unordered items are visible beyond ordered SKUs and quantities.",
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
