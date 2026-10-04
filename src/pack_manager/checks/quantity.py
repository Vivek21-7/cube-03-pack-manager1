"""Quantity counting: observed instance counts vs expected order quantities."""

from __future__ import annotations

import time

from pack_manager.checks._common import (
    RULES_MODEL,
    UNCERTAIN_SKU,
    UNKNOWN_SKU,
    expected_qty_by_sku,
    mean_confidence,
    observed_qty_by_sku,
    timed_ms,
)
from pack_manager.models import CheckKey, CheckResult, DetectedItem, Order, Verdict


def run_quantity_match(
    order: Order,
    detections: list[DetectedItem],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()
    expected = expected_qty_by_sku(order)
    observed = observed_qty_by_sku(detections)

    if any(d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45 for d in detections) or not detections:
        return CheckResult(
            check_key=CheckKey.QUANTITY_MATCH,
            verdict=Verdict.UNCERTAIN,
            confidence=mean_confidence(detections),
            detail=(
                "Counts cannot be certified: detections are missing, ambiguous, "
                "or below the confidence floor. UNCERTAIN is not a low-confidence PASS."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    mismatches: list[str] = []
    for sku, exp_qty in expected.items():
        obs_qty = observed.get(sku, 0)
        if obs_qty != exp_qty:
            mismatches.append(f"{sku}: expected {exp_qty}, observed {obs_qty}")

    extras_unknown = observed.get(UNKNOWN_SKU, 0)
    if extras_unknown:
        mismatches.append(f"unidentified instances counted: {extras_unknown}")

    if mismatches:
        return CheckResult(
            check_key=CheckKey.QUANTITY_MATCH,
            verdict=Verdict.FAIL,
            confidence=mean_confidence(detections),
            detail="Quantity mismatch grounded in detections vs order: " + "; ".join(mismatches) + ".",
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    rows = ", ".join(f"{sku}={observed.get(sku, 0)}" for sku in expected)
    return CheckResult(
        check_key=CheckKey.QUANTITY_MATCH,
        verdict=Verdict.PASS,
        confidence=mean_confidence(detections),
        detail=f"Observed quantities match the order for every SKU ({rows}).",
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
