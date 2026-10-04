"""Wrong-item / identity: a visible item does not match the expected SKU."""

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


def run_item_identity(
    order: Order,
    detections: list[DetectedItem],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()
    expected_skus = set(expected_qty_by_sku(order))

    if not detections:
        return CheckResult(
            check_key=CheckKey.ITEM_IDENTITY,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            detail="No detections available to compare identity against ordered SKUs.",
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    uncertain = [d for d in detections if d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45]
    if uncertain:
        return CheckResult(
            check_key=CheckKey.ITEM_IDENTITY,
            verdict=Verdict.UNCERTAIN,
            confidence=mean_confidence(uncertain),
            detail=(
                "Identity cannot be certified for one or more visible objects "
                "(occlusion, blur, or near-duplicate colourway). Not forcing FAIL or PASS."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    wrong = [
        d
        for d in detections
        if d.sku and d.sku not in expected_skus and d.sku != UNKNOWN_SKU
    ]
    unidentified = [d for d in detections if d.sku == UNKNOWN_SKU]

    if wrong or unidentified:
        parts = [f"{d.label}→{d.sku}" for d in wrong + unidentified]
        return CheckResult(
            check_key=CheckKey.ITEM_IDENTITY,
            verdict=Verdict.FAIL,
            confidence=mean_confidence(wrong + unidentified) if (wrong or unidentified) else 0.0,
            detail=(
                "Visible item(s) do not match expected order SKUs: "
                + ", ".join(parts)
                + ". Classic substitution case (e.g. Red Cap vs Blue Cap) is FAIL."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    return CheckResult(
        check_key=CheckKey.ITEM_IDENTITY,
        verdict=Verdict.PASS,
        confidence=mean_confidence(detections),
        detail="Every confidently detected SKU is on the order.",
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
