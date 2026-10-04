"""Object/product identification: are items visually present in evidence photos?"""

from __future__ import annotations

import time

from pack_manager.checks._common import RULES_MODEL, mean_confidence, timed_ms
from pack_manager.models import CheckKey, CheckResult, DetectedItem, Verdict


def run_item_presence(
    detections: list[DetectedItem],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()

    if not detections:
        return CheckResult(
            check_key=CheckKey.ITEM_PRESENCE,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            detail=(
                "No products were identified in the evidence photograph(s). "
                "Absence of detections is treated as insufficient evidence, not as an empty box."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    ambiguous = [
        d for d in detections if d.sku is None or d.confidence < 0.45 or d.sku == "__uncertain__"
    ]
    if ambiguous:
        labels = ", ".join(sorted({d.label for d in ambiguous}))
        return CheckResult(
            check_key=CheckKey.ITEM_PRESENCE,
            verdict=Verdict.UNCERTAIN,
            confidence=mean_confidence(ambiguous),
            detail=(
                f"Photograph contains ambiguous region(s) ({labels}). "
                "Item presence cannot be positively confirmed."
            ),
            model_version=model_version,
            latency_ms=timed_ms(started, time.perf_counter()),
        )

    identified = ", ".join(
        f"{d.sku} x{d.quantity} (conf={d.confidence:.2f})" for d in detections
    )
    return CheckResult(
        check_key=CheckKey.ITEM_PRESENCE,
        verdict=Verdict.PASS,
        confidence=mean_confidence(detections),
        detail=f"Identified visible products grounded in the photograph: {identified}.",
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
