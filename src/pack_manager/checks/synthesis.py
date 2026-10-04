"""Decision synthesis: SEAL only when every required check is PASS."""

from __future__ import annotations

import time

from pack_manager.checks._common import RULES_MODEL, timed_ms
from pack_manager.models import (
    REQUIRED_CHECK_KEYS,
    CheckKey,
    CheckResult,
    Decision,
    Verdict,
)


def synthesize_decision(checks: list[CheckResult]) -> Decision:
    by_key = {c.check_key: c for c in checks}
    for key in REQUIRED_CHECK_KEYS:
        if key == CheckKey.DECISION_SYNTHESIS:
            continue
        check = by_key.get(key)
        if check is None or check.verdict != Verdict.PASS:
            return Decision.STOP_AND_FIX
    return Decision.SEAL


def run_decision_synthesis(
    prior_checks: list[CheckResult],
    *,
    model_version: str = RULES_MODEL,
) -> CheckResult:
    started = time.perf_counter()
    decision = synthesize_decision(prior_checks)
    blocking = [
        f"{c.check_key.value}={c.verdict.value}"
        for c in prior_checks
        if c.verdict != Verdict.PASS
    ]
    if decision == Decision.SEAL:
        detail = "All required checks PASS. Decision is SEAL."
        verdict = Verdict.PASS
        confidence = min(c.confidence for c in prior_checks) if prior_checks else 0.0
    else:
        detail = (
            "STOP_AND_FIX because at least one required check is FAIL or UNCERTAIN "
            f"({', '.join(blocking) or 'missing required checks'}). "
            "UNCERTAIN never auto-seals."
        )
        verdict = Verdict.FAIL if any(c.verdict == Verdict.FAIL for c in prior_checks) else Verdict.UNCERTAIN
        confidence = min((c.confidence for c in prior_checks), default=0.0)

    return CheckResult(
        check_key=CheckKey.DECISION_SYNTHESIS,
        verdict=verdict,
        confidence=confidence,
        detail=detail,
        model_version=model_version,
        latency_ms=timed_ms(started, time.perf_counter()),
    )
