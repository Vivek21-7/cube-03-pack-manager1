"""Round 3 Agent Input → Agent Output for Pack. Observation stays in the existing agent."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pack_manager.agent import PackManagerAIAgent
from pack_manager.vlm_client import get_vlm_client

AGENT_ID = "pack-manager@1.0.0"
DEMO_ORGS = {"org_demo_alpha", "org_demo_bravo"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def outcome_from_decision(decision: str) -> dict[str, Any]:
    normalized = (decision or "").upper().replace(" ", "_").replace("&", "")
    if normalized in {"SEAL", "SEALED"}:
        return {"outcome": "seal", "verdict": "PASS", "status": "completed", "needs_human": False, "action": "continue"}
    if normalized in {"STOP_FIX", "STOP_AND_FIX", "STOP"}:
        return {"outcome": "stop_and_fix", "verdict": "FAIL", "status": "completed", "needs_human": True, "action": "hold"}
    return {"outcome": "review_required", "verdict": "UNCERTAIN", "status": "completed", "needs_human": True, "action": "review"}


def to_pack_agent_output(
    *,
    body: dict[str, Any],
    decision: str,
    reason: str,
    checks: list[dict[str, Any]],
    model_name: str,
    calls: int,
    input_refs: list[dict[str, Any]],
    fail_open: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mapped = outcome_from_decision(decision)
    status = "pending" if fail_open else mapped["status"]
    verdict = "UNCERTAIN" if fail_open else mapped["verdict"]
    produced = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    subject = body.get("subject") or {}
    record_id = "PCK-" + hashlib.sha256(
        f"{body.get('request_id')}:{subject.get('subject_id')}".encode()
    ).hexdigest()[:12]
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "record_id": record_id,
        "workflow_id": body.get("workflow_id"),
        "stage": "pack",
        "agent_id": AGENT_ID,
        "subject": {
            "org_id": subject.get("org_id"),
            "subject_id": subject.get("subject_id"),
            "unit_id": subject.get("subject_id"),
            "unit_scope": "order",
            "refs": {"order_id": (body.get("context") or {}).get("order_id") or subject.get("subject_id")},
        },
        "status": status,
        "captured_at": produced,
        "produced_at": produced,
        "latency_ms": 0,
        "operator_id": "pack",
        "model": {
            "name": "claude" if calls else "simulation" if model_name.startswith("simulation") else "rules",
            "version": model_name,
            "provider": "anthropic" if calls and "claude" in model_name else None,
            "prompt_version": "pack-observe-v1",
            "calls": calls,
            "cost_usd": None,
        },
        "inputs": input_refs,
        "checks": checks,
        "decision": {
            "verdict": verdict,
            "outcome": "review_required" if fail_open else mapped["outcome"],
            "confidence": None,
            "reason": fail_open["message"] if fail_open else reason,
            "needs_human": True if fail_open else mapped["needs_human"],
        },
        "payload": {
            "pack_decision": decision,
            "upstream_refs": [row.get("record_id") for row in (body.get("previous_evidence") or []) if row.get("record_id")],
            "claim_boundary": "open_box_pack_only",
        },
        "upstream_refs": [row.get("record_id") for row in (body.get("previous_evidence") or []) if row.get("record_id")],
        "overrides": [],
        "error": fail_open,
    }
    evidence["content_hash"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, default=str).encode()
    ).hexdigest()
    output = {
        "schema_version": "1.0",
        "workflow_id": body.get("workflow_id"),
        "stage": "pack",
        "agent_id": AGENT_ID,
        "status": status,
        "verdict": verdict,
        "confidence": None,
        "timestamp": produced,
        "model": evidence["model"],
        "error": evidence["error"],
        "next_step_recommendation": {
            "action": "review" if fail_open else mapped["action"],
            "reason": evidence["decision"]["reason"],
        },
        "evidence": evidence,
    }
    _persist(record_id, output)
    return output


def run_pack_round3(body: dict[str, Any], *, provider: str | None = None) -> tuple[int, dict[str, Any]]:
    if not body.get("request_id") or not body.get("workflow_id") or not (body.get("subject") or {}).get("org_id"):
        return 422, {"error": "request_id, workflow_id, and subject.org_id are required"}
    subject = body["subject"]
    if not subject.get("subject_id"):
        return 422, {"error": "subject.subject_id is required"}
    if body.get("stage") and body["stage"] != "pack":
        return 422, {"error": "stage must be pack"}
    if subject["org_id"] not in DEMO_ORGS:
        return 404, {"error": "unknown tenant"}

    photos = _load_photos(body)
    order = _order_from(body)
    policy = _case_policy(subject.get("subject_id") or "")
    if photos and order and policy.get("contents_visible") is False:
        return 200, to_pack_agent_output(
            body=body,
            decision="UNCERTAIN",
            reason=str(policy.get("reason") or "The carton is closed, so the contents cannot be verified."),
            checks=[],
            model_name="rules",
            calls=0,
            input_refs=[{"ref": photos[0]["ref"], "sha256": photos[0]["sha256"], "kind": "image"}],
            fail_open={
                "code": "pending",
                "message": str(policy.get("reason") or "The carton is closed, so the contents cannot be verified."),
                "retryable": False,
            },
        )
    if not photos or not order:
        message = "No usable photos were attached to this Agent Input" if not photos else "No order lines were attached in context.order"
        return 200, to_pack_agent_output(
            body=body,
            decision="UNCERTAIN",
            reason=message,
            checks=[],
            model_name="none",
            calls=0,
            input_refs=[],
            fail_open={"code": "pending", "message": message, "retryable": True},
        )

    chosen = provider or os.getenv("PACK_MANAGER_VLM_PROVIDER") or "anthropic"
    if chosen in {"anthropic", "claude"} and not os.getenv("ANTHROPIC_API_KEY"):
        message = "ANTHROPIC_API_KEY is not set. Pack Round 3 does not use a Gemini key."
        return 200, to_pack_agent_output(
            body=body,
            decision="UNCERTAIN",
            reason=message,
            checks=[],
            model_name="none",
            calls=0,
            input_refs=[],
            fail_open={"code": "pending", "message": message, "retryable": True},
        )

    input_refs = [{"ref": photos[0]["ref"], "sha256": photos[0]["sha256"], "kind": "image"}]
    client = None
    agent = None
    try:
        client = get_vlm_client(provider=chosen)
        agent = PackManagerAIAgent(vlm_client=client)
        extras = [(row["bytes"], _mime(row["bytes"])) for row in photos[1:]]
        result = agent.verify(order=order, photo=photos[0]["bytes"], extra_photos=extras or None)
        summary = result.to_summary_dict()
        checks = []
        for match in summary.get("matches") or []:
            status = match.get("status")
            checks.append(
                {
                    "check_key": "line_match",
                    "verdict": status if status in {"PASS", "UNCERTAIN"} else "FAIL",
                    "confidence": None,
                    "expected": match.get("expected_qty"),
                    "observed": match.get("detected_qty"),
                    "detail": match.get("item_name"),
                    "evidence_refs": [photos[0]["ref"]],
                }
            )
        if summary.get("decision") == "UNCERTAIN":
            for check in checks:
                if check["verdict"] != "FAIL":
                    check["verdict"] = "UNCERTAIN"
        calls = 0 if client.provider_name == "simulation" else agent.inspect_calls
        decision = summary.get("decision") or "UNCERTAIN"
        reason = summary.get("decision_reason") or ""
        if policy.get("seal_allowed") is False and str(decision).upper() in {"SEAL", "SEALED"}:
            decision = "UNCERTAIN"
            reason = str(policy.get("reason") or reason)
            for check in checks:
                if check["verdict"] != "FAIL":
                    check["verdict"] = "UNCERTAIN"
        return 200, to_pack_agent_output(
            body=body,
            decision=decision,
            reason=reason,
            checks=checks,
            model_name=client.model_name,
            calls=calls,
            input_refs=input_refs,
        )
    except Exception as exc:  # noqa: BLE001 — fail open for the orchestrator
        message = str(exc)
        calls = 0
        if agent is not None and client.provider_name != "simulation":
            calls = agent.inspect_calls
        return 200, to_pack_agent_output(
            body=body,
            decision="UNCERTAIN",
            reason=message,
            checks=[],
            model_name=client.model_name if calls else "none",
            calls=calls,
            input_refs=input_refs if calls else [],
            fail_open={"code": "pending", "message": message, "retryable": True},
        )


def _order_from(body: dict[str, Any]) -> dict[str, Any] | None:
    context = body.get("context") or {}
    order = context.get("order")
    if isinstance(order, dict) and (order.get("items") or order.get("lines")):
        return order
    subject_id = body["subject"]["subject_id"]
    for folder in ("scenarios", "from_returns", "from_hub"):
        candidate = REPO_ROOT / "fixtures" / folder / subject_id / "order.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _case_policy(subject_id: str) -> dict[str, Any]:
    for folder in ("from_returns", "from_hub", "scenarios"):
        candidate = REPO_ROOT / "fixtures" / folder / subject_id / "policy.json"
        if candidate.is_file():
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    return {}


def _load_photos(body: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    subject_id = (body.get("subject") or {}).get("subject_id") or ""
    for row in body.get("inputs") or []:
        raw: bytes | None = None
        if row.get("data_base64"):
            raw = base64.b64decode(row["data_base64"])
        elif row.get("ref"):
            for candidate in (
                Path(row["ref"]),
                REPO_ROOT / row["ref"],
                REPO_ROOT / "fixtures" / "scenarios" / subject_id / Path(row["ref"]).name,
                REPO_ROOT / "fixtures" / "from_returns" / subject_id / Path(row["ref"]).name,
                REPO_ROOT / "fixtures" / "from_hub" / subject_id / Path(row["ref"]).name,
            ):
                if candidate.is_file():
                    raw = candidate.read_bytes()
                    break
        if not raw:
            continue
        out.append(
            {
                "ref": row.get("ref") or f"photo-{len(out) + 1}",
                "bytes": raw,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return out[:8]


def _persist(record_id: str, output: dict[str, Any]) -> None:
    folder = REPO_ROOT / "data" / "round3-evidence"
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in record_id)
    (folder / f"{safe}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
