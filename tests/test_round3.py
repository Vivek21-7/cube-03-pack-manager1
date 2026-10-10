import base64
import io
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from pack_manager import round3
from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import MatchResultAI
from pack_manager.round3 import outcome_from_decision, run_pack_round3, to_pack_agent_output


def test_pack_outcomes() -> None:
    assert outcome_from_decision("SEAL")["outcome"] == "seal"
    assert outcome_from_decision("STOP_FIX")["verdict"] == "FAIL"
    uncertain = outcome_from_decision("UNCERTAIN")
    assert uncertain["status"] == "completed"
    assert uncertain["verdict"] == "UNCERTAIN"
    assert uncertain["needs_human"] is True


POUCH = "Oblique Sally tech pouch"


class FakeClaude:
    provider_name = "anthropic"
    model_name = "claude-sonnet-4-5"

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls = 0

    def inspect(self, prompt, image_bytes, mime_type="image/png", extra_images=None) -> str:
        self.calls += 1
        return self.raw


def pouch_reply(line_status: str, decision: str = "UNCERTAIN", confidence: str = "medium") -> str:
    return json.dumps(
        {
            "order_id": "PRODUCT-04",
            "order_items": [{"name": POUCH, "expected_qty": 1, "variant": "Tan"}],
            "detected_items": [
                {"name": POUCH, "detected_qty": 1, "variant": "Tan", "confidence": "medium", "notes": "Wrapped in tissue"}
            ],
            "matches": [
                {"item_name": POUCH, "expected_qty": 1, "detected_qty": 1, "variant_match": True, "status": line_status}
            ],
            "missing_items": [],
            "extra_items": [],
            "product_condition": {"visible_damage": False, "notes": "No visible damage"},
            "decision": decision,
            "decision_reason": "Tissue covers the pouch, so the brand mark cannot be read.",
            "confidence": confidence,
            "evidence": ["Tan pouch partly wrapped in tissue"],
        }
    )


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 170, 120)).save(buf, format="PNG")
    return buf.getvalue()


def pouch_body(org_id: str = "org_demo_alpha") -> dict:
    return {
        "request_id": "WF-org_demo_alpha-PRODUCT-04:pack",
        "workflow_id": "WF-org_demo_alpha-PRODUCT-04",
        "stage": "pack",
        "subject": {"org_id": org_id, "subject_id": "PRODUCT-04"},
        "inputs": [{"ref": "PRODUCT-04/pack/01.png", "data_base64": base64.b64encode(png_bytes()).decode()}],
        "previous_evidence": [{"record_id": "RCV-1"}, {"record_id": "PRP-1"}],
        "context": {"order": {"order_id": "PRODUCT-04", "items": [{"sku": "8906126921460", "name": POUCH, "qty": 1}]}},
    }


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(round3, "REPO_ROOT", tmp_path)
    holder = {}

    def use(raw: str) -> FakeClaude:
        client = FakeClaude(raw)
        holder["client"] = client
        monkeypatch.setattr(round3, "get_vlm_client", lambda provider=None: client)
        return client

    return use


@pytest.mark.parametrize("status", ["PASS", "FAIL", "UNCERTAIN"])
def test_match_status_accepts_contract_verdicts(status) -> None:
    assert MatchResultAI(item_name=POUCH, expected_qty=1, detected_qty=1, status=status).status == status


def test_match_status_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        MatchResultAI(item_name=POUCH, expected_qty=1, detected_qty=1, status="MAYBE")


def test_uncertain_line_is_never_sealed() -> None:
    client = FakeClaude(pouch_reply("UNCERTAIN", decision="SEAL", confidence="high"))
    agent = PackManagerAIAgent(vlm_client=client)
    result = agent.verify(
        order={"order_id": "PRODUCT-04", "items": [{"name": POUCH, "expected_qty": 1}]}, photo=png_bytes()
    )
    assert [m.status for m in result.matches] == ["UNCERTAIN"]
    assert result.decision == "UNCERTAIN"
    assert POUCH in result.decision_reason
    assert "brand mark cannot be read" in result.decision_reason
    assert agent.inspect_calls == 1


def test_pass_line_still_seals() -> None:
    agent = PackManagerAIAgent(vlm_client=FakeClaude(pouch_reply("PASS", decision="SEAL", confidence="high")))
    result = agent.verify(
        order={"order_id": "PRODUCT-04", "items": [{"name": POUCH, "expected_qty": 1}]}, photo=png_bytes()
    )
    assert [m.status for m in result.matches] == ["PASS"]
    assert result.decision == "SEAL"


def test_uncertain_pouch_persists_completed_record(fake_model, tmp_path) -> None:
    client = fake_model(pouch_reply("UNCERTAIN"))
    status, output = run_pack_round3(pouch_body())
    assert status == 200
    ev = output["evidence"]
    assert output["status"] == ev["status"] == "completed"
    assert output["verdict"] == ev["decision"]["verdict"] == "UNCERTAIN"
    assert ev["decision"]["needs_human"] is True
    assert ev["error"] is None
    assert [c["verdict"] for c in ev["checks"]] == ["UNCERTAIN"]
    assert client.calls == 1
    assert ev["model"]["calls"] == 1
    assert ev["model"]["provider"] == "anthropic"
    assert ev["inputs"][0]["ref"] == "PRODUCT-04/pack/01.png"
    assert ev["workflow_id"] == "WF-org_demo_alpha-PRODUCT-04"
    assert (ev["subject"]["org_id"], ev["subject"]["subject_id"]) == ("org_demo_alpha", "PRODUCT-04")
    assert ev["upstream_refs"] == ["RCV-1", "PRP-1"]
    saved = json.loads((tmp_path / "data" / "round3-evidence" / f"{ev['record_id']}.json").read_text())
    assert saved == output


def test_invalid_model_output_fails_open_and_counts_the_call(fake_model) -> None:
    fake_model(pouch_reply("MAYBE"))
    status, output = run_pack_round3(pouch_body())
    assert status == 200
    assert output["status"] == "pending"
    assert output["verdict"] == "UNCERTAIN"
    assert output["error"]["retryable"] is True
    assert "MAYBE" in output["error"]["message"]
    assert output["model"]["calls"] == 1
    assert output["evidence"]["inputs"][0]["ref"] == "PRODUCT-04/pack/01.png"


def test_same_request_writes_one_record(fake_model, tmp_path) -> None:
    fake_model(pouch_reply("UNCERTAIN"))
    _, first = run_pack_round3(pouch_body())
    _, second = run_pack_round3(pouch_body())
    assert first["evidence"]["record_id"] == second["evidence"]["record_id"]
    assert len(list((tmp_path / "data" / "round3-evidence").iterdir())) == 1


def test_unknown_tenant_is_refused_before_any_model_call(fake_model) -> None:
    client = fake_model(pouch_reply("UNCERTAIN"))
    status, _ = run_pack_round3(pouch_body(org_id="org_someone_else"))
    assert status == 404
    assert client.calls == 0


def test_pack_envelope() -> None:
    output = to_pack_agent_output(
        body={
            "request_id": "req-1",
            "workflow_id": "WF-1",
            "subject": {"org_id": "org_demo_alpha", "subject_id": "ORD-1"},
            "previous_evidence": [{"record_id": "PRP-1"}],
        },
        decision="SEAL",
        reason="lines match",
        checks=[{"check_key": "line_match", "verdict": "PASS", "detail": "cap"}],
        model_name="claude-sonnet-4-5",
        calls=1,
        input_refs=[{"ref": "box.png", "sha256": "abc", "kind": "image"}],
    )
    assert output["stage"] == "pack"
    assert output["evidence"]["decision"]["outcome"] == "seal"
    assert output["evidence"]["payload"]["upstream_refs"] == ["PRP-1"]
    assert len(output["evidence"]["content_hash"]) == 64


def test_closed_shipper_does_not_seal(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, output = run_pack_round3(
        {
            "request_id": "req-closed",
            "workflow_id": "WF-closed",
            "stage": "pack",
            "subject": {"org_id": "org_demo_alpha", "subject_id": "HUB-CLINDACAN-600"},
            "inputs": [{"ref": "carton.jpg"}],
        }
    )
    assert status == 200
    assert output["status"] == "pending"
    assert output["verdict"] == "UNCERTAIN"
    assert output["model"]["calls"] == 0
    assert "closed" in output["evidence"]["decision"]["reason"].lower()


def test_missing_photo_fail_open(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, output = run_pack_round3(
        {
            "request_id": "req-2",
            "workflow_id": "WF-2",
            "stage": "pack",
            "subject": {"org_id": "org_demo_alpha", "subject_id": "ORD-2"},
            "inputs": [],
            "context": {"order": {"order_id": "ORD-2", "items": [{"name": "Cap", "expected_qty": 1}]}},
        }
    )
    assert status == 200
    assert output["status"] == "pending"
    assert output["error"]["retryable"] is True
