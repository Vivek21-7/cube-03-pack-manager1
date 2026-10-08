from pack_manager.round3 import outcome_from_decision, run_pack_round3, to_pack_agent_output


def test_pack_outcomes() -> None:
    assert outcome_from_decision("SEAL")["outcome"] == "seal"
    assert outcome_from_decision("STOP_FIX")["verdict"] == "FAIL"
    assert outcome_from_decision("UNCERTAIN")["status"] == "pending"


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
