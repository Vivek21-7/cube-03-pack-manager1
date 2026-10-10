"""Pack inside the real Pod orchestrator: an UNCERTAIN Pack judgment is a completed stage, not a finished workflow.

Needs the Pod repository (POD_REPO, default ../cube-round3-pod) and its dependencies (jsonschema).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

POD_REPO = Path(os.environ.get("POD_REPO", Path(__file__).resolve().parents[2] / "cube-round3-pod"))
if POD_REPO.is_dir() and str(POD_REPO) not in sys.path:
    sys.path.insert(0, str(POD_REPO))
pytest.importorskip("jsonschema")
orchestrator = pytest.importorskip("orchestration.orchestrator")
store_mod = pytest.importorskip("orchestration.store")
records = pytest.importorskip("shared.utils.records")

from pack_manager import round3  # noqa: E402
from test_round3 import FakeClaude, POUCH, png_bytes, pouch_reply  # noqa: E402

CASE = {
    "org_id": "org_demo_alpha",
    "subject_id": "PRODUCT-04",
    "route": "fba",
    "returned": True,
    "order": {"order_id": "PRODUCT-04", "items": [{"sku": "8906126921460", "name": POUCH, "qty": 1}]},
    "charges": [],
}


class Stage:
    """A finished upstream/downstream stage with a fixed verdict."""

    def __init__(self, verdict: str, outcome: str, needs_human: bool | None = None) -> None:
        self.verdict, self.outcome, self.needs_human, self.calls = verdict, outcome, needs_human, 0

    def run(self, request, timeout_s):
        self.calls += 1
        stage = request["stage"]
        rec = records.build_record(
            request, agent_id=f"{stage}-fake@1", record_id=f"{records.PREFIX[stage]}-{request['request_id'].replace(':', '-')}",
            captured_at="2026-10-09T16:09:30Z", checks=[records.check("c1", self.verdict, None)], outcome=self.outcome,
            reason="fake", model={"name": "fake", "version": "1"}, needs_human=self.needs_human,
        )
        return records.build_output(rec)


class RealPack:
    """The production Round 3 handler, fed by the orchestrator's real Agent Input."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, request, timeout_s):
        self.calls += 1
        status, body = round3.run_pack_round3(request)
        assert status == 200
        return body


@pytest.fixture
def pod(monkeypatch, tmp_path):
    photo_dir = tmp_path / "input" / "PRODUCT-04" / "pack"
    photo_dir.mkdir(parents=True)
    (photo_dir / "01-interior.png").write_bytes(png_bytes())
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(round3, "REPO_ROOT", tmp_path)
    clients = {
        "receiving": Stage("UNCERTAIN", "pending_review"),
        "prep": Stage("FAIL", "stop_and_fix", needs_human=True),
        "pack": RealPack(),
        "returns": Stage("PASS", "restock"),
        "recovery": Stage("UNCERTAIN", "insufficient_evidence", needs_human=False),
    }
    flow = orchestrator.load_flow(POD_REPO / "orchestration" / "flow.pod4-tracers.json")
    return clients, flow, store_mod.MemoryStore()


def use_model(monkeypatch, raw: str) -> FakeClaude:
    client = FakeClaude(raw)
    monkeypatch.setattr(round3, "get_vlm_client", lambda provider=None: client)
    return client


def test_pack_resume_completes_stage_but_workflow_stays_blocked(pod, monkeypatch) -> None:
    clients, flow, store = pod
    use_model(monkeypatch, pouch_reply("MAYBE"))
    wf = orchestrator.run_workflow(CASE, flow, store, clients)
    pack = next(sr for sr in wf["stage_results"] if sr["stage"] == "pack")
    assert wf["status"] == "FAILED"
    assert pack["state"] == "error"
    first_record = pack["record_id"]
    assert store.get_evidence(first_record)["model"]["calls"] == 1

    model = use_model(monkeypatch, pouch_reply("UNCERTAIN"))
    wf = orchestrator.resume(wf["workflow_id"], flow, store, clients)

    stages = {sr["stage"]: sr for sr in wf["stage_results"]}
    pack = stages["pack"]
    record = store.get_evidence(pack["record_id"])
    assert model.calls == 1
    assert pack["state"] == "completed" and pack["runs"] == 2
    assert record["status"] == "completed"
    assert record["decision"]["verdict"] == "UNCERTAIN" and record["decision"]["needs_human"] is True
    assert [c["verdict"] for c in record["checks"]] == ["UNCERTAIN"]
    assert record["model"]["calls"] == 1
    assert (record["subject"]["org_id"], record["subject"]["subject_id"], record["workflow_id"]) == (
        "org_demo_alpha", "PRODUCT-04", "WF-org_demo_alpha-PRODUCT-04")

    for stage in ("receiving", "prep", "returns", "recovery"):
        assert stages[stage]["runs"] == 1
        assert clients[stage].calls == 1

    assert wf["status"] == "BLOCKED"
    final = wf["final_outcome"]
    assert final["outcome"] == "EXCEPTION" and final["needs_human"] is True
    assert final["effective_verdicts"]["receiving"] == "UNCERTAIN"
    assert final["effective_verdicts"]["prep"] == "FAIL"
    assert final["effective_verdicts"]["pack"] == "UNCERTAIN"

    assert first_record != pack["record_id"]
    assert store.get_evidence(first_record) is not None
    assert first_record in wf["evidence_references"]

    refs_before = list(wf["evidence_references"])
    wf = orchestrator.resume(wf["workflow_id"], flow, store, clients)
    assert wf["evidence_references"] == refs_before
    assert len(wf["evidence_references"]) == len(set(wf["evidence_references"]))
    assert model.calls == 1
    assert clients["pack"].calls == 2


def test_pack_record_for_another_tenant_is_rejected(pod, monkeypatch) -> None:
    clients, flow, store = pod
    use_model(monkeypatch, pouch_reply("UNCERTAIN"))

    class OtherTenant(RealPack):
        def run(self, request, timeout_s):
            out = super().run(request, timeout_s)
            out["evidence"]["subject"]["org_id"] = "org_demo_bravo"
            return out

    clients["pack"] = OtherTenant()
    wf = orchestrator.run_workflow(CASE, flow, store, clients)
    pack = next(sr for sr in wf["stage_results"] if sr["stage"] == "pack")
    assert pack["state"] == "error"
    assert any(e["code"] == "tenant_mismatch" for e in wf["errors"])
