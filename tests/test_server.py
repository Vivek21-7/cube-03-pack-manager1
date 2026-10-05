"""Tests for the Pack Manager HTTP server and REST endpoints."""

from __future__ import annotations

import base64
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from pack_manager.server import PackManagerRequestHandler, REPO_ROOT

PORT = 8991
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def live_server():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), PackManagerRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield BASE_URL
    server.shutdown()
    server.server_close()


def test_server_index_html(live_server: str) -> None:
    resp = httpx.get(f"{live_server}/")
    assert resp.status_code == 200
    assert "Pack Manager AI" in resp.text
    assert "<title>" in resp.text


def test_server_get_scenarios(live_server: str) -> None:
    resp = httpx.get(f"{live_server}/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 5


def test_server_get_scenario_detail(live_server: str) -> None:
    resp = httpx.get(f"{live_server}/api/scenario/example_1_correct_order")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == "example_1_correct_order"
    assert "photo_base64" in data
    assert "order" in data


def test_server_post_verify_correct(live_server: str) -> None:
    scen_resp = httpx.get(f"{live_server}/api/scenario/example_1_correct_order")
    scen_data = scen_resp.json()

    verify_resp = httpx.post(
        f"{live_server}/api/verify",
        json={
            "order": scen_data["order"],
            "photo_base64": scen_data["photo_base64"],
            "provider": "simulation",
        },
    )
    assert verify_resp.status_code == 200
    res = verify_resp.json()
    assert res["decision"] == "SEAL"
    assert res["confidence"] == "high"


def test_server_post_verify_wrong_item(live_server: str) -> None:
    scen_resp = httpx.get(f"{live_server}/api/scenario/example_2_wrong_item")
    scen_data = scen_resp.json()

    verify_resp = httpx.post(
        f"{live_server}/api/verify",
        json={
            "order": scen_data["order"],
            "photo_base64": scen_data["photo_base64"],
            "provider": "simulation",
        },
    )
    assert verify_resp.status_code == 200
    res = verify_resp.json()
    assert res["decision"] == "STOP_FIX"
