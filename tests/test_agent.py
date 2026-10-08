"""Tests for the Pack Manager AI Agent, schemas, and prompt-driven verification rules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pack_manager.agent import PackManagerAIAgent, _clean_json_markdown
from pack_manager.agent_schemas import (
    DetectedItemAI,
    MatchResultAI,
    MissingItemAI,
    OrderItemInput,
    PackManagerAIResponse,
    ProductConditionAI,
)
from pack_manager.fixture_scenarios import generate_all_scenarios
from pack_manager.models import Decision, PackEvent
from pack_manager.pipeline import evaluate_pack
from pack_manager.vlm_client import SimulationVLMClient, get_vlm_client

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "scenarios"


@pytest.fixture(scope="session", autouse=True)
def scenarios() -> dict[str, dict[str, Path]]:
    return generate_all_scenarios(FIXTURES)


def test_clean_json_markdown() -> None:
    raw_fenced = "```json\n{\"order_id\": \"123\"}\n```"
    assert _clean_json_markdown(raw_fenced) == '{"order_id": "123"}'

    raw_plain = '{"order_id": "123"}'
    assert _clean_json_markdown(raw_plain) == '{"order_id": "123"}'

    raw_fenced_no_lang = "```\n{\"order_id\": \"123\"}\n```"
    assert _clean_json_markdown(raw_fenced_no_lang) == '{"order_id": "123"}'


def test_seal_prohibited_if_missing_items() -> None:
    # Model validation rule: even if raw decision is SEAL, missing items force STOP_FIX
    resp = PackManagerAIResponse(
        order_id="O1",
        order_items=[OrderItemInput(name="Cap", expected_qty=1)],
        detected_items=[],
        matches=[],
        missing_items=[MissingItemAI(item_name="Cap", expected_qty=1, reason="missing")],
        extra_items=[],
        product_condition=ProductConditionAI(visible_damage=False),
        decision="SEAL",
        decision_reason="Attempted seal",
        confidence="high",
        evidence=[],
    )
    assert resp.decision == "STOP_FIX"


def test_seal_prohibited_if_damage() -> None:
    resp = PackManagerAIResponse(
        order_id="O1",
        order_items=[],
        detected_items=[],
        matches=[],
        missing_items=[],
        extra_items=[],
        product_condition=ProductConditionAI(visible_damage=True, notes="Tear on packaging"),
        decision="SEAL",
        decision_reason="All present",
        confidence="high",
        evidence=[],
    )
    assert resp.decision == "STOP_FIX"


def test_example_1_correct_order_seals(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_1_correct_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "SEAL"
    assert result.confidence == "high"
    assert len(result.detected_items) == 2
    assert all(m.status == "PASS" for m in result.matches)
    assert result.missing_items == []
    assert result.extra_items == []
    assert result.product_condition.visible_damage is False


def test_example_2_wrong_item_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_2_wrong_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "STOP_FIX"
    assert "color mismatch" in result.decision_reason.lower() or "variant" in result.decision_reason.lower()
    cap_match = [m for m in result.matches if "cap" in m.item_name.lower()][0]
    assert cap_match.status == "FAIL"
    assert cap_match.variant_match is False


def test_example_3_missing_item_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_3_missing_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "STOP_FIX"
    assert len(result.missing_items) == 1
    assert "manual" in result.missing_items[0].item_name.lower()


def test_example_4_extra_item_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_4_extra_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "STOP_FIX"
    assert len(result.extra_items) == 1
    assert "scarf" in result.extra_items[0].item_name.lower()


def test_example_5_unclear_photo_uncertain(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_5_unclear_photo"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "UNCERTAIN"
    assert result.confidence == "low"
    assert "photo quality" in result.decision_reason.lower() or "clearer" in result.decision_reason.lower()


def test_example_6_damaged_goods_stops_fix(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_6_damaged_goods"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    result = agent.verify(order=scen["order"], photo=scen["photo"])

    assert result.decision == "STOP_FIX"
    assert result.product_condition.visible_damage is True


def test_agent_vision_backend_pipeline_adapter() -> None:
    from pack_manager.io import load_pack_event

    correct_fixtures = ROOT / "fixtures" / "correct_order"
    event = load_pack_event(
        order_path=correct_fixtures / "order.json",
        catalogue_path=correct_fixtures / "catalogue.json",
        photos_path=correct_fixtures / "photos.json",
    )
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    adapter = agent.as_vision_backend()
    result = evaluate_pack(event, vision=adapter)

    assert result.evidence.agent == "pack-manager"
    assert result.comparison.decision == Decision.SEAL
    assert result.evidence.content_hash.startswith("sha256:")


# ==============================================================================
# STAGE 1: VISUAL OBSERVATION UNIT & INTEGRATION TESTS
# ==============================================================================

def test_stage_1_observation_schema_validation() -> None:
    from pack_manager.agent_schemas import (
        BoundingBox2D,
        ContainerObservation,
        ObservedPhysicalItem,
        PackageVisualObservation,
    )

    box = BoundingBox2D(ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6)
    assert box.ymin == 0.1 and box.xmax == 0.6

    # Strict rejection of inverted bounding boxes without guessing width or swapping
    with pytest.raises(ValueError):
        BoundingBox2D(ymin=0.8, xmin=0.2, ymax=0.3, xmax=0.6)  # ymin > ymax

    with pytest.raises(ValueError):
        BoundingBox2D(ymin=0.1, xmin=0.8, ymax=0.5, xmax=0.2)  # xmin > xmax

    item = ObservedPhysicalItem(
        object_id="obj-1",
        label="Folded T-Shirt",
        visual_attributes={"color": "black", "text": "M"},
        confidence=0.95,
        bbox=box,
    )
    obs = PackageVisualObservation(
        image_id="test-img-1",
        container=ContainerObservation(container_type="cardboard_box", visible_damage=False),
        observed_items=[item],
    )
    assert obs.total_observed_products() == 1
    assert obs.container.visible_damage is False


def test_stage_1_observation_correct_order_blind_to_order(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_1_correct_order"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    # Observe package without passing any order manifest
    obs = agent.observe_package(photo=scen["photo"], image_id="box-test-1")

    assert obs.image_quality == "clear"
    assert obs.clarity_score > 0.9
    assert obs.container.visible_damage is False

    # Must return 1 detection per physical product unit (2 shirts + 1 cap = 3 items)
    assert len(obs.observed_items) == 3
    assert obs.total_observed_products() == 3

    # Every item must have valid normalized bounding box and confidence
    for item in obs.observed_items:
        assert 0.0 <= item.bbox.ymin <= item.bbox.ymax <= 1.0
        assert 0.0 <= item.bbox.xmin <= item.bbox.xmax <= 1.0
        assert 0.8 <= item.confidence <= 1.0
        assert item.occlusion.is_occluded is False


def test_stage_1_observation_wrong_item_observes_red_variant(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_2_wrong_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    obs = agent.observe_package(photo=scen["photo"])
    assert len(obs.observed_items) == 3

    cap = next(it for it in obs.observed_items if "cap" in it.label.lower())
    assert cap.visual_attributes.get("color") == "red"
    assert "RED CAP" in cap.visual_attributes.get("text", "")


def test_stage_1_observation_missing_item_only_reports_visible(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_3_missing_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    # In missing_item scenario, the manual is physically absent from the box photo.
    # Stage 1 must NOT claim anything is missing because it does not know the order!
    obs = agent.observe_package(photo=scen["photo"])
    assert len(obs.observed_items) == 3  # Only the 2 shirts and 1 cap are physically seen
    assert all("manual" not in it.label.lower() for it in obs.observed_items)


def test_stage_1_observation_extra_item_detects_all_physical_units(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_4_extra_item"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    obs = agent.observe_package(photo=scen["photo"])
    # 2 shirts + 1 cap + 1 scarf = 4 physical items detected
    assert len(obs.observed_items) == 4
    labels = [it.label.lower() for it in obs.observed_items]
    assert any("scarf" in l for l in labels)


def test_stage_1_observation_unclear_photo_flags_blur_ambiguity(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_5_unclear_photo"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    obs = agent.observe_package(photo=scen["photo"])
    assert obs.image_quality == "blurry"
    assert obs.clarity_score < 0.5
    assert len(obs.ambiguity_flags) > 0


def test_stage_1_observation_detects_container_damage(scenarios: dict[str, dict[str, Path]]) -> None:
    scen = scenarios["example_6_damaged_goods"]
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())

    obs = agent.observe_package(photo=scen["photo"])
    assert obs.container.visible_damage is True
    assert obs.container.damage_notes is not None
