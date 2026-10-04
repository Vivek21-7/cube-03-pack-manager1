"""Correct-order path: expected Blue Cap, photo contains Blue Cap → SEAL."""

from __future__ import annotations

from pathlib import Path

import pytest

from pack_manager.checks.synthesis import synthesize_decision
from pack_manager.errors import MissingCatalogueError
from pack_manager.fixtures_draw import write_correct_order_images
from pack_manager.io import load_pack_event
from pack_manager.models import CheckKey, Decision, PackEvent, Verdict
from pack_manager.pipeline import evaluate_pack

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "correct_order"


@pytest.fixture(scope="session")
def correct_images() -> dict[str, Path]:
    return write_correct_order_images(FIXTURES)


@pytest.fixture
def correct_event(correct_images: dict[str, Path]) -> PackEvent:
    return load_pack_event(
        order_path=FIXTURES / "order.json",
        catalogue_path=FIXTURES / "catalogue.json",
        photos_path=FIXTURES / "photos.json",
    )


def test_correct_order_seals(correct_event: PackEvent) -> None:
    result = evaluate_pack(correct_event)
    assert result.comparison.decision == Decision.SEAL
    assert result.evidence.outcome.decision == Decision.SEAL
    assert result.evidence.agent == "pack-manager"
    assert result.evidence.content_hash.startswith("sha256:")
    assert len(result.evidence.checks) == 6
    for check in result.evidence.checks:
        assert check.verdict == Verdict.PASS
        assert 0.0 <= check.confidence <= 1.0
        assert check.latency_ms >= 0
        assert check.detail
        assert check.model_version
    table = result.comparison.quantity_table
    assert len(table) == 1
    assert table[0].sku == "CAP-BLUE-001"
    assert table[0].expected_qty == 1
    assert table[0].observed_qty == 1
    assert table[0].delta == 0
    assert result.comparison.missing == []
    assert result.comparison.wrong == []
    assert result.comparison.extra == []
    keys = {c.check_key for c in result.evidence.checks}
    assert keys == set(CheckKey)


def test_missing_catalogue_is_hard_error(correct_event: PackEvent) -> None:
    event = correct_event.model_copy(update={"catalogue": []})
    with pytest.raises(Exception):
        PackEvent.model_validate(event.model_dump())


def test_ordered_sku_without_catalog_entry(correct_event: PackEvent) -> None:
    other = correct_event.catalogue[0].model_copy(update={"sku": "NOT-THE-ORDERED-SKU"})
    event = correct_event.model_copy(update={"catalogue": [other]})
    with pytest.raises(MissingCatalogueError):
        evaluate_pack(event)


def test_synthesis_never_seals_uncertain() -> None:
    from pack_manager.models import CheckResult

    prior = [
        CheckResult(
            check_key=key,
            verdict=Verdict.PASS if key != CheckKey.ITEM_PRESENCE else Verdict.UNCERTAIN,
            confidence=0.4,
            detail="x",
            model_version="t",
            latency_ms=1,
        )
        for key in CheckKey
        if key != CheckKey.DECISION_SYNTHESIS
    ]
    assert synthesize_decision(prior) == Decision.STOP_AND_FIX
