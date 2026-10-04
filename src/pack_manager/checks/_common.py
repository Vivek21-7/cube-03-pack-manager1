"""Shared helpers for check modules."""

from __future__ import annotations

from collections import defaultdict

from pack_manager.models import DetectedItem, Order, OrderLine

RULES_MODEL = "rules-0.1.0"
UNCERTAIN_SKU = "__uncertain__"
UNKNOWN_SKU = "__unknown__"


def timed_ms(started: float, finished: float) -> int:
    return max(0, int(round((finished - started) * 1000)))


def catalogue_name(order: Order, sku: str) -> str:
    for line in order.lines:
        if line.sku == sku:
            return line.product_name
    return sku


def expected_qty_by_sku(order: Order) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for line in order.lines:
        totals[line.sku] += line.expected_qty
    return dict(totals)


def observed_qty_by_sku(detections: list[DetectedItem]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for item in detections:
        key = item.sku if item.sku else UNKNOWN_SKU
        totals[key] += item.quantity
    return dict(totals)


def mean_confidence(detections: list[DetectedItem]) -> float:
    if not detections:
        return 0.0
    return sum(item.confidence for item in detections) / len(detections)


def lines_by_sku(order: Order) -> dict[str, list[OrderLine]]:
    grouped: dict[str, list[OrderLine]] = defaultdict(list)
    for line in order.lines:
        grouped[line.sku].append(line)
    return dict(grouped)


def is_ambiguous(detections: list[DetectedItem]) -> bool:
    return any(
        item.sku in {None, UNCERTAIN_SKU} or item.confidence < 0.45
        for item in detections
    )
