"""Pack evaluation pipeline: validate → detect → checks → evidence → decision."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pack_manager.checks import (
    run_decision_synthesis,
    run_extra_item_detection,
    run_item_identity,
    run_item_presence,
    run_missing_item_detection,
    run_quantity_match,
)
from pack_manager.checks._common import (
    UNCERTAIN_SKU,
    UNKNOWN_SKU,
    catalogue_name,
    expected_qty_by_sku,
    observed_qty_by_sku,
)
from pack_manager.checks.synthesis import synthesize_decision
from pack_manager.errors import CorruptImageError, InvalidInputError, MissingCatalogueError
from pack_manager.evidence import seal_hash
from pack_manager.models import (
    CatalogItem,
    Decision,
    DetectedItem,
    EvidenceRecord,
    EvidenceStatus,
    ImageRef,
    ImageView,
    ItemMismatch,
    Order,
    Outcome,
    PackComparison,
    PackEvent,
    PackResult,
    QuantityRow,
    Subject,
    Verdict,
)
from pack_manager.scope import AGENT_NAME
from pack_manager.vision import HeuristicColorBlobBackend, VisionBackend


def _file_sha256(uri: str) -> str:
    digest = hashlib.sha256(Path(uri).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def validate_event(event: PackEvent) -> dict[str, CatalogItem]:
    if event.organization_id != event.order.organization_id:
        raise InvalidInputError("organization_id on the pack event does not match the order")
    if event.client_id != event.order.client_id:
        raise InvalidInputError("client_id on the pack event does not match the order")

    by_sku = {item.sku: item for item in event.catalogue}
    missing = [line.sku for line in event.order.lines if line.sku not in by_sku]
    if missing:
        raise MissingCatalogueError(
            "ordered SKU(s) have no catalogue entry: " + ", ".join(sorted(set(missing)))
        )

    for photo in event.photos:
        path = Path(photo.uri)
        if not path.is_file():
            raise CorruptImageError(f"evidence photo not found: {photo.uri}")
    return by_sku


def _quantity_table(order: Order, detections: list[DetectedItem]) -> list[QuantityRow]:
    expected = expected_qty_by_sku(order)
    observed = observed_qty_by_sku(detections)
    ambiguous = any(
        d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45 for d in detections
    ) or not detections
    rows: list[QuantityRow] = []
    for sku, exp in expected.items():
        obs = None if ambiguous else observed.get(sku, 0)
        delta = None if obs is None else obs - exp
        rows.append(
            QuantityRow(
                sku=sku,
                product_name=catalogue_name(order, sku),
                expected_qty=exp,
                observed_qty=obs,
                delta=delta,
            )
        )
    return rows


def _mismatches(order: Order, detections: list[DetectedItem]) -> tuple[
    list[ItemMismatch],
    list[ItemMismatch],
    list[ItemMismatch],
]:
    expected = expected_qty_by_sku(order)
    observed = observed_qty_by_sku(detections)
    ambiguous = any(d.sku in {None, UNCERTAIN_SKU} or d.confidence < 0.45 for d in detections)

    missing: list[ItemMismatch] = []
    extra: list[ItemMismatch] = []
    wrong: list[ItemMismatch] = []
    if ambiguous or not detections:
        return missing, wrong, extra

    for sku, exp in expected.items():
        obs = observed.get(sku, 0)
        if obs < exp:
            missing.append(
                ItemMismatch(
                    sku=sku,
                    product_name=catalogue_name(order, sku),
                    quantity=exp - obs,
                    reason=f"expected {exp}, observed {obs}",
                )
            )

    for det in detections:
        if det.sku in expected:
            continue
        target = extra if det.sku in {None, UNKNOWN_SKU} else wrong
        target.append(
            ItemMismatch(
                sku=None if det.sku in {None, UNKNOWN_SKU} else det.sku,
                product_name=det.label,
                quantity=det.quantity,
                reason=det.reasoning or "visible item is not on the order",
                related_detection_ids=[det.detection_id],
            )
        )

    for sku, obs in observed.items():
        if sku in expected and obs > expected[sku]:
            extra.append(
                ItemMismatch(
                    sku=sku,
                    product_name=catalogue_name(order, sku),
                    quantity=obs - expected[sku],
                    reason=f"observed {obs} vs expected {expected[sku]}",
                )
            )
    return missing, wrong, extra


def build_comparison(order: Order, detections: list[DetectedItem], decision: Decision) -> PackComparison:
    missing, wrong, extra = _mismatches(order, detections)
    return PackComparison(
        detected_items=detections,
        expected_items=order.lines,
        quantity_table=_quantity_table(order, detections),
        missing=missing,
        wrong=wrong,
        extra=extra,
        decision=decision,
    )


def evaluate_pack(
    event: PackEvent,
    *,
    vision: VisionBackend | None = None,
    decided_by: str = AGENT_NAME,
) -> PackResult:
    validate_event(event)
    backend = vision or HeuristicColorBlobBackend()
    detections, vision_latency_ms = backend.detect(event)
    model_version = getattr(backend, "model_version", "unknown")

    checks = [
        run_item_presence(detections, model_version=model_version),
        run_quantity_match(event.order, detections, model_version=model_version),
        run_item_identity(event.order, detections, model_version=model_version),
        run_extra_item_detection(event.order, detections, model_version=model_version),
        run_missing_item_detection(event.order, detections, model_version=model_version),
    ]
    # Presence/quantity checks already include their own latency; add vision cost
    # onto identification so model cost is traceable on the vision-bound check.
    presence = checks[0].model_copy(
        update={"latency_ms": checks[0].latency_ms + vision_latency_ms}
    )
    checks[0] = presence
    synthesis = run_decision_synthesis(checks)
    all_checks = [*checks, synthesis]
    decision = synthesize_decision(checks)

    now = datetime.now(timezone.utc)
    captured = max(photo.captured_at for photo in event.photos)
    images = [
        ImageRef(
            image_id=photo.image_id,
            uri=photo.uri,
            sha256=_file_sha256(photo.uri),
            role="evidence",
            view=photo.view if isinstance(photo.view, ImageView) else ImageView.UNKNOWN,
        )
        for photo in event.photos
    ]

    needs_review = any(c.verdict == Verdict.UNCERTAIN for c in all_checks)
    status = EvidenceStatus.NEEDS_REVIEW if needs_review else EvidenceStatus.COMPLETE
    if decision == Decision.STOP_AND_FIX and needs_review:
        status = EvidenceStatus.NEEDS_REVIEW

    record = EvidenceRecord(
        record_id=str(uuid.uuid4()),
        organization_id=event.organization_id,
        client_id=event.client_id,
        agent=AGENT_NAME,
        subject=Subject(order_id=event.order.order_id, package_id=event.order.package_id),
        captured_at=captured,
        operator_label=event.operator_label,
        images=images,
        checks=all_checks,
        outcome=Outcome(decision=decision, decided_by=decided_by, decided_at=now),
        overrides=[],
        status=status,
        content_hash="",
    )
    record = seal_hash(record)
    return PackResult(comparison=build_comparison(event.order, detections, decision), evidence=record)
