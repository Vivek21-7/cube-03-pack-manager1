"""JSON loaders for the confirmed interchange format."""

from __future__ import annotations

import json
from pathlib import Path

from pack_manager.models import CatalogItem, ImageRef, Order, PackEvent, PackPhoto

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_uri(uri: str, *, relative_to: Path) -> str:
    path = Path(uri)
    if path.is_file():
        return str(path.resolve())
    candidates = [
        relative_to / uri,
        REPO_ROOT / uri,
        relative_to / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str((REPO_ROOT / uri).resolve())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalogue(path: Path) -> list[CatalogItem]:
    payload = load_json(path)
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    loaded: list[CatalogItem] = []
    base = path.parent
    for item in items:
        model = CatalogItem.model_validate(item)
        refs = [
            ImageRef.model_validate(
                {**ref.model_dump(), "uri": _resolve_uri(ref.uri, relative_to=base)}
            )
            for ref in model.reference_images
        ]
        loaded.append(model.model_copy(update={"reference_images": refs}))
    return loaded


def load_order(path: Path) -> Order:
    return Order.model_validate(load_json(path))


def load_photos(path: Path) -> list[PackPhoto]:
    payload = load_json(path)
    photos = payload["photos"] if isinstance(payload, dict) and "photos" in payload else payload
    loaded: list[PackPhoto] = []
    base = path.parent
    for photo in photos:
        model = PackPhoto.model_validate(photo)
        loaded.append(model.model_copy(update={"uri": _resolve_uri(model.uri, relative_to=base)}))
    return loaded


def load_pack_event(
    *,
    order_path: Path,
    catalogue_path: Path,
    photos_path: Path,
    operator_label: str | None = None,
) -> PackEvent:
    order = load_order(order_path)
    photos = load_photos(photos_path)
    return PackEvent(
        organization_id=order.organization_id,
        client_id=order.client_id,
        order=order,
        catalogue=load_catalogue(catalogue_path),
        photos=photos,
        operator_label=operator_label or photos[0].operator_label,
    )
