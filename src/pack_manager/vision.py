"""Pluggable visual detection. MVP backend is a deterministic color-blob matcher."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageFilter, ImageStat

from pack_manager.checks._common import UNCERTAIN_SKU, UNKNOWN_SKU
from pack_manager.errors import CorruptImageError
from pack_manager.models import BoundingBox, CatalogItem, DetectedItem, PackEvent, PackPhoto

VISION_MODEL_VERSION = "heuristic-color-blob-0.1.0"

# Hue distance below this means two SKUs are visually confusable.
SIMILAR_HUE_THRESHOLD = 12.0
# Pixel must be this close in hue to a catalogue reference to match.
MATCH_HUE_THRESHOLD = 18.0
MIN_BLOB_PIXELS = 400
AMBIGUOUS_SHARPNESS = 18.0


class VisionBackend(Protocol):
    model_version: str

    def detect(self, event: PackEvent) -> tuple[list[DetectedItem], int]:
        """Return detections and backend latency in milliseconds."""


@dataclass
class _SkuSignature:
    sku: str
    name: str
    hue: float


def _open_rgb(uri: str) -> Image.Image:
    path = Path(uri)
    if not path.is_file():
        raise CorruptImageError(f"image not found: {uri}")
    try:
        image = Image.open(path)
        image.load()
    except OSError as exc:
        raise CorruptImageError(f"corrupt or unreadable image: {uri}") from exc
    return image.convert("RGB")


def _hue(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta == 0:
        return 0.0
    if mx == r:
        h = ((g - b) / delta) % 6.0
    elif mx == g:
        h = (b - r) / delta + 2.0
    else:
        h = (r - g) / delta + 4.0
    return h * 60.0


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _dominant_object_hue(image: Image.Image) -> float | None:
    """Mean hue of pixels that are not near-white / near-gray cardboard."""
    rgb_image = image.convert("RGB")
    px = rgb_image.load()
    width, height = rgb_image.size
    hues: list[float] = []
    for y in range(height):
        for x in range(width):
            r, g, b = px[x, y]
            mx, mn = max(r, g, b), min(r, g, b)
            if mx < 40:
                continue
            saturation = 0 if mx == 0 else (mx - mn) / mx
            if saturation < 0.25:
                continue
            hues.append(_hue((r, g, b)))
    if len(hues) < 50:
        return None
    hues.sort()
    mid = hues[len(hues) // 2]
    return mid


def _sharpness(image: Image.Image) -> float:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(gray).stddev[0])


def _connected_components(
    mask: list[list[bool]],
    min_pixels: int,
) -> list[tuple[int, int, int, int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    blobs: list[tuple[int, int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if count >= min_pixels:
                blobs.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, count))
    return blobs


def _boxes_near(
    a: tuple[int, int, int, int, int],
    b: tuple[int, int, int, int, int],
    gap: int,
) -> bool:
    ax, ay, aw, ah, _ = a
    bx, by, bw, bh, _ = b
    return not (
        ax + aw + gap < bx
        or bx + bw + gap < ax
        or ay + ah + gap < by
        or by + bh + gap < ay
    )


def _merge_nearby_blobs(
    blobs: list[tuple[int, int, int, int, int]],
    gap: int,
) -> list[tuple[int, int, int, int, int]]:
    remaining = list(blobs)
    merged: list[tuple[int, int, int, int, int]] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[tuple[int, int, int, int, int]] = []
            for other in remaining:
                if _boxes_near(current, other, gap):
                    x1 = min(current[0], other[0])
                    y1 = min(current[1], other[1])
                    x2 = max(current[0] + current[2], other[0] + other[2])
                    y2 = max(current[1] + current[3], other[1] + other[3])
                    current = (x1, y1, x2 - x1, y2 - y1, current[4] + other[4])
                    changed = True
                else:
                    keep.append(other)
            remaining = keep
        merged.append(current)
    return merged


def _signature_for_item(item: CatalogItem) -> _SkuSignature | None:
    hue: float | None = None
    for ref in item.reference_images:
        image = _open_rgb(ref.uri)
        hue = _dominant_object_hue(image)
        if hue is not None:
            break
    if hue is None:
        named = item.attributes.get("color", "").strip().lower()
        named_hues = {
            "red": 5.0,
            "blue": 220.0,
            "green": 120.0,
            "yellow": 55.0,
            "orange": 30.0,
            "purple": 280.0,
            "black": 0.0,
            "white": 0.0,
        }
        hue = named_hues.get(named)
    if hue is None:
        return None
    return _SkuSignature(sku=item.sku, name=item.product_name, hue=hue)


class HeuristicColorBlobBackend:
    """Match saturated blobs in the pack photo to catalogue reference hues.

    This is intentionally simple and deterministic for fixtures. It does not invent
    SKUs: unmatched blobs become UNKNOWN; confusable colourways become UNCERTAIN.
    """

    model_version = VISION_MODEL_VERSION

    def detect(self, event: PackEvent) -> tuple[list[DetectedItem], int]:
        started = time.perf_counter()
        signatures = []
        for item in event.catalogue:
            sig = _signature_for_item(item)
            if sig is not None:
                signatures.append(sig)

        detections: list[DetectedItem] = []
        for photo in event.photos:
            detections.extend(self._detect_photo(photo, signatures))
        latency_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        return detections, latency_ms

    def _detect_photo(
        self,
        photo: PackPhoto,
        signatures: list[_SkuSignature],
    ) -> list[DetectedItem]:
        image = _open_rgb(photo.uri)
        sharpness = _sharpness(image)
        width, height = image.size
        rgb = image.load()

        confusable = set()
        for i, a in enumerate(signatures):
            for b in signatures[i + 1 :]:
                if _hue_distance(a.hue, b.hue) < SIMILAR_HUE_THRESHOLD:
                    confusable.add(a.sku)
                    confusable.add(b.sku)

        # Downsample mask for speed on large photos.
        step = 2
        mask_h = height // step
        mask_w = width // step
        pixel_hues = [[None] * mask_w for _ in range(mask_h)]
        sat_mask = [[False] * mask_w for _ in range(mask_h)]
        for my in range(mask_h):
            for mx in range(mask_w):
                r, g, b = rgb[mx * step, my * step]
                mxv, mnv = max(r, g, b), min(r, g, b)
                sat = 0 if mxv == 0 else (mxv - mnv) / mxv
                if sat >= 0.28 and mxv >= 40:
                    sat_mask[my][mx] = True
                    pixel_hues[my][mx] = _hue((r, g, b))

        blobs = _connected_components(sat_mask, max(40, MIN_BLOB_PIXELS // (step * step)))
        blobs = _merge_nearby_blobs(blobs, gap=8)
        if sharpness < AMBIGUOUS_SHARPNESS and blobs:
            return [
                DetectedItem(
                    detection_id=str(uuid.uuid4()),
                    sku=UNCERTAIN_SKU,
                    label="ambiguous-blur",
                    quantity=1,
                    confidence=0.2,
                    image_id=photo.image_id,
                    reasoning=(
                        f"Evidence photo sharpness={sharpness:.1f} is below the "
                        "ambiguity threshold; identity is not asserted."
                    ),
                )
            ]

        detections: list[DetectedItem] = []
        for min_x, min_y, bw, bh, count in blobs:
            hues_in_blob = [
                pixel_hues[my][mx]
                for my in range(min_y, min_y + bh)
                for mx in range(min_x, min_x + bw)
                if sat_mask[my][mx] and pixel_hues[my][mx] is not None
            ]
            if not hues_in_blob:
                continue
            hues_in_blob.sort()
            blob_hue = hues_in_blob[len(hues_in_blob) // 2]

            ranked = sorted(signatures, key=lambda s: _hue_distance(s.hue, blob_hue))
            if not ranked:
                sku, name, dist = UNKNOWN_SKU, "unidentified-object", 999.0
            else:
                best = ranked[0]
                dist = _hue_distance(best.hue, blob_hue)
                sku, name = best.sku, best.name

            if dist > MATCH_HUE_THRESHOLD:
                sku, name = UNKNOWN_SKU, "unidentified-object"
                confidence = 0.55
                reason = f"Blob hue {blob_hue:.0f} is not within match distance of any catalogue SKU."
            elif sku in confusable:
                sku = UNCERTAIN_SKU
                confidence = 0.35
                reason = (
                    f"Blob hue {blob_hue:.0f} is consistent with visually similar catalogue "
                    "colourways; identity left UNCERTAIN."
                )
            else:
                confidence = max(0.5, min(0.98, 1.0 - dist / 40.0))
                reason = f"Blob hue {blob_hue:.0f} matches {name} ({sku}); hue distance {dist:.1f}."

            detections.append(
                DetectedItem(
                    detection_id=str(uuid.uuid4()),
                    sku=sku,
                    label=name,
                    quantity=1,
                    confidence=confidence,
                    image_id=photo.image_id,
                    bbox=BoundingBox(
                        x=float(min_x * step),
                        y=float(min_y * step),
                        w=float(bw * step),
                        h=float(bh * step),
                    ),
                    reasoning=reason,
                )
            )
        return detections
