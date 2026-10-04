"""Deterministic PNG fixtures for the correct-order path."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def draw_reference_cap(color: tuple[int, int, int], path: Path) -> None:
    image = Image.new("RGB", (160, 160), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 48, 135, 140), fill=color)
    draw.rectangle((40, 36, 120, 70), fill=color)
    _save(image, path)


def draw_open_box_with_caps(
    path: Path,
    cap_colors: list[tuple[int, int, int]],
    box_size: tuple[int, int] = (480, 360),
) -> None:
    width, height = box_size
    # Background stays desaturated so only product blobs pass the hue mask.
    image = Image.new("RGB", (width, height), (196, 196, 192))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, width - 21, height - 21), outline=(110, 110, 108), width=8)
    n = max(1, len(cap_colors))
    for i, color in enumerate(cap_colors):
        cx = 90 + i * min(140, (width - 120) // n)
        cy = height // 2
        draw.ellipse((cx - 50, cy - 40, cx + 50, cy + 55), fill=color)
        draw.rectangle((cx - 38, cy - 48, cx + 38, cy - 22), fill=color)
    _save(image, path)


def write_correct_order_images(root: Path) -> dict[str, Path]:
    ref = root / "catalogue" / "blue_cap_ref.png"
    pack = root / "photos" / "pack_open.png"
    draw_reference_cap((30, 90, 200), ref)
    draw_open_box_with_caps(pack, [(30, 90, 200)])
    return {"reference": ref, "pack": pack}
