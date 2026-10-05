"""Generate realistic multi-scenario fixtures matching the Pack Manager AI specification."""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, PngImagePlugin


def _draw_box(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    # Outer packing box / cardboard styling
    draw.rectangle([10, 10, width - 10, height - 10], outline=(150, 115, 80), width=8)
    draw.rectangle([18, 18, width - 18, height - 18], fill=(225, 210, 185))
    # Box flap inner seams
    draw.line([(18, 50), (width - 18, 50)], fill=(180, 150, 115), width=2)
    draw.line([(18, height - 50), (width - 18, height - 50)], fill=(180, 150, 115), width=2)


def _draw_black_tshirt(draw: ImageDraw.ImageDraw, x: int, y: int, label: str = "T-Shirt") -> None:
    # Folded black garment rectangle with collar crease
    draw.rectangle([x, y, x + 130, y + 100], fill=(28, 28, 30), outline=(50, 50, 55), width=2)
    # Crewneck collar curve
    draw.arc([x + 35, y + 5, x + 95, y + 35], 0, 180, fill=(70, 70, 75), width=3)
    # Fold lines
    draw.line([(x + 10, y + 50), (x + 120, y + 50)], fill=(45, 45, 50), width=1)
    draw.text((x + 35, y + 65), "M • 100% COTTON", fill=(120, 120, 130))


def _draw_cap(draw: ImageDraw.ImageDraw, x: int, y: int, color_rgb: tuple[int, int, int], text: str) -> None:
    # Cap crown
    draw.ellipse([x, y, x + 110, y + 90], fill=color_rgb, outline=(30, 30, 30), width=2)
    # Cap visor / bill
    draw.chord([x + 15, y + 50, x + 130, y + 115], 10, 170, fill=color_rgb, outline=(20, 20, 20), width=2)
    # Button on crown
    draw.ellipse([x + 50, y + 10, x + 60, y + 20], fill=(240, 240, 240))
    draw.text((x + 30, y + 35), text, fill=(255, 255, 255))


def _draw_manual(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    # White booklet
    draw.rectangle([x, y, x + 85, y + 110], fill=(250, 250, 250), outline=(160, 160, 160), width=2)
    draw.rectangle([x + 10, y + 10, x + 75, y + 25], fill=(40, 100, 200))
    draw.text((x + 15, y + 12), "USER GUIDE", fill=(255, 255, 255))
    for row in range(5):
        ly = y + 35 + row * 12
        draw.line([(x + 10, ly), (x + 75, ly)], fill=(200, 200, 200), width=2)


def _draw_scarf(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    # Red scarf with fringe
    draw.rectangle([x, y, x + 120, y + 70], fill=(210, 30, 45), outline=(140, 15, 25), width=2)
    for fringe in range(0, 120, 10):
        draw.line([(x + fringe, y + 70), (x + fringe, y + 85)], fill=(210, 30, 45), width=2)
    draw.text((x + 20, y + 25), "RED SCARF", fill=(255, 230, 230))


def _draw_damage(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    # Punctures and tear marks across the box
    draw.polygon([(80, 40), (130, 80), (105, 120), (60, 90)], fill=(60, 40, 20), outline=(20, 10, 5), width=2)
    draw.line([(60, 90), (40, 150)], fill=(30, 15, 5), width=3)
    draw.line([(130, 80), (170, 70)], fill=(30, 15, 5), width=3)


def generate_all_scenarios(output_dir: Path) -> dict[str, dict[str, Path]]:
    """Generate sample fixtures and order JSONs for all 5 prompt examples (+ damaged case)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: dict[str, dict[str, Path]] = {}

    # Common order baseline for T-Shirt + Cap
    standard_order = {
        "order_id": "ORD-2024-001",
        "items": [
            {
                "sku": "TSHIRT-BLK-M",
                "name": "Black T-Shirt",
                "expected_qty": 2,
                "variant": "black",
            },
            {
                "sku": "CAP-BLU-001",
                "name": "Blue Cap",
                "expected_qty": 1,
                "variant": "blue",
            },
        ],
    }

    # 1. Correct Order (SEAL)
    s1_dir = output_dir / "example_1_correct_order"
    s1_dir.mkdir(exist_ok=True)
    img1 = Image.new("RGB", (560, 400), (240, 240, 240))
    d1 = ImageDraw.Draw(img1)
    _draw_box(d1, 560, 400)
    _draw_black_tshirt(d1, 40, 80)
    _draw_black_tshirt(d1, 190, 80)
    _draw_cap(d1, 350, 100, (30, 90, 210), "BLUE CAP")
    info1 = PngImagePlugin.PngInfo()
    info1.add_text("scenario", "correct_order")
    p1 = s1_dir / "box_photo.png"
    img1.save(p1, format="PNG", pnginfo=info1)
    o1 = s1_dir / "order.json"
    o1.write_text(json.dumps(standard_order, indent=2), encoding="utf-8")
    scenarios["example_1_correct_order"] = {"order": o1, "photo": p1}

    # 2. Wrong Item (STOP & FIX) - Red cap instead of blue cap
    s2_dir = output_dir / "example_2_wrong_item"
    s2_dir.mkdir(exist_ok=True)
    img2 = Image.new("RGB", (560, 400), (240, 240, 240))
    d2 = ImageDraw.Draw(img2)
    _draw_box(d2, 560, 400)
    _draw_black_tshirt(d2, 40, 80)
    _draw_black_tshirt(d2, 190, 80)
    _draw_cap(d2, 350, 100, (215, 35, 40), "RED CAP")
    info2 = PngImagePlugin.PngInfo()
    info2.add_text("scenario", "wrong_item")
    p2 = s2_dir / "box_photo.png"
    img2.save(p2, format="PNG", pnginfo=info2)
    o2 = s2_dir / "order.json"
    o2.write_text(json.dumps(standard_order, indent=2), encoding="utf-8")
    scenarios["example_2_wrong_item"] = {"order": o2, "photo": p2}

    # 3. Missing Item (STOP & FIX) - Expected Manual missing
    s3_dir = output_dir / "example_3_missing_item"
    s3_dir.mkdir(exist_ok=True)
    order_with_manual = {
        "order_id": "ORD-2024-003",
        "items": [
            {
                "sku": "TSHIRT-BLK-M",
                "name": "Black T-Shirt",
                "expected_qty": 2,
                "variant": "black",
            },
            {
                "sku": "CAP-BLU-001",
                "name": "Blue Cap",
                "expected_qty": 1,
                "variant": "blue",
            },
            {
                "sku": "DOC-MANUAL-01",
                "name": "Manual",
                "expected_qty": 1,
                "variant": "standard",
            },
        ],
    }
    img3 = Image.new("RGB", (560, 400), (240, 240, 240))
    d3 = ImageDraw.Draw(img3)
    _draw_box(d3, 560, 400)
    _draw_black_tshirt(d3, 40, 80)
    _draw_black_tshirt(d3, 190, 80)
    _draw_cap(d3, 350, 100, (30, 90, 210), "BLUE CAP")
    # Note: Manual is intentionally NOT drawn
    info3 = PngImagePlugin.PngInfo()
    info3.add_text("scenario", "missing_item")
    p3 = s3_dir / "box_photo.png"
    img3.save(p3, format="PNG", pnginfo=info3)
    o3 = s3_dir / "order.json"
    o3.write_text(json.dumps(order_with_manual, indent=2), encoding="utf-8")
    scenarios["example_3_missing_item"] = {"order": o3, "photo": p3}

    # 4. Extra Item (STOP & FIX) - Unexpected Red Scarf
    s4_dir = output_dir / "example_4_extra_item"
    s4_dir.mkdir(exist_ok=True)
    img4 = Image.new("RGB", (560, 400), (240, 240, 240))
    d4 = ImageDraw.Draw(img4)
    _draw_box(d4, 560, 400)
    _draw_black_tshirt(d4, 30, 80)
    _draw_black_tshirt(d4, 170, 80)
    _draw_cap(d4, 320, 70, (30, 90, 210), "BLUE CAP")
    _draw_scarf(d4, 320, 200)
    info4 = PngImagePlugin.PngInfo()
    info4.add_text("scenario", "extra_item")
    p4 = s4_dir / "box_photo.png"
    img4.save(p4, format="PNG", pnginfo=info4)
    o4 = s4_dir / "order.json"
    o4.write_text(json.dumps(standard_order, indent=2), encoding="utf-8")
    scenarios["example_4_extra_item"] = {"order": o4, "photo": p4}

    # 5. Unclear Photo (UNCERTAIN) - Blurry image
    s5_dir = output_dir / "example_5_unclear_photo"
    s5_dir.mkdir(exist_ok=True)
    img5 = Image.new("RGB", (560, 400), (240, 240, 240))
    d5 = ImageDraw.Draw(img5)
    _draw_box(d5, 560, 400)
    _draw_black_tshirt(d5, 40, 80)
    _draw_black_tshirt(d5, 190, 80)
    _draw_cap(d5, 350, 100, (30, 90, 210), "BLUE CAP")
    # Apply heavy blur filter to simulate motion blur / poor camera focus
    img5 = img5.filter(ImageFilter.GaussianBlur(radius=14))
    info5 = PngImagePlugin.PngInfo()
    info5.add_text("scenario", "unclear_photo")
    p5 = s5_dir / "box_photo.png"
    img5.save(p5, format="PNG", pnginfo=info5)
    o5 = s5_dir / "order.json"
    o5.write_text(json.dumps(standard_order, indent=2), encoding="utf-8")
    scenarios["example_5_unclear_photo"] = {"order": o5, "photo": p5}

    # 6. Damaged Box / Product (STOP & FIX)
    s6_dir = output_dir / "example_6_damaged_goods"
    s6_dir.mkdir(exist_ok=True)
    img6 = Image.new("RGB", (560, 400), (240, 240, 240))
    d6 = ImageDraw.Draw(img6)
    _draw_box(d6, 560, 400)
    _draw_black_tshirt(d6, 40, 80)
    _draw_black_tshirt(d6, 190, 80)
    _draw_cap(d6, 350, 100, (30, 90, 210), "BLUE CAP")
    _draw_damage(d6, 560, 400)
    info6 = PngImagePlugin.PngInfo()
    info6.add_text("scenario", "damaged_product")
    p6 = s6_dir / "box_photo.png"
    img6.save(p6, format="PNG", pnginfo=info6)
    o6 = s6_dir / "order.json"
    o6.write_text(json.dumps(standard_order, indent=2), encoding="utf-8")
    scenarios["example_6_damaged_goods"] = {"order": o6, "photo": p6}

    # 7. Electronics Order (SEAL) - Action Camera / Electronic Device
    s7_dir = output_dir / "example_7_electronics_order"
    s7_dir.mkdir(exist_ok=True)
    electronics_order = {
        "order_id": "ORD-2024-009",
        "items": [
            {
                "sku": "CAM-ACT-001",
                "name": "Action Camera / Electronic Device",
                "expected_qty": 1,
                "variant": "matte-black",
            }
        ],
    }
    img7 = Image.new("RGB", (560, 400), (240, 240, 240))
    d7 = ImageDraw.Draw(img7)
    _draw_box(d7, 560, 400)
    # Draw rigid dark camera body with circular lens
    d7.rectangle([170, 110, 390, 290], fill=(28, 30, 34), outline=(15, 15, 18), width=3)
    d7.ellipse([230, 140, 330, 240], fill=(70, 75, 85), outline=(180, 190, 205), width=4)
    d7.ellipse([255, 165, 305, 215], fill=(20, 22, 28), outline=(100, 120, 150), width=3)
    d7.text((215, 255), "ACTION CAM 4K", fill=(200, 200, 210))
    info7 = PngImagePlugin.PngInfo()
    info7.add_text("scenario", "electronics_order")
    p7 = s7_dir / "box_photo.png"
    img7.save(p7, format="PNG", pnginfo=info7)
    o7 = s7_dir / "order.json"
    o7.write_text(json.dumps(electronics_order, indent=2), encoding="utf-8")
    scenarios["example_7_electronics_order"] = {"order": o7, "photo": p7}

    return scenarios
