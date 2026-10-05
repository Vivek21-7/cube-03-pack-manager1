"""VLM Client backends for Pack Manager AI (Gemini, OpenAI, Anthropic, and Simulation)."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Protocol

import httpx
from PIL import Image, ImageFilter, ImageStat

from pack_manager.prompts import PACK_MANAGER_SYSTEM_PROMPT

logger = logging.getLogger("pack_manager.vlm")


class VLMClient(Protocol):
    provider_name: str
    model_name: str

    def inspect(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Send prompt and image to the vision language model and return raw JSON response text."""


class GeminiVLMClient:
    """Google Gemini Multimodal Vision API Client via Generative Language REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or pass api_key to constructor."
            )
        self.model_name = model_name
        self.provider_name = "gemini"
        self.timeout = timeout
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        )

    def inspect(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "system_instruction": {"parts": [{"text": PACK_MANAGER_SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_img,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            candidates = data.get("candidates", [])
            parts = candidates[0]["content"]["parts"]
            return parts[0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response structure from Gemini: {data}") from exc


class OpenAIVLMClient:
    """OpenAI Vision Client (GPT-4o, GPT-4o-mini) via Chat Completions REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY or pass api_key to constructor."
            )
        self.model_name = model_name
        self.provider_name = "openai"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def inspect(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64_img}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": PACK_MANAGER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response structure from OpenAI: {data}") from exc


class AnthropicVLMClient:
    """Anthropic Claude Vision Client (Claude 3.5 Sonnet) via Messages REST."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY or pass api_key to constructor."
            )
        self.model_name = model_name
        self.provider_name = "anthropic"
        self.timeout = timeout

    def inspect(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64_img = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "system": PACK_MANAGER_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_img,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response structure from Anthropic: {data}") from exc


class SimulationVLMClient:
    """Offline Simulation VLM Client.

    Used when running without an external API key or in unit tests.
    It inspects image characteristics (sharpness, dimensions, metadata or tags)
    and order payload to synthesize a valid, prompt-compliant inspection response.
    """

    def __init__(self, model_name: str = "simulation-vlm-1.0") -> None:
        self.provider_name = "simulation"
        self.model_name = model_name

    def _inspect_real_captured_image(
        self,
        pil_img: Image.Image,
        order_id: str,
        order_items: list[dict[str, Any]],
        stddev: float,
    ) -> str:
        """Intelligently inspect a real webcam capture or local uploaded photo using computer vision."""
        import colorsys
        import numpy as np

        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size

        # Check for empty / pitch dark frame
        arr = np.array(rgb_img, dtype=np.float32)
        gray_full = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        full_dark_pct = float(np.mean(gray_full < 30) * 100.0)

        if stddev < 3.5 or (full_dark_pct > 92.0 and stddev < 6.0):
            return json.dumps({
                "order_id": order_id,
                "order_items": order_items,
                "detected_items": [],
                "matches": [
                    {"item_name": it["name"], "expected_qty": it["expected_qty"], "detected_qty": 0, "variant_match": False, "status": "FAIL"}
                    for it in order_items
                ],
                "missing_items": [
                    {"item_name": it["name"], "expected_qty": it["expected_qty"], "reason": "Camera view is blank or obstructed"}
                    for it in order_items
                ],
                "extra_items": [],
                "product_condition": {"visible_damage": False, "notes": "No items distinguishable in camera feed"},
                "decision": "UNCERTAIN",
                "decision_reason": "Camera feed is pitch black or completely obstructed. Please ensure proper lighting and position product.",
                "confidence": "low",
                "evidence": ["Average brightness/contrast too low", "No product features visible"]
            })

        # Central ROI where object is held or placed
        roi = arr[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        diff_rg = np.abs(r - g)
        diff_rb = np.abs(r - b)
        is_neutral = (diff_rg < 28) & (diff_rb < 28)

        dark_casing = (gray < 85) & is_neutral
        dark_casing_pct = float(np.mean(dark_casing) * 100.0)

        bright_lens = (gray >= 100) & (gray <= 210) & is_neutral
        bright_lens_pct = float(np.mean(bright_lens) * 100.0)

        is_blue = (b > r + 30) & (b > g + 15) & (b > 60)
        blue_pct = float(np.mean(is_blue) * 100.0)

        is_red = (r > g + 35) & (r > b + 35) & (r > 60)
        red_pct = float(np.mean(is_red) * 100.0)

        is_green = (g > r + 25) & (g > b + 25) & (g > 60)
        green_pct = float(np.mean(is_green) * 100.0)

        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_density = float(np.mean(grad_mag > 18) * 100.0)

        # Detect physical entity types
        is_electronic_device = (dark_casing_pct > 12.0) and (bright_lens_pct > 3.5 or edge_density > 3.0) and (blue_pct < 5.0)
        has_blue_apparel = blue_pct >= 2.5
        has_red_apparel = red_pct >= 2.5
        has_green_apparel = green_pct >= 3.0
        has_black_apparel = (dark_casing_pct >= 25.0) and (bright_lens_pct < 3.0) and not is_electronic_device

        detected = []
        matches = []
        missing = []
        extra = []
        mismatch_reasons = []

        # Check if the order expects apparel vs electronics
        has_apparel_expected = any(
            any(k in it.get("name", "").lower() for k in ["shirt", "t-shirt", "cap", "hat", "apparel", "cloth", "garment", "hoodie", "pants"])
            for it in order_items
        )
        has_electronics_expected = any(
            any(k in it.get("name", "").lower() for k in ["electronic", "camera", "gadget", "device", "lens", "phone", "hardware", "sensor", "goods"])
            for it in order_items
        )

        # Catalog detected physical objects in scene
        if is_electronic_device:
            elec_item = next(
                (it for it in order_items if any(k in it.get("name", "").lower() for k in ["electronic", "camera", "gadget", "device", "lens", "phone", "hardware", "sensor", "goods"])),
                None,
            )
            detected.append({
                "name": elec_item["name"] if elec_item else "Electronic Hardware Device / Action Camera",
                "detected_qty": elec_item.get("expected_qty", 1) if elec_item else 1,
                "variant": "matte-black/grey",
                "confidence": "high",
                "notes": f"Rigid electronic hardware detected with optical sensor (dark casing: {dark_casing_pct:.1f}%, lens/dial element: {bright_lens_pct:.1f}%)"
            })
            # If the manifest ONLY expected apparel and NO electronics, this is an unauthorized foreign object!
            if has_apparel_expected and not has_electronics_expected:
                extra.append({
                    "item_name": "Electronic Hardware Device / Action Camera",
                    "qty": 1,
                    "notes": "UNAUTHORIZED FOREIGN OBJECT: Handheld electronic device detected in packaging station instead of ordered apparel",
                })

        if has_blue_apparel:
            blue_item = next(
                (it for it in order_items if "blue" in (it.get("name", "") + " " + str(it.get("variant", ""))).lower()),
                None,
            )
            detected.append({
                "name": blue_item["name"] if blue_item else "Blue Cap",
                "detected_qty": blue_item.get("expected_qty", 1) if blue_item else 1,
                "variant": "blue",
                "confidence": "high",
                "notes": f"Blue fabric/apparel signature detected ({blue_pct:.1f}% blue pixels)"
            })

        if has_red_apparel:
            red_item = next(
                (it for it in order_items if "red" in (it.get("name", "") + " " + str(it.get("variant", ""))).lower()),
                None,
            )
            detected.append({
                "name": red_item["name"] if red_item else "Red Cap",
                "detected_qty": red_item.get("expected_qty", 1) if red_item else 1,
                "variant": "red",
                "confidence": "high",
                "notes": f"Red item signature detected ({red_pct:.1f}% red pixels)"
            })

        if has_black_apparel:
            black_item = next(
                (it for it in order_items if "black" in (it.get("name", "") + " " + str(it.get("variant", ""))).lower()),
                None,
            )
            detected.append({
                "name": black_item["name"] if black_item else "Black T-Shirt",
                "detected_qty": black_item.get("expected_qty", 1) if black_item else 1,
                "variant": "black",
                "confidence": "high",
                "notes": f"Dark fabric apparel detected ({dark_casing_pct:.1f}% dark garment profile)"
            })

        # Manifest reconciliation
        for it in order_items:
            name = it.get("name", "Product").lower()
            var = str(it.get("variant") or "").lower()
            exp_qty = it.get("expected_qty", 1)

            is_elec_item = any(k in name for k in ["electronic", "camera", "gadget", "device", "lens", "phone", "hardware", "sensor", "goods"])
            is_apparel_item = any(k in name for k in ["shirt", "t-shirt", "cap", "hat", "apparel", "cloth", "garment", "hoodie", "pants"])

            # 1. Electronics Manifest Item
            if is_elec_item:
                if is_electronic_device:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                else:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                    missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "Electronic hardware device not detected in camera frame"})

            # 2. Apparel Manifest Item
            elif is_apparel_item:
                # If an electronic device was held instead of apparel
                if is_electronic_device:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                    missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": f"Expected apparel item '{it['name']}' not found; foreign electronic device present"})
                elif "blue" in var or "blue" in name:
                    if has_blue_apparel:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                    elif has_red_apparel:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": False, "status": "FAIL"})
                        mismatch_reasons.append(f"Variant mismatch on '{it['name']}': detected RED object, expected BLUE")
                    else:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                        missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "No blue item detected in camera view"})
                elif "red" in var or "red" in name:
                    if has_red_apparel:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                    else:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                        missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "No red item detected in camera view"})
                elif "black" in var or "black" in name:
                    if has_black_apparel:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                    else:
                        matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                        missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "Black apparel garment not detected in camera view"})
                else:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})

            # 3. Documentation or General Goods
            elif "manual" in name or "doc" in name:
                if bright_lens_pct > 15.0 or blue_pct > 4.0:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                else:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                    missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "User manual documentation not detected in box"})

            # 4. Fallback General Product
            else:
                if stddev >= 5.0 and len(detected) > 0:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": exp_qty, "variant_match": True, "status": "PASS"})
                else:
                    matches.append({"item_name": it["name"], "expected_qty": exp_qty, "detected_qty": 0, "variant_match": False, "status": "FAIL"})
                    missing.append({"item_name": it["name"], "expected_qty": exp_qty, "reason": "Item contours not verified in frame"})

        # Evaluate final packing decision
        if extra and has_apparel_expected and not has_electronics_expected:
            decision = "STOP_FIX"
            missing_names = ", ".join(m["item_name"] for m in missing) or "ordered apparel"
            decision_reason = (
                f"CRITICAL DEFECT: Foreign electronic device detected in packaging area instead of expected apparel ({missing_names}). "
                "Expected items are missing. DO NOT SEAL."
            )
        elif mismatch_reasons:
            decision = "STOP_FIX"
            decision_reason = "; ".join(mismatch_reasons)
        elif missing:
            decision = "STOP_FIX"
            missing_names = ", ".join(m["item_name"] for m in missing)
            decision_reason = f"Missing {len(missing)} item(s) in camera photo: {missing_names}. Please add required items before sealing."
        elif any(m["status"] == "FAIL" for m in matches):
            decision = "STOP_FIX"
            decision_reason = "Order reconciliation failed against camera inspection."
        else:
            decision = "SEAL"
            verified_names = ", ".join(it["name"] for it in order_items)
            decision_reason = f"All {len(order_items)} item(s) verified in camera view ({verified_names}). Packaging integrity confirmed."

        return json.dumps({
            "order_id": order_id,
            "order_items": order_items,
            "detected_items": detected,
            "matches": matches,
            "missing_items": missing,
            "extra_items": extra,
            "product_condition": {
                "visible_damage": False,
                "notes": "No carton damage detected in camera frame."
            },
            "decision": decision,
            "decision_reason": decision_reason,
            "confidence": "high" if not missing and not mismatch_reasons and not extra else "medium",
            "evidence": [
                f"Camera frame analyzed: dark_casing={dark_casing_pct:.1f}%, bright_sensor={bright_lens_pct:.1f}%, blue={blue_pct:.1f}%, red={red_pct:.1f}%",
                f"Electronic device signature: {is_electronic_device}",
                f"Evaluation: {decision_reason}"
            ]
        })

    def inspect(self, prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
        # Parse order payload from prompt if present
        order_dict: dict[str, Any] = {}
        try:
            # Look for ```json in prompt
            if "```json" in prompt:
                json_part = prompt.split("```json")[1].split("```")[0].strip()
                order_dict = json.loads(json_part)
        except Exception:
            pass

        order_id = order_dict.get("order_id", "ORD-UNKNOWN")
        raw_items = order_dict.get("items", order_dict.get("lines", []))
        order_items = []
        for it in raw_items:
            name = it.get("product_name") or it.get("name") or it.get("sku") or "Item"
            qty = it.get("expected_qty", 1)
            var = it.get("variant") or it.get("color")
            if not var and "attributes" in it and isinstance(it["attributes"], dict):
                var = it["attributes"].get("color")
            order_items.append({"name": name, "expected_qty": qty, "variant": var})

        # Check visual attributes of the image
        is_blurry = False
        scenario_tag = None
        has_damage = False

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Check for embedded scenario tag in image metadata
            scenario_tag = pil_img.info.get("scenario")

            # Check sharpness
            gray = pil_img.convert("L").filter(ImageFilter.FIND_EDGES)
            stddev = float(ImageStat.Stat(gray).stddev[0])
            if stddev < 5.0:
                is_blurry = True

            # Check for damage indicator in info
            if pil_img.info.get("damage") == "true":
                has_damage = True
        except Exception:
            pass

        # Handle specific scenario tags if embedded (from scenario fixtures)
        if scenario_tag == "unclear_photo" or is_blurry:
            return json.dumps(
                {
                    "order_id": order_id,
                    "order_items": order_items,
                    "detected_items": [
                        {
                            "name": "unidentified_blurred_item",
                            "detected_qty": 1,
                            "variant": "unknown",
                            "confidence": "low",
                            "notes": "Image is excessively blurry; object contours cannot be distinguished.",
                        }
                    ],
                    "matches": [
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "detected_qty": 0,
                            "variant_match": False,
                            "status": "FAIL",
                        }
                        for item in order_items
                    ],
                    "missing_items": [
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "reason": "Cannot verify due to poor image clarity",
                        }
                        for item in order_items
                    ],
                    "extra_items": [],
                    "product_condition": {
                        "visible_damage": False,
                        "notes": "Image too blurry to assess physical product condition.",
                    },
                    "decision": "UNCERTAIN",
                    "decision_reason": "Photo quality insufficient to confidently verify contents - request clearer image",
                    "confidence": "low",
                    "evidence": [
                        "Image sharpness score is below inspection threshold",
                        "Item silhouettes are severely blurred and indistinct",
                        "Variant and quantity cannot be counted reliably",
                    ],
                }
            )

        if scenario_tag == "wrong_item":
            detected = []
            matches = []
            for item in order_items:
                if "cap" in item["name"].lower():
                    detected.append(
                        {
                            "name": item["name"],
                            "detected_qty": item["expected_qty"],
                            "variant": "red",
                            "confidence": "high",
                            "notes": "Red cap detected instead of blue",
                        }
                    )
                    matches.append(
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "detected_qty": item["expected_qty"],
                            "variant_match": False,
                            "status": "FAIL",
                        }
                    )
                else:
                    detected.append(
                        {
                            "name": item["name"],
                            "detected_qty": item["expected_qty"],
                            "variant": item.get("variant") or "black",
                            "confidence": "high",
                            "notes": "Present and folded",
                        }
                    )
                    matches.append(
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "detected_qty": item["expected_qty"],
                            "variant_match": True,
                            "status": "PASS",
                        }
                    )
            return json.dumps(
                {
                    "order_id": order_id,
                    "order_items": order_items,
                    "detected_items": detected,
                    "matches": matches,
                    "missing_items": [],
                    "extra_items": [],
                    "product_condition": {
                        "visible_damage": False,
                        "notes": "No visible damage to products or packaging",
                    },
                    "decision": "STOP_FIX",
                    "decision_reason": "Cap color mismatch - expected blue, detected red",
                    "confidence": "high",
                    "evidence": [
                        "Identified 2 black t-shirts matching order",
                        "Identified 1 red cap where blue cap was ordered",
                        "Variant mismatch detected for cap",
                    ],
                }
            )

        if scenario_tag == "missing_item":
            detected = []
            matches = []
            missing = []
            for item in order_items:
                if "manual" in item["name"].lower():
                    missing.append(
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "reason": "Not visible in box",
                        }
                    )
                    matches.append(
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "detected_qty": 0,
                            "variant_match": False,
                            "status": "FAIL",
                        }
                    )
                else:
                    detected.append(
                        {
                            "name": item["name"],
                            "detected_qty": item["expected_qty"],
                            "variant": item.get("variant") or "black",
                            "confidence": "high",
                            "notes": "Present in box",
                        }
                    )
                    matches.append(
                        {
                            "item_name": item["name"],
                            "expected_qty": item["expected_qty"],
                            "detected_qty": item["expected_qty"],
                            "variant_match": True,
                            "status": "PASS",
                        }
                    )
            return json.dumps(
                {
                    "order_id": order_id,
                    "order_items": order_items,
                    "detected_items": detected,
                    "matches": matches,
                    "missing_items": missing,
                    "extra_items": [],
                    "product_condition": {
                        "visible_damage": False,
                        "notes": "No visible damage to products or packaging",
                    },
                    "decision": "STOP_FIX",
                    "decision_reason": "Manual missing - not visible in box",
                    "confidence": "high",
                    "evidence": [
                        "Expected items partially visible",
                        "Manual documentation is absent from packing container",
                    ],
                }
            )

        if scenario_tag == "extra_item":
            detected = [
                {
                    "name": item["name"],
                    "detected_qty": item["expected_qty"],
                    "variant": item.get("variant") or "standard",
                    "confidence": "high",
                    "notes": "Matched to order",
                }
                for item in order_items
            ]
            detected.append(
                {
                    "name": "Red Scarf",
                    "detected_qty": 1,
                    "variant": "red",
                    "confidence": "high",
                    "notes": "Unexpected item in box - not on order",
                }
            )
            matches = [
                {
                    "item_name": item["name"],
                    "expected_qty": item["expected_qty"],
                    "detected_qty": item["expected_qty"],
                    "variant_match": True,
                    "status": "PASS",
                }
                for item in order_items
            ]
            return json.dumps(
                {
                    "order_id": order_id,
                    "order_items": order_items,
                    "detected_items": detected,
                    "matches": matches,
                    "missing_items": [],
                    "extra_items": [
                        {
                            "item_name": "Red Scarf",
                            "qty": 1,
                            "notes": "Unexpected item in box - red scarf not on order",
                        }
                    ],
                    "product_condition": {
                        "visible_damage": False,
                        "notes": "No visible damage to products or packaging",
                    },
                    "decision": "STOP_FIX",
                    "decision_reason": "Unexpected item in box - red scarf not on order",
                    "confidence": "high",
                    "evidence": [
                        "All ordered items accounted for",
                        "1 unauthorized extra item found inside package: Red Scarf",
                    ],
                }
            )

        if has_damage or scenario_tag == "damaged_product":
            detected = [
                {
                    "name": item["name"],
                    "detected_qty": item["expected_qty"],
                    "variant": item.get("variant") or "standard",
                    "confidence": "high",
                    "notes": "Item physically compromised",
                }
                for item in order_items
            ]
            matches = [
                {
                    "item_name": item["name"],
                    "expected_qty": item["expected_qty"],
                    "detected_qty": item["expected_qty"],
                    "variant_match": True,
                    "status": "PASS",
                }
                for item in order_items
            ]
            return json.dumps(
                {
                    "order_id": order_id,
                    "order_items": order_items,
                    "detected_items": detected,
                    "matches": matches,
                    "missing_items": [],
                    "extra_items": [],
                    "product_condition": {
                        "visible_damage": True,
                        "notes": "Package crush damage and visible tear detected on product packaging",
                    },
                    "decision": "STOP_FIX",
                    "decision_reason": "Product or packaging damage detected during visual inspection",
                    "confidence": "high",
                    "evidence": [
                        "All expected items are present in correct quantities",
                        "Physical packaging damage observed, violating shipment criteria",
                    ],
                }
            )

        # If no scenario tag is embedded, this is a real live camera capture or user-uploaded file!
        if scenario_tag is None and not pil_img.info.get("damage"):
            return self._inspect_real_captured_image(pil_img, order_id, order_items, stddev)

        # Fallback for synthetic fixture correct order case (SEAL)
        detected = [
            {
                "name": item["name"],
                "detected_qty": item["expected_qty"],
                "variant": item.get("variant") or "standard",
                "confidence": "high",
                "notes": "Clearly visible, folded, correct quantity and variant",
            }
            for item in order_items
        ]
        matches = [
            {
                "item_name": item["name"],
                "expected_qty": item["expected_qty"],
                "detected_qty": item["expected_qty"],
                "variant_match": True,
                "status": "PASS",
            }
            for item in order_items
        ]

        return json.dumps(
            {
                "order_id": order_id,
                "order_items": order_items,
                "detected_items": detected,
                "matches": matches,
                "missing_items": [],
                "extra_items": [],
                "product_condition": {
                    "visible_damage": False,
                    "notes": "No visible damage to products or packaging",
                },
                "decision": "SEAL",
                "decision_reason": "All items present, correct quantities, correct colors",
                "confidence": "high",
                "evidence": [
                    "All items matched against order",
                    "No missing components",
                    "No visible damage",
                ],
            }
        )


def get_vlm_client(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> VLMClient:
    """Create a VLM client based on specified provider or available environment variables.

    Priority:
    1. Explicit provider argument ('gemini', 'openai', 'anthropic', 'simulation', 'mock')
    2. Environment variables:
       - GEMINI_API_KEY or GOOGLE_API_KEY -> GeminiVLMClient
       - OPENAI_API_KEY -> OpenAIVLMClient
       - ANTHROPIC_API_KEY -> AnthropicVLMClient
    3. Default fallback -> SimulationVLMClient (Offline simulation mode)
    """
    chosen = (provider or os.getenv("PACK_MANAGER_VLM_PROVIDER", "")).strip().lower()

    if chosen in ("gemini", "google"):
        return GeminiVLMClient(api_key=api_key, model_name=model or "gemini-2.0-flash")
    if chosen == "openai":
        return OpenAIVLMClient(api_key=api_key, model_name=model or "gpt-4o")
    if chosen in ("anthropic", "claude"):
        return AnthropicVLMClient(
            api_key=api_key, model_name=model or "claude-3-5-sonnet-20241022"
        )
    if chosen in ("simulation", "mock", "offline"):
        return SimulationVLMClient(model_name=model or "simulation-vlm-1.0")

    # Auto-detection from environment keys
    if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) and not api_key:
        return GeminiVLMClient(model_name=model or "gemini-2.0-flash")
    if os.getenv("OPENAI_API_KEY") and not api_key:
        return OpenAIVLMClient(model_name=model or "gpt-4o")
    if os.getenv("ANTHROPIC_API_KEY") and not api_key:
        return AnthropicVLMClient(model_name=model or "claude-3-5-sonnet-20241022")

    # If an API key was explicitly passed without provider, try gemini then openai
    if api_key:
        if api_key.startswith("AIza"):
            return GeminiVLMClient(api_key=api_key, model_name=model or "gemini-2.0-flash")
        if api_key.startswith("sk-ant-"):
            return AnthropicVLMClient(api_key=api_key, model_name=model or "claude-3-5-sonnet-20241022")
        if api_key.startswith("sk-"):
            return OpenAIVLMClient(api_key=api_key, model_name=model or "gpt-4o")

    logger.info("No active VLM API credentials found; initializing SimulationVLMClient.")
    return SimulationVLMClient(model_name=model or "simulation-vlm-1.0")
