"""Pack Manager AI Agent: Autonomous logistics verification agent powered by VLM."""

from __future__ import annotations

import argparse
import io
import json
import logging
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from pack_manager.agent_schemas import (
    OrderItemInput,
    PackManagerAIResponse,
    MatchResultAI,
    MissingItemAI,
    ExtraItemAI,
)
from pack_manager.models import BoundingBox, DetectedItem, Order, PackEvent
from pack_manager.prompts import build_user_prompt
from pack_manager.vlm_client import SimulationVLMClient, VLMClient, get_vlm_client

logger = logging.getLogger("pack_manager.agent")


def _clean_json_markdown(raw: str) -> str:
    """Strip markdown code blocks or extra whitespace that models might emit."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Match ```json or ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Strip initial line and trailing ```
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
    return cleaned


def _normalize_order(order_input: Any) -> dict[str, Any]:
    """Convert Order model, dict, or path to standardized order dict for the agent prompt."""
    if isinstance(order_input, (str, Path)) and Path(order_input).is_file():
        order_input = json.loads(Path(order_input).read_text(encoding="utf-8"))

    if isinstance(order_input, Order):
        lines = []
        for line in order_input.lines:
            lines.append(
                {
                    "name": line.product_name,
                    "sku": line.sku,
                    "expected_qty": line.expected_qty,
                    "variant": None,
                }
            )
        return {
            "order_id": order_input.order_id,
            "items": lines,
        }

    if isinstance(order_input, dict):
        order_id = order_input.get("order_id", "ORD-UNKNOWN")
        raw_items = order_input.get("items") or order_input.get("order_items") or order_input.get("lines") or []
        items = []
        for it in raw_items:
            if isinstance(it, dict):
                name = it.get("name") or it.get("product_name") or it.get("sku") or "Item"
                expected_qty = it.get("expected_qty", 1)
                variant = it.get("variant")
                if not variant and "attributes" in it and isinstance(it["attributes"], dict):
                    variant = it["attributes"].get("color")
                sku = it.get("sku")
                items.append(
                    {
                        "name": name,
                        "expected_qty": int(expected_qty),
                        "variant": variant,
                        "sku": sku,
                    }
                )
        return {"order_id": order_id, "items": items}

    raise ValueError(f"Unsupported order input format: {type(order_input)}")


def _load_image_bytes(photo: Path | str | bytes | Image.Image) -> tuple[bytes, str]:
    """Extract raw image bytes and mime type from multiple formats."""
    if isinstance(photo, bytes):
        return photo, "image/png"

    if isinstance(photo, Image.Image):
        buf = io.BytesIO()
        # Preserve info (such as scenario tags) if available
        photo.save(buf, format="PNG", pnginfo=photo.info.get("pnginfo"))
        return buf.getvalue(), "image/png"

    path = Path(photo)
    if not path.is_file():
        raise FileNotFoundError(f"Box photo image not found at {path}")

    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    return path.read_bytes(), mime


class PackManagerAIAgent:
    """The Real AI Logistics Inspector Agent executing order verification via Vision LLM."""

    def __init__(self, vlm_client: VLMClient | None = None) -> None:
        self.vlm = vlm_client or get_vlm_client()

    def verify(
        self,
        order: dict[str, Any] | Order | Path | str,
        photo: Path | str | bytes | Image.Image,
    ) -> PackManagerAIResponse:
        """Analyze box photograph against expected order and output canonical verification response."""
        order_dict = _normalize_order(order)
        image_bytes, mime_type = _load_image_bytes(photo)

        prompt = build_user_prompt(order_dict)
        raw_output = self.vlm.inspect(prompt, image_bytes, mime_type=mime_type)
        cleaned_json = _clean_json_markdown(raw_output)

        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            logger.error("VLM did not return valid JSON. Raw output:\n%s", raw_output)
            raise ValueError(f"Failed to parse VLM response as JSON: {exc}") from exc

        # First, ensure we have the expected order items in the parsed response
        if "order_items" not in parsed or not parsed["order_items"]:
            parsed["order_items"] = order_dict["items"]

        # Validate base structure (LLM output)
        response = PackManagerAIResponse.model_validate(parsed)
        
        # Enforce deterministic comparison engine
        return self._reconcile_inspection_result(response)

    def _reconcile_inspection_result(self, response: PackManagerAIResponse) -> PackManagerAIResponse:
        matches = []
        missing_items = []
        extra_items = []
        
        detected_unmatched = list(response.detected_items)
        
        for expected in response.order_items:
            best_match_idx = -1
            for i, det in enumerate(detected_unmatched):
                if det.name.lower() == expected.name.lower():
                    best_match_idx = i
                    break
                if expected.name.lower() in det.name.lower() or det.name.lower() in expected.name.lower():
                    best_match_idx = i
                    break
            
            if best_match_idx >= 0:
                det = detected_unmatched.pop(best_match_idx)
                variant_match = True
                if expected.variant and det.variant:
                    # Ignore case and spacing
                    if expected.variant.strip().lower() != det.variant.strip().lower():
                        variant_match = False
                
                detected_qty = det.detected_qty
                status = "PASS" if detected_qty == expected.expected_qty and variant_match else "FAIL"
                
                matches.append(MatchResultAI(
                    item_name=expected.name,
                    expected_qty=expected.expected_qty,
                    detected_qty=detected_qty,
                    variant_match=variant_match,
                    status=status
                ))
                
                if detected_qty < expected.expected_qty:
                    missing_items.append(MissingItemAI(
                        item_name=expected.name,
                        expected_qty=expected.expected_qty - detected_qty,
                        reason=f"Detected {detected_qty} but expected {expected.expected_qty}"
                    ))
                elif detected_qty > expected.expected_qty:
                    extra_items.append(ExtraItemAI(
                        item_name=expected.name,
                        qty=detected_qty - expected.expected_qty,
                        notes=f"Detected {detected_qty} but expected {expected.expected_qty}"
                    ))
            else:
                matches.append(MatchResultAI(
                    item_name=expected.name,
                    expected_qty=expected.expected_qty,
                    detected_qty=0,
                    variant_match=False,
                    status="FAIL"
                ))
                missing_items.append(MissingItemAI(
                    item_name=expected.name,
                    expected_qty=expected.expected_qty,
                    reason="Not visible in box"
                ))
                
        for det in detected_unmatched:
            extra_items.append(ExtraItemAI(
                item_name=det.name,
                qty=det.detected_qty,
                notes="Not in order"
            ))
            
        response.matches = matches
        response.missing_items = missing_items
        response.extra_items = extra_items
        
        has_missing = len(missing_items) > 0
        has_extra = len(extra_items) > 0
        has_failed_match = any(m.status == "FAIL" for m in matches)
        has_damage = response.product_condition.visible_damage

        if any(d.confidence == "low" for d in response.detected_items) or response.confidence == "low":
            response.decision = "UNCERTAIN"
            response.decision_reason = "Photo quality insufficient to confidently verify contents - request clearer image."
        elif has_missing or has_extra or has_failed_match or has_damage:
            response.decision = "STOP_FIX"
            reasons = []
            if has_missing: reasons.append(f"Missing {sum(m.expected_qty for m in missing_items)} item(s)")
            if has_extra: reasons.append(f"Extra {sum(e.qty for e in extra_items)} item(s)")
            if has_failed_match and not has_missing and not has_extra: reasons.append("Variant/color mismatch")
            if has_damage: reasons.append("Visible damage detected")
            response.decision_reason = " | ".join(reasons) + "."
        else:
            response.decision = "SEAL"
            response.decision_reason = "All expected items present, correct quantity, correct variant, NO extra items."

        # Return validated response to ensure schema compliance
        return PackManagerAIResponse.model_validate(response.model_dump())

    def as_vision_backend(self) -> AgentVisionBackend:
        """Return a VisionBackend adapter that can plug into existing pipeline.evaluate_pack."""
        return AgentVisionBackend(agent=self)


class AgentVisionBackend:
    """VisionBackend adapter bridging PackManagerAIAgent to pack_manager.pipeline."""

    def __init__(self, agent: PackManagerAIAgent) -> None:
        self.agent = agent
        self.model_version = f"agent-{getattr(agent.vlm, 'provider_name', 'vlm')}-{getattr(agent.vlm, 'model_name', '1.0')}"

    def detect(self, event: PackEvent) -> tuple[list[DetectedItem], int]:
        started = time.perf_counter()
        if not event.photos:
            return [], 0

        photo_path = event.photos[0].uri
        ai_resp = self.agent.verify(event.order, photo_path)
        latency_ms = max(0, int(round((time.perf_counter() - started) * 1000)))

        detections: list[DetectedItem] = []
        # Build catalog SKU mapping by product name / color
        cat_map = {item.product_name.lower(): item.sku for item in event.catalogue}
        for item in event.catalogue:
            cat_map[item.sku.lower()] = item.sku

        for det in ai_resp.detected_items:
            sku = cat_map.get(det.name.lower())
            conf_val = 0.95 if det.confidence == "high" else (0.7 if det.confidence == "medium" else 0.3)
            if det.name == "unidentified_blurred_item" or det.confidence == "low":
                sku = "__uncertain__"

            for i in range(max(1, det.detected_qty)):
                detections.append(
                    DetectedItem(
                        detection_id=str(uuid.uuid4()),
                        sku=sku,
                        label=det.name,
                        quantity=1,
                        confidence=conf_val,
                        image_id=event.photos[0].image_id,
                        reasoning=det.notes or f"Identified by {self.model_version}",
                    )
                )

        return detections, latency_ms


def format_cli_output(response: PackManagerAIResponse) -> str:
    """Render a terminal-friendly summary report."""
    badge = f"[{response.decision}]"

    lines = [
        "=" * 64,
        f" PACK MANAGER AI INSPECTION REPORT: {response.order_id}",
        "=" * 64,
        f"DECISION:        {badge}",
        f"DECISION REASON: {response.decision_reason}",
        f"CONFIDENCE:      {response.confidence.upper()}",
        f"DAMAGE DETECTED: {'YES' if response.product_condition.visible_damage else 'NO'} ({response.product_condition.notes})",
        "",
        "--- DETECTED ITEMS IN BOX ---",
    ]
    for d in response.detected_items:
        var = f" (variant: {d.variant})" if d.variant else ""
        lines.append(f"  * {d.detected_qty}x {d.name}{var} [conf: {d.confidence}]")
        if d.notes:
            lines.append(f"    Notes: {d.notes}")

    lines.append("")
    lines.append("--- ORDER MATCH VERIFICATION ---")
    for m in response.matches:
        stat = "PASS" if m.status == "PASS" else "FAIL"
        lines.append(
            f"  [{stat}] {m.item_name} | Expected: {m.expected_qty} | Detected: {m.detected_qty} | Variant Match: {m.variant_match}"
        )

    if response.missing_items:
        lines.append("")
        lines.append("--- MISSING ITEMS ---")
        for mi in response.missing_items:
            lines.append(f"  [MISSING] {mi.expected_qty}x {mi.item_name} -> {mi.reason}")

    if response.extra_items:
        lines.append("")
        lines.append("--- EXTRA / UNORDERED ITEMS ---")
        for ei in response.extra_items:
            lines.append(f"  [EXTRA] {ei.qty}x {ei.item_name} -> {ei.notes}")

    if response.evidence:
        lines.append("")
        lines.append("--- VISUAL EVIDENCE ---")
        for ev in response.evidence:
            lines.append(f"  + {ev}")

    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack Manager AI Agent: Visual Inspection at Packing Stage")
    parser.add_argument("--order", type=Path, required=True, help="Path to Expected Order JSON file")
    parser.add_argument("--photo", type=Path, required=True, help="Path to Box Photograph (image file)")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "simulation"],
        default=None,
        help="VLM provider to use (default: auto-detected or simulation)",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key for chosen provider")
    parser.add_argument("--model", default=None, help="Model name (e.g. gemini-2.0-flash, gpt-4o)")
    parser.add_argument("--json-out", type=Path, default=None, help="Write raw output JSON to file")
    args = parser.parse_args(argv)

    vlm = get_vlm_client(provider=args.provider, api_key=args.api_key, model=args.model)
    agent = PackManagerAIAgent(vlm_client=vlm)

    print(f"Running Pack Manager AI Agent using [{vlm.provider_name} : {vlm.model_name}]...")
    result = agent.verify(order=args.order, photo=args.photo)
    print(format_cli_output(result))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Inspection JSON written to {args.json_out}")

    return 0 if result.decision == "SEAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
