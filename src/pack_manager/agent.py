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
    DecisionResult,
    PackVerificationReport,
    ManifestReconciliation,
    PackageSKUResolution,
    Stage2Config,
    PackageVisualObservation,
    ObservedPhysicalItem,
    BoundingBox2D,
    OrderItemInput,
    PackManagerAIResponse,
    MatchResultAI,
    MissingItemAI,
    ExtraItemAI,
)
from pack_manager.models import CatalogItem, BoundingBox, DetectedItem, Order, PackEvent
from pack_manager.prompts import build_user_prompt, build_observation_user_prompt
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
                brand = it.get("brand")
                if not variant and "attributes" in it and isinstance(it["attributes"], dict):
                    variant = it["attributes"].get("color")
                sku = it.get("sku")
                items.append(
                    {
                        "name": name,
                        "brand": brand,
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
        mime = "image/png"
        try:
            fmt = Image.open(io.BytesIO(photo)).format
            if fmt == "JPEG":
                mime = "image/jpeg"
            elif fmt == "WEBP":
                mime = "image/webp"
            elif fmt == "PNG":
                mime = "image/png"
        except Exception:
            pass
        return photo, mime

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

    def observe_package(
        self,
        photo: Path | str | bytes | Image.Image,
        image_id: str = "box-photo-0",
        max_retries: int = 1,
    ) -> PackageVisualObservation:
        """STAGE 1: Perform visual observation on package photograph without order knowledge.
        Strict safety: If output is malformed or bounding boxes are rejected, retries once
        without guessing or fabricating spatial data.
        """
        image_bytes, mime_type = _load_image_bytes(photo)
        prompt = build_observation_user_prompt(image_id)

        last_error: Exception | None = None
        raw_output = ""
        for attempt in range(max_retries + 1):
            raw_output = self.vlm.observe(prompt, image_bytes, mime_type=mime_type)
            cleaned_json = _clean_json_markdown(raw_output)

            try:
                parsed = json.loads(cleaned_json)
                return PackageVisualObservation.model_validate(parsed)
            except Exception as exc:
                last_error = exc
                logger.warning("Stage 1 observation attempt %d failed: %s", attempt + 1, exc)
                if attempt < max_retries:
                    continue

        logger.error("VLM observation failed after %d attempt(s):\n%s", max_retries + 1, raw_output)
        raise ValueError(f"Failed to obtain valid visual observation: {last_error}") from last_error


    def resolve_skus(
        self,
        observation: PackageVisualObservation,
        catalogue: list[CatalogItem] | Path | str | dict[str, Any] | None = None,
        *,
        config: Optional[Stage2Config] = None,
        package_image: Optional[Path | str | bytes | Image.Image] = None,
        **kwargs: Any,
    ) -> PackageSKUResolution:
        """STAGE 2: Order-Blind SKU / Catalogue Resolution.

        Determines which catalogue SKU each visible product in the Stage 1 observation
        most likely corresponds to. Strictly order-blind: Does NOT receive customer order
        or expected quantities.
        """
        # Guard: Strict rejection of order manifests
        if "order" in kwargs or "order_lines" in kwargs or "expected" in kwargs:
            raise TypeError("Stage 2 is strictly order-blind and must NOT receive order manifests or expected quantities.")

        for arg in [observation, catalogue]:
            arg_type = type(arg).__name__
            if "Order" in arg_type or "OrderLine" in arg_type:
                raise TypeError(f"Stage 2 cannot accept '{arg_type}'. Only PackageVisualObservation and catalogue are permitted.")

        # Ingest catalogue
        cat_items: list[CatalogItem] = []
        if catalogue is None:
            from pack_manager.catalogue_resolver import DEMO_FIXTURE_CATALOGUE
            cat_items = DEMO_FIXTURE_CATALOGUE
        elif isinstance(catalogue, list):
            cat_items = [
                CatalogItem.model_validate(it) if isinstance(it, dict) else it
                for it in catalogue
            ]
        elif isinstance(catalogue, (str, Path)):
            p = Path(catalogue)
            if p.is_file():
                import json
                raw_cat = json.loads(p.read_text(encoding="utf-8"))
                items_data = raw_cat.get("items", raw_cat) if isinstance(raw_cat, dict) else raw_cat
                cat_items = [CatalogItem.model_validate(it) for it in items_data]
            else:
                raise FileNotFoundError(f"Catalogue file not found: {catalogue}")
        elif isinstance(catalogue, dict):
            items_data = catalogue.get("items", catalogue)
            cat_items = [CatalogItem.model_validate(it) for it in items_data]
        else:
            raise TypeError(f"Unsupported catalogue input type: {type(catalogue)}")

        from pack_manager.catalogue_resolver import resolve_catalogue_skus
        return resolve_catalogue_skus(
            observation=observation,
            catalogue=cat_items,
            config=config,
            package_image=package_image,
        )


    def reconcile_manifest(
        self,
        order: dict[str, Any] | Order | Path | str,
        resolution: PackageSKUResolution,
    ) -> ManifestReconciliation:
        """STAGE 3: Deterministic Manifest Reconciliation.

        Compares expected order manifest against Stage 2 SKU resolution using pure Python logic.
        Produces structured categories: matched, missing, extra, quantity mismatches, unverified items,
        preserving unresolved products and foreign objects. Strictly does NOT make shipping decisions.
        """
        from pack_manager.manifest_reconciler import reconcile_manifest
        return reconcile_manifest(order=order, resolution=resolution)


    def decide(
        self,
        reconciliation: ManifestReconciliation,
        observation: Optional[PackageVisualObservation] = None,
    ) -> DecisionResult:
        """STAGE 4: Final Deterministic Packing Decision Engine.

        Evaluates Stage 3 manifest reconciliation and optional Stage 1 visual observation.
        Strict precedence: UNCERTAIN > STOP_FIX > SEAL.
        """
        from pack_manager.decision_engine import evaluate_decision
        return evaluate_decision(reconciliation=reconciliation, observation=observation)


    def create_verification_report(
        self,
        reconciliation: ManifestReconciliation,
        decision: DecisionResult,
        observation: PackageVisualObservation,
        resolution: PackageSKUResolution,
        order: Optional[Order | dict[str, Any] | Path | str] = None,
        *,
        image_bytes: Optional[bytes] = None,
        image_sha256: Optional[str] = None,
    ) -> PackVerificationReport:
        """STAGE 5: Final Grounded Evidence & Audit Trail Packaging.

        Packages Stages 1-4 into an immutable, audit-grade verification report.
        Strictly preserves decision, links every observed item to bounding boxes,
        and provides grounded operator action.
        """
        from pack_manager.grounded_audit import build_verification_report
        return build_verification_report(
            reconciliation=reconciliation,
            decision=decision,
            observation=observation,
            resolution=resolution,
            order=order,
            image_bytes=image_bytes,
            image_sha256=image_sha256,
            model_name=getattr(self.vlm, "model_name", None),
        )


    def verify_full(
        self,
        order: dict[str, Any] | Order | Path | str,
        catalogue: list[CatalogItem] | Path | str | dict[str, Any] | None = None,
        photo: Path | str | bytes | Image.Image = None,
        image_id: str = "box-photo-0",
    ) -> PackVerificationReport:
        """END-TO-END VERIFICATION PIPELINE (Stages 1-5).

        Connects all 5 deterministic and perceptual stages:
        1. observe_package(photo) -> PackageVisualObservation (order-blind perception)
        2. resolve_skus(observation, catalogue) -> PackageSKUResolution (order-blind SKU mapping)
        3. reconcile_manifest(order, resolution) -> ManifestReconciliation (deterministic Python comparison)
        4. decide(reconciliation, observation) -> DecisionResult (SEAL / STOP_FIX / UNCERTAIN)
        5. create_verification_report(...) -> PackVerificationReport (traceable grounded evidence)
        """
        if photo is None:
            raise ValueError("Photo input is required for verification.")

        image_bytes, _ = _load_image_bytes(photo)

        cat_items: list[CatalogItem] = []
        if catalogue is None:
            from pack_manager.catalogue_resolver import DEMO_FIXTURE_CATALOGUE
            cat_items = DEMO_FIXTURE_CATALOGUE
        elif isinstance(catalogue, list):
            cat_items = [
                CatalogItem.model_validate(it) if isinstance(it, dict) else it
                for it in catalogue
            ]
        elif isinstance(catalogue, (str, Path)):
            p = Path(catalogue)
            if p.is_file():
                raw_cat = json.loads(p.read_text(encoding="utf-8"))
                items_data = raw_cat.get("items", raw_cat) if isinstance(raw_cat, dict) else raw_cat
                cat_items = [CatalogItem.model_validate(it) for it in items_data]
            else:
                from pack_manager.catalogue_resolver import DEFAULT_WAREHOUSE_CATALOGUE
                cat_items = DEFAULT_WAREHOUSE_CATALOGUE
        elif isinstance(catalogue, dict):
            items_data = catalogue.get("items", catalogue)
            cat_items = [CatalogItem.model_validate(it) for it in items_data]
        else:
            cat_items = list(catalogue)

        # STAGE 1: Visual Perception (Order-Blind)
        observation = self.observe_package(photo=photo, image_id=image_id)

        # STAGE 2: SKU / Catalogue Resolution (Order-Blind)
        resolution = self.resolve_skus(
            observation=observation,
            catalogue=cat_items,
            package_image=photo,
        )

        # STAGE 3: Deterministic Manifest Reconciliation (Pure Python)
        reconciliation = self.reconcile_manifest(
            order=order,
            resolution=resolution,
        )

        # STAGE 4: Final Decision Engine (Precedence Gating)
        decision = self.decide(
            reconciliation=reconciliation,
            observation=observation,
        )

        # STAGE 5: Grounded Evidence & Audit Trail
        report = self.create_verification_report(
            reconciliation=reconciliation,
            decision=decision,
            observation=observation,
            resolution=resolution,
            order=order,
            image_bytes=image_bytes,
        )

        return report

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
                
                brand_match = True
                if expected.brand and det.brand:
                    if expected.brand.strip().lower() != det.brand.strip().lower() and det.brand.lower() != "unknown":
                        brand_match = False
                elif expected.brand and det.brand == "unknown":
                    pass # Handled by confidence or notes usually, but we'll let it pass identity and let the user review
                
                detected_qty = det.detected_qty
                status = "PASS" if detected_qty == expected.expected_qty and variant_match and brand_match else "FAIL"
                
                matches.append(MatchResultAI(
                    item_name=expected.name,
                    expected_qty=expected.expected_qty,
                    detected_qty=detected_qty,
                    variant_match=variant_match,
                    brand_match=brand_match,
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

        if any(d.confidence == "low" for d in response.detected_items) or response.confidence == "low" or any(d.brand == "Unknown" for d in response.detected_items):
            response.decision = "UNCERTAIN"
            response.decision_reason = "Photo quality insufficient or brand identity unclear - request clearer image."
        elif has_missing or has_extra or has_failed_match or has_damage:
            response.decision = "STOP_FIX"
            reasons = []
            if has_missing: reasons.append(f"Missing {sum(m.expected_qty for m in missing_items)} item(s)")
            if has_extra: reasons.append(f"Extra {sum(e.qty for e in extra_items)} item(s)")
            if has_failed_match and not has_missing and not has_extra:
                failed = [m for m in matches if m.status == "FAIL"]
                if any(not m.brand_match for m in failed):
                    reasons.append("Brand/product mismatch")
                elif any(not m.variant_match for m in failed):
                    reasons.append("Variant/color mismatch")
                else:
                    reasons.append("Quantity mismatch")
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
