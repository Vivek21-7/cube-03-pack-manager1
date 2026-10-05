"""Minimal demo: one correct-order pack → comparison + evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pack_manager.fixtures_draw import write_correct_order_images
from pack_manager.io import load_pack_event
from pack_manager.pipeline import evaluate_pack

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "correct_order"


def _print_table(result) -> None:
    print("=== Pack Manager demo (correct order) ===")
    print(f"order:     {result.evidence.subject.order_id}")
    print(f"decision:  {result.comparison.decision.value}")
    print(f"status:    {result.evidence.status.value}")
    print()
    print("expected vs observed")
    print(f"{'sku':<16}{'expected':>10}{'observed':>10}{'delta':>8}")
    for row in result.comparison.quantity_table:
        obs = "-" if row.observed_qty is None else str(row.observed_qty)
        delta = "-" if row.delta is None else str(row.delta)
        print(f"{row.sku:<16}{row.expected_qty:>10}{obs:>10}{delta:>8}")
    print()
    print("checks")
    for check in result.evidence.checks:
        print(
            f"  {check.check_key.value:<24} {check.verdict.value:<10} "
            f"conf={check.confidence:.2f}  {check.latency_ms}ms"
        )
        print(f"    {check.detail}")
    print()
    print(f"content_hash: {result.evidence.content_hash}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pack Manager on the correct-order fixture.")
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Regenerate synthetic reference and pack PNGs before evaluating.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the evidence record as JSON.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the PackManagerAIAgent VLM backend instead of the heuristic blob backend.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "simulation"],
        default=None,
        help="VLM provider if --agent is used.",
    )
    args = parser.parse_args(argv)

    if args.write_images or not (FIXTURES / "photos" / "pack_open.png").exists():
        write_correct_order_images(FIXTURES)

    event = load_pack_event(
        order_path=FIXTURES / "order.json",
        catalogue_path=FIXTURES / "catalogue.json",
        photos_path=FIXTURES / "photos.json",
    )

    vision_backend = None
    if args.agent:
        from pack_manager.agent import PackManagerAIAgent
        from pack_manager.vlm_client import get_vlm_client

        agent = PackManagerAIAgent(vlm_client=get_vlm_client(provider=args.provider))
        vision_backend = agent.as_vision_backend()

    result = evaluate_pack(event, vision=vision_backend)
    _print_table(result)
    payload = {
        "comparison": result.comparison.model_dump(mode="json"),
        "evidence": result.evidence.model_dump(mode="json"),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if result.comparison.decision.value == "SEAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
