"""STAGE 3: DETERMINISTIC MANIFEST RECONCILIATION ENGINE

Reconciles Stage 2 visual SKU resolutions against the expected customer order manifest.
Uses PURE PYTHON logic only (NO LLMs, NO re-perception, NO image access).

Answers:
"What is the exact difference between what was ordered and what was confidently observed?"

Does NOT decide final SEAL / STOP & FIX / UNCERTAIN (belongs to Stage 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pack_manager.agent_schemas import (
    CandidateSKU,
    ManifestReconciliation,
    PackageSKUResolution,
    ReconciledLine,
    ResolutionStatus,
    ResolvedObjectItem,
    UnverifiedLine,
)
from pack_manager.models import Order, OrderLine


def _normalize_order_manifest(
    order: Order | dict[str, Any] | Path | str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Extract order_id and aggregated expected quantities by SKU.
    Handles duplicate order lines for the same SKU by summing expected quantities.
    """
    order_id = "unknown-order"
    raw_lines: list[dict[str, Any]] = []

    if isinstance(order, Order):
        order_id = order.order_id
        for l in order.lines:
            raw_lines.append({
                "sku": l.sku.strip(),
                "expected_qty": l.expected_qty,
                "product_name": l.product_name,
            })
    elif isinstance(order, (str, Path)):
        p = Path(order)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            order_id = data.get("order_id", p.stem)
            raw_lines = data.get("lines") or data.get("items") or []
        else:
            raise FileNotFoundError(f"Order file not found: {order}")
    elif isinstance(order, dict):
        order_id = order.get("order_id", "unknown-order")
        raw_lines = order.get("lines") or order.get("items") or []
    else:
        raise TypeError(f"Unsupported order manifest type: {type(order)}")

    # Aggregate by SKU (combining duplicate order lines)
    aggregated: dict[str, dict[str, Any]] = {}
    for item in raw_lines:
        sku = str(item.get("sku") or item.get("name", "")).strip()
        if not sku:
            continue
        qty = int(item.get("expected_qty", item.get("quantity", 1)))
        prod_name = item.get("product_name") or item.get("name", sku)

        if sku in aggregated:
            aggregated[sku]["expected_qty"] += qty
        else:
            aggregated[sku] = {
                "sku": sku,
                "expected_qty": qty,
                "product_name": prod_name,
            }

    return order_id, aggregated


def reconcile_manifest(
    order: Order | dict[str, Any] | Path | str,
    resolution: PackageSKUResolution,
) -> ManifestReconciliation:
    """STAGE 3: Pure Python deterministic manifest reconciliation.

    Compares expected order SKU counts against Stage 2 resolved observations.
    Categorizes results into:
    - MATCHED: Exactly observed as ordered
    - MISSING: Expected SKU absent with NO ambiguous candidates
    - EXTRA: Confirmed observed SKU not on order manifest
    - QUANTITY_MISMATCH: Count deficit/surplus with NO ambiguous candidates
    - UNVERIFIED: Expected SKU shortfall where ambiguous candidates exist (avoids false missing)
    - UNRESOLVED_PRODUCTS: Preserved uncatalogued visible products
    - FOREIGN_OBJECTS: Preserved foreign objects (tools, phone, trash)
    """
    order_id, expected_by_sku = _normalize_order_manifest(order)

    # 1. Aggregate Confidently Resolved Stage 2 Items
    # Key: resolved_sku -> {"qty": count, "object_ids": [...], "product_name": ...}
    resolved_by_sku: dict[str, dict[str, Any]] = {}

    for item in resolution.resolved_objects:
        if item.status == ResolutionStatus.RESOLVED and item.resolved_sku:
            sku = item.resolved_sku
            if sku not in resolved_by_sku:
                resolved_by_sku[sku] = {
                    "qty": 0,
                    "object_ids": [],
                    "product_name": item.candidate_skus[0].product_name if item.candidate_skus else sku,
                }
            resolved_by_sku[sku]["qty"] += 1
            resolved_by_sku[sku]["object_ids"].append(item.object_id)

    # 2. Index Ambiguous Objects by Candidate SKUs
    # candidate_sku -> list of (ResolvedObjectItem, CandidateSKU)
    ambiguous_by_candidate_sku: dict[str, list[tuple[ResolvedObjectItem, CandidateSKU]]] = {}
    ambiguous_items = [o for o in resolution.resolved_objects if o.status == ResolutionStatus.AMBIGUOUS]

    for item in ambiguous_items:
        for cand in item.candidate_skus:
            cand_sku = cand.sku
            if cand_sku not in ambiguous_by_candidate_sku:
                ambiguous_by_candidate_sku[cand_sku] = []
            ambiguous_by_candidate_sku[cand_sku].append((item, cand))

    # 3. Preserve Foreign Objects and Unresolved Products
    foreign_objects = [o for o in resolution.resolved_objects if o.semantic_type == "foreign_object"]
    unresolved_products = [
        o for o in resolution.resolved_objects
        if o.semantic_type == "product" and o.status == ResolutionStatus.UNRESOLVED
    ]

    matched: list[ReconciledLine] = []
    missing: list[ReconciledLine] = []
    extra: list[ReconciledLine] = []
    quantity_mismatches: list[ReconciledLine] = []
    unverified: list[UnverifiedLine] = []

    # 4. Reconcile Every Expected SKU
    for sku, exp_info in expected_by_sku.items():
        exp_qty = exp_info["expected_qty"]
        prod_name = exp_info["product_name"]

        res_info = resolved_by_sku.get(sku)
        obs_qty = res_info["qty"] if res_info else 0
        obj_ids = res_info["object_ids"] if res_info else []

        # Find any ambiguous objects that could plausibly be this SKU
        amb_matches = ambiguous_by_candidate_sku.get(sku, [])

        if obs_qty == exp_qty:
            # Exact Match
            matched.append(ReconciledLine(
                sku=sku,
                product_name=prod_name,
                expected_qty=exp_qty,
                observed_qty=obs_qty,
                delta=0,
                status="MATCHED",
                object_ids=obj_ids,
                notes=f"All {exp_qty} unit(s) verified",
            ))

        elif obs_qty > exp_qty:
            # Over-pack: observed count exceeds expected count
            quantity_mismatches.append(ReconciledLine(
                sku=sku,
                product_name=prod_name,
                expected_qty=exp_qty,
                observed_qty=obs_qty,
                delta=obs_qty - exp_qty,
                status="QUANTITY_MISMATCH",
                object_ids=obj_ids,
                notes=f"Over-pack: Expected {exp_qty}, observed {obs_qty} (+{obs_qty - exp_qty})",
            ))

        else:
            # Shortage: obs_qty < exp_qty
            shortage = exp_qty - obs_qty

            if amb_matches:
                # Ambiguous observation exists that could account for the shortfall!
                # Do NOT classify as confirmed MISSING or confirmed QUANTITY_MISMATCH!
                possible_obj_ids = list({item.object_id for item, _ in amb_matches})
                all_candidate_skus: list[CandidateSKU] = []
                seen_skus = set()
                for item, _ in amb_matches:
                    for c in item.candidate_skus:
                        if c.sku not in seen_skus:
                            seen_skus.add(c.sku)
                            all_candidate_skus.append(c)

                evidence_lines = []
                for item, _ in amb_matches:
                    evidence_lines.extend(item.matching_evidence)

                unverified.append(UnverifiedLine(
                    expected_sku=sku,
                    product_name=prod_name,
                    expected_qty=exp_qty,
                    observed_resolved_qty=obs_qty,
                    unverified_qty=min(shortage, len(possible_obj_ids)),
                    possible_object_ids=possible_obj_ids,
                    candidate_skus=all_candidate_skus,
                    reason=(
                        f"Shortfall of {shortage} unit(s) for {sku} cannot be confirmed missing: "
                        f"ambiguous observation(s) {possible_obj_ids} list {sku} as a plausible candidate."
                    ),
                    evidence=list(dict.fromkeys(evidence_lines)),
                ))

                # If there were partial resolved units (e.g. ordered 2, resolved 1, 1 ambiguous),
                # record the resolved unit as partially matched
                if obs_qty > 0:
                    matched.append(ReconciledLine(
                        sku=sku,
                        product_name=prod_name,
                        expected_qty=obs_qty,
                        observed_qty=obs_qty,
                        delta=0,
                        status="MATCHED",
                        object_ids=obj_ids,
                        notes=f"{obs_qty} of {exp_qty} unit(s) confirmed; remaining {shortage} unit(s) unverified",
                    ))

            else:
                # No ambiguous objects could be this SKU: Confirmed Failure
                if obs_qty == 0:
                    missing.append(ReconciledLine(
                        sku=sku,
                        product_name=prod_name,
                        expected_qty=exp_qty,
                        observed_qty=0,
                        delta=-exp_qty,
                        status="MISSING",
                        object_ids=[],
                        notes=f"Confirmed missing: Expected {exp_qty} unit(s), 0 observed",
                    ))
                else:
                    quantity_mismatches.append(ReconciledLine(
                        sku=sku,
                        product_name=prod_name,
                        expected_qty=exp_qty,
                        observed_qty=obs_qty,
                        delta=obs_qty - exp_qty,
                        status="QUANTITY_MISMATCH",
                        object_ids=obj_ids,
                        notes=f"Under-pack: Expected {exp_qty}, observed {obs_qty} (-{shortage})",
                    ))

    # 5. Check for Extra Observed Resolved SKUs (Not on Expected Manifest)
    for sku, res_info in resolved_by_sku.items():
        if sku not in expected_by_sku:
            extra.append(ReconciledLine(
                sku=sku,
                product_name=res_info["product_name"],
                expected_qty=0,
                observed_qty=res_info["qty"],
                delta=res_info["qty"],
                status="EXTRA",
                object_ids=res_info["object_ids"],
                notes=f"Extra item: Observed {res_info['qty']} unit(s) of unexpected SKU {sku}",
            ))

    # 6. Check for Unclaimed Ambiguous Objects (Must not be silently dropped)
    claimed_ambiguous_ids = set()
    for u in unverified:
        claimed_ambiguous_ids.update(u.possible_object_ids)

    for amb in ambiguous_items:
        if amb.object_id not in claimed_ambiguous_ids:
            amb_name = amb.candidate_skus[0].product_name if amb.candidate_skus else "Ambiguous package item"
            unverified.append(UnverifiedLine(
                expected_sku="UNEXPECTED_AMBIGUOUS",
                product_name=amb_name,
                expected_qty=0,
                observed_resolved_qty=0,
                unverified_qty=1,
                possible_object_ids=[amb.object_id],
                candidate_skus=amb.candidate_skus,
                reason=(
                    f"Unexpected ambiguous object '{amb.object_id}' in carton "
                    f"does not match any ordered SKU."
                ),
                evidence=amb.matching_evidence,
            ))

    return ManifestReconciliation(
        order_id=order_id,
        matched=matched,
        missing=missing,
        extra=extra,
        quantity_mismatches=quantity_mismatches,
        unverified=unverified,
        unresolved_products=unresolved_products,
        foreign_objects=foreign_objects,
    )
