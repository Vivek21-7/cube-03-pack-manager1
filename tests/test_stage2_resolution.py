"""Tests for STAGE 2: SKU / CATALOGUE RESOLUTION ENGINE

Verifies:
1. unique obvious SKU -> RESOLVED
2. wrong color candidate eliminated via hard conflict
3. two similar SKUs -> AMBIGUOUS (score margin < 0.15)
4. low-confidence Stage 1 observation -> AMBIGUOUS
5. severe occlusion -> AMBIGUOUS
6. product not in catalogue -> UNRESOLVED (unknown_product)
7. foreign object -> UNRESOLVED + preserved foreign_object type
8. reference image breaks metadata tie
9. no reference image still works cleanly without crash
10. Stage 2 interface strictly cannot receive order manifest
"""

import tempfile
from pathlib import Path
import pytest
from PIL import Image

from pack_manager.agent import PackManagerAIAgent
from pack_manager.agent_schemas import (
    BoundingBox2D,
    ContainerObservation,
    OcclusionDetail,
    ObservedPhysicalItem,
    PackageVisualObservation,
    ResolutionStatus,
    Stage2Config,
)
from pack_manager.catalogue_resolver import (
    DEFAULT_AMBIGUITY_THRESHOLD,
    DEFAULT_MAX_OCCLUSION_FOR_RESOLUTION,
    DEFAULT_MIN_OBSERVATION_CONFIDENCE,
    DEFAULT_MIN_RESOLUTION_MARGIN,
    DEFAULT_RESOLUTION_THRESHOLD,
    resolve_catalogue_skus,
)
from pack_manager.models import CatalogItem, ImageRef, Order, OrderLine, OrderStatus
from pack_manager.vlm_client import SimulationVLMClient


@pytest.fixture
def sample_catalogue() -> list[CatalogItem]:
    return [
        CatalogItem(
            sku="CAP-BLU-01",
            product_name="Blue Baseball Cap",
            category="apparel",
            attributes={"color": "blue", "style": "baseball cap"},
        ),
        CatalogItem(
            sku="CAP-NAVY-01",
            product_name="Navy Baseball Cap",
            category="apparel",
            attributes={"color": "navy", "style": "baseball cap"},
        ),
        CatalogItem(
            sku="CAP-RED-01",
            product_name="Red Baseball Cap",
            category="apparel",
            attributes={"color": "red", "style": "baseball cap"},
        ),
        CatalogItem(
            sku="SHIRT-BLK-01",
            product_name="Black Cotton T-Shirt",
            category="apparel",
            attributes={"color": "black", "size": "M"},
        ),
    ]


def test_unique_obvious_sku_resolved(sample_catalogue: list[CatalogItem]) -> None:
    """1. Unique obvious SKU with clear attributes must be definitively RESOLVED."""
    item = ObservedPhysicalItem(
        object_id="obj-1",
        label="Blue Baseball Cap",
        category="product",
        visual_attributes={"color": "blue"},
        confidence=0.95,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, sample_catalogue)
    assert len(res.resolved_objects) == 1
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.RESOLVED
    assert resolved_item.resolved_sku == "CAP-BLU-01"
    assert resolved_item.confidence >= DEFAULT_RESOLUTION_THRESHOLD
    assert any("Observed color 'blue' matches" in ev for ev in resolved_item.matching_evidence)


def test_wrong_color_candidate_eliminated(sample_catalogue: list[CatalogItem]) -> None:
    """2. Wrong color candidate must be eliminated by hard compatibility check."""
    # Only supply Blue Cap in catalogue
    catalogue = [sample_catalogue[0]]  # CAP-BLU-01 (blue)

    # Observed product is clearly Red
    item = ObservedPhysicalItem(
        object_id="obj-red",
        label="Baseball Cap",
        category="product",
        visual_attributes={"color": "red"},
        confidence=0.95,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, catalogue)
    resolved_item = res.resolved_objects[0]

    # CAP-BLU-01 must be eliminated, leaving no candidates
    assert resolved_item.status == ResolutionStatus.UNRESOLVED
    assert resolved_item.resolved_sku is None
    assert len(resolved_item.candidate_skus) == 0
    assert resolved_item.unresolved_reason == "UNKNOWN_PRODUCT"


def test_two_similar_skus_ambiguous(sample_catalogue: list[CatalogItem]) -> None:
    """3. Two similar SKUs with narrow score delta must be AMBIGUOUS (never guess)."""
    # Observed product with ambiguous/dark blue shade
    item = ObservedPhysicalItem(
        object_id="obj-dark-cap",
        label="Baseball Cap",
        category="product",
        visual_attributes={"color": "blue"},
        confidence=0.90,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    # Catalogue only has Blue and Navy caps
    two_caps_catalogue = [sample_catalogue[0], sample_catalogue[1]]
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    # If scores are close (e.g. within 0.15), it must be flagged AMBIGUOUS
    cfg = Stage2Config(min_resolution_margin=0.25)  # Enforce separation
    res = resolve_catalogue_skus(obs, two_caps_catalogue, config=cfg)
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.AMBIGUOUS
    assert resolved_item.resolved_sku is None
    assert resolved_item.unresolved_reason in ("AMBIGUOUS_COMPETITORS", "INSUFFICIENT_CONFIDENCE")
    assert len(resolved_item.candidate_skus) >= 2


def test_low_confidence_observation_ambiguous(sample_catalogue: list[CatalogItem]) -> None:
    """4. Low-confidence Stage 1 observation must result in AMBIGUOUS."""
    item = ObservedPhysicalItem(
        object_id="obj-low-conf",
        label="Blue Baseball Cap",
        category="product",
        visual_attributes={"color": "blue"},
        confidence=0.45,  # Below DEFAULT_MIN_OBSERVATION_CONFIDENCE (0.60)
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, sample_catalogue)
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.AMBIGUOUS
    assert resolved_item.resolved_sku is None
    assert resolved_item.unresolved_reason == "LOW_PERCEPTION_CONFIDENCE"


def test_severe_occlusion_ambiguous(sample_catalogue: list[CatalogItem]) -> None:
    """5. Severe occlusion (> 40%) must force resolution status to AMBIGUOUS."""
    item = ObservedPhysicalItem(
        object_id="obj-occluded",
        label="Blue Baseball Cap",
        category="product",
        visual_attributes={"color": "blue"},
        confidence=0.95,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
        occlusion=OcclusionDetail(
            is_occluded=True,
            occlusion_ratio=0.65,  # Above MAX_OCCLUSION_FOR_RESOLUTION (0.40)
            notes="65% obscured behind carton flap",
        ),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, sample_catalogue)
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.AMBIGUOUS
    assert resolved_item.resolved_sku is None
    assert resolved_item.unresolved_reason == "SEVERE_OCCLUSION"


def test_product_not_in_catalogue_unresolved(sample_catalogue: list[CatalogItem]) -> None:
    """6. Product not present in catalogue must be UNRESOLVED with semantic_type='product'."""
    item = ObservedPhysicalItem(
        object_id="obj-unknown-toy",
        label="Yellow Plastic Duck",
        category="product",
        visual_attributes={"color": "yellow", "material": "plastic"},
        confidence=0.92,
        bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, sample_catalogue)
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.UNRESOLVED
    assert resolved_item.resolved_sku is None
    assert resolved_item.semantic_type == "product"
    assert resolved_item.unresolved_reason == "UNKNOWN_PRODUCT"


def test_foreign_object_unresolved_preserved(sample_catalogue: list[CatalogItem]) -> None:
    """7. Foreign object must be UNRESOLVED and preserve foreign_object semantic type."""
    item = ObservedPhysicalItem(
        object_id="obj-scissors",
        label="Metal Packing Scissors",
        category="foreign_object",
        visual_attributes={"material": "metal"},
        confidence=0.98,
        bbox=BoundingBox2D(ymin=0.6, xmin=0.6, ymax=0.8, xmax=0.8),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

    res = resolve_catalogue_skus(obs, sample_catalogue)
    resolved_item = res.resolved_objects[0]

    assert resolved_item.status == ResolutionStatus.UNRESOLVED
    assert resolved_item.resolved_sku is None
    assert resolved_item.semantic_type == "foreign_object"
    assert resolved_item.unresolved_reason == "FOREIGN_OBJECT"
    assert len(res.foreign_objects) == 1


def test_reference_image_breaks_metadata_tie() -> None:
    """8. Reference image visual comparison breaks tie between two near-duplicate SKUs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 1. Create reference image 1: solid red square
        ref1_path = tmp_path / "ref_red.png"
        img_red = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img_red.save(ref1_path)

        # 2. Create reference image 2: solid blue square
        ref2_path = tmp_path / "ref_blue.png"
        img_blue = Image.new("RGB", (64, 64), color=(0, 0, 255))
        img_blue.save(ref2_path)

        # 3. Create package image: containing a blue box in the crop region
        pkg_img_path = tmp_path / "pkg.png"
        pkg_img = Image.new("RGB", (200, 200), color=(200, 200, 200))
        # Draw blue region at (20, 20, 80, 80) -> normalized (0.1, 0.1, 0.4, 0.4)
        for x in range(20, 80):
            for y in range(20, 80):
                pkg_img.putpixel((x, y), (0, 0, 255))
        pkg_img.save(pkg_img_path)

        cat_red = CatalogItem(
            sku="ITEM-RED",
            product_name="Generic Widget Red",
            category="hardware",
            attributes={"color": "red"},
            reference_images=[ImageRef(image_id="ref1", uri=str(ref1_path), role="reference")],
        )
        cat_blue = CatalogItem(
            sku="ITEM-BLUE",
            product_name="Generic Widget Blue",
            category="hardware",
            attributes={"color": "blue"},
            reference_images=[ImageRef(image_id="ref2", uri=str(ref2_path), role="reference")],
        )

        item = ObservedPhysicalItem(
            object_id="obj-widget",
            label="Generic Widget",
            category="product",
            visual_attributes={},  # No color in metadata, causing potential tie
            confidence=0.90,
            bbox=BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4),
        )
        obs = PackageVisualObservation(image_id="img-1", observed_items=[item])

        res = resolve_catalogue_skus(
            obs,
            [cat_red, cat_blue],
            package_image=pkg_img,
        )
        resolved_item = res.resolved_objects[0]

        # The blue reference image matches the package crop, so ITEM-BLUE should rank higher
        assert len(resolved_item.candidate_skus) >= 1
        top_sku = resolved_item.candidate_skus[0].sku
        assert top_sku == "ITEM-BLUE"
        assert any("Reference image visual comparison" in ev for ev in resolved_item.matching_evidence)


def test_no_reference_image_still_works(sample_catalogue: list[CatalogItem]) -> None:
    """9. Absence of reference images falls back to metadata cleanly without crashing."""
    # Ensure catalogue has empty reference_images
    for item in sample_catalogue:
        item.reference_images = []

    obs_item = ObservedPhysicalItem(
        object_id="obj-shirt",
        label="Black T-Shirt M",
        category="product",
        visual_attributes={"color": "black", "size": "M"},
        confidence=0.95,
        bbox=BoundingBox2D(ymin=0.2, xmin=0.2, ymax=0.7, xmax=0.7),
    )
    obs = PackageVisualObservation(image_id="img-1", observed_items=[obs_item])

    res = resolve_catalogue_skus(obs, sample_catalogue, package_image=None)
    resolved = res.resolved_objects[0]

    assert resolved.status == ResolutionStatus.RESOLVED
    assert resolved.resolved_sku == "SHIRT-BLK-01"


def test_stage2_interface_cannot_receive_order_manifest(sample_catalogue: list[CatalogItem]) -> None:
    """10. Stage 2 interface strictly cannot receive Order, OrderLine, or expected quantities."""
    obs = PackageVisualObservation(image_id="img-1", observed_items=[])
    order = Order(
        order_id="ord-1",
        organization_id="org-1",
        client_id="client-1",
        lines=[OrderLine(line_id="l1", sku="CAP-BLU-01", expected_qty=1, product_name="Cap")],
        status=OrderStatus.READY_TO_PACK,
    )

    # 1. Test kwargs rejection
    with pytest.raises(TypeError, match="strictly order-blind"):
        resolve_catalogue_skus(obs, sample_catalogue, order=order)  # type: ignore

    # 2. Test passing Order as observation argument
    with pytest.raises(TypeError, match="cannot accept 'Order'"):
        resolve_catalogue_skus(order, sample_catalogue)  # type: ignore

    # 3. Test agent.resolve_skus rejection
    agent = PackManagerAIAgent(vlm_client=SimulationVLMClient())
    with pytest.raises(TypeError, match="strictly order-blind"):
        agent.resolve_skus(obs, sample_catalogue, order=order)  # type: ignore
