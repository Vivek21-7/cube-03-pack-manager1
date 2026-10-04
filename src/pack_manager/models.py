"""Canonical data models for Pack Manager inputs and intermediate outputs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from pack_manager.assumptions import SCHEMA_VERSION


class ImageView(str, Enum):
    TOP = "top"
    SIDE = "side"
    DETAIL = "detail"
    UNKNOWN = "unknown"


class ImageRef(BaseModel):
    image_id: str
    uri: str
    sha256: Optional[str] = None
    role: Literal["evidence", "reference"] = "evidence"
    view: ImageView = ImageView.UNKNOWN


class CatalogItem(BaseModel):
    sku: str
    asin: Optional[str] = None
    product_name: str
    category: str
    attributes: dict[str, str] = Field(default_factory=dict)
    reference_images: list[ImageRef] = Field(default_factory=list)
    barcode: Optional[str] = None

    @field_validator("sku")
    @classmethod
    def sku_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sku must be non-empty")
        return value.strip()


class OrderLine(BaseModel):
    line_id: str
    sku: str
    expected_qty: int = Field(ge=1)
    product_name: str
    asin: Optional[str] = None


class OrderStatus(str, Enum):
    READY_TO_PACK = "ready_to_pack"


class Order(BaseModel):
    order_id: str
    organization_id: str
    client_id: str
    lines: list[OrderLine]
    package_id: Optional[str] = None
    status: OrderStatus = OrderStatus.READY_TO_PACK

    @field_validator("lines")
    @classmethod
    def at_least_one_line(cls, value: list[OrderLine]) -> list[OrderLine]:
        if not value:
            raise ValueError("order must contain at least one line")
        return value


class PackPhoto(BaseModel):
    image_id: str
    uri: str
    captured_at: datetime
    operator_label: str
    view: ImageView = ImageView.UNKNOWN
    sha256: Optional[str] = None


class PackEvent(BaseModel):
    organization_id: str
    client_id: str
    order: Order
    catalogue: list[CatalogItem]
    photos: list[PackPhoto]
    operator_label: str

    @field_validator("photos")
    @classmethod
    def at_least_one_photo(cls, value: list[PackPhoto]) -> list[PackPhoto]:
        if not value:
            raise ValueError("pack event requires at least one photograph")
        return value

    @field_validator("catalogue")
    @classmethod
    def at_least_one_catalog_item(cls, value: list[CatalogItem]) -> list[CatalogItem]:
        if not value:
            raise ValueError("pack event requires a non-empty catalogue")
        return value


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectedItem(BaseModel):
    detection_id: str
    label: str
    quantity: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    image_id: str
    sku: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    reasoning: Optional[str] = None


class QuantityRow(BaseModel):
    sku: str
    product_name: str
    expected_qty: int
    observed_qty: Optional[int] = None
    delta: Optional[int] = None


class ItemMismatch(BaseModel):
    sku: Optional[str] = None
    product_name: str
    quantity: int
    reason: str
    related_detection_ids: list[str] = Field(default_factory=list)


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class Decision(str, Enum):
    SEAL = "SEAL"
    STOP_AND_FIX = "STOP_AND_FIX"


class EvidenceStatus(str, Enum):
    COMPLETE = "complete"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class CheckKey(str, Enum):
    ITEM_PRESENCE = "item_presence"
    QUANTITY_MATCH = "quantity_match"
    ITEM_IDENTITY = "item_identity"
    EXTRA_ITEM_DETECTION = "extra_item_detection"
    MISSING_ITEM_DETECTION = "missing_item_detection"
    DECISION_SYNTHESIS = "decision_synthesis"


REQUIRED_CHECK_KEYS = (
    CheckKey.ITEM_PRESENCE,
    CheckKey.QUANTITY_MATCH,
    CheckKey.ITEM_IDENTITY,
    CheckKey.EXTRA_ITEM_DETECTION,
    CheckKey.MISSING_ITEM_DETECTION,
    CheckKey.DECISION_SYNTHESIS,
)


class CheckResult(BaseModel):
    check_key: CheckKey
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    detail: str
    model_version: str
    latency_ms: int = Field(ge=0)


class Subject(BaseModel):
    order_id: str
    package_id: Optional[str] = None


class Outcome(BaseModel):
    decision: Decision
    decided_by: str
    decided_at: datetime


class Override(BaseModel):
    actor: str
    reason: str
    from_decision: Decision
    to_decision: Decision
    at: datetime


class EvidenceRecord(BaseModel):
    record_id: str
    schema_version: str = SCHEMA_VERSION
    organization_id: str
    client_id: str
    agent: str
    subject: Subject
    captured_at: datetime
    operator_label: str
    images: list[ImageRef]
    checks: list[CheckResult]
    outcome: Outcome
    overrides: list[Override] = Field(default_factory=list)
    status: EvidenceStatus
    content_hash: str


class PackComparison(BaseModel):
    detected_items: list[DetectedItem]
    expected_items: list[OrderLine]
    quantity_table: list[QuantityRow]
    missing: list[ItemMismatch]
    wrong: list[ItemMismatch]
    extra: list[ItemMismatch]
    decision: Decision


class PackResult(BaseModel):
    comparison: PackComparison
    evidence: EvidenceRecord
