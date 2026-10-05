"""Pydantic data schemas strictly matching the Pack Manager AI prompt specification."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


ConfidenceLevel = Literal["high", "medium", "low"]
DecisionType = Literal["SEAL", "STOP_FIX", "UNCERTAIN"]
MatchStatusType = Literal["PASS", "FAIL"]


class OrderItemInput(BaseModel):
    name: str
    expected_qty: int = Field(ge=0)
    variant: Optional[str] = None
    sku: Optional[str] = None


class DetectedItemAI(BaseModel):
    name: str
    detected_qty: int = Field(ge=0)
    variant: Optional[str] = None
    confidence: ConfidenceLevel = "high"
    notes: Optional[str] = ""


class MatchResultAI(BaseModel):
    item_name: str
    expected_qty: int = Field(ge=0)
    detected_qty: int = Field(ge=0)
    variant_match: bool = True
    status: MatchStatusType = "PASS"


class MissingItemAI(BaseModel):
    item_name: str
    expected_qty: int = Field(ge=1)
    reason: str = "Not visible in box"


class ExtraItemAI(BaseModel):
    item_name: str
    qty: int = Field(ge=1)
    notes: str = "Not in order"


class ProductConditionAI(BaseModel):
    visible_damage: bool = False
    notes: str = "No visible damage to products or packaging"


class PackManagerAIResponse(BaseModel):
    """The canonical output payload emitted by the Pack Manager AI Agent."""

    order_id: str
    order_items: list[OrderItemInput] = Field(default_factory=list)
    detected_items: list[DetectedItemAI] = Field(default_factory=list)
    matches: list[MatchResultAI] = Field(default_factory=list)
    missing_items: list[MissingItemAI] = Field(default_factory=list)
    extra_items: list[ExtraItemAI] = Field(default_factory=list)
    product_condition: ProductConditionAI = Field(default_factory=ProductConditionAI)
    decision: DecisionType
    decision_reason: str
    confidence: ConfidenceLevel
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> PackManagerAIResponse:
        """Enforce strict safety rules: SEAL is prohibited if defects or ambiguities exist."""
        has_missing = len(self.missing_items) > 0
        has_extra = len(self.extra_items) > 0
        has_failed_match = any(m.status == "FAIL" for m in self.matches)
        has_damage = self.product_condition.visible_damage

        if self.decision == "SEAL":
            if has_missing or has_extra or has_failed_match or has_damage:
                # Disallow SEAL when defects are detected
                self.decision = "STOP_FIX"
                if not self.decision_reason:
                    self.decision_reason = "Defects detected during package verification."
            elif self.confidence == "low":
                self.decision = "UNCERTAIN"
                if not self.decision_reason:
                    self.decision_reason = "Inspection confidence is low; manual review required."
        return self

    def to_summary_dict(self) -> dict[str, Any]:
        """Convenience dictionary representation."""
        return self.model_dump(mode="json")
