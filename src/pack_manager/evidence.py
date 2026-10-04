"""Evidence record hashing and JSON Schema (contract v1.0.0)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pack_manager.models import EvidenceRecord

SCHEMA_PATH_RELATIVE = "schemas/evidence_record.v1.json"


def canonical_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_content_hash(record: EvidenceRecord) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("content_hash", None)
    digest = hashlib.sha256(canonical_dump(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def seal_hash(record: EvidenceRecord) -> EvidenceRecord:
    return record.model_copy(update={"content_hash": compute_content_hash(record)})
