"""Locked defaults after schema confirmation (2026-10-04).

These are explicit so catalogue/order format is not invented silently later.
"""

SCHEMA_VERSION = "1.0.0"

# Primary identity for matching is SKU. ASIN is optional metadata.
PRIMARY_ITEM_KEY = "sku"

# One pack event evaluates one order against one open box.
# package_id is optional; when absent, subject.package_id is null.
ONE_EVENT_ONE_PACKAGE = True

# Input interchange for the MVP is JSON (files or API body), not CSV.
INPUT_FORMAT = "json"

# Catalogue may attach multiple reference images per SKU.
ALLOW_MULTIPLE_REFERENCE_IMAGES = True

# Vision is pluggable. MVP uses a deterministic color/blob heuristic so the
# correct-order path and eval harness run without a paid VLM.
VISION_BACKEND_MVP = "heuristic-color-blob"

# Record and detection IDs are UUIDs unless a caller supplies them.
ID_SCHEME = "uuid4"
