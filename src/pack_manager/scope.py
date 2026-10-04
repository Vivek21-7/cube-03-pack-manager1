"""Track boundary. Round 3 will compose this agent with other tracks.

This package verifies an open outbound pack against an order.
It does not receive inventory, prep units, process returns, or recover exceptions.
"""

TRACK_ID = "03"
AGENT_NAME = "pack-manager"

IN_SCOPE = (
    "Compare open-package photograph(s) to expected order line items.",
    "Identify, count, and match visible products to catalogue SKUs.",
    "Detect missing, wrong, extra, and quantity-mismatch items.",
    "Emit SEAL or STOP_AND_FIX with a structured evidence record.",
)

OUT_OF_SCOPE = (
    "Receiving / inbound verification",
    "Prep / labeling / polybagging quality",
    "Returns inspection",
    "Post-exception recovery workflows",
    "FBA / Amazon-controlled packing",
)
