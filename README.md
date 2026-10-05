# Pack Manager AI — Autonomous Logistics Inspector (CUBE Track 03)

Pack Manager AI is an intelligent visual verification system for outbound order packing. It analyzes open package photographs against order data to autonomously determine whether an order can be **SEALED** for shipping, must **STOP & FIX**, or is **UNCERTAIN** (requiring clearer evidence).

---

## Key Capabilities

- **Multimodal AI Agent**: Powered by Vision Language Models (Google Gemini 2.0 Flash, OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, or local offline simulation).
- **Exact Prompt Specification**: Implements the official Pack Manager AI system prompt, tasks, constraints, and canonical JSON output schema.
- **Defensive Inspection Logic**:
  - `SEAL`: All expected items present, exact quantities, correct variants/colors, zero extra items, no physical damage.
  - `STOP & FIX`: Any missing items, quantity mismatches, colorway/variant differences, extra items, or product/box damage.
  - `UNCERTAIN`: Blurry photos, obscured items, poor lighting, or ambiguous counts (prompts for retake rather than guessing).
- **Interactive Web Inspection Dashboard**: High-visibility UI with real-time AI scanning, photo dropzone, order editor, and 1-click test scenario presets.
- **Audit-Ready Evidence Records**: Deterministic SHA-256 content hashes, check results, and latency tracking compliant with Track 03 schemas.

---

## Quickstart

### 1. Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install package with dependencies
pip install -e ".[dev]"
```

### 2. Launch Interactive Web Dashboard

```bash
python -m pack_manager.server --port 8000
# or: pack-manager-server
```

Open your browser at **`http://localhost:8000`**. You can:
- Switch between all 5 prompt scenarios with 1 click.
- Upload custom package photos or edit the expected order JSON.
- Toggle between **Offline Simulation Mode**, **Google Gemini**, **OpenAI**, or **Anthropic Claude**.
- View live SEAL / STOP & FIX verdicts, item matching tables, defect alerts, and raw JSON.

---

## CLI Usage

### Direct Agent Inspection

Run the AI agent on any order JSON and box photograph:

```bash
python -m pack_manager.agent \
  --order fixtures/scenarios/example_1_correct_order/order.json \
  --photo fixtures/scenarios/example_1_correct_order/box_photo.png \
  --provider simulation
```

#### With Live Multimodal Models:

```bash
# Using Google Gemini:
set GEMINI_API_KEY="your-api-key"
python -m pack_manager.agent --order order.json --photo box.jpg --provider gemini

# Using OpenAI GPT-4o:
set OPENAI_API_KEY="your-api-key"
python -m pack_manager.agent --order order.json --photo box.jpg --provider openai

# Save output JSON:
python -m pack_manager.agent --order order.json --photo box.jpg --json-out result.json
```

### Pipeline Integration Demo

```bash
# Run with Real Agent backend:
python -m pack_manager.demo --agent

# Run with baseline heuristic backend:
python -m pack_manager.demo
```

---

## Canonical Evaluation Scenarios

Pack Manager includes synthetic fixtures and automated tests for the prompt's scenarios:

| Scenario | Expected Order | Photo Contents | Decision | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1: Correct Order** | 2× Black T-Shirt, 1× Blue Cap | 2 black t-shirts, 1 blue cap | **SEAL** | All items present, correct quantities and colors |
| **Example 2: Wrong Item** | 2× Black T-Shirt, 1× Blue Cap | 2 black t-shirts, 1 RED cap | **STOP & FIX** | Cap color mismatch (expected blue, detected red) |
| **Example 3: Missing Item** | 2× Black T-Shirt, 1× Blue Cap, 1× Manual | 2 black t-shirts, 1 blue cap | **STOP & FIX** | Manual missing - not visible in box |
| **Example 4: Extra Item** | 2× Black T-Shirt, 1× Blue Cap | 2 black t-shirts, 1 blue cap, 1 red scarf | **STOP & FIX** | Unexpected item in box (red scarf not on order) |
| **Example 5: Unclear Photo** | 2× Black T-Shirt, 1× Blue Cap | Blurry / obscured image | **UNCERTAIN** | Photo quality insufficient - request clearer image |
| **Example 6: Damaged Goods** | 2× Black T-Shirt, 1× Blue Cap | Punctured packaging / crushed box | **STOP & FIX** | Packaging damage detected during inspection |

---

## Output JSON Schema

The agent returns canonical JSON strictly adhering to the prompt specification:

```json
{
  "order_id": "ORD-2024-001",
  "order_items": [
    { "name": "Black T-Shirt", "expected_qty": 2, "variant": "black" },
    { "name": "Blue Cap", "expected_qty": 1, "variant": "blue" }
  ],
  "detected_items": [
    {
      "name": "Black T-Shirt",
      "detected_qty": 2,
      "variant": "black",
      "confidence": "high",
      "notes": "Clearly visible, folded, correct quantity"
    },
    {
      "name": "Blue Cap",
      "detected_qty": 1,
      "variant": "blue",
      "confidence": "high",
      "notes": "Clearly visible, folded, correct quantity"
    }
  ],
  "matches": [
    {
      "item_name": "Black T-Shirt",
      "expected_qty": 2,
      "detected_qty": 2,
      "variant_match": true,
      "status": "PASS"
    },
    {
      "item_name": "Blue Cap",
      "expected_qty": 1,
      "detected_qty": 1,
      "variant_match": true,
      "status": "PASS"
    }
  ],
  "missing_items": [],
  "extra_items": [],
  "product_condition": {
    "visible_damage": false,
    "notes": "No visible damage to products or packaging"
  },
  "decision": "SEAL",
  "decision_reason": "All items present, correct quantities, correct colors",
  "confidence": "high",
  "evidence": [
    "All items matched against order",
    "No missing components",
    "No visible damage"
  ]
}
```

---

## Running the Test Suite

```bash
.venv\Scripts\pytest -v
```

All 26 unit and integration tests validate schema enforcement, prompt adherence, failure modes, and REST endpoints.
