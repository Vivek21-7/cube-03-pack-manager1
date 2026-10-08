# Stage 1 Visual Perception Evaluation: Scope and Limitations

## Evaluation Classification
**Live Gemini validation on synthetic package fixtures**

> **IMPORTANT NOTICE:**  
> The Gemini tests conducted for Stage 1 visual observation utilize real, live multimodal calls to Google Gemini (`gemini-3.5-flash-lite` / `gemini-2.0-flash`).  
> However, the underlying input images (`fixtures/scenarios/example_*`) are **synthetic package fixtures** generated to simulate warehouse packing states (correct order, wrong item colorway, missing item, extra foreign object, low lighting / blur, carton crush damage).
>
> This evaluation does **not** represent real-world warehouse camera validation.  
> Real warehouse operational photographs will be integrated into the evaluation benchmark suite in a subsequent phase.

---

## Stage 1 Perception Contract
1. **Order-Blind Perception**:
   The perception agent does not receive customer order items, expected quantities, or SKU codes. It acts purely as a spatial visual sensor.
2. **One Unit Per Entry**:
   Each visible product unit is detected individually (e.g., two identical shirts yield two distinct `ObservedPhysicalItem` records: `obj-1` and `obj-2`).
3. **Bounding Box Rigor**:
   Coordinates are normalized to `[0.0, 1.0]`. Known `0-1000` coordinate formats are scaled deterministically. Malformed boxes (where `xmin > xmax` or `ymin > ymax`) are strictly rejected without guessing or fabricating spatial dimensions.
4. **Environment Isolation**:
   Pack Manager strictly loads credentials from `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or local `.env` / `.env.local`. It is completely decoupled from any external projects.
