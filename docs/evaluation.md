# Pack evaluation

These figures are one Claude run against one label sheet of 15 open-box orders. They are not an overall accuracy. No other model is part of this score.

## The set

| | Count |
|---|---|
| Orders scored | 14. ORD-005 is excluded |
| Model calls | 15, one per order, including the excluded order |
| Model | claude-sonnet-4-5 |
| Run | 2026-10-09, wall clock 200 seconds |
| Photos | 23, one or two views of the same open box |
| Ingestion failures | 0 |
| Labeler | Ruthvik, one sheet |
| Second labeler | none |

The orders and photos came from the Drive folder shared for this run. A closed carton is not in this set. The earlier three subjects (RTN-019, RTN-001, HUB-CLINDACAN-600) are not part of this denominator.

ORD-005 was called, then removed from the score. Its carton photo includes a printed picture of a person’s head, so that image is not reused. The row below is the score without it.

## Verdict

The agent decision maps `SEAL` to `PASS`, `STOP_FIX` to `FAIL`, and `UNCERTAIN` to `UNCERTAIN`.

| | Count |
|---|---|
| Orders | 14 |
| Verdict matches | 3 |
| Verdict match | 3/14 (21.4%) |
| False seals | 0 |
| Label `PASS`, agent `FAIL` | 5 |
| Label `PASS`, agent `UNCERTAIN` | 2 |
| Label `FAIL`, agent `UNCERTAIN` | 3 |
| Label `UNCERTAIN`, agent `FAIL` | 1 |

The agent sealed nothing. Every labeled pass was stopped or left uncertain. The three matches are all labeled fails that the agent also failed: ORD-002, ORD-010, ORD-021.

| Order | Label | Agent | Why the agent stopped or held |
|---|---|---|---|
| ORD-002 | FAIL | FAIL | Extra items |
| ORD-004 | PASS | FAIL | Missing 1, extra 6 |
| ORD-010 | FAIL | FAIL | Extra items |
| ORD-015 | PASS | UNCERTAIN | Low confidence or unclear brand |
| ORD-018 | FAIL | UNCERTAIN | Low confidence or unclear brand |
| ORD-021 | FAIL | FAIL | Extra items |
| ORD-022 | PASS | UNCERTAIN | Low confidence or unclear brand |
| ORD-027 | PASS | FAIL | Extra items |
| ORD-028 | FAIL | UNCERTAIN | Low confidence or unclear brand |
| ORD-031 | PASS | FAIL | Extra items |
| ORD-040 | UNCERTAIN | FAIL | Variant or colour mismatch |
| ORD-042 | PASS | FAIL | Missing 1, extra 7 |
| ORD-044 | FAIL | UNCERTAIN | Low confidence or unclear brand |
| ORD-050 | PASS | FAIL | Extra items |

## Fields

`unclear` on the sheet is left out of that field. A model `UNCERTAIN` verdict does not erase an extra or a missing line the model still listed.

| Field | Scorable | Agree |
|---|---:|---:|
| extra item | 14 | 3 |
| missing item | 13 | 10 |
| product damage | 13 | 13 |
| item present | 14 | 12 |

Product damage agreed on every scorable row: the sheet said no, and the agent reported no product damage. The extra-item field is the weak one. The agent treats cables, remotes, manuals, and stands that belong with the product as extras, then refuses to seal.

## Cost

Token totals were not stored. Fifteen calls, about 200 seconds. That is not an invoice.

## Files

| What | Where |
|---|---|
| This explanation | `docs/evaluation.md` |
| Claude outputs | `docs/held-out-15order-claude-results.json` |
| Labels and photos | the Drive folder for this run, not copied into the repo |
