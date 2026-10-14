# Pre-registration: the LangGraph version must not regress

Committed before any run of the graph engine.

## Purpose

To show that the agent can be expressed in LangGraph, **not** to improve scores. The graph version shares
tool validation/execution (`execute_call`) with the reference loop and is tested to send the model the
identical conversation on 16 scripted scenarios. If both engines are deterministic against the same model
state, the live results should be identical; any difference is therefore either model nondeterminism or a
defect in the graph version, and is reported either way.

## Matrix

Engines `loop` and `graph`; same session, same backend (snippet 800), same model state, policy `single`
unless stated, retry on, one run per cell.

| Set | Cases | Limits |
|---|---|---|
| tool_calls_v1 (dev) | 26 | 4 turns / 3 calls |
| tool_calls_heldout_v1 | 27 | 4 / 3 |
| tool_calls_heldout_v2 | 22 | 4 / 3 |
| cross_document_v1, policy single | 12 | 6 / 4 |
| cross_document_v1, policy decompose | 12 | 6 / 4 |

## Criterion, fixed in advance

- **Non-regression:** in every row the graph passes at least as many cases as the loop. A row where it
  passes fewer is a failure of this criterion and is investigated and reported, not averaged away.
- **Equivalence (informative):** the set of cases whose pass/fail differs between engines, and any
  difference in tool-call count, is listed per row.
- Nothing in the graph version is tuned after seeing these results.

## What this cannot show

That LangGraph is better for this agent (the loop is shorter and does the same), only that the graph form
works and does not lose behaviour. One run per cell; the model is the same one throughout.
