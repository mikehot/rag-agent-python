# Re-checking the headline numbers after the first-run effect — local, 2026-10-14

## Why

The LangGraph comparison showed that the first run of a prompt set in a session differed from later runs
(answer texts, and two questions swapping pass/fail). Many cells elsewhere in this repository were single
runs, mostly first runs. This re-ran the cells behind the claims we make, with more repeats, and reversed
the order of the retry on/off comparison, whose original order (off then on) could have confounded it.

## Result: no claim changed

Totals across every agent capture taken on 2026-10-14 (same code from the control-marker fix onward; the
oldest runs predate it and are included where noted in the table):

| Set | Snippet | Runs | Answerable | All cases | Safety-row failures | Empty replies |
|---|---|---|---|---|---|---|
| dev (26) | 800 | 3 | 7/7 each | 26/26 each | 0 | 0 |
| dev (26) | 360 | 3 | 3/7 each | 22/26 each | 0 | 0 |
| held-out v1 (27) | 800 | 6 | 12/13 each | 26/27 each | 0 | 0 |
| held-out v1 (27) | 360, retry on | 3 | 13/13 each | 27/27 each | 0 | 0 |
| held-out v1 (27) | 360, retry off or absent | 4 | 11/13 each | 25/27 each | 0 | 8 (2 per run) |
| held-out v2 (22) | 800 | 4 | 10/11 each | 21/22 each | 0 | 0 |
| held-out v2 (22) | 360 | 3 | 7/11 each | 18/22 each | 0 | 0 |

Safety rows = ACL no-leak, boundary, out of scope, not-in-knowledge-base, no-tool, list documents.

- **Snippet length (360 vs 800):** the totals did not move in any repeat. The effect is question-dependent
  (large on dev and v2, small on v1) exactly as reported before.
- **Empty-reply retry:** with the order reversed (on, off, on, off) it is 27/27, 25/27, 27/27, 25/27 and the
  failing cases are always the same two (HO-07, HO-12). The original off-then-on result was not an order
  effect.
- **Safety rows:** no failure in any of the runs above.
- **Cross-document set** (all cells at snippet 800):

| Cell | Runs | Totals | Per question |
|---|---|---|---|
| Java `POST /api/ask` | 3 | 6/12 each | identical every run |
| Agent `single` | 7 | 9/12 each | XD-09 and XD-10 failed 0/7; XD-01 passed 1/7 and XD-03 passed 6/7; the other 8 passed 7/7 |
| Agent `decompose` | 5 | 12/12 each | every question passed in every run |

## What did change

- **Which questions fail under `single` is not fixed.** The total is 9 in all seven runs because XD-01 and
  XD-03 trade places: when one passes the other fails. The earlier statement "single missed XD-01, XD-09
  and XD-10" was true of one run. The stable statements are: XD-09 and XD-10 are missed by `single` and
  answered by `decompose`; XD-01/XD-03 are an unstable pair under `single` and both pass under `decompose`.
- `decompose` is the more reproducible policy here: 5 of 5 runs perfect against a `single` that reshuffles.

## What this does not explain

Earlier I suggested that the first-run effect might explain the Java project's 43/45 against 38/45. These
data do not support that: the first-run effect changed texts and swapped a pair, but did **not** change a
total in any cell. The 5-case gap remains unexplained, and a restart-separated rerun of the Java harness
has not been done.

## Still not done

A repeat that includes a model reload (Eject and load from the LM Studio UI, Thinking off) has not been
done for the agent sets: all repeats above are within one model-server session. One author, one model,
literal term matching, as before.
