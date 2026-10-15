# Pre-registration: forcing a retrieval on every turn of a remembered conversation

Committed before any run of `multi_turn_v2.jsonl` or `revocation_v1.jsonl`.

## Why

The two-turn report (2026-10-14) found that nothing forces retrieval on a follow-up: two of eight
follow-ups were answered from the earlier answer with no search. That is harmless when the earlier
answer is still correct, and a permission bypass when the caller's access was revoked between the two
turns, because retrieval is where the caller's current permissions are applied. That report said a
stronger rule was a candidate for the next evaluation on new conversations; this is it.

## The change under test

When a request carries earlier turns and the current turn has had no successful retrieval, a final
text reply is not accepted: one changed retry (an added instruction), and if the model still does not
retrieve, the response is a fixed refusal with `stop_reason="no_retrieval"`. Text is not streamed
before the first successful retrieval. Without earlier turns nothing changes. A setting
(`RAG_AGENT_ENFORCE_RETRIEVAL=false`) restores the previous behaviour for comparison.

## Two experiments

**A. `multi_turn_v2.jsonl`**: 10 new two-turn conversations (8 follow-ups, 2 negative ones), written
before looking at any output; facts were checked against the corpus text. Same runner and rules as
`multi_turn_v1`. Run with enforcement off and on, two runs each.

**B. `revocation_v1.jsonl`**: 5 cases. The actor starts with no grant on the document; an
administrator grants USER READ; turn 1 is asked with a remembered session; the administrator revokes;
turn 2 only asks the model to repeat what turn 1 said. Run with enforcement off and on, two runs each.
A run counts only if turn 1 stated the fact and the revoke returned HTTP 200 (reported separately).

## Predictions (written now)

1. B, off: turn 2 repeats the revoked fact in at least 3 of the valid runs. If it repeats it in 0, the
   premise of this change is wrong for this model and this prompt, and that is the headline.
2. B, on: turn 2 repeats the revoked fact in 0 valid runs, with a retrieval in every run.
3. A, on vs off: follow-ups correct with the session drop by at most 1 of 8 in each run. The
   conversations where the fact was already in turn 1's answer are where enforcement costs something,
   because the model now has to retrieve it again.
4. A, on: `no_retrieval` appears in at most 1 of 10 conversations per run.

## What will be reported, whichever way it falls

Both tables, per-case rows, every `no_retrieval`, and any case where enforcement changed a correct
answer into a wrong one. No prompt, threshold or dataset is changed between the runs. If prediction 2
fails, the change is not shipped as a "fix".

## Limits

Ten and five conversations, one author, literal matching, one model, one local backend, two runs each.
Experiment B tests one revocation path (USER grant, same session id, same process). It does not test
several agent replicas, a session store shared between processes, or a revoke that arrives during a
turn. The refusal text does not tell the user *why* nothing was found, by design.
