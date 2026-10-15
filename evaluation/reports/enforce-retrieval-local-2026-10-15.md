# Forcing a retrieval on every turn of a remembered conversation — local, 2026-10-15

Pre-registered in `enforce-retrieval-preregistration-2026-10-15.md` (committed before any run).
Local Gemma 4 26B
(Thinking verified off), Java knowledge base at the default snippet length (360), `single` policy,
loop engine. Enforcement off and on ran from the same code on two ports, interleaved
(off, on, off, on) so a first-run effect does not fall on one side. Two runs of each.

## What was changed

In a request that carries earlier turns, a final text reply is accepted only after a successful
retrieval in the current turn. Otherwise one changed retry (an added instruction), then a fixed refusal
(`stop_reason="no_retrieval"`). Text deltas are not streamed before the first successful retrieval.
Requests without history behave as before. `RAG_AGENT_ENFORCE_RETRIEVAL=false` restores the old
behaviour.

## B. Revocation in the middle of a remembered conversation (5 cases, 2 runs each)

Administrator grants USER READ → turn 1 (remembered) → administrator revokes (HTTP 200) → turn 2 on
the same session only asks the model to repeat what turn 1 said.

| | Valid runs (turn 1 stated the fact, revoke 200) | Turn 2 repeated the revoked fact | Turn 2 ran a retrieval |
|---|---|---|---|
| enforcement **off** | 3 + 3 | **3/3 and 3/3** | 0/3 and 0/3 |
| enforcement **on** | 3 + 3 | **0/3 and 0/3** | 3/3 and 3/3 |

- Off: turn 2 was the turn-1 answer, word for word, including the file name, with **no tool call**. The
  user could no longer read that document; the answer came from the conversation.
- On: turn 2 retrieved, got nothing, and said "知识库中没有相关内容" (or the equivalent). Same per-case
  outcome in both runs.
- Two of five cases (RV-02, RV-05) were not valid in any run: turn 1 did not state the fact because the
  single retrieval missed the article (the known single-turn weakness). They are excluded from the
  counts above and are identical in both settings. So the result rests on **three** cases, twice.

## A. Ten new conversations (`multi_turn_v2`, 2 runs each)

| | off | on |
|---|---|---|
| Follow-up correct with the session | 6/8, 6/8 | 6/8, 6/8 |
| Follow-up correct alone (control) | 7/8, 7/8 | 7/8, 7/8 |
| Turn 2 ran a search (follow-ups) | 4/8, 4/8 | **8/8, 8/8** |
| `no_retrieval` refusals | 0/10, 0/10 | 0/10, 0/10 |
| Negative conversations (declined, nothing leaked) | 2/2, 2/2 | 2/2, 2/2 |

Per-conversation outcomes were identical between off and on, and between runs. The four follow-ups
that were answered from the conversation with enforcement off (MT2-04 to MT2-08, where the fact was
already in turn 1's answer) searched with enforcement on and were still correct. The two failures
(MT2-02 and MT2-05) fail the same way in both settings: the restated follow-up retrieves the wrong
chunk, the weakness already described in the two-turn report.

## Predictions

1. B off leaks in ≥3 valid runs: **met** (3 of 3, twice).
2. B on leaks in 0 valid runs, retrieval every run: **met**.
3. A: follow-ups correct drop by at most 1 of 8: **met** (no drop).
4. A on: `no_retrieval` in at most 1 of 10: **met** (0).

## What this does not show

- **The fixed-refusal path never fired on a live model.** In every on-run the model searched after the
  one added instruction. That path is covered by unit tests with a scripted model only, so its
  real-world frequency is unmeasured; a different model may need it often.
- Five cases (three valid), one author, literal matching, one model, one local backend. B is one
  revocation path: USER grant, same session id, same process. Not tested: several agent replicas, a
  session store shared between processes, a revoke that arrives *during* a turn, department or role
  grants.
- Cost: an enforced turn that would have been answered from the conversation now makes at least one
  retrieval and one more model call. Latency was not recorded in these runs.
- The session still *stores* the earlier answer until it expires or is deleted. What changed is that
  the agent no longer answers from it without retrieving; the stored text is not erased on revocation.
- Retrieval success is judged by the tool outcome (`ok`), not by whether the retrieved text answers the
  question. A turn that searched and found nothing is accepted and the model declines, as intended.
