# Two-turn conversations with opt-in memory — local, 2026-10-14

Pre-registered in `multi-turn-preregistration-2026-10-14.md` (committed before any run). Local Gemma 4 26B
(Thinking off), the Java knowledge base at the default snippet length (360), 10 conversations, two
identical runs.

## Result

| | Passed |
|---|---|
| Follow-up, turn 2 **with** the session | 5/8 |
| Follow-up, turn 2 **alone** (control) | 2/8 |
| Turn 2 ran a search (with the session) | 6/8 |
| Negative conversations (declined, nothing withheld leaked) | 2/2 |

Both runs gave exactly the same per-conversation outcome.

### Is memory needed, and does it work?

Needed: without the session 2 of 8 follow-ups were answered, with it 5 of 8. The three that fail with the
session are not memory failures (below).

### Is retrieval repeated on turn 2? (the design intent)

Stratified after the fact on whether turn 1's own answer already contained the expected fact:

| | Conversations | Searched on turn 2 | Correct |
|---|---|---|---|
| Fact already in turn 1's answer | 4 | 2 (the other 2 answered from the conversation) | 4 |
| Fact **not** in turn 1's answer | 4 | **4** | **1** |

Where the model had to look something up, it always did. The two cases that skipped retrieval were
questions whose answer was already on the screen. That is acceptable for these two, but it shows that
nothing *forces* retrieval: the follow-up rule is a prompt, and a user whose permissions were revoked
mid-session could still be handed what an earlier answer already said. This set cannot test that (it
needs an ACL change between turns) and it was not tested.

### Where it actually fails

| Conversation | What happened |
|---|---|
| MT-01 Deployment rolling update -> "最多允许多少比例的 Pod 不可用" | The follow-up was understood (answered about `maxUnavailable`) and retrieved, but the retrieved text did not contain the 25% default |
| MT-04 Job failure -> "最多重试几次" | Understood and retrieved; the answer describes `backoffLimit` but not the default of 6 |
| MT-06 daily working hours -> "那每周呢？" | The restated query retrieved unrelated chunks (part-time work, one day of rest) instead of the weekly limit |

These are the same single-turn weaknesses as before: whether the chunk that holds the number is
retrieved. A restated follow-up is a *new* query written by a small model and it retrieves less reliably
than a user's own wording. Memory is not what is breaking.

## What this does not show

Ten conversations, one author, literal matching, one model. No conversation longer than two turns, no
concurrent sessions under load, and no test of revocation mid-session. The control ("turn 2 alone") is
deliberately unfair to the model; it measures whether memory is necessary, not how good the agent is.

## Operational notes

- Sessions are opt-in (`remember` or an existing `session_id`), process-local, owner-bound by a keyed
  hash of the bearer token, expire after 30 minutes by default and keep at most 6 turns.
- Only questions and final answers are kept. A session id belonging to someone else is
  indistinguishable from an unknown one (both 404).
