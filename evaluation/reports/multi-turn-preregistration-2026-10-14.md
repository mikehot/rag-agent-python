# Pre-registration: do follow-up questions work with conversation memory, and is retrieval repeated?

Committed before any run of `multi_turn_v1.jsonl`.

## Why

Sessions keep earlier questions and final answers (never tool output) so that a follow-up such as
"那最多重试几次？" can be understood. The design intent is that **every turn retrieves again** with the
caller's current permissions. A live smoke check on 2026-10-14 showed a follow-up answered with no tool call
at all (`turns` 1), from the previous answer. That is correct here and not what the design intends.

## Design

10 two-turn conversations (8 follow-ups that depend on turn 1, 2 negative ones whose turn 1 is declined).
Each follow-up's expected fact was verified to occur in its source document and not in its own question.
Per conversation: turn 1 remembered, turn 2 with the session, and turn 2 alone as a control. One run, then
a second run to see stability.

## What will be reported, whichever way it falls

- Turn 2 correct with the session vs alone (is memory needed, and does it work).
- How often turn 2 ran **no search**, and, stratified after the fact, how often that happened when turn 1's
  answer did *not* already contain the fact (the case where skipping retrieval is a defect).
- Negative conversations: turn 2 must still decline and must not contain the withheld facts.

No prompt is changed between the two runs. If skipping retrieval turns out to be common, a stronger rule
is a candidate for the next evaluation on new conversations; it is not applied here.

## Limits

Ten conversations, one author, literal matching, one model, one session. It cannot test permission
revocation mid-session, which is the reason repeated retrieval matters; that needs an ACL change between
turns and is not covered.
