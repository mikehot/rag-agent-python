# Pre-registration: does question decomposition help on cross-document questions?

Written and committed **before** any run of `cross_document_v1.jsonl`.

## Why this question

The Java service answers from one retrieval of the user's wording. Its known weak spot is a question
whose parts live in different documents: cross-document questions scored 0/3 in every earlier
configuration, and CORP-035 / CORP2-045 fail even though both documents' evidence is retrieved. An agent
can call the search tool more than once, so it is the one component in this system that could do better
here. It might also do worse: earlier work found that a small model rewriting queries lowered retrieval
quality. This experiment is meant to find out which, not to confirm a hope.

## Design

| Item | Value |
|---|---|
| Questions | 12, each asking two facts that live in two different documents (`cross_document_v1.jsonl`) |
| Validity checks (done before this file) | no duplicate of any earlier question; every accepted term occurs in its own document; no accepted term occurs in the *other* document (so one document cannot answer alone); no accepted term appears in its own question |
| Cells | (a) Java `POST /api/ask`; (b) Python agent, policy `single`; (c) Python agent, policy `decompose` |
| Same for (b) and (c) | model, backend, snippet length 800 (so truncation does not confound), 6 model turns, 4 tool calls |
| Regression control | the 27-case held-out v1 set under (b) and (c) |
| Runs | one per cell, one model-server session |

`decompose` differs from `single` by exactly one paragraph appended to the system prompt, asking the model
to split a multi-part question into self-contained sub-questions and search for each.

## Decision rule, fixed in advance

- Decomposition is credited with a benefit **only if** it passes at least 3 more of the 12 questions than
  `single` **and** the control set is not worse by more than 1 case.
- Otherwise the result is reported as "no demonstrated benefit", including the case where it is worse.
- Comparison with the Java baseline is reported separately and does not enter the rule.

## What this cannot show

Twelve questions, one author, literal term matching, one model, one run per cell. Three or more extra
passes out of twelve is an observation worth following up, not an established effect. The scoring checks
that both facts are present, not that each is attributed to the right document.
