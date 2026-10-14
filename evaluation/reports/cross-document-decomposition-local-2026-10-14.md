# Cross-document questions: Java single-shot vs the Python agent — local, 2026-10-14

Pre-registered in `cross-document-preregistration-2026-10-14.md` (committed before any run). Same model
(`gemma-4-26b-a4b-qat`, Thinking off, temperature 0), same knowledge base, snippet length 800, 6 model turns
and 4 tool calls for both agent policies. One run per cell.

## Result (pre-registered set: 12 compound questions, two facts from two different documents)

| Cell | Passed | Tool calls |
|---|---|---|
| Java `POST /api/ask` (one retrieval of the user's wording) | 6/12 | n/a |
| Python agent, policy `single` | 9/12 | 16 |
| Python agent, policy `decompose` | **12/12** | 24 |

Regression control, held-out v1 (27 cases): `single` 26/27 and `decompose` 26/27, identical per category,
22 tool calls each. Decomposition did not add calls on single-fact questions.

**Decision rule (fixed in advance): at least 3 more passes than `single`, and the control no worse by more
than 1.** Met: +3, control +0. Read as an observation on twelve questions, not as an established effect.

## What the answers show

In the first run the three questions `single` missed were each answered from one document only: XD-09
never retrieved the "important data" half, XD-01 never retrieved the Job chunk, and XD-10 answered the
minors question from the personal-information law instead of the labour law. Over seven runs, XD-09 and
XD-10 failed every time, while XD-01 and XD-03 traded places (see the re-check report); `decompose` passed
all twelve in all five runs. Under `decompose` each fact appeared with the right
document. I read all twelve; the passes are not coincidental term matches.

Note that `single` is not strictly single-query: in 4 of 12 questions the model searched twice on its own.
The policies differ by one prompt paragraph, not by capability.

## Exploratory, added after the pre-registered run (not pre-registered)

The 12 questions are explicitly compound ("A? Also B?"), which is the easy shape to decompose. The three
cross-document questions already in the Java datasets (and their paraphrases) are less explicit.

| Cell, 6 questions (CORP-034/035/045 and CORP2-034/035/045) | Passed |
|---|---|
| Java `POST /api/ask` | 4/6 |
| Python agent `single` | 6/6 |
| Python agent `decompose` | 6/6 |

On these the agent beats the Java baseline and decomposition adds nothing. Two of the three questions name
both laws in the question itself, so this is still not a test of fully implicit questions.

## What this supports, and what it does not

- **Supports:** the Python layer can do something the single-shot Java endpoint does not, namely retrieve
  more than once and combine the results; on cross-document questions it scored 9/12 and 12/12 against
  6/12, and 6/6 against 4/6. That gives the layer a reason to exist beyond forwarding.
- **Does not support:** a general claim that decomposition improves answers. On single-fact questions it
  changed nothing; on the exploratory set it changed nothing; on the 12 compound questions it changed 3.
- **The comparison with Java is not clean.** The agent also reads whole chunks (snippet 800) and may issue
  several searches freely, while `/api/ask` has its own prompt, Top-K and similarity threshold. "Agent beats
  pipeline" here means "this agent configuration beat this pipeline configuration on these questions".
- **Scoring checks that both facts appear, not that each is attributed to the right document.** That was
  checked by reading, not by the rule.

## Two incidental findings

- **A chat-template control marker leaked into answers.** `<|channel>` appeared at the start of free-text
  answers in 10 of the 416 agent answers captured today (7 of 19 capture files; the Java endpoint, which
  returns structured JSON, was unaffected). It was invisible to the scorer. The agent now strips such markers; this is a hygiene fix, not
  an improvement of answer quality, and the earlier numbers in this repository were measured with it present.
- **The `decompose` prompt makes the model quote document titles and article numbers** ("依据《数据安全法》
  第二十七条") more readily. Nothing here checks that those citations are accurate; the knowledge base's own
  citation mechanism (backend-owned, validated) is not used by the agent path.

## Limits

Twelve plus six questions, one author, one model, one session, literal term matching. The Java project's
corpus-v2 addendum shows that model-server state alone moved a result by 5 cases between two days; a repeat
after a model reload has not been done for this set.
