# tool-calls-v1 — a local model calling the knowledge base through MCP — 2026-10-14

What this measures: not answer quality in general, but whether an agent built on a small local model
picks the right tool, passes acceptable arguments, stays inside the caller's access, and declines when it
should. 26 hand-written cases, scored by literal rules (`rag_agent/tool_eval.py`).

## Setup

| Item | Value |
|---|---|
| Agent model | `google/gemma-4-26b-a4b-qat` via LM Studio, Thinking off, temperature 0 |
| Knowledge base | the Java service's disposable stack: 28 documents, bge-m3, Top-K=5 |
| Tools | `search_knowledge`, `list_documents`, `get_document_status` over `POST /mcp`, caller's own JWT |
| Limits | 4 model turns, 3 executed tool calls per question |
| Cases | 26: 7 answerable, 3 list documents, 3 no tool needed, 5 ACL no-leak, 5 boundary, 3 out of scope |
| Runs | **one run per configuration** (see Limits of this evidence) |

## Result

| Configuration | Passed | Answerable | Everything else |
|---|---|---|---|
| v1: first prompt, snippet 360 chars | 19/26 | 2/7 | 17/19 |
| v2: policy changes, snippet 360 chars | 22/26 | 3/7 | 19/19 |
| v2 + snippet 800 chars (Java `RAG_AGENT_SNIPPET_CHARS=800`) | **25/26** | 6/7 | 19/19 |

Across all three configurations: no executed call ever carried an identity-bearing argument, no answer
contained text the caller was not allowed to see, and out-of-scope and unauthorized questions were
declined in every case.

## What was found, in the order it was found

1. **A small model fills optional arguments with placeholders.** Given an optional `knowledgeBaseId`, Gemma
   sent `"your_knowledge_base_id_here"` and `"default"`. The server correctly answered `INVALID_ARGUMENTS`,
   so nothing leaked, but every question failed. The agent now hides arguments the application decides
   (`knowledgeBaseId`) from the model, validates `format: uuid` locally, and feeds the reason back so the
   model can retry. Scoping is a policy decision, not something to leave to the model.
2. **Query rewriting by the model hurt retrieval.** The model compressed questions to keywords
   ("生孩子产假至少多少天" → "生孩子产假天数"); similarities dropped (top-1 0.54 for that case) and the evidence
   chunk fell out of the Top-5. The Java `/ask` endpoint embeds the user's own wording. The prompt now asks
   for the user's words verbatim.
3. **The model sometimes declined without searching** ("知识库中没有…" with no tool call). The prompt now says
   factual questions must be searched first. This was one case, not a trend.
4. **The tool returned truncated evidence.** With verbatim queries the evidence chunk *was* retrieved (rank 2
   for the maternity-leave question), but `search_knowledge` returned only the first 360 characters of each
   chunk and the answer sat at character 457 of 635. `/ask` hands the model whole chunks, so the same
   question was answerable there and not here. Making the length configurable in the Java service moved
   answerable cases from 3/7 to 6/7. The default stays 360; whether to raise it is a product decision about
   tool-output size, taken separately.
5. **`search_knowledge` is nearest-neighbour only.** An out-of-scope or unauthorized-topic question still
   returns five chunks with similarity scores; there is no relevance cutoff in the tool. "I don't know" has to
   come from the agent. Because of this the agent reports `retrieved` candidates, never `sources`: a reply
   that says "not in the knowledge base" must not be shown next to five citations.

## The one remaining failure is a scoring false negative

TC-005-S ("backoffLimit 不设的话是多少"): the answer says the default is **6** (correct, and it adds the
indexed-job value). The rule copied from corpus-v2 looks for adjacent substrings such as "是 6", which the
markdown bold in "是 **6**" breaks. The rule was not edited to keep the comparison fixed; read as intended,
the result is 26/26.

## Limits of this evidence

- **One run per configuration.** At temperature 0 inside one model-server session a repeat mostly re-measures
  the same decoding; the Java project's own experience is that the same configuration moved by up to 5 cases
  between days (see its corpus-v2 addendum, 2026-10-14). Treat 22 → 25 as an observation to re-measure, not a
  stable gain.
- **The policy changes were made after reading v1's failures on these same 26 cases.** The v2 and snippet
  numbers are therefore not held-out. A fresh set written afterwards is the real test.
- **Small, one author.** 26 cases written by the person who built the agent; scoring is literal substring
  matching (see the false negative above) and does not judge faithfulness or completeness.
- **Nothing here compares models.** Only one local model was run.
