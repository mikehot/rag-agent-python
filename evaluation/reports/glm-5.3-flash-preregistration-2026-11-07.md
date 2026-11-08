# Pre-registration: third model, cloud `GLM-5.3-Flash-Event` (SCNet) + local `text-embedding-bge-m3`

Committed before any run. Mirror copy of `rag-knowledge-base/evaluation/reports/glm-5.3-flash-preregistration-2026-11-07.md`
for traceability across both repositories. Covers Cell A in `rag-knowledge-base` and Cells B, C, D in
`rag-agent-python`.

## Why

The evaluation suite across both repositories has so far measured two local 4-bit MLX models
(`google/gemma-4-26b-a4b-qat` and `qwen/qwen3.8-27b`). This pre-registration tests a cloud model,
`GLM-5.3-Flash-Event` on SCNet (`https://api.scnet.cn/api/llm/v1`), while keeping the knowledge base's
embedding model (`text-embedding-bge-m3` via local LM Studio) and retrieval settings 100% unchanged.

## What stays fixed

- **Knowledge base retrieval**: `text-embedding-bge-m3` (local LM Studio, 1024-dim), `VECTOR` Top-K=5,
  similarity threshold 0.35, chunk 700/100, citation mode `REPAIR`.
- **Chat model**: `GLM-5.3-Flash-Event` at `temperature=0`, `reasoning_effort="none"` (`reasoning_tokens=0`,
  matching the Thinking-off condition of the local evaluations), `AI_RESPONSE_FORMAT=json_object` on the
  Java backend.

## Four cells

1. **Cell A (`corpus_v2.jsonl`, 57 cases × 3 runs, Java `POST /api/ask`)**: 45 answerable, 12 refusals,
   backend restarted before each run.
2. **Cell B (`cross_document_v1.jsonl`, 12 compound cases + `tool_calls_heldout_v1.jsonl`, 27 control cases)**:
   `RAG_AGENT_SNIPPET_CHARS=800`, 6 turns, 4 tool calls. Compare Java `POST /api/ask`, Python agent `single`
   (×3 runs), and Python agent `decompose` (×3 runs). Decision rule: `decompose` credited with a benefit
   only if ≥ +3/12 over `single` and held-out control drops by ≤ 1.
3. **Cell C (`multi_turn_v2.jsonl`, 10 two-turn conversations × 3 runs)**: snippet length 360,
   comparing turn 2 with session vs alone (`RAG_AGENT_ENFORCE_RETRIEVAL=true` and `false`).
4. **Cell D (`revocation_v1.jsonl`, 5 mid-session ACL revocation cases × 3 runs)**: comparing
   `RAG_AGENT_ENFORCE_RETRIEVAL=true` vs `false`. Gate: 0 leaks on valid runs when enforcement is `on`,
   with a retrieval on every valid turn 2.
