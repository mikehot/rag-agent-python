# rag-agent-python

A small Python service that sits **in front of** an existing Java RAG knowledge base and gives it an
agent: it decides which knowledge-base tool to call, calls it over [MCP](https://modelcontextprotocol.io/)
with the caller's identity, and answers from what comes back.

It deliberately does **not** re-implement chunking, embeddings, or vector search. Those live in the Java
service ([`mikehot/rag-knowledge-base`](https://github.com/mikehot/rag-knowledge-base)), which also owns
access control, citations, and the safety checks. This repository owns the part that is naturally Python:
orchestration, tool selection, and evaluating whether the agent calls tools correctly.

## Status

Early scaffold. What exists today:

| Piece | State |
|---|---|
| FastAPI app, `GET /health` | done |
| `POST /ask` | done: runs the bounded agent loop with the caller's own bearer token |
| MCP client for the Java service | done: offline contract tests + 12 live tests (skipped without a knowledge base) |
| Bounded tool-calling loop | done: turn and tool-call limits, local argument validation, hidden scope arguments |
| Tool-call evaluation | done: 26 dev + 27 held-out cases, one run per cell; see [the report](evaluation/reports/tool-calls-v1-local-2026-10-14.md) |

## Try it

The caller's knowledge-base token is forwarded unchanged; this service stores no credentials.

```bash
RAG_AGENT_KB_BASE_URL=http://localhost:8080 \
  uv run uvicorn rag_agent.main:app --port 8090

curl -s -X POST localhost:8090/ask \
  -H "Authorization: Bearer $KB_TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "Pod 的重启策略有哪些？默认是哪个？"}'
```

| Field | Meaning |
|---|---|
| `answered` | `false` when a limit was reached or the model gave nothing; read `stop_reason` |
| `tool_calls` | every tool the agent tried and how it ended (`ok`, `tool_error`, `rejected_arguments`, ...) |
| `retrieved` | candidates the knowledge base returned; **not citations** |
| `cited_documents` | the subset of `retrieved` whose filename the answer text actually names |

`401` for a missing or rejected token, `502` when the knowledge base or the model server is unreachable.
A model that fails never produces an invented answer.

## What was learned so far

A small local model needed four changes before it used the knowledge base correctly through MCP, and one of
them was on the Java side (tool output was cut at 360 characters, hiding evidence that had been retrieved).
The full account, including what the numbers do not show, is in the
[tool-call report](evaluation/reports/tool-calls-v1-local-2026-10-14.md).

## Plan

1. ~~MCP client with contract tests~~, ~~bounded agent loop~~, ~~tool-call evaluation~~, ~~held-out set~~.
2. Repeats that include a model-server restart; look at the wrong "not in the knowledge base" answers (H2-05).
3. Wire the loop to `POST /ask`, then: LangGraph wrapper, streaming, multi-turn memory.

Everything runs locally against a local model. No hosted service is assumed.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run uvicorn rag_agent.main:app --reload
```

Live contract tests and the evaluation need a running knowledge base and a model server:

```bash
export RAG_AGENT_LIVE_KB_URL=http://localhost:8080
export RAG_AGENT_LIVE_CREDENTIALS=/path/to/credentials.json   # {actor: {username, password}}
uv run pytest tests/test_live_contract.py
uv run python evaluation/run_tool_eval.py --dataset evaluation/datasets/tool_calls_v1.jsonl \
  --credentials "$RAG_AGENT_LIVE_CREDENTIALS" --label local --out-dir /tmp/tool-eval
```

Configuration comes from `RAG_AGENT_*` environment variables (see `src/rag_agent/config.py`).
Credentials for the knowledge base are never stored in the repository.

## License

MIT
