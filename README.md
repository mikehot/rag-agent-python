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
| `POST /ask` | **placeholder**, returns `501` until the agent loop exists |
| MCP client for the Java service | not started |
| Bounded tool-calling loop | not started |
| Tool-call evaluation set | not started |

## Plan

1. **MCP client** (standard library + `httpx`) with contract tests against the Java service's `/mcp`.
2. **Bounded agent loop**: step limit, tool-call budget, and no data leakage when access is denied.
3. **Tool-call evaluation**: right tool, right arguments, nothing leaked for unauthorized users, repeated
   runs with variance reported, the same way the Java project reports its evaluation.
4. Later: LangGraph wrapper, streaming, multi-turn memory.

Everything runs locally against a local model. No hosted service is assumed.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run uvicorn rag_agent.main:app --reload
```

Configuration comes from `RAG_AGENT_*` environment variables (see `src/rag_agent/config.py`).
Credentials for the knowledge base are never stored in the repository.

## License

MIT
