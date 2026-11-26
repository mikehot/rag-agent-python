# Security Policy

## Security Model & Boundaries

- **Per-caller identity passthrough:** This service holds no knowledge-base credentials. Every `/ask` and `/ask/stream` request must supply the caller's own `Authorization: Bearer` token, which is forwarded unchanged to `rag-knowledge-base`.
- **Session isolation & revocation guard:** Multi-turn sessions are keyed to a salted digest of the caller's bearer token (in-memory or optional SQLite WAL via `RAG_AGENT_SESSION_DB_PATH`). With `RAG_AGENT_ENFORCE_RETRIEVAL=true` (default), remembered turns require a live retrieval in the current turn so revoked documents cannot be answered from conversation memory.
- **No sensitive logging / fixed error envelopes:** User questions, retrieved chunks, bearer tokens, and LLM API keys are never logged or exported to OTel spans. `/docs`, `/redoc`, and `/openapi.json` are disabled; 422 validation errors return a fixed `invalid_request` envelope without echoing input.
- **Container hardening:** The Docker image runs as unprivileged user `agent` (UID 10001) with `read_only: true`, `cap_drop: [ALL]`, `no-new-privileges:true`, and loopback-only binding (`127.0.0.1`) by default.

## Reporting a Vulnerability

Please report suspected security issues privately to `mikehot668@gmail.com` rather than opening a public GitHub issue.
