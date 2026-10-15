# rag-agent-python

A small Python service that sits **in front of** an existing Java RAG knowledge base and gives it an
agent: it decides which knowledge-base tool to call, calls it over [MCP](https://modelcontextprotocol.io/)
with the caller's identity, and answers from what comes back.

It deliberately does **not** re-implement chunking, embeddings, or vector search. Those live in the Java
service ([`mikehot/rag-knowledge-base`](https://github.com/mikehot/rag-knowledge-base)), which also owns
access control, citations, and the safety checks. This repository owns the part that is naturally Python:
orchestration, tool selection, and evaluating whether the agent calls tools correctly.

## Status

A working service with small, pre-registered evaluations. It runs locally against a local model; there is
no hosted deployment, and every number below comes from datasets of 5 to 27 cases written by one person,
scored by literal string matching, on one model (Gemma 4 26B, Thinking off).

| Piece | State |
|---|---|
| FastAPI app, `GET /health` | done |
| `POST /ask`, `POST /ask/stream` | done: bounded agent loop with the caller's own bearer token; SSE streaming; opt-in conversation memory |
| MCP client for the Java service | done: offline contract tests + 12 live tests (skipped without a knowledge base) |
| Bounded tool-calling loop | done: turn and tool-call limits, local argument validation, hidden scope arguments |
| LangGraph version of the same agent | done: `RAG_AGENT_ENGINE=graph`; identical to the loop on scripted scenarios (including remembered conversations); same pass counts on 5 live sets; see [the report](evaluation/reports/langgraph-local-2026-10-14.md) |
| Tool-call evaluation | done: 26 dev + 27 held-out cases, one run per cell; see [the report](evaluation/reports/tool-calls-v1-local-2026-10-14.md) |
| Cross-document questions | done, **opt-in**: `RAG_AGENT_POLICY=decompose`. On 12 compound questions the Java single retrieval passed 6, the agent 9, decomposition 12, with the held-out control unchanged (26/27); an observation on twelve questions, not an established effect ([report](evaluation/reports/cross-document-decomposition-local-2026-10-14.md)) |
| Conversation memory | done, opt-in, in-process; follow-ups 5/8 with a session vs 2/8 without ([report](evaluation/reports/multi-turn-local-2026-10-14.md)) |
| Retrieval enforced on every remembered turn | done: revoked-document repeat 3/3 off, 0/3 on, three valid cases ([report](evaluation/reports/enforce-retrieval-local-2026-10-15.md)) |
| OpenTelemetry tracing | done: allow-listed attributes only, console output only (no network exporter) |
| Docker / compose | done: non-root, read-only filesystem, health check; one-command stack with the Java repository |
| CI | `uv sync --locked`, `ruff check`, `ruff format --check`, `pytest` on pushes to `main` and on every pull request |
| Demo | scripted drivers (`scripts/demo.py`, `scripts/demo_admin.py`) and a silent 3 min 47 s terminal recording in the Java repository ([notes](docs/DEMO_VIDEO.md)) |

**Not done:** a network trace exporter; a session store shared between processes (sessions are
in-process, so several replicas are untested); rate limiting and any load test; evaluation on a second
model; a larger or independently written test set; and the fixed refusal after a failed retry has never
been triggered by a live model.

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

## Streaming and conversations

```bash
# server-sent events: session, tool_call, delta..., answer, done
curl -N -X POST localhost:8090/ask/stream \
  -H "Authorization: Bearer $KB_TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "Pod 的重启策略有哪些？", "remember": true}'

# a follow-up that only makes sense with the earlier turn
curl -s -X POST localhost:8090/ask \
  -H "Authorization: Bearer $KB_TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "那默认值是哪个？", "session_id": "<id from the first call>"}'

curl -s -X DELETE localhost:8090/sessions/<id> -H "Authorization: Bearer $KB_TOKEN"
```

- `delta` events are a preview. The `answer` event is authoritative: it has control markers removed and is
  what `cited_documents` is computed from.
- **Nothing is remembered unless asked.** A session holds earlier questions and final answers only (never
  tool output), is bound to the token that created it, and expires (default 30 minutes, 6 turns, in memory).
  Retrieval runs again every turn with the caller's current permissions, and now it is **enforced**: in a
  conversation with earlier turns a final reply is accepted only after a successful retrieval in the
  current turn (one changed retry, then a fixed refusal with `stop_reason="no_retrieval"`; no text is
  streamed before the first retrieval). With enforcement off, a model asked to repeat a document that
  had just been revoked did so verbatim in 3 of 3 valid runs, twice; with it on, 0 of 3, twice, and every
  turn retrieved ([report](evaluation/reports/enforce-retrieval-local-2026-10-15.md)). Three valid cases,
  one model, one revocation path; the stored earlier answer is not erased on revocation, and the
  fixed-refusal path was never exercised by a live model.
  `RAG_AGENT_ENFORCE_RETRIEVAL=false` turns it off, for comparison only.
- Measured on 10 two-turn conversations: memory turned 2/8 follow-ups into 5/8 and retrieval was repeated
  wherever the fact was new, but restated follow-up queries retrieve less reliably than a user's own
  wording ([report](evaluation/reports/multi-turn-local-2026-10-14.md)).

## Tracing

OpenTelemetry spans for every request: `agent.ask` (root) with a child `llm.chat` per model call and a
`tool.call` per tool call. Off by default; `RAG_AGENT_TRACING=console` prints finished spans to standard
output. There is no network exporter in this version.

Spans carry behaviour only: engine, policy, durations, counts, outcomes, model name, finish reason and
token counts. **They never carry the question, the answer, tool arguments, retrieved text, a bearer
token, a username or a session id.** The allow-list of attribute names lives in
`src/rag_agent/telemetry.py` and setting any other name raises; tests assert that no span in a real
request contains any of those values.

## Run with Docker

The image is built from `uv.lock` only (`uv sync --locked`), runs as an unprivileged user with a read-only
filesystem and no capabilities, and has a health check.

```bash
# the agent alone, pointed at a knowledge base and a model server on the host
docker compose up --build
```

```bash
# database + Java backend + agent from one command (the two repositories must be siblings)
export AI_EMBEDDING_MODEL_ID=text-embedding-bge-m3 AI_TEMPERATURE=0
docker compose -p ragdemo \
  -f ../rag-knowledge-base/docker-compose.yml -f docker-compose.yml -f docker-compose.with-kb.yml \
  up -d --build
```

The model server (LM Studio) still runs on the host, with **Enable Thinking off**. Verified on 2026-10-14:
the whole stack came up in 11 seconds with a warm build cache and answered real questions through the
agent container. Things to know:

- **The first build of the Java image took 16 minutes** on the machine used here: Maven Central answered
  at about 0.7 MB/s from this network and the build prints nothing under `-q`. Later builds hit the cache.
- Relative paths in a layered `-f` command resolve against the *first* file's directory, which is why
  `docker-compose.with-kb.yml` restates the build context.
- The Java compose file pins `container_name`s, so it collides with containers left from an earlier run of
  the same file. Remove those, or give the stack different names with an extra override file.
- A `RAG_AGENT_*` setting this version does not know is ignored on purpose, but a warning is logged at
  startup (it caught a real case where an image predated a setting).

## What was learned so far

A small local model needed four changes before it used the knowledge base correctly through MCP, and one of
them was on the Java side (tool output was cut at 360 characters, hiding evidence that had been retrieved).
The full account, including what the numbers do not show, is in the
[tool-call report](evaluation/reports/tool-calls-v1-local-2026-10-14.md).

## Plan

1. ~~MCP client with contract tests~~, ~~bounded agent loop~~, ~~tool-call evaluation~~, ~~held-out set~~.
2. ~~`POST /ask`, LangGraph version, streaming, opt-in memory, tracing, Docker, demo recording~~.
3. Open: repeats that include a model-server restart; the wrong "not in the knowledge base" answers
   (H2-05); a second model on the same sets; a session store that survives more than one process.

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
