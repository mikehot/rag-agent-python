# rag-agent-python: Cloud `GLM-5.3-Flash-Event` (SCNet) Evaluation — 2026-11-07

Pre-registered in [`glm-5.3-flash-preregistration-2026-11-07.md`](glm-5.3-flash-preregistration-2026-11-07.md) before any run.
Java backend Cell A (`corpus_v2.jsonl`, 57 cases × 3 runs) and the LLM-as-Judge semantic audit are recorded in
`rag-knowledge-base/evaluation/reports/glm-5.3-flash-corpus-v2-cloud-2026-11-07.md` (`45/45`, `44/45`, `45/45`, mean `44.67/45`, `DISTINGUISHABLE` vs Gemma 4 26B).

---

## 1. Pre-flight: Native Tool-Calling Probe (`evaluation/probe_agent.py`)

Configuration: `RAG_AGENT_LLM_BASE_URL=https://api.scnet.cn/api/llm/v1`, `RAG_AGENT_LLM_MODEL=GLM-5.3-Flash-Event`, `RAG_AGENT_KB_BASE_URL=http://localhost:18080` (disposable corpus fixture, local `text-embedding-bge-m3`).

Result: **8 / 8 passed on first run** (native OpenAI `tool_calls` emitted, zero narrated tool calls, zero identity-argument hallucinations):

| Actor | Case | Outcome | Turns | Retrieved | Latency | Tools Called |
|---|---|---|---:|---:|---:|---|
| `corp.sre` | `search` | `answered` | 2 | 5 | 14.6s | `search_knowledge:ok` |
| `corp.hr` | `search` | `answered` | 2 | 5 | 10.1s | `search_knowledge:ok` |
| `corp.sre` | `list` | `answered` | 2 | 0 | 6.9s | `list_documents:ok` |
| `corp.outsider` | `search_then_none` | `answered` (declined) | 3 | 0 | 8.2s | `search_knowledge:ok,search_knowledge:ok` |
| `corp.sre` | `search_then_none` | `answered` (declined) | 2 | 5 | 6.0s | `search_knowledge:ok` |
| `corp.sre` | `no_tool` | `answered` | 1 | 0 | 2.6s | `-` |
| `corp.sre` | `no_identity_args` | `answered` | 2 | 0 | 14.7s | `list_documents:ok` |
| `corp.sre` | `no_tool_or_refuse` | `answered` (refused delete) | 1 | 0 | 2.3s | `-` |

---

## 2. Cell C: Multi-Turn Conversation & Session Memory (`multi_turn_v2.jsonl`, 3 runs × 2 modes)

Ran 3 repeated runs of `multi_turn_v2.jsonl` (8 positive follow-up conversations + 2 negative/ACL-boundary conversations) against two agent instances:
- **`enforced=off`** (`RAG_AGENT_MEMORY_ENFORCE_RETRIEVAL=false` on port 18091)
- **`enforced=on`** (`RAG_AGENT_MEMORY_ENFORCE_RETRIEVAL=true` on port 18090)

### Results Across 3 Runs (`GLM-5.3-Flash-Event` vs Local `Gemma 4 26B` Baseline)

| Metric | Local Gemma 4 26B (`enforced=off`) | **Cloud `GLM-5.3-Flash-Event` (`enforced=off`, 3 runs)** | **Cloud `GLM-5.3-Flash-Event` (`enforced=on`, 3 runs)** |
|---|---:|---:|---:|
| **Turn 2 with session** (`/8`) | `5 / 8` | **`6 / 8`, `6 / 8`, `6 / 8` (mean `6.00`)** | **`6 / 8`, `5 / 8`, `5 / 8` (mean `5.33`)** |
| **Turn 2 alone (no session)** (`/8`) | `2 / 8` | `5 / 8`, `5 / 8`, `5 / 8` (mean `5.00`) | `5 / 8`, `5 / 8`, `5 / 8` (mean `5.00`) |
| **Turn 2 ran a tool search** (`/8`) | `5 / 8` | **`8 / 8`, `7 / 8`, `8 / 8` (mean `7.67`)** | **`8 / 8`, `8 / 8`, `8 / 8` (mean `8.00`)** |
| **Turn 2 stopped with `no_retrieval`** (`/10`) | `0 / 10` | **`0 / 10` (all 3 runs)** | **`0 / 10` (all 3 runs)** |
| **Negative conversations declined with 0 leak** (`/2`) | `2 / 2` | **`2 / 2` (all 3 runs)** | **`2 / 2` (all 3 runs)** |

### Key Findings
1. **Spontaneous Re-Retrieval on Follow-up Turns**: Unlike local `Gemma 4 26B` (which skipped calling `search_knowledge` on Turn 2 in 3/8 conversations when `enforced=off`, answering from Turn 1 memory), `GLM-5.3-Flash-Event` proactively called `search_knowledge` on Turn 2 in **23 / 24 (`95.8%`)** follow-up turns even with `enforced=off`, and **24 / 24 (`100%`)** with `enforced=on`.
2. **Higher Session Accuracy & Zero ACL Leakage**: `GLM-5.3-Flash-Event` improved Turn 2 session accuracy from Gemma's `5/8` to `6/8` (`75%`) across all three `enforced=off` runs while maintaining **0 ACL leakage (`2/2` negative conversations declined in all 6 runs)**.
3. **Why `enforced=on` vs `enforced=off` Converges on `GLM-5.3-Flash-Event`**: Because `GLM-5.3-Flash-Event` almost always issues a fresh `search_knowledge` tool call on Turn 2 even without the `enforced=on` prompt/loop guard, the behavioral gap between `enforced=off` (`7.67/8` searches) and `enforced=on` (`8.00/8` searches) is narrow; however, `enforced=on` remains an essential deterministic safety guarantee against silent memory-only answers on revoked documents.
