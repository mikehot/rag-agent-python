import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from rag_agent import __version__
from rag_agent.config import Settings
from rag_agent.llm import AssistantTurn, LlmError, ToolCall
from rag_agent.main import bearer_token, create_app
from rag_agent.mcp_client import McpError, Tool, ToolResult

TOKEN = "caller-token-value"
SEARCH = Tool(
    "search_knowledge",
    "Search",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


class FakeKb:
    def __init__(self, token: str, *, hit: bool = True, fail: McpError | None = None) -> None:
        self.token, self.hit, self.fail = token, hit, fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "FakeKb":
        if self.fail:
            raise self.fail
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def list_tools(self) -> list[Tool]:
        return [SEARCH]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        self.calls.append((name, arguments or {}))
        sources = (
            [{"filename": "k8s-pods-pod-lifecycle.md", "locator": "chunk#10", "snippet": "x"}]
            if self.hit
            else []
        )
        return ToolResult(False, "ok", {"found": self.hit, "sources": sources})


class Scripted:
    def __init__(self, *turns: AssistantTurn | Exception) -> None:
        self.turns = list(turns)

    def complete(self, messages, tools) -> AssistantTurn:
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def search_then(text: str) -> Scripted:
    call = ToolCall(
        "c1", "search_knowledge", json.dumps({"query": "Pod 重启策略"}, ensure_ascii=False)
    )
    return Scripted(AssistantTurn(None, [call]), AssistantTurn(text))


def build(
    kb: FakeKb | None = None, chat=None, **settings: Any
) -> tuple[TestClient, dict[str, Any]]:
    seen: dict[str, Any] = {}

    def factory(base_url: str, token: str, timeout: float) -> FakeKb:
        seen.update(base_url=base_url, token=token, timeout=timeout)
        return kb or FakeKb(token)

    settings = Settings(kb_base_url="http://kb.invalid", **settings)
    return TestClient(create_app(settings, kb_factory=factory, chat=chat)), seen


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_health_reports_service_and_version() -> None:
    client, _ = build()
    assert client.get("/health").json() == {
        "status": "ok",
        "service": "rag-agent-python",
        "version": __version__,
    }


def test_ask_requires_a_bearer_token() -> None:
    client, _ = build(chat=Scripted())
    assert client.post("/ask", json={"question": "q"}).status_code == 401
    assert (
        client.post(
            "/ask", json={"question": "q"}, headers={"Authorization": "Basic x"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/ask", json={"question": "q"}, headers={"Authorization": "Bearer "}
        ).status_code
        == 401
    )


def test_ask_forwards_the_callers_token_unchanged_and_returns_a_traced_answer() -> None:
    kb = FakeKb(TOKEN)
    client, seen = build(kb, search_then("依据文档：k8s-pods-pod-lifecycle.md，取值为 Always。"))
    response = client.post("/ask", json={"question": "Pod 的重启策略？"}, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert seen["token"] == TOKEN and seen["base_url"] == "http://kb.invalid"
    assert body["answered"] is True and body["stop_reason"] == "answered" and body["turns"] == 2
    assert body["tool_calls"] == [{"name": "search_knowledge", "outcome": "ok"}]
    assert body["retrieved"] == [{"filename": "k8s-pods-pod-lifecycle.md", "locator": "chunk#10"}]
    assert body["cited_documents"] == ["k8s-pods-pod-lifecycle.md"]
    assert TOKEN not in response.text


def test_retrieved_documents_the_answer_does_not_name_are_not_reported_as_cited() -> None:
    client, _ = build(chat=search_then("知识库中没有相关内容。"))
    body = client.post("/ask", json={"question": "试用期？"}, headers=AUTH).json()
    assert body["retrieved"] and body["cited_documents"] == []


def test_token_rejected_by_the_knowledge_base_is_a_401() -> None:
    kb = FakeKb(TOKEN, fail=McpError("tools/list: HTTP 401", status=401))
    client, _ = build(kb, chat=Scripted())
    response = client.post("/ask", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 401 and response.json()["error"] == "unauthorized"


def test_unreachable_knowledge_base_is_a_502() -> None:
    kb = FakeKb(TOKEN, fail=McpError("transport error"))
    client, _ = build(kb, chat=Scripted())
    response = client.post("/ask", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"] == "knowledge_base_unavailable"


def test_model_failure_is_a_502_and_not_an_invented_answer() -> None:
    client, _ = build(chat=Scripted(LlmError("down")))
    response = client.post("/ask", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 502 and response.json()["error"] == "model_unavailable"
    assert "answer" not in response.json()


def test_a_run_that_hits_its_limits_is_200_with_answered_false() -> None:
    call = ToolCall("c", "search_knowledge", json.dumps({"query": "q"}))
    chat = Scripted(*[AssistantTurn(None, [call]) for _ in range(10)])
    client, _ = build(chat=chat, max_turns=2, max_tool_calls=1)
    body = client.post("/ask", json={"question": "q"}, headers=AUTH).json()
    assert body["answered"] is False and body["stop_reason"] in {"turn_limit", "tool_budget"}


def test_empty_and_overlong_questions_are_rejected() -> None:
    client, _ = build(chat=Scripted())
    assert client.post("/ask", json={"question": ""}, headers=AUTH).status_code == 422
    assert client.post("/ask", json={"question": "a" * 2001}, headers=AUTH).status_code == 422


def test_bearer_token_parsing() -> None:
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer abc") == "abc"
    assert bearer_token("Bearer") is None and bearer_token(None) is None


def test_settings_read_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AGENT_KB_BASE_URL", "http://example.invalid:9")
    monkeypatch.setenv("RAG_AGENT_MAX_TURNS", "6")
    s = Settings()
    assert s.kb_base_url == "http://example.invalid:9" and s.max_turns == 6


def test_httpx_is_importable_for_the_default_clients() -> None:
    assert httpx.__version__


def test_the_graph_engine_serves_the_same_answer() -> None:
    kb = FakeKb(TOKEN)
    client, _ = build(kb, search_then("依据文档：k8s-pods-pod-lifecycle.md。"), engine="graph")
    body = client.post("/ask", json={"question": "Pod 的重启策略？"}, headers=AUTH).json()
    assert body["answered"] is True and body["turns"] == 2
    assert body["tool_calls"] == [{"name": "search_knowledge", "outcome": "ok"}]
    assert body["cited_documents"] == ["k8s-pods-pod-lifecycle.md"]


def test_engine_defaults_to_the_reference_loop() -> None:
    assert Settings().engine == "loop"


def test_policy_setting_reaches_the_model_and_defaults_to_single() -> None:
    assert Settings().policy == "single"
    seen: list[str] = []

    class Spy(Scripted):
        def complete(self, messages, tools) -> AssistantTurn:
            seen.append(messages[0]["content"])
            return super().complete(messages, tools)

    client, _ = build(chat=Spy(AssistantTurn("好的")), policy="decompose")
    assert client.post("/ask", json={"question": "q"}, headers=AUTH).status_code == 200
    assert "拆成彼此独立" in seen[0]
