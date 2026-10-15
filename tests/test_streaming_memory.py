import json
from typing import Any

import httpx
import pytest

from rag_agent.agent import FOLLOWUP_ADDENDUM, initial_messages, run_agent
from rag_agent.llm import AssistantTurn, ChatClient, MarkerFilter, ToolCall
from rag_agent.sessions import SessionStore
from test_app import AUTH, TOKEN, FakeKb, Scripted, build

# ---------------------------------------------------------------- marker filter and stream parsing


def test_marker_filter_removes_a_marker_split_across_chunks() -> None:
    f = MarkerFilter()
    out = f.feed("<|chan") + f.feed("nel>你好") + f.feed("，世界") + f.flush()
    assert out == "你好，世界"


def test_marker_filter_releases_text_that_only_looked_like_a_marker() -> None:
    f = MarkerFilter()
    assert f.feed("a < b") + f.feed(" and c") + f.flush() == "a < b and c"
    f = MarkerFilter()
    assert f.feed("x <") + f.feed("y") + f.flush() == "x <y"


def sse_body(chunks: list[dict[str, Any]]) -> str:
    return (
        "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
        + "data: [DONE]\n\n"
    )


def stream_client(body: str, status: int = 200) -> ChatClient:
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text=body))
    return ChatClient("http://llm.invalid/v1", "m", transport=transport)


def test_complete_stream_forwards_clean_deltas_and_returns_the_whole_turn() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "<|channel>"}}]},
        {"choices": [{"delta": {"content": "答案是"}}]},
        {"choices": [{"delta": {"content": "六个月"}, "finish_reason": "stop"}]},
    ]
    seen: list[str] = []
    turn = stream_client(sse_body(chunks)).complete_stream([], [], seen.append)
    assert "".join(seen) == "答案是六个月"
    assert turn.finish_reason == "stop" and not turn.tool_calls


def test_complete_stream_assembles_tool_calls_from_fragments() -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "search_knowledge", "arguments": '{"que'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'ry": "试用期"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    turn = stream_client(sse_body(chunks)).complete_stream([], [], lambda t: None)
    assert [(c.id, c.name, c.arguments_raw) for c in turn.tool_calls] == [
        ("c1", "search_knowledge", '{"query": "试用期"}')
    ]


def test_complete_stream_reports_http_failure_as_an_llm_error() -> None:
    from rag_agent.llm import LlmError

    with pytest.raises(LlmError):
        stream_client("", status=503).complete_stream([], [], lambda t: None)


# ------------------------------------------------------------------------------- conversation


def test_history_is_questions_and_answers_only_and_adds_the_followup_rule() -> None:
    msgs = initial_messages("那默认值呢？", "single", [("Pod 重启策略？", "有三种。")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert FOLLOWUP_ADDENDUM in msgs[0]["content"]
    assert msgs[-1]["content"] == "那默认值呢？"
    assert all("tool_calls" not in m for m in msgs)


def test_no_history_means_no_followup_rule() -> None:
    msgs = initial_messages("q", "single", None)
    assert FOLLOWUP_ADDENDUM not in msgs[0]["content"]
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_agent_streams_events_for_tool_calls_and_content() -> None:
    class StreamingChat:
        def __init__(self) -> None:
            self.turns = [
                AssistantTurn(None, [ToolCall("c", "search_knowledge", '{"query": "q"}')]),
                AssistantTurn("最终答案"),
            ]

        def complete(self, messages, tools):  # not used when streaming is available
            raise AssertionError("complete() must not be used when complete_stream exists")

        def complete_stream(self, messages, tools, on_delta):
            turn = self.turns.pop(0)
            if turn.content:
                on_delta(turn.content)
            return turn

    events: list[dict[str, Any]] = []
    result = run_agent("q", FakeKb(TOKEN), StreamingChat(), on_event=events.append)
    assert result.answer == "最终答案"
    assert [e["type"] for e in events] == ["tool_call", "delta"]
    assert events[0] == {"type": "tool_call", "name": "search_knowledge", "outcome": "ok"}


# --------------------------------------------------------------------------- session store


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_session_is_bound_to_the_token_that_created_it() -> None:
    store = SessionStore()
    sid = store.create("token-a")
    store.append(sid, "token-a", "q1", "a1")
    assert store.history(sid, "token-a") == [("q1", "a1")]
    assert store.history(sid, "token-b") is None  # same as unknown
    assert store.history("nope", "token-a") is None
    store.append(sid, "token-b", "intruder", "x")  # ignored
    assert store.history(sid, "token-a") == [("q1", "a1")]
    assert store.forget(sid, "token-b") is False
    assert store.forget(sid, "token-a") is True


def test_the_store_does_not_keep_the_token() -> None:
    store = SessionStore()
    store.create("super-secret-token-value")
    assert "super-secret-token-value" not in repr(vars(store))
    assert all("super-secret-token-value" not in repr(s) for s in store._sessions.values())


def test_sessions_expire_and_are_bounded() -> None:
    clock = Clock()
    store = SessionStore(
        ttl_seconds=60, max_sessions=2, max_turns=2, max_answer_chars=5, clock=clock
    )
    sid = store.create("t")
    clock.now = 61
    assert store.history(sid, "t") is None
    a = store.create("t")
    clock.now += 1
    b = store.create("t")
    clock.now += 1
    c = store.create("t")  # evicts the oldest
    assert len(store) == 2 and store.history(a, "t") is None
    assert store.history(b, "t") == [] and store.history(c, "t") == []
    for i in range(4):
        store.append(c, "t", f"q{i}", "0123456789")
    assert store.history(c, "t") == [("q2", "01234"), ("q3", "01234")]


# ------------------------------------------------------------------------------------- HTTP


class Recorder(Scripted):
    def __init__(self, *turns: AssistantTurn) -> None:
        super().__init__(*turns)
        self.system_prompts: list[str] = []
        self.conversations: list[list[str]] = []

    def complete(self, messages, tools):
        self.system_prompts.append(messages[0]["content"])
        self.conversations.append([m["content"] for m in messages[1:] if m["role"] != "tool"])
        return super().complete(messages, tools)


def test_nothing_is_remembered_unless_asked() -> None:
    client, _ = build(chat=Scripted(AssistantTurn("好的")))
    body = client.post("/ask", json={"question": "q"}, headers=AUTH).json()
    assert body["session_id"] is None
    assert len(client.app.state.sessions) == 0


def test_a_remembered_conversation_carries_earlier_turns_to_the_model() -> None:
    search = ToolCall("c", "search_knowledge", json.dumps({"query": "默认值"}))
    chat = Recorder(
        AssistantTurn("有三种。"),
        AssistantTurn(None, [search]),
        AssistantTurn("默认是 Always。"),
    )
    client, _ = build(chat=chat)
    first = client.post("/ask", json={"question": "Pod 重启策略？", "remember": True}, headers=AUTH)
    sid = first.json()["session_id"]
    assert sid
    second = client.post("/ask", json={"question": "那默认值呢？", "session_id": sid}, headers=AUTH)
    assert second.status_code == 200 and second.json()["session_id"] == sid
    assert chat.conversations[1] == ["Pod 重启策略？", "有三种。", "那默认值呢？"]
    assert FOLLOWUP_ADDENDUM in chat.system_prompts[1]
    assert FOLLOWUP_ADDENDUM not in chat.system_prompts[0]


def test_someone_elses_session_looks_like_an_unknown_one() -> None:
    chat = Scripted(AssistantTurn("一"), AssistantTurn("二"))
    client, _ = build(chat=chat)
    sid = client.post("/ask", json={"question": "q", "remember": True}, headers=AUTH).json()[
        "session_id"
    ]
    other = {"Authorization": "Bearer another-users-token"}
    stolen = client.post("/ask", json={"question": "q2", "session_id": sid}, headers=other)
    missing = client.post("/ask", json={"question": "q2", "session_id": "no-such"}, headers=other)
    assert stolen.status_code == missing.status_code == 404
    assert stolen.json() == missing.json()
    assert client.delete(f"/sessions/{sid}", headers=other).status_code == 404
    assert client.delete(f"/sessions/{sid}", headers=AUTH).status_code == 204
    assert client.delete(f"/sessions/{sid}", headers=AUTH).status_code == 404


def test_a_failed_turn_is_not_remembered() -> None:
    from rag_agent.llm import LlmError

    chat = Scripted(AssistantTurn("好的"), LlmError("down"))
    client, _ = build(chat=chat)
    sid = client.post("/ask", json={"question": "q1", "remember": True}, headers=AUTH).json()[
        "session_id"
    ]
    assert (
        client.post("/ask", json={"question": "q2", "session_id": sid}, headers=AUTH).status_code
        == 502
    )
    assert client.app.state.sessions.history(sid, TOKEN) == [("q1", "好的")]


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_emits_steps_then_the_authoritative_answer() -> None:
    call = ToolCall("c", "search_knowledge", json.dumps({"query": "q"}))
    chat = Scripted(
        AssistantTurn(None, [call]), AssistantTurn("依据文档：k8s-pods-pod-lifecycle.md")
    )
    client, _ = build(FakeKb(TOKEN), chat)
    response = client.post("/ask/stream", json={"question": "q", "remember": True}, headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "session" and kinds[-1] == "done" and kinds[-2] == "answer"
    assert ("tool_call", {"name": "search_knowledge", "outcome": "ok"}) in events
    answer = dict(events)["answer"]
    assert answer["answered"] is True and answer["cited_documents"] == ["k8s-pods-pod-lifecycle.md"]
    assert answer["session_id"] == dict(events)["session"]["session_id"]
    assert TOKEN not in response.text


def test_stream_errors_before_streaming_are_ordinary_http_errors() -> None:
    client, _ = build(chat=Scripted())
    assert client.post("/ask/stream", json={"question": "q"}).status_code == 401
    assert (
        client.post(
            "/ask/stream", json={"question": "q", "session_id": "x"}, headers=AUTH
        ).status_code
        == 404
    )


def test_stream_reports_a_late_failure_as_an_error_event() -> None:
    from rag_agent.mcp_client import McpError

    client, _ = build(FakeKb(TOKEN, fail=McpError("down")), Scripted())
    events = parse_sse(client.post("/ask/stream", json={"question": "q"}, headers=AUTH).text)
    assert events[0][0] == "error" and events[0][1]["error"] == "knowledge_base_unavailable"
    assert events[-1][0] == "done"
