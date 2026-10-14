import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_agent import telemetry
from rag_agent.llm import AssistantTurn, ToolCall
from test_app import AUTH, TOKEN, FakeKb, Scripted, build


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry.use_provider(provider)
    yield exporter
    telemetry.reset()


def search_then(text: str) -> Scripted:
    call = ToolCall(
        "c", "search_knowledge", json.dumps({"query": "绝密的问题文本"}, ensure_ascii=False)
    )
    return Scripted(AssistantTurn(None, [call]), AssistantTurn(text))


def by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    out: dict[str, list] = {}
    for span in exporter.get_finished_spans():
        out.setdefault(span.name, []).append(span)
    return out


def test_a_request_produces_a_root_span_with_model_and_tool_children(spans) -> None:
    client, _ = build(FakeKb(TOKEN), search_then("依据文档：k8s-pods-pod-lifecycle.md"))
    assert client.post("/ask", json={"question": "绝密的问题文本"}, headers=AUTH).status_code == 200
    named = by_name(spans)
    assert set(named) == {"agent.ask", "llm.chat", "tool.call"}
    root = named["agent.ask"][0]
    assert len(named["llm.chat"]) == 2 and len(named["tool.call"]) == 1
    assert all(
        s.parent.span_id == root.context.span_id for s in named["llm.chat"] + named["tool.call"]
    )
    assert root.attributes["agent.stop_reason"] == "answered"
    assert root.attributes["agent.turns"] == 2
    assert root.attributes["agent.tool_calls"] == 1
    assert root.attributes["agent.cited_documents"] == 1
    assert named["tool.call"][0].attributes["tool.name"] == "search_knowledge"
    assert named["tool.call"][0].attributes["tool.outcome"] == "ok"


def test_spans_never_carry_user_text_answers_tokens_or_session_ids(spans) -> None:
    client, _ = build(FakeKb(TOKEN), search_then("机密答案正文 k8s-pods-pod-lifecycle.md"))
    sid = client.post(
        "/ask", json={"question": "绝密的问题文本", "remember": True}, headers=AUTH
    ).json()["session_id"]
    everything = json.dumps(
        [dict(s.attributes) | {"name": s.name} for s in spans.get_finished_spans()],
        ensure_ascii=False,
    )
    for secret in ("绝密的问题文本", "机密答案正文", TOKEN, sid):
        assert secret not in everything
    for span in spans.get_finished_spans():
        assert set(span.attributes) <= telemetry.ALLOWED_ATTRIBUTES


def test_setting_an_attribute_that_is_not_on_the_allow_list_is_an_error(spans) -> None:
    with telemetry.tracer().start_as_current_span("x") as span, pytest.raises(ValueError):
        telemetry.set_attributes(span, tool__arguments="{}")


def test_the_streaming_path_is_traced_from_the_worker_thread(spans) -> None:
    client, _ = build(FakeKb(TOKEN), search_then("好的"))
    response = client.post("/ask/stream", json={"question": "q"}, headers=AUTH)
    assert response.status_code == 200
    root = by_name(spans)["agent.ask"][0]
    assert root.attributes["agent.streaming"] is True


def test_tracing_is_inert_by_default() -> None:
    from rag_agent.config import Settings

    assert Settings().tracing == "off"
    telemetry.reset()
    client, _ = build(FakeKb(TOKEN), search_then("好的"))  # must work with no provider at all
    assert client.post("/ask", json={"question": "q"}, headers=AUTH).status_code == 200


def test_console_mode_prints_spans_and_unknown_modes_are_rejected(capsys) -> None:
    try:
        telemetry.configure("console")
        client, _ = build(FakeKb(TOKEN), search_then("好的"))
        client.post("/ask", json={"question": "绝密的问题文本"}, headers=AUTH)
        printed = capsys.readouterr().out
        assert '"name": "agent.ask"' in printed
        assert "绝密的问题文本" not in printed and TOKEN not in printed
    finally:
        telemetry.reset()
    with pytest.raises(ValueError):
        telemetry.configure("otlp")


def test_cited_documents_counts_files_not_chunks() -> None:
    from rag_agent.agent import AgentResult
    from rag_agent.main import cited_filenames

    result = AgentResult(
        answer="依据 a.md 和 b.md",
        stop_reason="answered",
        retrieved=[
            {"filename": "a.md", "locator": "chunk#1"},
            {"filename": "a.md", "locator": "chunk#2"},
            {"filename": "b.md", "locator": "chunk#9"},
            {"filename": "c.md", "locator": "chunk#3"},
        ],
    )
    assert cited_filenames(result) == ["a.md", "b.md"]
