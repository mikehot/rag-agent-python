"""The LangGraph version must behave exactly like the reference loop on the same scenarios."""

import json
from typing import Any

import pytest

from rag_agent.agent import NO_RETRIEVAL_ANSWER, Limits, run_agent
from rag_agent.graph import build_graph, run_graph_agent
from rag_agent.llm import AssistantTurn, LlmError, ToolCall
from rag_agent.mcp_client import McpError, ToolResult
from test_agent import FakeTools, ScriptedChat, call, say

HIT = ToolResult(
    False,
    "ok",
    {"found": True, "sources": [{"filename": "a.md", "locator": "chunk#1", "snippet": "x"}]},
)
REFUSED = ToolResult(True, "denied", None)


def search(i: int = 0) -> AssistantTurn:
    return call("search_knowledge", {"query": "q"}, f"c{i}")


# name -> (script factory, tools factory, limits)
SCENARIOS: dict[str, tuple[Any, Any, Limits]] = {
    "search_then_answer": (
        lambda: [search(), say("答案（来自 a.md）")],
        lambda: FakeTools({"search_knowledge": HIT}),
        Limits(),
    ),
    "answer_without_tools": (lambda: [say("你好")], lambda: FakeTools(), Limits()),
    "unknown_tool": (lambda: [call("delete_document", {}), say("不能")], FakeTools, Limits()),
    "identity_argument": (
        lambda: [call("list_documents", {"tenantId": "t"}), say("ok")],
        FakeTools,
        Limits(),
    ),
    "malformed_json": (
        lambda: [call("search_knowledge", "{not json"), say("ok")],
        FakeTools,
        Limits(),
    ),
    "tool_refusal": (
        lambda: [search(), say("没有权限")],
        lambda: FakeTools({"search_knowledge": REFUSED}),
        Limits(),
    ),
    "transport_error": (
        lambda: [search(), say("暂时不可用")],
        lambda: FakeTools({"search_knowledge": McpError("down")}),
        Limits(),
    ),
    "tool_budget": (
        lambda: [search(i) for i in range(10)],
        FakeTools,
        Limits(max_turns=10, max_tool_calls=2),
    ),
    "turn_limit": (
        lambda: [call("nope", {}, f"c{i}") for i in range(10)],
        FakeTools,
        Limits(max_turns=3),
    ),
    "model_error": (lambda: [LlmError("boom")], FakeTools, Limits()),
    "empty_answer_no_tools": (lambda: [say("  ")], FakeTools, Limits()),
    "empty_reply_then_retry_succeeds": (
        lambda: [search(), say(""), say("最终答案")],
        FakeTools,
        Limits(),
    ),
    "empty_reply_twice": (lambda: [search(), say(""), say("")], FakeTools, Limits()),
    "empty_reply_retry_switched_off": (
        lambda: [search(), say("")],
        FakeTools,
        Limits(retry_empty_reply=False),
    ),
    "control_marker_is_stripped": (lambda: [say("<|channel>最终答案")], FakeTools, Limits()),
    "two_calls_in_one_turn": (
        lambda: [
            AssistantTurn(
                None,
                [
                    ToolCall("a", "search_knowledge", json.dumps({"query": "q1"})),
                    ToolCall("b", "search_knowledge", json.dumps({"query": "q2"})),
                ],
            ),
            say("两部分都有"),
        ],
        lambda: FakeTools({"search_knowledge": HIT}),
        Limits(),
    ),
}


def snapshot(result) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "stop_reason": result.stop_reason,
        "turns": result.turns,
        "trace": [(t.name, t.arguments, t.outcome, t.is_error) for t in result.trace],
        "retrieved": result.retrieved,
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_graph_matches_the_reference_loop(name: str) -> None:
    script, tools, limits = SCENARIOS[name]
    loop_tools, graph_tools = tools(), tools()
    loop_chat, graph_chat = ScriptedChat(*script()), ScriptedChat(*script())
    expected = run_agent("问题", loop_tools, loop_chat, limits)
    actual = run_graph_agent("问题", graph_tools, graph_chat, limits)
    assert snapshot(actual) == snapshot(expected)
    # the backend must also have seen the same calls, in the same order
    assert graph_tools.calls == loop_tools.calls
    # and the model must have been shown the same conversation
    assert graph_chat.seen == loop_chat.seen


HISTORY = [("上一个问题", "上一个答案")]

# Conversations with earlier turns: a final reply needs a successful retrieval in this turn.
HISTORY_SCENARIOS: dict[str, tuple[Any, Any, Limits]] = {
    "search_then_answer": (
        lambda: [search(), say("答案（来自 a.md）")],
        lambda: FakeTools({"search_knowledge": HIT}),
        Limits(),
    ),
    "answer_from_history_then_nudged_to_search": (
        lambda: [say("上一个答案里有"), search(), say("检索后的答案")],
        lambda: FakeTools({"search_knowledge": HIT}),
        Limits(),
    ),
    "never_retrieves": (
        lambda: [say("上一个答案里有"), say("还是上一个答案里有")],
        FakeTools,
        Limits(),
    ),
    "empty_reply_without_retrieval": (lambda: [say(""), say("")], FakeTools, Limits()),
    "refused_retrieval_is_not_retrieval": (
        lambda: [search(), say("上一个答案里有"), say("上一个答案里有")],
        lambda: FakeTools({"search_knowledge": REFUSED}),
        Limits(),
    ),
    "enforcement_switched_off": (
        lambda: [say("上一个答案里有")],
        FakeTools,
        Limits(enforce_retrieval=False),
    ),
}


@pytest.mark.parametrize("name", sorted(HISTORY_SCENARIOS))
def test_graph_matches_the_reference_loop_with_history(name: str) -> None:
    script, tools, limits = HISTORY_SCENARIOS[name]
    loop_tools, graph_tools = tools(), tools()
    loop_chat, graph_chat = ScriptedChat(*script()), ScriptedChat(*script())
    expected = run_agent("问题", loop_tools, loop_chat, limits, history=HISTORY)
    actual = run_graph_agent("问题", graph_tools, graph_chat, limits, history=HISTORY)
    assert snapshot(actual) == snapshot(expected)
    assert graph_tools.calls == loop_tools.calls
    assert graph_chat.seen == loop_chat.seen
    if name in {
        "never_retrieves",
        "empty_reply_without_retrieval",
        "refused_retrieval_is_not_retrieval",
    }:
        assert actual.stop_reason == "no_retrieval" and actual.answer == NO_RETRIEVAL_ANSWER


def test_policy_reaches_the_graphs_system_message() -> None:
    chat = ScriptedChat(say("ok"))
    run_graph_agent("q", FakeTools(), chat, policy="decompose")
    assert "拆成彼此独立" in chat.seen[0][0]["content"]


def test_the_compiled_graph_has_exactly_two_nodes() -> None:
    graph = build_graph(FakeTools(), ScriptedChat(), Limits()).get_graph()
    assert {n for n in graph.nodes if not n.startswith("__")} == {"model", "tools"}


def test_langsmith_tracing_is_off_by_default() -> None:
    import os

    assert os.environ.get("LANGSMITH_TRACING") == "false"
