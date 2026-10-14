"""The bounded tool-calling agent expressed as a LangGraph ``StateGraph``.

Same behaviour as :func:`rag_agent.agent.run_agent`, different control structure: two nodes
(``model`` and ``tools``) and conditional edges instead of a ``for`` loop. Tool validation and
execution are the *same function* as in the loop version (``execute_call``), and the loop version is
kept as the reference implementation: ``tests/test_graph.py`` runs both on the same scripted
scenarios and requires identical results.

Only LangGraph's orchestration is used. Model calls go through this package's own ``ChatClient`` and
tool calls through ``McpClient``; no LangChain model wrappers and no tracing (LangSmith tracing is
explicitly left off).
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

# langsmith ships with langgraph and is a telemetry client; keep it off unless a deployer opts in.
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from langgraph.graph import END, START, StateGraph

from rag_agent.agent import (
    EMPTY_REPLY_NUDGE,
    NO_ANSWER,
    AgentResult,
    ChatBackend,
    Limits,
    ToolBackend,
    ToolTrace,
    assistant_message,
    clean_answer,
    execute_call,
    system_prompt,
    to_openai_tools,
)
from rag_agent.llm import LlmError, ToolCall


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    pending: list[ToolCall]
    turns: int
    executed: int
    nudged: bool
    answer: str
    stop_reason: str
    done: bool
    trace: list[ToolTrace]
    retrieved: list[dict[str, str]]


def build_graph(tools_backend: ToolBackend, chat: ChatBackend, limits: Limits):
    """Compile the graph for one knowledge-base connection. The tool list is read once, here."""
    tools = tools_backend.list_tools()
    by_name = {tool.name: tool for tool in tools}
    openai_tools = to_openai_tools(tools)

    def model(state: AgentState) -> dict[str, Any]:
        if state["turns"] >= limits.max_turns:
            return {"done": True, "stop_reason": "turn_limit"}
        turns = state["turns"] + 1
        try:
            turn = chat.complete(state["messages"], openai_tools)
        except LlmError:
            return {"turns": turns, "done": True, "stop_reason": "model_error"}
        if turn.tool_calls:
            return {
                "turns": turns,
                "messages": [*state["messages"], assistant_message(turn)],
                "pending": list(turn.tool_calls),
            }
        text = clean_answer(turn.content or "")
        if text:
            return {"turns": turns, "done": True, "answer": text, "stop_reason": "answered"}
        if state["nudged"] or not state["trace"] or not limits.retry_empty_reply:
            return {"turns": turns, "done": True, "stop_reason": "empty_answer"}
        # One changed retry, only after tool results were seen.
        return {
            "turns": turns,
            "nudged": True,
            "stop_reason": "empty_answer",
            "messages": [*state["messages"], {"role": "user", "content": EMPTY_REPLY_NUDGE}],
        }

    def run_tools(state: AgentState) -> dict[str, Any]:
        scratch = AgentResult(
            answer=NO_ANSWER,
            stop_reason=state["stop_reason"],
            trace=list(state["trace"]),
            retrieved=list(state["retrieved"]),
        )
        messages = list(state["messages"])
        executed = state["executed"]
        for call in state["pending"]:
            if executed >= limits.max_tool_calls:
                return {
                    "done": True,
                    "stop_reason": "tool_budget",
                    "pending": [],
                    "messages": messages,
                    "executed": executed,
                    "trace": scratch.trace,
                    "retrieved": scratch.retrieved,
                }
            executed += execute_call(call, by_name, tools_backend, scratch, messages)
        return {
            "pending": [],
            "messages": messages,
            "executed": executed,
            "trace": scratch.trace,
            "retrieved": scratch.retrieved,
        }

    def after_model(state: AgentState) -> str:
        if state["done"]:
            return END
        return "tools" if state["pending"] else "model"

    def after_tools(state: AgentState) -> str:
        return END if state["done"] else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", after_model, {END: END, "tools": "tools", "model": "model"}
    )
    graph.add_conditional_edges("tools", after_tools, {END: END, "model": "model"})
    return graph.compile()


def run_graph_agent(
    question: str,
    tools_backend: ToolBackend,
    chat: ChatBackend,
    limits: Limits | None = None,
    policy: str = "single",
) -> AgentResult:
    """Drop-in replacement for :func:`rag_agent.agent.run_agent`."""
    limits = limits or Limits()
    compiled = build_graph(tools_backend, chat, limits)
    initial: AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt(policy)},
            {"role": "user", "content": question},
        ],
        "pending": [],
        "turns": 0,
        "executed": 0,
        "nudged": False,
        "answer": NO_ANSWER,
        "stop_reason": "turn_limit",
        "done": False,
        "trace": [],
        "retrieved": [],
    }
    # Each model turn is at most two supersteps (model, tools); leave headroom for the retry.
    final = compiled.invoke(initial, {"recursion_limit": 2 * limits.max_turns + 6})
    return AgentResult(
        answer=final["answer"],
        stop_reason=final["stop_reason"],
        turns=final["turns"],
        trace=list(final["trace"]),
        retrieved=list(final["retrieved"]),
    )
