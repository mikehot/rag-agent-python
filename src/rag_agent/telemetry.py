"""Minimal OpenTelemetry tracing.

Code paths use only the OpenTelemetry *API*, which does nothing until a provider is configured, so
tracing costs nothing while it is off (the default). ``RAG_AGENT_TRACING=console`` prints spans to
standard output; there is deliberately no network exporter in this version.

What a span may carry is decided here, once. Spans describe *behaviour*: durations, counts,
outcomes, model name, finish reason, token counts. They never carry the question, the answer, tool
arguments, retrieved text, a bearer token, a username, or a session id: each of those is user data
or a secret, and a trace backend is rarely held to the same access rules as the knowledge base.
"""

from __future__ import annotations

import sys
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

_NAME = "rag_agent"
_tracer: Tracer = trace.get_tracer(_NAME)

# The only attribute keys any span in this package sets. Tests assert that nothing else appears.
ALLOWED_ATTRIBUTES = frozenset(
    {
        "agent.engine",
        "agent.policy",
        "agent.streaming",
        "agent.has_history",
        "agent.stop_reason",
        "agent.turns",
        "agent.tool_calls",
        "agent.retrieved",
        "agent.cited_documents",
        "llm.model",
        "llm.streaming",
        "llm.finish_reason",
        "llm.completion_tokens",
        "llm.tool_calls",
        "tool.name",
        "tool.outcome",
        "tool.is_error",
    }
)


def tracer() -> Tracer:
    return _tracer


def use_provider(provider: Any) -> None:
    """Route this package's spans to ``provider`` (used by setup and by tests)."""
    global _tracer
    _tracer = provider.get_tracer(_NAME)


def reset() -> None:
    global _tracer
    _tracer = trace.get_tracer(_NAME)


def configure(mode: str) -> None:
    """``off`` leaves tracing inert; ``console`` prints finished spans to standard output."""
    if mode == "off":
        return
    if mode != "console":
        raise ValueError(f"unknown tracing mode {mode!r}; expected 'off' or 'console'")
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "rag-agent-python"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stdout)))
    use_provider(provider)


def set_attributes(span: Any, **attributes: Any) -> None:
    for key, value in attributes.items():
        name = key.replace("__", ".")
        if name not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"span attribute {name!r} is not on the allow-list")
        span.set_attribute(name, value)
