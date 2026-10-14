from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from fastapi import FastAPI, Header, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_agent import __version__, telemetry
from rag_agent.agent import AgentResult, ChatBackend, Limits, ToolBackend, run_agent
from rag_agent.config import Settings, unknown_settings
from rag_agent.graph import run_graph_agent
from rag_agent.llm import ChatClient
from rag_agent.mcp_client import McpClient, McpError
from rag_agent.sessions import SessionStore

log = logging.getLogger("rag_agent")

KbFactory = Callable[[str, str, float], AbstractContextManager[ToolBackend]]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Continue an earlier conversation. Unknown, expired and someone else's ids look the same.
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    # Start a remembered conversation and return its id. Nothing is remembered unless asked.
    remember: bool = False


class ToolCallInfo(BaseModel):
    name: str
    outcome: str


class DocumentRef(BaseModel):
    filename: str
    locator: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    stop_reason: str
    turns: int
    tool_calls: list[ToolCallInfo]
    # Candidates the knowledge base returned. They are NOT citations: an answer may conclude that
    # none of them helps. `cited_documents` is the lexical subset the answer text actually names.
    retrieved: list[DocumentRef]
    cited_documents: list[str]
    session_id: str | None = None


def default_kb_factory(base_url: str, token: str, timeout: float) -> McpClient:
    return McpClient(base_url, token, timeout=timeout)


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


def cited_filenames(result: AgentResult) -> list[str]:
    """File names of retrieved documents that the answer text names, one entry per file."""
    named = [r["filename"] for r in result.retrieved if r["filename"] in result.answer]
    return list(dict.fromkeys(named))


def to_response(result: AgentResult, session_id: str | None = None) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        answered=result.stop_reason == "answered",
        stop_reason=result.stop_reason,
        turns=result.turns,
        tool_calls=[ToolCallInfo(name=t.name, outcome=t.outcome) for t in result.trace],
        retrieved=[DocumentRef(**r) for r in result.retrieved],
        cited_documents=cited_filenames(result),
        session_id=session_id,
    )


def error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def create_app(
    settings: Settings | None = None,
    kb_factory: KbFactory | None = None,
    chat: ChatBackend | None = None,
    sessions: SessionStore | None = None,
) -> FastAPI:
    settings = settings or Settings()
    telemetry.configure(settings.tracing)
    for name in unknown_settings(dict(os.environ)):
        log.warning("ignoring unknown setting %s (not supported by this version)", name)
    kb_factory = kb_factory or default_kb_factory
    chat = chat or ChatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    sessions = sessions or SessionStore(
        ttl_seconds=settings.session_ttl_seconds, max_sessions=settings.max_sessions
    )
    limits = Limits(max_turns=settings.max_turns, max_tool_calls=settings.max_tool_calls)
    runner = run_graph_agent if settings.engine == "graph" else run_agent
    app = FastAPI(title="rag-agent-python", version=__version__)
    app.state.settings = settings
    app.state.sessions = sessions

    def prepare(request: AskRequest, authorization: str | None):
        """Authenticate and resolve the conversation. Returns (token, session_id, history) or an
        error response."""
        token = bearer_token(authorization)
        if token is None:
            return error(
                401, "unauthorized", "send the knowledge-base bearer token as 'Authorization'"
            )
        session_id: str | None = None
        history = None
        if request.session_id is not None:
            history = sessions.history(request.session_id, token)
            if history is None:
                return error(404, "session_not_found", "unknown or expired session")
            session_id = request.session_id
        elif request.remember:
            session_id = sessions.create(token)
            history = []
        return token, session_id, history

    def execute(request: AskRequest, token: str, history, on_event=None) -> AgentResult:
        with telemetry.tracer().start_as_current_span("agent.ask") as span:
            telemetry.set_attributes(
                span,
                agent__engine=settings.engine,
                agent__policy=settings.policy,
                agent__streaming=on_event is not None,
                agent__has_history=bool(history),
            )
            with kb_factory(settings.kb_base_url, token, settings.request_timeout_seconds) as kb:
                result = runner(
                    request.question, kb, chat, limits, settings.policy, history, on_event
                )
            telemetry.set_attributes(
                span,
                agent__stop_reason=result.stop_reason,
                agent__turns=result.turns,
                agent__tool_calls=len(result.trace),
                agent__retrieved=len(result.retrieved),
                agent__cited_documents=len(cited_filenames(result)),
            )
            return result

    def remember_turn(request: AskRequest, token: str, session_id: str | None, result) -> None:
        if session_id is not None and result.stop_reason == "answered":
            sessions.append(session_id, token, request.question, result.answer)

    def failure(exc: McpError, request_id: str) -> JSONResponse:
        # Never log the question or the token; the status and our own request id are enough.
        log.warning("request=%s knowledge base error status=%s", request_id, exc.status)
        if exc.status in (401, 403):
            return error(401, "unauthorized", "the knowledge base rejected the token")
        return error(502, "knowledge_base_unavailable", "the knowledge base could not be reached")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "rag-agent-python", "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, authorization: str | None = Header(default=None)) -> Any:
        prepared = prepare(request, authorization)
        if isinstance(prepared, JSONResponse):
            return prepared
        token, session_id, history = prepared
        request_id = uuid.uuid4().hex[:12]
        try:
            result = execute(request, token, history)
        except McpError as exc:
            return failure(exc, request_id)
        log.info(
            "request=%s stop=%s turns=%s tools=%s",
            request_id,
            result.stop_reason,
            result.turns,
            len(result.trace),
        )
        if result.stop_reason == "model_error":
            return error(502, "model_unavailable", "the language model did not respond")
        remember_turn(request, token, session_id, result)
        return to_response(result, session_id)

    @app.post("/ask/stream")
    def ask_stream(request: AskRequest, authorization: str | None = Header(default=None)) -> Any:
        """Server-sent events: ``session``, ``tool_call``, ``delta``, then ``answer`` and ``done``.

        ``delta`` text is a preview; the ``answer`` event is authoritative (it has control markers
        removed, and is what ``cited_documents`` was computed from). Failures before streaming
        starts use ordinary HTTP errors; a failure afterwards arrives as an ``error`` event.
        """
        prepared = prepare(request, authorization)
        if isinstance(prepared, JSONResponse):
            return prepared
        token, session_id, history = prepared
        request_id = uuid.uuid4().hex[:12]
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

        def work() -> None:
            try:
                result = execute(request, token, history, lambda e: events.put((e.pop("type"), e)))
                if result.stop_reason == "model_error":
                    events.put(
                        (
                            "error",
                            {
                                "error": "model_unavailable",
                                "detail": "the language model did not respond",
                            },
                        )
                    )
                else:
                    remember_turn(request, token, session_id, result)
                    events.put(("answer", to_response(result, session_id).model_dump()))
            except McpError as exc:
                response = failure(exc, request_id)
                events.put(("error", json.loads(response.body)))
            except Exception:  # the stream must always end with an event
                log.exception("request=%s unexpected failure", request_id)
                events.put(("error", {"error": "internal_error", "detail": "unexpected failure"}))
            finally:
                events.put(None)

        def stream():
            if session_id is not None:
                yield sse("session", {"session_id": session_id})
            threading.Thread(target=work, daemon=True).start()
            while (item := events.get()) is not None:
                yield sse(*item)
            yield sse("done", {})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/sessions/{session_id}", status_code=204, response_class=Response)
    def forget(session_id: str, authorization: str | None = Header(default=None)) -> Response:
        token = bearer_token(authorization)
        if token is None:
            return error(
                401, "unauthorized", "send the knowledge-base bearer token as 'Authorization'"
            )
        if not sessions.forget(session_id, token):
            return error(404, "session_not_found", "unknown or expired session")
        return Response(status_code=204)

    return app


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()
