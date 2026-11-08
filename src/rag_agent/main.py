from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from fastapi import FastAPI, Header, Response
from fastapi.exceptions import RequestValidationError
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
    if scheme.lower() != "bearer" or not value:
        return None
    # A token is forwarded as an HTTP header value, so it must be visible ASCII without spaces.
    # Anything else (a latin-1 byte, a control character) cannot be a real token and would make
    # the knowledge-base client raise an encoding error, which surfaced as an HTTP 500.
    if not (value.isascii() and value.isprintable() and " " not in value):
        return None
    return value


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


class BodyLimit:
    """Reject request bodies over ``limit`` bytes with 413, before FastAPI buffers and parses them.

    Checks the declared Content-Length first and counts the bytes of chunked bodies as they arrive,
    so a body without a Content-Length cannot get past it.
    """

    def __init__(self, app, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        too_large = JSONResponse(
            status_code=413,
            content={"error": "request_too_large", "detail": "the request body is too large"},
        )
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.limit:
            await too_large(scope, receive, send)
            return
        seen = 0
        rejected = False

        async def limited_receive():
            nonlocal seen, rejected
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    rejected = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message) -> None:
            if rejected and message["type"] == "http.response.start":
                await too_large(scope, receive, send)
                raise _Dropped
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _Dropped:
            return


class _Dropped(Exception):
    pass


class Cancelled(Exception):
    """The client went away; stop spending model and knowledge-base time on its behalf."""


class _CancelAware:
    """Raises ``Cancelled`` at the next model or tool call once ``cancel`` is set.

    Wraps the two backends from outside, so the agent loop (and the LangGraph version) run exactly
    the code that was evaluated; a call already in flight is not interrupted, the next one never
    starts.
    """

    def __init__(self, inner: Any, cancel: threading.Event) -> None:
        self._inner, self._cancel = inner, cancel

    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)  # AttributeError stays one (hasattr checks rely on it)
        if name not in ("complete", "complete_stream", "list_tools", "call_tool"):
            return target

        def guarded(*args: Any, **kwargs: Any) -> Any:
            self._check()
            return target(*args, **kwargs)

        return guarded


class CancelOnExit(StreamingResponse):
    """A streaming response that sets ``cancel`` when the response ends for any reason.

    Normal completion ends the response after the worker has finished, so setting the flag then is
    a no-op; a client disconnect or a failed send ends it early and the worker stops at its next
    model or tool call.
    """

    def __init__(self, content: Any, cancel: threading.Event, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._cancel = cancel

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cancel.set()


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
        reasoning_effort=settings.llm_reasoning_effort,
        timeout=settings.llm_timeout_seconds,
    )
    sessions = (
        sessions
        if sessions is not None
        else SessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
            max_sessions_per_owner=settings.max_sessions_per_owner,
        )
    )
    limits = Limits(
        max_turns=settings.max_turns,
        max_tool_calls=settings.max_tool_calls,
        enforce_retrieval=settings.enforce_retrieval,
    )
    runner = run_graph_agent if settings.engine == "graph" else run_agent
    app = FastAPI(
        title="rag-agent-python",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    app.state.sessions = sessions

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Any, _exc: RequestValidationError) -> JSONResponse:
        return error(422, "invalid_request", "request validation failed")

    gate = (
        threading.BoundedSemaphore(settings.max_concurrent_asks)
        if (settings.max_concurrent_asks > 0)
        else None
    )

    def try_acquire() -> bool:
        return gate is None or gate.acquire(blocking=False)

    def release() -> None:
        if gate is not None:
            gate.release()

    def busy() -> JSONResponse:
        # Fixed text: no limit value, no counts, nothing about who else is using the service.
        return JSONResponse(
            status_code=429,
            content={"error": "too_many_requests", "detail": "the service is busy; retry shortly"},
            headers={"Retry-After": "1"},
        )

    def prepare(request: AskRequest, authorization: str | None):
        """Authenticate and resolve the conversation. Returns (token, session_id, history, created)
        or an error response. ``created`` is true when this request opened the session."""
        token = bearer_token(authorization)
        if token is None:
            return error(
                401, "unauthorized", "send the knowledge-base bearer token as 'Authorization'"
            )
        session_id: str | None = None
        history = None
        created = False
        if request.session_id is not None:
            history = sessions.history(request.session_id, token)
            if history is None:
                return error(404, "session_not_found", "unknown or expired session")
            session_id = request.session_id
        elif request.remember:
            session_id = sessions.create(token)
            history = []
            created = True
        return token, session_id, history, created

    def discard_if_new(session_id: str | None, token: str, created: bool) -> None:
        # The session exists before the knowledge base has judged the token. When it rejects it,
        # nothing may be left behind: the entry could never be used, and a flood of bogus tokens
        # would otherwise push real sessions out of the bounded store.
        if created and session_id is not None:
            sessions.forget(session_id, token)

    def execute(
        request: AskRequest, token: str, history, on_event=None, cancel=None
    ) -> AgentResult:
        with telemetry.tracer().start_as_current_span("agent.ask") as span:
            telemetry.set_attributes(
                span,
                agent__engine=settings.engine,
                agent__policy=settings.policy,
                agent__streaming=on_event is not None,
                agent__has_history=bool(history),
            )
            with kb_factory(settings.kb_base_url, token, settings.request_timeout_seconds) as kb:
                if cancel is not None:
                    kb_, chat_ = _CancelAware(kb, cancel), _CancelAware(chat, cancel)
                else:
                    kb_, chat_ = kb, chat
                result = runner(
                    request.question, kb_, chat_, limits, settings.policy, history, on_event
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
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "rag-agent-python", "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, authorization: str | None = Header(default=None)) -> Any:
        if not try_acquire():
            return busy()
        try:
            return answer_once(request, authorization)
        finally:
            release()

    def answer_once(request: AskRequest, authorization: str | None) -> Any:
        prepared = prepare(request, authorization)
        if isinstance(prepared, JSONResponse):
            return prepared
        token, session_id, history, created = prepared
        request_id = uuid.uuid4().hex[:12]
        try:
            result = execute(request, token, history)
        except McpError as exc:
            discard_if_new(session_id, token, created)  # the error carries no session id
            return failure(exc, request_id)
        log.info(
            "request=%s stop=%s turns=%s tools=%s",
            request_id,
            result.stop_reason,
            result.turns,
            len(result.trace),
        )
        if result.stop_reason == "model_error":
            discard_if_new(session_id, token, created)
            return error(502, "model_unavailable", "the language model did not respond")
        remember_turn(request, token, session_id, result)
        return to_response(result, session_id)

    @app.post("/ask/stream")
    async def ask_stream(
        request: AskRequest, authorization: str | None = Header(default=None)
    ) -> Any:
        """Server-sent events: ``session``, ``tool_call``, ``delta``, then ``answer`` and ``done``.

        ``delta`` text is a preview; the ``answer`` event is authoritative (it has control markers
        removed, and is what ``cited_documents`` was computed from). Failures before streaming
        starts use ordinary HTTP errors; a failure afterwards arrives as an ``error`` event.
        If the client disconnects, the worker stops at its next model or tool call.
        """
        if not try_acquire():
            return busy()
        try:
            prepared = prepare(request, authorization)
        except BaseException:
            release()
            raise
        if isinstance(prepared, JSONResponse):
            release()
            return prepared
        token, session_id, history, created = prepared
        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        cancel = threading.Event()

        def put(item: tuple[str, dict[str, Any]] | None) -> None:
            # RuntimeError: the loop is gone (server shutting down); nobody is listening.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(events.put_nowait, item)

        def on_event(event: dict[str, Any]) -> None:
            if cancel.is_set():
                raise Cancelled  # also abandons a model call that is still streaming deltas
            put((event.pop("type"), event))

        def work() -> None:
            try:
                result = execute(request, token, history, on_event, cancel)
                if result.stop_reason == "model_error":
                    put(
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
                    put(("answer", to_response(result, session_id).model_dump()))
            except Cancelled:
                log.info("request=%s client disconnected; work cancelled", request_id)
                discard_if_new(session_id, token, created)
            except McpError as exc:
                response = failure(exc, request_id)
                if response.status_code == 401:
                    discard_if_new(session_id, token, created)
                put(("error", json.loads(response.body)))
            except Exception:  # the stream must always end with an event
                log.exception("request=%s unexpected failure", request_id)
                put(("error", {"error": "internal_error", "detail": "unexpected failure"}))
            finally:
                put(None)
                release()

        async def stream():
            if session_id is not None:
                yield sse("session", {"session_id": session_id})
            while (item := await events.get()) is not None:
                yield sse(*item)
            yield sse("done", {})

        try:
            threading.Thread(target=work, daemon=True).start()
        except BaseException:
            release()
            raise
        return CancelOnExit(
            stream(),
            cancel,
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
