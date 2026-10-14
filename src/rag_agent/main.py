from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_agent import __version__
from rag_agent.agent import AgentResult, ChatBackend, Limits, ToolBackend, run_agent
from rag_agent.config import Settings
from rag_agent.graph import run_graph_agent
from rag_agent.llm import ChatClient
from rag_agent.mcp_client import McpClient, McpError

log = logging.getLogger("rag_agent")

KbFactory = Callable[[str, str, float], AbstractContextManager[ToolBackend]]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


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


def default_kb_factory(base_url: str, token: str, timeout: float) -> McpClient:
    return McpClient(base_url, token, timeout=timeout)


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


def to_response(result: AgentResult) -> AskResponse:
    cited = [r["filename"] for r in result.retrieved if r["filename"] in result.answer]
    return AskResponse(
        answer=result.answer,
        answered=result.stop_reason == "answered",
        stop_reason=result.stop_reason,
        turns=result.turns,
        tool_calls=[ToolCallInfo(name=t.name, outcome=t.outcome) for t in result.trace],
        retrieved=[DocumentRef(**r) for r in result.retrieved],
        cited_documents=list(dict.fromkeys(cited)),
    )


def error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def create_app(
    settings: Settings | None = None,
    kb_factory: KbFactory | None = None,
    chat: ChatBackend | None = None,
) -> FastAPI:
    settings = settings or Settings()
    kb_factory = kb_factory or default_kb_factory
    chat = chat or ChatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )
    limits = Limits(max_turns=settings.max_turns, max_tool_calls=settings.max_tool_calls)
    runner = run_graph_agent if settings.engine == "graph" else run_agent
    app = FastAPI(title="rag-agent-python", version=__version__)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "rag-agent-python", "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, authorization: str | None = Header(default=None)) -> Any:
        token = bearer_token(authorization)
        if token is None:
            return error(
                401, "unauthorized", "send the knowledge-base bearer token as 'Authorization'"
            )
        request_id = uuid.uuid4().hex[:12]
        try:
            with kb_factory(settings.kb_base_url, token, settings.request_timeout_seconds) as kb:
                result = runner(request.question, kb, chat, limits, settings.policy)
        except McpError as exc:
            # Never log the question or the token; the status and our own request id are enough.
            log.warning("request=%s knowledge base error status=%s", request_id, exc.status)
            if exc.status in (401, 403):
                return error(401, "unauthorized", "the knowledge base rejected the token")
            return error(
                502, "knowledge_base_unavailable", "the knowledge base could not be reached"
            )
        log.info(
            "request=%s stop=%s turns=%s tools=%s",
            request_id,
            result.stop_reason,
            result.turns,
            len(result.trace),
        )
        if result.stop_reason == "model_error":
            return error(502, "model_unavailable", "the language model did not respond")
        return to_response(result)

    return app


app = create_app()
