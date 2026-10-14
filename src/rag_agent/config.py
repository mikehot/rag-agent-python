from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from RAG_AGENT_* environment variables.

    The service holds no knowledge-base credentials: every request carries the caller's own bearer
    token, which is forwarded to the knowledge base unchanged. Only the model server's key (if any)
    is configured here, and it never appears in responses or logs.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_AGENT_", extra="ignore")

    kb_base_url: str = "http://localhost:8080"
    request_timeout_seconds: float = 30.0

    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "google/gemma-4-26b-a4b-qat"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 120.0

    # "loop" is the reference implementation; "graph" is the same agent as a LangGraph StateGraph.
    engine: Literal["loop", "graph"] = "loop"

    # "single" is what every evaluation before the cross-document one measured; "decompose" adds
    # one prompt paragraph asking the model to split multi-part questions (see the cross-document
    # report).
    policy: Literal["single", "decompose"] = "single"

    # "off" (default) or "console": spans to standard output, never any user text or secret.
    tracing: Literal["off", "console"] = "off"

    session_ttl_seconds: float = 1800.0
    max_sessions: int = 1000

    max_turns: int = 4
    max_tool_calls: int = 3


def unknown_settings(environ: dict[str, str]) -> list[str]:
    """Names of RAG_AGENT_* variables that this version does not know.

    Settings ignore extras on purpose (a shared environment may carry unrelated variables), but a
    misspelled or unsupported RAG_AGENT_* name would otherwise be dropped silently and the service
    would run with a default the operator did not intend.
    """
    known = {f"RAG_AGENT_{name}".upper() for name in Settings.model_fields}
    return sorted(
        k for k in environ if k.upper().startswith("RAG_AGENT_") and k.upper() not in known
    )
