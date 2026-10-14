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

    max_turns: int = 4
    max_tool_calls: int = 3
