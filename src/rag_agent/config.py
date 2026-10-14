from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from RAG_AGENT_* environment variables.

    Secrets (the knowledge-base credentials) are intentionally not defined yet: they arrive with
    the MCP client and must come from the environment, never from the repository.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_AGENT_", extra="ignore")

    kb_base_url: str = "http://localhost:8080"
    request_timeout_seconds: float = 30.0
