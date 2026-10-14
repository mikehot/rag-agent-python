from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_agent import __version__
from rag_agent.config import Settings


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="rag-agent-python", version=__version__)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "rag-agent-python", "version": __version__}

    @app.post("/ask")
    def ask(request: AskRequest) -> JSONResponse:
        # Honest placeholder: the agent loop needs the MCP client first (see README roadmap).
        return JSONResponse(
            status_code=501,
            content={"error": "not_implemented", "detail": "the agent loop is not built yet"},
        )

    return app


app = create_app()
