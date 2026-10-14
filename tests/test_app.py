from fastapi.testclient import TestClient

from rag_agent import __version__
from rag_agent.config import Settings
from rag_agent.main import create_app


def client() -> TestClient:
    return TestClient(create_app(Settings(kb_base_url="http://kb.invalid")))


def test_health_reports_service_and_version() -> None:
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-agent-python",
        "version": __version__,
    }


def test_ask_is_an_explicit_501_until_the_agent_exists() -> None:
    response = client().post("/ask", json={"question": "试用期最长几个月？"})
    assert response.status_code == 501
    assert response.json()["error"] == "not_implemented"


def test_ask_rejects_an_empty_question() -> None:
    assert client().post("/ask", json={"question": ""}).status_code == 422


def test_settings_read_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AGENT_KB_BASE_URL", "http://example.invalid:9")
    assert Settings().kb_base_url == "http://example.invalid:9"
