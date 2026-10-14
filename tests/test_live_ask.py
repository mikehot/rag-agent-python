"""End-to-end check of POST /ask against a running knowledge base and a running model server.

Skipped unless RAG_AGENT_LIVE_KB_URL, RAG_AGENT_LIVE_CREDENTIALS and RAG_AGENT_LIVE_LLM=1 are
set. It asserts behaviour that must hold for any reasonable model (access is respected, the trace
is honest), not particular wording.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.config import Settings
from rag_agent.kb_auth import login
from rag_agent.main import create_app

URL = os.environ.get("RAG_AGENT_LIVE_KB_URL")
CREDENTIALS = os.environ.get("RAG_AGENT_LIVE_CREDENTIALS")
LLM = os.environ.get("RAG_AGENT_LIVE_LLM") == "1"

pytestmark = pytest.mark.skipif(
    not (URL and CREDENTIALS and LLM), reason="live knowledge base and model not configured"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(Settings(kb_base_url=URL)))


def headers(actor: str) -> dict[str, str]:
    creds = json.loads(Path(CREDENTIALS).read_text(encoding="utf-8"))[actor]
    return {"Authorization": f"Bearer {login(URL, creds['username'], creds['password'])}"}


def test_authorized_question_is_answered_with_a_search_trace(client) -> None:
    response = client.post(
        "/ask", json={"question": "Pod 的重启策略有哪些？"}, headers=headers("corp.sre")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert {"name": "search_knowledge", "outcome": "ok"} in body["tool_calls"]
    assert body["retrieved"]
    assert all(r["filename"].startswith("k8s-") for r in body["retrieved"])


def test_unauthorized_user_gets_nothing_from_documents_they_cannot_read(client) -> None:
    response = client.post(
        "/ask",
        json={"question": "Pod 的重启策略有哪些？"},
        headers=headers("corp.outsider"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retrieved"] == [] and body["cited_documents"] == []
    assert "OnFailure" not in body["answer"]
