"""Contract tests against a running Java knowledge base.

Skipped unless both variables are set, so CI (which has no knowledge base) stays green:

    RAG_AGENT_LIVE_KB_URL           e.g. http://localhost:8080
    RAG_AGENT_LIVE_CREDENTIALS      JSON file {actor: {username, password}} for corp.sre, corp.hr,
                                    corp.outsider, as written by the Java repo's
                                    scripts/eval_stack.sh

They assert only behaviour the adapter documents: the tool allowlist, ACL-filtered results,
identical denials for forbidden and non-existent documents, and rejection of identity-bearing
arguments.
"""

import json
import os
from pathlib import Path

import pytest

from rag_agent.kb_auth import login
from rag_agent.mcp_client import McpClient, McpError

URL = os.environ.get("RAG_AGENT_LIVE_KB_URL")
CREDENTIALS = os.environ.get("RAG_AGENT_LIVE_CREDENTIALS")

pytestmark = pytest.mark.skipif(
    not (URL and CREDENTIALS), reason="live knowledge base not configured"
)


@pytest.fixture(scope="module")
def actors():
    creds = json.loads(Path(CREDENTIALS).read_text(encoding="utf-8"))
    clients = {
        actor: McpClient(URL, login(URL, c["username"], c["password"]))
        for actor, c in creds.items()
    }
    yield clients
    for client in clients.values():
        client.close()


def filenames(client: McpClient) -> set[str]:
    result = client.call_tool("list_documents", {})
    assert result.is_error is False and result.structured is not None
    return {item["filename"] for item in result.structured["items"]}


def test_discover_and_exact_tool_allowlist(actors) -> None:
    sre = actors["corp.sre"]
    assert sre.discover()["protocolVersion"]
    assert {t.name for t in sre.list_tools()} == {
        "search_knowledge",
        "list_documents",
        "get_document_status",
    }


def test_authorized_search_returns_cited_sources(actors) -> None:
    result = actors["corp.sre"].call_tool("search_knowledge", {"query": "Pod 的重启策略有哪些"})
    assert result.is_error is False
    assert result.structured["found"] is True
    assert result.structured["sources"]
    assert all(s["filename"].startswith("k8s-") for s in result.structured["sources"])


def test_outsider_gets_nothing(actors) -> None:
    outsider = actors["corp.outsider"]
    result = outsider.call_tool("search_knowledge", {"query": "Pod 的重启策略有哪些"})
    assert result.structured == {"found": False, "sources": []}
    assert filenames(outsider) == set()


def test_document_visibility_follows_group_grants(actors) -> None:
    sre, hr = filenames(actors["corp.sre"]), filenames(actors["corp.hr"])
    assert "labor-law.md" in hr and "labor-law.md" not in sre
    assert "k8s-pods-pod-lifecycle.md" in sre and "k8s-pods-pod-lifecycle.md" not in hr


def test_forbidden_and_nonexistent_documents_are_indistinguishable(actors) -> None:
    hr_doc = next(
        item
        for item in actors["corp.hr"].call_tool("list_documents", {}).structured["items"]
        if item["filename"] == "labor-law.md"
    )["documentId"]
    sre = actors["corp.sre"]
    forbidden = sre.call_tool("get_document_status", {"documentId": hr_doc})
    missing = sre.call_tool(
        "get_document_status", {"documentId": "00000000-0000-0000-0000-000000000001"}
    )
    assert forbidden.is_error and missing.is_error
    assert forbidden.text == missing.text
    assert forbidden.structured is None and missing.structured is None


@pytest.mark.parametrize("field", ["tenantId", "userId", "actorId", "groupIds"])
def test_identity_bearing_arguments_are_rejected(actors, field) -> None:
    result = actors["corp.sre"].call_tool("list_documents", {field: "anything"})
    assert result.is_error is True
    assert result.structured is None


def test_overlong_query_is_rejected(actors) -> None:
    result = actors["corp.sre"].call_tool("search_knowledge", {"query": "a" * 1001})
    assert result.is_error is True


def test_unknown_tool_is_refused(actors) -> None:
    try:
        result = actors["corp.sre"].call_tool("delete_document", {})
    except McpError:
        return
    assert result.is_error is True


def test_invalid_token_is_401() -> None:
    with McpClient(URL, "not-a-token") as client, pytest.raises(McpError) as caught:
        client.list_tools()
    assert caught.value.status == 401
