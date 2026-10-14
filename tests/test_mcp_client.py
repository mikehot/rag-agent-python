import json

import httpx
import pytest

from rag_agent.kb_auth import login
from rag_agent.mcp_client import PROTOCOL_VERSION, McpClient, McpError

TOKEN = "secret-token-value"


def rpc_ok(request: httpx.Request, result: dict) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def client(handler) -> McpClient:
    return McpClient("http://kb.invalid", TOKEN, transport=httpx.MockTransport(handler))


def test_discover_sends_protocol_and_method_headers() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return rpc_ok(request, {"protocolVersion": PROTOCOL_VERSION})

    assert client(handler).discover()["protocolVersion"] == PROTOCOL_VERSION
    assert seen["headers"]["authorization"] == f"Bearer {TOKEN}"
    assert seen["headers"]["mcp-protocol-version"] == PROTOCOL_VERSION
    assert seen["headers"]["mcp-method"] == "server/discover"
    assert "mcp-name" not in seen["headers"]
    assert seen["body"]["jsonrpc"] == "2.0"
    assert "params" not in seen["body"]


def test_discover_rejects_a_different_protocol_version() -> None:
    with pytest.raises(McpError, match="unsupported protocol version"):
        client(lambda r: rpc_ok(r, {"protocolVersion": "1999-01-01"})).discover()


def test_list_tools_parses_schemas() -> None:
    tool = {
        "name": "search_knowledge",
        "description": "Search",
        "inputSchema": {"type": "object", "required": ["query"]},
    }
    tools = client(lambda r: rpc_ok(r, {"tools": [tool], "ttlMs": 0})).list_tools()
    assert [t.name for t in tools] == ["search_knowledge"]
    assert tools[0].input_schema["required"] == ["query"]


def test_list_tools_rejects_malformed_entries() -> None:
    with pytest.raises(McpError, match="malformed"):
        client(lambda r: rpc_ok(r, {"tools": [{"description": "no name"}]})).list_tools()


def test_call_tool_sets_name_header_and_returns_structured_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["name_header"] = request.headers.get("mcp-name")
        seen["params"] = json.loads(request.content)["params"]
        return rpc_ok(
            request,
            {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
                "structuredContent": {"found": False, "sources": []},
            },
        )

    result = client(handler).call_tool("search_knowledge", {"query": "x"})
    assert seen["name_header"] == "search_knowledge"
    assert seen["params"] == {"name": "search_knowledge", "arguments": {"query": "x"}}
    assert result.is_error is False
    assert result.structured == {"found": False, "sources": []}


def test_tool_level_refusal_is_a_result_not_an_exception() -> None:
    result = client(
        lambda r: rpc_ok(r, {"content": [{"type": "text", "text": "denied"}], "isError": True})
    ).call_tool("list_documents", {"tenantId": "x"})
    assert result.is_error is True
    assert result.text == "denied"
    assert result.structured is None


def test_jsonrpc_error_raises_with_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        error = {"code": -32602, "message": "bad params"}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": error})

    with pytest.raises(McpError) as caught:
        client(handler).call_tool("search_knowledge", {})
    assert caught.value.code == -32602


def test_http_401_raises_and_does_not_leak_the_token() -> None:
    with pytest.raises(McpError) as caught:
        client(lambda r: httpx.Response(401, json={"code": 40101})).list_tools()
    assert caught.value.status == 401
    assert TOKEN not in str(caught.value)


def test_mismatched_response_id_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "someone-else", "result": {}})

    with pytest.raises(McpError, match="does not match"):
        client(handler).discover()


def test_transport_failure_hides_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(McpError) as caught:
        client(handler).discover()
    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None


def test_repr_does_not_contain_the_token() -> None:
    assert TOKEN not in repr(client(lambda r: httpx.Response(200)))


def test_login_returns_token_and_failure_does_not_echo_credentials() -> None:
    ok = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"code": 0, "data": {"token": "abc"}})
    )
    assert login("http://kb.invalid", "u", "p", transport=ok) == "abc"

    bad = httpx.MockTransport(lambda r: httpx.Response(401, json={"code": 40101}))
    with pytest.raises(McpError) as caught:
        login("http://kb.invalid", "user-x", "pass-y", transport=bad)
    assert "pass-y" not in str(caught.value)
    assert "user-x" not in str(caught.value)
