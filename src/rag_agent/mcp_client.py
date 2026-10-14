"""A small client for the Java knowledge base's stateless MCP adapter (``POST /mcp``).

It speaks only what that adapter documents: ``server/discover``, ``tools/list`` and
``tools/call`` over JSON-RPC 2.0, with the caller's bearer token. It is not a general MCP SDK, and
it makes no claim of full transport or authorization conformance.

Two properties matter more than convenience here:

* Identity comes only from the bearer token. Arguments that try to name a tenant, user, or group
  are rejected by the server's tool boundary; this client neither adds nor strips such fields.
* The token never appears in exceptions, ``repr`` output, or logs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

PROTOCOL_VERSION = "2026-07-28"


class McpError(Exception):
    """The adapter returned a transport-level or JSON-RPC error, or an unusable response."""

    def __init__(self, message: str, *, status: int | None = None, code: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one ``tools/call``.

    ``is_error`` means the tool ran and refused or failed (for example an argument the boundary
    rejects); transport and protocol failures raise :class:`McpError` instead.
    """

    is_error: bool
    text: str
    structured: dict[str, Any] | None


@dataclass
class McpClient:
    base_url: str
    token: str = field(repr=False)
    timeout: float = 30.0
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            base_url=self.base_url.rstrip("/"), timeout=self.timeout, transport=self.transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> McpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def discover(self) -> dict[str, Any]:
        result = self._rpc("server/discover", None)
        if result.get("protocolVersion") != PROTOCOL_VERSION:
            raise McpError(
                f"unsupported protocol version {result.get('protocolVersion')!r}; "
                f"this client implements {PROTOCOL_VERSION}"
            )
        return result

    def list_tools(self) -> list[Tool]:
        result = self._rpc("tools/list", {})
        raw = result.get("tools")
        if not isinstance(raw, list):
            raise McpError("tools/list returned no tool array")
        tools = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise McpError("tools/list returned a malformed tool entry")
            tools.append(
                Tool(
                    name=item["name"],
                    description=str(item.get("description", "")),
                    input_schema=item.get("inputSchema") or {},
                )
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        result = self._rpc(
            "tools/call", {"name": name, "arguments": arguments or {}}, tool_name=name
        )
        content = result.get("content")
        text = ""
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        structured = result.get("structuredContent")
        return ToolResult(
            is_error=bool(result.get("isError")),
            text=text,
            structured=structured if isinstance(structured, dict) else None,
        )

    def _rpc(
        self, method: str, params: dict[str, Any] | None, tool_name: str | None = None
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": method,
        }
        if tool_name:
            headers["Mcp-Name"] = tool_name
        try:
            response = self._http.post("/mcp", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            # Do not chain: the request object carries the Authorization header.
            raise McpError(f"transport error calling {method}: {type(exc).__name__}") from None
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise McpError(f"{method}: non-JSON response", status=response.status_code)
        error = body.get("error")
        if isinstance(error, dict):
            raise McpError(
                f"{method}: {error.get('message', 'JSON-RPC error')}",
                status=response.status_code,
                code=error.get("code") if isinstance(error.get("code"), int) else None,
            )
        if response.status_code != 200:
            raise McpError(f"{method}: HTTP {response.status_code}", status=response.status_code)
        if body.get("id") != request_id:
            raise McpError(f"{method}: response id does not match the request")
        result = body.get("result")
        if not isinstance(result, dict):
            raise McpError(f"{method}: response has no result object")
        return result
