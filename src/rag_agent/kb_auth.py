"""Obtain a bearer token from the Java knowledge base's login endpoint."""

from __future__ import annotations

import httpx

from rag_agent.mcp_client import McpError


def login(
    base_url: str,
    username: str,
    password: str,
    *,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> str:
    try:
        with httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        ) as http:
            response = http.post(
                "/api/auth/login", json={"username": username, "password": password}
            )
    except httpx.HTTPError as exc:
        raise McpError(f"transport error during login: {type(exc).__name__}") from None
    try:
        envelope = response.json()
    except ValueError:
        envelope = None
    data = envelope.get("data") if isinstance(envelope, dict) else None
    if response.status_code != 200 or not isinstance(data, dict) or not data.get("token"):
        # Never echo the username/password or the response body.
        raise McpError(f"login failed (HTTP {response.status_code})", status=response.status_code)
    return str(data["token"])
