"""A minimal OpenAI-compatible chat client (LM Studio, DeepSeek, ...), tools included."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


class LlmError(Exception):
    """The model server failed or returned something unusable."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_raw: str


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    completion_tokens: int | None = None


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    temperature: float | None = 0.0
    max_tokens: int = 1200
    timeout: float = 120.0
    transport: httpx.BaseTransport | None = None

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                response = http.post(
                    f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise LlmError(f"model server unreachable: {type(exc).__name__}") from None
        if response.status_code != 200:
            raise LlmError(f"model server returned HTTP {response.status_code}")
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LlmError("model server returned a malformed completion") from None
        calls = []
        for raw in message.get("tool_calls") or []:
            try:
                function = raw["function"]
                calls.append(
                    ToolCall(
                        id=str(raw.get("id") or f"call_{len(calls)}"),
                        name=str(function["name"]),
                        arguments_raw=str(function.get("arguments") or "{}"),
                    )
                )
            except (KeyError, TypeError):
                raise LlmError("model server returned a malformed tool call") from None
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return AssistantTurn(
            content=message.get("content"),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            completion_tokens=usage.get("completion_tokens"),
        )
