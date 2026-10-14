"""A minimal OpenAI-compatible chat client (LM Studio, DeepSeek, ...), tools included."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
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


class MarkerFilter:
    """Removes chat-template control markers (seen as ``<|channel>``) from a stream of text.

    A marker can be split across chunks, so text from a ``<|`` onwards is held back until the marker
    is complete (then dropped) or can no longer be one (then released). ``flush`` releases whatever
    is still held at the end of the stream.
    """

    _MARKER = re.compile(r"<\|[^<>\n|]{0,40}\|?>")
    _OPEN = re.compile(r"<\|[^<>\n|]{0,40}\|?$")

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        buffer = self._MARKER.sub("", self._held + text)
        hold = self._OPEN.search(buffer)
        partial = re.search(r"<$", buffer)
        cut = hold.start() if hold else (partial.start() if partial else None)
        if cut is None:
            self._held = ""
            return buffer
        self._held = buffer[cut:]
        return buffer[:cut]

    def flush(self) -> str:
        out, self._held = self._MARKER.sub("", self._held), ""
        return out


@dataclass
class ChatClient:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    temperature: float | None = 0.0
    max_tokens: int = 1200
    timeout: float = 120.0
    transport: httpx.BaseTransport | None = None

    def _request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], stream: bool
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if stream:
            body["stream"] = True
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return body, headers

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None],
    ) -> AssistantTurn:
        """Like :meth:`complete`, but forwards content as it arrives (control markers removed)."""
        body, headers = self._request(messages, tools, stream=True)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        content: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage: dict[str, Any] = {}
        cleaner = MarkerFilter()
        try:
            with (
                httpx.Client(timeout=self.timeout, transport=self.transport) as http,
                http.stream("POST", url, json=body, headers=headers) as response,
            ):
                if response.status_code != 200:
                    raise LlmError(f"model server returned HTTP {response.status_code}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            content.append(text)
                            shown = cleaner.feed(text)
                            if shown:
                                on_delta(shown)
                        for raw in delta.get("tool_calls") or []:
                            slot = calls.setdefault(
                                int(raw.get("index", 0)), {"id": "", "name": "", "arguments": ""}
                            )
                            if raw.get("id"):
                                slot["id"] = str(raw["id"])
                            function = raw.get("function") or {}
                            if function.get("name"):
                                slot["name"] = str(function["name"])
                            if function.get("arguments"):
                                slot["arguments"] += str(function["arguments"])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.HTTPError as exc:
            raise LlmError(f"model server unreachable: {type(exc).__name__}") from None
        tail = cleaner.flush()
        if tail:
            on_delta(tail)
        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments_raw=slot["arguments"] or "{}",
            )
            for index, slot in sorted(calls.items())
            if slot["name"]
        ]
        return AssistantTurn(
            content="".join(content) or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            completion_tokens=usage.get("completion_tokens"),
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        body, headers = self._request(messages, tools, stream=False)
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
