"""Per-owner session quota, the global /ask concurrency cap, and cancellation after a disconnect.

The network tests run a real uvicorn on 127.0.0.1 (no outside access): a disconnect cannot be
produced faithfully by TestClient.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn

from rag_agent.config import Settings
from rag_agent.llm import AssistantTurn, ToolCall
from rag_agent.main import create_app
from rag_agent.sessions import SessionStore
from test_app import FakeKb

# ------------------------------------------------------------------------------ session quota


def test_a_full_owner_evicts_its_own_oldest_session_not_anyone_elses() -> None:
    now = [0.0]
    store = SessionStore(max_sessions_per_owner=2, clock=lambda: now[0])

    def tick() -> float:
        now[0] += 1
        return now[0]

    other = store.create("token-b")
    tick()
    first = store.create("token-a")
    tick()
    second = store.create("token-a")
    tick()
    third = store.create("token-a")  # owner A is at its limit: A's oldest goes
    assert store.history(first, "token-a") is None
    assert store.history(second, "token-a") == []
    assert store.history(third, "token-a") == []
    assert store.history(other, "token-b") == []  # B untouched
    assert len(store) == 3


def test_touching_a_session_protects_it_from_the_owner_quota() -> None:
    now = [0.0]
    store = SessionStore(max_sessions_per_owner=2, clock=lambda: now[0])
    first = store.create("t")
    now[0] += 1
    second = store.create("t")
    now[0] += 1
    store.history(first, "t")  # first is now the most recently used
    now[0] += 1
    store.create("t")
    assert store.history(second, "t") is None
    assert store.history(first, "t") == []


def test_one_owner_cannot_push_other_owners_out_of_the_store() -> None:
    store = SessionStore(max_sessions=50, max_sessions_per_owner=5)
    victims = [store.create(f"user-{i}") for i in range(3)]
    for _ in range(200):
        store.create("greedy")
    assert len(store) == 3 + 5
    assert all(store.history(v, f"user-{i}") == [] for i, v in enumerate(victims))


def test_a_zero_owner_quota_means_unlimited() -> None:
    store = SessionStore(max_sessions=1000, max_sessions_per_owner=0)
    for _ in range(50):
        store.create("t")
    assert len(store) == 50


def test_defaults_keep_the_documented_values() -> None:
    s = Settings()
    assert s.max_sessions_per_owner == 20 and s.max_concurrent_asks == 0


# ------------------------------------------------------------------- live-server test harness


class SlowChat:
    """A model that sleeps each turn and keeps asking for a search until max_turns."""

    def __init__(self, delay: float, *, answer_after: int | None = None) -> None:
        self.delay, self.answer_after = delay, answer_after
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, messages: Any, tools: Any) -> AssistantTurn:
        with self.lock:
            self.calls += 1
            n = self.calls
        time.sleep(self.delay)
        if self.answer_after is not None and n >= self.answer_after:
            return AssistantTurn("好")
        call = ToolCall("c", "search_knowledge", json.dumps({"query": "q"}))
        return AssistantTurn(None, [call])


class Server:
    def __init__(self, chat: SlowChat, **settings: Any) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        app = create_app(
            Settings(kb_base_url="http://kb.invalid", **settings),
            kb_factory=lambda url, token, timeout: FakeKb(token),
            chat=chat,
        )
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "Server":
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self.server.started
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


HEADERS = {"Authorization": "Bearer tok-1"}


@pytest.fixture
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def read_until(response: httpx.Response, event: str) -> None:
    for line in response.iter_lines():
        if line == f"event: {event}":
            return
    raise AssertionError(f"stream ended before {event!r}")


# ------------------------------------------------------------------- concurrency cap and /health


def test_ask_over_the_cap_gets_429_and_health_stays_fast_then_recovers(no_proxy) -> None:
    chat = SlowChat(0.5, answer_after=2)
    with Server(chat, max_concurrent_asks=2, max_turns=4) as srv:
        results: list[int] = []

        def slow() -> None:
            r = httpx.post(srv.url + "/ask", json={"question": "q"}, headers=HEADERS, timeout=30)
            results.append(r.status_code)

        threads = [threading.Thread(target=slow) for _ in range(2)]
        for t in threads:
            t.start()
        assert wait_until(lambda: chat.calls >= 1)

        rejected = httpx.post(srv.url + "/ask", json={"question": "q"}, headers=HEADERS)
        assert rejected.status_code == 429
        assert int(rejected.headers["retry-after"]) >= 1
        assert rejected.json()["error"] == "too_many_requests"
        # The stream endpoint shares the cap, and a refusal creates no session.
        streamed = httpx.post(
            srv.url + "/ask/stream", json={"question": "q", "remember": True}, headers=HEADERS
        )
        assert streamed.status_code == 429
        # Nothing internal in the body: no limit value, no token, no exception text.
        body = rejected.text
        assert "tok-1" not in body and "2" not in body.replace("429", "")

        started = time.perf_counter()
        assert httpx.get(srv.url + "/health").status_code == 200
        assert time.perf_counter() - started < 0.3

        for t in threads:
            t.join()
        assert results == [200, 200]
        # Slots were released: a new request is served again.
        chat.calls = 0
        again = httpx.post(srv.url + "/ask", json={"question": "q"}, headers=HEADERS, timeout=30)
        assert again.status_code == 200


def test_the_cap_is_off_by_default(no_proxy) -> None:
    chat = SlowChat(0.2, answer_after=2)
    with Server(chat) as srv:
        codes: list[int] = []

        def one() -> None:
            codes.append(
                httpx.post(
                    srv.url + "/ask", json={"question": "q"}, headers=HEADERS, timeout=30
                ).status_code
            )

        threads = [threading.Thread(target=one) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert codes == [200] * 6


# --------------------------------------------------------------------- disconnect cancellation


@pytest.mark.parametrize("engine", ["loop", "graph"])
def test_a_dropped_stream_stops_the_work_before_max_turns(no_proxy, engine: str) -> None:
    max_turns = 8
    chat = SlowChat(0.3)  # never answers: without cancellation it runs all 8 turns
    with Server(
        chat, engine=engine, max_turns=max_turns, max_tool_calls=max_turns, max_concurrent_asks=1
    ) as srv:
        with (
            httpx.Client(timeout=30) as client,
            client.stream(
                "POST", srv.url + "/ask/stream", json={"question": "q"}, headers=HEADERS
            ) as response,
        ):
            read_until(response, "tool_call")  # one turn done, then walk away
        time.sleep(max_turns * 0.3 + 1.0)  # long enough for an uncancelled run to finish
        assert 1 <= chat.calls < max_turns, chat.calls
        # The thread and the concurrency slot were released.
        ok = httpx.post(srv.url + "/ask/stream", json={"question": "q"}, headers=HEADERS)
        assert ok.status_code == 200


def test_a_stream_that_is_read_to_the_end_is_never_cancelled(no_proxy) -> None:
    chat = SlowChat(0.05, answer_after=3)
    with Server(chat, max_turns=4, max_tool_calls=4) as srv:
        r = httpx.post(srv.url + "/ask/stream", json={"question": "q"}, headers=HEADERS, timeout=30)
        assert "event: answer" in r.text and "event: done" in r.text
        assert "internal_error" not in r.text
        assert chat.calls == 3


def test_the_app_applies_the_owner_quota_from_settings() -> None:
    from rag_agent.llm import AssistantTurn
    from test_app import AUTH, Scripted, build

    client, _ = build(
        chat=Scripted(*[AssistantTurn("好") for _ in range(4)]), max_sessions_per_owner=2
    )
    # No retrieval needed for a first turn; each remember=true opens one session for this token.
    for _ in range(4):
        assert (
            client.post("/ask", json={"question": "q", "remember": True}, headers=AUTH).status_code
            == 200
        )
    assert len(client.app.state.sessions) == 2


def test_ask_stream_releases_its_concurrency_slot_if_prepare_raises() -> None:
    from fastapi.testclient import TestClient

    from test_app import AUTH, Scripted

    store = SessionStore()
    app = create_app(
        Settings(kb_base_url="http://kb.invalid", max_concurrent_asks=1),
        kb_factory=lambda url, token, timeout: FakeKb(token),
        chat=Scripted(AssistantTurn("好")),
        sessions=store,
    )
    client = TestClient(app, raise_server_exceptions=False)

    orig_create = store.create

    def boom(token: str) -> str:
        raise RuntimeError("session store failure")

    store.create = boom  # type: ignore[method-assign]
    r1 = client.post("/ask/stream", json={"question": "q", "remember": True}, headers=AUTH)
    assert r1.status_code == 500

    store.create = orig_create  # type: ignore[method-assign]
    r2 = client.post("/ask/stream", json={"question": "q"}, headers=AUTH)
    assert r2.status_code == 200
    assert "event: answer" in r2.text
