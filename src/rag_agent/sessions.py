"""In-memory conversation history for multi-turn questions.

Design constraints, because this holds user text:

* **Owner-bound.** A session is usable only with the bearer token that created it. The token is
  never stored; a keyed hash with a per-process random salt is, so it cannot be recovered or
  compared across restarts. Presenting someone else's session id is indistinguishable from
  presenting an unknown one.
* **Opt-in.** Nothing is remembered unless the caller asks (``remember`` or a session id).
* **Only questions and final answers.** Tool output is never stored, so every turn retrieves again
  with the caller's *current* permissions. What a user was shown earlier can still reappear in an
  answer they were given, so sessions are short-lived and bounded.
* **Bounded and expiring.** Time to live, number of sessions, turns per session, answer length.
* **Process-local.** Restarting the service forgets every session; nothing is persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _Session:
    owner: str
    turns: list[tuple[str, str]] = field(default_factory=list)
    touched: float = 0.0


class SessionStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 1000,
        max_turns: int = 6,
        max_answer_chars: int = 2000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_turns = max_turns
        self._max_answer_chars = max_answer_chars
        self._clock = clock
        self._salt = secrets.token_bytes(32)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def _owner(self, token: str) -> str:
        return hmac.new(self._salt, token.encode("utf-8"), hashlib.sha256).hexdigest()

    def _expire(self) -> None:
        now = self._clock()
        for key in [k for k, s in self._sessions.items() if now - s.touched > self._ttl]:
            del self._sessions[key]

    def create(self, token: str) -> str:
        with self._lock:
            self._expire()
            while len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].touched)
                del self._sessions[oldest]
            session_id = secrets.token_urlsafe(18)
            self._sessions[session_id] = _Session(self._owner(token), touched=self._clock())
            return session_id

    def history(self, session_id: str, token: str) -> list[tuple[str, str]] | None:
        """Earlier turns, or ``None`` when the session is unknown, expired or not this caller's."""
        with self._lock:
            self._expire()
            session = self._sessions.get(session_id)
            if session is None or not hmac.compare_digest(session.owner, self._owner(token)):
                return None
            session.touched = self._clock()
            return list(session.turns)

    def append(self, session_id: str, token: str, question: str, answer: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not hmac.compare_digest(session.owner, self._owner(token)):
                return
            session.turns.append((question, answer[: self._max_answer_chars]))
            del session.turns[: -self._max_turns]
            session.touched = self._clock()

    def forget(self, session_id: str, token: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not hmac.compare_digest(session.owner, self._owner(token)):
                return False
            del self._sessions[session_id]
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
