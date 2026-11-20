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
import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


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
        max_sessions_per_owner: int = 0,
        max_turns: int = 6,
        max_answer_chars: int = 2000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        # 0 = no per-owner limit. Otherwise an owner at its limit loses its own oldest session.
        self._max_per_owner = max_sessions_per_owner
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
            owner = self._owner(token)
            if self._max_per_owner > 0:
                # Make room from the caller's own sessions first, so one owner cannot push
                # anybody else's out of the store.
                mine = sorted(
                    (k for k, s in self._sessions.items() if s.owner == owner),
                    key=lambda k: self._sessions[k].touched,
                )
                for key in mine[: max(0, len(mine) - self._max_per_owner + 1)]:
                    del self._sessions[key]
            while len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].touched)
                del self._sessions[oldest]
            session_id = secrets.token_urlsafe(18)
            self._sessions[session_id] = _Session(owner, touched=self._clock())
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


class SqliteSessionStore:
    """SQLite-backed session store preserving owner-bound security across restarts."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 1000,
        max_sessions_per_owner: int = 0,
        max_turns: int = 6,
        max_answer_chars: int = 2000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_per_owner = max_sessions_per_owner
        self._max_turns = max_turns
        self._max_answer_chars = max_answer_chars
        self._clock = clock
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                turns_json TEXT NOT NULL,
                touched REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_owner_touched ON sessions (owner, touched)"
        )
        self._salt = self._load_or_create_salt()
        self._conn.commit()

    def _load_or_create_salt(self) -> bytes:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'salt'").fetchone()
        if row is not None:
            return bytes(row[0])
        salt = secrets.token_bytes(32)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('salt', ?)", (salt,)
        )
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'salt'").fetchone()
        assert row is not None
        return bytes(row[0])

    def _owner(self, token: str) -> str:
        return hmac.new(self._salt, token.encode("utf-8"), hashlib.sha256).hexdigest()

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        self._conn.execute("DELETE FROM sessions WHERE touched < ?", (cutoff,))

    def create(self, token: str) -> str:
        with self._lock, self._conn:
            self._expire()
            owner = self._owner(token)
            if self._max_per_owner > 0:
                mine = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT session_id FROM sessions WHERE owner = ? ORDER BY touched ASC",
                        (owner,),
                    ).fetchall()
                ]
                excess = len(mine) - self._max_per_owner + 1
                if excess > 0:
                    for key in mine[:excess]:
                        self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (key,))
            total = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if total >= self._max_sessions:
                excess = total - self._max_sessions + 1
                oldest = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT session_id FROM sessions ORDER BY touched ASC LIMIT ?",
                        (excess,),
                    ).fetchall()
                ]
                for key in oldest:
                    self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (key,))
            session_id = secrets.token_urlsafe(18)
            self._conn.execute(
                "INSERT INTO sessions (session_id, owner, turns_json, touched) VALUES (?, ?, ?, ?)",
                (session_id, owner, "[]", self._clock()),
            )
            return session_id

    def history(self, session_id: str, token: str) -> list[tuple[str, str]] | None:
        with self._lock, self._conn:
            self._expire()
            row = self._conn.execute(
                "SELECT owner, turns_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row[0], self._owner(token)):
                return None
            self._conn.execute(
                "UPDATE sessions SET touched = ? WHERE session_id = ?",
                (self._clock(), session_id),
            )
            raw_turns = json.loads(row[1])
            return [(str(q), str(a)) for q, a in raw_turns]

    def append(self, session_id: str, token: str, question: str, answer: str) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT owner, turns_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row[0], self._owner(token)):
                return
            turns = json.loads(row[1])
            turns.append([question, answer[: self._max_answer_chars]])
            turns = turns[-self._max_turns :]
            self._conn.execute(
                "UPDATE sessions SET turns_json = ?, touched = ? WHERE session_id = ?",
                (json.dumps(turns, ensure_ascii=False), self._clock(), session_id),
            )

    def forget(self, session_id: str, token: str) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT owner FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or not hmac.compare_digest(row[0], self._owner(token)):
                return False
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
