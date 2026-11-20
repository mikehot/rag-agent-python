from pathlib import Path

from rag_agent.sessions import SqliteSessionStore


def test_sqlite_session_survives_restart_and_enforces_owner(tmp_path: Path) -> None:
    db_file = tmp_path / "sessions.db"
    now = [1000.0]

    store1 = SqliteSessionStore(db_file, ttl_seconds=60.0, clock=lambda: now[0])
    sid = store1.create("token-alice")
    store1.append(sid, "token-alice", "第一问？", "第一答。")
    store1.close()

    # Re-open with a fresh store instance (simulating process restart / another worker).
    store2 = SqliteSessionStore(db_file, ttl_seconds=60.0, clock=lambda: now[0])
    try:
        assert store2.history(sid, "token-bob") is None
        assert store2.forget(sid, "token-bob") is False
        assert store2.history(sid, "token-alice") == [("第一问？", "第一答。")]

        # Advance past TTL -> expires cleanly.
        now[0] += 61.0
        assert store2.history(sid, "token-alice") is None
        assert len(store2) == 0
    finally:
        store2.close()


def test_sqlite_session_enforces_per_owner_and_max_limits(tmp_path: Path) -> None:
    db_file = tmp_path / "limits.db"
    now = [100.0]

    def tick() -> float:
        now[0] += 1.0
        return now[0]

    store = SqliteSessionStore(
        db_file,
        max_sessions=3,
        max_sessions_per_owner=2,
        clock=tick,
    )
    try:
        a1 = store.create("alice")
        a2 = store.create("alice")
        a3 = store.create("alice")
        # Per-owner limit = 2 -> a1 dropped, a2 & a3 kept.
        assert store.history(a1, "alice") is None
        assert store.history(a2, "alice") == []
        assert store.history(a3, "alice") == []

        b1 = store.create("bob")
        b2 = store.create("bob")
        assert len(store) == 3
        assert store.history(b1, "bob") == []
        assert store.history(b2, "bob") == []
    finally:
        store.close()
