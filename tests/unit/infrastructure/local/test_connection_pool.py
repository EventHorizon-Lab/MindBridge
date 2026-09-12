"""What pooling connections must keep true, and the one cost it removes.

Opening a SQLite connection is not free: one `sqlite3_open` plus the four `PRAGMA` statements that
make a connection usable, measured at 450 us against a populated store. Nothing held a connection
between calls, so every store helper opened its own -- seven for one `add`, five for one `search`,
one for a `get` that then spent nine tenths of its latency on them.

The pool is a checkout pool rather than a connection per thread: a connection leaves it for
exactly one `with` block and is returned by the same `finally` that used to close it. That is what
these tests pin. Exclusive checkout is the whole safety argument -- it is why two threads never
execute on one connection and why two nested blocks still get two connections, as they did when
each opened its own. A block that raised must not return its connection, because a statement that
failed part way through can leave a transaction open that the next borrower would silently join.
And `close()` has to reclaim what the pool still holds, or a store would outlive its handles.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredMemory
from mindbridge.infrastructure.local.store._connections import Connections

_NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


def _memory(memory_id: str) -> StoredMemory:
    return StoredMemory(
        memory_id=memory_id,
        content=f"observation {memory_id}",
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_repeated_reads_open_one_connection_rather_than_one_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = 0
    real_open = Connections._open_connection

    def counting_open(store: Connections, *, secure_delete: bool = False) -> sqlite3.Connection:
        nonlocal opened
        opened += 1
        return real_open(store, secure_delete=secure_delete)

    with LocalStore(tmp_path) as store:
        store.records.write_memory(_memory("m-0"))
        monkeypatch.setattr(Connections, "_open_connection", counting_open)
        for _ in range(20):
            assert store.records.read_memories(("m-0",))[0].memory_id == "m-0"

    assert opened == 0, "a warm pool must satisfy a read without connecting"


def test_two_nested_checkouts_get_two_connections(tmp_path: Path) -> None:
    """The pool is what makes nesting safe: an inner block cannot join an outer transaction."""
    with (
        LocalStore(tmp_path) as store,
        store._connections.connection() as outer,
        store._connections.connection() as inner,
    ):
        assert outer is not inner
        outer.execute("BEGIN IMMEDIATE")
        # A second connection means the inner block has its own transaction scope, which is
        # exactly what a connection cached per caller would have taken away.
        inner.execute("BEGIN")
        inner.rollback()
        outer.rollback()


def test_concurrent_readers_never_hold_the_same_connection(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.records.write_memory(_memory("m-0"))
        both_inside = Barrier(2, timeout=30)

        def borrow() -> int:
            with store._connections.connection() as connection:
                both_inside.wait()
                connection.execute("SELECT 1").fetchone()
                return id(connection)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Both submitted before either result is awaited, or the barrier deadlocks.
            futures = [executor.submit(borrow) for _ in (0, 1)]
            first, second = (future.result() for future in futures)
        assert first != second


def test_a_failed_statement_does_not_poison_the_next_read(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.records.write_memory(_memory("m-0"))
        with pytest.raises(sqlite3.OperationalError), store._connections.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("SELECT * FROM a_table_that_does_not_exist")

        # The failed borrower left a write transaction open. Returning that connection to the pool
        # would hand the open transaction to whoever borrowed it next.
        assert store.records.read_memories(("m-0",))[0].memory_id == "m-0"
        store.records.write_memory(_memory("m-1"))
        assert {memory.memory_id for memory in store.records.read_memories(("m-0", "m-1"))} == {
            "m-0",
            "m-1",
        }


def test_close_reclaims_the_pooled_connections(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.records.write_memory(_memory("m-0"))
    store.records.read_memories(("m-0",))
    pooled = tuple(store._connections._pool)
    assert pooled, "a completed read must leave its connection idle in the pool"

    store.close()

    assert not store._connections._pool
    for connection in pooled:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
