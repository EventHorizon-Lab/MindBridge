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
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredMemory
from mindbridge.infrastructure.local._capacity import Capacity
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError
from mindbridge.infrastructure.local.store import _connections as connections_module
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


def test_connection_limit_includes_secure_delete_and_recovers(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        with ExitStack() as borrowers:
            for _ in range(32):
                borrowers.enter_context(store._connections.connection())
            with (
                pytest.raises(RuntimeError, match="capacity"),
                store._connections.connection(secure_delete=True),
            ):
                pytest.fail("overload opened another connection")
        with store._connections.connection(secure_delete=True) as connection:
            assert connection.execute("SELECT 1").fetchone()[0] == 1


def test_close_keeps_directory_ownership_until_borrowers_stop(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    entered, release = Event(), Event()

    def borrow() -> None:
        with store._connections.connection() as connection:
            entered.set()
            assert release.wait(5)
            assert connection.execute("SELECT 1").fetchone()[0] == 1

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            borrower = executor.submit(borrow)
            try:
                assert entered.wait(2)
                closing = executor.submit(store.close)
                with store._connections._pool_lock:
                    assert store._connections._pool_lock.wait_for(
                        lambda: store._connections._closed, timeout=1
                    )
                assert not closing.done()
                with pytest.raises(DataDirectoryInUseError):
                    LocalStore(tmp_path)
            finally:
                release.set()
            borrower.result(timeout=2)
            closing.result(timeout=2)
        with LocalStore(tmp_path) as reopened:
            assert reopened.records.read_memories(("absent",)) == ()
    finally:
        release.set()
        store.close()


def test_idle_connections_leave_capacity_for_other_directories(tmp_path: Path) -> None:
    with ExitStack() as owners:
        stores = [owners.enter_context(LocalStore(tmp_path / str(index))) for index in range(16)]
        for index, store in enumerate(stores):
            store.records.write_memory(_memory(f"m-{index}"))
            with ExitStack() as borrowers:
                connections = [
                    borrowers.enter_context(store._connections.connection()) for _ in range(12)
                ]
                assert len({id(connection) for connection in connections}) == 12
                assert all(
                    connection.execute("SELECT 1").fetchone()[0] == 1 for connection in connections
                )
        for index, store in enumerate(stores):
            assert [row.memory_id for row in store.records.read_memories((f"m-{index}",))] == [
                f"m-{index}"
            ]


def test_repeated_bursts_reuse_warm_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = 0
    original = Connections._open_connection

    def open_connection(owner: Connections, *, secure_delete: bool = False) -> sqlite3.Connection:
        nonlocal opened
        opened += 1
        return original(owner, secure_delete=secure_delete)

    with LocalStore(tmp_path) as store:
        monkeypatch.setattr(Connections, "_open_connection", open_connection)
        for _ in range(3):
            with ExitStack() as borrowers:
                handles = [
                    borrowers.enter_context(store._connections.connection()) for _ in range(12)
                ]
                assert len({id(handle) for handle in handles}) == 12
                assert all(handle.execute("SELECT 1").fetchone()[0] == 1 for handle in handles)
        assert opened == 12


def test_idle_eviction_preserves_recent_readers_and_directory_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connections_module, "_PROCESS_IDLE_CONNECTIONS", 4)
    monkeypatch.setattr(connections_module, "_DIRECTORY_IDLE_RESERVE", 1)
    with ExitStack() as owners:
        first, second, third = (
            owners.enter_context(LocalStore(tmp_path / str(index))) for index in range(3)
        )
        for index, store in enumerate((first, second)):
            store.update_metadata({"owner": str(index)})
            with ExitStack() as borrowers:
                for _ in range(3):
                    borrowers.enter_context(store._connections.connection())
        assert len(first._connections._pool) == 1
        with first._connections.connection() as recent:
            assert recent.execute("SELECT value FROM store_metadata").fetchone()[0] == "0"
        cold = second._connections._pool[0]
        with third._connections.connection() as fresh:
            assert fresh.execute("SELECT count(*) FROM store_metadata").fetchone()[0] == 0
        with pytest.raises(sqlite3.ProgrammingError):
            cold.execute("SELECT 1")
        assert recent in first._connections._pool
        assert sum(len(store._connections._pool) for store in (first, second, third)) == 4
        assert first.get_metadata("owner") == "0"
        assert second.get_metadata("owner") == "1"


def test_active_pressure_reclaims_idle_connections_before_rejecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connections_module, "_PROCESS_CONNECTIONS", Capacity(4))
    with LocalStore(tmp_path / "first") as first, LocalStore(tmp_path / "second") as second:
        first.update_metadata({"owner": "first"})
        second.update_metadata({"owner": "second"})
        with ExitStack() as borrowers:
            for _ in range(3):
                borrowers.enter_context(first._connections.connection())
        with ExitStack() as borrowers:
            handles = [borrowers.enter_context(second._connections.connection()) for _ in range(2)]
            assert all(
                handle.execute("SELECT value FROM store_metadata").fetchone()[0] == "second"
                for handle in handles
            )
            assert len(first._connections._pool) == 2
        assert first.get_metadata("owner") == "first"


def test_process_capacity_still_refuses_when_every_handle_is_active(tmp_path: Path) -> None:
    with ExitStack() as owners:
        stores = [owners.enter_context(LocalStore(tmp_path / str(index))) for index in range(5)]
        with ExitStack() as borrowers:
            for store in stores[:4]:
                for _ in range(32):
                    borrowers.enter_context(store._connections.connection())
            with (
                pytest.raises(RuntimeError, match="process SQLite connection capacity"),
                stores[4]._connections.connection(),
            ):
                pytest.fail("active handles were evicted to admit another borrower")
        with stores[4]._connections.connection() as recovered:
            assert recovered.execute("SELECT 1").fetchone()[0] == 1


def test_extra_idle_handles_do_not_displace_another_directorys_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connections_module, "_PROCESS_IDLE_CONNECTIONS", 4)
    monkeypatch.setattr(connections_module, "_DIRECTORY_IDLE_RESERVE", 2)
    with LocalStore(tmp_path / "first") as first, LocalStore(tmp_path / "second") as second:
        for store in (first, second):
            with ExitStack() as borrowers:
                for _ in range(2):
                    borrowers.enter_context(store._connections.connection())
        protected = tuple(second._connections._pool)
        with ExitStack() as borrowers:
            for _ in range(3):
                borrowers.enter_context(first._connections.connection())
        assert tuple(second._connections._pool) == protected
        assert all(handle.execute("SELECT 1").fetchone()[0] == 1 for handle in protected)
        assert len(first._connections._pool) == len(second._connections._pool) == 2


def test_eviction_close_keeps_directory_owned_without_blocking_warm_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connections_module, "_PROCESS_IDLE_CONNECTIONS", 2)
    monkeypatch.setattr(connections_module, "_DIRECTORY_IDLE_RESERVE", 1)
    monkeypatch.setattr(connections_module, "_CONNECTION_POOL_SIZE", 2)
    entered, release = Event(), Event()
    original = Connections._discard
    with LocalStore(tmp_path / "first") as first, LocalStore(tmp_path / "second") as second:
        with ExitStack() as borrowers:
            for _ in range(2):
                borrowers.enter_context(first._connections.connection())
        evicted = first._connections._pool[0]

        def discard(connection: sqlite3.Connection) -> None:
            if connection is evicted:
                entered.set()
                assert release.wait(5)
            original(connection)

        def borrow_second() -> None:
            with second._connections.connection() as connection:
                assert connection.execute("SELECT 1").fetchone()[0] == 1

        monkeypatch.setattr(Connections, "_discard", staticmethod(discard))
        with ThreadPoolExecutor(max_workers=2) as executor:
            returning = executor.submit(borrow_second)
            try:
                assert entered.wait(2)
                with (
                    first._connections.connection(),
                    pytest.raises(RuntimeError, match="local SQLite connection capacity"),
                    first._connections.connection(),
                ):
                    pytest.fail("eviction cleanup stopped counting against the handle limit")
                closing = executor.submit(first.close)
                with first._connections._pool_lock:
                    assert first._connections._pool_lock.wait_for(
                        lambda: first._connections._closed, timeout=1
                    )
                assert not closing.done()
                borrow_second()
                with pytest.raises(DataDirectoryInUseError):
                    LocalStore(tmp_path / "first")
            finally:
                release.set()
            returning.result(timeout=2)
            closing.result(timeout=2)
        with LocalStore(tmp_path / "first"):
            pass


def test_a_new_process_uses_fresh_idle_coordination(monkeypatch: pytest.MonkeyPatch) -> None:
    inherited = connections_module._process_idle_connections()
    monkeypatch.setattr(connections_module, "_IDLE_CONNECTIONS", inherited)
    monkeypatch.setattr(connections_module, "os", SimpleNamespace(getpid=lambda: inherited.pid + 1))

    current = connections_module._process_idle_connections()

    assert current is not inherited
    assert current.condition is not inherited.condition
    assert not current.entries
