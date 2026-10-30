"""One data directory's SQLite connections: pool, pragmas, and transactions."""

from __future__ import annotations

import os
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Condition
from weakref import ReferenceType, ref

from mindbridge.infrastructure.local._capacity import (
    FILE_DESCRIPTOR_LOCK,
    Capacity,
    ProcessLock,
    file_descriptor_usage,
)
from mindbridge.infrastructure.local._lock import DataDirectoryLock
from mindbridge.infrastructure.local.store._predicates import validate_predicate_schema
from mindbridge.infrastructure.local.store._recall import RECALL_FOLD_FUNCTION, recall_fold
from mindbridge.infrastructure.local.store._schema import (
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    create_schema,
    table_names,
    user_version,
    validate_evidence_clause_schema,
    validate_text_selector_schema,
)
from mindbridge.infrastructure.local.store.errors import (
    LocalStoreCapacityError,
    LocalStoreClosedError,
    UnsupportedSchemaError,
)

# Total connections, including checked-out, opening, and idle connections.
_CONNECTION_POOL_SIZE = 32
# Idle handles stay directory-owned, with half the process budget available to active work.
_PROCESS_IDLE_CONNECTIONS = 64
_DIRECTORY_IDLE_RESERVE = 4
_PROCESS_INSTANCES = Capacity(16)
_PROCESS_CONNECTIONS = Capacity(128)


class _IdleConnections:
    """Coordinate directory-owned idle handles; never lend one across directories."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.condition = Condition()
        self.entries: OrderedDict[int, ReferenceType[Connections]] = OrderedDict()

    def take_oldest(
        self, *, minimum_retained: int = 0
    ) -> tuple[Connections, sqlite3.Connection] | None:
        # The condition protects every directory's idle list as well as this eviction order.
        for handle_id in tuple(self.entries):
            owner = self.entries[handle_id]()
            if owner is None:
                del self.entries[handle_id]
                continue
            if len(owner._pool) <= minimum_retained:
                continue
            del self.entries[handle_id]
            connection = next(handle for handle in owner._pool if id(handle) == handle_id)
            owner._pool.remove(connection)
            owner._closing_connections += 1
            return owner, connection
        return None

    def finish_eviction(self, evicted: tuple[Connections, sqlite3.Connection]) -> None:
        owner, connection = evicted
        try:
            owner._discard(connection)
        finally:
            with self.condition:
                owner._closing_connections -= 1
                self.condition.notify_all()


_IDLE_CONNECTIONS = _IdleConnections()
_IDLE_CONNECTIONS_LOCK = ProcessLock()


def _process_idle_connections() -> _IdleConnections:
    global _IDLE_CONNECTIONS
    with _IDLE_CONNECTIONS_LOCK:
        if _IDLE_CONNECTIONS.pid != os.getpid():
            _IDLE_CONNECTIONS = _IdleConnections()
        return _IDLE_CONNECTIONS


class Connections:
    """The pooled SQLite connections for one data directory, and the transactions over them.

    The directory lock lives here too: it is what makes "one live owner" true, and the
    connections are exactly what it protects.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.database_path = data_dir / "state.sqlite3"
        self._closed = False
        self._schema_ready = False
        # Idle connections only: one is in the pool exactly while nobody holds it.
        self._pool: list[sqlite3.Connection] = []
        self._idle = _process_idle_connections()
        self._pool_lock = self._idle.condition
        self._active_connections = 0
        self._closing_connections = 0
        # Every commit invalidates the corpus digest. Counting it here rather than in the caller
        # keeps the cache where writes are observable: one data directory has one live owner, so
        # a commit this process did not make cannot exist.
        self._write_generation = 0
        self._read_state = (0, 0)
        self._directory_lock = DataDirectoryLock(self.data_dir)
        if not _PROCESS_INSTANCES.acquire():
            self._directory_lock.close()
            raise LocalStoreCapacityError("local instance capacity exceeded")
        try:
            self._initialize_schema()
            self._schema_ready = True
            if os.name != "nt":
                os.chmod(self.database_path, 0o600)
        except BaseException:
            self._close_pool()
            _PROCESS_INSTANCES.release()
            self._directory_lock.close()
            self._closed = True
            raise

    @property
    def write_generation(self) -> int:
        """How many transactions have committed; a reader's cache key over the whole store."""
        return self._write_generation

    @property
    def read_generation(self) -> int | None:
        """A reusable read version, or none while a writer can change committed state."""
        # Capture one immutable pair so readers never combine a new version with an old writer
        # count, and need not contend on the process-wide connection-pool condition.
        generation, writers = self._read_state
        return None if writers else generation

    def close(self) -> None:
        """Release the pooled connections and the directory; repeated calls are harmless."""
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            while self._active_connections or self._closing_connections:
                self._pool_lock.wait()
        self._close_pool()
        self._directory_lock.close()
        _PROCESS_INSTANCES.release()

    def _initialize_schema(self) -> None:
        """Create or validate the current schema; refuse every other version."""
        with self.connection() as connection:
            version = user_version(connection)
            tables = table_names(connection)
            if version == 0:
                create_schema(connection, tables)
                version = user_version(connection)
                tables = table_names(connection)
            if version != SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"unsupported local schema version {version}; expected {SCHEMA_VERSION}. "
                    + (
                        "Re-create the data directory and re-ingest, or open it with the "
                        "MindBridge version that wrote it."
                        if version < SCHEMA_VERSION
                        else "This directory was written by a newer MindBridge."
                    )
                )
            missing_tables = REQUIRED_TABLES - tables
            if missing_tables:
                names = ", ".join(sorted(missing_tables))
                raise UnsupportedSchemaError(f"local schema is missing required tables: {names}")
            validate_evidence_clause_schema(connection)
            validate_text_selector_schema(connection)
            validate_predicate_schema(connection)

    def _open_connection(self, *, secure_delete: bool = False) -> sqlite3.Connection:
        # `check_same_thread=False` disables sqlite3's own guard, not the ownership rule it
        # approximates. A pooled connection is checked out to exactly one caller at a time, so
        # no two threads ever execute on it concurrently; without this, a connection could only
        # ever be reused by the thread that happened to open it.
        while not _PROCESS_CONNECTIONS.acquire():
            with self._pool_lock:
                evicted = self._idle.take_oldest()
            if evicted is None:
                raise LocalStoreCapacityError("process SQLite connection capacity exceeded")
            self._idle.finish_eviction(evicted)
        try:
            with FILE_DESCRIPTOR_LOCK:
                usage = file_descriptor_usage()
                if usage is not None and usage[0] + 3 + 128 > usage[1]:
                    raise LocalStoreCapacityError(
                        "SQLite open refused before file descriptor exhaustion"
                    )
                return self._connect(secure_delete=secure_delete)
        except BaseException:
            _PROCESS_CONNECTIONS.release()
            raise

    def _connect(self, *, secure_delete: bool) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if not self._schema_ready:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.create_function(RECALL_FOLD_FUNCTION, 1, recall_fold, deterministic=True)
            if secure_delete:
                # Zero-fill freed cells instead of leaving them legible in free pages. Scoped
                # to erasure: it costs extra page writes on every DELETE, and the outbox
                # acknowledges by deleting rows on the hot path.
                connection.execute("PRAGMA secure_delete = ON")
        except BaseException:
            # The connection is being discarded either way, so a failure to close it must not
            # replace the pragma failure the caller has to diagnose.
            with suppress(sqlite3.Error):
                connection.close()
            raise
        return connection

    @contextmanager
    def connection(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
        """Borrow exclusively, refusing overload before opening another connection.

        Connecting is not free: one `sqlite3_open` plus the four `PRAGMA` statements that make a
        connection usable. One `add` paid for seven of them and one `search` five, because every
        helper opened its own; in the steady state they now open none.

        A connection leaves the pool for exactly one `with` block and is returned by the same
        `finally` that used to close it, so no two callers ever hold the same one -- including
        two nested blocks, which draw two separate connections exactly as they did before.

        A connection is closed rather than returned when the block raised, because a statement
        that failed part way through can leave a transaction open that the next borrower would
        silently join. `secure_delete` and pre-schema connections are never pooled: the first is
        a persistent per-connection pragma that would make every later DELETE pay for
        zero-filling, and the second runs during schema initialization.
        """
        with self._pool_lock:
            self.require_open()
            if self._active_connections + self._closing_connections >= _CONNECTION_POOL_SIZE:
                raise LocalStoreCapacityError("local SQLite connection capacity exceeded")
            self._active_connections += 1
            idle = self._pool.pop() if self._pool else None
            if idle is not None:
                del self._idle.entries[id(idle)]
        connection = None
        reusable = False
        try:
            if idle is not None and secure_delete:
                idle.close()
                _PROCESS_CONNECTIONS.release()
                idle = None
            connection = (
                self._open_connection(secure_delete=secure_delete) if idle is None else idle
            )
            yield connection
            reusable = self._schema_ready and not secure_delete
        finally:
            self._return_connection(connection, reusable=reusable)

    def _return_connection(self, connection: sqlite3.Connection | None, *, reusable: bool) -> None:
        # Keep admission until closing or returning the actual handle has finished.
        evicted = None
        discard = False
        with self._pool_lock:
            if connection is not None:
                if reusable and not self._closed:
                    if len(self._idle.entries) >= _PROCESS_IDLE_CONNECTIONS:
                        evicted = self._idle.take_oldest(minimum_retained=_DIRECTORY_IDLE_RESERVE)
                    if len(self._idle.entries) < _PROCESS_IDLE_CONNECTIONS:
                        self._pool.append(connection)
                        self._idle.entries[id(connection)] = ref(self)
                    else:
                        discard = True
                else:
                    discard = True
            pending_cleanup = evicted is not None or discard
            if not pending_cleanup:
                self._active_connections -= 1
                self._pool_lock.notify_all()
        if pending_cleanup:
            try:
                if evicted is not None:
                    self._idle.finish_eviction(evicted)
                elif connection is not None:
                    self._discard(connection)
            finally:
                with self._pool_lock:
                    self._active_connections -= 1
                    self._pool_lock.notify_all()

    @staticmethod
    def _discard(connection: sqlite3.Connection) -> None:
        try:
            with suppress(sqlite3.Error):
                connection.close()
        finally:
            _PROCESS_CONNECTIONS.release()

    def _close_pool(self) -> None:
        with self._pool_lock:
            connections = tuple(self._pool)
            self._pool.clear()
            for connection in connections:
                del self._idle.entries[id(connection)]
        for connection in connections:
            self._discard(connection)

    @contextmanager
    def transaction(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
        with self.connection(secure_delete=secure_delete) as connection:
            connection.execute("BEGIN IMMEDIATE")
            with self._pool_lock:
                generation, writers = self._read_state
                self._read_state = (generation, writers + 1)
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
                self._write_generation += 1
            finally:
                # Keep caches unavailable through commit/rollback, including the gap between
                # SQLite returning and Python updating its counters. A failed transaction also
                # invalidates read reuse: a commit error must never leave an assumed old view.
                with self._pool_lock:
                    generation, writers = self._read_state
                    self._read_state = (generation + 1, writers - 1)

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold one WAL snapshot across multi-query hydration."""
        with self.connection() as connection:
            connection.execute("BEGIN")
            try:
                yield connection
            finally:
                connection.rollback()

    def require_open(self) -> None:
        if self._closed:
            raise LocalStoreClosedError("local store is closed")
