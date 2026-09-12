"""One data directory's SQLite connections: pool, pragmas, and transactions."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock

from mindbridge.infrastructure.local._lock import DataDirectoryLock
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
    LocalStoreClosedError,
    UnsupportedSchemaError,
)

# Idle connections retained for reuse. Concurrency above this many simultaneous readers falls
# back to the previous open-and-close behaviour rather than growing the pool without bound.
_CONNECTION_POOL_SIZE = 32


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
        self._pool_lock = Lock()
        # Every commit invalidates the corpus digest. Counting it here rather than in the caller
        # keeps the cache where writes are observable: one data directory has one live owner, so
        # a commit this process did not make cannot exist.
        self._write_generation = 0
        self._directory_lock = DataDirectoryLock(self.data_dir)
        try:
            self._initialize_schema()
            self._schema_ready = True
            if os.name != "nt":
                os.chmod(self.database_path, 0o600)
        except BaseException:
            self._directory_lock.close()
            self._closed = True
            raise

    @property
    def write_generation(self) -> int:
        """How many transactions have committed; a reader's cache key over the whole store."""
        return self._write_generation

    def close(self) -> None:
        """Release the pooled connections and the directory; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        self._close_pool()
        self._directory_lock.close()

    def _initialize_schema(self) -> None:
        """Create the current schema in an empty directory, or refuse any other version.

        This build has no upgrade path. A directory written by an older MindBridge is refused
        rather than converted, because SQLite is the authoritative copy: re-create the directory
        and re-ingest, or open it with the version that wrote it.
        """
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

    def _open_connection(self, *, secure_delete: bool = False) -> sqlite3.Connection:
        # `check_same_thread=False` disables sqlite3's own guard, not the ownership rule it
        # approximates. A pooled connection is checked out to exactly one caller at a time, so
        # no two threads ever execute on it concurrently; without this, a connection could only
        # ever be reused by the thread that happened to open it.
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
        """Check one connection out of the pool, opening one when the pool is empty.

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
        zero-filling, and the second runs during migration.
        """
        self.require_open()
        if secure_delete or not self._schema_ready:
            connection = self._open_connection(secure_delete=secure_delete)
            try:
                yield connection
            except BaseException:
                with suppress(sqlite3.Error):
                    connection.close()
                raise
            else:
                connection.close()
            return
        with self._pool_lock:
            idle = self._pool.pop() if self._pool else None
        connection = self._open_connection() if idle is None else idle
        try:
            yield connection
        except BaseException:
            with suppress(sqlite3.Error):
                connection.close()
            raise
        else:
            self._release_connection(connection)

    def _release_connection(self, connection: sqlite3.Connection) -> None:
        with self._pool_lock:
            pooled = not self._closed and len(self._pool) < _CONNECTION_POOL_SIZE
            if pooled:
                self._pool.append(connection)
        if not pooled:
            with suppress(sqlite3.Error):
                connection.close()

    def _close_pool(self) -> None:
        with self._pool_lock:
            connections = tuple(self._pool)
            self._pool.clear()
        for connection in connections:
            with suppress(sqlite3.Error):
                connection.close()

    @contextmanager
    def transaction(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
        with self.connection(secure_delete=secure_delete) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
                self._write_generation += 1

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
