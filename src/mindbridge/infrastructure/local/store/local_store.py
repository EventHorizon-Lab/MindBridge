"""SQLite source of truth for the local MindBridge runtime.

`LocalStore` owns one data directory and exposes its storage surface by table family. The
families share one connection pool and one directory lock. Each family is the writer of its own
tables; a transaction that spans families -- a formation commit that writes records, evidence,
embeddings, and outbox rows together -- lives in the family that owns the primary write and
calls the other families' connection-level functions inside it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mindbridge.infrastructure.local.store._codec import datetime_text, row_text
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store.captures import CaptureQueue
from mindbridge.infrastructure.local.store.control import OperationLog
from mindbridge.infrastructure.local.store.identities import IdentityRegistry
from mindbridge.infrastructure.local.store.index import IndexOutbox
from mindbridge.infrastructure.local.store.media import MediaAnalyses
from mindbridge.infrastructure.local.store.recall import RecallReads
from mindbridge.infrastructure.local.store.records import MemoryRecords
from mindbridge.infrastructure.local.store.rows import require_identifier
from mindbridge.infrastructure.local.store.semantics import Semantics


class LocalStore:
    """Own one data directory and expose its transactional storage surface by table family."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._connections = Connections(self.data_dir)
        self.database_path = self._connections.database_path
        self.records = MemoryRecords(
            connections=self._connections,
        )
        self.index = IndexOutbox(
            connections=self._connections,
        )
        self.media = MediaAnalyses(
            connections=self._connections,
        )
        self.captures = CaptureQueue(
            connections=self._connections,
        )
        self.semantics = Semantics(
            connections=self._connections,
            records=self.records,
        )
        self.identities = IdentityRegistry(
            connections=self._connections,
            records=self.records,
        )
        self.recall = RecallReads(
            connections=self._connections,
            records=self.records,
        )
        self.control = OperationLog(
            connections=self._connections,
        )

    def __enter__(self) -> LocalStore:
        self._connections.require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the pooled connections and the directory; repeated calls are harmless."""
        self._connections.close()

    def set_metadata(self, key: str, value: str) -> None:
        """Set one store-level compatibility value."""
        require_identifier(key, "metadata key")
        with self._connections.transaction() as connection:
            connection.execute(
                """
                INSERT INTO store_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, datetime_text(datetime.now(timezone.utc))),
            )

    def get_metadata(self, key: str) -> str | None:
        """Read one store-level compatibility value."""
        require_identifier(key, "metadata key")
        with self._connections.connection() as connection:
            row = connection.execute(
                "SELECT value FROM store_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return None if row is None else row_text(row, "value")

    def delete_metadata(self, key: str) -> bool:
        """Delete one store-level compatibility value."""
        require_identifier(key, "metadata key")
        with self._connections.transaction() as connection:
            cursor = connection.execute("DELETE FROM store_metadata WHERE key = ?", (key,))
        return cursor.rowcount > 0
