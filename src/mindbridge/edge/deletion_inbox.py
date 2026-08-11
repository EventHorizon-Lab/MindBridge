"""Durable cloud tombstones and local evidence erasure for edge devices."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from mindbridge.contracts import DeletionPage, DeletionTombstoneView, ObserveRequest
from mindbridge.core import ForgetTargetType, MemoryIntegrityError, derive_observation_id
from mindbridge.edge.identity_schema import initialize_identity_tables


class SQLiteDeletionInbox:
    """Apply ordered cloud deletion barriers before an edge device can re-upload."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._database_path = database_path
        self._clock = clock or _utc_now
        with self._connect() as connection:
            initialize_deletion_tables(connection)
            initialize_identity_tables(connection)

    def apply_page(self, tenant_id: str, page: DeletionPage) -> int:
        """Erase local targets and advance the cursor in one recoverable transaction."""
        if page.next_cursor is not None and (
            not page.items or page.next_cursor != page.items[-1].tombstone_id
        ):
            raise MemoryIntegrityError("deletion page cursor does not match its final item")
        received_at = self._now().isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for tombstone in page.items:
                self._apply_tombstone(connection, tenant_id, tombstone, received_at)
            if page.items:
                self._write_cursor(
                    connection,
                    tenant_id,
                    page.items[-1].tombstone_id,
                    received_at,
                )
        return len(page.items)

    def read_cursor(self, tenant_id: str) -> str | None:
        """Return the last durably applied cloud tombstone ID for one tenant."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cursor FROM edge_deletion_cursors WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return None if row is None else str(row["cursor"])

    def tenant_ids(self) -> tuple[str, ...]:
        """List local tenants that need deletion polling, including empty outboxes."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tenant_id FROM edge_observation_outbox
                UNION SELECT tenant_id FROM edge_sync_watermarks
                UNION SELECT tenant_id FROM edge_observation_media
                UNION SELECT tenant_id FROM edge_deletion_cursors
                UNION SELECT tenant_id FROM edge_identity_templates
                ORDER BY tenant_id
                """
            ).fetchall()
        return tuple(str(row["tenant_id"]) for row in rows)

    def _apply_tombstone(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        tombstone: DeletionTombstoneView,
        received_at: str,
    ) -> None:
        self._require_consistent_identity(connection, tenant_id, tombstone)
        if tombstone.target_type is ForgetTargetType.OBSERVATION:
            self._erase_observation(connection, tenant_id, tombstone.target_id)
        try:
            connection.execute(
                """
                INSERT INTO edge_deletion_tombstones (
                    tenant_id, tombstone_id, target_type, target_id,
                    propagation_state, requested_at, completed_at, error_code, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, tombstone_id) DO UPDATE SET
                    propagation_state = excluded.propagation_state,
                    completed_at = excluded.completed_at,
                    error_code = excluded.error_code,
                    received_at = excluded.received_at
                """,
                (
                    tenant_id,
                    tombstone.tombstone_id,
                    tombstone.target_type.value,
                    tombstone.target_id,
                    tombstone.propagation_state.value,
                    tombstone.requested_at.isoformat(),
                    tombstone.completed_at.isoformat()
                    if tombstone.completed_at is not None
                    else None,
                    tombstone.error_code,
                    received_at,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise MemoryIntegrityError("edge tombstone conflicts with retained identity") from error

    def _erase_observation(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        observation_id: str,
    ) -> None:
        rows = connection.execute(
            """
            SELECT media.local_path
            FROM edge_observation_media AS media
            WHERE media.tenant_id = ? AND media.observation_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM edge_observation_media AS other
                  WHERE other.tenant_id = media.tenant_id
                    AND other.local_path = media.local_path
                    AND other.observation_id <> media.observation_id
              )
            """,
            (tenant_id, observation_id),
        ).fetchall()
        for row in rows:
            Path(row["local_path"]).unlink(missing_ok=True)
        pending = connection.execute(
            """
            SELECT outbox_id, request_json FROM edge_observation_outbox
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
        for row in pending:
            request = ObserveRequest.model_validate_json(row["request_json"])
            if (
                derive_observation_id(
                    request.tenant_id,
                    request.device_id,
                    request.boot_id,
                    request.sequence,
                )
                == observation_id
            ):
                connection.execute(
                    "DELETE FROM edge_observation_outbox WHERE outbox_id = ?",
                    (row["outbox_id"],),
                )
        connection.execute(
            """
            DELETE FROM edge_observation_media
            WHERE tenant_id = ? AND observation_id = ?
            """,
            (tenant_id, observation_id),
        )
        connection.execute(
            """
            DELETE FROM edge_identity_templates
            WHERE tenant_id = ? AND source_observation_id = ?
            """,
            (tenant_id, observation_id),
        )

    @staticmethod
    def _require_consistent_identity(
        connection: sqlite3.Connection,
        tenant_id: str,
        tombstone: DeletionTombstoneView,
    ) -> None:
        rows = connection.execute(
            """
            SELECT tombstone_id, target_type, target_id
            FROM edge_deletion_tombstones
            WHERE tenant_id = ?
              AND (tombstone_id = ? OR (target_type = ? AND target_id = ?))
            """,
            (
                tenant_id,
                tombstone.tombstone_id,
                tombstone.target_type.value,
                tombstone.target_id,
            ),
        ).fetchall()
        if any(
            row["tombstone_id"] != tombstone.tombstone_id
            or row["target_type"] != tombstone.target_type.value
            or row["target_id"] != tombstone.target_id
            for row in rows
        ):
            raise MemoryIntegrityError("edge tombstone identity changed")

    @staticmethod
    def _write_cursor(
        connection: sqlite3.Connection,
        tenant_id: str,
        cursor: str,
        synced_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO edge_deletion_cursors (tenant_id, cursor, synced_at)
            VALUES (?, ?, ?)
            ON CONFLICT (tenant_id) DO UPDATE SET
                cursor = excluded.cursor,
                synced_at = excluded.synced_at
            """,
            (tenant_id, cursor, synced_at),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("edge deletion clock must return a timezone-aware datetime")
        return now


def initialize_deletion_tables(connection: sqlite3.Connection) -> None:
    """Install the SQLite tables shared by capture and deletion synchronization."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS edge_observation_media (
            tenant_id TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            media_object_id TEXT NOT NULL,
            local_path TEXT NOT NULL,
            PRIMARY KEY (tenant_id, observation_id, media_object_id)
        );

        CREATE TABLE IF NOT EXISTS edge_deletion_tombstones (
            tenant_id TEXT NOT NULL,
            tombstone_id TEXT NOT NULL,
            target_type TEXT NOT NULL CHECK (target_type IN ('memory_record', 'observation')),
            target_id TEXT NOT NULL,
            propagation_state TEXT NOT NULL CHECK (
                propagation_state IN ('pending', 'propagating', 'complete', 'failed')
            ),
            requested_at TEXT NOT NULL,
            completed_at TEXT,
            error_code TEXT,
            received_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, tombstone_id),
            UNIQUE (tenant_id, target_type, target_id)
        );

        CREATE TABLE IF NOT EXISTS edge_deletion_cursors (
            tenant_id TEXT PRIMARY KEY,
            cursor TEXT NOT NULL,
            synced_at TEXT NOT NULL
        );
        """
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
