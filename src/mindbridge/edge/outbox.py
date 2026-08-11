"""File-backed SQLite outbox for offline-safe edge observations."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import StringConstraints, TypeAdapter, model_validator

from mindbridge.contracts import ContractModel, Identifier, ObservationReceipt, ObserveRequest
from mindbridge.core import (
    IdempotencyConflictError,
    MemoryDeletedError,
    MemoryIntegrityError,
    derive_observation_id,
    derive_stable_id,
)
from mindbridge.edge.deletion_inbox import initialize_deletion_tables

_SCHEMA_VERSION = 2


class EdgeMediaFile(ContractModel):
    """One completed local media file corresponding to request metadata."""

    media_object_id: Identifier
    local_path: Path
    content_type: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=3,
            max_length=127,
            pattern=r"^[!#$&^_.+\-\w]+/[!#$&^_.+\-\w]+$",
        ),
    ]

    @model_validator(mode="after")
    def require_absolute_path(self) -> EdgeMediaFile:
        """Keep queued paths stable across service restarts and working directories."""
        if not self.local_path.is_absolute():
            raise ValueError("edge media local_path must be absolute")
        return self


_MEDIA_FILES = TypeAdapter(tuple[EdgeMediaFile, ...])


@dataclass(frozen=True, slots=True)
class EdgeObservationOutboxItem:
    """One durable observation awaiting media upload or cloud acknowledgement."""

    outbox_id: str
    request: ObserveRequest
    media_files: tuple[EdgeMediaFile, ...]
    media_uploaded: bool
    attempts: int
    last_error_code: str | None


@dataclass(frozen=True, slots=True)
class EdgeSyncWatermark:
    """Latest cloud-acknowledged sequence for one tenant device boot."""

    tenant_id: str
    device_id: str
    boot_id: str
    sequence: int
    observation_id: str
    processing_job_id: str
    synced_at: datetime


class SQLiteObservationOutbox:
    """Persist edge observations until immutable media and metadata reach the cloud."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if str(database_path) == ":memory:":
            raise ValueError("edge outbox must use a file-backed SQLite database")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._clock = clock or _utc_now
        self._initialize()

    def enqueue(
        self,
        request: ObserveRequest,
        media_files: tuple[EdgeMediaFile, ...],
    ) -> bool:
        """Durably queue a new sequence; return false for an exact prior sequence."""
        request_media_ids = tuple(media.media_object_id for media in request.media_objects)
        file_media_ids = tuple(media.media_object_id for media in media_files)
        if request_media_ids != file_media_ids:
            raise ValueError("edge media files must match request media objects in order")
        outbox_id = derive_stable_id(
            "edge_observation",
            request.tenant_id,
            request.device_id,
            request.boot_id,
            str(request.sequence),
        )
        observation_id = derive_observation_id(
            request.tenant_id,
            request.device_id,
            request.boot_id,
            request.sequence,
        )
        request_json = request.model_dump_json()
        media_files_json = _MEDIA_FILES.dump_json(media_files).decode("utf-8")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tombstone = connection.execute(
                """
                SELECT 1 FROM edge_deletion_tombstones
                WHERE tenant_id = ? AND target_type = 'observation' AND target_id = ?
                """,
                (request.tenant_id, observation_id),
            ).fetchone()
            if tombstone is not None:
                raise MemoryDeletedError("edge observation was explicitly forgotten")
            watermark = connection.execute(
                """
                SELECT sequence
                FROM edge_sync_watermarks
                WHERE tenant_id = ? AND device_id = ? AND boot_id = ?
                """,
                (request.tenant_id, request.device_id, request.boot_id),
            ).fetchone()
            if watermark is not None and request.sequence <= int(watermark["sequence"]):
                return False
            existing = connection.execute(
                """
                SELECT request_json, media_files_json
                FROM edge_observation_outbox
                WHERE outbox_id = ?
                """,
                (outbox_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["request_json"] == request_json
                    and existing["media_files_json"] == media_files_json
                ):
                    return False
                raise IdempotencyConflictError(
                    "edge observation sequence was reused with different content"
                )
            for media_file in media_files:
                if not media_file.local_path.is_file():
                    raise FileNotFoundError(media_file.local_path)
            connection.execute(
                """
                INSERT INTO edge_observation_outbox (
                    outbox_id,
                    tenant_id,
                    device_id,
                    boot_id,
                    sequence,
                    request_json,
                    media_files_json,
                    media_uploaded,
                    attempts,
                    last_error_code,
                    enqueued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, ?)
                """,
                (
                    outbox_id,
                    request.tenant_id,
                    request.device_id,
                    request.boot_id,
                    request.sequence,
                    request_json,
                    media_files_json,
                    self._now().isoformat(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO edge_observation_media (
                    tenant_id, observation_id, media_object_id, local_path
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    (
                        request.tenant_id,
                        observation_id,
                        media_file.media_object_id,
                        str(media_file.local_path),
                    )
                    for media_file in media_files
                ),
            )
        return True

    def next_pending(self) -> EdgeObservationOutboxItem | None:
        """Return the oldest unsynchronized observation without removing it."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM edge_observation_outbox
                ORDER BY enqueued_at, sequence
                LIMIT 1
                """
            ).fetchone()
        return _outbox_item(row) if row is not None else None

    def mark_media_uploaded(self, outbox_id: str) -> None:
        """Avoid retransmitting media when only the observation API needs retrying."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE edge_observation_outbox
                SET media_uploaded = 1
                WHERE outbox_id = ?
                """,
                (outbox_id,),
            )

    def record_failure(self, outbox_id: str, error_code: str) -> None:
        """Persist a sanitized failure class without recording media or exception text."""
        error_code = error_code.strip()
        if not error_code or len(error_code) > 255:
            raise ValueError("edge outbox error_code must contain 1 to 255 characters")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE edge_observation_outbox
                SET attempts = attempts + 1, last_error_code = ?
                WHERE outbox_id = ?
                """,
                (error_code, outbox_id),
            )

    def acknowledge(
        self,
        item: EdgeObservationOutboxItem,
        receipt: ObservationReceipt,
    ) -> None:
        """Advance the watermark and remove the outbox row in one transaction."""
        expected_key = item.request.idempotency_key
        if expected_key is not None and receipt.idempotency_key != expected_key:
            raise MemoryIntegrityError("cloud receipt returned an unexpected idempotency key")
        synced_at = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stored = connection.execute(
                "SELECT request_json FROM edge_observation_outbox WHERE outbox_id = ?",
                (item.outbox_id,),
            ).fetchone()
            if stored is None:
                watermark = connection.execute(
                    """
                    SELECT sequence
                    FROM edge_sync_watermarks
                    WHERE tenant_id = ? AND device_id = ? AND boot_id = ?
                    """,
                    (
                        item.request.tenant_id,
                        item.request.device_id,
                        item.request.boot_id,
                    ),
                ).fetchone()
                if watermark is not None and int(watermark["sequence"]) >= item.request.sequence:
                    return
                raise MemoryIntegrityError("edge outbox item disappeared before acknowledgement")
            if stored["request_json"] != item.request.model_dump_json():
                raise MemoryIntegrityError("edge outbox item changed before acknowledgement")
            connection.execute(
                """
                INSERT INTO edge_sync_watermarks (
                    tenant_id,
                    device_id,
                    boot_id,
                    sequence,
                    observation_id,
                    processing_job_id,
                    synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, device_id, boot_id) DO UPDATE SET
                    sequence = excluded.sequence,
                    observation_id = excluded.observation_id,
                    processing_job_id = excluded.processing_job_id,
                    synced_at = excluded.synced_at
                WHERE excluded.sequence >= edge_sync_watermarks.sequence
                """,
                (
                    item.request.tenant_id,
                    item.request.device_id,
                    item.request.boot_id,
                    item.request.sequence,
                    receipt.observation_id,
                    receipt.processing_job_id,
                    synced_at.isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM edge_observation_outbox WHERE outbox_id = ?",
                (item.outbox_id,),
            )

    def read_watermark(
        self,
        tenant_id: str,
        device_id: str,
        boot_id: str,
    ) -> EdgeSyncWatermark | None:
        """Read the latest durable cloud acknowledgement for local retention decisions."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM edge_sync_watermarks
                WHERE tenant_id = ? AND device_id = ? AND boot_id = ?
                """,
                (tenant_id, device_id, boot_id),
            ).fetchone()
        if row is None:
            return None
        return EdgeSyncWatermark(
            tenant_id=row["tenant_id"],
            device_id=row["device_id"],
            boot_id=row["boot_id"],
            sequence=row["sequence"],
            observation_id=row["observation_id"],
            processing_job_id=row["processing_job_id"],
            synced_at=datetime.fromisoformat(row["synced_at"]),
        )

    def pending_count(self) -> int:
        """Return queue depth for health checks and disk-pressure policy."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM edge_observation_outbox"
            ).fetchone()
        assert row is not None
        return int(row["count"])

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, _SCHEMA_VERSION}:
                raise RuntimeError(f"unsupported edge outbox schema version {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS edge_observation_outbox (
                    outbox_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    boot_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    request_json TEXT NOT NULL,
                    media_files_json TEXT NOT NULL,
                    media_uploaded INTEGER NOT NULL CHECK (media_uploaded IN (0, 1)),
                    attempts INTEGER NOT NULL CHECK (attempts >= 0),
                    last_error_code TEXT,
                    enqueued_at TEXT NOT NULL,
                    UNIQUE (tenant_id, device_id, boot_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS edge_sync_watermarks (
                    tenant_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    boot_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 0),
                    observation_id TEXT NOT NULL,
                    processing_job_id TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, device_id, boot_id)
                );
                """
            )
            initialize_deletion_tables(connection)
            self._backfill_observation_media(connection)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        os.chmod(self._database_path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _backfill_observation_media(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT request_json, media_files_json FROM edge_observation_outbox"
        ).fetchall()
        for row in rows:
            request = ObserveRequest.model_validate_json(row["request_json"])
            media_files = _MEDIA_FILES.validate_json(row["media_files_json"])
            observation_id = derive_observation_id(
                request.tenant_id,
                request.device_id,
                request.boot_id,
                request.sequence,
            )
            connection.executemany(
                """
                INSERT INTO edge_observation_media (
                    tenant_id, observation_id, media_object_id, local_path
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    (
                        request.tenant_id,
                        observation_id,
                        media_file.media_object_id,
                        str(media_file.local_path),
                    )
                    for media_file in media_files
                ),
            )

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("edge outbox clock must return a timezone-aware datetime")
        return now


def _outbox_item(row: sqlite3.Row) -> EdgeObservationOutboxItem:
    return EdgeObservationOutboxItem(
        outbox_id=row["outbox_id"],
        request=ObserveRequest.model_validate_json(row["request_json"]),
        media_files=_MEDIA_FILES.validate_json(row["media_files_json"]),
        media_uploaded=bool(row["media_uploaded"]),
        attempts=row["attempts"],
        last_error_code=row["last_error_code"],
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
