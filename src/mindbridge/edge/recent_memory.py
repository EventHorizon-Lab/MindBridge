"""Small SQLite cache for recent cloud-derived memory on edge devices."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindbridge.contracts import EvidenceView, MemoryResult
from mindbridge.core import MemoryIntegrityError

_DEFAULT_RETENTION = timedelta(hours=24)


class SQLiteRecentMemory:
    """Keep a bounded-time, evidence-openable copy of recent cloud memory."""

    def __init__(
        self,
        database_path: Path,
        *,
        retention: timedelta = _DEFAULT_RETENTION,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if str(database_path) == ":memory:":
            raise ValueError("recent memory must use a file-backed SQLite database")
        if retention <= timedelta(0):
            raise ValueError("recent memory retention must be positive")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._retention = retention
        self._clock = clock or _utc_now
        with self._connect() as connection:
            initialize_recent_memory_tables(connection)
        os.chmod(database_path, 0o600)

    def cache_job_memories(
        self,
        tenant_id: str,
        observation_id: str,
        processing_job_id: str,
        expected_memory_ids: tuple[str, ...],
        memories: tuple[MemoryResult, ...],
    ) -> None:
        """Cache one complete job result and remove its pending marker atomically."""
        if tuple(memory.memory_id for memory in memories) != expected_memory_ids:
            raise MemoryIntegrityError("cloud job memory IDs do not match fetched memories")
        expires_at = self._now() + self._retention
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT observation_id FROM edge_processing_jobs
                WHERE tenant_id = ? AND processing_job_id = ?
                """,
                (tenant_id, processing_job_id),
            ).fetchone()
            if job is None:
                return
            if job["observation_id"] != observation_id:
                raise MemoryIntegrityError("edge processing job identity changed")
            for memory in memories:
                stored = connection.execute(
                    """
                    SELECT source_observation_id
                    FROM edge_recent_memories
                    WHERE tenant_id = ? AND memory_id = ?
                    """,
                    (tenant_id, memory.memory_id),
                ).fetchone()
                if stored is not None and stored["source_observation_id"] != observation_id:
                    raise MemoryIntegrityError("edge recent memory identity changed")
                local_memory = _with_local_evidence(
                    connection,
                    tenant_id,
                    observation_id,
                    memory,
                    expires_at,
                )
                connection.execute(
                    """
                    INSERT INTO edge_recent_memories (
                        tenant_id, memory_id, source_observation_id,
                        result_json, occurred_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (tenant_id, memory_id) DO UPDATE SET
                        result_json = excluded.result_json,
                        occurred_at = excluded.occurred_at,
                        expires_at = excluded.expires_at
                    """,
                    (
                        tenant_id,
                        memory.memory_id,
                        observation_id,
                        local_memory.model_dump_json(),
                        _utc_iso(memory.occurred_at),
                        _utc_iso(expires_at),
                    ),
                )
            connection.execute(
                """
                DELETE FROM edge_processing_jobs
                WHERE tenant_id = ? AND processing_job_id = ?
                """,
                (tenant_id, processing_job_id),
            )

    def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult | None:
        """Read one unexpired local memory without contacting the cloud."""
        with self._connect() as connection:
            self._delete_expired(connection)
            row = connection.execute(
                """
                SELECT result_json FROM edge_recent_memories
                WHERE tenant_id = ? AND memory_id = ?
                """,
                (tenant_id, memory_id),
            ).fetchone()
        return None if row is None else MemoryResult.model_validate_json(row["result_json"])

    def list_memories(self, tenant_id: str, *, limit: int = 20) -> tuple[MemoryResult, ...]:
        """List recent local memories newest-first without semantic overreach."""
        if not 1 <= limit <= 100:
            raise ValueError("recent memory limit must be 1..100")
        with self._connect() as connection:
            self._delete_expired(connection)
            rows = connection.execute(
                """
                SELECT result_json FROM edge_recent_memories
                WHERE tenant_id = ?
                ORDER BY occurred_at DESC, memory_id
                LIMIT ?
                """,
                (tenant_id, limit),
            ).fetchall()
        return tuple(MemoryResult.model_validate_json(row["result_json"]) for row in rows)

    def _delete_expired(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM edge_recent_memories WHERE expires_at <= ?",
            (_utc_iso(self._now()),),
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
            raise ValueError("recent memory clock must return a timezone-aware datetime")
        return now


def initialize_recent_memory_tables(connection: sqlite3.Connection) -> None:
    """Install the tables shared by acknowledgement, polling, and deletion."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS edge_processing_jobs (
            tenant_id TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            processing_job_id TEXT NOT NULL,
            queued_at TEXT NOT NULL,
            last_polled_at TEXT,
            PRIMARY KEY (tenant_id, processing_job_id),
            UNIQUE (tenant_id, observation_id)
        );

        CREATE TABLE IF NOT EXISTS edge_recent_memories (
            tenant_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            source_observation_id TEXT NOT NULL,
            result_json TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, memory_id)
        );

        CREATE INDEX IF NOT EXISTS edge_recent_memories_timeline_idx
            ON edge_recent_memories (tenant_id, occurred_at DESC, memory_id);
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(edge_processing_jobs)")}
    if "last_polled_at" not in columns:
        connection.execute("ALTER TABLE edge_processing_jobs ADD COLUMN last_polled_at TEXT")


def _with_local_evidence(
    connection: sqlite3.Connection,
    tenant_id: str,
    observation_id: str,
    memory: MemoryResult,
    expires_at: datetime,
) -> MemoryResult:
    evidence: list[EvidenceView] = []
    for item in memory.evidence:
        row = connection.execute(
            """
            SELECT local_path FROM edge_observation_media
            WHERE tenant_id = ? AND observation_id = ? AND media_object_id = ?
            LIMIT 1
            """,
            (tenant_id, observation_id, item.media_object_id),
        ).fetchone()
        if row is None:
            continue
        local_path = Path(row["local_path"])
        if local_path.is_file():
            evidence.append(
                item.model_copy(
                    update={
                        "media_url": local_path.as_uri(),
                        "media_url_expires_at": expires_at,
                    }
                )
            )
    return memory.model_copy(update={"evidence": tuple(evidence)})


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
