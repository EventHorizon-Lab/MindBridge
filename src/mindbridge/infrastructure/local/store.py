"""SQLite source of truth for the local MindBridge runtime."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NoReturn

from mindbridge.infrastructure.local._lock import DataDirectoryLock
from mindbridge.models.base import SpeechAnalysis
from mindbridge.types import SpeakerSegment

_SCHEMA_VERSION = 5
_SQLITE_PARAMETER_BATCH = 900
_MEMORY_MODALITIES = frozenset({"text", "image", "video", "audio", "omni"})
_MEMORY_TYPES = frozenset({"semantic", "episodic", "procedural"})
_ASSET_MODALITIES = frozenset({"image", "video", "audio"})
_SHA256_HEX_LENGTH = 64
_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")
_REQUIRED_V1_TABLES = frozenset(
    {"embeddings", "memory_records", "search_index_queue", "store_metadata"}
)
_REQUIRED_TABLES = frozenset(
    {
        "embeddings",
        "media_assets",
        "memory_assets",
        "memory_records",
        "search_index_queue",
        "speaker_identities",
        "speech_analyses",
        "speech_segments",
        "store_metadata",
    }
)
_ASSET_SCHEMA = """
CREATE TABLE media_assets (
    asset_id TEXT PRIMARY KEY,
    modality TEXT NOT NULL CHECK (modality IN ('image', 'video', 'audio')),
    mime_type TEXT NOT NULL CHECK (length(trim(mime_type)) > 0),
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    sha256 TEXT NOT NULL UNIQUE CHECK (
        length(sha256) = 64
        AND sha256 NOT GLOB '*[^0-9a-f]*'
        AND asset_id = sha256
    ),
    relative_path TEXT NOT NULL UNIQUE CHECK (length(trim(relative_path)) > 0),
    name TEXT,
    transcript TEXT,
    created_at TEXT NOT NULL,
    CHECK (name IS NULL OR length(trim(name)) > 0)
);

CREATE TABLE memory_assets (
    memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    asset_id TEXT NOT NULL REFERENCES media_assets (asset_id) ON DELETE RESTRICT,
    PRIMARY KEY (memory_id, position)
);

CREATE INDEX memory_assets_asset_idx ON memory_assets (asset_id);
"""

_INDEX_TRIGGERS = """
CREATE TRIGGER embeddings_queue_insert
AFTER INSERT ON embeddings
BEGIN
    INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
    VALUES (NEW.embedding_id, 'upsert', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
END;

CREATE TRIGGER embeddings_queue_update
AFTER UPDATE ON embeddings
BEGIN
    INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
    VALUES (NEW.embedding_id, 'upsert', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
END;

CREATE TRIGGER embeddings_queue_delete
AFTER DELETE ON embeddings
BEGIN
    INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
    VALUES (OLD.embedding_id, 'delete', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
END;
"""

_SPEECH_SCHEMA = """
CREATE TABLE speech_analyses (
    asset_id TEXT PRIMARY KEY REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    transcript TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE speaker_identities (
    speaker_id TEXT PRIMARY KEY CHECK (length(trim(speaker_id)) > 0),
    name TEXT CHECK (name IS NULL OR length(trim(name)) > 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    centroid BLOB NOT NULL CHECK (length(centroid) = dimension * 4),
    observations INTEGER NOT NULL CHECK (observations > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
);

CREATE INDEX speaker_identities_space_idx
    ON speaker_identities (space_id, dimension, speaker_id);

CREATE TABLE speech_segments (
    asset_id TEXT NOT NULL REFERENCES speech_analyses (asset_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    transcript TEXT NOT NULL CHECK (length(trim(transcript)) > 0),
    speaker_id TEXT REFERENCES speaker_identities (speaker_id) ON DELETE SET NULL,
    identity_score REAL CHECK (
        identity_score IS NULL OR (identity_score >= 0.0 AND identity_score <= 1.0)
    ),
    PRIMARY KEY (asset_id, position)
);

CREATE INDEX speech_segments_speaker_idx ON speech_segments (speaker_id);
"""

_SCHEMA_V5 = f"""
BEGIN IMMEDIATE;

CREATE TABLE memory_records (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'video', 'audio', 'omni')),
    memory_type TEXT NOT NULL DEFAULT 'semantic'
        CHECK (memory_type IN ('semantic', 'episodic', 'procedural')),
    metadata_json TEXT NOT NULL,
    occurred_at TEXT,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count BETWEEN 0 AND 20),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
);

CREATE INDEX memory_records_created_idx
    ON memory_records (created_at DESC, memory_id DESC);

CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    object_part INTEGER NOT NULL DEFAULT 0 CHECK (object_part >= 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    task TEXT NOT NULL CHECK (length(trim(task)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    vector BLOB NOT NULL CHECK (length(vector) = dimension * 4),
    created_at TEXT NOT NULL,
    UNIQUE (memory_id, object_part, model_id, task)
);

CREATE INDEX embeddings_memory_idx ON embeddings (memory_id);

{_ASSET_SCHEMA}
{_SPEECH_SCHEMA}

CREATE TABLE store_metadata (
    key TEXT PRIMARY KEY CHECK (length(trim(key)) > 0),
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE search_index_queue (
    operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('upsert', 'delete')),
    enqueued_at TEXT NOT NULL
);

CREATE INDEX search_index_queue_order_idx
    ON search_index_queue (operation_id);

{_INDEX_TRIGGERS}

PRAGMA user_version = 5;
COMMIT;
"""

_MIGRATE_V1_TO_V2 = f"""
BEGIN IMMEDIATE;

DROP TRIGGER IF EXISTS embeddings_queue_insert;
DROP TRIGGER IF EXISTS embeddings_queue_update;
DROP TRIGGER IF EXISTS embeddings_queue_delete;
DROP INDEX IF EXISTS embeddings_memory_idx;
DROP INDEX IF EXISTS memory_records_created_idx;

ALTER TABLE embeddings RENAME TO embeddings_v1;
ALTER TABLE memory_records RENAME TO memory_records_v1;

CREATE TABLE memory_records (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'video', 'audio', 'omni')),
    metadata_json TEXT NOT NULL,
    occurred_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
);

INSERT INTO memory_records (
    memory_id, content, modality, metadata_json, occurred_at, created_at, updated_at
)
SELECT memory_id, content, 'text', metadata_json, occurred_at, created_at, updated_at
FROM memory_records_v1;

CREATE INDEX memory_records_created_idx
    ON memory_records (created_at DESC, memory_id DESC);

CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    object_part INTEGER NOT NULL DEFAULT 0 CHECK (object_part >= 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    task TEXT NOT NULL CHECK (length(trim(task)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    vector BLOB NOT NULL CHECK (length(vector) = dimension * 4),
    created_at TEXT NOT NULL,
    UNIQUE (memory_id, object_part, model_id, task)
);

INSERT INTO embeddings (
    embedding_id, memory_id, object_part, model_id, space_id, task,
    dimension, normalized, vector, created_at
)
SELECT embedding_id, memory_id, object_part, model_id, space_id, task,
       dimension, normalized, vector, created_at
FROM embeddings_v1;

CREATE INDEX embeddings_memory_idx ON embeddings (memory_id);

DROP TABLE embeddings_v1;
DROP TABLE memory_records_v1;

{_ASSET_SCHEMA}
{_INDEX_TRIGGERS}

PRAGMA user_version = 2;
COMMIT;
"""

_MIGRATE_V2_TO_V4 = f"""
BEGIN IMMEDIATE;
{_SPEECH_SCHEMA}
PRAGMA user_version = 4;
COMMIT;
"""

_MIGRATE_V3_TO_V4 = """
BEGIN IMMEDIATE;
ALTER TABLE speaker_identities
ADD COLUMN name TEXT CHECK (name IS NULL OR length(trim(name)) > 0);
PRAGMA user_version = 4;
COMMIT;
"""


class LocalStoreClosedError(RuntimeError):
    """Raised when a closed local store is used."""


class UnsupportedSchemaError(RuntimeError):
    """Raised when the data directory has an unknown or incomplete schema."""


@dataclass(frozen=True, slots=True)
class StoredAsset:
    """Immutable metadata for one content-addressed local media asset."""

    asset_id: str
    modality: str
    mime_type: str
    size_bytes: int
    sha256: str
    relative_path: str
    created_at: datetime
    name: str | None = None
    transcript: str | None = None

    def __post_init__(self) -> None:
        digest = _sha256(self.sha256)
        if self.asset_id != digest:
            raise ValueError("asset_id must equal the asset sha256")
        object.__setattr__(self, "sha256", digest)
        modality = _modality(self.modality, asset=True)
        object.__setattr__(self, "modality", modality)
        mime_type = _mime_type(self.mime_type, modality)
        object.__setattr__(self, "mime_type", mime_type)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("size_bytes must be a positive integer")
        expected_path = _asset_relative_path(digest)
        if self.relative_path != expected_path:
            raise ValueError(f"relative_path must be {expected_path!r}")
        if self.name is not None:
            object.__setattr__(self, "name", validate_asset_name(self.name))
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise ValueError("transcript must be text or None")
        _require_aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class StoredMemory:
    """The authoritative local representation of one memory."""

    memory_id: str
    content: str
    metadata_json: str
    created_at: datetime
    updated_at: datetime
    occurred_at: datetime | None = None
    modality: str = "text"
    memory_type: str = "semantic"
    assets: tuple[StoredAsset, ...] = ()
    last_accessed_at: datetime | None = None
    access_count: int = 0

    def __post_init__(self) -> None:
        _require_identifier(self.memory_id, "memory_id")
        object.__setattr__(self, "modality", _modality(self.modality, asset=False))
        if self.memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        if not isinstance(self.assets, tuple) or not all(
            isinstance(asset, StoredAsset) for asset in self.assets
        ):
            raise ValueError("assets must be a tuple of StoredAsset values")
        asset_modalities = {asset.modality for asset in self.assets}
        if self.modality == "text" and self.assets:
            raise ValueError("text memories cannot contain media assets")
        if self.modality in _ASSET_MODALITIES and asset_modalities != {self.modality}:
            raise ValueError(f"{self.modality} memories require only {self.modality} assets")
        if self.modality == "omni" and len(asset_modalities) < 2:
            raise ValueError("omni memories require at least two media modalities")
        if not self.content.strip() and not self.assets:
            raise ValueError("a memory must contain text or at least one asset")
        object.__setattr__(self, "metadata_json", _canonical_object_json(self.metadata_json))
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.occurred_at is not None:
            _require_aware(self.occurred_at, "occurred_at")
        if self.last_accessed_at is not None:
            _require_aware(self.last_accessed_at, "last_accessed_at")
        _access_count(self.access_count)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    """An FP32 vector retained in SQLite so the search index is rebuildable."""

    embedding_id: str
    memory_id: str
    values: tuple[float, ...]
    model_id: str
    space_id: str
    task: str
    created_at: datetime
    object_part: int = 0
    normalized: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.embedding_id, "embedding_id")
        _require_identifier(self.memory_id, "memory_id")
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.space_id, "space_id")
        _require_identifier(self.task, "task")
        _require_aware(self.created_at, "created_at")
        if self.object_part < 0:
            raise ValueError("object_part must not be negative")
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError("embedding values must be finite and non-empty")
        if self.normalized and not math.isclose(
            math.sqrt(sum(value * value for value in self.values)),
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-6,
        ):
            raise ValueError("normalized embedding must have unit length")


@dataclass(frozen=True, slots=True)
class IndexOperation:
    """One durable mutation awaiting a successful search-index flush."""

    operation_id: int
    embedding_id: str
    action: Literal["upsert", "delete"]


@dataclass(frozen=True, slots=True)
class IndexDocument:
    """Current SQLite payload for an embedding queued for indexing."""

    embedding: StoredEmbedding
    content: str
    metadata_json: str
    memory_type: str = "semantic"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        if self.occurred_at is not None:
            _require_aware(self.occurred_at, "occurred_at")


class LocalStore:
    """Own one data directory and expose its small transactional storage surface."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.database_path = self.data_dir / "state.sqlite3"
        self._closed = False
        self._schema_ready = False
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

    def __enter__(self) -> LocalStore:
        self._require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the directory; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        self._directory_lock.close()

    def write_memory(
        self,
        memory: StoredMemory,
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Create or update a memory and its supplied embeddings atomically.

        Returns true when this call created the memory. Updating memory content queues every
        existing vector again because indexed text is derived from the authoritative row.
        """
        return self.write_memories((memory,), embeddings)[0]

    def write_memories(
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, ...]:
        """Create or update a batch with one commit and one durability sync."""
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        if not supplied_memories:
            return ()
        with self._transaction() as connection:
            created_flags = tuple(
                self._write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                )
                for memory in supplied_memories
            )
            for embedding in supplied_embeddings:
                self._write_embedding(connection, embedding)
        return created_flags

    def read_memory(self, memory_id: str) -> StoredMemory | None:
        """Return one memory, or none when it does not exist."""
        _require_identifier(memory_id, "memory_id")
        with self._read_transaction() as connection:
            row = connection.execute(
                """
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, last_accessed_at, access_count, created_at, updated_at
                FROM memory_records
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            assets = (
                () if row is None else self._read_memory_assets(connection, (memory_id,))[memory_id]
            )
        return None if row is None else _memory_from_row(row, assets=assets)

    def read_memories(self, memory_ids: Sequence[str]) -> tuple[StoredMemory, ...]:
        """Hydrate existing memories with one query and preserve input ranking."""
        if not memory_ids:
            return ()
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        rows: list[sqlite3.Row] = []
        with self._read_transaction() as connection:
            for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT memory_id, content, modality, memory_type, metadata_json,
                               occurred_at, last_accessed_at, access_count, created_at, updated_at
                        FROM memory_records
                        WHERE memory_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
            assets_by_memory = self._read_memory_assets(connection, tuple(memory_ids))
        by_id = {
            _row_text(row, "memory_id"): _memory_from_row(
                row,
                assets=assets_by_memory.get(_row_text(row, "memory_id"), ()),
            )
            for row in rows
        }
        return tuple(by_id[memory_id] for memory_id in memory_ids if memory_id in by_id)

    def list_memories(
        self,
        *,
        limit: int = 100,
        after: tuple[datetime, str] | None = None,
    ) -> tuple[StoredMemory, ...]:
        """List newest first with a stable `(created_at, memory_id)` keyset cursor."""
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        parameters: tuple[object, ...]
        where = ""
        if after is None:
            parameters = (limit,)
        else:
            created_at, memory_id = after
            _require_aware(created_at, "after created_at")
            _require_identifier(memory_id, "after memory_id")
            created_text = _datetime_text(created_at)
            where = """
                WHERE created_at < ? OR (created_at = ? AND memory_id < ?)
            """
            parameters = (created_text, created_text, memory_id, limit)
        with self._read_transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, last_accessed_at, access_count, created_at, updated_at
                FROM memory_records
                {where}
                ORDER BY created_at DESC, memory_id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            memory_ids = tuple(_row_text(row, "memory_id") for row in rows)
            assets_by_memory = self._read_memory_assets(connection, memory_ids)
        return tuple(
            _memory_from_row(
                row,
                assets=assets_by_memory.get(_row_text(row, "memory_id"), ()),
            )
            for row in rows
        )

    def reinforce_memories(self, memory_ids: Sequence[str], *, accessed_at: datetime) -> int:
        """Record one bounded retrieval reinforcement for each existing memory."""
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0
        for memory_id in unique_ids:
            _require_identifier(memory_id, "memory_id")
        _require_aware(accessed_at, "accessed_at")
        accessed_text = _datetime_text(accessed_at)
        changed = 0
        with self._transaction() as connection:
            for offset in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                cursor = connection.execute(
                    f"""
                    UPDATE memory_records
                    SET access_count = MIN(access_count + 1, 20),
                        last_accessed_at = CASE
                            WHEN last_accessed_at IS NULL OR last_accessed_at < ? THEN ?
                            ELSE last_accessed_at
                        END
                    WHERE memory_id IN ({placeholders})
                    """,
                    (accessed_text, accessed_text, *batch),
                )
                changed += cursor.rowcount
        return changed

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory; cascading embedding triggers enqueue index deletions."""
        deleted, _assets = self.delete_memory_with_assets(memory_id)
        return deleted

    def delete_memory_with_assets(self, memory_id: str) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory and return its assets that no remaining memory references."""
        _require_identifier(memory_id, "memory_id")
        with self._transaction() as connection:
            linked_ids = tuple(
                dict.fromkeys(
                    _row_text(row, "asset_id")
                    for row in connection.execute(
                        "SELECT asset_id FROM memory_assets WHERE memory_id = ? ORDER BY position",
                        (memory_id,),
                    ).fetchall()
                )
            )
            cursor = connection.execute(
                "DELETE FROM memory_records WHERE memory_id = ?",
                (memory_id,),
            )
            unreferenced = self._read_unreferenced_assets(connection, linked_ids)
        return cursor.rowcount > 0, unreferenced

    def read_asset(self, asset_id: str) -> StoredAsset | None:
        """Return one persisted asset descriptor, including a cached transcript."""
        _sha256(asset_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT asset_id, modality, mime_type, size_bytes, sha256,
                       relative_path, name, transcript, created_at
                FROM media_assets
                WHERE asset_id = ?
                """,
                (asset_id,),
            ).fetchone()
        return None if row is None else _asset_from_row(row)

    def read_assets(self, asset_ids: Sequence[str]) -> tuple[StoredAsset, ...]:
        """Resolve existing assets in caller order, preserving repeated IDs."""
        if not asset_ids:
            return ()
        for asset_id in asset_ids:
            _sha256(asset_id)
        rows: list[sqlite3.Row] = []
        unique_ids = tuple(dict.fromkeys(asset_ids))
        with self._connection() as connection:
            for offset in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT asset_id, modality, mime_type, size_bytes, sha256,
                               relative_path, name, transcript, created_at
                        FROM media_assets
                        WHERE asset_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        by_id = {_row_text(row, "asset_id"): _asset_from_row(row) for row in rows}
        return tuple(by_id[asset_id] for asset_id in asset_ids if asset_id in by_id)

    def write_asset(self, asset: StoredAsset) -> None:
        """Persist one media descriptor without attaching it to a memory."""
        if not isinstance(asset, StoredAsset):
            raise ValueError("asset must be a StoredAsset value")
        with self._transaction() as connection:
            self._write_asset(connection, asset)

    def read_unreferenced_assets(
        self,
        asset_ids: Sequence[str],
    ) -> tuple[StoredAsset, ...]:
        """Return supplied asset rows that no memory currently references."""
        for asset_id in asset_ids:
            _sha256(asset_id)
        with self._connection() as connection:
            return self._read_unreferenced_assets(connection, tuple(dict.fromkeys(asset_ids)))

    def set_asset_transcript(self, asset_id: str, text: str) -> bool:
        """Fill or replace an asset's cached transcript.

        Empty text is meaningful: it records that transcription completed without speech.
        """
        return self.set_asset_transcripts(((asset_id, text),)) == 1

    def set_asset_transcripts(self, values: Sequence[tuple[str, str]]) -> int:
        """Cache a batch of transcripts in one durable SQLite transaction."""
        supplied = tuple(values)
        if len({asset_id for asset_id, _text in supplied}) != len(supplied):
            raise ValueError("asset transcript IDs must be unique")
        for asset_id, text in supplied:
            _sha256(asset_id)
            if not isinstance(text, str):
                raise ValueError("asset transcript must be text")
        if not supplied:
            return 0
        with self._transaction() as connection:
            cursor = connection.executemany(
                "UPDATE media_assets SET transcript = ? WHERE asset_id = ?",
                ((text, asset_id) for asset_id, text in supplied),
            )
        return cursor.rowcount

    def read_speech(
        self,
        asset_id: str,
        *,
        space_id: str,
    ) -> tuple[SpeakerSegment, ...] | None:
        """Return cached speaker turns, including an empty completed analysis."""
        _sha256(asset_id)
        _require_identifier(space_id, "speech space_id")
        with self._connection() as connection:
            analysis = connection.execute(
                "SELECT 1 FROM speech_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if analysis is None:
                return None
            return self._read_speech(connection, asset_id)

    def write_speech(
        self,
        asset_id: str,
        analysis: SpeechAnalysis,
        *,
        model_id: str,
        space_id: str,
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.05,
    ) -> tuple[SpeakerSegment, ...]:
        """Persist one analysis and match its CAM++ centroids to local identities."""
        _sha256(asset_id)
        _require_identifier(model_id, "speech model_id")
        _require_identifier(space_id, "speech space_id")
        if not isinstance(analysis, SpeechAnalysis):
            raise ValueError("analysis must be a SpeechAnalysis value")
        for value, name in (
            (minimum_similarity, "minimum_similarity"),
            (minimum_margin, "minimum_margin"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one")
        speakers = {speaker.speaker_label: speaker.values for speaker in analysis.speakers}
        if len(speakers) != len(analysis.speakers):
            raise ValueError("speaker labels must be unique within one analysis")
        labels = {turn.speaker_label for turn in analysis.turns if turn.speaker_label is not None}
        if labels - speakers.keys():
            raise ValueError("every speaker turn label must have a centroid")
        dimensions = {len(values) for values in speakers.values()}
        if 0 in dimensions or len(dimensions) > 1:
            raise ValueError("speaker centroids must share one non-zero dimension")
        normalized = {
            label: _normalized_vector(values, "speaker centroid")
            for label, values in speakers.items()
        }
        now = datetime.now(timezone.utc)
        transcript = "\n".join(turn.text for turn in analysis.turns)
        with self._transaction() as connection:
            cached = connection.execute(
                "SELECT 1 FROM speech_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if cached is not None:
                return self._read_speech(connection, asset_id)
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("speech analysis requires a stored media asset")

            identities = self._match_speakers(
                connection,
                normalized,
                model_id=model_id,
                space_id=space_id,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
                now=now,
            )
            connection.execute("DELETE FROM speech_analyses WHERE asset_id = ?", (asset_id,))
            connection.execute(
                """
                INSERT INTO speech_analyses (asset_id, model_id, space_id, transcript, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, model_id, space_id, transcript, _datetime_text(now)),
            )
            connection.executemany(
                """
                INSERT INTO speech_segments (
                    asset_id, position, start_ms, end_ms, transcript,
                    speaker_id, identity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        asset_id,
                        position,
                        turn.start_ms,
                        turn.end_ms,
                        turn.text,
                        None if turn.speaker_label is None else identities[turn.speaker_label][0],
                        None if turn.speaker_label is None else identities[turn.speaker_label][1],
                    )
                    for position, turn in enumerate(analysis.turns)
                ),
            )
            connection.execute(
                "UPDATE media_assets SET transcript = ? WHERE asset_id = ?",
                (transcript, asset_id),
            )
            return self._read_speech(connection, asset_id)

    def register_speaker(self, speaker_id: str, name: str) -> bool:
        """Assign or replace the display name for one local speaker identity."""
        _require_identifier(speaker_id, "speaker_id")
        normalized_name = _speaker_name(name)
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE speaker_identities
                SET name = ?, updated_at = ?
                WHERE speaker_id = ?
                """,
                (normalized_name, now, speaker_id),
            )
        return cursor.rowcount > 0

    def list_unreferenced_assets(self, *, limit: int = 100) -> tuple[StoredAsset, ...]:
        """List asset rows eligible for physical garbage collection."""
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                       a.relative_path, a.name, a.transcript, a.created_at
                FROM media_assets AS a
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_assets AS ma WHERE ma.asset_id = a.asset_id
                )
                ORDER BY a.created_at, a.asset_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_asset_from_row(row) for row in rows)

    def delete_asset_if_unreferenced(self, asset_id: str) -> bool:
        """Delete one unreferenced descriptor after its CAS file has been removed."""
        _sha256(asset_id)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM media_assets
                WHERE asset_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_assets WHERE memory_assets.asset_id = media_assets.asset_id
                  )
                """,
                (asset_id,),
            )
        return cursor.rowcount > 0

    def write_embedding(self, embedding: StoredEmbedding) -> bool:
        """Create or update one authoritative vector and enqueue its index mutation."""
        with self._transaction() as connection:
            created = (
                connection.execute(
                    "SELECT 1 FROM embeddings WHERE embedding_id = ?",
                    (embedding.embedding_id,),
                ).fetchone()
                is None
            )
            self._write_embedding(connection, embedding)
        return created

    def read_embedding(self, embedding_id: str) -> StoredEmbedding | None:
        """Return one authoritative FP32 embedding."""
        _require_identifier(embedding_id, "embedding_id")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT embedding_id, memory_id, object_part, model_id, space_id, task,
                       dimension, normalized, vector, created_at
                FROM embeddings
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            ).fetchone()
        return None if row is None else _embedding_from_row(row)

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete one vector and durably enqueue removal from the search index."""
        _require_identifier(embedding_id, "embedding_id")
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM embeddings WHERE embedding_id = ?",
                (embedding_id,),
            )
        return cursor.rowcount > 0

    def read_index_document(self, embedding_id: str) -> IndexDocument | None:
        """Hydrate the current index payload from authoritative SQLite state."""
        _require_identifier(embedding_id, "embedding_id")
        documents = self.read_index_documents((embedding_id,))
        return documents[0] if documents else None

    def read_index_documents(
        self,
        embedding_ids: Sequence[str],
    ) -> tuple[IndexDocument, ...]:
        """Hydrate existing index payloads on one connection, preserving input order."""
        if not embedding_ids:
            return ()
        for embedding_id in embedding_ids:
            _require_identifier(embedding_id, "embedding_id")
        rows = []
        with self._connection() as connection:
            for offset in range(0, len(embedding_ids), _SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _embedding_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT e.embedding_id, e.memory_id, e.object_part, e.model_id, e.space_id,
                               e.task, e.dimension, e.normalized, e.vector, e.created_at,
                               m.content, m.metadata_json, m.memory_type, m.occurred_at
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.embedding_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        by_id: dict[str, IndexDocument] = {}
        for row in rows:
            document = IndexDocument(
                embedding=_embedding_from_row(row),
                content=_row_text(row, "content"),
                metadata_json=_row_text(row, "metadata_json"),
                memory_type=_row_text(row, "memory_type"),
                occurred_at=_optional_datetime_from_row(row, "occurred_at"),
            )
            by_id[document.embedding.embedding_id] = document
        return tuple(by_id[embedding_id] for embedding_id in embedding_ids if embedding_id in by_id)

    def pending_index_operations(self, *, limit: int = 100) -> tuple[IndexOperation, ...]:
        """Read queued mutations without acknowledging them."""
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT operation_id, embedding_id, action
                FROM search_index_queue
                ORDER BY operation_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            IndexOperation(
                operation_id=int(row["operation_id"]),
                embedding_id=_row_text(row, "embedding_id"),
                action=_index_action(row),
            )
            for row in rows
        )

    def acknowledge_index_operations(self, operations: Sequence[IndexOperation]) -> int:
        """Remove exactly the operations made durable by the search index."""
        if not operations:
            return 0
        operation_ids = [operation.operation_id for operation in operations]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("operation IDs must be unique")
        with self._transaction() as connection:
            cursor = connection.executemany(
                """
                DELETE FROM search_index_queue
                WHERE operation_id = ? AND embedding_id = ? AND action = ?
                """,
                (
                    (operation.operation_id, operation.embedding_id, operation.action)
                    for operation in operations
                ),
            )
        return cursor.rowcount

    def queue_all_embeddings(self) -> int:
        """Append one upsert per vector for a full search-index rebuild."""
        with self._transaction() as connection:
            now = _datetime_text(datetime.now(timezone.utc))
            cursor = connection.execute(
                """
                INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
                SELECT embeddings.embedding_id, 'upsert', ?
                FROM embeddings
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM search_index_queue
                    WHERE search_index_queue.embedding_id = embeddings.embedding_id
                      AND search_index_queue.action = 'upsert'
                )
                ORDER BY embeddings.embedding_id
                """,
                (now,),
            )
        return cursor.rowcount

    def set_metadata(self, key: str, value: str) -> None:
        """Set one store-level compatibility value."""
        _require_identifier(key, "metadata key")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO store_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _datetime_text(datetime.now(timezone.utc))),
            )

    def get_metadata(self, key: str) -> str | None:
        """Read one store-level compatibility value."""
        _require_identifier(key, "metadata key")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM store_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return None if row is None else _row_text(row, "value")

    def delete_metadata(self, key: str) -> bool:
        """Delete one store-level compatibility value."""
        _require_identifier(key, "metadata key")
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM store_metadata WHERE key = ?", (key,))
        return cursor.rowcount > 0

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = _table_names(connection)
            if version == 0:
                _create_schema(connection, tables)
            elif version == 1:
                _migrate_v1(connection, tables)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 2:
                _migrate_v2(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 3:
                _migrate_v3(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 4:
                _migrate_v4(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = _table_names(connection)
            if version != _SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"unsupported local schema version {version}; expected {_SCHEMA_VERSION}"
                )
            missing_tables = _REQUIRED_TABLES - tables
            if missing_tables:
                names = ", ".join(sorted(missing_tables))
                raise UnsupportedSchemaError(f"local schema is missing required tables: {names}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._require_open()
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if not self._schema_ready:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold one WAL snapshot across multi-query hydration."""
        with self._connection() as connection:
            connection.execute("BEGIN")
            try:
                yield connection
            finally:
                connection.rollback()

    def _write_embedding(
        self,
        connection: sqlite3.Connection,
        embedding: StoredEmbedding,
    ) -> None:
        connection.execute(
            """
            INSERT INTO embeddings (
                embedding_id, memory_id, object_part, model_id, space_id, task,
                dimension, normalized, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (embedding_id) DO UPDATE SET
                memory_id = excluded.memory_id,
                object_part = excluded.object_part,
                model_id = excluded.model_id,
                space_id = excluded.space_id,
                task = excluded.task,
                dimension = excluded.dimension,
                normalized = excluded.normalized,
                vector = excluded.vector,
                created_at = excluded.created_at
            """,
            (
                embedding.embedding_id,
                embedding.memory_id,
                embedding.object_part,
                embedding.model_id,
                embedding.space_id,
                embedding.task,
                len(embedding.values),
                int(embedding.normalized),
                _pack_vector(embedding.values),
                _datetime_text(embedding.created_at),
            ),
        )

    def _write_memory(
        self,
        connection: sqlite3.Connection,
        memory: StoredMemory,
        *,
        supplied_embedding_ids: set[str],
    ) -> bool:
        existing = connection.execute(
            """
            SELECT content, modality, memory_type, metadata_json, occurred_at
            FROM memory_records
            WHERE memory_id = ?
            """,
            (memory.memory_id,),
        ).fetchone()
        existing_asset_ids = tuple(
            _row_text(row, "asset_id")
            for row in connection.execute(
                """
                SELECT asset_id
                FROM memory_assets
                WHERE memory_id = ?
                ORDER BY position
                """,
                (memory.memory_id,),
            ).fetchall()
        )
        supplied_asset_ids = tuple(asset.asset_id for asset in memory.assets)
        index_content_changed = existing is not None and (
            _row_text(existing, "content") != memory.content
            or _row_text(existing, "modality") != memory.modality
            or _row_text(existing, "memory_type") != memory.memory_type
            or _row_text(existing, "metadata_json") != memory.metadata_json
            or existing["occurred_at"] != _optional_datetime_text(memory.occurred_at)
            or existing_asset_ids != supplied_asset_ids
        )
        connection.execute(
            """
            INSERT INTO memory_records (
                memory_id, content, modality, memory_type, metadata_json,
                occurred_at, last_accessed_at, access_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (memory_id) DO UPDATE SET
                content = excluded.content,
                modality = excluded.modality,
                memory_type = excluded.memory_type,
                metadata_json = excluded.metadata_json,
                occurred_at = excluded.occurred_at,
                updated_at = excluded.updated_at
            """,
            (
                memory.memory_id,
                memory.content,
                memory.modality,
                memory.memory_type,
                memory.metadata_json,
                _optional_datetime_text(memory.occurred_at),
                _optional_datetime_text(memory.last_accessed_at),
                memory.access_count,
                _datetime_text(memory.created_at),
                _datetime_text(memory.updated_at),
            ),
        )
        for asset in memory.assets:
            self._write_asset(connection, asset)
        if existing_asset_ids != supplied_asset_ids:
            connection.execute(
                "DELETE FROM memory_assets WHERE memory_id = ?",
                (memory.memory_id,),
            )
            connection.executemany(
                """
                INSERT INTO memory_assets (memory_id, position, asset_id)
                VALUES (?, ?, ?)
                """,
                (
                    (memory.memory_id, position, asset_id)
                    for position, asset_id in enumerate(supplied_asset_ids)
                ),
            )
        if index_content_changed:
            self._queue_memory_embeddings(
                connection,
                memory.memory_id,
                exclude=supplied_embedding_ids,
            )
        return existing is None

    @staticmethod
    def _write_asset(connection: sqlite3.Connection, asset: StoredAsset) -> None:
        connection.execute(
            """
            INSERT INTO media_assets (
                asset_id, modality, mime_type, size_bytes, sha256,
                relative_path, name, transcript, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (asset_id) DO NOTHING
            """,
            (
                asset.asset_id,
                asset.modality,
                asset.mime_type,
                asset.size_bytes,
                asset.sha256,
                asset.relative_path,
                asset.name,
                asset.transcript,
                _datetime_text(asset.created_at),
            ),
        )
        row = connection.execute(
            """
            SELECT asset_id, modality, mime_type, size_bytes, sha256,
                   relative_path, name, transcript, created_at
            FROM media_assets
            WHERE asset_id = ?
            """,
            (asset.asset_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("asset upsert did not produce a row")
        stored = _asset_from_row(row)
        immutable = (
            "modality",
            "mime_type",
            "size_bytes",
            "sha256",
            "relative_path",
        )
        if any(getattr(stored, field) != getattr(asset, field) for field in immutable):
            raise ValueError(f"asset {asset.asset_id!r} conflicts with stored metadata")
        name = stored.name if stored.name is not None else asset.name
        transcript = asset.transcript if asset.transcript is not None else stored.transcript
        if name != stored.name or transcript != stored.transcript:
            connection.execute(
                "UPDATE media_assets SET name = ?, transcript = ? WHERE asset_id = ?",
                (name, transcript, asset.asset_id),
            )

    @staticmethod
    def _read_memory_assets(
        connection: sqlite3.Connection,
        memory_ids: Sequence[str],
    ) -> dict[str, tuple[StoredAsset, ...]]:
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return {}
        collected: dict[str, list[StoredAsset]] = {memory_id: [] for memory_id in unique_ids}
        for offset in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH):
            batch = unique_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
            placeholders = ", ".join("?" for _memory_id in batch)
            rows = connection.execute(
                f"""
                SELECT ma.memory_id, ma.position,
                       a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                       a.relative_path, a.name, a.transcript, a.created_at
                FROM memory_assets AS ma
                JOIN media_assets AS a ON a.asset_id = ma.asset_id
                WHERE ma.memory_id IN ({placeholders})
                ORDER BY ma.memory_id, ma.position
                """,
                tuple(batch),
            ).fetchall()
            for row in rows:
                collected[_row_text(row, "memory_id")].append(_asset_from_row(row))
        return {memory_id: tuple(assets) for memory_id, assets in collected.items()}

    @staticmethod
    def _read_unreferenced_assets(
        connection: sqlite3.Connection,
        asset_ids: Sequence[str],
    ) -> tuple[StoredAsset, ...]:
        if not asset_ids:
            return ()
        found: dict[str, StoredAsset] = {}
        for offset in range(0, len(asset_ids), _SQLITE_PARAMETER_BATCH):
            batch = asset_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
            placeholders = ", ".join("?" for _asset_id in batch)
            rows = connection.execute(
                f"""
                SELECT a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                       a.relative_path, a.name, a.transcript, a.created_at
                FROM media_assets AS a
                WHERE a.asset_id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_assets AS ma WHERE ma.asset_id = a.asset_id
                  )
                """,
                tuple(batch),
            ).fetchall()
            found.update((_row_text(row, "asset_id"), _asset_from_row(row)) for row in rows)
        return tuple(found[asset_id] for asset_id in asset_ids if asset_id in found)

    @staticmethod
    def _queue_memory_embeddings(
        connection: sqlite3.Connection,
        memory_id: str,
        *,
        exclude: set[str],
    ) -> None:
        embedding_ids = connection.execute(
            "SELECT embedding_id FROM embeddings WHERE memory_id = ? ORDER BY embedding_id",
            (memory_id,),
        ).fetchall()
        now = _datetime_text(datetime.now(timezone.utc))
        connection.executemany(
            """
            INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
            VALUES (?, 'upsert', ?)
            """,
            (
                (_row_text(row, "embedding_id"), now)
                for row in embedding_ids
                if _row_text(row, "embedding_id") not in exclude
            ),
        )

    @staticmethod
    def _read_speech(
        connection: sqlite3.Connection,
        asset_id: str,
    ) -> tuple[SpeakerSegment, ...]:
        rows = connection.execute(
            """
            SELECT s.start_ms, s.end_ms, s.transcript, s.speaker_id,
                   i.name AS speaker_name, s.identity_score
            FROM speech_segments AS s
            LEFT JOIN speaker_identities AS i ON i.speaker_id = s.speaker_id
            WHERE s.asset_id = ?
            ORDER BY s.position
            """,
            (asset_id,),
        ).fetchall()
        return tuple(
            SpeakerSegment(
                asset_id=asset_id,
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                text=_row_text(row, "transcript"),
                speaker_id=(None if row["speaker_id"] is None else _row_text(row, "speaker_id")),
                speaker_name=(
                    None if row["speaker_name"] is None else _row_text(row, "speaker_name")
                ),
                identity_score=(
                    None if row["identity_score"] is None else float(row["identity_score"])
                ),
            )
            for row in rows
        )

    @staticmethod
    def _match_speakers(
        connection: sqlite3.Connection,
        speakers: dict[str, tuple[float, ...]],
        *,
        model_id: str,
        space_id: str,
        minimum_similarity: float,
        minimum_margin: float,
        now: datetime,
    ) -> dict[str, tuple[str, float | None]]:
        if not speakers:
            return {}
        dimension = len(next(iter(speakers.values())))
        rows = connection.execute(
            """
            SELECT speaker_id, centroid, observations, created_at
            FROM speaker_identities
            WHERE space_id = ? AND dimension = ?
            ORDER BY speaker_id
            """,
            (space_id, dimension),
        ).fetchall()
        existing = {
            _row_text(row, "speaker_id"): (
                _normalized_vector(
                    _unpack_vector(_row_blob(row, "centroid"), dimension),
                    "stored speaker centroid",
                ),
                int(row["observations"]),
            )
            for row in rows
        }
        claimed: set[str] = set()
        matches: dict[str, tuple[str, float | None]] = {}
        for label, centroid in sorted(speakers.items()):
            # ponytail: local speaker populations use a linear scan; add a vector index only
            # after profiling shows identity matching matters beside model inference.
            ranked = sorted(
                (
                    (speaker_id, math.fsum(a * b for a, b in zip(centroid, stored, strict=True)))
                    for speaker_id, (stored, _observations) in existing.items()
                    if speaker_id not in claimed
                ),
                key=lambda item: (-item[1], item[0]),
            )
            best = ranked[0] if ranked else None
            ambiguous = len(ranked) > 1 and ranked[0][1] - ranked[1][1] < minimum_margin
            if best is None or best[1] < minimum_similarity or ambiguous:
                speaker_id = f"speaker_{uuid.uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO speaker_identities (
                        speaker_id, model_id, space_id, dimension, centroid,
                        observations, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        speaker_id,
                        model_id,
                        space_id,
                        dimension,
                        _pack_vector(centroid),
                        _datetime_text(now),
                        _datetime_text(now),
                    ),
                )
                existing[speaker_id] = (centroid, 1)
                matches[label] = (speaker_id, None)
            else:
                speaker_id, score = best
                previous, observations = existing[speaker_id]
                updated = _normalized_vector(
                    tuple(
                        (old * observations + new) / (observations + 1)
                        for old, new in zip(previous, centroid, strict=True)
                    ),
                    "updated speaker centroid",
                )
                connection.execute(
                    """
                    UPDATE speaker_identities
                    SET centroid = ?, observations = ?, updated_at = ?
                    WHERE speaker_id = ?
                    """,
                    (
                        _pack_vector(updated),
                        observations + 1,
                        _datetime_text(now),
                        speaker_id,
                    ),
                )
                existing[speaker_id] = (updated, observations + 1)
                matches[label] = (speaker_id, max(0.0, min(1.0, score)))
            claimed.add(matches[label][0])
        return matches

    def _require_open(self) -> None:
        if self._closed:
            raise LocalStoreClosedError("local store is closed")


def _table_names(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    return frozenset(_row_text(row, "name") for row in rows)


def _prepare_write_batch(
    memories: Iterable[StoredMemory],
    embeddings: Iterable[StoredEmbedding],
) -> tuple[
    tuple[StoredMemory, ...],
    tuple[StoredEmbedding, ...],
    dict[str, set[str]],
]:
    supplied_memories = tuple(memories)
    supplied_embeddings = tuple(embeddings)
    if not supplied_memories and supplied_embeddings:
        raise ValueError("embeddings require at least one written memory")
    memory_ids = [memory.memory_id for memory in supplied_memories]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory IDs must be unique")
    memory_id_set = set(memory_ids)
    if any(embedding.memory_id not in memory_id_set for embedding in supplied_embeddings):
        raise ValueError("all embeddings must belong to a written memory")
    embedding_ids = [embedding.embedding_id for embedding in supplied_embeddings]
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("embedding IDs must be unique")
    supplied_by_memory: dict[str, set[str]] = {memory_id: set() for memory_id in memory_ids}
    for embedding in supplied_embeddings:
        supplied_by_memory[embedding.memory_id].add(embedding.embedding_id)
    return supplied_memories, supplied_embeddings, supplied_by_memory


def _create_schema(
    connection: sqlite3.Connection,
    existing_tables: frozenset[str],
) -> None:
    if existing_tables:
        raise UnsupportedSchemaError("local data directory contains an unversioned SQLite schema")
    try:
        connection.executescript(_SCHEMA_V5)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v1(
    connection: sqlite3.Connection,
    existing_tables: frozenset[str],
) -> None:
    missing_tables = _REQUIRED_V1_TABLES - existing_tables
    if missing_tables:
        names = ", ".join(sorted(missing_tables))
        raise UnsupportedSchemaError(f"local schema v1 is missing required tables: {names}")
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.executescript(_MIGRATE_V1_TO_V2)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise UnsupportedSchemaError("local schema migration produced invalid foreign keys")


def _migrate_v2(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(_MIGRATE_V2_TO_V4)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v3(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(_MIGRATE_V3_TO_V4)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v4(connection: sqlite3.Connection) -> None:
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(memory_records)")}
        connection.execute("BEGIN IMMEDIATE")
        if "memory_type" not in columns:
            connection.execute(
                """
                ALTER TABLE memory_records
                ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'semantic'
                    CHECK (memory_type IN ('semantic', 'episodic', 'procedural'))
                """
            )
        if "last_accessed_at" not in columns:
            connection.execute("ALTER TABLE memory_records ADD COLUMN last_accessed_at TEXT")
        if "access_count" not in columns:
            connection.execute(
                """
                ALTER TABLE memory_records
                ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0
                    CHECK (access_count BETWEEN 0 AND 20)
                """
            )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _memory_from_row(
    row: sqlite3.Row,
    *,
    assets: tuple[StoredAsset, ...] = (),
) -> StoredMemory:
    return StoredMemory(
        memory_id=_row_text(row, "memory_id"),
        content=_row_text(row, "content"),
        modality=_row_text(row, "modality"),
        memory_type=_row_text(row, "memory_type"),
        assets=assets,
        metadata_json=_row_text(row, "metadata_json"),
        occurred_at=_optional_datetime_from_row(row, "occurred_at"),
        last_accessed_at=_optional_datetime_from_row(row, "last_accessed_at"),
        access_count=int(row["access_count"]),
        created_at=_parse_datetime(_row_text(row, "created_at")),
        updated_at=_parse_datetime(_row_text(row, "updated_at")),
    )


def _access_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 20:
        raise ValueError("access_count must be between zero and twenty")
    return value


def _asset_from_row(row: sqlite3.Row) -> StoredAsset:
    return StoredAsset(
        asset_id=_row_text(row, "asset_id"),
        modality=_row_text(row, "modality"),
        mime_type=_row_text(row, "mime_type"),
        size_bytes=int(row["size_bytes"]),
        sha256=_row_text(row, "sha256"),
        relative_path=_row_text(row, "relative_path"),
        name=None if row["name"] is None else _row_text(row, "name"),
        transcript=(None if row["transcript"] is None else _row_text(row, "transcript")),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _embedding_from_row(row: sqlite3.Row) -> StoredEmbedding:
    dimension = int(row["dimension"])
    vector = row["vector"]
    if not isinstance(vector, bytes):
        raise RuntimeError("stored embedding vector is not a BLOB")
    return StoredEmbedding(
        embedding_id=_row_text(row, "embedding_id"),
        memory_id=_row_text(row, "memory_id"),
        object_part=int(row["object_part"]),
        model_id=_row_text(row, "model_id"),
        space_id=_row_text(row, "space_id"),
        task=_row_text(row, "task"),
        values=_unpack_vector(vector, dimension),
        normalized=bool(int(row["normalized"])),
        created_at=_parse_datetime(_row_text(row, "created_at")),
    )


def _index_action(row: sqlite3.Row) -> Literal["upsert", "delete"]:
    action = _row_text(row, "action")
    if action == "upsert":
        return "upsert"
    if action == "delete":
        return "delete"
    raise RuntimeError(f"invalid queued index action: {action}")


def _row_text(row: sqlite3.Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise RuntimeError(f"stored {column} is not text")
    return value


def _row_blob(row: sqlite3.Row, column: str) -> bytes:
    value = row[column]
    if not isinstance(value, bytes):
        raise RuntimeError(f"stored {column} is not a BLOB")
    return value


def _pack_vector(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(value: bytes, dimension: int) -> tuple[float, ...]:
    if dimension <= 0 or len(value) != dimension * 4:
        raise RuntimeError("stored embedding dimension does not match its FP32 BLOB")
    return struct.unpack(f"<{dimension}f", value)


def _normalized_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain finite values")
    magnitude = math.sqrt(math.fsum(value * value for value in normalized))
    if magnitude == 0.0:
        raise ValueError(f"{name} must not be a zero vector")
    return tuple(value / magnitude for value in normalized)


def _datetime_text(value: datetime) -> str:
    _require_aware(value, "datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _optional_datetime_text(value: datetime | None) -> str | None:
    return None if value is None else _datetime_text(value)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require_aware(parsed, "stored datetime")
    return parsed


def _optional_datetime_from_row(row: sqlite3.Row, column: str) -> datetime | None:
    value = row[column]
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"stored {column} is not text")
    return _parse_datetime(value)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_identifier(value: str, name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


def _speaker_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("speaker name must be non-empty text")
    normalized = value.strip()
    if len(normalized) > 255 or not normalized.isprintable():
        raise ValueError("speaker name must be at most 255 printable characters")
    return normalized


def _sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def _modality(value: str, *, asset: bool) -> str:
    choices = _ASSET_MODALITIES if asset else _MEMORY_MODALITIES
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"modality must be one of: {allowed}")
    return value


def _mime_type(value: str, modality: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("mime_type must be non-empty and trimmed")
    canonical = value.casefold()
    if _MEDIA_TYPE.fullmatch(canonical) is None or canonical.split("/", 1)[0] != modality:
        raise ValueError(f"mime_type must be a canonical {modality} media type")
    return canonical


def _asset_relative_path(digest: str) -> str:
    return f"assets/{digest[:2]}/{digest}"


def validate_asset_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 255
    ):
        raise ValueError("asset name must be a safe filename of at most 255 bytes")
    return value


def _canonical_object_json(value: str) -> str:
    try:
        decoded: object = json.loads(value, parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata_json must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json must encode an object")
    return json.dumps(
        decoded, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not supported: {value}")
