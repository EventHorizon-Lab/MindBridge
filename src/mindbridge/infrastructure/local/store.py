"""SQLite source of truth for the local MindBridge runtime."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Literal, NoReturn

from mindbridge.infrastructure.local._lock import DataDirectoryLock
from mindbridge.models.base import FaceAnalysis, SpeechAnalysis
from mindbridge.types import (
    EvidenceBasis,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryContext,
    MemoryKind,
    Modality,
    PendingCapture,
    SpatialAnchor,
    SpatialContext,
    SpeakerSegment,
)

_SCHEMA_VERSION = 11
_SQLITE_PARAMETER_BATCH = 900
# Kinds whose claims can contradict inside one lineage. This must stay the set the context
# compiler reports conflicts for; `tests/unit/test_memory_control_plane.py` asserts they agree,
# because the compiler is a product module and this is infrastructure.
_CONFLICT_KINDS: tuple[str, ...] = ("state", "relation", "trait")
_CONFLICT_KIND_PLACEHOLDERS = ", ".join("?" for _kind in _CONFLICT_KINDS)
_FACE_EXEMPLAR_LIMIT = 10
_VOICE_EXEMPLAR_LIMIT = 20
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
        "face_analyses",
        "face_observations",
        "identities",
        "identity_aliases",
        "identity_exemplars",
        "identity_link_evidence",
        "speech_analyses",
        "speech_segments",
        "store_metadata",
        "formation_runs",
        "memory_evidence",
        "memory_semantics",
        "memory_versions",
        "capture_queue",
        "memory_operations",
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

_LEGACY_SPEECH_SCHEMA = """
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

_IDENTITY_LINK_EVIDENCE_DDL = (
    """CREATE TABLE identity_link_evidence (
    voice_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    face_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (voice_id, face_id, asset_id)
)""",
    "CREATE INDEX identity_link_evidence_face_idx ON identity_link_evidence (face_id)",
    "CREATE INDEX identity_link_evidence_asset_idx ON identity_link_evidence (asset_id)",
)

_IDENTITY_LINK_EVIDENCE_SCHEMA = "".join(
    f"{statement};\n" for statement in _IDENTITY_LINK_EVIDENCE_DDL
)

_IDENTITY_SCHEMA = f"""
CREATE TABLE identities (
    identity_id TEXT PRIMARY KEY CHECK (length(trim(identity_id)) > 0),
    name TEXT CHECK (name IS NULL OR length(trim(name)) > 0),
    relationship TEXT CHECK (relationship IS NULL OR length(trim(relationship)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
);

CREATE TABLE identity_aliases (
    alias_id TEXT PRIMARY KEY CHECK (length(trim(alias_id)) > 0),
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    contributed_modality TEXT CHECK (
        contributed_modality IS NULL OR contributed_modality IN ('face', 'voice')
    ),
    CHECK (alias_id <> identity_id)
);

CREATE INDEX identity_aliases_identity_idx ON identity_aliases (identity_id);

{_IDENTITY_LINK_EVIDENCE_SCHEMA}

CREATE TABLE identity_exemplars (
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK (modality IN ('face', 'voice')),
    position INTEGER NOT NULL CHECK (position >= 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL CHECK (length(vector) = dimension * 4),
    created_at TEXT NOT NULL,
    PRIMARY KEY (identity_id, modality, position)
);

CREATE INDEX identity_exemplars_space_idx
    ON identity_exemplars (modality, space_id, dimension, identity_id, position);

CREATE TABLE speech_analyses (
    asset_id TEXT PRIMARY KEY REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    transcript TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE speech_segments (
    asset_id TEXT NOT NULL REFERENCES speech_analyses (asset_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    transcript TEXT NOT NULL CHECK (length(trim(transcript)) > 0),
    speaker_id TEXT REFERENCES identities (identity_id) ON DELETE SET NULL,
    identity_score REAL CHECK (
        identity_score IS NULL OR (identity_score >= 0.0 AND identity_score <= 1.0)
    ),
    PRIMARY KEY (asset_id, position)
);

CREATE INDEX speech_segments_speaker_idx ON speech_segments (speaker_id);

CREATE TABLE face_analyses (
    asset_id TEXT PRIMARY KEY REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE face_observations (
    asset_id TEXT NOT NULL REFERENCES face_analyses (asset_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    observed_at_ms INTEGER CHECK (observed_at_ms IS NULL OR observed_at_ms >= 0),
    box_x REAL NOT NULL CHECK (box_x >= 0.0 AND box_x <= 1.0),
    box_y REAL NOT NULL CHECK (box_y >= 0.0 AND box_y <= 1.0),
    box_width REAL NOT NULL CHECK (box_width > 0.0 AND box_x + box_width <= 1.0),
    box_height REAL NOT NULL CHECK (box_height > 0.0 AND box_y + box_height <= 1.0),
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE RESTRICT,
    identity_score REAL CHECK (
        identity_score IS NULL OR (identity_score >= 0.0 AND identity_score <= 1.0)
    ),
    PRIMARY KEY (asset_id, position)
);

CREATE INDEX face_observations_identity_idx ON face_observations (identity_id);
"""

_SEMANTIC_SCHEMA = """
CREATE TABLE memory_semantics (
    memory_id TEXT PRIMARY KEY REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    lineage_id TEXT NOT NULL CHECK (length(trim(lineage_id)) > 0),
    kind TEXT NOT NULL CHECK (
        kind IN (
            'observation', 'entity', 'event', 'state', 'relation',
            'affect', 'trait', 'response_policy'
        )
    ),
    basis TEXT NOT NULL CHECK (
        basis IN ('observation', 'user_statement', 'model_inference', 'response_feedback')
    ),
    source_id TEXT,
    subject TEXT,
    predicate TEXT,
    value TEXT,
    model_id TEXT,
    recipe TEXT,
    cue_modality TEXT CHECK (
        cue_modality IS NULL OR cue_modality IN ('text', 'image', 'video', 'audio')
    ),
    valence REAL CHECK (valence IS NULL OR (valence >= -1.0 AND valence <= 1.0)),
    arousal REAL CHECK (arousal IS NULL OR (arousal >= 0.0 AND arousal <= 1.0)),
    spatial_frame_id TEXT,
    spatial_anchor TEXT CHECK (
        spatial_anchor IS NULL OR spatial_anchor IN ('observer', 'subject')
    ),
    spatial_x REAL,
    spatial_y REAL,
    spatial_z REAL,
    spatial_qx REAL,
    spatial_qy REAL,
    spatial_qz REAL,
    spatial_qw REAL,
    spatial_uncertainty_m REAL CHECK (
        spatial_uncertainty_m IS NULL OR spatial_uncertainty_m >= 0.0
    ),
    CHECK (
        (spatial_frame_id IS NULL AND spatial_anchor IS NULL
         AND spatial_x IS NULL AND spatial_y IS NULL AND spatial_z IS NULL)
        OR
        (spatial_frame_id IS NOT NULL AND spatial_anchor IS NOT NULL
         AND spatial_x IS NOT NULL AND spatial_y IS NOT NULL AND spatial_z IS NOT NULL)
    )
);

CREATE INDEX memory_semantics_lineage_idx
    ON memory_semantics (lineage_id, memory_id);
CREATE INDEX memory_semantics_spatial_idx
    ON memory_semantics (spatial_frame_id, spatial_anchor, memory_id);

CREATE TABLE memory_versions (
    memory_id TEXT NOT NULL REFERENCES memory_semantics (memory_id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    valid_from TEXT,
    valid_until TEXT CHECK (
        valid_until IS NULL OR (valid_from IS NOT NULL AND valid_until > valid_from)
    ),
    recorded_at TEXT NOT NULL,
    retired_at TEXT CHECK (retired_at IS NULL OR retired_at > recorded_at),
    visible INTEGER NOT NULL DEFAULT 1 CHECK (visible IN (0, 1)),
    supersedes_id TEXT REFERENCES memory_records (memory_id) ON DELETE SET NULL,
    PRIMARY KEY (memory_id, version)
);

CREATE INDEX memory_versions_current_idx
    ON memory_versions (memory_id, retired_at, recorded_at, version);

CREATE TABLE memory_evidence (
    memory_id TEXT NOT NULL REFERENCES memory_semantics (memory_id) ON DELETE CASCADE,
    source_memory_id TEXT NOT NULL,
    source_group_id TEXT NOT NULL CHECK (length(trim(source_group_id)) > 0),
    position INTEGER NOT NULL CHECK (position >= 0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    recorded_at TEXT NOT NULL,
    retired_at TEXT CHECK (retired_at IS NULL OR retired_at > recorded_at),
    PRIMARY KEY (memory_id, position),
    CHECK (memory_id <> source_memory_id)
);

CREATE INDEX memory_evidence_source_idx
    ON memory_evidence (source_memory_id, retired_at, memory_id);
CREATE UNIQUE INDEX memory_evidence_current_idx
    ON memory_evidence (memory_id, source_memory_id)
    WHERE retired_at IS NULL;

CREATE TABLE formation_runs (
    source_memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    recipe TEXT NOT NULL CHECK (length(trim(recipe)) > 0),
    completed_at TEXT NOT NULL,
    PRIMARY KEY (source_memory_id, recipe)
);
"""

_CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS capture_queue (
    memory_id TEXT PRIMARY KEY REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    enqueued_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT CHECK (last_error IS NULL OR length(trim(last_error)) > 0)
);

CREATE TABLE IF NOT EXISTS memory_operations (
    operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_key TEXT NOT NULL CHECK (length(trim(operation_key)) > 0),
    intent TEXT NOT NULL CHECK (intent IN ('reinforce', 'consolidate', 'correct', 'forget')),
    trigger TEXT NOT NULL CHECK (length(trim(trigger)) > 0),
    model_id TEXT,
    recipe TEXT,
    operation_json TEXT NOT NULL,
    effects_json TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    rolled_back_at TEXT CHECK (rolled_back_at IS NULL OR rolled_back_at >= applied_at)
);

CREATE UNIQUE INDEX IF NOT EXISTS memory_operations_active_key_idx
    ON memory_operations (operation_key)
    WHERE rolled_back_at IS NULL;
"""

_PLACE_COLUMN_DDL = """
ALTER TABLE memory_records
ADD COLUMN place_id TEXT CHECK (
    place_id IS NULL OR (length(place_id) > 0 AND place_id = trim(place_id))
)
"""

# A symbolic room-level label, complementary to the metric pose on `memory_semantics`: the pose
# answers "within 2 m of here", this answers "in the kitchen", which is what a household query
# asks and the only spatial label a robot can supply when it cannot localise. It lives on
# `memory_records` rather than on the semantic row because the semantic row is conditional -- a
# memory added without a former and without an `ObservationContext` has none -- and a place scope
# that silently skipped those memories would be worse than no place scope. Partial, so a store
# that labels nothing carries no index pages.
_PLACE_INDEX_DDL = """
CREATE INDEX memory_records_place_idx
    ON memory_records (place_id, memory_id)
    WHERE place_id IS NOT NULL
"""

_SCHEMA_V10 = f"""
BEGIN IMMEDIATE;

CREATE TABLE memory_records (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'video', 'audio', 'omni')),
    memory_type TEXT NOT NULL DEFAULT 'semantic'
        CHECK (memory_type IN ('semantic', 'episodic', 'procedural')),
    metadata_json TEXT NOT NULL,
    occurred_at TEXT,
    occurred_end TEXT CHECK (
        occurred_end IS NULL OR (occurred_at IS NOT NULL AND occurred_end > occurred_at)
    ),
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count BETWEEN 0 AND 20),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at),
    place_id TEXT CHECK (
        place_id IS NULL OR (length(place_id) > 0 AND place_id = trim(place_id))
    ),
    forgotten_at TEXT
);

CREATE INDEX memory_records_created_idx
    ON memory_records (created_at DESC, memory_id DESC);
{_PLACE_INDEX_DDL};

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
{_IDENTITY_SCHEMA}
{_SEMANTIC_SCHEMA}
{_CONTROL_SCHEMA}

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

PRAGMA user_version = 11;
COMMIT;
"""

_MIGRATE_V7_TO_V8 = f"""
BEGIN IMMEDIATE;
{_SEMANTIC_SCHEMA}
PRAGMA user_version = 8;
COMMIT;
"""

_MIGRATE_V5_TO_V6 = """
BEGIN IMMEDIATE;
ALTER TABLE memory_records ADD COLUMN occurred_end TEXT CHECK (
    occurred_end IS NULL OR (occurred_at IS NOT NULL AND occurred_end > occurred_at)
);
PRAGMA user_version = 6;
COMMIT;
"""

_MIGRATE_V6_TO_V7 = """
BEGIN IMMEDIATE;

CREATE TABLE identities (
    identity_id TEXT PRIMARY KEY CHECK (length(trim(identity_id)) > 0),
    name TEXT CHECK (name IS NULL OR length(trim(name)) > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL CHECK (updated_at >= created_at)
);

INSERT INTO identities (identity_id, name, created_at, updated_at)
SELECT speaker_id, name, created_at, updated_at
FROM speaker_identities;

CREATE TABLE identity_aliases (
    alias_id TEXT PRIMARY KEY CHECK (length(trim(alias_id)) > 0),
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    CHECK (alias_id <> identity_id)
);

CREATE INDEX identity_aliases_identity_idx ON identity_aliases (identity_id);

CREATE TABLE identity_exemplars (
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK (modality IN ('face', 'voice')),
    position INTEGER NOT NULL CHECK (position >= 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    vector BLOB NOT NULL CHECK (length(vector) = dimension * 4),
    created_at TEXT NOT NULL,
    PRIMARY KEY (identity_id, modality, position)
);

INSERT INTO identity_exemplars (
    identity_id, modality, position, model_id, space_id, dimension, vector, created_at
)
SELECT speaker_id, 'voice', 0, model_id, space_id, dimension, centroid, created_at
FROM speaker_identities;

CREATE INDEX identity_exemplars_space_idx
    ON identity_exemplars (modality, space_id, dimension, identity_id, position);

CREATE TABLE speech_segments_v7 (
    asset_id TEXT NOT NULL REFERENCES speech_analyses (asset_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    transcript TEXT NOT NULL CHECK (length(trim(transcript)) > 0),
    speaker_id TEXT REFERENCES identities (identity_id) ON DELETE SET NULL,
    identity_score REAL CHECK (
        identity_score IS NULL OR (identity_score >= 0.0 AND identity_score <= 1.0)
    ),
    PRIMARY KEY (asset_id, position)
);

INSERT INTO speech_segments_v7 (
    asset_id, position, start_ms, end_ms, transcript, speaker_id, identity_score
)
SELECT asset_id, position, start_ms, end_ms, transcript, speaker_id, identity_score
FROM speech_segments;

DROP TABLE speech_segments;
DROP TABLE speaker_identities;
ALTER TABLE speech_segments_v7 RENAME TO speech_segments;
CREATE INDEX speech_segments_speaker_idx ON speech_segments (speaker_id);

CREATE TABLE face_analyses (
    asset_id TEXT PRIMARY KEY REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE face_observations (
    asset_id TEXT NOT NULL REFERENCES face_analyses (asset_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    observed_at_ms INTEGER CHECK (observed_at_ms IS NULL OR observed_at_ms >= 0),
    box_x REAL NOT NULL CHECK (box_x >= 0.0 AND box_x <= 1.0),
    box_y REAL NOT NULL CHECK (box_y >= 0.0 AND box_y <= 1.0),
    box_width REAL NOT NULL CHECK (box_width > 0.0 AND box_x + box_width <= 1.0),
    box_height REAL NOT NULL CHECK (box_height > 0.0 AND box_y + box_height <= 1.0),
    identity_id TEXT NOT NULL REFERENCES identities (identity_id) ON DELETE RESTRICT,
    identity_score REAL CHECK (
        identity_score IS NULL OR (identity_score >= 0.0 AND identity_score <= 1.0)
    ),
    PRIMARY KEY (asset_id, position)
);

CREATE INDEX face_observations_identity_idx ON face_observations (identity_id);

PRAGMA user_version = 7;
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
{_LEGACY_SPEECH_SCHEMA}
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


class StaleOperationError(RuntimeError):
    """Raised when a control-plane operation's preconditions moved before it committed.

    The control plane reads and validates records, then applies them in a later transaction. The
    apply transaction re-checks the preconditions the proposal was built on; when a target or
    cited source was forgotten, corrected, deleted, or already linked in between, nothing is
    written and the caller reports the proposal as stale rather than partially applying it.
    """


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
    occurred_end: datetime | None = None
    modality: str = "text"
    memory_type: str = "semantic"
    assets: tuple[StoredAsset, ...] = ()
    last_accessed_at: datetime | None = None
    access_count: int = 0
    forgotten_at: datetime | None = None
    context: MemoryContext | None = None
    place_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.memory_id, "memory_id")
        _require_optional_identifier(self.place_id, "place_id")
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
        _require_interval(self.occurred_at, self.occurred_end)
        if self.last_accessed_at is not None:
            _require_aware(self.last_accessed_at, "last_accessed_at")
        _access_count(self.access_count)
        if self.context is not None and not isinstance(self.context, MemoryContext):
            raise ValueError("context must be a MemoryContext or None")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    """An FP32 vector retained in SQLite so the search index is rebuildable.

    Vector *content* is checked once, where a vector enters SQLite (`_write_embedding`), and not
    here. Hydrating a search candidate rebuilds this value ~100 times per search while the search
    path reads only `embedding_id`, `memory_id` and `object_part` from it -- the vector itself is
    consumed solely by `ZvecIndex.upsert`, which validates what it consumes. Re-checking finiteness
    and unit length in `__post_init__` therefore cost two O(dimension) Python loops per candidate
    per search and bought nobody anything: 4.91 ms to hydrate 100 candidates of 1024 dimensions, of
    which 3.39 ms was those loops.
    """

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


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredOperation:
    """One append-only control-plane operation-log row.

    The same value describes a pending write and a logged read: a caller supplies everything but
    `operation_id`, and the store returns a copy carrying the assigned id and the effects it
    actually applied.
    """

    operation_key: str
    intent: str
    trigger: str
    operation_json: str
    applied_at: datetime
    model_id: str | None = None
    recipe: str | None = None
    operation_id: int = 0
    created_ids: tuple[str, ...] = ()
    changed_ids: tuple[str, ...] = ()
    # Records this operation moved out of ordinary recall, whether as a FORGET intent or as the
    # consolidation forgetting a CONSOLIDATE carried. Rollback clears exactly these.
    forgotten_ids: tuple[str, ...] = ()
    # Evidence rows this operation actually inserted. Rollback retires exactly these, so a link
    # that predated the operation survives it.
    linked: tuple[tuple[str, str], ...] = ()
    rolled_back_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.operation_key, "operation_key")
        _require_identifier(self.intent, "intent")
        _require_identifier(self.trigger, "trigger")
        if not self.operation_json.strip():
            raise ValueError("operation_json must not be blank")
        _require_aware(self.applied_at, "applied_at")
        if self.rolled_back_at is not None:
            _require_aware(self.rolled_back_at, "rolled_back_at")
        if self.operation_id < 0:
            raise ValueError("operation_id must not be negative")
        for name in ("created_ids", "changed_ids", "forgotten_ids"):
            for memory_id in getattr(self, name):
                _require_identifier(memory_id, name)
        for memory_id, source_memory_id in self.linked:
            _require_identifier(memory_id, "memory_id")
            _require_identifier(source_memory_id, "source_memory_id")


@dataclass(frozen=True, slots=True)
class StoredCandidate:
    """One unit of deliberation work derived from state the store already keeps."""

    trigger: str
    memory_ids: tuple[str, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.trigger, "trigger")
        if not self.memory_ids:
            raise ValueError("a candidate must name at least one memory")
        for memory_id in self.memory_ids:
            _require_identifier(memory_id, "memory_id")
        if self.evidence_count <= 0:
            raise ValueError("evidence_count must be positive")


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
    occurred_end: datetime | None = None

    def __post_init__(self) -> None:
        if self.memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        _require_interval(self.occurred_at, self.occurred_end)


@dataclass(frozen=True, slots=True)
class IndexCandidate:
    """Ranking-path projection of an indexed embedding.

    Retrieval ranks on index scores and event times; it never reads the stored vector or the
    memory content. Hydrating a full ``IndexDocument`` for that would unpack and revalidate one
    FP32 vector per candidate, which measured at twenty-one times the cost of the query that
    produced the row. This projection reads the four columns ranking uses.
    """

    embedding_id: str
    memory_id: str
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.embedding_id, "embedding_id")
        _require_identifier(self.memory_id, "memory_id")
        _require_interval(self.occurred_at, self.occurred_end)


@dataclass(frozen=True, slots=True)
class IdentityLink:
    """A validated face-and-voice merge that can be committed atomically."""

    target_id: str
    source_id: str
    name: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class _IdentityExemplarState:
    modality: Literal["face", "voice"]
    position: int
    model_id: str
    space_id: str
    dimension: int
    vector: bytes
    created_at: str


@dataclass(frozen=True, slots=True)
class _IdentityState:
    identity_id: str
    name: str | None
    created_at: str
    updated_at: str
    exemplars: tuple[_IdentityExemplarState, ...]


@dataclass(frozen=True, slots=True)
class _IdentityChange:
    identity_id: str
    previous: _IdentityState | None


@dataclass(frozen=True, slots=True)
class SpeechRollback:
    """Internal undo token for an unreferenced speech analysis."""

    asset_id: str
    identities: tuple[_IdentityChange, ...]


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
        *,
        formation_pending_at: datetime | None = None,
    ) -> tuple[bool, ...]:
        """Create or update a batch with one commit and one durability sync.

        `formation_pending_at` enqueues each written memory for the follow-up work the caller has
        not done yet, in the same transaction that makes the record durable. The strong `add()`
        path uses it so a crash between this commit and formation leaves a queue row the next
        `settle()` finds, instead of a searchable record nothing will ever form. The row carries
        no state of its own: a queued record that already has vectors owes formation only, which
        is what `read_embedding` tells the settler.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        if not supplied_memories:
            return ()
        if formation_pending_at is not None:
            _require_aware(formation_pending_at, "formation_pending_at")
        with self._transaction() as connection:
            transaction_memory_ids: set[str] = set()
            created_flags = []
            for memory in supplied_memories:
                created_flags.append(
                    self._write_memory(
                        connection,
                        memory,
                        supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                        transaction_memory_ids=transaction_memory_ids,
                    )
                )
                transaction_memory_ids.add(memory.memory_id)
                if formation_pending_at is not None:
                    connection.execute(
                        """
                        INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)
                        ON CONFLICT (memory_id) DO NOTHING
                        """,
                        (memory.memory_id, _datetime_text(formation_pending_at)),
                    )
            for embedding in supplied_embeddings:
                self._write_embedding(connection, embedding)
        return tuple(created_flags)

    def write_captures(
        self,
        memories: Iterable[StoredMemory],
        *,
        enqueued_at: datetime,
    ) -> tuple[str, ...]:
        """Commit new captured records, their media, their context, and their queue rows.

        Returns the IDs this call enqueued. A memory that already exists is left exactly as it is,
        so capturing content that was already added or already queued neither rewrites derived
        content nor re-enqueues work.
        """
        supplied_memories, _embeddings, supplied_by_memory = _prepare_write_batch(memories, ())
        if not supplied_memories:
            return ()
        _require_aware(enqueued_at, "enqueued_at")
        enqueued: list[str] = []
        with self._transaction() as connection:
            transaction_memory_ids: set[str] = set()
            for memory in supplied_memories:
                if (
                    connection.execute(
                        "SELECT 1 FROM memory_records WHERE memory_id = ?",
                        (memory.memory_id,),
                    ).fetchone()
                    is not None
                ):
                    continue
                self._write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                )
                transaction_memory_ids.add(memory.memory_id)
                connection.execute(
                    "INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)",
                    (memory.memory_id, _datetime_text(enqueued_at)),
                )
                enqueued.append(memory.memory_id)
        return tuple(enqueued)

    def settle_capture(
        self,
        memory: StoredMemory,
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Store one captured record's derived content and vectors while it stays queued.

        Returns false when the record is not queued, so a settlement that lost a race writes
        nothing. The queue row is removed by `complete_captures` once every deferred stage,
        formation included, has succeeded; a retry after a later failure re-runs this write, which
        is an idempotent upsert of the same derived content and vectors.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            (memory,),
            embeddings,
        )
        with self._transaction() as connection:
            queued = connection.execute(
                "SELECT 1 FROM capture_queue WHERE memory_id = ?",
                (memory.memory_id,),
            ).fetchone()
            if queued is None:
                return False
            self._write_memory(
                connection,
                supplied_memories[0],
                supplied_embedding_ids=supplied_by_memory[memory.memory_id],
            )
            for embedding in supplied_embeddings:
                self._write_embedding(connection, embedding)
        return True

    def complete_captures(self, memory_ids: Sequence[str]) -> int:
        """Remove captures from the queue after every deferred stage succeeded."""
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            _require_identifier(memory_id, "memory_id")
        if not selected:
            return 0
        removed = 0
        with self._transaction() as connection:
            for offset in range(0, len(selected), _SQLITE_PARAMETER_BATCH):
                batch = selected[offset : offset + _SQLITE_PARAMETER_BATCH]
                removed += connection.execute(
                    f"DELETE FROM capture_queue WHERE memory_id IN "
                    f"({', '.join('?' for _value in batch)})",
                    batch,
                ).rowcount
        return removed

    def record_capture_failure(self, memory_id: str, error: str) -> None:
        """Count one failed settlement and store its reason, leaving the row queued."""
        _require_identifier(memory_id, "memory_id")
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE capture_queue
                SET attempts = attempts + 1, last_error = ?
                WHERE memory_id = ?
                """,
                (error.strip() or None, memory_id),
            )

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
        max_attempts: int | None = None,
    ) -> tuple[PendingCapture, ...]:
        """Return queued captures in enqueue order, optionally restricted or attempt-capped.

        `max_attempts` excludes rows that already failed that many times. They stay queued and
        stay visible, so one poisoned record cannot block every record enqueued after it.

        `awaiting` is read from the vectors themselves: a row whose part-0 embedding exists is
        already searchable and owes formation only, which is the same test settlement applies
        before deciding whether to re-run the model stages.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if memory_ids is not None:
            if not memory_ids:
                return ()
            for memory_id in memory_ids:
                _require_identifier(memory_id, "memory_id")
        selected = None if memory_ids is None else tuple(dict.fromkeys(memory_ids))
        batches: list[tuple[str, ...] | None] = (
            [None]
            if selected is None
            else [
                selected[offset : offset + _SQLITE_PARAMETER_BATCH]
                for offset in range(0, len(selected), _SQLITE_PARAMETER_BATCH)
            ]
        )
        queued: list[PendingCapture] = []
        with self._read_transaction() as connection:
            for batch in batches:
                clauses = (
                    []
                    if batch is None
                    else [f"memory_id IN ({', '.join('?' for _value in batch)})"]
                )
                if max_attempts is not None:
                    clauses.append("attempts < ?")
                restriction = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                queued.extend(
                    PendingCapture(
                        memory_id=_row_text(row, "memory_id"),
                        enqueued_at=_parse_datetime(_row_text(row, "enqueued_at")),
                        attempts=int(row["attempts"]),
                        last_error=row["last_error"],
                        awaiting="formation" if row["embedded"] else "enrichment",
                    )
                    for row in connection.execute(
                        f"""
                        SELECT memory_id, enqueued_at, attempts, last_error,
                               EXISTS (
                                   SELECT 1 FROM embeddings
                                   WHERE embeddings.memory_id = capture_queue.memory_id
                                     AND embeddings.object_part = 0
                               ) AS embedded
                        FROM capture_queue
                        {restriction}
                        ORDER BY enqueued_at, memory_id
                        LIMIT ?
                        """,
                        (
                            *(batch or ()),
                            *(() if max_attempts is None else (max_attempts,)),
                            limit - len(queued),
                        ),
                    ).fetchall()
                )
                if len(queued) >= limit:
                    break
        return tuple(queued[:limit])

    def read_consolidation_candidates(self, *, limit: int) -> tuple[StoredCandidate, ...]:
        """Derive due deliberation work from evidence, lineage, and feedback already recorded.

        There is no queue and no timer behind this: every row is a fact the store already holds,
        read back as a question. Rows are interleaved across triggers so a busy trigger cannot
        starve the others out of the window.
        """
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._read_transaction() as connection:
            consumed = _consumed_at_by_memory(connection)
            groups = (
                _evidence_candidates(connection, consumed, limit),
                _contradiction_candidates(connection, limit),
                _feedback_candidates(connection, consumed, limit),
            )
        interleaved = [
            candidate for row in zip_longest(*groups) for candidate in row if candidate is not None
        ]
        return tuple(interleaved[:limit])

    def apply_formation(
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding],
        *,
        evidence: Sequence[tuple[str, str, float]],
        source_memory_ids: Sequence[str],
        recipe: str,
        completed_at: datetime,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
    ) -> bool:
        """Commit derived records, evidence, and source completion in one transaction.

        `operation` logs the control-plane operation that produced these records in the same
        transaction. Consolidation passes no `source_memory_ids`, so no `formation_runs` marker
        is written and the derived record stays open to further independent evidence.
        `forget_ids` is consolidation forgetting: sources the derived record replaces in ordinary
        recall, retired in this same commit and cleared again by rolling the operation back.
        `require_active` names memories that must still exist and still be un-forgotten inside
        this transaction; if any moved since the caller validated it, nothing is written and
        `StaleOperationError` is raised.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        sources = tuple(dict.fromkeys(source_memory_ids))
        _require_identifier(recipe, "recipe")
        _require_aware(completed_at, "completed_at")
        _validate_formation_links(sources, forget_ids, evidence)
        with self._transaction() as connection:
            # Same in-transaction idempotency check `apply_control_operation` makes: a duplicate
            # that arrives after the caller's pre-check must be refused, not surface as a
            # unique-index violation.
            if (
                operation is not None
                and _active_operation_id(connection, operation.operation_key) is not None
            ):
                return False
            _require_active(connection, require_active)
            if any(
                connection.execute(
                    """
                    SELECT 1 FROM formation_runs
                    WHERE source_memory_id = ? AND recipe = ?
                    """,
                    (source_memory_id, recipe),
                ).fetchone()
                is not None
                for source_memory_id in sources
            ):
                return False
            transaction_memory_ids: set[str] = set()
            for memory in supplied_memories:
                self._write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                )
                transaction_memory_ids.add(memory.memory_id)
            for embedding in supplied_embeddings:
                self._write_embedding(connection, embedding)
            linked: list[tuple[str, str]] = []
            for memory_id, source_memory_id, confidence in evidence:
                if _add_memory_evidence(
                    connection,
                    memory_id,
                    source_memory_id,
                    confidence=confidence,
                    recorded_at=completed_at,
                ):
                    linked.append((memory_id, source_memory_id))
            _refresh_multi_source_projections(
                connection,
                supplied_memories,
                changed_at=completed_at,
            )
            connection.executemany(
                """
                INSERT INTO formation_runs (source_memory_id, recipe, completed_at)
                VALUES (?, ?, ?)
                """,
                (
                    (source_memory_id, recipe, _datetime_text(completed_at))
                    for source_memory_id in sources
                ),
            )
            if operation is not None:
                forgotten = _set_forgotten(connection, forget_ids, forgotten_at=completed_at)
                _insert_operation(
                    connection,
                    replace(operation, forgotten_ids=forgotten, linked=tuple(linked)),
                )
        return True

    def replace_memory_embeddings(
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding],
    ) -> None:
        """Atomically replace every vector belonging to supplied existing memories."""
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        if not supplied_memories:
            return
        with self._transaction() as connection:
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )

    def formation_completed(self, source_memory_id: str, recipe: str) -> bool:
        """Return whether one source was successfully formed with this recipe."""
        _require_identifier(source_memory_id, "source_memory_id")
        _require_identifier(recipe, "recipe")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM formation_runs
                WHERE source_memory_id = ? AND recipe = ?
                """,
                (source_memory_id, recipe),
            ).fetchone()
        return row is not None

    def mark_formation_completed(
        self,
        source_memory_id: str,
        recipe: str,
        *,
        completed_at: datetime,
    ) -> None:
        """Durably mark an idempotent automatic formation run as complete."""
        _require_identifier(source_memory_id, "source_memory_id")
        _require_identifier(recipe, "recipe")
        _require_aware(completed_at, "completed_at")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO formation_runs (
                    source_memory_id, recipe, completed_at
                ) VALUES (?, ?, ?)
                """,
                (source_memory_id, recipe, _datetime_text(completed_at)),
            )

    def add_memory_evidence(
        self,
        memory_id: str,
        source_memory_id: str,
        *,
        confidence: float,
        recorded_at: datetime,
    ) -> bool:
        """Attach independent evidence and version the derived confidence projection."""
        _require_identifier(memory_id, "memory_id")
        _require_identifier(source_memory_id, "source_memory_id")
        _require_aware(recorded_at, "recorded_at")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        with self._transaction() as connection:
            return _add_memory_evidence(
                connection,
                memory_id,
                source_memory_id,
                confidence=float(confidence),
                recorded_at=recorded_at,
            )

    def apply_control_operation(
        self,
        operation: StoredOperation,
        *,
        reinforce: Sequence[tuple[str, str]] = (),
        correct_ids: Sequence[str] = (),
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
    ) -> StoredOperation | None:
        """Apply one already-validated operation and its log row in one transaction.

        The caller owns policy; this only executes the supplied effects. It returns `None` when
        the operation key is already applied and not rolled back.

        `require_active` names memories the caller validated and that must still exist and still
        be un-forgotten here. That check and the all-or-nothing effect check below are what make
        the gap between validation and this transaction safe: a target or source that moved in
        between raises `StaleOperationError` with nothing written, so the log never claims an
        effect that did not happen.
        """
        for memory_id, source_memory_id in reinforce:
            _require_identifier(memory_id, "memory_id")
            _require_identifier(source_memory_id, "source_memory_id")
        for memory_id in (*correct_ids, *forget_ids):
            _require_identifier(memory_id, "memory_id")
        with self._transaction() as connection:
            if _active_operation_id(connection, operation.operation_key) is not None:
                return None
            _require_active(connection, require_active)
            changed: list[str] = []
            linked: list[tuple[str, str]] = []
            for memory_id, source_memory_id in reinforce:
                if not _add_memory_evidence(
                    connection,
                    memory_id,
                    source_memory_id,
                    confidence=_asserted_confidence(connection, memory_id),
                    recorded_at=operation.applied_at,
                ):
                    raise StaleOperationError(f"{source_memory_id} already supports {memory_id}")
                changed.append(memory_id)
                linked.append((memory_id, source_memory_id))
            retired = _retire_memory_versions(
                connection, correct_ids, retired_at=operation.applied_at
            )
            _require_every(retired, correct_ids, "retire")
            changed.extend(retired)
            forgotten = _set_forgotten(connection, forget_ids, forgotten_at=operation.applied_at)
            _require_every(forgotten, forget_ids, "forget")
            changed.extend(forgotten)
            applied = replace(
                operation,
                changed_ids=tuple(dict.fromkeys(changed)),
                forgotten_ids=forgotten,
                linked=tuple(linked),
            )
            return replace(applied, operation_id=_insert_operation(connection, applied))

    def rollback_operation(
        self,
        operation_id: int,
        *,
        rolled_back_at: datetime,
        retire_evidence: Sequence[tuple[str, str]] = (),
        restore_versions: Sequence[str] = (),
        clear_forgotten: Sequence[str] = (),
        delete_memory_ids: Sequence[str] = (),
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Apply the caller's reversal and mark one operation rolled back, atomically.

        Returns `(False, ())` when the operation is unknown or already rolled back, in which case
        no reversal is applied. Otherwise the second value lists the assets that the deleted
        records were the last to reference; index cleanup follows through the durable outbox.
        """
        _require_aware(rolled_back_at, "rolled_back_at")
        for memory_id in delete_memory_ids:
            _require_identifier(memory_id, "memory_id")
        unreferenced: list[StoredAsset] = []
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT applied_at FROM memory_operations
                WHERE operation_id = ? AND rolled_back_at IS NULL
                """,
                (operation_id,),
            ).fetchone()
            if row is None:
                return False, ()
            reverted_at = max(rolled_back_at, _parse_datetime(_row_text(row, "applied_at")))
            for memory_id in dict.fromkeys(delete_memory_ids):
                _deleted, orphaned = self._delete_memory(connection, memory_id)
                unreferenced.extend(orphaned)
            _retire_memory_evidence(connection, retire_evidence, retired_at=reverted_at)
            _restore_memory_versions(connection, restore_versions, recorded_at=reverted_at)
            _set_forgotten(connection, clear_forgotten, forgotten_at=None)
            connection.execute(
                "UPDATE memory_operations SET rolled_back_at = ? WHERE operation_id = ?",
                (_datetime_text(reverted_at), operation_id),
            )
        return True, tuple({asset.asset_id: asset for asset in unreferenced}.values())

    def read_operations(
        self,
        *,
        limit: int = 100,
        operation_id: int | None = None,
        operation_key: str | None = None,
    ) -> tuple[StoredOperation, ...]:
        """Return logged operations newest first, or the one matching an id or active key."""
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        where = ""
        parameters: tuple[object, ...] = ()
        if operation_id is not None:
            where = "WHERE operation_id = ?"
            parameters = (operation_id,)
        elif operation_key is not None:
            _require_identifier(operation_key, "operation_key")
            where = "WHERE operation_key = ? AND rolled_back_at IS NULL"
            parameters = (operation_key,)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT operation_id, operation_key, intent, trigger, model_id, recipe,
                       operation_json, effects_json, applied_at, rolled_back_at
                FROM memory_operations
                {where}
                ORDER BY operation_id DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def read_memory(self, memory_id: str) -> StoredMemory | None:
        """Return one memory, or none when it does not exist."""
        _require_identifier(memory_id, "memory_id")
        with self._read_transaction() as connection:
            row = connection.execute(
                """
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, occurred_end, last_accessed_at, access_count,
                       created_at, updated_at, place_id, forgotten_at
                FROM memory_records
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            assets = (
                () if row is None else self._read_memory_assets(connection, (memory_id,))[memory_id]
            )
            contexts, _semantic_ids = _read_memory_contexts(connection, (memory_id,))
        return (
            None
            if row is None
            else _memory_from_row(row, assets=assets, context=contexts.get(memory_id))
        )

    def read_memories(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        active_only: bool = False,
    ) -> tuple[StoredMemory, ...]:
        """Hydrate existing memories with one query and preserve input ranking.

        `place_id` scopes the slate to one symbolic place. It is a hard filter, unlike `near`/
        `radius_m` which the caller applies to metric pose, and it is applied in SQL so a scoped
        hydration reads only the rows it returns.
        """
        if not memory_ids:
            return ()
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        _require_optional_identifier(place_id, "place_id")
        place_clause = "" if place_id is None else "AND place_id = ?"
        place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
        rows: list[sqlite3.Row] = []
        with self._read_transaction() as connection:
            for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT memory_id, content, modality, memory_type, metadata_json,
                               occurred_at, occurred_end, last_accessed_at, access_count,
                               created_at, updated_at, place_id, forgotten_at
                        FROM memory_records
                        WHERE memory_id IN ({placeholders})
                        {place_clause}
                        """,
                        (*batch, *place_parameters),
                    ).fetchall()
                )
            assets_by_memory = self._read_memory_assets(connection, tuple(memory_ids))
            contexts, semantic_ids = _read_memory_contexts(
                connection,
                tuple(memory_ids),
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                active_only=active_only,
            )
        by_id = {
            _row_text(row, "memory_id"): _memory_from_row(
                row,
                assets=assets_by_memory.get(_row_text(row, "memory_id"), ()),
                context=contexts.get(_row_text(row, "memory_id")),
            )
            for row in rows
            if not active_only
            or (
                row["forgotten_at"] is None
                and (
                    (
                        # A record with no `memory_semantics` row declares no validity interval, so
                        # it is valid at every `valid_at` exactly like a version row whose
                        # `valid_from` and `valid_until` are NULL; only recorded time can exclude
                        # it, and there `created_at` is the honest bound. `near` is separate and
                        # spatial: a record with no pose is not at any location, so it drops just
                        # as a semantic row without a pose does above.
                        near is None
                        and _row_text(row, "memory_id") not in semantic_ids
                        and (
                            known_at is None
                            or _parse_datetime(_row_text(row, "created_at")) <= known_at
                        )
                    )
                    or _row_text(row, "memory_id") in contexts
                )
            )
        }
        return tuple(by_id[memory_id] for memory_id in memory_ids if memory_id in by_id)

    def embedding_ids_in_range(
        self,
        occurred_from: datetime,
        occurred_until: datetime,
        *,
        space_id: str,
        task: str,
        memory_type: str | None = None,
    ) -> tuple[frozenset[str], int]:
        """Return current aggregate embedding IDs in a time range and the searchable total."""
        _require_aware(occurred_from, "occurred_from")
        _require_aware(occurred_until, "occurred_until")
        if occurred_until <= occurred_from:
            raise ValueError("occurred_until must be later than occurred_from")
        _require_identifier(space_id, "space_id")
        _require_identifier(task, "task")
        if memory_type is not None and memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type is invalid")
        start = _datetime_text(occurred_from)
        until = _datetime_text(occurred_until)
        type_clause = "" if memory_type is None else "AND m.memory_type = ?"
        scope: tuple[object, ...] = (
            (space_id, task) if memory_type is None else (space_id, task, memory_type)
        )
        with self._read_transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT e.embedding_id
                FROM embeddings AS e
                JOIN memory_records AS m ON m.memory_id = e.memory_id
                WHERE e.space_id = ? AND e.task = ? AND e.object_part = 0
                  AND m.occurred_at IS NOT NULL
                  AND (
                      (m.occurred_end IS NOT NULL AND m.occurred_end > ?)
                      OR (m.occurred_end IS NULL AND m.occurred_at >= ?)
                  )
                  AND m.occurred_at < ?
                  {type_clause}
                """,
                (*scope[:2], start, start, until, *scope[2:]),
            ).fetchall()
            total_row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM embeddings AS e
                JOIN memory_records AS m ON m.memory_id = e.memory_id
                WHERE e.space_id = ? AND e.task = ? AND e.object_part = 0
                {type_clause}
                """,
                scope,
            ).fetchone()
        total = 0 if total_row is None else int(total_row["count"])
        return frozenset(_row_text(row, "embedding_id") for row in rows), total

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
                       occurred_at, occurred_end, last_accessed_at, access_count,
                       created_at, updated_at, place_id, forgotten_at
                FROM memory_records
                {where}
                ORDER BY created_at DESC, memory_id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            memory_ids = tuple(_row_text(row, "memory_id") for row in rows)
            assets_by_memory = self._read_memory_assets(connection, memory_ids)
            contexts, _semantic_ids = _read_memory_contexts(connection, memory_ids)
        return tuple(
            _memory_from_row(
                row,
                assets=assets_by_memory.get(_row_text(row, "memory_id"), ()),
                context=contexts.get(_row_text(row, "memory_id")),
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

    def set_forgotten(
        self,
        memory_ids: Sequence[str],
        *,
        forgotten_at: datetime | None,
    ) -> tuple[str, ...]:
        """Set or clear cognitive forgetting and return the ids whose state changed."""
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            _require_identifier(memory_id, "memory_id")
        if forgotten_at is not None:
            _require_aware(forgotten_at, "forgotten_at")
        if not ids:
            return ()
        with self._transaction() as connection:
            return _set_forgotten(connection, ids, forgotten_at=forgotten_at)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory; cascading embedding triggers enqueue index deletions."""
        deleted, _assets = self.delete_memory_with_assets(memory_id)
        return deleted

    def delete_memory_with_assets(self, memory_id: str) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory and return its assets that no remaining memory references."""
        _require_identifier(memory_id, "memory_id")
        with self._transaction() as connection:
            return self._delete_memory(connection, memory_id)

    def _delete_memory(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory inside the caller's transaction; see `delete_memory_with_assets`."""
        linked_ids = [
            _row_text(row, "asset_id")
            for row in connection.execute(
                "SELECT asset_id FROM memory_assets WHERE memory_id = ? ORDER BY position",
                (memory_id,),
            ).fetchall()
        ]
        target_semantic = connection.execute(
            """
            SELECT lineage_id, kind, basis
            FROM memory_semantics WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        reconciled: set[tuple[str, str]] = set()
        if target_semantic is not None and (
            _row_text(target_semantic, "kind") == MemoryKind.STATE.value
            or (
                _row_text(target_semantic, "kind") == MemoryKind.TRAIT.value
                and _row_text(target_semantic, "basis") == EvidenceBasis.USER_STATEMENT.value
            )
        ):
            reconciled.add(
                (
                    _row_text(target_semantic, "lineage_id"),
                    _row_text(target_semantic, "kind"),
                )
            )
        dependent_rows = connection.execute(
            """
            SELECT e.memory_id, e.recorded_at, s.lineage_id, s.kind, s.basis
            FROM memory_evidence AS e
            JOIN memory_semantics AS s ON s.memory_id = e.memory_id
            WHERE e.source_memory_id = ? AND e.retired_at IS NULL
            ORDER BY e.memory_id
            """,
            (memory_id,),
        ).fetchall()
        for dependent in dependent_rows:
            if _row_text(dependent, "kind") == MemoryKind.STATE.value or (
                _row_text(dependent, "kind") == MemoryKind.TRAIT.value
                and _row_text(dependent, "basis") == EvidenceBasis.USER_STATEMENT.value
            ):
                reconciled.add(
                    (
                        _row_text(dependent, "lineage_id"),
                        _row_text(dependent, "kind"),
                    )
                )
        affected_ids = [_row_text(row, "memory_id") for row in dependent_rows]
        for lineage_id, kind in reconciled:
            affected_ids.extend(
                _row_text(row, "memory_id")
                for row in connection.execute(
                    """
                    SELECT memory_id FROM memory_semantics
                    WHERE lineage_id = ? AND kind = ?
                    """,
                    (lineage_id, kind),
                ).fetchall()
            )
        changed_at = _next_semantic_transaction_time(
            connection,
            datetime.now(timezone.utc),
            affected_ids,
        )
        connection.execute(
            """
            UPDATE memory_evidence SET retired_at = ?
            WHERE source_memory_id = ? AND retired_at IS NULL
            """,
            (_datetime_text(changed_at), memory_id),
        )
        for dependent in dependent_rows:
            dependent_id = _row_text(dependent, "memory_id")
            evidence_count, _confidence = _evidence_summary(connection, dependent_id)
            if evidence_count:
                _refresh_evidence_projection(connection, dependent_id, changed_at)
                continue
            linked_ids.extend(
                _row_text(row, "asset_id")
                for row in connection.execute(
                    """
                    SELECT asset_id FROM memory_assets
                    WHERE memory_id = ? ORDER BY position
                    """,
                    (dependent_id,),
                ).fetchall()
            )
            connection.execute(
                "DELETE FROM memory_records WHERE memory_id = ?",
                (dependent_id,),
            )
        cursor = connection.execute(
            "DELETE FROM memory_records WHERE memory_id = ?",
            (memory_id,),
        )
        for lineage_id, kind in sorted(reconciled):
            _rebuild_reconciled_lineage(
                connection,
                lineage_id,
                kind,
                changed_at=changed_at,
            )
        unreferenced = self._read_unreferenced_assets(
            connection,
            tuple(dict.fromkeys(linked_ids)),
        )
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
        # Provenance and calibration live on MemoryConfig.speaker_similarity/speaker_margin in
        # plugins.py; every product call site passes them, so treat these literals as a test
        # convenience and not as a settled threshold.
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[SpeakerSegment, ...]:
        """Persist one analysis and match its CAM++ exemplars to local identities."""
        segments, _rollback = self.write_speech_reversible(
            asset_id,
            analysis,
            model_id=model_id,
            space_id=space_id,
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
            preferred_identity=preferred_identity,
        )
        return segments

    def write_speech_reversible(
        self,
        asset_id: str,
        analysis: SpeechAnalysis,
        *,
        model_id: str,
        space_id: str,
        # See write_speech: MemoryConfig.speaker_similarity/speaker_margin own these values.
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[tuple[SpeakerSegment, ...], SpeechRollback | None]:
        """Persist speech and return an undo token when this call created the analysis."""
        _sha256(asset_id)
        _require_identifier(model_id, "speech model_id")
        _require_identifier(space_id, "speech space_id")
        if preferred_identity is not None:
            _require_identifier(preferred_identity, "preferred_identity")
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
            raise ValueError("every speaker turn label must have an exemplar")
        dimensions = {len(values) for values in speakers.values()}
        if 0 in dimensions or len(dimensions) > 1:
            raise ValueError("speaker exemplars must share one non-zero dimension")
        normalized = {
            label: _normalized_vector(values, "speaker exemplar")
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
                return self._read_speech(connection, asset_id), None
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("speech analysis requires a stored media asset")

            identities, identity_changes = self._match_speakers(
                connection,
                normalized,
                model_id=model_id,
                space_id=space_id,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
                now=now,
                preferred_identity=preferred_identity,
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
            return self._read_speech(connection, asset_id), SpeechRollback(
                asset_id,
                identity_changes,
            )

    def rollback_speech(self, rollback: SpeechRollback) -> bool:
        """Undo one new analysis while its asset is still unreferenced by a memory."""
        if not isinstance(rollback, SpeechRollback):
            raise ValueError("rollback must be a SpeechRollback value")
        with self._transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM memory_assets WHERE asset_id = ? LIMIT 1",
                    (rollback.asset_id,),
                ).fetchone()
                is not None
            ):
                return False
            deleted = connection.execute(
                "DELETE FROM speech_analyses WHERE asset_id = ?",
                (rollback.asset_id,),
            )
            if deleted.rowcount == 0:
                return False
            for change in reversed(rollback.identities):
                previous = change.previous
                if previous is None:
                    removed = connection.execute(
                        """
                        DELETE FROM identities
                        WHERE identity_id = ?
                          AND NOT EXISTS (
                              SELECT 1 FROM speech_segments WHERE speaker_id = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM face_observations WHERE identity_id = ?
                          )
                        """,
                        (change.identity_id, change.identity_id, change.identity_id),
                    )
                    if removed.rowcount != 1:
                        raise RuntimeError("new speaker identity could not be rolled back")
                    continue
                connection.execute(
                    """
                    INSERT INTO identities (identity_id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (identity_id) DO UPDATE SET
                        name = excluded.name,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        previous.identity_id,
                        previous.name,
                        previous.created_at,
                        previous.updated_at,
                    ),
                )
                connection.execute(
                    "DELETE FROM identity_exemplars WHERE identity_id = ?",
                    (previous.identity_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO identity_exemplars (
                        identity_id, modality, position, model_id, space_id,
                        dimension, vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            previous.identity_id,
                            exemplar.modality,
                            exemplar.position,
                            exemplar.model_id,
                            exemplar.space_id,
                            exemplar.dimension,
                            exemplar.vector,
                            exemplar.created_at,
                        )
                        for exemplar in previous.exemplars
                    ),
                )
            return True

    def read_faces(
        self,
        asset_id: str,
        *,
        space_id: str,
    ) -> tuple[FaceObservation, ...] | None:
        """Return cached face observations, including an empty completed analysis."""
        _sha256(asset_id)
        _require_identifier(space_id, "face space_id")
        with self._connection() as connection:
            analysis = connection.execute(
                "SELECT 1 FROM face_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if analysis is None:
                return None
            return self._read_faces(connection, asset_id)

    def write_faces(
        self,
        asset_id: str,
        analysis: FaceAnalysis,
        *,
        model_id: str,
        space_id: str,
        analysis_space_id: str | None = None,
        # See write_speech: MemoryConfig.face_similarity/face_margin own these values.
        minimum_similarity: float = 0.363,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[FaceObservation, ...]:
        """Persist detected faces and match them against durable identity exemplars.

        ``preferred_identity`` lets a caller enroll this modality into an identity that lacks it,
        for a single unambiguous observation. The product write path deliberately does not use it:
        adopting an asset's lone voice for its lone face is a cross-modal claim made from one
        asset, which `Memory` instead routes through corroborated linking so that it is recorded,
        observable, and reversible. Pass it only where that claim is already established.
        """
        _sha256(asset_id)
        _require_identifier(model_id, "face model_id")
        _require_identifier(space_id, "face space_id")
        selected_analysis_space = space_id if analysis_space_id is None else analysis_space_id
        _require_identifier(selected_analysis_space, "face analysis_space_id")
        if preferred_identity is not None:
            _require_identifier(preferred_identity, "preferred_identity")
        if not isinstance(analysis, FaceAnalysis):
            raise ValueError("analysis must be a FaceAnalysis value")
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
        dimensions = {len(face.values) for face in analysis.faces}
        if 0 in dimensions or len(dimensions) > 1:
            raise ValueError("face exemplars must share one non-zero dimension")
        observations = {
            face.face_label: (_normalized_vector(face.values, "face exemplar"),)
            for face in analysis.faces
        }
        claim_groups = {face.face_label: face.observed_at_ms for face in analysis.faces}
        now = datetime.now(timezone.utc)
        with self._transaction() as connection:
            cached = connection.execute(
                "SELECT 1 FROM face_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, selected_analysis_space),
            ).fetchone()
            if cached is not None:
                return self._read_faces(connection, asset_id)
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("face analysis requires a stored media asset")
            matches, _changes = self._match_identities(
                connection,
                observations,
                claim_groups=claim_groups,
                modality="face",
                model_id=model_id,
                space_id=space_id,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
                exemplar_limit=_FACE_EXEMPLAR_LIMIT,
                now=now,
                preferred_identity=preferred_identity,
            )
            connection.execute("DELETE FROM face_analyses WHERE asset_id = ?", (asset_id,))
            connection.execute(
                """
                INSERT INTO face_analyses (asset_id, model_id, space_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (asset_id, model_id, selected_analysis_space, _datetime_text(now)),
            )
            connection.executemany(
                """
                INSERT INTO face_observations (
                    asset_id, position, observed_at_ms, box_x, box_y,
                    box_width, box_height, identity_id, identity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        asset_id,
                        position,
                        face.observed_at_ms,
                        *face.bounding_box,
                        matches[face.face_label][0],
                        matches[face.face_label][1],
                    )
                    for position, face in enumerate(analysis.faces)
                ),
            )
            return self._read_faces(connection, asset_id)

    def resolve_identity_id(self, identity_id: str) -> str | None:
        """Resolve a current or merged identity ID to its canonical identity."""
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            return _resolve_identity_id(connection, identity_id)

    def speaker_memory_ids(self, speaker_id: str) -> tuple[str, ...] | None:
        """Return memories containing a speaker, or none when the identity is unknown."""
        _require_identifier(speaker_id, "speaker_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, speaker_id)
            if resolved_id is None:
                return None
            if (
                connection.execute(
                    """
                    SELECT 1
                    FROM identity_exemplars
                    WHERE identity_id = ? AND modality = 'voice'
                    LIMIT 1
                    """,
                    (resolved_id,),
                ).fetchone()
                is None
            ):
                return None
            rows = connection.execute(
                """
                SELECT DISTINCT ma.memory_id
                FROM speech_segments AS s
                JOIN memory_assets AS ma ON ma.asset_id = s.asset_id
                WHERE s.speaker_id = ?
                ORDER BY ma.memory_id
                """,
                (resolved_id,),
            ).fetchall()
        return tuple(_row_text(row, "memory_id") for row in rows)

    def identity_memory_ids(self, identity_id: str) -> tuple[str, ...] | None:
        """Return memories containing a face or voice occurrence for one identity."""
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            rows = connection.execute(
                """
                SELECT DISTINCT ma.memory_id
                FROM memory_assets AS ma
                WHERE EXISTS (
                    SELECT 1 FROM speech_segments AS s
                    WHERE s.asset_id = ma.asset_id AND s.speaker_id = ?
                ) OR EXISTS (
                    SELECT 1 FROM face_observations AS f
                    WHERE f.asset_id = ma.asset_id AND f.identity_id = ?
                )
                ORDER BY ma.memory_id
                """,
                (resolved_id, resolved_id),
            ).fetchall()
        return tuple(_row_text(row, "memory_id") for row in rows)

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Assign a display name and atomically replace affected indexed memories."""
        _require_identifier(speaker_id, "speaker_id")
        return self.register_identity(
            speaker_id,
            name,
            memories=memories,
            embeddings=embeddings,
        )

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Assign a display name and atomically replace affected indexed memories.

        A None relationship keeps whatever relationship is already recorded, so
        renaming an identity never silently drops it.
        """
        _require_identifier(identity_id, "identity_id")
        normalized_name = _identity_name(name)
        normalized_relationship = (
            None if relationship is None else _identity_name(relationship, "relationship")
        )
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return False
            cursor = connection.execute(
                """
                UPDATE identities
                SET name = ?,
                    relationship = COALESCE(?, relationship),
                    updated_at = ?
                WHERE identity_id = ?
                """,
                (normalized_name, normalized_relationship, now, resolved_id),
            )
            if cursor.rowcount == 0:
                return False
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        return cursor.rowcount > 0

    def identity_profile(self, identity_id: str) -> IdentityProfile | None:
        """Return one identity's profile, or None when it does not exist.

        A merged identity resolves through its alias, like every other identity read, and the
        returned profile carries the canonical ID so a caller never has to resolve it twice.
        """
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            row = connection.execute(
                "SELECT name, relationship FROM identities WHERE identity_id = ?",
                (resolved_id,),
            ).fetchone()
        if row is None:
            return None
        return IdentityProfile(
            identity_id=resolved_id,
            name=_optional_row_text(row, "name"),
            relationship=_optional_row_text(row, "relationship"),
        )

    def record_identity_link_evidence(self, voice_id: str, face_id: str, asset_id: str) -> int:
        """Record one voice-and-face co-occurrence and count the pair's distinct assets.

        Recording the same triple again is idempotent. Returns zero when either
        identity is unknown, or when both resolve to the same identity because the pair has
        already merged, so a caller can accumulate corroboration across assets before it
        commits an irreversible cross-modal merge.
        """
        _require_identifier(voice_id, "voice identity_id")
        _require_identifier(face_id, "face identity_id")
        _sha256(asset_id)
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            voice = _resolve_identity_id(connection, voice_id)
            face = _resolve_identity_id(connection, face_id)
            if voice is None or face is None or voice == face:
                return 0
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("identity link evidence requires a stored media asset")
            connection.execute(
                """
                INSERT INTO identity_link_evidence (voice_id, face_id, asset_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (voice_id, face_id, asset_id) DO NOTHING
                """,
                (voice, face, asset_id, now),
            )
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT asset_id) AS assets
                FROM identity_link_evidence
                WHERE voice_id = ? AND face_id = ?
                """,
                (voice, face),
            ).fetchone()
        return int(row["assets"])

    def identity_link_plan(
        self,
        first_id: str,
        second_id: str,
        *,
        allow_shared_modality: bool = False,
    ) -> IdentityLink | None:
        """Return the currently valid merge plan for two complementary identities.

        With allow_shared_modality a shared modality no longer blocks the plan, so an
        established identity can re-absorb a single-modality fragment instead of that
        fragment staying orphaned forever. Two identities that each hold face and
        voice stay refused even then: that merge fuses two complete people and is not
        recoverable in bulk.
        """
        _require_identifier(first_id, "first identity_id")
        _require_identifier(second_id, "second identity_id")
        with self._connection() as connection:
            return self._identity_link_plan(
                connection,
                first_id,
                second_id,
                allow_shared_modality=allow_shared_modality,
            )

    def link_identities(
        self,
        first_id: str,
        second_id: str,
        *,
        expected: IdentityLink | None = None,
        allow_shared_modality: bool = False,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> str | None:
        """Merge two identities under one stable ID, recording a reversible alias.

        Pass allow_shared_modality to commit a plan obtained with the same intent.
        """
        _require_identifier(first_id, "first identity_id")
        _require_identifier(second_id, "second identity_id")
        if expected is not None and not isinstance(expected, IdentityLink):
            raise ValueError("expected must be an IdentityLink value or None")
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        with self._transaction() as connection:
            plan = self._identity_link_plan(
                connection,
                first_id,
                second_id,
                allow_shared_modality=allow_shared_modality,
            )
            if plan is None:
                first = _resolve_identity_id(connection, first_id)
                second = _resolve_identity_id(connection, second_id)
                return first if first is not None and first == second else None
            if expected is not None and plan != expected:
                return None
            target, source = plan.target_id, plan.source_id
            contributed = _sole_identity_modality(connection, source)
            now = _datetime_text(datetime.now(timezone.utc))
            connection.execute(
                "UPDATE speech_segments SET speaker_id = ? WHERE speaker_id = ?",
                (target, source),
            )
            connection.execute(
                "UPDATE face_observations SET identity_id = ? WHERE identity_id = ?",
                (target, source),
            )
            _merge_identity_exemplars(connection, target, source)
            connection.execute(
                """
                UPDATE identities
                SET name = ?, created_at = ?, updated_at = ?
                WHERE identity_id = ?
                """,
                (
                    plan.name,
                    plan.created_at,
                    now,
                    target,
                ),
            )
            connection.execute(
                "UPDATE identity_aliases SET identity_id = ? WHERE identity_id = ?",
                (target, source),
            )
            # Re-point accumulated evidence by copying it onto the target first: the
            # same asset may already carry a row for the target, and the primary key
            # would reject a plain UPDATE.
            connection.execute(
                """
                INSERT INTO identity_link_evidence (voice_id, face_id, asset_id, created_at)
                SELECT CASE WHEN voice_id = ? THEN ? ELSE voice_id END,
                       CASE WHEN face_id = ? THEN ? ELSE face_id END,
                       asset_id,
                       created_at
                FROM identity_link_evidence
                WHERE voice_id = ? OR face_id = ?
                ON CONFLICT (voice_id, face_id, asset_id) DO NOTHING
                """,
                (source, target, source, target, source, source),
            )
            connection.execute(
                "DELETE FROM identity_link_evidence WHERE voice_id = ? OR face_id = ?",
                (source, source),
            )
            # Re-pointing turns this pair's own evidence into voice_id = face_id rows, which
            # describe an identity co-occurring with itself and can never yield a plan.
            connection.execute("DELETE FROM identity_link_evidence WHERE voice_id = face_id")
            connection.execute(
                """
                INSERT INTO identity_aliases (
                    alias_id, identity_id, created_at, contributed_modality
                ) VALUES (?, ?, ?, ?)
                """,
                (source, target, now, contributed),
            )
            connection.execute("DELETE FROM identities WHERE identity_id = ?", (source,))
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return target

    def unlink_identity(self, alias_id: str) -> str | None:
        """Reverse one recorded merge, restoring alias_id as an independent identity.

        Returns the restored identity_id, or None when alias_id is not a reversible
        alias: an unknown alias, an alias whose source held both modalities, and an
        alias merged before this schema recorded the contributed modality all have no
        recorded split point. Also returns None, changing nothing, when the target no
        longer holds the other modality, because the split would leave it with no
        exemplars at all.

        The restored identity gets no name and no relationship: a merge keeps only one
        profile and the target keeps it, so name the restored identity again if it
        needs one. Every exemplar and observation of the contributed modality moves
        back, not only the rows the source originally supplied, so unlinking a
        re-absorbed fragment also hands over what the target learned in that modality.

        Unlinking clears the pair's accumulated link evidence but does not suppress the
        pair, so continued ingestion can corroborate and merge them again. Treat this as
        resetting the evidence, not as recording that a human rejected the merge.
        """
        _require_identifier(alias_id, "alias_id")
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            alias = connection.execute(
                """
                SELECT identity_id, created_at, contributed_modality
                FROM identity_aliases
                WHERE alias_id = ?
                """,
                (alias_id,),
            ).fetchone()
            if alias is None or alias["contributed_modality"] is None:
                return None
            target = _row_text(alias, "identity_id")
            modality = _identity_modality(alias["contributed_modality"])
            modalities = {
                _identity_modality(row["modality"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT modality
                    FROM identity_exemplars
                    WHERE identity_id = ?
                    """,
                    (target,),
                ).fetchall()
            }
            if modalities != {"face", "voice"}:
                return None
            connection.execute(
                """
                INSERT INTO identities (identity_id, name, relationship, created_at, updated_at)
                VALUES (?, NULL, NULL, ?, ?)
                """,
                (alias_id, _row_text(alias, "created_at"), now),
            )
            connection.execute(
                """
                UPDATE identity_exemplars
                SET identity_id = ?
                WHERE identity_id = ? AND modality = ?
                """,
                (alias_id, target, modality),
            )
            if modality == "face":
                connection.execute(
                    "UPDATE face_observations SET identity_id = ? WHERE identity_id = ?",
                    (alias_id, target),
                )
            else:
                connection.execute(
                    "UPDATE speech_segments SET speaker_id = ? WHERE speaker_id = ?",
                    (alias_id, target),
                )
            connection.execute("DELETE FROM identity_aliases WHERE alias_id = ?", (alias_id,))
            connection.execute(
                """
                DELETE FROM identity_link_evidence
                WHERE voice_id IN (?, ?) AND face_id IN (?, ?)
                """,
                (target, alias_id, target, alias_id),
            )
            return alias_id

    def identity_equivalence_class(self, identity_id: str) -> tuple[str, ...] | None:
        """Return every ID that names one identity: the canonical ID first, then its aliases.

        Accepts a canonical ID or any merged alias, and returns None when none of them is
        known. Note what this is *not* useful for: `link_identities` re-points every speech
        segment, face observation and exemplar onto the canonical ID, so expanding a read
        across the returned class retrieves exactly what the canonical ID alone retrieves.
        The class is the erasure and audit surface -- which IDs still admit a forgotten
        person -- not a recall lever.
        """
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            return self._identity_equivalence_class(connection, identity_id)

    @staticmethod
    def _identity_equivalence_class(
        connection: sqlite3.Connection,
        identity_id: str,
    ) -> tuple[str, ...] | None:
        resolved_id = _resolve_identity_id(connection, identity_id)
        if resolved_id is None:
            return None
        aliases = connection.execute(
            "SELECT alias_id FROM identity_aliases WHERE identity_id = ? ORDER BY alias_id",
            (resolved_id,),
        ).fetchall()
        return (resolved_id, *(_row_text(row, "alias_id") for row in aliases))

    def forget_identity(
        self,
        identity_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> IdentityErasure | None:
        """Erase one person's identity cluster, keeping the memories that mention them.

        Accepts a canonical ID or any merged alias and removes the whole cluster: the profile,
        every face and voice exemplar, every alias, and the accumulated cross-modal link
        evidence. Returns None, changing nothing, when no such identity exists.

        Memories, their content and their media assets survive -- deleting a person must not
        delete the family's memory of the events. Their identity annotations do not: speech
        segments keep their transcript with `speaker_id` scrubbed to NULL, and face
        observations are removed outright, because a face row's entire payload is a box plus
        the identity claim and stripping the claim leaves only a biometric locator. The
        cached `face_analyses`/`speech_analyses` rows deliberately stay, so re-analysing
        already stored media cannot re-mint the person from the same clip.

        No tombstone is recorded, and this deliberately does not stop a future encounter from
        minting a fresh identity. Recognising someone as previously-forgotten requires keeping
        their template, which is the one thing the request asked to destroy; a deployment that
        wants "never recognise this person again" needs a retained blocklist, which is not a
        deletion and must not be spelled like one.

        Pass `memories` and `embeddings` to atomically replace indexed documents that named the
        person, exactly as `register_identity` does -- the erasure commits in SQLite before the
        outbox tells the projection.

        Recoverability, stated plainly: freed cells are zero-filled (`PRAGMA secure_delete`) and
        the write-ahead log is checkpointed and truncated afterwards, so the exemplar bytes are
        no longer present in `state.sqlite3` or its `-wal`. This store runs no `VACUUM`, and
        nothing here reaches filesystem snapshots, backups, or blocks an SSD retains through
        wear levelling; full-disk encryption remains the only defence against those.
        """
        _require_identifier(identity_id, "identity_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        with self._transaction(secure_delete=True) as connection:
            members = self._identity_equivalence_class(connection, identity_id)
            if members is None:
                return None
            resolved_id = members[0]
            exemplars = {
                _identity_modality(row["modality"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT modality, COUNT(*) AS count
                    FROM identity_exemplars
                    WHERE identity_id = ?
                    GROUP BY modality
                    """,
                    (resolved_id,),
                ).fetchall()
            }
            segments = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM speech_segments WHERE speaker_id = ?",
                    (resolved_id,),
                ).fetchone()["count"]
            )
            # face_observations.identity_id is NOT NULL, so the RESTRICT that guards it cannot
            # be satisfied by anonymising in place the way speech segments are.
            observations = connection.execute(
                "DELETE FROM face_observations WHERE identity_id = ?",
                (resolved_id,),
            ).rowcount
            # Cascades the aliases, exemplars and link evidence; NULLs the speech segments.
            connection.execute("DELETE FROM identities WHERE identity_id = ?", (resolved_id,))
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        # After the commit, and best effort: a busy checkpoint leaves the zeroed pages in the
        # log rather than losing them, and the next checkpoint still applies them.
        with self._connection() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return IdentityErasure(
            identity_id=resolved_id,
            alias_ids=members[1:],
            face_exemplars=exemplars.get("face", 0),
            voice_exemplars=exemplars.get("voice", 0),
            face_observations=observations,
            speech_segments=segments,
        )

    @staticmethod
    def _identity_link_plan(
        connection: sqlite3.Connection,
        first_id: str,
        second_id: str,
        *,
        allow_shared_modality: bool = False,
    ) -> IdentityLink | None:
        first = _resolve_identity_id(connection, first_id)
        second = _resolve_identity_id(connection, second_id)
        if first is None or second is None or first == second:
            return None
        identities = (first, second)
        rows = connection.execute(
            """
            SELECT identity_id, name, created_at
            FROM identities
            WHERE identity_id IN (?, ?)
            ORDER BY identity_id
            """,
            identities,
        ).fetchall()
        if len(rows) != 2:
            return None
        modalities = {
            identity_id: {
                _identity_modality(row["modality"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT modality
                    FROM identity_exemplars
                    WHERE identity_id = ?
                    """,
                    (identity_id,),
                ).fetchall()
            }
            for identity_id in identities
        }
        if not modalities[first] or not modalities[second]:
            return None
        # A shared modality only stops blocking the plan while one side is still a
        # single-modality fragment being re-absorbed. Two identities that both hold
        # face and voice stay refused: fusing two complete people is worse than
        # leaving a fragment orphaned.
        fragment = min(len(modalities[first]), len(modalities[second])) == 1
        if modalities[first] & modalities[second] and not (allow_shared_modality and fragment):
            return None
        names = {
            _row_text(row, "identity_id"): (None if row["name"] is None else _row_text(row, "name"))
            for row in rows
        }
        if names[first] is not None and names[second] not in {None, names[first]}:
            return None
        created = {_row_text(row, "identity_id"): _row_text(row, "created_at") for row in rows}
        target, source = sorted(
            identities,
            key=lambda identity_id: (
                names[identity_id] is None,
                created[identity_id],
                identity_id,
            ),
        )
        return IdentityLink(
            target,
            source,
            names[target] or names[source],
            min(created[target], created[source]),
        )

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

    def asset_retention_candidates(
        self,
        *,
        created_before: datetime | None = None,
        limit: int = 100,
    ) -> tuple[StoredAsset, ...]:
        """List stored assets oldest first, so a retention window can be expressed.

        Media is the overwhelming majority of storage growth, so retention is a cost and
        privacy mechanism. This is the read half only: it reports what a policy could drop,
        never drops anything, and includes assets a memory still references -- `memory_assets`
        holds those under RESTRICT, so dropping one is a separate decision with its own
        contract. Pair with `asset_storage_bytes` for a size budget and
        `list_unreferenced_assets` for the already-collectable subset.
        """
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if created_before is not None:
            _require_aware(created_before, "created_before")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT asset_id, modality, mime_type, size_bytes, sha256,
                       relative_path, name, transcript, created_at
                FROM media_assets
                WHERE ? IS NULL OR created_at < ?
                ORDER BY created_at, asset_id
                LIMIT ?
                """,
                (
                    _optional_datetime_text(created_before),
                    _optional_datetime_text(created_before),
                    limit,
                ),
            ).fetchall()
        return tuple(_asset_from_row(row) for row in rows)

    def asset_storage_bytes(self) -> int:
        """Return the total size of every stored media asset descriptor."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM media_assets"
            ).fetchone()
        return int(row["total"])

    def delete_asset_if_unreferenced(self, asset_id: str) -> bool:
        """Delete one unreferenced descriptor after its CAS file has been removed."""
        _sha256(asset_id)
        with self._transaction() as connection:
            # Read before the delete cascades the observations away: these are the only
            # identities this asset can have orphaned.
            observed = tuple(
                _row_text(row, "identity_id")
                for row in connection.execute(
                    """
                    SELECT identity_id FROM face_observations WHERE asset_id = ?
                    UNION
                    SELECT speaker_id FROM speech_segments
                    WHERE asset_id = ? AND speaker_id IS NOT NULL
                    """,
                    (asset_id, asset_id),
                ).fetchall()
            )
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
            if cursor.rowcount > 0:
                _delete_unobserved_identities(connection, observed)
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
                               m.content, m.metadata_json, m.memory_type,
                               m.occurred_at, m.occurred_end
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.embedding_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        by_id: dict[str, IndexDocument] = {}
        for row in rows:
            document = _index_document_from_row(row)
            by_id[document.embedding.embedding_id] = document
        return tuple(by_id[embedding_id] for embedding_id in embedding_ids if embedding_id in by_id)

    def read_index_candidates(
        self,
        embedding_ids: Sequence[str],
    ) -> tuple[IndexCandidate, ...]:
        """Project indexed embeddings onto the columns ranking reads, preserving input order."""
        if not embedding_ids:
            return ()
        for embedding_id in embedding_ids:
            _require_identifier(embedding_id, "embedding_id")
        by_id: dict[str, IndexCandidate] = {}
        with self._connection() as connection:
            for offset in range(0, len(embedding_ids), _SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _embedding_id in batch)
                for row in connection.execute(
                    f"""
                    SELECT e.embedding_id, e.memory_id, m.occurred_at, m.occurred_end
                    FROM embeddings AS e
                    JOIN memory_records AS m ON m.memory_id = e.memory_id
                    WHERE e.embedding_id IN ({placeholders})
                    """,
                    tuple(batch),
                ):
                    candidate = IndexCandidate(
                        embedding_id=_row_text(row, "embedding_id"),
                        memory_id=_row_text(row, "memory_id"),
                        occurred_at=_optional_datetime_from_row(row, "occurred_at"),
                        occurred_end=_optional_datetime_from_row(row, "occurred_end"),
                    )
                    by_id[candidate.embedding_id] = candidate
        return tuple(by_id[embedding_id] for embedding_id in embedding_ids if embedding_id in by_id)

    def read_memory_index_documents(
        self,
        memory_ids: Sequence[str],
    ) -> tuple[IndexDocument, ...]:
        """Hydrate every embedding for each memory, preserving memory and part order."""
        if not memory_ids:
            return ()
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        rows = []
        with self._connection() as connection:
            for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT e.embedding_id, e.memory_id, e.object_part, e.model_id, e.space_id,
                               e.task, e.dimension, e.normalized, e.vector, e.created_at,
                               m.content, m.metadata_json, m.memory_type,
                               m.occurred_at, m.occurred_end
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.memory_id IN ({placeholders})
                        ORDER BY e.object_part, e.embedding_id
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        by_memory: dict[str, list[IndexDocument]] = {}
        for row in rows:
            document = _index_document_from_row(row)
            by_memory.setdefault(document.embedding.memory_id, []).append(document)
        return tuple(
            document for memory_id in memory_ids for document in by_memory.get(memory_id, ())
        )

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

    def _initialize_schema(self) -> None:  # noqa: C901 - sequential migrations stay explicit
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
            if version == 5:
                _migrate_v5(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 6:
                _migrate_v6(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 7:
                _migrate_v7(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 8:
                _migrate_v8(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 9:
                _migrate_v9(connection)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 10:
                _migrate_v10(connection)
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
    def _connection(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
        self._require_open()
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if not self._schema_ready:
                connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            if secure_delete:
                # Zero-fill freed cells instead of leaving them legible in free pages. Scoped
                # to erasure: it costs extra page writes on every DELETE, and the outbox
                # acknowledges by deleting rows on the hot path.
                connection.execute("PRAGMA secure_delete = ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connection(secure_delete=secure_delete) as connection:
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
        # The only place a caller-supplied vector reaches the authoritative table, and so the only
        # place its content is checked. See StoredEmbedding for why hydration does not repeat this.
        _require_storable_vector(embedding.values, normalized=embedding.normalized)
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

    def _replace_memory_embeddings(
        self,
        connection: sqlite3.Connection,
        memories: Sequence[StoredMemory],
        embeddings: Sequence[StoredEmbedding],
        embedding_ids: dict[str, set[str]],
    ) -> None:
        for memory in memories:
            if (
                connection.execute(
                    "SELECT 1 FROM memory_records WHERE memory_id = ?",
                    (memory.memory_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("replacement embeddings require an existing memory")
            connection.execute(
                "DELETE FROM embeddings WHERE memory_id = ?",
                (memory.memory_id,),
            )
            self._write_memory(
                connection,
                memory,
                supplied_embedding_ids=embedding_ids[memory.memory_id],
            )
        for embedding in embeddings:
            self._write_embedding(connection, embedding)

    def _write_memory(
        self,
        connection: sqlite3.Connection,
        memory: StoredMemory,
        *,
        supplied_embedding_ids: set[str],
        transaction_memory_ids: set[str] | None = None,
    ) -> bool:
        existing = connection.execute(
            """
            SELECT content, modality, memory_type, metadata_json, occurred_at, occurred_end
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
            or existing["occurred_end"] != _optional_datetime_text(memory.occurred_end)
            or existing_asset_ids != supplied_asset_ids
        )
        connection.execute(
            """
            INSERT INTO memory_records (
                memory_id, content, modality, memory_type, metadata_json,
                occurred_at, occurred_end, last_accessed_at, access_count, created_at, updated_at,
                place_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (memory_id) DO UPDATE SET
                content = excluded.content,
                modality = excluded.modality,
                memory_type = excluded.memory_type,
                metadata_json = excluded.metadata_json,
                occurred_at = excluded.occurred_at,
                occurred_end = excluded.occurred_end,
                updated_at = MAX(memory_records.updated_at, excluded.updated_at),
                place_id = excluded.place_id
            """,
            (
                memory.memory_id,
                memory.content,
                memory.modality,
                memory.memory_type,
                memory.metadata_json,
                _optional_datetime_text(memory.occurred_at),
                _optional_datetime_text(memory.occurred_end),
                _optional_datetime_text(memory.last_accessed_at),
                memory.access_count,
                _datetime_text(memory.created_at),
                _datetime_text(memory.updated_at),
                memory.place_id,
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
        if memory.context is not None:
            _write_memory_context(
                connection,
                memory.memory_id,
                memory.context,
                transaction_memory_ids=transaction_memory_ids,
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
            LEFT JOIN identities AS i ON i.identity_id = s.speaker_id
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
    def _read_faces(
        connection: sqlite3.Connection,
        asset_id: str,
    ) -> tuple[FaceObservation, ...]:
        rows = connection.execute(
            """
            SELECT f.observed_at_ms, f.box_x, f.box_y, f.box_width, f.box_height,
                   f.identity_id, i.name AS identity_name, f.identity_score
            FROM face_observations AS f
            JOIN identities AS i ON i.identity_id = f.identity_id
            WHERE f.asset_id = ?
            ORDER BY f.position
            """,
            (asset_id,),
        ).fetchall()
        return tuple(
            FaceObservation(
                asset_id=asset_id,
                observed_at_ms=(
                    None if row["observed_at_ms"] is None else int(row["observed_at_ms"])
                ),
                bounding_box=(
                    float(row["box_x"]),
                    float(row["box_y"]),
                    float(row["box_width"]),
                    float(row["box_height"]),
                ),
                identity_id=_row_text(row, "identity_id"),
                identity_name=(
                    None if row["identity_name"] is None else _row_text(row, "identity_name")
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
        preferred_identity: str | None = None,
    ) -> tuple[
        dict[str, tuple[str, float | None]],
        tuple[_IdentityChange, ...],
    ]:
        observations = {label: (vector,) for label, vector in speakers.items()}
        claim_groups = dict.fromkeys(observations, 0)
        return LocalStore._match_identities(
            connection,
            observations,
            claim_groups=claim_groups,
            modality="voice",
            model_id=model_id,
            space_id=space_id,
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
            exemplar_limit=_VOICE_EXEMPLAR_LIMIT,
            now=now,
            preferred_identity=preferred_identity,
        )

    @staticmethod
    def _match_identities(
        connection: sqlite3.Connection,
        observations: Mapping[str, tuple[tuple[float, ...], ...]],
        *,
        claim_groups: Mapping[str, int | None],
        modality: Literal["face", "voice"],
        model_id: str,
        space_id: str,
        minimum_similarity: float,
        minimum_margin: float,
        exemplar_limit: int,
        now: datetime,
        preferred_identity: str | None = None,
    ) -> tuple[dict[str, tuple[str, float | None]], tuple[_IdentityChange, ...]]:
        if not observations:
            return {}, ()
        dimension = len(next(iter(observations.values()))[0])
        rows = connection.execute(
            """
            SELECT identity_id, position, vector, created_at
            FROM identity_exemplars
            WHERE modality = ? AND space_id = ? AND dimension = ?
            ORDER BY identity_id, position
            """,
            (modality, space_id, dimension),
        ).fetchall()
        existing: dict[str, list[tuple[tuple[float, ...], str]]] = {}
        for row in rows:
            identity_id = _row_text(row, "identity_id")
            existing.setdefault(identity_id, []).append(
                (
                    _normalized_vector(
                        _unpack_vector(_row_blob(row, "vector"), dimension),
                        f"stored {modality} exemplar",
                    ),
                    _row_text(row, "created_at"),
                )
            )
        preferred_identity = (
            None
            if preferred_identity is None
            else _resolve_identity_id(connection, preferred_identity)
        )
        preferred_exists = preferred_identity is not None
        preferred_missing_modality = (
            preferred_identity is not None
            and connection.execute(
                """
                SELECT 1 FROM identity_exemplars
                WHERE identity_id = ? AND modality = ?
                LIMIT 1
                """,
                (preferred_identity, modality),
            ).fetchone()
            is None
        )
        known_identities = set(existing)
        if preferred_exists and preferred_identity is not None:
            known_identities.add(preferred_identity)
        claimed: dict[int | None, set[str]] = {}
        matches: dict[str, tuple[str, float | None]] = {}
        changes: dict[str, _IdentityChange] = {}
        now_text = _datetime_text(now)
        for label, vectors in observations.items():
            group_claims = claimed.setdefault(claim_groups[label], set())
            # ponytail: local identity populations use a linear scan; add a vector index only
            # after profiling shows identity matching matters beside model inference.
            accepted = _accepted_identity(
                existing,
                vectors,
                claimed=group_claims,
                minimum_similarity=minimum_similarity,
                minimum_margin=minimum_margin,
            )
            if accepted is not None:
                identity_id, score = accepted
            elif (
                preferred_exists
                and preferred_missing_modality
                and preferred_identity is not None
                and len(observations) == 1
                and preferred_identity not in group_claims
            ):
                identity_id, score = preferred_identity, None
            else:
                identity_id, score = f"identity_{uuid.uuid4().hex}", None
            if identity_id not in changes:
                previous = (
                    None
                    if identity_id not in known_identities
                    else LocalStore._identity_state(connection, identity_id)
                )
                changes[identity_id] = _IdentityChange(identity_id, previous)
            identity_exists = identity_id in known_identities
            existing[identity_id] = _write_identity_exemplars(
                connection,
                identity_id,
                existing.get(identity_id, ()),
                vectors,
                modality=modality,
                model_id=model_id,
                space_id=space_id,
                dimension=dimension,
                exemplar_limit=exemplar_limit,
                identity_exists=identity_exists,
                now_text=now_text,
            )
            known_identities.add(identity_id)
            matches[label] = (
                identity_id,
                None if score is None else max(0.0, min(1.0, score)),
            )
            group_claims.add(identity_id)
        return matches, tuple(changes.values())

    @staticmethod
    def _identity_state(connection: sqlite3.Connection, identity_id: str) -> _IdentityState:
        row = connection.execute(
            "SELECT name, created_at, updated_at FROM identities WHERE identity_id = ?",
            (identity_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("matched identity disappeared during recognition")
        exemplars = connection.execute(
            """
            SELECT modality, position, model_id, space_id, dimension, vector, created_at
            FROM identity_exemplars
            WHERE identity_id = ?
            ORDER BY modality, position
            """,
            (identity_id,),
        ).fetchall()
        return _IdentityState(
            identity_id=identity_id,
            name=None if row["name"] is None else _row_text(row, "name"),
            created_at=_row_text(row, "created_at"),
            updated_at=_row_text(row, "updated_at"),
            exemplars=tuple(
                _IdentityExemplarState(
                    modality=_identity_modality(exemplar["modality"]),
                    position=int(exemplar["position"]),
                    model_id=_row_text(exemplar, "model_id"),
                    space_id=_row_text(exemplar, "space_id"),
                    dimension=int(exemplar["dimension"]),
                    vector=_row_blob(exemplar, "vector"),
                    created_at=_row_text(exemplar, "created_at"),
                )
                for exemplar in exemplars
            ),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise LocalStoreClosedError("local store is closed")


def _delete_unobserved_identities(
    connection: sqlite3.Connection,
    identity_ids: Sequence[str],
) -> None:
    """Drop the anonymous identities the deleted media just left with no observation at all.

    An exemplar is a biometric template derived from stored media, so it is content: leaving one
    behind after its last face observation and speech segment are gone would make `delete()`
    incomplete. A named or merged identity is a person the caller asserted, not a by-product of
    one recording; it survives, and `forget_identity()` is what erases a person.

    Only identities this asset observed are candidates. A sweep of every unobserved identity
    would also destroy one `unlink_identity()` deliberately left anonymous with its exemplars and
    no observations, which is the state continued ingestion is supposed to corroborate again.
    """
    if not identity_ids:
        return
    placeholders = ", ".join("?" for _identity_id in identity_ids)
    connection.execute(
        f"""
        DELETE FROM identities
        WHERE identity_id IN ({placeholders})
          AND name IS NULL AND relationship IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM face_observations AS f
              WHERE f.identity_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM speech_segments AS s
              WHERE s.speaker_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM identity_aliases AS a
              WHERE a.identity_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM identity_link_evidence AS e
              WHERE e.voice_id = identities.identity_id OR e.face_id = identities.identity_id
          )
        """,
        tuple(identity_ids),
    )


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
        connection.executescript(_SCHEMA_V10)
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


def _migrate_v5(connection: sqlite3.Connection) -> None:
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(memory_records)")}
        if "occurred_end" not in columns:
            connection.executescript(_MIGRATE_V5_TO_V6)
        else:
            connection.execute("PRAGMA user_version = 6")
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v6(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(_MIGRATE_V6_TO_V7)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v7(connection: sqlite3.Connection) -> None:
    try:
        semantic_tables = {
            "formation_runs",
            "memory_evidence",
            "memory_semantics",
            "memory_versions",
        }
        existing = semantic_tables & _table_names(connection)
        if existing == semantic_tables:
            connection.execute("PRAGMA user_version = 8")
            return
        if existing:
            names = ", ".join(sorted(semantic_tables - existing))
            raise UnsupportedSchemaError(
                f"local schema has an incomplete v8 semantic projection; missing: {names}"
            )
        connection.executescript(_MIGRATE_V7_TO_V8)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v8(connection: sqlite3.Connection) -> None:
    try:
        identity_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(identities)")
        }
        alias_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(identity_aliases)")
        }
        has_evidence = "identity_link_evidence" in _table_names(connection)
        connection.execute("BEGIN IMMEDIATE")
        if "relationship" not in identity_columns:
            connection.execute(
                """
                ALTER TABLE identities
                ADD COLUMN relationship TEXT
                    CHECK (relationship IS NULL OR length(trim(relationship)) > 0)
                """
            )
        if "contributed_modality" not in alias_columns:
            connection.execute(
                """
                ALTER TABLE identity_aliases
                ADD COLUMN contributed_modality TEXT
                    CHECK (
                        contributed_modality IS NULL
                        OR contributed_modality IN ('face', 'voice')
                    )
                """
            )
        if not has_evidence:
            for statement in _IDENTITY_LINK_EVIDENCE_DDL:
                connection.execute(statement)
        connection.execute("PRAGMA user_version = 9")
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v9(connection: sqlite3.Connection) -> None:
    try:
        record_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(memory_records)")
        }
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(memory_records)")
            if row[1] is not None
        }
        connection.execute("BEGIN IMMEDIATE")
        if "place_id" not in record_columns:
            connection.execute(_PLACE_COLUMN_DDL)
        if "memory_records_place_idx" not in indexes:
            connection.execute(_PLACE_INDEX_DDL)
        connection.execute("PRAGMA user_version = 10")
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _migrate_v10(connection: sqlite3.Connection) -> None:
    """Add forgetting state and the control-plane tables.

    A store created by a v10 development build may carry either the place column or the
    forgetting column but not both, so every step is guarded rather than assumed.
    """
    try:
        record_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(memory_records)")
        }
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(memory_records)")
            if row[1] is not None
        }
        connection.execute("BEGIN IMMEDIATE")
        if "place_id" not in record_columns:
            connection.execute(_PLACE_COLUMN_DDL)
        if "memory_records_place_idx" not in indexes:
            connection.execute(_PLACE_INDEX_DDL)
        if "forgotten_at" not in record_columns:
            connection.execute("ALTER TABLE memory_records ADD COLUMN forgotten_at TEXT")
        connection.executescript(_CONTROL_SCHEMA)
        connection.execute("PRAGMA user_version = 11")
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _write_memory_context(
    connection: sqlite3.Connection,
    memory_id: str,
    context: MemoryContext,
    *,
    transaction_memory_ids: set[str] | None = None,
) -> None:
    existing = connection.execute(
        "SELECT 1 FROM memory_semantics WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if existing is not None:
        for source_memory_id in context.evidence_ids:
            _add_memory_evidence(
                connection,
                memory_id,
                source_memory_id,
                confidence=context.confidence,
                recorded_at=context.recorded_at,
            )
        return

    lineage_id = context.lineage_id or memory_id
    recorded_at = _next_lineage_transaction_time(
        connection,
        context.recorded_at,
        lineage_id=lineage_id,
        kind=context.kind.value,
        transaction_memory_ids=transaction_memory_ids,
    )
    valid_from = context.valid_from
    valid_until = context.valid_until
    spatial = context.spatial
    orientation = None if spatial is None else spatial.orientation_xyzw
    connection.execute(
        """
        INSERT INTO memory_semantics (
            memory_id, lineage_id, kind, basis, source_id,
            subject, predicate, value, model_id, recipe,
            cue_modality, valence, arousal,
            spatial_frame_id, spatial_anchor, spatial_x, spatial_y, spatial_z,
            spatial_qx, spatial_qy, spatial_qz, spatial_qw, spatial_uncertainty_m
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            memory_id,
            lineage_id,
            context.kind.value,
            context.basis.value,
            context.source_id,
            context.subject,
            context.predicate,
            context.value,
            context.model_id,
            context.recipe,
            None if context.cue_modality is None else context.cue_modality.value,
            context.valence,
            context.arousal,
            None if spatial is None else spatial.frame_id,
            None if spatial is None else spatial.anchor.value,
            None if spatial is None else spatial.x,
            None if spatial is None else spatial.y,
            None if spatial is None else spatial.z,
            None if orientation is None else orientation[0],
            None if orientation is None else orientation[1],
            None if orientation is None else orientation[2],
            None if orientation is None else orientation[3],
            None if spatial is None else spatial.position_uncertainty_m,
        ),
    )
    for source_memory_id in context.evidence_ids:
        _insert_memory_evidence(
            connection,
            memory_id,
            source_memory_id,
            confidence=context.confidence,
            recorded_at=recorded_at,
        )

    supersedes_id = context.supersedes_id
    reconcile_lineage = context.kind is MemoryKind.STATE or (
        context.kind is MemoryKind.TRAIT and context.basis is EvidenceBasis.USER_STATEMENT
    )
    if reconcile_lineage:
        old_rows = connection.execute(
            """
            SELECT
                s.memory_id, v.version, v.confidence, v.valid_from, v.valid_until,
                v.recorded_at, v.visible, v.supersedes_id
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            WHERE s.lineage_id = ? AND s.kind = ?
              AND s.memory_id <> ? AND v.retired_at IS NULL
            ORDER BY v.recorded_at, s.memory_id, v.version
            """,
            (lineage_id, context.kind.value, memory_id),
        ).fetchall()
        for old in old_rows:
            old_from = _optional_datetime_from_row(old, "valid_from")
            old_until = _optional_datetime_from_row(old, "valid_until")
            old_recorded = _parse_datetime(_row_text(old, "recorded_at"))
            # Independent assertions in one storage batch remain conflicting. Wall-clock equality
            # alone cannot identify a batch on low-resolution or frozen clocks.
            if (
                transaction_memory_ids is not None
                and _row_text(old, "memory_id") in transaction_memory_ids
            ) or not _intervals_overlap(
                valid_from,
                valid_until,
                old_from,
                old_until,
            ):
                continue
            tx_time = max(recorded_at, old_recorded + timedelta(microseconds=1))
            recorded_at = tx_time
            connection.execute(
                """
                UPDATE memory_versions
                SET retired_at = ?
                WHERE memory_id = ? AND version = ? AND retired_at IS NULL
                """,
                (
                    _datetime_text(tx_time),
                    _row_text(old, "memory_id"),
                    int(old["version"]),
                ),
            )
            if valid_from is not None and (old_from is None or old_from < valid_from):
                _carry_memory_version(
                    connection,
                    old,
                    valid_from=old_from,
                    valid_until=valid_from,
                    recorded_at=tx_time,
                )
            if valid_until is not None and (old_until is None or valid_until < old_until):
                _carry_memory_version(
                    connection,
                    old,
                    valid_from=valid_until,
                    valid_until=old_until,
                    recorded_at=tx_time,
                )
            supersedes_id = supersedes_id or _row_text(old, "memory_id")

    evidence_count, _confidence = _evidence_summary(connection, memory_id)
    visible = _semantic_visibility(
        connection,
        memory_id=memory_id,
        lineage_id=lineage_id,
        kind=context.kind.value,
        basis=context.basis.value,
        evidence_count=evidence_count,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    connection.execute(
        """
        INSERT INTO memory_versions (
            memory_id, version, confidence, valid_from, valid_until,
            recorded_at, retired_at, visible, supersedes_id
        ) VALUES (?, 1, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            memory_id,
            context.confidence,
            _optional_datetime_text(valid_from),
            _optional_datetime_text(valid_until),
            _datetime_text(recorded_at),
            int(visible),
            supersedes_id,
        ),
    )


def _intervals_overlap(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> bool:
    return not (
        (left_until is not None and right_from is not None and left_until <= right_from)
        or (right_until is not None and left_from is not None and right_until <= left_from)
    )


def _carry_memory_version(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    valid_from: datetime | None,
    valid_until: datetime | None,
    recorded_at: datetime,
) -> None:
    memory_id = _row_text(row, "memory_id")
    latest = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM memory_versions WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if latest is None:
        raise RuntimeError("failed to allocate a memory version")
    connection.execute(
        """
        INSERT INTO memory_versions (
            memory_id, version, confidence, valid_from, valid_until,
            recorded_at, retired_at, visible, supersedes_id
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            memory_id,
            int(latest["version"]) + 1,
            float(row["confidence"]),
            _optional_datetime_text(valid_from),
            _optional_datetime_text(valid_until),
            _datetime_text(recorded_at),
            int(row["visible"]),
            _optional_row_text(row, "supersedes_id"),
        ),
    )


def _next_semantic_transaction_time(
    connection: sqlite3.Connection,
    proposed: datetime,
    memory_ids: Iterable[str],
) -> datetime:
    unique_ids = tuple(dict.fromkeys(memory_ids))
    latest = proposed
    for offset in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH):
        batch = unique_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        for table in ("memory_versions", "memory_evidence"):
            row = connection.execute(
                f"""
                SELECT MAX(recorded_at) AS recorded_at, MAX(retired_at) AS retired_at
                FROM {table} WHERE memory_id IN ({placeholders})
                """,
                batch,
            ).fetchone()
            if row is None:
                continue
            for field in ("recorded_at", "retired_at"):
                value = _optional_row_text(row, field)
                if value is not None:
                    latest = max(latest, _parse_datetime(value) + timedelta(microseconds=1))
    return latest


def _next_lineage_transaction_time(
    connection: sqlite3.Connection,
    proposed: datetime,
    *,
    lineage_id: str,
    kind: str,
    transaction_memory_ids: set[str] | None,
) -> datetime:
    current_batch = transaction_memory_ids or set()
    memory_ids = tuple(
        _row_text(row, "memory_id")
        for row in connection.execute(
            "SELECT memory_id FROM memory_semantics WHERE lineage_id = ? AND kind = ?",
            (lineage_id, kind),
        ).fetchall()
        if _row_text(row, "memory_id") not in current_batch
    )
    return _next_semantic_transaction_time(connection, proposed, memory_ids)


def _insert_memory_evidence(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
    *,
    confidence: float,
    recorded_at: datetime,
) -> bool:
    if (
        connection.execute(
            """
        SELECT 1 FROM memory_evidence
        WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
        """,
            (memory_id, source_memory_id),
        ).fetchone()
        is not None
    ):
        return False
    source = connection.execute(
        """
        SELECT COALESCE(s.source_id, r.memory_id) AS source_group_id
        FROM memory_records AS r
        LEFT JOIN memory_semantics AS s
          ON s.memory_id = r.memory_id AND s.kind = 'observation'
        WHERE r.memory_id = ?
        """,
        (source_memory_id,),
    ).fetchone()
    if source is None:
        raise sqlite3.IntegrityError("evidence source memory does not exist")
    position_row = connection.execute(
        """
        SELECT COALESCE(MAX(position) + 1, 0) AS position
        FROM memory_evidence
        WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if position_row is None:
        raise RuntimeError("failed to allocate an evidence position")
    connection.execute(
        """
        INSERT INTO memory_evidence (
            memory_id, source_memory_id, source_group_id, position,
            confidence, recorded_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            memory_id,
            source_memory_id,
            _row_text(source, "source_group_id"),
            int(position_row["position"]),
            confidence,
            _datetime_text(recorded_at),
        ),
    )
    return True


def _add_memory_evidence(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
    *,
    confidence: float,
    recorded_at: datetime,
) -> bool:
    previous_count, previous_confidence = _evidence_summary(connection, memory_id)
    tx_time = _next_semantic_transaction_time(connection, recorded_at, (memory_id,))
    if not _insert_memory_evidence(
        connection,
        memory_id,
        source_memory_id,
        confidence=confidence,
        recorded_at=tx_time,
    ):
        return False
    evidence_count, combined = _evidence_summary(connection, memory_id)
    if evidence_count == previous_count and combined == previous_confidence:
        return True
    _refresh_evidence_projection(connection, memory_id, tx_time)
    return True


def _require_active(connection: sqlite3.Connection, memory_ids: Sequence[str]) -> None:
    """Fail the transaction unless every named memory still exists and is still un-forgotten."""
    for memory_id in dict.fromkeys(memory_ids):
        _require_identifier(memory_id, "memory_id")
        row = connection.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ? AND forgotten_at IS NULL",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise StaleOperationError(f"{memory_id} is no longer active")


def _require_every(changed: Sequence[str], requested: Sequence[str], effect: str) -> None:
    """Fail the transaction unless every requested target actually took the effect."""
    if set(changed) != set(requested):
        missing = sorted(set(requested) - set(changed))
        raise StaleOperationError(f"could not {effect} {', '.join(missing)}")


def _set_forgotten(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    forgotten_at: datetime | None,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id in dict.fromkeys(memory_ids):
        cursor = connection.execute(
            """
            UPDATE memory_records
            SET forgotten_at = ?
            WHERE memory_id = ? AND (forgotten_at IS NULL) != (? IS NULL)
            """,
            (
                None if forgotten_at is None else _datetime_text(forgotten_at),
                memory_id,
                None if forgotten_at is None else 1,
            ),
        )
        if cursor.rowcount:
            changed.append(memory_id)
    return tuple(changed)


def _refresh_multi_source_projections(
    connection: sqlite3.Connection,
    memories: Sequence[StoredMemory],
    *,
    changed_at: datetime,
) -> None:
    """Project confidence and visibility for records written with several sources at once.

    A record inserted with all of its evidence rows already present never reaches the per-row
    projection in `_add_memory_evidence`. Formation cites one source and needs nothing here;
    consolidation needs it for the noisy-OR confidence and trait visibility.
    """
    for memory in memories:
        context = memory.context
        if context is not None and len(context.evidence_ids) > 1:
            _refresh_evidence_projection(connection, memory.memory_id, changed_at)


def _asserted_confidence(connection: sqlite3.Connection, memory_id: str) -> float:
    """Return the confidence one source lends this assertion, not the noisy-OR projection.

    Version 1 never changes, so reinforcing the same record twice combines equal independent
    support instead of compounding whatever the last projection happened to be.
    """
    row = connection.execute(
        "SELECT confidence FROM memory_versions WHERE memory_id = ? ORDER BY version LIMIT 1",
        (memory_id,),
    ).fetchone()
    return 1.0 if row is None else float(row["confidence"])


def _retire_memory_versions(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id in dict.fromkeys(memory_ids):
        tx_time = _next_semantic_transaction_time(connection, retired_at, (memory_id,))
        cursor = connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND retired_at IS NULL
            """,
            (_datetime_text(tx_time), memory_id),
        )
        if cursor.rowcount:
            changed.append(memory_id)
    return tuple(changed)


def _restore_memory_versions(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    recorded_at: datetime,
) -> None:
    for memory_id in dict.fromkeys(memory_ids):
        row = connection.execute(
            """
            SELECT memory_id, version, confidence, valid_from, valid_until,
                   recorded_at, visible, supersedes_id
            FROM memory_versions WHERE memory_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            continue
        _carry_memory_version(
            connection,
            row,
            valid_from=_optional_datetime_from_row(row, "valid_from"),
            valid_until=_optional_datetime_from_row(row, "valid_until"),
            recorded_at=_next_semantic_transaction_time(connection, recorded_at, (memory_id,)),
        )


def _retire_memory_evidence(
    connection: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id, source_memory_id in pairs:
        tx_time = _next_semantic_transaction_time(connection, retired_at, (memory_id,))
        cursor = connection.execute(
            """
            UPDATE memory_evidence SET retired_at = ?
            WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
            """,
            (_datetime_text(tx_time), memory_id, source_memory_id),
        )
        if cursor.rowcount:
            changed.append(memory_id)
    for memory_id in dict.fromkeys(changed):
        _refresh_evidence_projection(
            connection,
            memory_id,
            _next_semantic_transaction_time(connection, retired_at, (memory_id,)),
        )
    return tuple(dict.fromkeys(changed))


def _validate_formation_links(
    sources: Sequence[str],
    forget_ids: Sequence[str],
    evidence: Sequence[tuple[str, str, float]],
) -> None:
    """Check every identifier and confidence a formation commit is about to write."""
    for source_memory_id in sources:
        _require_identifier(source_memory_id, "source_memory_id")
    for memory_id in forget_ids:
        _require_identifier(memory_id, "forget_id")
    for memory_id, source_memory_id, confidence in evidence:
        _require_identifier(memory_id, "memory_id")
        _require_identifier(source_memory_id, "source_memory_id")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")


def _consumed_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when a still-standing operation last consumed or produced it.

    The log stores IDs inside its two JSON documents rather than in columns, so this reads them
    back with `json_tree`. A rolled-back operation consumed nothing that still stands.
    """
    # ponytail: one full scan of the standing operation log per call, JSON-parsed row by row,
    # because both consumers need the map before they pick their candidate windows. Narrow it to
    # `j.value IN (...)` over those windows -- which means computing them first -- once a store's
    # log is long enough for the scan to show up next to the two window queries.
    return {
        _row_text(row, "memory_id"): _parse_datetime(_row_text(row, "consumed_at"))
        for row in connection.execute(
            """
            SELECT j.value AS memory_id, MAX(o.applied_at) AS consumed_at
            FROM memory_operations AS o
            JOIN json_tree(json_array(
                     json_extract(o.operation_json, '$.evidence_ids'),
                     json_extract(o.operation_json, '$.target_ids'),
                     json_extract(o.effects_json, '$.created_ids'),
                     json_extract(o.effects_json, '$.changed_ids')
                 )) AS j
            WHERE o.rolled_back_at IS NULL AND j.type = 'text'
            GROUP BY j.value
            """
        ).fetchall()
    }


def _evidence_candidates(
    connection: sqlite3.Connection,
    consumed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Derived records that gained independent evidence no standing operation has weighed."""
    # ponytail: over-fetch the newest window and filter, rather than push the consumed-time
    # comparison into SQL. Raise the factor only if a store whose newest records are all already
    # deliberated measurably under-fills the window.
    newest = connection.execute(
        """
        SELECT e.memory_id AS memory_id, MAX(e.recorded_at) AS newest
        FROM memory_evidence AS e
        JOIN memory_records AS r ON r.memory_id = e.memory_id
        WHERE e.retired_at IS NULL AND r.forgotten_at IS NULL
        GROUP BY e.memory_id
        ORDER BY newest DESC, e.memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    supported = {
        _row_text(row, "memory_id"): _parse_datetime(_row_text(row, "newest")) for row in newest
    }
    due = {
        memory_id: consumed.get(memory_id)
        for memory_id, latest in supported.items()
        if memory_id not in consumed or latest > consumed[memory_id]
    }
    if not due:
        return []
    placeholders = ", ".join("?" for _memory_id in due)
    fresh: dict[str, list[str]] = {memory_id: [] for memory_id in due}
    # A forgotten source is dropped here, not later: `consolidate()` reads the shown records with
    # `active_only=True`, so naming one would inflate `evidence_count` with an ID the deliberation
    # never sees. A candidate left with no remaining sources falls out of the `if sources` guard.
    for row in connection.execute(
        f"""
        SELECT e.memory_id AS memory_id, e.source_memory_id AS source_memory_id,
               e.recorded_at AS recorded_at
        FROM memory_evidence AS e
        JOIN memory_records AS s
          ON s.memory_id = e.source_memory_id AND s.forgotten_at IS NULL
        WHERE e.retired_at IS NULL AND e.memory_id IN ({placeholders})
        ORDER BY e.memory_id, e.position
        """,
        tuple(due),
    ).fetchall():
        memory_id = _row_text(row, "memory_id")
        threshold = due[memory_id]
        if threshold is None or _parse_datetime(_row_text(row, "recorded_at")) > threshold:
            fresh[memory_id].append(_row_text(row, "source_memory_id"))
    return [
        StoredCandidate(
            trigger="evidence",
            memory_ids=(memory_id, *sources),
            evidence_count=len(sources),
        )
        for memory_id, sources in fresh.items()
        if sources
    ][:limit]


def _contradiction_candidates(
    connection: sqlite3.Connection,
    limit: int,
) -> list[StoredCandidate]:
    """Lineages whose current visible claims disagree, the way the compiler reports them.

    A contradiction stays listed until the data stops contradicting: retiring one side through
    `CORRECT` is what clears it, so this needs no separate record of having been reported.
    """
    lineages = {
        _row_text(row, "lineage_id"): int(row["values_count"])
        for row in connection.execute(
            f"""
            SELECT s.lineage_id AS lineage_id, COUNT(DISTINCT s.value) AS values_count
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            JOIN memory_records AS r ON r.memory_id = s.memory_id
            WHERE v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
              AND s.value IS NOT NULL AND s.kind IN ({_CONFLICT_KIND_PLACEHOLDERS})
            GROUP BY s.lineage_id
            HAVING values_count > 1
            ORDER BY s.lineage_id
            LIMIT ?
            """,
            (*_CONFLICT_KINDS, limit),
        ).fetchall()
    }
    if not lineages:
        return []
    placeholders = ", ".join("?" for _lineage_id in lineages)
    members: dict[str, list[str]] = {lineage_id: [] for lineage_id in lineages}
    for row in connection.execute(
        f"""
        SELECT s.lineage_id AS lineage_id, s.memory_id AS memory_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
          AND s.value IS NOT NULL AND s.lineage_id IN ({placeholders})
        ORDER BY s.lineage_id, s.memory_id
        """,
        tuple(lineages),
    ).fetchall():
        members[_row_text(row, "lineage_id")].append(_row_text(row, "memory_id"))
    return [
        StoredCandidate(
            trigger="contradiction",
            memory_ids=tuple(dict.fromkeys(memory_ids)),
            evidence_count=lineages[lineage_id],
        )
        for lineage_id, memory_ids in members.items()
        if memory_ids
    ]


def _feedback_candidates(
    connection: sqlite3.Connection,
    consumed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Records whose recall was confirmed since a standing operation last saw them.

    `reinforce_memories` is one writer of `last_accessed_at`; under the default
    `reinforce_on_answer`, an `ask()` answer citing a record is the other.
    """
    rows = connection.execute(
        """
        SELECT memory_id, last_accessed_at, access_count
        FROM memory_records
        WHERE last_accessed_at IS NOT NULL AND forgotten_at IS NULL
        ORDER BY last_accessed_at DESC, memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    candidates: list[StoredCandidate] = []
    for row in rows:
        memory_id = _row_text(row, "memory_id")
        threshold = consumed.get(memory_id)
        accessed = _parse_datetime(_row_text(row, "last_accessed_at"))
        count = int(row["access_count"])
        if count > 0 and (threshold is None or accessed > threshold):
            candidates.append(
                StoredCandidate(
                    trigger="feedback",
                    memory_ids=(memory_id,),
                    evidence_count=count,
                )
            )
    return candidates[:limit]


def _active_operation_id(connection: sqlite3.Connection, operation_key: str) -> int | None:
    row = connection.execute(
        """
        SELECT operation_id FROM memory_operations
        WHERE operation_key = ? AND rolled_back_at IS NULL
        """,
        (operation_key,),
    ).fetchone()
    return None if row is None else int(row["operation_id"])


def _insert_operation(connection: sqlite3.Connection, operation: StoredOperation) -> int:
    cursor = connection.execute(
        """
        INSERT INTO memory_operations (
            operation_key, intent, trigger, model_id, recipe,
            operation_json, effects_json, applied_at, rolled_back_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            operation.operation_key,
            operation.intent,
            operation.trigger,
            operation.model_id,
            operation.recipe,
            operation.operation_json,
            json.dumps(
                {
                    "created_ids": list(operation.created_ids),
                    "changed_ids": list(operation.changed_ids),
                    "forgotten_ids": list(operation.forgotten_ids),
                    "linked": [list(pair) for pair in operation.linked],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            _datetime_text(operation.applied_at),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("failed to log a memory operation")
    return int(cursor.lastrowid)


def _operation_from_row(row: sqlite3.Row) -> StoredOperation:
    effects = json.loads(_row_text(row, "effects_json"))
    if not isinstance(effects, dict):
        raise ValueError("logged operation effects must encode an object")
    return StoredOperation(
        operation_id=int(row["operation_id"]),
        operation_key=_row_text(row, "operation_key"),
        intent=_row_text(row, "intent"),
        trigger=_row_text(row, "trigger"),
        model_id=_optional_row_text(row, "model_id"),
        recipe=_optional_row_text(row, "recipe"),
        operation_json=_row_text(row, "operation_json"),
        created_ids=tuple(effects.get("created_ids") or ()),
        changed_ids=tuple(effects.get("changed_ids") or ()),
        forgotten_ids=tuple(effects.get("forgotten_ids") or ()),
        linked=tuple((pair[0], pair[1]) for pair in effects.get("linked") or ()),
        applied_at=_parse_datetime(_row_text(row, "applied_at")),
        rolled_back_at=_optional_datetime_from_row(row, "rolled_back_at"),
    )


def _evidence_summary(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[int, float]:
    rows = connection.execute(
        """
        SELECT MAX(e.confidence) AS confidence
        FROM memory_evidence AS e
        WHERE e.memory_id = ? AND e.retired_at IS NULL
        GROUP BY e.source_group_id
        """,
        (memory_id,),
    ).fetchall()
    combined = 0.0
    for row in rows:
        combined = 1.0 - (1.0 - combined) * (1.0 - float(row["confidence"]))
    return len(rows), combined


def _semantic_visibility(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    lineage_id: str,
    kind: str,
    basis: str,
    evidence_count: int,
    valid_from: datetime | None,
    valid_until: datetime | None,
    explicit_intervals: Sequence[tuple[datetime | None, datetime | None]] | None = None,
) -> bool:
    if kind != MemoryKind.TRAIT.value or basis == EvidenceBasis.USER_STATEMENT.value:
        return True
    if evidence_count < 2:
        return False
    if explicit_intervals is None:
        rows = connection.execute(
            """
            SELECT v.valid_from, v.valid_until
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            WHERE s.lineage_id = ? AND s.kind = ? AND s.basis = ?
              AND s.memory_id <> ? AND v.retired_at IS NULL
            """,
            (
                lineage_id,
                MemoryKind.TRAIT.value,
                EvidenceBasis.USER_STATEMENT.value,
                memory_id,
            ),
        ).fetchall()
        explicit_intervals = tuple(
            (
                _optional_datetime_from_row(row, "valid_from"),
                _optional_datetime_from_row(row, "valid_until"),
            )
            for row in rows
        )
    return not any(
        _intervals_overlap(valid_from, valid_until, explicit_from, explicit_until)
        for explicit_from, explicit_until in explicit_intervals
    )


def _refresh_evidence_projection(
    connection: sqlite3.Connection,
    memory_id: str,
    changed_at: datetime,
) -> None:
    evidence_count, confidence = _evidence_summary(connection, memory_id)
    rows = connection.execute(
        """
        SELECT v.*, s.lineage_id, s.kind, s.basis
        FROM memory_versions AS v
        JOIN memory_semantics AS s ON s.memory_id = v.memory_id
        WHERE v.memory_id = ? AND v.retired_at IS NULL
        ORDER BY v.version
        """,
        (memory_id,),
    ).fetchall()
    for row in rows:
        recorded_at = _parse_datetime(_row_text(row, "recorded_at"))
        tx_time = max(changed_at, recorded_at + timedelta(microseconds=1))
        connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (_datetime_text(tx_time), memory_id, int(row["version"])),
        )
        visible = _semantic_visibility(
            connection,
            memory_id=memory_id,
            lineage_id=_row_text(row, "lineage_id"),
            kind=_row_text(row, "kind"),
            basis=_row_text(row, "basis"),
            evidence_count=evidence_count,
            valid_from=_optional_datetime_from_row(row, "valid_from"),
            valid_until=_optional_datetime_from_row(row, "valid_until"),
        )
        _carry_memory_version(
            connection,
            row,
            valid_from=_optional_datetime_from_row(row, "valid_from"),
            valid_until=_optional_datetime_from_row(row, "valid_until"),
            recorded_at=tx_time,
        )
        latest = connection.execute(
            """
            SELECT MAX(version) AS version FROM memory_versions WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        if latest is None:
            raise RuntimeError("failed to update evidence projection")
        connection.execute(
            """
            UPDATE memory_versions SET confidence = ?, visible = ?
            WHERE memory_id = ? AND version = ?
            """,
            (confidence, int(visible), memory_id, int(latest["version"])),
        )


def _rebuild_reconciled_lineage(  # noqa: C901 - replay order is the state contract
    connection: sqlite3.Connection,
    lineage_id: str,
    kind: str,
    *,
    changed_at: datetime,
) -> None:
    assertions = connection.execute(
        """
        SELECT v.*, s.kind, s.basis
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id AND v.version = 1
        WHERE s.lineage_id = ? AND s.kind = ?
          AND EXISTS (
              SELECT 1 FROM memory_evidence AS e
              WHERE e.memory_id = s.memory_id AND e.retired_at IS NULL
          )
        ORDER BY v.recorded_at, s.memory_id
        """,
        (lineage_id, kind),
    ).fetchall()
    current = connection.execute(
        """
        SELECT v.*
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        WHERE s.lineage_id = ? AND s.kind = ? AND v.retired_at IS NULL
        ORDER BY s.memory_id, v.version
        """,
        (lineage_id, kind),
    ).fetchall()
    tx_time = changed_at
    for row in current:
        tx_time = max(
            tx_time,
            _parse_datetime(_row_text(row, "recorded_at")) + timedelta(microseconds=1),
        )
    for row in current:
        connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (
                _datetime_text(tx_time),
                _row_text(row, "memory_id"),
                int(row["version"]),
            ),
        )

    # ponytail: replay is O(assertions²); add a compacted assertion ledger only if long-lived
    # lineages make deletion latency measurable.
    segments: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
    offset = 0
    while offset < len(assertions):
        recorded_at = _row_text(assertions[offset], "recorded_at")
        group: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
        while (
            offset < len(assertions) and _row_text(assertions[offset], "recorded_at") == recorded_at
        ):
            row = assertions[offset]
            group.append(
                (
                    row,
                    _optional_datetime_from_row(row, "valid_from"),
                    _optional_datetime_from_row(row, "valid_until"),
                )
            )
            offset += 1
        for cut_owner, cut_from, cut_until in group:
            if (
                kind == MemoryKind.TRAIT.value
                and _row_text(cut_owner, "basis") != EvidenceBasis.USER_STATEMENT.value
            ):
                continue
            remaining: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
            for owner, valid_from, valid_until in segments:
                if not _intervals_overlap(valid_from, valid_until, cut_from, cut_until):
                    remaining.append((owner, valid_from, valid_until))
                    continue
                if cut_from is not None and (valid_from is None or valid_from < cut_from):
                    remaining.append((owner, valid_from, cut_from))
                if cut_until is not None and (valid_until is None or cut_until < valid_until):
                    remaining.append((owner, cut_until, valid_until))
            segments = remaining
        segments.extend(group)

    explicit_intervals = tuple(
        (valid_from, valid_until)
        for row, valid_from, valid_until in segments
        if _row_text(row, "basis") == EvidenceBasis.USER_STATEMENT.value
    )
    summaries: dict[str, tuple[int, float]] = {}
    for row, valid_from, valid_until in segments:
        memory_id = _row_text(row, "memory_id")
        evidence_count, confidence = summaries.setdefault(
            memory_id,
            _evidence_summary(connection, memory_id),
        )
        visible = _semantic_visibility(
            connection,
            memory_id=memory_id,
            lineage_id=lineage_id,
            kind=kind,
            basis=_row_text(row, "basis"),
            evidence_count=evidence_count,
            valid_from=valid_from,
            valid_until=valid_until,
            explicit_intervals=explicit_intervals,
        )
        _carry_memory_version(
            connection,
            row,
            valid_from=valid_from,
            valid_until=valid_until,
            recorded_at=tx_time,
        )
        latest = connection.execute(
            "SELECT MAX(version) AS version FROM memory_versions WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if latest is None:
            raise RuntimeError("failed to rebuild a reconciled memory lineage")
        connection.execute(
            """
            UPDATE memory_versions SET confidence = ?, visible = ?
            WHERE memory_id = ? AND version = ?
            """,
            (confidence, int(visible), memory_id, int(latest["version"])),
        )


def _memory_from_row(
    row: sqlite3.Row,
    *,
    assets: tuple[StoredAsset, ...] = (),
    context: MemoryContext | None = None,
) -> StoredMemory:
    return StoredMemory(
        memory_id=_row_text(row, "memory_id"),
        content=_row_text(row, "content"),
        modality=_row_text(row, "modality"),
        memory_type=_row_text(row, "memory_type"),
        place_id=_optional_row_text(row, "place_id"),
        assets=assets,
        metadata_json=_row_text(row, "metadata_json"),
        occurred_at=_optional_datetime_from_row(row, "occurred_at"),
        occurred_end=_optional_datetime_from_row(row, "occurred_end"),
        last_accessed_at=_optional_datetime_from_row(row, "last_accessed_at"),
        access_count=int(row["access_count"]),
        created_at=_parse_datetime(_row_text(row, "created_at")),
        updated_at=_parse_datetime(_row_text(row, "updated_at")),
        forgotten_at=_optional_datetime_from_row(row, "forgotten_at"),
        context=context,
    )


def _read_memory_contexts(  # noqa: C901 - one authoritative bitemporal hydration pass
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    near: SpatialContext | None = None,
    radius_m: float | None = None,
    active_only: bool = False,
) -> tuple[dict[str, MemoryContext], frozenset[str]]:
    if not memory_ids:
        return {}, frozenset()
    if (near is None) != (radius_m is None):
        raise ValueError("near and radius_m must be supplied together")
    if near is not None and not isinstance(near, SpatialContext):
        raise ValueError("near must be a SpatialContext")
    if radius_m is not None and (
        isinstance(radius_m, bool)
        or not isinstance(radius_m, int | float)
        or not math.isfinite(float(radius_m))
        or radius_m < 0
    ):
        raise ValueError("radius_m must be a non-negative finite number")
    if valid_at is not None:
        _require_aware(valid_at, "valid_at")
    if known_at is not None:
        _require_aware(known_at, "known_at")
    query_valid_at = valid_at or datetime.now(timezone.utc)
    query_known_at = known_at or datetime.now(timezone.utc)

    rows: list[sqlite3.Row] = []
    evidence: dict[str, list[str]] = {}
    for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        rows.extend(
            connection.execute(
                f"""
                SELECT
                    s.memory_id AS semantic_memory_id,
                    s.lineage_id, s.kind, s.basis, s.source_id,
                    s.subject, s.predicate, s.value, s.model_id, s.recipe,
                    s.cue_modality, s.valence, s.arousal,
                    s.spatial_frame_id, s.spatial_anchor,
                    s.spatial_x, s.spatial_y, s.spatial_z,
                    s.spatial_qx, s.spatial_qy, s.spatial_qz, s.spatial_qw,
                    s.spatial_uncertainty_m,
                    v.version, v.confidence, v.valid_from, v.valid_until,
                    v.recorded_at, v.retired_at, v.visible, v.supersedes_id
                FROM memory_semantics AS s
                JOIN memory_versions AS v ON v.memory_id = s.memory_id
                WHERE s.memory_id IN ({placeholders})
                ORDER BY s.memory_id, v.recorded_at DESC, v.version DESC
                """,
                tuple(batch),
            ).fetchall()
        )
        evidence_time = (
            "retired_at IS NULL"
            if known_at is None
            else "recorded_at <= ? AND (retired_at IS NULL OR retired_at > ?)"
        )
        evidence_parameters: tuple[object, ...] = tuple(batch)
        if known_at is not None:
            known_text = _datetime_text(known_at)
            evidence_parameters += (known_text, known_text)
        for row in connection.execute(
            f"""
            SELECT memory_id, source_memory_id
            FROM memory_evidence
            WHERE memory_id IN ({placeholders}) AND {evidence_time}
            ORDER BY memory_id, position
            """,
            evidence_parameters,
        ).fetchall():
            evidence.setdefault(_row_text(row, "memory_id"), []).append(
                _row_text(row, "source_memory_id")
            )

    semantic_ids = frozenset(_row_text(row, "semantic_memory_id") for row in rows)
    selected: dict[str, sqlite3.Row] = {}
    for row in rows:
        memory_id = _row_text(row, "semantic_memory_id")
        if memory_id in selected:
            continue
        recorded_at = _parse_datetime(_row_text(row, "recorded_at"))
        retired_at = _optional_datetime_from_row(row, "retired_at")
        row_valid_from = _optional_datetime_from_row(row, "valid_from")
        row_valid_until = _optional_datetime_from_row(row, "valid_until")
        if active_only and (
            recorded_at > query_known_at
            or (retired_at is not None and retired_at <= query_known_at)
            or (row_valid_from is not None and row_valid_from > query_valid_at)
            or (row_valid_until is not None and row_valid_until <= query_valid_at)
            or not bool(int(row["visible"]))
        ):
            continue
        selected[memory_id] = row

    contexts: dict[str, MemoryContext] = {}
    for memory_id, row in selected.items():
        spatial: SpatialContext | None = None
        if row["spatial_frame_id"] is not None:
            orientation = None
            if row["spatial_qx"] is not None:
                orientation = (
                    float(row["spatial_qx"]),
                    float(row["spatial_qy"]),
                    float(row["spatial_qz"]),
                    float(row["spatial_qw"]),
                )
            spatial = SpatialContext(
                frame_id=_row_text(row, "spatial_frame_id"),
                anchor=SpatialAnchor(_row_text(row, "spatial_anchor")),
                x=float(row["spatial_x"]),
                y=float(row["spatial_y"]),
                z=float(row["spatial_z"]),
                orientation_xyzw=orientation,
                position_uncertainty_m=(
                    None
                    if row["spatial_uncertainty_m"] is None
                    else float(row["spatial_uncertainty_m"])
                ),
            )
        if near is not None:
            assert radius_m is not None
            if (
                spatial is None
                or spatial.frame_id != near.frame_id
                or spatial.anchor is not near.anchor
            ):
                continue
            distance = math.sqrt(
                (spatial.x - near.x) ** 2 + (spatial.y - near.y) ** 2 + (spatial.z - near.z) ** 2
            )
            tolerance = radius_m
            tolerance += spatial.position_uncertainty_m or 0.0
            tolerance += near.position_uncertainty_m or 0.0
            if distance > tolerance:
                continue
        retired_at = _optional_datetime_from_row(row, "retired_at")
        if known_at is not None and retired_at is not None and known_at < retired_at:
            retired_at = None
        contexts[memory_id] = MemoryContext(
            kind=MemoryKind(_row_text(row, "kind")),
            basis=EvidenceBasis(_row_text(row, "basis")),
            confidence=float(row["confidence"]),
            valid_from=_optional_datetime_from_row(row, "valid_from"),
            valid_until=_optional_datetime_from_row(row, "valid_until"),
            recorded_at=_parse_datetime(_row_text(row, "recorded_at")),
            visible=bool(int(row["visible"])),
            retired_at=retired_at,
            lineage_id=_row_text(row, "lineage_id"),
            source_id=_optional_row_text(row, "source_id"),
            subject=_optional_row_text(row, "subject"),
            predicate=_optional_row_text(row, "predicate"),
            value=_optional_row_text(row, "value"),
            evidence_ids=tuple(evidence.get(memory_id, ())),
            supersedes_id=_optional_row_text(row, "supersedes_id"),
            model_id=_optional_row_text(row, "model_id"),
            recipe=_optional_row_text(row, "recipe"),
            spatial=spatial,
            cue_modality=(
                None if row["cue_modality"] is None else Modality(_row_text(row, "cue_modality"))
            ),
            valence=None if row["valence"] is None else float(row["valence"]),
            arousal=None if row["arousal"] is None else float(row["arousal"]),
        )
    return contexts, semantic_ids


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


def _index_document_from_row(row: sqlite3.Row) -> IndexDocument:
    return IndexDocument(
        embedding=_embedding_from_row(row),
        content=_row_text(row, "content"),
        metadata_json=_row_text(row, "metadata_json"),
        memory_type=_row_text(row, "memory_type"),
        occurred_at=_optional_datetime_from_row(row, "occurred_at"),
        occurred_end=_optional_datetime_from_row(row, "occurred_end"),
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


def _optional_row_text(row: sqlite3.Row, column: str) -> str | None:
    return None if row[column] is None else _row_text(row, column)


def _row_blob(row: sqlite3.Row, column: str) -> bytes:
    value = row[column]
    if not isinstance(value, bytes):
        raise RuntimeError(f"stored {column} is not a BLOB")
    return value


def _resolve_identity_id(connection: sqlite3.Connection, identity_id: str) -> str | None:
    row = connection.execute(
        """
        SELECT identity_id FROM identities WHERE identity_id = ?
        UNION ALL
        SELECT identity_id FROM identity_aliases WHERE alias_id = ?
        LIMIT 1
        """,
        (identity_id, identity_id),
    ).fetchone()
    return None if row is None else _row_text(row, "identity_id")


def _sole_identity_modality(
    connection: sqlite3.Connection,
    identity_id: str,
) -> Literal["face", "voice"] | None:
    """Return the only modality an identity holds, or None when it holds several."""
    rows = connection.execute(
        "SELECT DISTINCT modality FROM identity_exemplars WHERE identity_id = ?",
        (identity_id,),
    ).fetchall()
    if len(rows) != 1:
        return None
    return _identity_modality(rows[0]["modality"])


def _merge_identity_exemplars(
    connection: sqlite3.Connection,
    target: str,
    source: str,
) -> None:
    """Move every exemplar of one identity onto another, keeping each bound intact.

    A shared-modality merge would collide on (identity_id, modality, position), so the
    source's exemplars are appended after the positions the target already holds.
    """
    for modality, limit in (("face", _FACE_EXEMPLAR_LIMIT), ("voice", _VOICE_EXEMPLAR_LIMIT)):
        offset = connection.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1
            FROM identity_exemplars
            WHERE identity_id = ? AND modality = ?
            """,
            (target, modality),
        ).fetchone()[0]
        connection.execute(
            """
            UPDATE identity_exemplars
            SET identity_id = ?, position = position + ?
            WHERE identity_id = ? AND modality = ?
            """,
            (target, int(offset), source, modality),
        )
        # ponytail: an overflowing merge keeps the target's exemplars by position rather
        # than re-running the diversity selection; the next observation of this identity
        # re-selects within the bound anyway.
        connection.execute(
            """
            DELETE FROM identity_exemplars
            WHERE identity_id = ? AND modality = ? AND position >= ?
            """,
            (target, modality, limit),
        )


def _require_storable_vector(values: tuple[float, ...], *, normalized: bool) -> None:
    """Check vector content once, at the boundary where it becomes authoritative."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding values must be finite and non-empty")
    if normalized and not math.isclose(
        math.sqrt(sum(value * value for value in values)),
        1.0,
        rel_tol=1e-4,
        abs_tol=1e-6,
    ):
        raise ValueError("normalized embedding must have unit length")


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


def _accepted_identity(
    exemplars_by_identity: dict[str, list[tuple[tuple[float, ...], str]]],
    vectors: Sequence[tuple[float, ...]],
    *,
    claimed: set[str],
    minimum_similarity: float,
    minimum_margin: float,
) -> tuple[str, float] | None:
    ranked = sorted(
        (
            (
                identity_id,
                max(
                    math.fsum(a * b for a, b in zip(vector, stored, strict=True))
                    for vector in vectors
                    for stored, _created_at in exemplars
                ),
            )
            for identity_id, exemplars in exemplars_by_identity.items()
            if identity_id not in claimed
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] < minimum_similarity:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < minimum_margin:
        return None
    return ranked[0]


def _write_identity_exemplars(
    connection: sqlite3.Connection,
    identity_id: str,
    stored: Sequence[tuple[tuple[float, ...], str]],
    observed: Sequence[tuple[float, ...]],
    *,
    modality: Literal["face", "voice"],
    model_id: str,
    space_id: str,
    dimension: int,
    exemplar_limit: int,
    identity_exists: bool,
    now_text: str,
) -> list[tuple[tuple[float, ...], str]]:
    if not identity_exists:
        connection.execute(
            "INSERT INTO identities (identity_id, created_at, updated_at) VALUES (?, ?, ?)",
            (identity_id, now_text, now_text),
        )
    exemplars = list(stored)
    for vector in observed:
        vector = _normalized_vector(
            _unpack_vector(_pack_vector(vector), dimension),
            f"{modality} exemplar",
        )
        if all(vector != existing for existing, _created_at in exemplars):
            exemplars.append((vector, now_text))
    selected = _diverse_exemplars(exemplars, limit=exemplar_limit)
    connection.execute(
        "DELETE FROM identity_exemplars WHERE identity_id = ? AND modality = ?",
        (identity_id, modality),
    )
    connection.executemany(
        """
        INSERT INTO identity_exemplars (
            identity_id, modality, position, model_id, space_id,
            dimension, vector, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                identity_id,
                modality,
                position,
                model_id,
                space_id,
                dimension,
                _pack_vector(vector),
                created_at,
            )
            for position, (vector, created_at) in enumerate(selected)
        ),
    )
    if identity_exists:
        connection.execute(
            "UPDATE identities SET updated_at = ? WHERE identity_id = ?",
            (now_text, identity_id),
        )
    return selected


def _diverse_exemplars(
    exemplars: Sequence[tuple[tuple[float, ...], str]],
    *,
    limit: int,
) -> list[tuple[tuple[float, ...], str]]:
    selected = list(exemplars)
    while len(selected) > limit:
        sums = tuple(
            math.fsum(vector[index] for vector, _created_at in selected)
            for index in range(len(selected[0][0]))
        )
        magnitude = math.sqrt(math.fsum(value * value for value in sums))
        if magnitude:
            centroid = tuple(value / magnitude for value in sums)
            remove = max(
                range(len(selected)),
                key=lambda index: (
                    math.fsum(
                        value * center
                        for value, center in zip(selected[index][0], centroid, strict=True)
                    ),
                    -index,
                ),
            )
        else:
            remove = max(
                range(len(selected)),
                key=lambda index: (
                    max(
                        math.fsum(
                            left * right
                            for left, right in zip(selected[index][0], candidate[0], strict=True)
                        )
                        for candidate_index, candidate in enumerate(selected)
                        if candidate_index != index
                    ),
                    -index,
                ),
            )
        selected.pop(remove)
    return selected


def _identity_modality(value: object) -> Literal["face", "voice"]:
    if value == "face" or value == "voice":
        return value
    raise RuntimeError(f"invalid stored identity modality: {value!r}")


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


def _require_interval(start: datetime | None, end: datetime | None) -> None:
    if start is not None:
        _require_aware(start, "occurred_at")
    if end is not None:
        _require_aware(end, "occurred_end")
        if start is None or end <= start:
            raise ValueError("occurred_end must be later than occurred_at")


def _require_identifier(value: str, name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


def _require_optional_identifier(value: str | None, name: str) -> None:
    if value is not None:
        _require_identifier(value, name)


def _identity_name(value: str, field: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"identity {field} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > 255 or not normalized.isprintable():
        raise ValueError(f"identity {field} must be at most 255 printable characters")
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
