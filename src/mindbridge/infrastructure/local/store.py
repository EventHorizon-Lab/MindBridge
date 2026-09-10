"""SQLite source of truth for the local MindBridge runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import unicodedata
import uuid
from collections.abc import (
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from itertools import groupby, zip_longest
from pathlib import Path
from threading import Lock
from typing import Literal, NoReturn

from mindbridge.infrastructure.local._lock import DataDirectoryLock
from mindbridge.models.base import FaceAnalysis, SpeechAnalysis
from mindbridge.types import (
    ConsentState,
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

_SCHEMA_VERSION = 18
_SQLITE_PARAMETER_BATCH = 900
# Idle connections retained for reuse. Concurrency above this many simultaneous readers falls
# back to the previous open-and-close behaviour rather than growing the pool without bound.
_CONNECTION_POOL_SIZE = 32
# Which identities one memory is "in": the records whose semantic subject is that identity, plus
# the records whose media carries a diarised speech segment or a face observation of them. This
# direction answers the search index projection, one document at a time. A merge re-points every
# one of these rows onto the surviving identity, so only canonical IDs ever appear here.
_IDENTITY_MEMORY_SQL = """
    SELECT memory_id AS memory_id, ms.identity_id AS identity_id
    FROM memory_semantics AS ms
    WHERE ms.identity_id IS NOT NULL AND ({predicate})
    UNION
    SELECT memory_id, ss.speaker_id
    FROM memory_assets AS ma
    JOIN speech_segments AS ss ON ss.asset_id = ma.asset_id
    WHERE ss.speaker_id IS NOT NULL AND ({predicate})
    UNION
    SELECT memory_id, fo.identity_id
    FROM memory_assets AS ma
    JOIN face_observations AS fo ON fo.asset_id = ma.asset_id
    WHERE {predicate}
"""
# The same membership asked the other way round -- which memories one identity is in -- resolved
# once from three bound copies of the canonical ID. Correlating the projection statement against
# the row being filtered instead made SQLite re-run its three-branch UNION for every candidate
# `memory_records` row, and for every embedding in the store on a merge. The two directions are
# one membership rule; `test_identity_membership_sql_agrees_in_both_directions` pins that.
_IDENTITY_MEMORIES_SQL = """
    SELECT ms.memory_id AS memory_id
    FROM memory_semantics AS ms
    WHERE ms.identity_id = ?
    UNION
    SELECT ma.memory_id
    FROM memory_assets AS ma
    JOIN speech_segments AS ss ON ss.asset_id = ma.asset_id
    WHERE ss.speaker_id = ?
    UNION
    SELECT ma.memory_id
    FROM memory_assets AS ma
    JOIN face_observations AS fo ON fo.asset_id = ma.asset_id
    WHERE fo.identity_id = ?
"""
_IDENTITY_SCOPE_CLAUSE = f"AND memory_records.memory_id IN ({_IDENTITY_MEMORIES_SQL})"
# The predicate that makes one identity-bound STATE assertion a consent statement. Shared with
# `mindbridge.memory`, which writes those assertions, so the writer and the projection can never
# disagree about which records they are.
CONSENT_PREDICATE = "consent"
# The one basis a consent statement can carry, hardcoded by `_consent_proposal`. Both consent
# reads below filter on it as well as on the predicate, so a row reaching this table by any
# other route -- a future writer, a hand-edited database -- cannot be read as a person's own
# statement. Defence in depth: `Memory._bound_identity` already reserves the predicate, so no
# model-originated claim is bound to an identity in the first place.
_CONSENT_BASIS = EvidenceBasis.USER_STATEMENT.value
# Consent states under which the kernel stops enrolling new biometric exemplars for a person.
# Recognition against exemplars already held is unaffected: erasing what is held is
# `forget_identity`, and answering a question about a photo is not new processing of a person.
_RESTRAINING_CONSENT = frozenset({ConsentState.WITHHELD.value, ConsentState.WITHDRAWN.value})
# Claims whose write contract gives one lineage one standing value. Relations are accumulating,
# as are model-inferred traits; only a person's own trait assertion has the single-value contract.
# Keep the complete SQL predicate in one place so all functional scans agree about which claims
# can participate in a conflict.
_FUNCTIONAL_CLAIM_SQL = "(s.kind = ? OR (s.kind = ? AND s.basis = ?))"
_FUNCTIONAL_CLAIM_PARAMETERS = (
    MemoryKind.STATE.value,
    MemoryKind.TRAIT.value,
    EvidenceBasis.USER_STATEMENT.value,
)
_FACE_EXEMPLAR_LIMIT = 10
_VOICE_EXEMPLAR_LIMIT = 20
_MEMORY_MODALITIES = frozenset({"text", "image", "video", "audio", "omni"})
_MEMORY_TYPES = frozenset({"semantic", "episodic", "procedural"})
_ASSET_MODALITIES = frozenset({"image", "video", "audio"})
_SHA256_HEX_LENGTH = 64
_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")
_REQUIRED_TABLES = frozenset(
    {
        "embeddings",
        "embedding_text_selectors",
        "embedding_text_span_pieces",
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
        "visual_descriptions",
        "store_metadata",
        "formation_runs",
        "memory_evidence",
        "memory_evidence_clauses",
        "memory_evidence_clause_members",
        "memory_evidence_clause_versions",
        "memory_semantics",
        "memory_versions",
        "capture_queue",
        "memory_operations",
        "memory_deliberations",
        "memory_deliberation_memories",
        "query_failures",
    }
)

_TEXT_SELECTOR_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS embedding_text_selectors (
        embedding_id TEXT NOT NULL REFERENCES embeddings (embedding_id) ON DELETE CASCADE,
        selector_position INTEGER NOT NULL CHECK (selector_position >= 0),
        parent_content_sha256 TEXT NOT NULL CHECK (
            length(parent_content_sha256) = 64
            AND parent_content_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        embedding_input_sha256 TEXT NOT NULL CHECK (
            length(embedding_input_sha256) = 64
            AND embedding_input_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        recipe_version TEXT NOT NULL CHECK (length(trim(recipe_version)) > 0),
        PRIMARY KEY (embedding_id, selector_position)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_text_span_pieces (
        embedding_id TEXT NOT NULL,
        selector_position INTEGER NOT NULL,
        piece_position INTEGER NOT NULL CHECK (piece_position >= 0),
        role TEXT NOT NULL CHECK (role IN ('context', 'body')),
        start_codepoint INTEGER NOT NULL CHECK (start_codepoint >= 0),
        end_codepoint INTEGER NOT NULL CHECK (end_codepoint > start_codepoint),
        piece_sha256 TEXT NOT NULL CHECK (
            length(piece_sha256) = 64
            AND piece_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        PRIMARY KEY (embedding_id, selector_position, piece_position),
        FOREIGN KEY (embedding_id, selector_position)
          REFERENCES embedding_text_selectors (embedding_id, selector_position) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS embedding_text_span_pieces_embedding_idx
        ON embedding_text_span_pieces (embedding_id, selector_position, piece_position)
    """,
)
_TEXT_SELECTOR_SCHEMA_SCRIPT = ";".join(_TEXT_SELECTOR_SCHEMA) + ";"
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
    identity_id TEXT REFERENCES identities (identity_id) ON DELETE SET NULL,
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
CREATE INDEX memory_semantics_identity_idx
    ON memory_semantics (identity_id, memory_id);

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

-- Clauses are authoritative provenance: one clause is an AND of its members, and several
-- clauses for a record are alternatives. `memory_evidence` remains the compatibility union.
CREATE TABLE memory_evidence_clauses (
    memory_id TEXT NOT NULL REFERENCES memory_semantics (memory_id) ON DELETE CASCADE,
    clause_id TEXT NOT NULL CHECK (length(clause_id) = 64 AND clause_id NOT GLOB '*[^0-9a-f]*'),
    member_count INTEGER NOT NULL CHECK (member_count > 0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    recorded_at TEXT NOT NULL,
    retired_at TEXT CHECK (retired_at IS NULL OR retired_at > recorded_at),
    PRIMARY KEY (memory_id, clause_id)
);

CREATE INDEX memory_evidence_clauses_current_idx
    ON memory_evidence_clauses (memory_id, retired_at, recorded_at);

CREATE TABLE memory_evidence_clause_members (
    memory_id TEXT NOT NULL,
    clause_id TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (memory_id, clause_id, source_memory_id),
    UNIQUE (memory_id, clause_id, position),
    FOREIGN KEY (memory_id, clause_id)
      REFERENCES memory_evidence_clauses (memory_id, clause_id) ON DELETE CASCADE
);

CREATE INDEX memory_evidence_clause_members_source_idx
    ON memory_evidence_clause_members (source_memory_id, memory_id, clause_id);

CREATE TABLE memory_evidence_clause_versions (
    memory_id TEXT NOT NULL,
    clause_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    recorded_at TEXT NOT NULL,
    retired_at TEXT CHECK (retired_at IS NULL OR retired_at > recorded_at),
    restores_version INTEGER CHECK (restores_version IS NULL OR restores_version > 0),
    PRIMARY KEY (memory_id, clause_id, version),
    FOREIGN KEY (memory_id, clause_id)
      REFERENCES memory_evidence_clauses (memory_id, clause_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX memory_evidence_clause_versions_current_idx
    ON memory_evidence_clause_versions (memory_id, clause_id)
    WHERE retired_at IS NULL;

CREATE TABLE formation_runs (
    source_memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    recipe TEXT NOT NULL CHECK (length(trim(recipe)) > 0),
    completed_at TEXT NOT NULL,
    PRIMARY KEY (source_memory_id, recipe)
);
"""

# One statement per entry, joined below. Kept as a tuple so a caller that must stay inside its
# own transaction can execute them one at a time: `executescript` issues an implicit COMMIT first.
_CONTROL_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS capture_queue (
        memory_id TEXT PRIMARY KEY REFERENCES memory_records (memory_id) ON DELETE CASCADE,
        enqueued_at TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        last_error TEXT CHECK (last_error IS NULL OR length(trim(last_error)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_operations (
        operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation_key TEXT NOT NULL CHECK (length(trim(operation_key)) > 0),
        intent TEXT NOT NULL CHECK (
            intent IN (
                'reinforce', 'consolidate', 'correct', 'forget', 'identify', 'merge', 'consent'
            )
        ),
        trigger TEXT NOT NULL CHECK (length(trim(trigger)) > 0),
        model_id TEXT,
        recipe TEXT,
        operation_json TEXT NOT NULL,
        effects_json TEXT NOT NULL,
        applied_at TEXT NOT NULL,
        rolled_back_at TEXT CHECK (rolled_back_at IS NULL OR rolled_back_at >= applied_at),
        -- Post-hoc judgement, written only by `record_outcome`. Nullable because "nobody has
        -- judged this yet" is the normal state and is not the same as unrefuted.
        outcome TEXT CHECK (outcome IS NULL OR outcome IN ('confirmed', 'refuted')),
        outcome_note TEXT CHECK (outcome_note IS NULL OR length(trim(outcome_note)) > 0),
        CHECK (outcome IS NOT NULL OR outcome_note IS NULL)
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS memory_operations_active_key_idx
        ON memory_operations (operation_key)
        WHERE rolled_back_at IS NULL
    """,
)
_CONTROL_SCHEMA_SCRIPT = ";".join(_CONTROL_SCHEMA) + ";"

# The scheduler's own durable state, added in v13. Two facts the store could not answer before:
# "has anything already weighed this?" and "which recalls came back empty?".
#
# `memory_deliberations` is the already-weighed marker. Without it a candidate whose deliberation
# yielded nothing -- the backend proposed nothing, or the kernel refused everything -- left no
# trace and came back every round, paying the model each time. A row is written per pass, zero
# yield included, and candidate derivation excludes a candidate weighed since its own newest
# signal.
#
# `query_failures` is the QUERY_FAILURE signal: a recall that returned nothing. It holds the
# owner's own queries, which is their own memory domain, and is bounded on write rather than
# growing without limit.
_SCHEDULER_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_deliberations (
    deliberation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL CHECK (length(trim(trigger)) > 0),
    weighed_at TEXT NOT NULL,
    proposed INTEGER NOT NULL CHECK (proposed >= 0),
    applied INTEGER NOT NULL CHECK (applied >= 0),
    rejected INTEGER NOT NULL CHECK (rejected >= 0)
);

CREATE TABLE IF NOT EXISTS memory_deliberation_memories (
    deliberation_id INTEGER NOT NULL
        REFERENCES memory_deliberations (deliberation_id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_records (memory_id) ON DELETE CASCADE,
    PRIMARY KEY (deliberation_id, memory_id)
);

CREATE INDEX IF NOT EXISTS memory_deliberation_memories_memory_idx
    ON memory_deliberation_memories (memory_id, deliberation_id);

CREATE TABLE IF NOT EXISTS query_failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL CHECK (length(trim(query)) > 0),
    normalized TEXT NOT NULL CHECK (length(normalized) > 0),
    failed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS query_failures_normalized_idx
    ON query_failures (normalized, failed_at);
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

# `_feedback_candidates` polls the most recently recalled records on every
# `consolidation_candidates()`; without this it is a full scan plus a temp sort of the whole
# store. Partial and in the query's own order, so a store that never reinforces carries no pages.
_ACCESSED_INDEX_DDL = """
CREATE INDEX memory_records_accessed_idx
    ON memory_records (last_accessed_at DESC, memory_id)
    WHERE last_accessed_at IS NOT NULL
"""

# A derived caption is a paid model call whose output is indexed text, so it is cached per asset
# *content* -- `media_assets` enforces `asset_id = sha256`, so this key is the asset's own digest
# and a second memory over the same bytes reuses the row. `space_id` is the describer's
# `vision_space`, which covers the prompt as well as the model, and it is part of the primary key
# rather than a plain column (as on `speech_analyses`) so two describers can hold their own caption
# for one asset instead of the newer recipe evicting the older one.
_VISUAL_DESCRIPTION_DDL = """
CREATE TABLE visual_descriptions (
    asset_id TEXT NOT NULL REFERENCES media_assets (asset_id) ON DELETE CASCADE,
    space_id TEXT NOT NULL CHECK (length(trim(space_id)) > 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    description TEXT NOT NULL CHECK (length(trim(description)) > 0),
    created_at TEXT NOT NULL,
    PRIMARY KEY (asset_id, space_id)
)
"""

_SCHEMA_CURRENT = f"""
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
{_ACCESSED_INDEX_DDL};

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

{_TEXT_SELECTOR_SCHEMA_SCRIPT}

{_ASSET_SCHEMA}
{_IDENTITY_SCHEMA}
{_SEMANTIC_SCHEMA}
{_CONTROL_SCHEMA_SCRIPT}

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

{_SCHEDULER_SCHEMA}

{_INDEX_TRIGGERS}

{_VISUAL_DESCRIPTION_DDL};

PRAGMA user_version = {_SCHEMA_VERSION};
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
class StoredTextSpanPiece:
    """One exact, digest-bound source range used to build a text embedding key."""

    role: Literal["context", "body"]
    start_codepoint: int
    end_codepoint: int
    piece_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"context", "body"}:
            raise ValueError("text span role must be context or body")
        if self.start_codepoint < 0 or self.end_codepoint <= self.start_codepoint:
            raise ValueError("text span offsets must be a non-empty half-open interval")
        _sha256(self.piece_sha256)


@dataclass(frozen=True, slots=True)
class StoredTextSelector:
    """One exact source-piece recipe for an embedding input."""

    parent_content_sha256: str
    embedding_input_sha256: str
    recipe_version: str
    pieces: tuple[StoredTextSpanPiece, ...]

    def __post_init__(self) -> None:
        _sha256(self.parent_content_sha256)
        _sha256(self.embedding_input_sha256)
        _require_identifier(self.recipe_version, "text selector recipe version")
        if not self.pieces:
            raise ValueError("a text selector must contain at least one source piece")
        if any(not isinstance(piece, StoredTextSpanPiece) for piece in self.pieces):
            raise ValueError("text selector pieces must be StoredTextSpanPiece values")
        positions = tuple(
            (piece.start_codepoint, piece.end_codepoint, piece.role) for piece in self.pieces
        )
        if len(positions) != len(set(positions)):
            raise ValueError("text selector pieces must be unique")
        roles = tuple(piece.role for piece in self.pieces)
        if roles not in {("body",), ("context", "body")}:
            raise ValueError("text selector pieces must be one body or one context then one body")


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
    text_selectors: tuple[StoredTextSelector, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.embedding_id, "embedding_id")
        _require_identifier(self.memory_id, "memory_id")
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.space_id, "space_id")
        _require_identifier(self.task, "task")
        _require_aware(self.created_at, "created_at")
        if self.object_part < 0:
            raise ValueError("object_part must not be negative")
        if any(not isinstance(value, StoredTextSelector) for value in self.text_selectors):
            raise ValueError("text_selectors must contain StoredTextSelector values")


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredEvidenceClauseChange:
    """One reversible clause mutation captured in an operation's durable effects."""

    memory_id: str
    clause_id: str
    previous_active: bool
    previous_confidence: float | None
    applied_confidence: float
    applied_recorded_at: datetime
    applied_version: int

    def __post_init__(self) -> None:
        _require_identifier(self.memory_id, "memory_id")
        _sha256(self.clause_id)
        for confidence in (self.previous_confidence, self.applied_confidence):
            if confidence is not None and (
                not math.isfinite(confidence) or not 0 <= confidence <= 1
            ):
                raise ValueError("clause confidence must be between zero and one")
        _require_aware(self.applied_recorded_at, "applied_recorded_at")
        if self.applied_version <= 0:
            raise ValueError("applied clause version must be positive")


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
    # Existing deterministic records whose retired version this operation restated. Rollback
    # retires this operation's new version without pretending the physical record was created.
    activated_ids: tuple[str, ...] = ()
    # Records this operation moved out of ordinary recall, whether as a FORGET intent or as the
    # consolidation forgetting a CONSOLIDATE carried. Rollback clears exactly these.
    forgotten_ids: tuple[str, ...] = ()
    # Evidence rows this operation actually inserted. Rollback retires exactly these, so a link
    # that predated the operation survives it.
    linked: tuple[tuple[str, str], ...] = ()
    # Complete support clauses this operation inserted or reactivated. Unlike `linked`, this
    # preserves a conjunction for rollback rather than treating each member as an alternative.
    # Kept for logs written by the incomplete schema-17 development build; new rows use the
    # before/after snapshots below.
    linked_clauses: tuple[tuple[str, str], ...] = ()
    clause_changes: tuple[StoredEvidenceClauseChange, ...] = ()
    # `(memory_id, version)` pairs this operation's own lineage rule superseded: the current
    # versions of the other records in the derived record's lineage whose validity it overlapped.
    # Rollback restores exactly these, so a supersession the backend never named is reversible.
    superseded: tuple[tuple[str, int], ...] = ()
    rolled_back_at: datetime | None = None
    # Post-hoc judgement of this operation, written only by `record_operation_outcome`.
    outcome: str | None = None
    outcome_note: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - validates durable operation effects
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
        for name in ("created_ids", "changed_ids", "activated_ids", "forgotten_ids"):
            for memory_id in getattr(self, name):
                _require_identifier(memory_id, name)
        for memory_id, source_memory_id in self.linked:
            _require_identifier(memory_id, "memory_id")
            _require_identifier(source_memory_id, "source_memory_id")
        for memory_id, clause_id in self.linked_clauses:
            _require_identifier(memory_id, "memory_id")
            _sha256(clause_id)
        if not all(
            isinstance(change, StoredEvidenceClauseChange) for change in self.clause_changes
        ):
            raise ValueError("clause_changes must contain StoredEvidenceClauseChange values")
        for memory_id, version in self.superseded:
            _require_identifier(memory_id, "memory_id")
            if version <= 0:
                raise ValueError("superseded version must be positive")
        _require_optional_identifier(self.outcome, "outcome")
        if self.outcome is None and self.outcome_note is not None:
            raise ValueError("an outcome note requires an outcome")


@dataclass(frozen=True, slots=True)
class StoredQueryFailure:
    """One group of near-equal recalls that came back empty, newest failure last."""

    query: str
    normalized: str
    failures: int
    failed_at: datetime

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.normalized:
            raise ValueError("a query failure must carry its query text")
        if self.failures <= 0:
            raise ValueError("failures must be positive")
        _require_aware(self.failed_at, "failed_at")


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
    place_id: str | None = None
    # Every identity this memory is about: its semantic subject, plus everyone observed in its
    # media. Projected so the search index can filter on it; SQLite stays authoritative.
    identity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_type not in _MEMORY_TYPES:
            raise ValueError("memory_type must be semantic, episodic, or procedural")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_optional_identifier(self.place_id, "place_id")
        for identity_id in self.identity_ids:
            _require_identifier(identity_id, "identity_id")


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
        # Idle connections only: one is in the pool exactly while nobody holds it.
        self._pool: list[sqlite3.Connection] = []
        self._pool_lock = Lock()
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
        """Release the pooled connections and the directory; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        self._close_pool()
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
            _reproject_named_identities(
                connection,
                tuple(memory.memory_id for memory in supplied_memories),
            )
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
        """Remove captures from the queue after every deferred stage succeeded.

        Retention also uses it, to abandon captures whose repeated failures have aged out: the
        queue row is the promise to keep retrying, and that is all it drops.
        """
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
                            limit,
                        ),
                    ).fetchall()
                )
        # Each chunk is only oldest-first within itself, so the chunks are merged rather than
        # concatenated: stopping at the first `limit` rows would starve ids past the 900th
        # forever, however long they had been queued.
        queued.sort(key=lambda row: (row.enqueued_at, row.memory_id))
        return tuple(queued[:limit])

    def read_consolidation_candidates(
        self,
        *,
        limit: int,
        idle: bool = False,
        record_budget: int | None = None,
    ) -> tuple[StoredCandidate, ...]:
        """Derive due deliberation work from evidence, lineage, and feedback already recorded.

        There is no queue and no timer behind this: every row is a fact the store already holds,
        read back as a question. Rows are interleaved across triggers so a busy trigger cannot
        starve the others out of the window.

        `record_budget` is the configured ceiling on active records; exceeding it derives
        `PRESSURE` rows. `idle` is the operator declaring an approved window, never the store
        guessing from a clock, and derives `IDLE` rows for lineages nothing has ever weighed.
        """
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if record_budget is not None and record_budget <= 0:
            raise ValueError("record_budget must be positive")
        with self._read_transaction() as connection:
            deliberated = _deliberated_at_by_memory(connection)
            weighed = _merge_weighed(_consumed_at_by_memory(connection), deliberated)
            groups = [
                _evidence_candidates(connection, weighed, limit),
                # Deliberations only, not the whole weighed map: the two consolidations that
                # created a pair of disagreeing claims weighed their own sources, not the
                # disagreement between the results, so a fresh contradiction is due even though
                # an operation touched both records a moment ago.
                _contradiction_candidates(connection, deliberated, limit),
                _feedback_candidates(connection, weighed, limit),
                _pressure_candidates(connection, weighed, limit, record_budget),
            ]
            if idle:
                groups.append(_idle_candidates(connection, weighed, limit))
        interleaved = [
            candidate for row in zip_longest(*groups) for candidate in row if candidate is not None
        ]
        return tuple(interleaved[:limit])

    def read_weighed_at(self, memory_ids: Sequence[str]) -> dict[str, datetime]:
        """Return, per named memory, when a standing operation or deliberation last weighed it."""
        # ponytail: builds the whole weighed map and then filters, because the two contributing
        # queries already aggregate over their whole tables. Narrow both to `IN (...)` over the
        # named IDs once a store's log is long enough for the scan to matter.
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        if not memory_ids:
            return {}
        with self._read_transaction() as connection:
            weighed = _weighed_at_by_memory(connection)
        return {memory_id: weighed[memory_id] for memory_id in memory_ids if memory_id in weighed}

    def record_query_failure(
        self,
        query: str,
        normalized: str,
        *,
        failed_at: datetime,
        keep: int,
    ) -> None:
        """Append one empty-recall signal, keeping at most `keep` of them.

        Bounded on write rather than by a sweeper: the table is a signal buffer, so the oldest
        rows falling out is the retention policy, not a loss.
        """
        if not query.strip() or not normalized:
            raise ValueError("a query failure must carry its query text")
        _require_aware(failed_at, "failed_at")
        if keep <= 0:
            raise ValueError("keep must be positive")
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO query_failures (query, normalized, failed_at) VALUES (?, ?, ?)",
                (query, normalized, _datetime_text(failed_at)),
            )
            connection.execute(
                """
                DELETE FROM query_failures WHERE failure_id NOT IN (
                    SELECT failure_id FROM query_failures ORDER BY failure_id DESC LIMIT ?
                )
                """,
                (keep,),
            )

    def read_repeated_query_failures(
        self,
        *,
        limit: int,
        since: datetime,
        minimum: int,
    ) -> tuple[StoredQueryFailure, ...]:
        """Return near-equal queries that failed at least `minimum` times since `since`."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        _require_aware(since, "since")
        if minimum < 2:
            raise ValueError("a repeated failure needs at least two failures")
        with self._read_transaction() as connection:
            rows = connection.execute(
                """
                SELECT normalized, COUNT(*) AS failures, MAX(failed_at) AS failed_at,
                       MAX(query) AS query
                FROM query_failures
                WHERE failed_at >= ?
                GROUP BY normalized
                HAVING failures >= ?
                ORDER BY failed_at DESC, normalized
                LIMIT ?
                """,
                (_datetime_text(since), minimum, limit),
            ).fetchall()
        return tuple(
            StoredQueryFailure(
                query=_row_text(row, "query"),
                normalized=_row_text(row, "normalized"),
                failures=int(row["failures"]),
                failed_at=_parse_datetime(_row_text(row, "failed_at")),
            )
            for row in rows
        )

    def record_deliberation(
        self,
        trigger: str,
        memory_ids: Sequence[str],
        *,
        weighed_at: datetime,
        proposed: int,
        applied: int,
        rejected: int,
    ) -> int:
        """Mark one evidence set weighed, whatever the pass yielded.

        A zero-yield pass records the same row as a productive one. That is the whole point: the
        candidate that produced nothing must stop coming back until its own signal moves.
        """
        _require_identifier(trigger, "trigger")
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        _require_aware(weighed_at, "weighed_at")
        if min(proposed, applied, rejected) < 0:
            raise ValueError("deliberation counts must not be negative")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_deliberations (
                    trigger, weighed_at, proposed, applied, rejected
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (trigger, _datetime_text(weighed_at), proposed, applied, rejected),
            )
            deliberation_id = int(cursor.lastrowid or 0)
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_deliberation_memories (deliberation_id, memory_id)
                VALUES (?, ?)
                """,
                tuple((deliberation_id, memory_id) for memory_id in dict.fromkeys(memory_ids)),
            )
        return deliberation_id

    def record_operation_outcome(
        self,
        operation_id: int,
        *,
        outcome: str,
        note: str | None,
    ) -> bool:
        """Record what later evidence said about one logged operation.

        Reports `False` for an unknown operation. The kernel never reads this back into a
        decision: it exists so consolidation precision, false retirement, and contradiction
        recovery are derivable from the log.
        """
        _require_identifier(outcome, "outcome")
        if note is not None and not note.strip():
            raise ValueError("an outcome note must not be blank")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_operations SET outcome = ?, outcome_note = ? WHERE operation_id = ?",
                (outcome, note, operation_id),
            )
        return cursor.rowcount > 0

    def apply_formation(  # noqa: C901 - one atomic formation transaction
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding],
        *,
        evidence: Sequence[tuple[str, str, float]],
        evidence_clauses: Sequence[tuple[str, tuple[str, ...], float]] = (),
        source_memory_ids: Sequence[str],
        narrowed: Sequence[tuple[str, str | None, str]] = (),
        recipe: str,
        completed_at: datetime,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
        projection_identity_id: str | None = None,
        projection_memories: Iterable[StoredMemory] = (),
        projection_embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Commit derived records, evidence, and source completion in one transaction.

        `operation` logs the control-plane operation that produced these records in the same
        transaction. Consolidation passes no `source_memory_ids`, so no `formation_runs` marker
        is written and the derived record stays open to further independent evidence.
        `forget_ids` is consolidation forgetting: sources the derived record replaces in ordinary
        recall, retired in this same commit and cleared again by rolling the operation back.
        `narrowed` is `(memory_id, place_id, metadata_json)` for records already written whose
        inherited columns no longer hold for all their evidence. Only those two columns change,
        and neither reaches the search index, so nothing is queued for re-indexing.
        `require_active` names memories that must still exist and still be un-forgotten inside
        this transaction; if any moved since the caller validated it, nothing is written and
        `StaleOperationError` is raised. `require_unretired` names memories whose current version
        must additionally still stand, which is what a cited derived source needs and what
        `require_active` deliberately does not check.

        The lineage rule the kernel applies to a `STATE` or user-stated `TRAIT` supersedes the
        current version of every other record in the same lineage with overlapping validity,
        including records nobody showed the backend. Those `(memory_id, version)` pairs are
        recorded on the operation row as `superseded` so `rollback_operation` can restore exactly
        them.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        projected_memories, projected_embeddings, projected_by_memory = _prepare_write_batch(
            projection_memories,
            projection_embeddings,
        )
        if projection_identity_id is not None:
            _require_identifier(projection_identity_id, "projection_identity_id")
        elif projected_memories:
            raise ValueError("projection memories require projection_identity_id")
        sources = tuple(dict.fromkeys(source_memory_ids))
        _require_identifier(recipe, "recipe")
        _require_aware(completed_at, "completed_at")
        _validate_formation_links(sources, forget_ids, evidence)
        for memory_id, members, confidence in evidence_clauses:
            _require_identifier(memory_id, "memory_id")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("evidence clause confidence must be between zero and one")
            if not members or len(set(members)) != len(members):
                raise ValueError("evidence clause members must be non-empty and unique")
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
            _require_unretired(connection, require_unretired)
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
            # What this operation has to roll back is measured before anything is written: a
            # restated record the store already holds attaches its own cited evidence on the way
            # through `_write_memory`, so the link's absence beforehand, not who inserted it,
            # says whether this operation created it.
            already_linked = {
                (memory_id, source_memory_id)
                for memory_id, source_memory_id, _confidence in evidence
                if _memory_evidence_linked(connection, memory_id, source_memory_id)
            }
            activated = tuple(
                memory.memory_id
                for memory in supplied_memories
                if memory.context is not None and _version_retired(connection, memory.memory_id)
            )
            transaction_memory_ids: set[str] = set()
            superseded: list[tuple[str, int]] = []
            clause_memory_ids = {memory_id for memory_id, _members, _confidence in evidence_clauses}
            for memory in supplied_memories:
                has_explicit_clause = memory.memory_id in clause_memory_ids
                self._write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                    superseded=superseded,
                    write_context_evidence=not has_explicit_clause,
                    context_recorded_at=completed_at if has_explicit_clause else None,
                )
                transaction_memory_ids.add(memory.memory_id)
            connection.executemany(
                """
                UPDATE memory_records
                SET place_id = ?,
                    metadata_json = ?,
                    updated_at = MAX(updated_at, ?)
                WHERE memory_id = ?
                """,
                (
                    (
                        place_id,
                        _canonical_object_json(metadata_json),
                        _datetime_text(completed_at),
                        memory_id,
                    )
                    for memory_id, place_id, metadata_json in narrowed
                ),
            )
            for embedding in supplied_embeddings:
                self._write_embedding(connection, embedding)
            linked: list[tuple[str, str]] = []
            clause_changes: list[StoredEvidenceClauseChange] = []
            for memory_id, members, confidence in evidence_clauses:
                change = _add_evidence_clause(
                    connection,
                    memory_id,
                    members,
                    confidence=confidence,
                    recorded_at=completed_at,
                )
                if change is not None:
                    clause_changes.append(change)
            for memory_id, source_memory_id, confidence in evidence:
                _add_memory_evidence(
                    connection,
                    memory_id,
                    source_memory_id,
                    confidence=confidence,
                    recorded_at=completed_at,
                )
                pair = (memory_id, source_memory_id)
                if pair not in already_linked:
                    already_linked.add(pair)
                    linked.append(pair)
            _refresh_multi_source_projections(
                connection,
                supplied_memories,
                changed_at=completed_at,
            )
            # Forgetting comes first: the projection has to be computed against the records that
            # remain, so consolidation forgetting a naming assertion stops projecting its name.
            forgotten: tuple[str, ...] = ()
            if operation is not None:
                forgotten = _set_forgotten(connection, forget_ids, forgotten_at=completed_at)
            _reproject_named_identities(
                connection,
                tuple(memory.memory_id for memory in supplied_memories)
                + tuple(memory_id for memory_id, _source, _confidence in evidence)
                + tuple(forget_ids),
            )
            if projection_identity_id is not None:
                if _resolve_identity_id(connection, projection_identity_id) is None:
                    raise StaleOperationError(
                        f"{projection_identity_id} is no longer an active identity"
                    )
                self._replace_memory_embeddings(
                    connection,
                    projected_memories,
                    projected_embeddings,
                    projected_by_memory,
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
                _insert_operation(
                    connection,
                    replace(
                        operation,
                        activated_ids=activated,
                        forgotten_ids=forgotten,
                        linked=tuple(linked),
                        clause_changes=tuple(clause_changes),
                        superseded=tuple(dict.fromkeys(superseded)),
                    ),
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
            added = _add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=float(confidence),
                recorded_at=recorded_at,
            )
            # Independent evidence is what makes an inferred naming assertion visible, so it is
            # also what can move the projection.
            _reproject_named_identities(connection, (memory_id,))
            return added is not None

    def apply_control_operation(
        self,
        operation: StoredOperation,
        *,
        reinforce: Sequence[tuple[str, str]] = (),
        correct_ids: Sequence[str] = (),
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
    ) -> StoredOperation | None:
        """Apply one already-validated operation and its log row in one transaction.

        The caller owns policy; this only executes the supplied effects. It returns `None` when
        the operation key is already applied and not rolled back.

        `require_active` names memories the caller validated and that must still exist and still
        be un-forgotten here. That check and the all-or-nothing effect check below are what make
        the gap between validation and this transaction safe: a target or source that moved in
        between raises `StaleOperationError` with nothing written, so the log never claims an
        effect that did not happen. `require_unretired` is the stricter half of that check, for
        the callers that need the current version of a record to still stand -- a `REINFORCE`
        target must not have been corrected in between -- which `require_active` must not check
        globally, because the host's own `forget()` may forget an already-corrected record.
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
            _require_unretired(connection, require_unretired)
            changed: list[str] = []
            linked: list[tuple[str, str]] = []
            clause_changes: list[StoredEvidenceClauseChange] = []
            for memory_id, source_memory_id in reinforce:
                change = _add_evidence_clause(
                    connection,
                    memory_id,
                    (source_memory_id,),
                    confidence=_asserted_confidence(connection, memory_id),
                    recorded_at=operation.applied_at,
                )
                if change is None:
                    raise StaleOperationError(f"{source_memory_id} already supports {memory_id}")
                changed.append(memory_id)
                linked.append((memory_id, source_memory_id))
                clause_changes.append(change)
            retired = _retire_memory_versions(
                connection, correct_ids, retired_at=operation.applied_at
            )
            _require_every(retired, correct_ids, "retire")
            changed.extend(retired)
            forgotten = _set_forgotten(connection, forget_ids, forgotten_at=operation.applied_at)
            _require_every(forgotten, forget_ids, "forget")
            changed.extend(forgotten)
            _reproject_named_identities(connection, (*changed, *correct_ids, *forget_ids))
            applied = replace(
                operation,
                changed_ids=tuple(dict.fromkeys(changed)),
                forgotten_ids=forgotten,
                linked=tuple(linked),
                clause_changes=tuple(clause_changes),
            )
            return replace(applied, operation_id=_insert_operation(connection, applied))

    def rollback_operation(
        self,
        operation_id: int,
        *,
        rolled_back_at: datetime,
        retire_evidence: Sequence[tuple[str, str]] = (),
        retire_clauses: Sequence[tuple[str, str]] = (),
        reverse_clause_changes: Sequence[StoredEvidenceClauseChange] = (),
        restore_versions: Sequence[str | tuple[str, int]] = (),
        retire_versions: Sequence[str] = (),
        clear_forgotten: Sequence[str] = (),
        delete_memory_ids: Sequence[str] = (),
        require_in_force: Sequence[str] = (),
        split_identity: str | None = None,
        merge_identities: tuple[str, str] | None = None,
        require_no_later_dependencies: Sequence[str] = (),
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Apply the caller's reversal and mark one operation rolled back, atomically.

        Returns `(False, ())` when the operation is unknown, already rolled back, or no longer
        reversible, in which case no reversal is applied. Otherwise the second value lists the
        assets that the deleted records were the last to reference; index cleanup follows through
        the durable outbox.

        `retire_versions` names records whose current version this reversal retires without
        deleting the record, which is what retracting a naming assertion is: the record and its
        log row stay readable while `restore_versions` brings back what it displaced.

        `require_in_force` names the records this operation put in force. If a later standing
        operation has since superseded one of them, reversing this one would restore a version
        beside a newer one and delete a record that newer one supersedes, so the reversal is
        refused instead: operations on one lineage reverse newest first.

        `split_identity` reverses a logged cross-modal merge by restoring that alias as its own
        identity, and `merge_identities` reverses a logged split by re-merging `(survivor,
        restored)` back under the survivor. Both are refused -- `(False, ())`, nothing written --
        when the identity graph no longer admits the reversal, which is how identity operations
        on one person also reverse newest first: a later split has already removed the alias a
        merge would need, and an erasure has removed both.

        `require_no_later_dependencies` names deterministic outputs this operation introduced or
        restated. A later standing log row that cites or changes one makes this operation no
        longer independently reversible.
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
            if _later_operation_depends_on(
                connection,
                operation_id,
                require_no_later_dependencies,
            ):
                return False, ()
            if any(_version_retired(connection, memory_id) for memory_id in require_in_force):
                return False, ()
            if not _evidence_clause_changes_are_current(connection, reverse_clause_changes):
                return False, ()
            # Both identity reversals refuse before they write, so a refusal here leaves the
            # transaction with nothing to undo. The merge plan is checked rather than trusted:
            # re-merging under the wrong survivor would silently rename a person.
            if merge_identities is not None:
                survivor, restored = merge_identities
                plan = self._identity_link_plan(connection, survivor, restored)
                if plan is None or plan.target_id != survivor:
                    return False, ()
                self._merge_identities(connection, plan)
            restored_id = (
                None if split_identity is None else self._split_identity(connection, split_identity)
            )
            if split_identity is not None and restored_id is None:
                return False, ()
            reverted_at = max(rolled_back_at, _parse_datetime(_row_text(row, "applied_at")))
            for memory_id in dict.fromkeys(delete_memory_ids):
                _deleted, orphaned = self._delete_memory(connection, memory_id)
                unreferenced.extend(orphaned)
            _retire_memory_evidence(connection, retire_evidence, retired_at=reverted_at)
            _retire_evidence_clauses(
                connection,
                retire_clauses,
                retired_at=reverted_at,
            )
            _reverse_evidence_clause_changes(
                connection,
                reverse_clause_changes,
                reversed_at=reverted_at,
            )
            # Retired before the restore, so an assertion and the one it displaced are never both
            # in force: a naming assertion is retracted rather than deleted, because the log row
            # that recorded it stays readable and the audit trail must show both names.
            _retire_memory_versions(connection, retire_versions, retired_at=reverted_at)
            _restore_memory_versions(connection, restore_versions, recorded_at=reverted_at)
            _set_forgotten(connection, clear_forgotten, forgotten_at=None)
            _reproject_named_identities(
                connection,
                (
                    *(memory_id for memory_id, _source in retire_evidence),
                    *(memory_id for memory_id, _clause in retire_clauses),
                    *retire_versions,
                    *(entry if isinstance(entry, str) else entry[0] for entry in restore_versions),
                    *clear_forgotten,
                ),
            )
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
        before_operation_id: int | None = None,
    ) -> tuple[StoredOperation, ...]:
        """Return logged operations newest first, or the one matching an id or active key.

        `before_operation_id` continues a newest-first scan, which is what lets a caller that
        must see the whole log -- a data-subject export -- page it instead of truncating it.
        """
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
        elif before_operation_id is not None:
            where = "WHERE operation_id < ?"
            parameters = (before_operation_id,)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT operation_id, operation_key, intent, trigger, model_id, recipe,
                       operation_json, effects_json, applied_at, rolled_back_at,
                       outcome, outcome_note
                FROM memory_operations
                {where}
                ORDER BY operation_id DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def naming_operation_key(self, operation_key: str, memory_id: str) -> str | None:
        """Return a log key for a naming attempt, or None for a standing duplicate.

        A superseded deterministic assertion still has an active historical operation with the
        base key. Restating that retired assertion is a new auditable operation, keyed by the
        version it is about to create; repeating a currently standing assertion remains a no-op.
        """
        _require_identifier(operation_key, "operation_key")
        _require_identifier(memory_id, "memory_id")
        with self._connection() as connection:
            if _active_operation_id(connection, operation_key) is None:
                return operation_key
            if not _version_retired(connection, memory_id):
                return None
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to read naming assertion version")
        payload = f"mindbridge-operation-restatement-v1:{operation_key}:{int(row['version']) + 1}"
        return hashlib.sha256(payload.encode()).hexdigest()

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
        identity_id: str | None = None,
        active_only: bool = False,
    ) -> tuple[StoredMemory, ...]:
        """Hydrate existing memories with one query and preserve input ranking.

        `place_id` scopes the slate to one symbolic place. It is a hard filter, unlike `near`/
        `radius_m` which the caller applies to metric pose, and it is applied in SQL so a scoped
        hydration reads only the rows it returns.

        `identity_id` scopes it to one person, accepting a merged alias, and is a hard filter for
        the same reason: it is the authoritative answer the search index only approximates.
        """
        if not memory_ids:
            return ()
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        _require_optional_identifier(place_id, "place_id")
        _require_optional_identifier(identity_id, "identity_id")
        place_clause = "" if place_id is None else "AND place_id = ?"
        place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
        rows: list[sqlite3.Row] = []
        with self._read_transaction() as connection:
            identity_clause, identity_parameters = _identity_scope(connection, identity_id)
            if identity_clause is None:
                return ()
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
                        {identity_clause}
                        """,
                        (*batch, *place_parameters, *identity_parameters),
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
            if _memory_in_scope(
                row,
                active_only=active_only,
                known_at=known_at,
                near=near,
                semantic_ids=semantic_ids,
                scoped_ids=contexts.keys(),
            )
        }
        return tuple(by_id[memory_id] for memory_id in memory_ids if memory_id in by_id)

    def read_functional_lineage_representatives(
        self,
        lineage_ids: Sequence[str],
        *,
        per_lineage_limit: int,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> tuple[tuple[StoredMemory, ...], frozenset[str]]:
        """Read bounded current representatives for functional claim lineages.

        This is a completion read for context compilation, not a second retrieval route.  It starts
        from the lineage index, then uses ``read_memories`` for authoritative bitemporal, place,
        identity, and metric-scope filtering before it retains the rows for each distinct value.
        The compiler chooses one deliverable representative only after confidence, support, and
        consent checks, so an ineligible copy cannot hide an eligible copy of the same assertion.
        The value bound is applied after scope, so repeated or historical copies of the same
        assertion do not by themselves make a current claim incomplete.  The read may scan all
        current visible candidate rows of a requested lineage before identity and metric scope
        can prove which values survive; it never scans an unrelated lineage or uses a relevance
        score to choose a representative.
        """
        if not lineage_ids:
            return (), frozenset()
        for lineage_id in lineage_ids:
            _require_identifier(lineage_id, "lineage_id")
        if (
            isinstance(per_lineage_limit, bool)
            or not isinstance(per_lineage_limit, int)
            or per_lineage_limit <= 0
        ):
            raise ValueError("per_lineage_limit must be a positive integer")
        query_valid_at = valid_at or datetime.now(timezone.utc)
        query_known_at = known_at or datetime.now(timezone.utc)
        valid_text = _datetime_text(query_valid_at)
        known_text = _datetime_text(query_known_at)
        selected_ids: list[str] = []
        with self._read_transaction() as connection:
            for lineage_id in dict.fromkeys(lineage_ids):
                rows = connection.execute(
                    f"""
                    SELECT s.memory_id AS memory_id
                    FROM memory_semantics AS s
                    JOIN memory_versions AS v ON v.memory_id = s.memory_id
                    JOIN memory_records AS r ON r.memory_id = s.memory_id
                    WHERE s.lineage_id = ?
                      AND {_FUNCTIONAL_CLAIM_SQL}
                      AND r.forgotten_at IS NULL
                      AND v.visible = 1
                      AND v.recorded_at <= ?
                      AND (v.retired_at IS NULL OR v.retired_at > ?)
                      AND (v.valid_from IS NULL OR v.valid_from <= ?)
                      AND (v.valid_until IS NULL OR v.valid_until > ?)
                    ORDER BY s.memory_id
                    """,
                    (
                        lineage_id,
                        *_FUNCTIONAL_CLAIM_PARAMETERS,
                        known_text,
                        known_text,
                        valid_text,
                        valid_text,
                    ),
                ).fetchall()
                selected_ids.extend(_row_text(row, "memory_id") for row in rows)
        scoped = self.read_memories(
            selected_ids,
            valid_at=valid_at,
            known_at=known_at,
            near=near,
            radius_m=radius_m,
            place_id=place_id,
            identity_id=identity_id,
            active_only=True,
        )
        representatives: list[StoredMemory] = []
        values: dict[str, set[str]] = {}
        incomplete: set[str] = set()
        for memory in scoped:
            context = memory.context
            if context is None or context.lineage_id is None or context.value is None:
                continue
            lineage_values = values.setdefault(context.lineage_id, set())
            if context.value in lineage_values:
                representatives.append(memory)
                continue
            if len(lineage_values) >= per_lineage_limit:
                incomplete.add(context.lineage_id)
                continue
            lineage_values.add(context.value)
            representatives.append(memory)
        return tuple(representatives), frozenset(incomplete)

    def count_memories(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        active_only: bool = False,
    ) -> int:
        """Count what `read_memories` would return for the same arguments.

        Applies every scope predicate through the same selection pass, and reads neither
        content, media assets, nor typed context rows: a caller that only needs the number of
        surviving memories must not pay for the records.
        """
        if not memory_ids:
            return 0
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        _require_optional_identifier(place_id, "place_id")
        _require_optional_identifier(identity_id, "identity_id")
        place_clause = "" if place_id is None else "AND place_id = ?"
        place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
        by_id: dict[str, sqlite3.Row] = {}
        with self._read_transaction() as connection:
            identity_clause, identity_parameters = _identity_scope(connection, identity_id)
            if identity_clause is None:
                return 0
            for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                for row in connection.execute(
                    f"""
                    SELECT memory_id, created_at, forgotten_at
                    FROM memory_records
                    WHERE memory_id IN ({placeholders})
                    {place_clause}
                    {identity_clause}
                    """,
                    (*batch, *place_parameters, *identity_parameters),
                ).fetchall():
                    by_id[_row_text(row, "memory_id")] = row
            scoped, semantic_ids = _select_memory_contexts(
                connection,
                tuple(memory_ids),
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                active_only=active_only,
            )
        surviving = {
            memory_id
            for memory_id, row in by_id.items()
            if _memory_in_scope(
                row,
                active_only=active_only,
                known_at=known_at,
                near=near,
                semantic_ids=semantic_ids,
                scoped_ids=scoped.keys(),
            )
        }
        # Counted per requested ID, not per distinct row, because `read_memories` repeats a
        # memory that its caller asked for twice.
        return sum(1 for memory_id in memory_ids if memory_id in surviving)

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
            changed = _set_forgotten(connection, ids, forgotten_at=forgotten_at)
            _reproject_named_identities(connection, changed)
            return changed

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory; cascading embedding triggers enqueue index deletions."""
        deleted, _assets = self.delete_memory_with_assets(memory_id)
        return deleted

    def naming_projection_after_delete(
        self,
        memory_id: str,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]]:
        """Return what deleting this record removes, and the name each identity is left with.

        A naming assertion is an ordinary memory record, so an ordinary caller can delete it --
        and so is a record an assertion cites, which takes the assertion with it when it was
        the last evidence. Both move a projection, and the caller has to rebuild the indexed
        text that quoted the name in the same commit. This is how it finds out which names
        change, to what, and which records it must not hand back as replacements because this
        delete is about to remove them.
        """
        _require_identifier(memory_id, "memory_id")
        with self._connection() as connection:
            removed = _deletion_cascade(connection, (memory_id,))
            affected = _active_evidence_dependent_closure(connection, memory_id)
            identities = _naming_assertion_identities(connection, (memory_id, *affected))
            connection.execute("SAVEPOINT deletion_projection")
            try:
                self._delete_memory(connection, memory_id)
                projection = tuple(
                    (
                        identity_id,
                        _optional_row_text(row, "name")
                        if (
                            row := connection.execute(
                                "SELECT name FROM identities WHERE identity_id = ?",
                                (identity_id,),
                            ).fetchone()
                        )
                        is not None
                        else None,
                    )
                    for identity_id in identities
                )
            finally:
                connection.execute("ROLLBACK TO deletion_projection")
                connection.execute("RELEASE deletion_projection")
        return removed, projection

    def deletion_cascade(self, memory_ids: Sequence[str]) -> tuple[str, ...]:
        """Predict the exact bounded record cascade for deleting all supplied roots."""
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            _require_identifier(memory_id, "memory_id")
        if not selected:
            return ()
        with self._connection() as connection:
            return _deletion_cascade(connection, selected)

    def assets_orphaned_by_deletion(self, memory_ids: Sequence[str]) -> tuple[str, ...]:
        """Return assets whose every current memory reference is in the supplied set."""
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            _require_identifier(memory_id, "memory_id")
        if not selected:
            return ()
        with self._read_transaction() as connection:
            # Both halves of the orphan predicate must see the complete deletion set. Binding it
            # twice in one statement exceeds SQLite's conservative parameter budget as soon as a
            # retention page crosses one batch, while checking each batch independently would
            # misclassify an asset shared by memories in different batches. A connection-local
            # table keeps that set exact without changing authoritative state.
            connection.execute(
                "CREATE TEMP TABLE retention_deletions (memory_id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT INTO retention_deletions (memory_id) VALUES (?)",
                ((memory_id,) for memory_id in selected),
            )
            rows = connection.execute(
                """
                SELECT DISTINCT ma.asset_id
                FROM memory_assets AS ma
                JOIN temp.retention_deletions AS selected
                  ON selected.memory_id = ma.memory_id
                JOIN media_assets AS a ON a.asset_id = ma.asset_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_assets AS other
                    LEFT JOIN temp.retention_deletions AS deleting
                      ON deleting.memory_id = other.memory_id
                    WHERE other.asset_id = ma.asset_id
                      AND deleting.memory_id IS NULL
                )
                ORDER BY a.created_at, ma.asset_id
                """
            ).fetchall()
        return tuple(_row_text(row, "asset_id") for row in rows)

    def delete_memory_with_assets(
        self,
        memory_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory and return its assets that no remaining memory references.

        Pass `memories` and `embeddings` to atomically replace indexed documents in the same
        commit, which deleting a naming assertion needs: the projection it fed is recomputed
        here, so the indexed text quoting the name has to be rebuilt with it.
        """
        _require_identifier(memory_id, "memory_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        with self._transaction() as connection:
            deleted, unreferenced = self._delete_memory(connection, memory_id)
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return deleted, unreferenced

    def _delete_memory(  # noqa: C901 - one atomic dependency and lineage teardown
        self,
        connection: sqlite3.Connection,
        memory_id: str,
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory inside the caller's transaction; see `delete_memory_with_assets`."""
        if (
            connection.execute(
                "SELECT 1 FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            is None
        ):
            return False, ()
        now = datetime.now(timezone.utc)
        changed_at = now
        affected = _active_evidence_dependent_closure(connection, memory_id)
        clause_changes = _retire_evidence_clauses_for_source(
            connection,
            memory_id,
            retired_at=changed_at,
        )
        if clause_changes:
            changed_at = max(changed_at, *clause_changes.values())
        grounded = _grounded_affected_memory_ids(
            connection,
            affected,
            excluding=(memory_id,),
        )
        unsupported = tuple(
            dependent_id for dependent_id in affected if dependent_id not in grounded
        )
        for source_memory_id in unsupported:
            downstream_changes = _retire_evidence_clauses_for_source(
                connection,
                source_memory_id,
                retired_at=changed_at,
            )
            for dependent_id, dependent_changed_at in downstream_changes.items():
                clause_changes[dependent_id] = max(
                    clause_changes.get(dependent_id, dependent_changed_at),
                    dependent_changed_at,
                )
            if downstream_changes:
                changed_at = max(changed_at, *downstream_changes.values())
        removed_ids = (memory_id, *unsupported)
        surviving_dependents = tuple(
            dependent_id for dependent_id in clause_changes if dependent_id not in removed_ids
        )
        for dependent_id in surviving_dependents:
            _refresh_evidence_projection(
                connection,
                dependent_id,
                clause_changes[dependent_id],
            )

        reconciled: set[tuple[str, str]] = set()
        for source_memory_id in removed_ids:
            semantic = connection.execute(
                """
                SELECT lineage_id, kind, basis
                FROM memory_semantics WHERE memory_id = ?
                """,
                (source_memory_id,),
            ).fetchone()
            if semantic is not None and (
                _row_text(semantic, "kind") == MemoryKind.STATE.value
                or (
                    _row_text(semantic, "kind") == MemoryKind.TRAIT.value
                    and _row_text(semantic, "basis") == EvidenceBasis.USER_STATEMENT.value
                )
            ):
                reconciled.add((_row_text(semantic, "lineage_id"), _row_text(semantic, "kind")))
        # Read before deletion: a derived naming assertion may be several dependency hops from
        # the requested source. Every unsupported node must stop feeding the identity projection.
        named_identities = _naming_assertion_identities(
            connection,
            (*removed_ids, *surviving_dependents),
        )
        displaced = [
            candidate
            for removed_id in removed_ids
            for candidate in _displaced_naming_versions(connection, removed_id)
        ]
        linked_ids = [
            _row_text(row, "asset_id")
            for removed_id in removed_ids
            for row in connection.execute(
                """
                SELECT asset_id FROM memory_assets
                WHERE memory_id = ? ORDER BY position
                """,
                (removed_id,),
            ).fetchall()
        ]
        _restamp_dependent_evidence(connection, surviving_dependents, changed_at)
        for removed_id in removed_ids:
            if removed_id != memory_id:
                connection.execute(
                    "DELETE FROM memory_records WHERE memory_id = ?",
                    (removed_id,),
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
        _restore_memory_versions(
            connection,
            tuple(
                candidate
                for candidate in dict.fromkeys(displaced)
                if _version_retired(connection, candidate)
            ),
            recorded_at=changed_at,
        )
        _reproject_identities(connection, named_identities)
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

    def read_visual_descriptions(
        self,
        asset_ids: Sequence[str],
        *,
        space_id: str,
    ) -> dict[str, str]:
        """Return the cached caption for every supplied asset described in this vision space.

        Batched because the write path describes a whole `add_many` at once, and an asset with no
        caption in this space is simply absent rather than an error: a miss is the normal state.
        """
        _require_identifier(space_id, "vision space_id")
        for asset_id in asset_ids:
            _sha256(asset_id)
        if not asset_ids:
            return {}
        unique_ids = tuple(dict.fromkeys(asset_ids))
        found: dict[str, str] = {}
        with self._connection() as connection:
            for offset in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                found.update(
                    (_row_text(row, "asset_id"), _row_text(row, "description"))
                    for row in connection.execute(
                        f"""
                        SELECT asset_id, description
                        FROM visual_descriptions
                        WHERE space_id = ? AND asset_id IN ({placeholders})
                        """,
                        (space_id, *batch),
                    ).fetchall()
                )
        return found

    def write_visual_descriptions(
        self,
        descriptions: Mapping[str, str],
        *,
        model_id: str,
        space_id: str,
    ) -> int:
        """Cache derived captions for stored assets in one durable SQLite transaction.

        An asset already described in this space keeps its stored caption: the point of the row is
        that two ingests of one picture build the same document, so a concurrent writer that
        described the same bytes first wins and the later caption is dropped.
        """
        _require_identifier(model_id, "vision model_id")
        _require_identifier(space_id, "vision space_id")
        for asset_id, description in descriptions.items():
            _sha256(asset_id)
            if not isinstance(description, str) or not description.strip():
                raise ValueError("visual description must not be blank")
        if not descriptions:
            return 0
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            cursor = connection.executemany(
                """
                INSERT INTO visual_descriptions (
                    asset_id, space_id, model_id, description, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (asset_id, space_id) DO NOTHING
                """,
                (
                    (asset_id, space_id, model_id, description, now)
                    for asset_id, description in descriptions.items()
                ),
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
            _queue_asset_identity_projection(connection, asset_id)
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
            _queue_asset_identity_projection(connection, asset_id)
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

    def projected_identity_name(
        self,
        identity_id: str,
        *,
        excluding: Sequence[str] = (),
    ) -> tuple[str | None, str | None]:
        """Return the name and relationship the current visible naming assertion projects.

        Both are `None` for an unknown or as yet unnamed identity, which is exactly what the
        projection should then write. `excluding` ignores assertions the caller is about to
        delete, so it can rebuild indexed text for the projection the deletion will leave.
        """
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            row = (
                None
                if resolved_id is None
                else _current_naming_assertion(connection, resolved_id, excluding=excluding)
            )
        if row is None:
            return None, None
        return _optional_row_text(row, "subject"), _optional_row_text(row, "value")

    def naming_projection_after_assertion(
        self,
        identity_id: str,
        memory_id: str,
        context: MemoryContext,
    ) -> tuple[str | None, str | None]:
        """Preview the projection after applying one bound naming assertion.

        The caller uses this while holding its write lock to build speech documents before the
        assertion transaction. The store remains the source of the evidence-group visibility
        rule, so the preview and the commit cannot disagree about corroboration.
        """
        _require_identifier(identity_id, "identity_id")
        _require_identifier(memory_id, "memory_id")
        if context.kind is not MemoryKind.ENTITY or context.identity_id != identity_id:
            raise ValueError("context must be a naming assertion for identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None, None
            groups = {
                _row_text(row, "source_group_id")
                for row in connection.execute(
                    """
                    SELECT source_group_id FROM memory_evidence
                    WHERE memory_id = ? AND retired_at IS NULL
                    """,
                    (memory_id,),
                ).fetchall()
            }
            for source_memory_id in context.evidence_ids:
                source = connection.execute(_SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
                if source is None:
                    raise sqlite3.IntegrityError("evidence source memory does not exist")
                groups.add(_row_text(source, "source_group_id"))
            visible = _semantic_visibility(
                connection,
                memory_id=memory_id,
                lineage_id=context.lineage_id or memory_id,
                kind=context.kind.value,
                basis=context.basis.value,
                identity_id=context.identity_id,
                evidence_count=len(groups),
                valid_from=context.valid_from,
                valid_until=context.valid_until,
            )
            if visible:
                return context.subject, context.value
            current = _current_naming_assertion(connection, resolved_id)
        if current is None:
            return None, None
        return _optional_row_text(current, "subject"), _optional_row_text(current, "value")

    def refresh_identity_projection(
        self,
        identity_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> IdentityProfile | None:
        """Recompute `identities.name`/`relationship` from the current naming assertion.

        The assertion is the record and this is the read path, the way `memory_versions`
        carries the evidence and `_refresh_evidence_projection` recomputes what reads see.
        Supplied documents replace their indexed vectors in the same transaction, so stored
        text can never disagree with the projection it was rebuilt for. Returns the projected
        profile, or None when the identity does not exist.
        """
        _require_identifier(identity_id, "identity_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        now = _datetime_text(datetime.now(timezone.utc))
        with self._transaction() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            row = _current_naming_assertion(connection, resolved_id)
            name = None if row is None else _optional_row_text(row, "subject")
            relationship = None if row is None else _optional_row_text(row, "value")
            cursor = connection.execute(
                """
                UPDATE identities
                SET name = ?, relationship = ?, updated_at = ?
                WHERE identity_id = ?
                """,
                (name, relationship, now, resolved_id),
            )
            if cursor.rowcount == 0:
                return None
            evidence_ids = (
                ()
                if row is None
                else _current_evidence_ids(connection, _row_text(row, "memory_id"))
            )
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        return IdentityProfile(
            identity_id=resolved_id,
            name=name,
            relationship=relationship,
            confirmed=row is not None,
            evidence_ids=evidence_ids,
        )

    def identity_for_subject(self, subject: str) -> str | None:
        """Return the identity whose visible naming assertion canonically names this subject.

        Deterministic and never a model's decision: the comparison is the same NFKC casefold
        the semantic layer already uses, and two identities that currently project the same
        canonical name resolve to neither, because binding a claim to the wrong person is
        worse than leaving it unbound.
        """
        canonical = _canonical_subject(subject)
        if canonical is None:
            return None
        with self._connection() as connection:
            matches = {
                _row_text(row, "identity_id")
                for row in _visible_naming_assertions(connection)
                if _canonical_subject(_optional_row_text(row, "subject")) == canonical
            }
        return matches.pop() if len(matches) == 1 else None

    def provisional_identities(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return, per memory, the identities it observes that no visible assertion names.

        A person a memory saw or heard but nobody has named yet: present in the evidence and
        absent from every projection. Empty entries are dropped, so a caller can treat a
        missing key as "nobody unnamed here".
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            _require_identifier(memory_id, "memory_id")
        if not ids:
            return {}
        provisional: dict[str, tuple[str, ...]] = {}
        observed: list[tuple[str, str]] = []
        with self._connection() as connection:
            for offset in range(0, len(ids), _SQLITE_PARAMETER_BATCH // 2):
                batch = ids[offset : offset + _SQLITE_PARAMETER_BATCH // 2]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows = connection.execute(
                    f"""
                    SELECT ma.memory_id AS memory_id, s.speaker_id AS identity_id
                    FROM memory_assets AS ma
                    JOIN speech_segments AS s ON s.asset_id = ma.asset_id
                    WHERE ma.memory_id IN ({placeholders}) AND s.speaker_id IS NOT NULL
                    UNION
                    SELECT ma.memory_id AS memory_id, f.identity_id AS identity_id
                    FROM memory_assets AS ma
                    JOIN face_observations AS f ON f.asset_id = ma.asset_id
                    WHERE ma.memory_id IN ({placeholders})
                    """,
                    (*batch, *batch),
                ).fetchall()
                observed.extend(
                    (_row_text(row, "memory_id"), _row_text(row, "identity_id")) for row in rows
                )
            # Asked only about the people actually observed here: the assertion scan is over
            # every named person otherwise, to answer about a handful.
            named = {
                _row_text(row, "identity_id")
                for row in _visible_naming_assertions(
                    connection,
                    identity_ids=tuple(identity_id for _memory_id, identity_id in observed),
                    valid_at=valid_at,
                    known_at=known_at,
                )
            }
        for memory_id, identity_id in observed:
            if identity_id not in named:
                provisional[memory_id] = (*provisional.get(memory_id, ()), identity_id)
        return {
            memory_id: tuple(sorted(set(identity_ids)))
            for memory_id, identity_ids in sorted(provisional.items())
        }

    def named_actors(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
    ) -> dict[str, tuple[tuple[str, str, str | None], ...]]:
        """Return, per memory, the NAMED identities its identity edge resolves to.

        A memory carries an identity edge two ways: its own semantic assertion may be bound to
        an identity (`memory_semantics.identity_id`, the field `MemoryContext.identity_id`
        projects), or its media asset may have recognized one -- the same asset-keyed join
        `provisional_identities` reads for the unnamed case. Each entry is `(identity_id,
        name, naming_memory_id)`. The naming assertion's own memory is never reported for
        itself, because it already renders as its own actors hit. Empty entries are dropped,
        so a caller can treat a missing key as "nobody named here".
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            _require_identifier(memory_id, "memory_id")
        if not ids:
            return {}
        with self._connection() as connection:
            named = _named_identities(connection, valid_at=valid_at, known_at=known_at)
            if not named:
                return {}
            links: dict[str, list[tuple[str, str, str]]] = {}
            for offset in range(0, len(ids), _SQLITE_PARAMETER_BATCH // 3):
                batch = ids[offset : offset + _SQLITE_PARAMETER_BATCH // 3]
                _merge_named_identity_links(connection, batch, named, links)
        return {memory_id: tuple(sorted(entries)) for memory_id, entries in sorted(links.items())}

    def co_derived_events(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return, per memory, the active event records formed from the same observations.

        The edge is shared evidence: an event is reported for a memory exactly when both cite
        one `memory_evidence.source_memory_id`. That is co-occurrence inside one capture and
        not a cause, and the memory itself is never its own co-derived event. The candidates are
        then read through `read_memories(active_only=True)` under every scope axis a search
        hydration applies, so a retired version, a hidden assertion, a forgotten record, and
        anything outside the asked-for validity, pose, or place are all left out. That second
        read is its own transaction: a `known_at` pins what both see, and without one a write
        landing between them can only remove an event from the hop, never add one the join did
        not already hold. Empty entries are dropped, so a missing key means no event shares this
        memory's observations.
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            _require_identifier(memory_id, "memory_id")
        # Eagerly, because the hydration that would otherwise validate these is skipped whenever
        # the join finds nothing.
        _require_scope_axes(valid_at=valid_at, known_at=known_at, near=near, radius_m=radius_m)
        _require_optional_identifier(place_id, "place_id")
        _require_optional_identifier(identity_id, "identity_id")
        if not ids:
            return {}
        # Both evidence sides are read as of the same transaction time the versions are, so a
        # link retired after `known_at` still joins the pair it joined then.
        current = "{alias}.retired_at IS NULL"
        as_of = (
            "{alias}.recorded_at <= ? AND ({alias}.retired_at IS NULL OR {alias}.retired_at > ?)"
        )
        clause = current if known_at is None else as_of
        known_parameters: tuple[object, ...] = (
            () if known_at is None else (_datetime_text(known_at), _datetime_text(known_at))
        )
        derived: dict[str, list[str]] = {}
        # Half a batch: each statement binds the cue IDs once and the time bounds twice.
        with self._read_transaction() as connection:
            for offset in range(0, len(ids), _SQLITE_PARAMETER_BATCH // 2):
                batch = ids[offset : offset + _SQLITE_PARAMETER_BATCH // 2]
                placeholders = ", ".join("?" for _memory_id in batch)
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT cue.memory_id AS memory_id, ev.memory_id AS event_id
                    FROM memory_evidence AS cue
                    JOIN memory_evidence AS ev
                        ON ev.source_memory_id = cue.source_memory_id
                        AND ev.memory_id <> cue.memory_id
                    JOIN memory_semantics AS s ON s.memory_id = ev.memory_id
                    WHERE cue.memory_id IN ({placeholders}) AND s.kind = ?
                      AND {clause.format(alias="cue")} AND {clause.format(alias="ev")}
                    """,
                    (*batch, MemoryKind.EVENT.value, *known_parameters, *known_parameters),
                ).fetchall():
                    derived.setdefault(_row_text(row, "memory_id"), []).append(
                        _row_text(row, "event_id")
                    )
        if not derived:
            return {}
        candidates = tuple(
            dict.fromkeys(event_id for events in derived.values() for event_id in events)
        )
        active = {
            memory.memory_id
            for memory in self.read_memories(
                candidates,
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                place_id=place_id,
                identity_id=identity_id,
                active_only=True,
            )
        }
        return {
            memory_id: standing
            for memory_id, events in sorted(derived.items())
            if (standing := tuple(sorted(event for event in events if event in active)))
        }

    def identity_profile(self, identity_id: str) -> IdentityProfile | None:
        """Return one identity's profile, or None when it does not exist.

        A merged identity resolves through its alias, like every other identity read, and the
        returned profile carries the canonical ID so a caller never has to resolve it twice.

        `confirmed` and `evidence_ids` are derived, not stored: a person is confirmed exactly
        while a visible naming assertion names them, and the evidence is that assertion's.
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
            assertion = _current_naming_assertion(connection, resolved_id)
            evidence_ids = (
                ()
                if assertion is None
                else _current_evidence_ids(connection, _row_text(assertion, "memory_id"))
            )
        return IdentityProfile(
            identity_id=resolved_id,
            name=_optional_row_text(row, "name"),
            relationship=_optional_row_text(row, "relationship"),
            confirmed=assertion is not None,
            evidence_ids=evidence_ids,
        )

    def identity_consent(self, identity_id: str) -> ConsentState | None:
        """Return the consent state one identity's standing assertion projects, or None.

        `None` is "nobody has recorded a statement", which is not consent and not a refusal.
        A merged alias resolves to its canonical identity, like every other identity read.
        """
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            value = _current_consent_state(connection, resolved_id)
        return None if value is None else ConsentState(value)

    def restrained_identities(self) -> frozenset[str]:
        """Return every identity whose standing consent withholds or withdraws processing."""
        with self._connection() as connection:
            return _restrained_identities(connection)

    def identity_assertion_memory_ids(self, identity_id: str) -> tuple[str, ...]:
        """Return every record bound to one identity, standing or superseded.

        This is the assertion half of what is held about a person -- their names and their
        consent statements, every version of each -- which `identity_memory_ids` does not cover
        because those records observe nobody.
        """
        _require_identifier(identity_id, "identity_id")
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return ()
            rows = connection.execute(
                """
                SELECT memory_id FROM memory_semantics
                WHERE identity_id = ? ORDER BY memory_id
                """,
                (resolved_id,),
            ).fetchall()
        return tuple(_row_text(row, "memory_id") for row in rows)

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
        operation: StoredOperation | None = None,
    ) -> str | None:
        """Merge two identities under one stable ID, recording a reversible alias.

        Pass allow_shared_modality to commit a plan obtained with the same intent.

        `operation` logs the control-plane operation that committed this merge in the same
        transaction, so a cross-modal bind is visible in the operation log and reversible
        through it. The log row carries the naming assertions the merge moved onto the survivor
        as `changed_ids`. Returns None, changing nothing, when that operation key is already
        applied and not rolled back.
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
            if (
                operation is not None
                and _active_operation_id(connection, operation.operation_key) is not None
            ):
                return None
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
            moved_claims = self._merge_identities(connection, plan)
            if operation is not None:
                _insert_operation(connection, replace(operation, changed_ids=moved_claims))
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return plan.target_id

    @staticmethod
    def _merge_identities(
        connection: sqlite3.Connection,
        plan: IdentityLink,
    ) -> tuple[str, ...]:
        """Apply one validated merge inside an open transaction.

        Returns the naming assertions that moved onto the survivor, which is the one effect a
        merge has on memory records and therefore the one an operation log row can name.
        """
        target, source = plan.target_id, plan.source_id
        contributed = _sole_identity_modality(connection, source)
        now = _datetime_text(datetime.now(timezone.utc))
        # Read before the source's `identities` row is deleted below, so the alias row can carry
        # forward the absorbed identity's own creation time rather than the merge time. Rollback
        # (`_split_identity`) restores the identity from this value; storing `now` here instead
        # would make every merge-then-rollback cycle overwrite the original `created_at`.
        source_created_at = _row_text(
            connection.execute(
                "SELECT created_at FROM identities WHERE identity_id = ?",
                (source,),
            ).fetchone(),
            "created_at",
        )
        # Before the re-pointing below, while the rows the index projection reads still say
        # `source`: every memory it names is about `target` afterwards, and only the outbox can
        # tell the index that.
        _queue_identity_projection(connection, source)
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
            SET created_at = ?, updated_at = ?
            WHERE identity_id = ?
            """,
            (plan.created_at, now, target),
        )
        moved_claims = tuple(
            _row_text(row, "memory_id")
            for row in connection.execute(
                "SELECT memory_id FROM memory_semantics WHERE identity_id = ? ORDER BY memory_id",
                (source,),
            ).fetchall()
        )
        # The source's naming assertions move to the survivor before the source row goes,
        # or `ON DELETE SET NULL` would strand the name that was asserted about this person
        # and the projection would go on reporting a name nothing supports. The name is a
        # projection here too: recomputed from the assertions, never assigned.
        connection.execute(
            "UPDATE memory_semantics SET identity_id = ? WHERE identity_id = ?",
            (target, source),
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
            (source, target, source_created_at, contributed),
        )
        connection.execute("DELETE FROM identities WHERE identity_id = ?", (source,))
        if _has_naming_assertion(connection, target):
            _reproject_identities(connection, (target,))
        return moved_claims

    def identity_alias_modality(self, alias_id: str) -> Literal["face", "voice"] | None:
        """Return the modality one merged alias contributed, or None when none is recorded.

        A caller that must rebuild derived text before `unlink_identity` has to know which
        modality is about to move back, and the alias row is the only record of it.
        """
        _require_identifier(alias_id, "alias_id")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT contributed_modality FROM identity_aliases WHERE alias_id = ?",
                (alias_id,),
            ).fetchone()
        if row is None or row["contributed_modality"] is None:
            return None
        return _identity_modality(row["contributed_modality"])

    def unlink_identity(
        self,
        alias_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
        operation: StoredOperation | None = None,
    ) -> str | None:
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

        Pass `memories` and `embeddings` to atomically replace indexed documents that named
        the merged person, exactly as `register_identity` does. They are applied only when the
        unlink actually commits, so a refused unlink leaves the projection untouched.

        Claims derived while the two were one identity are re-evaluated in the same
        transaction: one resting only on media that moved back is re-attributed to the
        restored identity, one resting on both people's media is unbound, and one that never
        involved the restored modality keeps its binding.

        `operation` logs the control-plane operation that split this merge, in the same
        transaction, so a split is visible in the operation log and reversible through it. It is
        written only when the split actually commits.
        """
        _require_identifier(alias_id, "alias_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = _prepare_write_batch(
            memories,
            embeddings,
        )
        with self._transaction() as connection:
            if self._split_identity(connection, alias_id) is None:
                return None
            if operation is not None:
                _insert_operation(connection, operation)
            self._replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return alias_id

    @staticmethod
    def _split_identity(connection: sqlite3.Connection, alias_id: str) -> str | None:
        """Reverse one recorded merge inside an open transaction.

        Returns the restored identity, or None -- having changed nothing -- when the alias has
        no recorded split point or the survivor would be left with no exemplars.
        """
        now = _datetime_text(datetime.now(timezone.utc))
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
        # Same reason as the merge: the rows below move from `target` to `alias_id`, so the
        # index projection of every memory naming `target` has to be rebuilt.
        _queue_identity_projection(connection, target)
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
        _rebind_unlinked_claims(
            connection,
            target=target,
            restored=alias_id,
            modality=modality,
        )
        if _has_naming_assertion(connection, target):
            _reproject_identities(connection, (target,))
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
        operation: StoredOperation | None = None,
    ) -> tuple[IdentityErasure, tuple[StoredAsset, ...]] | None:
        """Erase one person's identity cluster, keeping the memories that mention them.

        Accepts a canonical ID or any merged alias and removes the whole cluster: the profile,
        every face and voice exemplar, every alias, and the accumulated cross-modal link
        evidence. Returns the erasure and the assets no remaining memory references, the way
        `delete_memory_with_assets` does, or None, changing nothing, when no such identity
        exists.

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

        `operation` logs, in the same transaction, that this erasure happened. The row is audit
        history and nothing more: it names the identity and its aliases, and the naming
        assertions the erasure deleted, so `operations()` can show that a person was erased
        without holding anything a rollback could restore.

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
            # Before the erasure strips the rows the index projection reads.
            _queue_identity_projection(connection, resolved_id)
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
            # The naming assertions go with the person. `ON DELETE SET NULL` below would keep
            # them as unattributed records still carrying the erased name in their content and
            # their vectors, which is exactly what erasure promises to remove. This goes through
            # the ordinary delete, so whatever cited the assertion is reconciled and reprojected
            # exactly as it is when a caller deletes the assertion itself.
            erased_claims = tuple(
                _row_text(row, "memory_id")
                for row in connection.execute(
                    """
                    SELECT memory_id FROM memory_semantics
                    WHERE identity_id = ? AND kind = ?
                    ORDER BY memory_id
                    """,
                    (resolved_id, MemoryKind.ENTITY.value),
                ).fetchall()
            )
            unreferenced: list[StoredAsset] = []
            for claim_id in erased_claims:
                _deleted, orphaned = self._delete_memory(connection, claim_id)
                unreferenced.extend(orphaned)
            # Cascades the aliases, exemplars and link evidence; NULLs the speech segments.
            connection.execute("DELETE FROM identities WHERE identity_id = ?", (resolved_id,))
            if operation is not None:
                _insert_operation(connection, replace(operation, changed_ids=erased_claims))
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
        ), tuple({asset.asset_id: asset for asset in unreferenced}.values())

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

    def asset_memory_ids(self, asset_ids: Sequence[str]) -> tuple[str, ...]:
        """Return every memory that references any of these assets, oldest memory first."""
        selected = tuple(dict.fromkeys(asset_ids))
        for asset_id in selected:
            _sha256(asset_id)
        if not selected:
            return ()
        found: list[str] = []
        with self._connection() as connection:
            for offset in range(0, len(selected), _SQLITE_PARAMETER_BATCH):
                batch = selected[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                found.extend(
                    _row_text(row, "memory_id")
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT memory_id FROM memory_assets
                        WHERE asset_id IN ({placeholders})
                        ORDER BY memory_id
                        """,
                        batch,
                    ).fetchall()
                )
        return tuple(dict.fromkeys(found))

    def forgotten_memory_ids(
        self,
        *,
        forgotten_before: datetime,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """Return memories cognitively forgotten before a moment, oldest forgetting first.

        The read half of retention over `forget()`: cognitive forgetting is reversible by
        design, so something has to decide when it becomes final, and only a declared policy or
        a person may.
        """
        _require_aware(forgotten_before, "forgotten_before")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM memory_records
                WHERE forgotten_at IS NOT NULL AND forgotten_at < ?
                ORDER BY forgotten_at, memory_id
                LIMIT ?
                """,
                (_datetime_text(forgotten_before), limit),
            ).fetchall()
        return tuple(_row_text(row, "memory_id") for row in rows)

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

    def read_text_selectors(
        self,
        embedding_ids: Sequence[str],
    ) -> dict[str, tuple[StoredTextSelector, ...]]:
        """Read exact text selectors for a bounded embedding-ID batch."""
        if not embedding_ids:
            return {}
        for embedding_id in embedding_ids:
            _require_identifier(embedding_id, "embedding_id")
        selector_rows: list[sqlite3.Row] = []
        piece_rows: list[sqlite3.Row] = []
        with self._connection() as connection:
            for offset in range(0, len(embedding_ids), _SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _embedding_id in batch)
                selector_rows.extend(
                    connection.execute(
                        f"""
                        SELECT embedding_id, selector_position, parent_content_sha256,
                               embedding_input_sha256, recipe_version
                        FROM embedding_text_selectors
                        WHERE embedding_id IN ({placeholders})
                        ORDER BY embedding_id, selector_position
                        """,
                        tuple(batch),
                    ).fetchall()
                )
                piece_rows.extend(
                    connection.execute(
                        f"""
                        SELECT embedding_id, selector_position, piece_position, role,
                               start_codepoint, end_codepoint, piece_sha256
                        FROM embedding_text_span_pieces
                        WHERE embedding_id IN ({placeholders})
                        ORDER BY embedding_id, selector_position, piece_position
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        pieces: dict[tuple[str, int], list[StoredTextSpanPiece]] = {}
        for row in piece_rows:
            key = (_row_text(row, "embedding_id"), int(row["selector_position"]))
            pieces.setdefault(key, []).append(
                StoredTextSpanPiece(
                    role=_row_text(row, "role"),  # type: ignore[arg-type]
                    start_codepoint=int(row["start_codepoint"]),
                    end_codepoint=int(row["end_codepoint"]),
                    piece_sha256=_row_text(row, "piece_sha256"),
                )
            )
        grouped: dict[str, list[StoredTextSelector]] = {}
        for row in selector_rows:
            embedding_id = _row_text(row, "embedding_id")
            position = int(row["selector_position"])
            selector_pieces = tuple(pieces.get((embedding_id, position), ()))
            if not selector_pieces:
                continue
            grouped.setdefault(embedding_id, []).append(
                StoredTextSelector(
                    parent_content_sha256=_row_text(row, "parent_content_sha256"),
                    embedding_input_sha256=_row_text(row, "embedding_input_sha256"),
                    recipe_version=_row_text(row, "recipe_version"),
                    pieces=selector_pieces,
                )
            )
        return {
            embedding_id: tuple(grouped.get(embedding_id, ())) for embedding_id in embedding_ids
        }

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
                               m.occurred_at, m.occurred_end, m.place_id
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.embedding_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
            identity_ids = _identity_projections(
                connection,
                tuple({_row_text(row, "memory_id") for row in rows}),
            )
        by_id: dict[str, IndexDocument] = {}
        for row in rows:
            document = _index_document_from_row(row, identity_ids)
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
                               m.occurred_at, m.occurred_end, m.place_id
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.memory_id IN ({placeholders})
                        ORDER BY e.object_part, e.embedding_id
                        """,
                        tuple(batch),
                    ).fetchall()
                )
            identity_ids = _identity_projections(connection, memory_ids)
        by_memory: dict[str, list[IndexDocument]] = {}
        for row in rows:
            document = _index_document_from_row(row, identity_ids)
            by_memory.setdefault(document.embedding.memory_id, []).append(document)
        return tuple(
            document for memory_id in memory_ids for document in by_memory.get(memory_id, ())
        )

    def iter_memory_embedding_vectors(
        self,
        memory_ids: Sequence[str],
        *,
        space_id: str,
        task: str,
    ) -> Generator[tuple[str, tuple[float, ...]], None, None]:
        """Stream matching vectors without hydrating parent memories or index payloads.

        The repeated memory ID is intentional: one memory can have an aggregate embedding and
        several retrieval-part embeddings. Callers that score parent memories must consider every
        row and reduce them under their own scoring rule.
        """
        if not memory_ids:
            return
        for memory_id in memory_ids:
            _require_identifier(memory_id, "memory_id")
        _require_identifier(space_id, "space_id")
        _require_identifier(task, "task")
        with self._connection() as connection:
            for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows = connection.execute(
                    f"""
                    SELECT memory_id, object_part, dimension, vector
                    FROM embeddings
                    WHERE memory_id IN ({placeholders})
                      AND space_id = ? AND task = ?
                    """,
                    (*batch, space_id, task),
                )
                for row in rows:
                    vector = row["vector"]
                    if not isinstance(vector, bytes):
                        raise RuntimeError("stored embedding vector is not a BLOB")
                    yield _row_text(row, "memory_id"), _unpack_vector(vector, int(row["dimension"]))

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
        """Create the current schema in an empty directory, or refuse any other version.

        This build has no upgrade path. A directory written by an older MindBridge is refused
        rather than converted, because SQLite is the authoritative copy: re-create the directory
        and re-ingest, or open it with the version that wrote it.
        """
        with self._connection() as connection:
            version = _user_version(connection)
            tables = _table_names(connection)
            if version == 0:
                _create_schema(connection, tables)
                version = _user_version(connection)
                tables = _table_names(connection)
            if version != _SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"unsupported local schema version {version}; expected {_SCHEMA_VERSION}. "
                    + (
                        "Re-create the data directory and re-ingest, or open it with the "
                        "MindBridge version that wrote it."
                        if version < _SCHEMA_VERSION
                        else "This directory was written by a newer MindBridge."
                    )
                )
            missing_tables = _REQUIRED_TABLES - tables
            if missing_tables:
                names = ", ".join(sorted(missing_tables))
                raise UnsupportedSchemaError(f"local schema is missing required tables: {names}")
            _validate_evidence_clause_schema(connection)
            _validate_text_selector_schema(connection)

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
    def _connection(self, *, secure_delete: bool = False) -> Iterator[sqlite3.Connection]:
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
        self._require_open()
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
        connection.execute(
            "DELETE FROM embedding_text_selectors WHERE embedding_id = ?",
            (embedding.embedding_id,),
        )
        if not embedding.text_selectors:
            return
        row = connection.execute(
            "SELECT content FROM memory_records WHERE memory_id = ?",
            (embedding.memory_id,),
        ).fetchone()
        if row is None:
            raise ValueError("text selectors require an existing parent memory")
        content = _row_text(row, "content")
        parent_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        for selector_position, selector in enumerate(embedding.text_selectors):
            if selector.parent_content_sha256 != parent_digest:
                raise ValueError("text selector parent digest does not match stored content")
            source_pieces = tuple(
                content[piece.start_codepoint : piece.end_codepoint] for piece in selector.pieces
            )
            for piece, source_text in zip(selector.pieces, source_pieces, strict=True):
                if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != piece.piece_sha256:
                    raise ValueError("text selector piece does not match stored content")
            embedding_input = (
                source_pieces[0]
                if len(source_pieces) == 1
                else f"{source_pieces[0]}\n\n{source_pieces[1]}"
            )
            if (
                hashlib.sha256(embedding_input.encode("utf-8")).hexdigest()
                != selector.embedding_input_sha256
            ):
                raise ValueError("text selector does not reconstruct its embedding input")
            connection.execute(
                """
                INSERT INTO embedding_text_selectors (
                    embedding_id, selector_position, parent_content_sha256,
                    embedding_input_sha256, recipe_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    embedding.embedding_id,
                    selector_position,
                    selector.parent_content_sha256,
                    selector.embedding_input_sha256,
                    selector.recipe_version,
                ),
            )
            for piece_position, piece in enumerate(selector.pieces):
                connection.execute(
                    """
                    INSERT INTO embedding_text_span_pieces (
                        embedding_id, selector_position, piece_position, role,
                        start_codepoint, end_codepoint, piece_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        embedding.embedding_id,
                        selector_position,
                        piece_position,
                        piece.role,
                        piece.start_codepoint,
                        piece.end_codepoint,
                        piece.piece_sha256,
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
        superseded: list[tuple[str, int]] | None = None,
        write_context_evidence: bool = True,
        context_recorded_at: datetime | None = None,
    ) -> bool:
        existing = connection.execute(
            """
            SELECT content, modality, memory_type, metadata_json, occurred_at, occurred_end,
                   place_id
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
        reactivating = memory.context is not None and _version_retired(connection, memory.memory_id)
        index_content_changed = existing is not None and (
            _row_text(existing, "content") != memory.content
            or _row_text(existing, "modality") != memory.modality
            or _row_text(existing, "memory_type") != memory.memory_type
            or _row_text(existing, "metadata_json") != memory.metadata_json
            or existing["occurred_at"] != _optional_datetime_text(memory.occurred_at)
            or existing["occurred_end"] != _optional_datetime_text(memory.occurred_end)
            or existing_asset_ids != supplied_asset_ids
            # The index carries `place_id` as a filter field, so relabelling a room is a change
            # to the indexed document even though none of its text moved.
            or existing["place_id"] != memory.place_id
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
        if index_content_changed or reactivating:
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
                superseded=superseded,
                write_evidence=write_context_evidence,
                recorded_at=context_recorded_at,
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
        # Consent restrains enrolment, not recognition. A person who withheld or withdrew it
        # still matches the exemplars already held -- destroying those is `forget_identity` --
        # but this observation adds nothing to their template, so the bank stops growing from
        # the moment they said so.
        restrained = _restrained_identities(connection)
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
            if identity_id in restrained:
                # Recognized, reported, and nothing written: no exemplar, no `identities` row
                # update, and so nothing for a rollback to undo either.
                matches[label] = (
                    identity_id,
                    None if score is None else max(0.0, min(1.0, score)),
                )
                group_claims.add(identity_id)
                continue
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

    def identity_evidence_memory_ids(
        self,
        identity_id: str,
        memory_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return which of `memory_ids` carry a face or voice occurrence of one identity.

        The control plane asks this before it lets a proposal name a person: a name may only be
        pinned on somebody the cited evidence actually contains. Unknown identities and unknown
        memory IDs simply do not match, so the caller reads an empty result as "not involved".
        """
        _require_identifier(identity_id, "identity_id")
        candidates = tuple(dict.fromkeys(memory_ids))
        if not candidates:
            return ()
        placeholders = ", ".join("?" for _memory_id in candidates)
        with self._connection() as connection:
            resolved_id = _resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return ()
            rows = connection.execute(
                f"""
                SELECT DISTINCT ma.memory_id
                FROM memory_assets AS ma
                WHERE ma.memory_id IN ({placeholders}) AND (
                    EXISTS (
                        SELECT 1 FROM speech_segments AS s
                        WHERE s.asset_id = ma.asset_id AND s.speaker_id = ?
                    ) OR EXISTS (
                        SELECT 1 FROM face_observations AS f
                        WHERE f.asset_id = ma.asset_id AND f.identity_id = ?
                    )
                )
                ORDER BY ma.memory_id
                """,
                (*candidates, resolved_id, resolved_id),
            ).fetchall()
        return tuple(_row_text(row, "memory_id") for row in rows)


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


def _user_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


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
        connection.executescript(_SCHEMA_CURRENT)
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise


def _validate_text_selector_schema(connection: sqlite3.Connection) -> None:
    """Reject a partial or incompatible selector projection before accepting schema 18."""
    expected_columns = {
        "embedding_text_selectors": (
            ("embedding_id", "TEXT", 1, 1),
            ("selector_position", "INTEGER", 1, 2),
            ("parent_content_sha256", "TEXT", 1, 0),
            ("embedding_input_sha256", "TEXT", 1, 0),
            ("recipe_version", "TEXT", 1, 0),
        ),
        "embedding_text_span_pieces": (
            ("embedding_id", "TEXT", 1, 1),
            ("selector_position", "INTEGER", 1, 2),
            ("piece_position", "INTEGER", 1, 3),
            ("role", "TEXT", 1, 0),
            ("start_codepoint", "INTEGER", 1, 0),
            ("end_codepoint", "INTEGER", 1, 0),
            ("piece_sha256", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise UnsupportedSchemaError(f"local schema has an invalid {table} table")

    selector_foreign_keys = tuple(
        tuple(row[index] for index in range(8))
        for row in connection.execute("PRAGMA foreign_key_list(embedding_text_selectors)")
    )
    if selector_foreign_keys != (
        (0, 0, "embeddings", "embedding_id", "embedding_id", "NO ACTION", "CASCADE", "NONE"),
    ):
        raise UnsupportedSchemaError(
            "local schema has invalid foreign keys on embedding_text_selectors"
        )
    piece_foreign_keys = tuple(
        tuple(row[index] for index in range(8))
        for row in connection.execute("PRAGMA foreign_key_list(embedding_text_span_pieces)")
    )
    if piece_foreign_keys != (
        (
            0,
            0,
            "embedding_text_selectors",
            "embedding_id",
            "embedding_id",
            "NO ACTION",
            "CASCADE",
            "NONE",
        ),
        (
            0,
            1,
            "embedding_text_selectors",
            "selector_position",
            "selector_position",
            "NO ACTION",
            "CASCADE",
            "NONE",
        ),
    ):
        raise UnsupportedSchemaError(
            "local schema has invalid foreign keys on embedding_text_span_pieces"
        )

    indexes = {
        str(row[1]): (bool(row[2]), str(row[3]), bool(row[4]))
        for row in connection.execute("PRAGMA index_list(embedding_text_span_pieces)")
        if row[1] is not None
    }
    if indexes.get("embedding_text_span_pieces_embedding_idx") != (False, "c", False):
        raise UnsupportedSchemaError(
            "local schema has invalid indexes on embedding_text_span_pieces"
        )
    index_columns = tuple(
        str(row[2])
        for row in connection.execute("PRAGMA index_info(embedding_text_span_pieces_embedding_idx)")
    )
    if index_columns != ("embedding_id", "selector_position", "piece_position"):
        raise UnsupportedSchemaError(
            "local schema has invalid indexes on embedding_text_span_pieces"
        )

    required_checks = {
        "embedding_text_selectors": (
            "check (selector_position >= 0)",
            "check ( length(parent_content_sha256) = 64 and parent_content_sha256 not glob '*[^0-9a-f]*' )",
            "check ( length(embedding_input_sha256) = 64 and embedding_input_sha256 not glob '*[^0-9a-f]*' )",
            "check (length(trim(recipe_version)) > 0)",
        ),
        "embedding_text_span_pieces": (
            "check (piece_position >= 0)",
            "check (role in ('context', 'body'))",
            "check (start_codepoint >= 0)",
            "check (end_codepoint > start_codepoint)",
            "check ( length(piece_sha256) = 64 and piece_sha256 not glob '*[^0-9a-f]*' )",
        ),
    }
    for table, checks in required_checks.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        sql = "" if row is None or row[0] is None else " ".join(str(row[0]).lower().split())
        if any(" ".join(check.lower().split()) not in sql for check in checks):
            raise UnsupportedSchemaError(f"local schema has invalid checks on {table}")


def _validate_evidence_clause_schema(connection: sqlite3.Connection) -> None:
    """Reject a partial development schema before migration writes any support rows."""
    expected_columns = {
        "memory_evidence_clauses": (
            "memory_id",
            "clause_id",
            "member_count",
            "confidence",
            "recorded_at",
            "retired_at",
        ),
        "memory_evidence_clause_members": (
            "memory_id",
            "clause_id",
            "source_memory_id",
            "position",
        ),
        "memory_evidence_clause_versions": (
            "memory_id",
            "clause_id",
            "version",
            "confidence",
            "recorded_at",
            "retired_at",
            "restores_version",
        ),
    }
    for table, expected in expected_columns.items():
        info = connection.execute(f"PRAGMA table_info({table})").fetchall()
        actual = tuple(str(row[1]) for row in info)
        if actual != expected:
            raise UnsupportedSchemaError(f"local schema has an invalid {table} table")
        expected_primary_key = {
            "memory_evidence_clauses": (1, 2, 0, 0, 0, 0),
            "memory_evidence_clause_members": (1, 2, 3, 0),
            "memory_evidence_clause_versions": (1, 2, 3, 0, 0, 0, 0),
        }[table]
        if tuple(int(row[5]) for row in info) != expected_primary_key:
            raise UnsupportedSchemaError(f"local schema has an invalid {table} primary key")

    member_foreign_keys = connection.execute(
        "PRAGMA foreign_key_list(memory_evidence_clause_members)"
    ).fetchall()
    version_foreign_keys = connection.execute(
        "PRAGMA foreign_key_list(memory_evidence_clause_versions)"
    ).fetchall()
    for table, rows in (
        ("memory_evidence_clause_members", member_foreign_keys),
        ("memory_evidence_clause_versions", version_foreign_keys),
    ):
        target_pairs = {(str(row[3]), str(row[4])) for row in rows}
        if {str(row[2]) for row in rows} != {"memory_evidence_clauses"} or target_pairs != {
            ("memory_id", "memory_id"),
            ("clause_id", "clause_id"),
        }:
            raise UnsupportedSchemaError(f"local schema has invalid foreign keys on {table}")
    if any(str(row[3]) == "source_memory_id" for row in member_foreign_keys):
        raise UnsupportedSchemaError("local schema must not cascade evidence clause source members")
    expected_indexes = {
        "memory_evidence_clauses": {"memory_evidence_clauses_current_idx": (False, False)},
        "memory_evidence_clause_members": {
            "memory_evidence_clause_members_source_idx": (False, False)
        },
        "memory_evidence_clause_versions": {
            "memory_evidence_clause_versions_current_idx": (True, True)
        },
    }
    for table, expected_index_shapes in expected_indexes.items():
        indexes = {
            str(row[1]): (bool(row[2]), bool(row[4]))
            for row in connection.execute(f"PRAGMA index_list({table})")
            if row[1] is not None
        }
        if any(indexes.get(name) != shape for name, shape in expected_index_shapes.items()):
            raise UnsupportedSchemaError(f"local schema has invalid indexes on {table}")


# The kernel derives a naming assertion's IDs in `memory.py`; importing it here would invert the
# dependency, so the two payloads are restated for the migration and pinned by a contract test
# that registers the same name through the kernel and compares the IDs it mints.
_NAMING_RECIPE = "mindbridge-identity-naming-v1"
_NAMING_PREDICATE = "identity"


def _naming_assertion_ids(
    identity_id: str,
    name: str,
    relationship: str | None,
) -> tuple[str, str]:
    """Return the `(memory_id, lineage_id)` the kernel mints for one host naming assertion."""
    lineage = _canonical_json(
        {
            "kind": MemoryKind.ENTITY.value,
            "predicate": _NAMING_PREDICATE,
            "frame_id": None,
            "anchor": None,
            "subject": None,
            "identity_id": identity_id,
        }
    )
    formation = _canonical_json(
        {
            "recipe": _NAMING_RECIPE,
            "kind": MemoryKind.ENTITY.value,
            "identity_id": identity_id,
            "subject": _canonical_subject(name),
            "predicate": _NAMING_PREDICATE,
            "value": _canonical_subject(relationship),
            "assertion_basis": EvidenceBasis.USER_STATEMENT.value,
            "cue_modality": None,
            "episode_source": None,
            "content": None,
            "valid_from": None,
            "valid_until": None,
            "spatial": None,
        }
    )
    return (
        hashlib.sha256(f"mindbridge-formation-v1:{formation}".encode()).hexdigest(),
        hashlib.sha256(f"mindbridge-lineage-v1:{lineage}".encode()).hexdigest(),
    )


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_memory_context(
    connection: sqlite3.Connection,
    memory_id: str,
    context: MemoryContext,
    *,
    transaction_memory_ids: set[str] | None = None,
    superseded: list[tuple[str, int]] | None = None,
    write_evidence: bool = True,
    recorded_at: datetime | None = None,
) -> None:
    context_recorded_at = context.recorded_at if recorded_at is None else recorded_at
    existing = connection.execute(
        "SELECT lineage_id FROM memory_semantics WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if existing is not None and write_evidence:
        for source_memory_id in context.evidence_ids:
            _add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=context.confidence,
                recorded_at=context_recorded_at,
            )
        if not _version_retired(connection, memory_id):
            return
        # Every version of this claim is retired: it was superseded or rolled back, and asserting
        # it again is a fresh claim rather than a repeat of a standing one. Restating it as the
        # next version reconciles the lineage the same way a first assertion does, so renaming a
        # person back, or re-registering a name after `rollback`, lands instead of doing nothing.

    lineage_id = (
        _row_text(existing, "lineage_id")
        if existing is not None
        else context.lineage_id or memory_id
    )
    recorded_at = _next_lineage_transaction_time(
        connection,
        context_recorded_at,
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
        INSERT OR IGNORE INTO memory_semantics (
            memory_id, lineage_id, kind, basis, source_id,
            subject, predicate, value, model_id, recipe, identity_id,
            cue_modality, valence, arousal,
            spatial_frame_id, spatial_anchor, spatial_x, spatial_y, spatial_z,
            spatial_qx, spatial_qy, spatial_qz, spatial_qw, spatial_uncertainty_m
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
            context.identity_id,
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
    if existing is None and write_evidence:
        for source_memory_id in context.evidence_ids:
            _add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=context.confidence,
                recorded_at=recorded_at,
            )

    # Explicit formation clauses are inserted immediately after this semantic row because their
    # parent FK requires it to exist first. Evaluate the pending clause as one assessment here so
    # a STATE performs its lineage reconciliation in the same atomic formation transaction. The
    # later clause insertion recomputes the same projection from durable rows. A joint witness is
    # still one assessment, so it does not clear the two-group TRAIT/naming threshold.
    evidence_count = (
        1
        if not write_evidence and context.evidence_ids
        else _evidence_summary(connection, memory_id)[0]
    )
    # Decided against the lineage as it stands, before this write retires anything: an assertion
    # nobody can see must not be what displaces the standing one, and the explicit statement that
    # suppresses a guess only counts while it is still unretired.
    visible = _semantic_visibility(
        connection,
        memory_id=memory_id,
        lineage_id=lineage_id,
        kind=context.kind.value,
        basis=context.basis.value,
        identity_id=context.identity_id,
        evidence_count=evidence_count,
        valid_from=valid_from,
        valid_until=valid_until,
    )

    supersedes_id = context.supersedes_id
    # A state changes, an asserted trait replaces the last one, and naming a person supersedes
    # whatever they were called before -- so the retracted name stops answering to active reads
    # instead of sitting beside the current one. Everything else accumulates evidence instead,
    # as does a claim that is not visible: it waits beside the standing one for corroboration.
    reconcile_lineage = visible and (
        context.kind is MemoryKind.STATE
        or (context.kind is MemoryKind.TRAIT and context.basis is EvidenceBasis.USER_STATEMENT)
        or (context.kind is MemoryKind.ENTITY and context.identity_id is not None)
    )
    if reconcile_lineage:
        recorded_at, supersedes_id = _retire_displaced_lineage(
            connection,
            memory_id,
            lineage_id=lineage_id,
            kind=context.kind.value,
            recorded_at=recorded_at,
            supersedes_id=supersedes_id,
            valid_from=valid_from,
            valid_until=valid_until,
            transaction_memory_ids=transaction_memory_ids,
            superseded=superseded,
        )

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
            context.confidence,
            _optional_datetime_text(valid_from),
            _optional_datetime_text(valid_until),
            _datetime_text(recorded_at),
            int(visible),
            supersedes_id,
        ),
    )


def _retire_displaced_lineage(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    lineage_id: str,
    kind: str,
    recorded_at: datetime,
    supersedes_id: str | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
    transaction_memory_ids: set[str] | None,
    superseded: list[tuple[str, int]] | None,
) -> tuple[datetime, str | None]:
    """Retire every standing version this claim displaces; return its time and what it replaced.

    A version whose validity only partly overlaps is carried forward over the interval that
    survives, so retiring the whole of it never silently drops the part the new claim says
    nothing about.
    """
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
        (lineage_id, kind, memory_id),
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
        ) or not _intervals_overlap(valid_from, valid_until, old_from, old_until):
            continue
        tx_time = max(recorded_at, old_recorded + timedelta(microseconds=1))
        recorded_at = tx_time
        connection.execute(
            """
            UPDATE memory_versions
            SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (_datetime_text(tx_time), _row_text(old, "memory_id"), int(old["version"])),
        )
        if superseded is not None:
            superseded.append((_row_text(old, "memory_id"), int(old["version"])))
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
    return recorded_at, supersedes_id


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


def _memory_evidence_linked(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
) -> bool:
    """Return whether this record already cites this source in an unretired evidence row."""
    return (
        connection.execute(
            """
            SELECT 1 FROM memory_evidence
            WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
            """,
            (memory_id, source_memory_id),
        ).fetchone()
        is not None
    )


def _evidence_clause_id(source_memory_ids: Sequence[str]) -> str:
    """Stable identity for one conjunction; caller preserves declaration order separately."""
    canonical = tuple(sorted(source_memory_ids))
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"mindbridge-evidence-clause-v1:{payload}".encode()).hexdigest()


def _add_evidence_clause(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_ids: Sequence[str],
    *,
    confidence: float,
    recorded_at: datetime,
) -> StoredEvidenceClauseChange | None:
    """Persist one complete support conjunction and project its members into flat evidence."""
    members = tuple(source_memory_ids)
    if not members or len(set(members)) != len(members):
        raise ValueError("evidence clause members must be non-empty and unique")
    for source_memory_id in members:
        _require_identifier(source_memory_id, "source_memory_id")
    clause_id = _evidence_clause_id(members)
    row = connection.execute(
        """
        SELECT confidence, recorded_at, retired_at FROM memory_evidence_clauses
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is not None and row["retired_at"] is None and float(row["confidence"]) == confidence:
        return None
    tx_time = _next_clause_transaction_time(connection, memory_id, clause_id, recorded_at)
    previous_active = row is not None and row["retired_at"] is None
    previous_confidence = None if row is None else float(row["confidence"])
    version_row = connection.execute(
        """
        SELECT COALESCE(MAX(version), 0) AS version
        FROM memory_evidence_clause_versions
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if version_row is None:
        raise RuntimeError("failed to allocate an evidence clause version")
    version = int(version_row["version"]) + 1
    if previous_active:
        connection.execute(
            """
            UPDATE memory_evidence_clause_versions SET retired_at = ?
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (_datetime_text(tx_time), memory_id, clause_id),
        )
    if row is None:
        connection.execute(
            """
            INSERT INTO memory_evidence_clauses (
                memory_id, clause_id, member_count, confidence, recorded_at, retired_at
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (memory_id, clause_id, len(members), confidence, _datetime_text(tx_time)),
        )
        connection.executemany(
            """
            INSERT INTO memory_evidence_clause_members (
                memory_id, clause_id, source_memory_id, position
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (memory_id, clause_id, source_memory_id, position)
                for position, source_memory_id in enumerate(members)
            ),
        )
    else:
        connection.execute(
            """
            UPDATE memory_evidence_clauses
            SET retired_at = NULL, confidence = ?, recorded_at = ?
            WHERE memory_id = ? AND clause_id = ?
            """,
            (confidence, _datetime_text(tx_time), memory_id, clause_id),
        )
    connection.execute(
        """
        INSERT INTO memory_evidence_clause_versions (
            memory_id, clause_id, version, confidence, recorded_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, NULL)
        """,
        (memory_id, clause_id, version, confidence, _datetime_text(tx_time)),
    )
    _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
    _refresh_evidence_projection(connection, memory_id, tx_time)
    _restamp_dependent_evidence(connection, (memory_id,), tx_time)
    return StoredEvidenceClauseChange(
        memory_id=memory_id,
        clause_id=clause_id,
        previous_active=previous_active,
        previous_confidence=previous_confidence,
        applied_confidence=confidence,
        applied_recorded_at=tx_time,
        applied_version=version,
    )


def _next_clause_transaction_time(
    connection: sqlite3.Connection,
    memory_id: str,
    clause_id: str,
    proposed: datetime,
) -> datetime:
    """Return a time strictly after this clause's latest state transition."""
    tx_time = _next_semantic_transaction_time(connection, proposed, (memory_id,))
    row = connection.execute(
        """
        SELECT MAX(recorded_at) AS recorded_at, MAX(retired_at) AS retired_at
        FROM memory_evidence_clause_versions
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is None:
        return tx_time
    for field in ("recorded_at", "retired_at"):
        value = _optional_datetime_from_row(row, field)
        if value is not None:
            tx_time = max(tx_time, value + timedelta(microseconds=1))
    return tx_time


def _rebuild_flat_evidence_from_clauses(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    changed_at: datetime,
) -> None:
    """Make legacy flat links the exact union of complete active support clauses."""
    rows = connection.execute(
        """
        SELECT m.source_memory_id, MAX(c.confidence) AS confidence
        FROM memory_evidence_clauses AS c
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
        WHERE c.memory_id = ? AND c.retired_at IS NULL
        GROUP BY m.source_memory_id
        HAVING COUNT(*) > 0
        ORDER BY MIN(m.position), m.source_memory_id
        """,
        (memory_id,),
    ).fetchall()
    wanted = {_row_text(row, "source_memory_id"): float(row["confidence"]) for row in rows}
    active = {
        _row_text(row, "source_memory_id")
        for row in connection.execute(
            """
            SELECT source_memory_id FROM memory_evidence
            WHERE memory_id = ? AND retired_at IS NULL
            """,
            (memory_id,),
        ).fetchall()
    }
    retired = active - set(wanted)
    if retired:
        connection.execute(
            f"""
            UPDATE memory_evidence SET retired_at = ?
            WHERE memory_id = ? AND retired_at IS NULL
              AND source_memory_id IN ({", ".join("?" for _ in retired)})
            """,
            (_datetime_text(changed_at), memory_id, *sorted(retired)),
        )
    for source_memory_id, confidence in wanted.items():
        if source_memory_id not in active:
            _insert_memory_evidence(
                connection,
                memory_id,
                source_memory_id,
                confidence=confidence,
                recorded_at=changed_at,
            )
        else:
            connection.execute(
                """
                UPDATE memory_evidence SET confidence = ?
                WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
                """,
                (confidence, memory_id, source_memory_id),
            )


def _retire_evidence_clauses_for_source(
    connection: sqlite3.Connection,
    source_memory_id: str,
    *,
    retired_at: datetime,
) -> dict[str, datetime]:
    """Retire every conjunction containing a withdrawn source; never shorten it."""
    rows = connection.execute(
        """
        SELECT c.memory_id, c.clause_id
        FROM memory_evidence_clauses AS c
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
        WHERE m.source_memory_id = ? AND c.retired_at IS NULL
        ORDER BY c.memory_id, c.clause_id
        """,
        (source_memory_id,),
    ).fetchall()
    if not rows:
        return {}
    changed: dict[str, datetime] = {}
    for row in rows:
        memory_id = _row_text(row, "memory_id")
        changed_at = _retire_active_evidence_clause(
            connection,
            memory_id,
            _row_text(row, "clause_id"),
            retired_at=retired_at,
        )
        if changed_at is not None:
            changed[memory_id] = max(changed.get(memory_id, changed_at), changed_at)
    for memory_id, changed_at in changed.items():
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=changed_at)
    return changed


def _active_evidence_dependent_closure(
    connection: sqlite3.Connection,
    source_memory_id: str,
) -> tuple[str, ...]:
    """Return active derived dependents reachable from one source before it is withdrawn."""
    closure: dict[str, None] = {}
    frontier = [source_memory_id]
    while frontier:
        source_id = frontier.pop()
        for row in connection.execute(
            """
            SELECT DISTINCT c.memory_id
            FROM memory_evidence_clauses AS c
            JOIN memory_evidence_clause_members AS m
              ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
            WHERE m.source_memory_id = ? AND c.retired_at IS NULL
            ORDER BY c.memory_id
            """,
            (source_id,),
        ).fetchall():
            dependent_id = _row_text(row, "memory_id")
            if dependent_id == source_memory_id or dependent_id in closure:
                continue
            closure[dependent_id] = None
            frontier.append(dependent_id)
    return tuple(closure)


def _deletion_cascade(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return existing roots followed by unsupported records in their reverse closure."""
    selected = tuple(
        memory_id
        for memory_id in dict.fromkeys(memory_ids)
        if connection.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        is not None
    )
    selected_set = set(selected)
    affected: dict[str, None] = {}
    for memory_id in selected:
        for dependent_id in _active_evidence_dependent_closure(connection, memory_id):
            if dependent_id not in selected_set:
                affected.setdefault(dependent_id, None)
    grounded = _grounded_affected_memory_ids(
        connection,
        tuple(affected),
        excluding=selected,
    )
    return (
        *selected,
        *(memory_id for memory_id in affected if memory_id not in grounded),
    )


def _grounded_affected_memory_ids(  # noqa: C901 - bounded support fixed point
    connection: sqlite3.Connection,
    affected: Sequence[str],
    *,
    excluding: Sequence[str],
) -> frozenset[str]:
    """Solve support only inside one withdrawal's dependent subgraph.

    Existing visible support outside the subgraph is unchanged and therefore acts as a seed.
    Inside it, observations and host assertions seed themselves; every other record needs one
    complete clause whose members are already grounded. A mutual-support cycle cannot seed itself.
    """
    affected_set = set(affected)
    excluded = set(excluding)
    grounded = {
        memory_id
        for memory_id in affected
        if memory_id not in excluded and _is_evidence_root(connection, memory_id)
    }
    clauses: dict[str, dict[str, set[str]]] = {}
    for memory_id in affected:
        if memory_id in excluded:
            continue
        for row in connection.execute(
            """
            SELECT c.clause_id, m.source_memory_id
            FROM memory_evidence_clauses AS c
            JOIN memory_evidence_clause_members AS m
              ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
            WHERE c.memory_id = ? AND c.retired_at IS NULL
            ORDER BY c.clause_id, m.position
            """,
            (memory_id,),
        ).fetchall():
            clauses.setdefault(memory_id, {}).setdefault(_row_text(row, "clause_id"), set()).add(
                _row_text(row, "source_memory_id")
            )
    outside_grounded: dict[str, bool] = {}

    def member_is_grounded(source_id: str) -> bool:
        if source_id in excluded:
            return False
        if source_id in affected_set:
            return source_id in grounded
        if source_id not in outside_grounded:
            # A record outside the withdrawal's subgraph keeps whatever support it had: a hidden
            # trait still below its visibility threshold is supported, not unsupported, so it
            # must keep grounding the alternatives that cite it.
            outside_grounded[source_id] = _is_evidence_root(connection, source_id) or (
                connection.execute(
                    """
                    SELECT 1 FROM memory_versions AS v
                    WHERE v.memory_id = ? AND v.retired_at IS NULL
                      AND (
                        v.visible = 1
                        OR EXISTS (
                            SELECT 1 FROM memory_evidence_clauses AS c
                            WHERE c.memory_id = v.memory_id AND c.retired_at IS NULL
                        )
                      )
                    """,
                    (source_id,),
                ).fetchone()
                is not None
            )
        return outside_grounded[source_id]

    moved = True
    while moved:
        moved = False
        for memory_id, alternatives in clauses.items():
            if memory_id in grounded:
                continue
            if any(
                members and all(member_is_grounded(source_id) for source_id in members)
                for members in alternatives.values()
            ):
                grounded.add(memory_id)
                moved = True
    return frozenset(grounded)


def _is_evidence_root(connection: sqlite3.Connection, memory_id: str) -> bool:
    """Return whether a record stands without another memory's support."""
    row = connection.execute(
        """
        SELECT s.kind, s.basis
        FROM memory_records AS r
        LEFT JOIN memory_semantics AS s ON s.memory_id = r.memory_id
        WHERE r.memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        return False
    return (
        row["kind"] is None
        or _row_text(row, "kind") == MemoryKind.OBSERVATION.value
        or _row_text(row, "basis")
        in {EvidenceBasis.USER_STATEMENT.value, EvidenceBasis.RESPONSE_FEEDBACK.value}
    )


def _retire_active_evidence_clause(
    connection: sqlite3.Connection,
    memory_id: str,
    clause_id: str,
    *,
    retired_at: datetime,
) -> datetime | None:
    """Close one current clause interval and mirror that state on the clause projection."""
    row = connection.execute(
        """
        SELECT 1 FROM memory_evidence_clauses
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is None:
        return None
    tx_time = _next_clause_transaction_time(connection, memory_id, clause_id, retired_at)
    connection.execute(
        """
        UPDATE memory_evidence_clause_versions SET retired_at = ?
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (_datetime_text(tx_time), memory_id, clause_id),
    )
    connection.execute(
        """
        UPDATE memory_evidence_clauses SET retired_at = ?
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (_datetime_text(tx_time), memory_id, clause_id),
    )
    return tx_time


# Independence is counted per capture: a raw observation resolves to the `source_id` its
# `ObservationContext` carried, falling back to the observation record itself when the caller
# supplied none. A derived source (an affect cue, say) inherits the group its own evidence
# already resolves to, so two observations of one capture -- and every cue formed from them --
# stay one group and cannot corroborate each other into a visible trait. A source resolving to
# several groups keeps its own identity: nothing that traces back to a single capture can then
# reach two groups. `MIN = MAX` says "exactly one distinct group" without the temp B-tree a
# `COUNT(DISTINCT ...)` would build, and holds because `source_group_id` is `TEXT NOT NULL`.
_SOURCE_GROUP_QUERY = """
    SELECT COALESCE(
        (
            SELECT MIN(d.source_group_id)
            FROM memory_evidence AS d
            WHERE d.memory_id = r.memory_id AND d.retired_at IS NULL
            HAVING MIN(d.source_group_id) = MAX(d.source_group_id)
        ),
        s.source_id,
        r.memory_id
    ) AS source_group_id
    FROM memory_records AS r
    LEFT JOIN memory_semantics AS s
      ON s.memory_id = r.memory_id AND s.kind = 'observation'
    WHERE r.memory_id = ?
"""


def _restamp_dependent_evidence(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    changed_at: datetime,
) -> None:
    """Re-resolve the inherited group on every row citing a record whose own evidence moved.

    `source_group_id` is copied from the cited source when the citing row is written, so
    reinforcing, rolling back, or cascading a delete through that source later leaves the rows
    that cite it stamped with a group the source no longer resolves to -- and a dependent trait
    keeps counting corroboration that is gone. Restamp the whole reachable citation subgraph in
    the same transaction and reproject what changed.
    """
    closure: dict[str, None] = {}
    frontier = list(dict.fromkeys(memory_ids))
    while frontier:
        source_memory_id = frontier.pop()
        if source_memory_id in closure:
            continue
        closure[source_memory_id] = None
        frontier.extend(
            _row_text(row, "memory_id")
            for row in connection.execute(
                """
                SELECT memory_id FROM memory_evidence
                WHERE source_memory_id = ? AND retired_at IS NULL
                """,
                (source_memory_id,),
            ).fetchall()
        )
    # A sweep in arbitrary order can restamp a record before one of its own sources settles, so
    # sweep until nothing moves. The citation graph is acyclic, so each sweep settles at least
    # one more record and the bound is only there to stop a corrupted cycle from spinning.
    for _sweep in range(len(closure)):
        moved = False
        for source_memory_id in closure:
            row = connection.execute(_SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
            if row is None:
                continue
            group = _row_text(row, "source_group_id")
            dependent_ids = tuple(
                dict.fromkeys(
                    _row_text(dependent, "memory_id")
                    for dependent in connection.execute(
                        """
                        SELECT memory_id FROM memory_evidence
                        WHERE source_memory_id = ?
                          AND retired_at IS NULL AND source_group_id <> ?
                        """,
                        (source_memory_id, group),
                    ).fetchall()
                )
            )
            if not dependent_ids:
                continue
            moved = True
            connection.execute(
                """
                UPDATE memory_evidence SET source_group_id = ?
                WHERE source_memory_id = ? AND retired_at IS NULL
                """,
                (group, source_memory_id),
            )
            for dependent_id in dependent_ids:
                _refresh_evidence_projection(
                    connection,
                    dependent_id,
                    _next_semantic_transaction_time(connection, changed_at, (dependent_id,)),
                )
        if not moved:
            break


def _insert_memory_evidence(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
    *,
    confidence: float,
    recorded_at: datetime,
) -> bool:
    if _memory_evidence_linked(connection, memory_id, source_memory_id):
        return False
    source = connection.execute(_SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
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
    # This record's own active evidence just changed, so whatever cites it may now inherit a
    # different group.
    _restamp_dependent_evidence(connection, (memory_id,), recorded_at)
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


def _version_retired(connection: sqlite3.Connection, memory_id: str) -> bool:
    """Return whether a typed record exists whose every version has been retired."""
    row = connection.execute(
        """
        SELECT COUNT(*) AS total, COUNT(retired_at) AS retired
        FROM memory_versions WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    return row is not None and bool(int(row["total"])) and int(row["total"]) == int(row["retired"])


def _require_unretired(connection: sqlite3.Connection, memory_ids: Sequence[str]) -> None:
    """Fail the transaction unless every named memory's current version still stands.

    Deliberately narrower than `_require_active`: it says nothing about `forgotten_at`, and it
    never looks at `visible`, because a hidden inferred `TRAIT` is a legitimate `REINFORCE`
    target. A record with no typed version at all -- a plain observation -- passes.
    """
    for memory_id in dict.fromkeys(memory_ids):
        _require_identifier(memory_id, "memory_id")
        if _version_retired(connection, memory_id):
            raise StaleOperationError(f"{memory_id} has no current version")


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
    memory_ids: Sequence[str | tuple[str, int]],
    *,
    recorded_at: datetime,
) -> None:
    """Carry a retired version forward again, by memory ID or by exact `(id, version)` pair.

    A `CORRECT` retires whatever version was current, so naming the record is enough. A lineage
    supersession may also have split the record's remaining validity into carried versions, so
    the operation row names the exact version it retired and that one is restored.
    """
    for entry in dict.fromkeys(memory_ids):
        memory_id, version = (entry, None) if isinstance(entry, str) else entry
        row = connection.execute(
            f"""
            SELECT memory_id, version, confidence, valid_from, valid_until,
                   recorded_at, visible, supersedes_id
            FROM memory_versions
            WHERE memory_id = ?{"" if version is None else " AND version = ?"}
            ORDER BY version DESC LIMIT 1
            """,
            (memory_id,) if version is None else (memory_id, version),
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


def _displaced_naming_versions(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[str, ...]:
    """Return what this record's in-force naming assertion displaced, if it is one.

    Deleting the assertion that renamed somebody has to leave the previous name standing, the
    same reversal `rollback_operation` applies -- a bound `ENTITY` row supersedes a lineage
    rather than accumulating beside it, so nothing else brings the predecessor back. A record
    whose version is already retired displaced nothing that deleting it can restore, and a
    first assertion supersedes nothing, so a retracted name never returns on its own.
    """
    return tuple(
        _row_text(row, "supersedes_id")
        for row in connection.execute(
            """
            SELECT v.supersedes_id
            FROM memory_versions AS v
            JOIN memory_semantics AS s ON s.memory_id = v.memory_id
            WHERE v.memory_id = ? AND v.retired_at IS NULL AND v.supersedes_id IS NOT NULL
              AND s.kind = ? AND s.identity_id IS NOT NULL
            ORDER BY v.version
            """,
            (memory_id, MemoryKind.ENTITY.value),
        ).fetchall()
    )


def _current_naming_assertion(
    connection: sqlite3.Connection,
    identity_id: str,
    *,
    excluding: Sequence[str] = (),
) -> sqlite3.Row | None:
    """Return the naming assertion `identities.name` currently projects, newest first.

    `excluding` drops records a caller is about to delete, which is how it can rebuild indexed
    text for the projection a deletion is going to leave behind rather than the current one.
    """
    dropped = tuple(dict.fromkeys(excluding))
    placeholders = ", ".join("?" for _memory_id in dropped)
    row: sqlite3.Row | None = connection.execute(
        f"""
        SELECT s.memory_id, s.subject, s.value
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.identity_id = ? AND s.kind = ?
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
          {f"AND s.memory_id NOT IN ({placeholders})" if dropped else ""}
        ORDER BY v.recorded_at DESC, s.memory_id DESC
        LIMIT 1
        """,
        (identity_id, MemoryKind.ENTITY.value, *dropped),
    ).fetchone()
    return row


def _current_consent_state(connection: sqlite3.Connection, identity_id: str) -> str | None:
    """Return the consent state one identity's standing assertion projects, or None.

    Consent is a bound STATE assertion rather than a bound ENTITY one, which is what keeps it
    out of `_current_naming_assertion` above: the two live in separate lineages, so recording
    consent never displaces a name and renaming somebody never disturbs their consent.
    """
    row: sqlite3.Row | None = connection.execute(
        """
        SELECT s.value
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.identity_id = ? AND s.kind = ? AND s.predicate = ? AND s.basis = ?
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
        ORDER BY v.recorded_at DESC, s.memory_id DESC
        LIMIT 1
        """,
        (identity_id, MemoryKind.STATE.value, CONSENT_PREDICATE, _CONSENT_BASIS),
    ).fetchone()
    return None if row is None else _optional_row_text(row, "value")


def _restrained_identities(connection: sqlite3.Connection) -> frozenset[str]:
    """Return every identity whose standing consent assertion restrains the kernel.

    Withheld and withdrawn restrain identically; the distinction between them is audit history,
    not policy. Read fresh on each recognition rather than cached, because a person withdrawing
    consent has to take effect on the next observation, not on the next process start.
    """
    # Ordered oldest-first per identity so the newest standing assertion is the one that
    # survives the dict build, which is the same rule `_current_naming_assertion` applies.
    standing: dict[str, str | None] = {
        _row_text(row, "identity_id"): _optional_row_text(row, "value")
        for row in connection.execute(
            """
            SELECT s.identity_id, s.value
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            JOIN memory_records AS r ON r.memory_id = s.memory_id
            WHERE s.kind = ? AND s.predicate = ? AND s.basis = ? AND s.identity_id IS NOT NULL
              AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
            ORDER BY s.identity_id, v.recorded_at, s.memory_id
            """,
            (MemoryKind.STATE.value, CONSENT_PREDICATE, _CONSENT_BASIS),
        ).fetchall()
    }
    return frozenset(
        identity_id for identity_id, value in standing.items() if value in _RESTRAINING_CONSENT
    )


def _visible_naming_assertions(
    connection: sqlite3.Connection,
    *,
    identity_ids: Sequence[str] | None = None,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> tuple[sqlite3.Row, ...]:
    """Return the identity, subject, and memory id of every currently visible naming assertion.

    `identity_ids` narrows the scan to the people the caller already has in hand, which is what
    a read path asking "is this handful of observed people named" wants; None means everybody.
    """
    query = """
        SELECT s.identity_id, s.subject, s.memory_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.kind = ? AND s.identity_id IS NOT NULL
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
    """
    parameters: list[object] = [MemoryKind.ENTITY.value]
    if known_at is not None:
        known_text = _datetime_text(known_at)
        query = query.replace(
            "AND v.retired_at IS NULL",
            "AND v.recorded_at <= ? AND (v.retired_at IS NULL OR v.retired_at > ?)",
        )
        parameters.extend((known_text, known_text))
    if valid_at is not None:
        valid_text = _datetime_text(valid_at)
        query += " AND (v.valid_from IS NULL OR v.valid_from <= ?) AND (v.valid_until IS NULL OR v.valid_until > ?)"
        parameters.extend((valid_text, valid_text))
    if identity_ids is None:
        return tuple(connection.execute(query, parameters).fetchall())
    wanted = tuple(dict.fromkeys(identity_ids))
    rows: list[sqlite3.Row] = []
    for offset in range(0, len(wanted), _SQLITE_PARAMETER_BATCH):
        batch = wanted[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _identity_id in batch)
        rows.extend(
            connection.execute(
                f"{query} AND s.identity_id IN ({placeholders})",
                (*parameters, *batch),
            ).fetchall()
        )
    return tuple(rows)


def _named_identities(
    connection: sqlite3.Connection,
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> dict[str, tuple[str, str]]:
    """Return, per identity, the `(name, naming_memory_id)` its visible naming assertion gives.

    An identity with no visible assertion, or one whose subject is somehow unset, is simply
    absent -- `named_actors` treats a missing key as "not currently named".
    """
    named: dict[str, tuple[str, str]] = {}
    for row in _visible_naming_assertions(connection, valid_at=valid_at, known_at=known_at):
        name = _optional_row_text(row, "subject")
        if name is not None:
            named[_row_text(row, "identity_id")] = (name, _row_text(row, "memory_id"))
    return named


def _merge_named_identity_links(
    connection: sqlite3.Connection,
    batch: Sequence[str],
    named: Mapping[str, tuple[str, str]],
    links: dict[str, list[tuple[str, str, str]]],
) -> None:
    """Resolve one batch of memories' identity edges and add the named ones to `links`.

    The identity edge is read two ways in one query: the asset-keyed speech speaker or face
    observation `provisional_identities` reads for the unnamed case, and the memory's own
    bound semantic assertion (`memory_semantics.identity_id`). A memory naming itself -- the
    naming assertion's own row -- is skipped, because it already renders as its own actors hit.
    """
    placeholders = ", ".join("?" for _memory_id in batch)
    rows = connection.execute(
        f"""
        SELECT ma.memory_id AS memory_id, sp.speaker_id AS identity_id
        FROM memory_assets AS ma
        JOIN speech_segments AS sp ON sp.asset_id = ma.asset_id
        WHERE ma.memory_id IN ({placeholders}) AND sp.speaker_id IS NOT NULL
        UNION
        SELECT ma.memory_id AS memory_id, f.identity_id AS identity_id
        FROM memory_assets AS ma
        JOIN face_observations AS f ON f.asset_id = ma.asset_id
        WHERE ma.memory_id IN ({placeholders})
        UNION
        SELECT s.memory_id AS memory_id, s.identity_id AS identity_id
        FROM memory_semantics AS s
        WHERE s.memory_id IN ({placeholders}) AND s.identity_id IS NOT NULL
        """,
        (*batch, *batch, *batch),
    ).fetchall()
    for row in rows:
        identity_id = _row_text(row, "identity_id")
        entry = named.get(identity_id)
        if entry is None:
            continue
        memory_id = _row_text(row, "memory_id")
        name, naming_memory_id = entry
        if memory_id == naming_memory_id:
            continue
        bucket = links.setdefault(memory_id, [])
        if not any(identity_id == existing[0] for existing in bucket):
            bucket.append((identity_id, name, naming_memory_id))


def _has_naming_assertion(connection: sqlite3.Connection, identity_id: str) -> bool:
    """Return whether any naming assertion, visible or not, is bound to this identity.

    A merge only recomputes a projection when there is an assertion to project. An identity
    named through the store's own `register_identity`, with no assertion behind it, keeps the
    name the merge plan chose -- which is always the surviving identity's own name.
    """
    return (
        connection.execute(
            "SELECT 1 FROM memory_semantics WHERE identity_id = ? AND kind = ? LIMIT 1",
            (identity_id, MemoryKind.ENTITY.value),
        ).fetchone()
        is not None
    )


def _canonical_subject(subject: str | None) -> str | None:
    """Normalize a semantic subject the one way the whole kernel compares them."""
    if subject is None:
        return None
    canonical = unicodedata.normalize("NFKC", subject).casefold().strip()
    return canonical or None


def _naming_assertion_identities(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return the identities these records name, for the records that name one."""
    ids = tuple(dict.fromkeys(memory_ids))
    found: list[str] = []
    for offset in range(0, len(ids), _SQLITE_PARAMETER_BATCH):
        batch = ids[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        found.extend(
            _row_text(row, "identity_id")
            for row in connection.execute(
                f"""
                SELECT DISTINCT identity_id
                FROM memory_semantics
                WHERE kind = ? AND identity_id IS NOT NULL
                  AND memory_id IN ({placeholders})
                """,
                (MemoryKind.ENTITY.value, *batch),
            ).fetchall()
        )
    return tuple(dict.fromkeys(found))


def _reproject_identities(
    connection: sqlite3.Connection,
    identity_ids: Sequence[str],
) -> tuple[str, ...]:
    """Recompute `identities.name` from each identity's current visible naming assertion.

    This is the one rule behind the projection: whenever a bound naming assertion is written,
    retired, hidden, re-pointed or deleted, the registry is recomputed in the same transaction,
    so `identities.name` is never a name no visible assertion supports. Returns the identities
    whose projection actually moved, which is what a caller has to repaint indexed text for.

    `identities.updated_at` is transaction time and is taken here rather than from the caller.
    Every path that moves the projection carries a semantic time of its own -- a record's
    `recorded_at`, an operation's `applied_at`, a `forgotten_at` a host may deliberately backdate
    -- and none of them says when this row changed. Stamping one of those could put `updated_at`
    before the row's own `created_at`, which the schema refuses.
    """
    now = _datetime_text(datetime.now(timezone.utc))
    changed: list[str] = []
    for identity_id in dict.fromkeys(identity_ids):
        current = connection.execute(
            "SELECT name, relationship FROM identities WHERE identity_id = ?",
            (identity_id,),
        ).fetchone()
        if current is None:
            continue
        row = _current_naming_assertion(connection, identity_id)
        projected = (
            (None, None)
            if row is None
            else (_optional_row_text(row, "subject"), _optional_row_text(row, "value"))
        )
        if projected == (
            _optional_row_text(current, "name"),
            _optional_row_text(current, "relationship"),
        ):
            continue
        connection.execute(
            """
            UPDATE identities
            SET name = ?, relationship = ?, updated_at = ?
            WHERE identity_id = ?
            """,
            (*projected, now, identity_id),
        )
        changed.append(identity_id)
    return tuple(changed)


def _reproject_named_identities(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Reproject every identity named by one of these records. The projection hook."""
    identity_ids = _naming_assertion_identities(connection, memory_ids)
    if not identity_ids:
        return ()
    return _reproject_identities(connection, identity_ids)


def _rebind_unlinked_claims(
    connection: sqlite3.Connection,
    *,
    target: str,
    restored: str,
    modality: Literal["face", "voice"],
) -> tuple[str, ...]:
    """Re-evaluate every claim bound to a person a merge is being undone for.

    A merge made two people one, and claims derived while they were one are bound to the
    survivor. Splitting them re-examines exactly the claims whose evidence involves media that
    just moved back: a claim resting only on media that is now solely the restored person's is
    re-attributed to them, and a claim resting on media the two still share, or on both
    people's media, is unbound. Neither is left attributed to someone it was never about. A
    claim whose evidence never touched the restored person is untouched, and naming assertions
    stay with the survivor, which is the identity that keeps the profile.

    Returns the memories whose binding changed. Nothing here alters indexed text: the binding
    is a semantic column, and the projected name is repainted by the caller's index refresh.
    """
    restored_assets = _identity_asset_ids(connection, restored)
    # A clip that still shows the survivor is not evidence about the restored person alone,
    # even though their modality moved out of it.
    moved_assets = restored_assets - _identity_asset_ids(connection, target)
    if not restored_assets:
        return ()
    changed: list[str] = []
    for row in connection.execute(
        """
        SELECT memory_id FROM memory_semantics
        WHERE identity_id = ? AND kind <> ?
        ORDER BY memory_id
        """,
        (target, MemoryKind.ENTITY.value),
    ).fetchall():
        memory_id = _row_text(row, "memory_id")
        assets = {
            asset_id
            for source_id in _current_evidence_ids(connection, memory_id)
            for asset_id in _memory_asset_ids(connection, source_id)
        }
        if not assets & restored_assets:
            continue
        connection.execute(
            "UPDATE memory_semantics SET identity_id = ? WHERE memory_id = ?",
            (restored if assets <= moved_assets else None, memory_id),
        )
        changed.append(memory_id)
    return tuple(changed)


def _identity_asset_ids(connection: sqlite3.Connection, identity_id: str) -> set[str]:
    """Return every media asset that observed one identity, seen or heard."""
    return {
        _row_text(row, "asset_id")
        for row in connection.execute(
            """
            SELECT DISTINCT asset_id FROM speech_segments WHERE speaker_id = ?
            UNION
            SELECT DISTINCT asset_id FROM face_observations WHERE identity_id = ?
            """,
            (identity_id, identity_id),
        ).fetchall()
    }


def _memory_asset_ids(connection: sqlite3.Connection, memory_id: str) -> tuple[str, ...]:
    """Return the media assets one memory references, in stored order."""
    return tuple(
        _row_text(row, "asset_id")
        for row in connection.execute(
            "SELECT asset_id FROM memory_assets WHERE memory_id = ? ORDER BY position",
            (memory_id,),
        ).fetchall()
    )


def _current_evidence_ids(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[str, ...]:
    """Return the memories currently cited as evidence for one derived record."""
    return tuple(
        _row_text(row, "source_memory_id")
        for row in connection.execute(
            """
            SELECT source_memory_id FROM memory_evidence
            WHERE memory_id = ? AND retired_at IS NULL
            ORDER BY position
            """,
            (memory_id,),
        ).fetchall()
    )


def _retire_memory_evidence(
    connection: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id, source_memory_id in pairs:
        # Control-plane reinforcement has always been a singleton source link. Retire that
        # exact legacy clause, never every alternative that happens to mention the same source.
        tx_time = _retire_active_evidence_clause(
            connection,
            memory_id,
            _evidence_clause_id((source_memory_id,)),
            retired_at=retired_at,
        )
        if tx_time is None:
            continue
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
        changed.append(memory_id)
    for memory_id in dict.fromkeys(changed):
        _refresh_evidence_projection(
            connection,
            memory_id,
            _next_semantic_transaction_time(connection, retired_at, (memory_id,)),
        )
    _restamp_dependent_evidence(connection, tuple(dict.fromkeys(changed)), retired_at)
    return tuple(dict.fromkeys(changed))


def _retire_evidence_clauses(
    connection: sqlite3.Connection,
    clauses: Sequence[tuple[str, str]],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    """Retire precisely the clauses an operation introduced, preserving alternatives."""
    changed: list[str] = []
    for memory_id, clause_id in clauses:
        tx_time = _retire_active_evidence_clause(
            connection,
            memory_id,
            clause_id,
            retired_at=retired_at,
        )
        if tx_time is None:
            continue
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
        changed.append(memory_id)
    for memory_id in dict.fromkeys(changed):
        _refresh_evidence_projection(
            connection,
            memory_id,
            _next_semantic_transaction_time(connection, retired_at, (memory_id,)),
        )
    _restamp_dependent_evidence(connection, tuple(dict.fromkeys(changed)), retired_at)
    return tuple(dict.fromkeys(changed))


def _evidence_clause_changes_are_current(
    connection: sqlite3.Connection,
    changes: Sequence[StoredEvidenceClauseChange],
) -> bool:
    """Require the operation's version or an explicit inverse chain restoring that version.

    Confidence equality is insufficient: an unrelated A -> B -> A sequence is still a later
    mutation. An inverse version records the exact predecessor it restored, so rolling back the
    newer operation makes the older one reversible again without weakening the ABA guard.
    """
    for change in changes:
        # Deleting the output this operation created cascaded its clause and every version of
        # it. `_reverse_evidence_clause_changes` treats that deletion as the complete inverse;
        # the currency guard has to agree, or the operation can never be rolled back.
        if (
            connection.execute(
                """
                SELECT 1 FROM memory_evidence_clauses
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            is None
        ):
            continue
        applied = connection.execute(
            """
            SELECT confidence, recorded_at
            FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND version = ?
            """,
            (change.memory_id, change.clause_id, change.applied_version),
        ).fetchone()
        if (
            applied is None
            or float(applied["confidence"]) != change.applied_confidence
            or _parse_datetime(_row_text(applied, "recorded_at")) != change.applied_recorded_at
        ):
            return False
        current = connection.execute(
            """
            SELECT version, confidence, restores_version
            FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (change.memory_id, change.clause_id),
        ).fetchone()
        if current is None or float(current["confidence"]) != change.applied_confidence:
            return False
        version = int(current["version"])
        visited: set[int] = set()
        while version != change.applied_version:
            if version in visited:
                return False
            visited.add(version)
            row = connection.execute(
                """
                SELECT restores_version FROM memory_evidence_clause_versions
                WHERE memory_id = ? AND clause_id = ? AND version = ?
                """,
                (change.memory_id, change.clause_id, version),
            ).fetchone()
            if row is None or row["restores_version"] is None:
                return False
            version = int(row["restores_version"])
    return True


def _reverse_evidence_clause_changes(
    connection: sqlite3.Connection,
    changes: Sequence[StoredEvidenceClauseChange],
    *,
    reversed_at: datetime,
) -> tuple[str, ...]:
    """Append the inverse clause state at rollback time without rewriting earlier intervals."""
    changed: dict[str, datetime] = {}
    for change in changes:
        # Deleting an output created by this operation cascades its clauses first. That deletion
        # is already the complete inverse for the output and leaves nothing here to restate.
        if (
            connection.execute(
                """
                SELECT 1 FROM memory_evidence_clauses
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            is None
        ):
            continue
        tx_time = _next_clause_transaction_time(
            connection,
            change.memory_id,
            change.clause_id,
            reversed_at,
        )
        current = connection.execute(
            """
            SELECT version FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (change.memory_id, change.clause_id),
        ).fetchone()
        if current is None:
            raise RuntimeError("a reversible clause change requires an active current version")
        connection.execute(
            """
            UPDATE memory_evidence_clause_versions SET retired_at = ?
            WHERE memory_id = ? AND clause_id = ? AND version = ? AND retired_at IS NULL
            """,
            (
                _datetime_text(tx_time),
                change.memory_id,
                change.clause_id,
                int(current["version"]),
            ),
        )
        if change.previous_active:
            if change.previous_confidence is None:
                raise RuntimeError("an active previous clause requires a confidence")
            latest = connection.execute(
                """
                SELECT MAX(version) AS version FROM memory_evidence_clause_versions
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            if latest is None or latest["version"] is None:
                raise RuntimeError("failed to allocate an inverse clause version")
            version = int(latest["version"]) + 1
            connection.execute(
                """
                INSERT INTO memory_evidence_clause_versions (
                    memory_id, clause_id, version, confidence, recorded_at, retired_at,
                    restores_version
                ) VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    change.memory_id,
                    change.clause_id,
                    version,
                    change.previous_confidence,
                    _datetime_text(tx_time),
                    change.applied_version - 1,
                ),
            )
            connection.execute(
                """
                UPDATE memory_evidence_clauses
                SET confidence = ?, recorded_at = ?, retired_at = NULL
                WHERE memory_id = ? AND clause_id = ?
                """,
                (
                    change.previous_confidence,
                    _datetime_text(tx_time),
                    change.memory_id,
                    change.clause_id,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE memory_evidence_clauses SET retired_at = ?
                WHERE memory_id = ? AND clause_id = ?
                """,
                (_datetime_text(tx_time), change.memory_id, change.clause_id),
            )
        changed[change.memory_id] = max(changed.get(change.memory_id, tx_time), tx_time)
    for memory_id, changed_at in changed.items():
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=changed_at)
        _refresh_evidence_projection(connection, memory_id, changed_at)
    _restamp_dependent_evidence(connection, tuple(changed), reversed_at)
    return tuple(changed)


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


def _weighed_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when anything last weighed it: an operation or a deliberation.

    Both halves are needed. An applied operation proves the record was considered *and* acted
    on; a deliberation row proves only that it was considered, which is exactly the case an
    operation log cannot record -- the backend proposed nothing, or the kernel refused
    everything. Without the second half a zero-yield candidate returns every round and is paid
    for every round.
    """
    return _merge_weighed(
        _consumed_at_by_memory(connection),
        _deliberated_at_by_memory(connection),
    )


def _merge_weighed(
    consumed: dict[str, datetime],
    deliberated: Mapping[str, datetime],
) -> dict[str, datetime]:
    """Keep the newer of the two marks per memory, mutating and returning the first."""
    for memory_id, deliberated_at in deliberated.items():
        previous = consumed.get(memory_id)
        if previous is None or deliberated_at > previous:
            consumed[memory_id] = deliberated_at
    return consumed


def _deliberated_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when a deliberation last had it in its evidence set."""
    return {
        _row_text(row, "memory_id"): _parse_datetime(_row_text(row, "weighed_at"))
        for row in connection.execute(
            """
            SELECT m.memory_id AS memory_id, MAX(d.weighed_at) AS weighed_at
            FROM memory_deliberation_memories AS m
            JOIN memory_deliberations AS d ON d.deliberation_id = m.deliberation_id
            GROUP BY m.memory_id
            """
        ).fetchall()
    }


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
    """Derived records that gained independent evidence nothing has weighed yet."""
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
    weighed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Lineages whose current visible claims disagree, the way the compiler reports them.

    Retiring one side through `CORRECT` is what clears a contradiction, but a deliberation the
    model could not resolve clears nothing -- so the lineage is dropped while nothing about it
    has changed since it was last weighed, and returns as soon as a further claim is recorded in
    it. The signal is the newest transaction time among the disagreeing claims.
    """
    entries: dict[str, list[tuple[str, str, datetime]]] = {}
    # A correlated overlap predicate made SQLite re-discover the same conflicting peers for each
    # member (and then do it again while hydrating members).  Claims in a lineage are instead
    # consumed together.  The two sweeps retain only the best two endpoints for distinct values:
    # an interval participates iff an earlier or later interval with another value overlaps it.
    rows = connection.execute(
        f"""
        SELECT s.lineage_id AS lineage_id, s.memory_id AS memory_id, s.value AS value,
               v.recorded_at AS recorded_at, v.valid_from AS valid_from,
               v.valid_until AS valid_until
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
          AND s.lineage_id IS NOT NULL AND s.value IS NOT NULL AND {_FUNCTIONAL_CLAIM_SQL}
        ORDER BY s.lineage_id, s.memory_id
        """,
        _FUNCTIONAL_CLAIM_PARAMETERS,
    )
    for lineage_id, lineage_rows in groupby(rows, key=lambda row: _row_text(row, "lineage_id")):
        members = _overlapping_functional_members(lineage_rows)
        if members:
            entries[lineage_id] = members
            # The old query applies the lineage limit before due-state filtering.  Stop here so a
            # previously weighed earlier lineage continues to consume a slot in exactly the same way.
            if len(entries) == limit:
                break
    candidates: list[StoredCandidate] = []
    for lineage_entries in entries.values():
        memory_ids = tuple(entry[0] for entry in lineage_entries)
        signal_at = max(entry[2] for entry in lineage_entries)
        if _is_due(signal_at, memory_ids, weighed):
            candidates.append(
                StoredCandidate(
                    trigger="contradiction",
                    memory_ids=memory_ids,
                    evidence_count=len({entry[1] for entry in lineage_entries}),
                )
            )
    return candidates


def _overlapping_functional_members(
    rows: Iterable[sqlite3.Row],
) -> list[tuple[str, str, datetime]]:
    """Return only members with a differently-valued half-open overlap in one lineage."""
    intervals = [
        (
            _row_text(row, "memory_id"),
            _row_text(row, "value"),
            _parse_datetime(_row_text(row, "recorded_at")),
            _optional_datetime_from_row(row, "valid_from"),
            _optional_datetime_from_row(row, "valid_until"),
        )
        for row in rows
    ]
    intervals.sort(key=lambda entry: (entry[3] is not None, entry[3], entry[0]))
    overlapping_ids: set[str] = set()

    # Each entry is (value, endpoint).  An endpoint of None means unbounded: -infinity for a
    # start and +infinity for an end.  Keeping two different values is sufficient because a
    # query excludes only its own value; a discarded value cannot become extremal without a later
    # row of that value re-entering the two slots.
    latest_ends: list[tuple[str, datetime | None]] = []
    for memory_id, value, _recorded_at, valid_from, valid_until in intervals:
        other_end = _other_extremum(latest_ends, value)
        if other_end is not None and _end_after(other_end[1], valid_from):
            overlapping_ids.add(memory_id)
        _update_endpoint_extrema(latest_ends, value, valid_until, latest=True)

    earliest_starts: list[tuple[str, datetime | None]] = []
    for memory_id, value, _recorded_at, valid_from, valid_until in reversed(intervals):
        other_start = _other_extremum(earliest_starts, value)
        if other_start is not None and _end_after(valid_until, other_start[1]):
            overlapping_ids.add(memory_id)
        _update_endpoint_extrema(earliest_starts, value, valid_from, latest=False)

    members = [
        (memory_id, value, recorded_at)
        for memory_id, value, recorded_at, _valid_from, _valid_until in intervals
        if memory_id in overlapping_ids
    ]
    return sorted(members, key=lambda entry: entry[0])


def _other_extremum(
    extrema: Sequence[tuple[str, datetime | None]], value: str
) -> tuple[str, datetime | None] | None:
    for other_value, endpoint in extrema:
        if other_value != value:
            return other_value, endpoint
    return None


def _end_after(end: datetime | None, start: datetime | None) -> bool:
    return end is None or start is None or end > start


def _update_endpoint_extrema(
    extrema: list[tuple[str, datetime | None]],
    value: str,
    endpoint: datetime | None,
    *,
    latest: bool,
) -> None:
    """Update two distinct value extrema in constant time."""
    for index, (known_value, known_endpoint) in enumerate(extrema):
        if known_value == value:
            if _endpoint_precedes(known_endpoint, endpoint, latest=latest):
                extrema[index] = (value, endpoint)
            break
    else:
        extrema.append((value, endpoint))
    extrema.sort(key=lambda item: _endpoint_sort_key(item[1], latest=latest), reverse=latest)
    del extrema[2:]


def _endpoint_precedes(left: datetime | None, right: datetime | None, *, latest: bool) -> bool:
    if latest:
        return _endpoint_sort_key(left, latest=True) < _endpoint_sort_key(right, latest=True)
    return _endpoint_sort_key(left, latest=False) > _endpoint_sort_key(right, latest=False)


def _endpoint_sort_key(endpoint: datetime | None, *, latest: bool) -> tuple[bool, datetime | None]:
    # For latest ends, None sorts after all timestamps; for earliest starts, before them.
    return (endpoint is None if latest else endpoint is not None, endpoint)


def _is_due(
    signal_at: datetime, memory_ids: Sequence[str], weighed: Mapping[str, datetime]
) -> bool:
    """Report whether a group's own signal is newer than anything that has weighed it."""
    marks = [weighed[memory_id] for memory_id in memory_ids if memory_id in weighed]
    return not marks or signal_at > max(marks)


def _pressure_candidates(
    connection: sqlite3.Connection,
    weighed: Mapping[str, datetime],
    limit: int,
    record_budget: int | None,
) -> list[StoredCandidate]:
    """Oldest never-weighed records, while the store holds more than its configured budget.

    Pressure has no signal time of its own -- the condition is a level, not an event -- so
    "already weighed" here means weighed at all rather than weighed since some moment. Once
    everything in the store has been weighed once, pressure derives nothing further: a
    deliberation that keeps re-proposing the same forgetting is the disease this marker exists to
    cure, and the honest answer is that there is no new work.
    """
    if record_budget is None:
        return []
    active = int(
        connection.execute(
            "SELECT COUNT(*) AS records FROM memory_records WHERE forgotten_at IS NULL"
        ).fetchone()["records"]
    )
    if active <= record_budget:
        return []
    rows = connection.execute(
        """
        SELECT memory_id
        FROM memory_records
        WHERE forgotten_at IS NULL
        ORDER BY COALESCE(last_accessed_at, created_at), memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    over = active - record_budget
    return [
        StoredCandidate(
            trigger="pressure",
            memory_ids=(memory_id,),
            evidence_count=over,
        )
        for memory_id in (_row_text(row, "memory_id") for row in rows)
        if memory_id not in weighed
    ][:limit]


def _idle_candidates(
    connection: sqlite3.Connection,
    weighed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Derived lineages nothing has ever weighed, oldest first.

    Only reached when the operator declares the window. The kernel never infers idleness from a
    clock: whether the device is idle or charging is the host's knowledge, not the store's.
    """
    rows = connection.execute(
        """
        SELECT s.memory_id AS memory_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE v.retired_at IS NULL AND r.forgotten_at IS NULL
        ORDER BY v.recorded_at, s.memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    return [
        StoredCandidate(trigger="idle", memory_ids=(memory_id,), evidence_count=1)
        for memory_id in (_row_text(row, "memory_id") for row in rows)
        if memory_id not in weighed
    ][:limit]


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


def _later_operation_depends_on(
    connection: sqlite3.Connection,
    operation_id: int,
    memory_ids: Sequence[str],
) -> bool:
    """Return whether a later standing operation names one of these outputs."""
    ids = tuple(dict.fromkeys(memory_ids))
    for memory_id in ids:
        _require_identifier(memory_id, "memory_id")
    for offset in range(0, len(ids), _SQLITE_PARAMETER_BATCH - 1):
        batch = ids[offset : offset + _SQLITE_PARAMETER_BATCH - 1]
        placeholders = ", ".join("?" for _memory_id in batch)
        if (
            connection.execute(
                f"""
            SELECT 1
            FROM memory_operations AS o
            JOIN json_tree(json_array(
                json_extract(o.operation_json, '$.evidence_ids'),
                json_extract(o.operation_json, '$.target_ids'),
                json_extract(o.effects_json, '$.created_ids'),
                json_extract(o.effects_json, '$.changed_ids'),
                json_extract(o.effects_json, '$.activated_ids'),
                json_extract(o.effects_json, '$.linked'),
                json_extract(o.effects_json, '$.superseded')
            )) AS j
            WHERE o.operation_id > ? AND o.rolled_back_at IS NULL
              AND j.type = 'text' AND j.value IN ({placeholders})
            LIMIT 1
            """,
                (operation_id, *batch),
            ).fetchone()
            is not None
        ):
            return True
    return False


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
                    "activated_ids": list(operation.activated_ids),
                    "forgotten_ids": list(operation.forgotten_ids),
                    "linked": [list(pair) for pair in operation.linked],
                    "linked_clauses": [list(pair) for pair in operation.linked_clauses],
                    "clause_changes": [
                        {
                            "memory_id": change.memory_id,
                            "clause_id": change.clause_id,
                            "previous_active": change.previous_active,
                            "previous_confidence": change.previous_confidence,
                            "applied_confidence": change.applied_confidence,
                            "applied_recorded_at": _datetime_text(change.applied_recorded_at),
                            "applied_version": change.applied_version,
                        }
                        for change in operation.clause_changes
                    ],
                    "superseded": [list(pair) for pair in operation.superseded],
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
        activated_ids=tuple(effects.get("activated_ids") or ()),
        forgotten_ids=tuple(effects.get("forgotten_ids") or ()),
        linked=tuple((pair[0], pair[1]) for pair in effects.get("linked") or ()),
        linked_clauses=tuple((pair[0], pair[1]) for pair in effects.get("linked_clauses") or ()),
        clause_changes=tuple(
            StoredEvidenceClauseChange(
                memory_id=change["memory_id"],
                clause_id=change["clause_id"],
                previous_active=change["previous_active"],
                previous_confidence=change.get("previous_confidence"),
                applied_confidence=change["applied_confidence"],
                applied_recorded_at=_parse_datetime(change["applied_recorded_at"]),
                applied_version=int(change["applied_version"]),
            )
            for change in effects.get("clause_changes") or ()
        ),
        superseded=tuple((pair[0], int(pair[1])) for pair in effects.get("superseded") or ()),
        applied_at=_parse_datetime(_row_text(row, "applied_at")),
        rolled_back_at=_optional_datetime_from_row(row, "rolled_back_at"),
        outcome=_optional_row_text(row, "outcome"),
        outcome_note=_optional_row_text(row, "outcome_note"),
    )


def _evidence_summary(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[int, float]:
    # Historic singleton clauses retain the capture-group noisy-OR projection. A multi-member
    # clause is one model assessment, however many sources it names: treat all such assessments
    # as one conservative alternative rather than inventing independent votes from its members.
    rows = connection.execute(
        """
        SELECT MAX(c.confidence) AS confidence
        FROM memory_evidence AS e
        JOIN memory_evidence_clauses AS c
          ON c.memory_id = e.memory_id AND c.retired_at IS NULL AND c.member_count = 1
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
             AND m.source_memory_id = e.source_memory_id
        WHERE e.memory_id = ? AND e.retired_at IS NULL
        GROUP BY e.source_group_id
        """,
        (memory_id,),
    ).fetchall()
    combined = 0.0
    for row in rows:
        combined = 1.0 - (1.0 - combined) * (1.0 - float(row["confidence"]))
    joint = connection.execute(
        """
        SELECT MAX(confidence) AS confidence
        FROM memory_evidence_clauses
        WHERE memory_id = ? AND retired_at IS NULL AND member_count > 1
        """,
        (memory_id,),
    ).fetchone()
    joint_confidence = (
        0.0 if joint is None or joint["confidence"] is None else float(joint["confidence"])
    )
    return max(len(rows), 1 if joint_confidence else 0), max(combined, joint_confidence)


def _semantic_visibility(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    lineage_id: str,
    kind: str,
    basis: str,
    identity_id: str | None,
    evidence_count: int,
    valid_from: datetime | None,
    valid_until: datetime | None,
    explicit_intervals: Sequence[tuple[datetime | None, datetime | None]] | None = None,
) -> bool:
    if basis in {
        EvidenceBasis.USER_STATEMENT.value,
        EvidenceBasis.RESPONSE_FEEDBACK.value,
    }:
        return True
    if kind != MemoryKind.OBSERVATION.value and evidence_count == 0:
        return False
    # A naming assertion bound to an identity is corroborated like an inferred trait: what an
    # agent claims somebody is called stays hidden, and so stays out of the projected
    # `identities.name`, until two independent evidence groups support it.
    if kind != MemoryKind.TRAIT.value and not (
        kind == MemoryKind.ENTITY.value and identity_id is not None
    ):
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
                kind,
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


def _refresh_inherited_columns(connection: sqlite3.Connection, memory_id: str) -> None:
    """Re-derive the place and the metadata one derived record inherits from its live evidence.

    The kernel clears both columns when two sources disagree, and evidence is not permanent: a
    deleted source or a rolled-back consolidation retires it. Recomputing them here, with the
    kernel's rule -- every remaining source agrees, or nothing -- keeps a record whose survivors
    all stand in one room from staying unscoped and out of every place-filtered read.

    A naming assertion -- a bound `ENTITY` -- inherits neither column from the clips that show
    the person, so it is left alone: who somebody is does not stop being true in another room.
    """
    rows = connection.execute(
        """
        SELECT r.place_id, r.metadata_json
        FROM memory_evidence AS e
        JOIN memory_records AS r ON r.memory_id = e.source_memory_id
        WHERE e.memory_id = ? AND e.retired_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM memory_semantics AS s
              WHERE s.memory_id = e.memory_id AND s.kind = ? AND s.identity_id IS NOT NULL
          )
        """,
        (memory_id, MemoryKind.ENTITY.value),
    ).fetchall()
    if not rows:
        # Nothing supports the record any more, or nothing may reach it. The caller removes an
        # unsupported record; clearing its columns first would only alter the row it deletes.
        return
    places = {_optional_row_text(row, "place_id") for row in rows}
    tags = {_canonical_object_json(_row_text(row, "metadata_json")) for row in rows}
    place_id = places.pop() if len(places) == 1 else None
    current = connection.execute(
        "SELECT place_id FROM memory_records WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    connection.execute(
        "UPDATE memory_records SET place_id = ?, metadata_json = ? WHERE memory_id = ?",
        (place_id, tags.pop() if len(tags) == 1 else "{}", memory_id),
    )
    # The search index carries `place_id` as a filter field, so a record that regains or loses
    # its inherited place has to be reprojected -- this UPDATE bypasses `_write_memory`, which
    # is where every other place change is noticed.
    if current is not None and _optional_row_text(current, "place_id") != place_id:
        LocalStore._queue_memory_embeddings(connection, memory_id, exclude=set())


def _refresh_evidence_projection(
    connection: sqlite3.Connection,
    memory_id: str,
    changed_at: datetime,
) -> None:
    _refresh_inherited_columns(connection, memory_id)
    evidence_count, confidence = _evidence_summary(connection, memory_id)
    rows = connection.execute(
        """
        SELECT v.*, s.lineage_id, s.kind, s.basis, s.identity_id
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
            identity_id=_optional_row_text(row, "identity_id"),
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
        SELECT v.*, s.kind, s.basis, s.identity_id
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
            identity_id=_optional_row_text(row, "identity_id"),
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


def _require_scope_axes(
    *,
    valid_at: datetime | None,
    known_at: datetime | None,
    near: SpatialContext | None,
    radius_m: float | None,
) -> None:
    """Validate the scope axes a hydration reads under, for callers that may not reach one."""
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


def _memory_in_scope(
    row: sqlite3.Row,
    *,
    active_only: bool,
    known_at: datetime | None,
    near: SpatialContext | None,
    semantic_ids: frozenset[str],
    scoped_ids: Collection[str],
) -> bool:
    """Decide whether one `memory_records` row survives the requested scope.

    `row` needs only `memory_id`, `created_at`, and `forgotten_at`, so both the scoped hydration
    and the scoped count reach this one predicate.
    """
    if not active_only:
        return True
    # Forgetting is a column, not a deletion: the row stays readable by ID and drops out of every
    # active slate, whatever its validity interval or pose says.
    if row["forgotten_at"] is not None:
        return False
    memory_id = _row_text(row, "memory_id")
    if memory_id in scoped_ids:
        return True
    # A record with no `memory_semantics` row declares no validity interval, so it is valid at
    # every `valid_at` exactly like a version row whose `valid_from` and `valid_until` are NULL;
    # only recorded time can exclude it, and there `created_at` is the honest bound. `near` is
    # separate and spatial: a record with no pose is not at any location, so it drops just as a
    # semantic row without a pose does in the selection pass.
    return (
        near is None
        and memory_id not in semantic_ids
        and (known_at is None or _parse_datetime(_row_text(row, "created_at")) <= known_at)
    )


def _select_memory_contexts(  # noqa: C901 - one authoritative bitemporal selection pass
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    near: SpatialContext | None = None,
    radius_m: float | None = None,
    active_only: bool = False,
) -> tuple[dict[str, tuple[sqlite3.Row, SpatialContext | None]], frozenset[str]]:
    """Choose the one current version row per memory and apply every scope predicate.

    Shared by `read_memories` and `count_memories` so a scoped count cannot drift from the
    scoped hydration it predicts.
    """
    if not memory_ids:
        return {}, frozenset()
    _require_scope_axes(valid_at=valid_at, known_at=known_at, near=near, radius_m=radius_m)
    query_valid_at = valid_at or datetime.now(timezone.utc)
    query_known_at = known_at or datetime.now(timezone.utc)

    rows: list[sqlite3.Row] = []
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
                    s.identity_id, s.cue_modality, s.valence, s.arousal,
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

    scoped: dict[str, tuple[sqlite3.Row, SpatialContext | None]] = {}
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
        scoped[memory_id] = (row, spatial)
    return scoped, semantic_ids


def _read_memory_contexts(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    near: SpatialContext | None = None,
    radius_m: float | None = None,
    active_only: bool = False,
) -> tuple[dict[str, MemoryContext], frozenset[str]]:
    scoped, semantic_ids = _select_memory_contexts(
        connection,
        memory_ids,
        valid_at=valid_at,
        known_at=known_at,
        near=near,
        radius_m=radius_m,
        active_only=active_only,
    )
    evidence = _read_memory_evidence(connection, tuple(scoped), known_at=known_at)
    contexts: dict[str, MemoryContext] = {}
    for memory_id, (row, spatial) in scoped.items():
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
            identity_id=_optional_row_text(row, "identity_id"),
            spatial=spatial,
            cue_modality=(
                None if row["cue_modality"] is None else Modality(_row_text(row, "cue_modality"))
            ),
            valence=None if row["valence"] is None else float(row["valence"]),
            arousal=None if row["arousal"] is None else float(row["arousal"]),
        )
    return contexts, semantic_ids


def _read_memory_evidence(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    known_at: datetime | None,
) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    evidence_time = (
        "retired_at IS NULL"
        if known_at is None
        else "recorded_at <= ? AND (retired_at IS NULL OR retired_at > ?)"
    )
    for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        parameters: tuple[object, ...] = tuple(batch)
        if known_at is not None:
            known_text = _datetime_text(known_at)
            parameters += (known_text, known_text)
        for row in connection.execute(
            f"""
            SELECT memory_id, source_memory_id
            FROM memory_evidence
            WHERE memory_id IN ({placeholders}) AND {evidence_time}
            ORDER BY memory_id, position
            """,
            parameters,
        ).fetchall():
            evidence.setdefault(_row_text(row, "memory_id"), []).append(
                _row_text(row, "source_memory_id")
            )
    return evidence


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


def _index_document_from_row(
    row: sqlite3.Row,
    identity_ids: Mapping[str, tuple[str, ...]],
) -> IndexDocument:
    return IndexDocument(
        embedding=_embedding_from_row(row),
        content=_row_text(row, "content"),
        metadata_json=_row_text(row, "metadata_json"),
        memory_type=_row_text(row, "memory_type"),
        occurred_at=_optional_datetime_from_row(row, "occurred_at"),
        occurred_end=_optional_datetime_from_row(row, "occurred_end"),
        place_id=_optional_row_text(row, "place_id"),
        identity_ids=identity_ids.get(_row_text(row, "memory_id"), ()),
    )


def _identity_scope(
    connection: sqlite3.Connection,
    identity_id: str | None,
) -> tuple[str | None, tuple[object, ...]]:
    """Return the SQL clause and parameters that scope `memory_records` to one identity.

    A `None` clause means the requested identity does not exist, so nothing is in scope.
    """
    if identity_id is None:
        return "", ()
    resolved_id = _resolve_identity_id(connection, identity_id)
    if resolved_id is None:
        return None, ()
    return _IDENTITY_SCOPE_CLAUSE, (resolved_id, resolved_id, resolved_id)


def _identity_projections(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Project every identity each memory is about, for the search index to filter on."""
    projections: dict[str, list[str]] = {}
    for offset in range(0, len(memory_ids), _SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + _SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        statement = _IDENTITY_MEMORY_SQL.format(predicate=f"memory_id IN ({placeholders})")
        for row in connection.execute(statement, (*batch, *batch, *batch)).fetchall():
            projections.setdefault(_row_text(row, "memory_id"), []).append(
                _row_text(row, "identity_id")
            )
    return {memory_id: tuple(sorted(values)) for memory_id, values in projections.items()}


def _queue_asset_identity_projection(connection: sqlite3.Connection, asset_id: str) -> None:
    """Re-enqueue the memories that already reference an asset whose people just became known.

    Speech and face analysis usually run while the memory is being written, but a caller can also
    ask for either after the fact, and then the indexed identity projection of an already-indexed
    memory is out of date the moment the observations land.
    """
    connection.execute(
        """
        INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
        SELECT e.embedding_id, 'upsert', ?
        FROM embeddings AS e
        WHERE EXISTS (
            SELECT 1 FROM memory_assets AS ma
            WHERE ma.memory_id = e.memory_id AND ma.asset_id = ?
        )
        ORDER BY e.embedding_id
        """,
        (_datetime_text(datetime.now(timezone.utc)), asset_id),
    )


def _queue_identity_projection(connection: sqlite3.Connection, identity_id: str) -> None:
    """Re-enqueue every memory whose indexed identity projection this identity appears in.

    Merging, splitting and erasing an identity all re-point the rows the projection reads, and
    none of them touches the embeddings, so nothing else would tell the index its filter field
    is now wrong. Called before the re-pointing, while the rows still name this identity.
    """
    connection.execute(
        f"""
        INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
        SELECT e.embedding_id, 'upsert', ?
        FROM embeddings AS e
        WHERE e.memory_id IN ({_IDENTITY_MEMORIES_SQL})
        ORDER BY e.embedding_id
        """,
        (_datetime_text(datetime.now(timezone.utc)), identity_id, identity_id, identity_id),
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


# Hydration re-canonicalizes metadata the store itself wrote, so the round trip is a no-op that a
# search pays ~140 times; the result is a pure function of the text, and invalid JSON is not cached.
@lru_cache(maxsize=1024)
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
