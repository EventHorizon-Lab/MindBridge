"""The current SQLite schema, and the refusal of every other version."""

from __future__ import annotations

import sqlite3

from mindbridge.infrastructure.local.store._codec import row_text
from mindbridge.infrastructure.local.store.errors import UnsupportedSchemaError

SCHEMA_VERSION = 18


REQUIRED_TABLES = frozenset(
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


# `feedback_candidates` polls the most recently recalled records on every
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

PRAGMA user_version = {SCHEMA_VERSION};
COMMIT;
"""


def user_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def table_names(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    return frozenset(row_text(row, "name") for row in rows)


def create_schema(
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


def validate_text_selector_schema(connection: sqlite3.Connection) -> None:
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


def validate_evidence_clause_schema(connection: sqlite3.Connection) -> None:
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
