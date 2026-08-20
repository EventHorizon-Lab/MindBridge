"""SQLite schema shared by local identity learning and deletion sync."""

import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_identity_templates (
    tenant_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('face', 'voice')),
    identity_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    nonce BLOB NOT NULL CHECK (length(nonce) = 12),
    encrypted_embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, device_id, kind, model_id, sample_id)
);

CREATE INDEX IF NOT EXISTS edge_identity_match_idx
    ON edge_identity_templates (
        tenant_id, device_id, kind, model_id, dimension, identity_id
    );

CREATE INDEX IF NOT EXISTS edge_identity_observation_idx
    ON edge_identity_templates (tenant_id, source_observation_id);

CREATE TABLE IF NOT EXISTS edge_face_voice_evidence (
    tenant_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    association_model_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    source_observation_id TEXT NOT NULL,
    face_identity_id TEXT NOT NULL,
    voice_identity_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, device_id, association_model_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS edge_face_voice_pair_idx
    ON edge_face_voice_evidence (
        tenant_id, device_id, association_model_id,
        face_identity_id, voice_identity_id
    );

CREATE INDEX IF NOT EXISTS edge_face_voice_observation_idx
    ON edge_face_voice_evidence (tenant_id, device_id, source_observation_id);

"""

# Both tables keyed their rows by the producing model's revision, which SQLite will not let an
# upgraded device drop -- it refuses to drop a primary key column. Copying the rows forward would
# not help either: a template's revision also sits inside its encrypted payload, whose model
# forbids unknown fields, so every legacy row would fail to decrypt. The device relearns the
# identities it still sees instead. Without this, `CREATE TABLE IF NOT EXISTS` is a silent no-op
# and the next insert fails on the legacy NOT NULL column.
_TABLES_RETIRED_WITH_THEIR_REVISION_COLUMN = (
    ("edge_identity_templates", "model_revision"),
    ("edge_face_voice_evidence", "association_model_revision"),
)


def initialize_identity_tables(connection: sqlite3.Connection) -> None:
    """Install encrypted identity samples and face/voice association evidence."""
    _drop_tables_still_keyed_by_a_model_revision(connection)
    connection.executescript(_SCHEMA)


def _drop_tables_still_keyed_by_a_model_revision(connection: sqlite3.Connection) -> None:
    for table, retired_column in _TABLES_RETIRED_WITH_THEIR_REVISION_COLUMN:
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if retired_column in columns:
            connection.execute(f"DROP TABLE {table}")
