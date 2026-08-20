"""SQLite schema shared by local identity learning and deletion sync."""

import sqlite3


def initialize_identity_tables(connection: sqlite3.Connection) -> None:
    """Install encrypted identity samples and face/voice association evidence."""
    connection.executescript(
        """
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
            PRIMARY KEY (
                tenant_id, device_id, kind, model_id, sample_id
            )
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
            PRIMARY KEY (
                tenant_id, device_id, association_model_id, evidence_id
            )
        );

        CREATE INDEX IF NOT EXISTS edge_face_voice_pair_idx
            ON edge_face_voice_evidence (
                tenant_id, device_id, association_model_id,
                face_identity_id, voice_identity_id
            );

        CREATE INDEX IF NOT EXISTS edge_face_voice_observation_idx
            ON edge_face_voice_evidence (tenant_id, device_id, source_observation_id);

        """
    )
