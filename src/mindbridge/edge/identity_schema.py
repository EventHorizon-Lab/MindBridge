"""SQLite schema shared by local identity learning and deletion sync."""

import sqlite3


def initialize_identity_tables(connection: sqlite3.Connection) -> None:
    """Install the encrypted identity sample table."""
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
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            nonce BLOB NOT NULL CHECK (length(nonce) = 12),
            encrypted_embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (
                tenant_id, device_id, kind, model_id, model_revision, sample_id
            )
        );

        CREATE INDEX IF NOT EXISTS edge_identity_match_idx
            ON edge_identity_templates (
                tenant_id, device_id, kind, model_id, model_revision, dimension, identity_id
            );

        CREATE INDEX IF NOT EXISTS edge_identity_observation_idx
            ON edge_identity_templates (tenant_id, source_observation_id);
        """
    )
