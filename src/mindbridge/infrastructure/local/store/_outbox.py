"""Enqueueing search-index projection work inside the transaction that changed the truth."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from mindbridge.infrastructure.local.store._codec import datetime_text, row_text
from mindbridge.infrastructure.local.store._membership import IDENTITY_MEMORIES_SQL


def enqueue_capture(connection: sqlite3.Connection, memory_id: str, enqueued_at: datetime) -> None:
    """Queue a captured memory for settlement; one already waiting keeps its earlier slot."""
    connection.execute(
        """
        INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)
        ON CONFLICT (memory_id) DO NOTHING
        """,
        (memory_id, datetime_text(enqueued_at)),
    )


def queue_asset_identity_projection(connection: sqlite3.Connection, asset_id: str) -> None:
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
        (datetime_text(datetime.now(timezone.utc)), asset_id),
    )


def queue_identity_projection(connection: sqlite3.Connection, identity_id: str) -> None:
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
        WHERE e.memory_id IN ({IDENTITY_MEMORIES_SQL})
        ORDER BY e.embedding_id
        """,
        (datetime_text(datetime.now(timezone.utc)), identity_id, identity_id, identity_id),
    )


def queue_memory_embeddings(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    exclude: set[str],
) -> None:
    embedding_ids = connection.execute(
        "SELECT embedding_id FROM embeddings WHERE memory_id = ? ORDER BY embedding_id",
        (memory_id,),
    ).fetchall()
    now = datetime_text(datetime.now(timezone.utc))
    connection.executemany(
        """
        INSERT INTO search_index_queue (embedding_id, action, enqueued_at)
        VALUES (?, 'upsert', ?)
        """,
        (
            (row_text(row, "embedding_id"), now)
            for row in embedding_ids
            if row_text(row, "embedding_id") not in exclude
        ),
    )
