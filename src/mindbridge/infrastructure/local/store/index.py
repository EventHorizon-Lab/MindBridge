"""Embeddings, text selectors, index documents, and the durable search-index outbox."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Sequence
from datetime import datetime, timezone

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    embedding_from_row,
    index_action,
    index_document_from_row,
    optional_datetime_from_row,
    row_text,
    unpack_vector,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._membership import identity_projections
from mindbridge.infrastructure.local.store.records import write_embedding
from mindbridge.infrastructure.local.store.rows import (
    MEMORY_TYPES,
    IndexCandidate,
    IndexDocument,
    IndexOperation,
    StoredEmbedding,
    StoredTextSelector,
    StoredTextSpanPiece,
    require_aware,
    require_identifier,
)


class IndexOutbox:
    """FP32 embeddings, text selectors, and the outbox the Zvec projection replays."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

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
        require_aware(occurred_from, "occurred_from")
        require_aware(occurred_until, "occurred_until")
        if occurred_until <= occurred_from:
            raise ValueError("occurred_until must be later than occurred_from")
        require_identifier(space_id, "space_id")
        require_identifier(task, "task")
        if memory_type is not None and memory_type not in MEMORY_TYPES:
            raise ValueError("memory_type is invalid")
        start = datetime_text(occurred_from)
        until = datetime_text(occurred_until)
        type_clause = "" if memory_type is None else "AND m.memory_type = ?"
        scope: tuple[object, ...] = (
            (space_id, task) if memory_type is None else (space_id, task, memory_type)
        )
        with self._connections.read_transaction() as connection:
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
        return frozenset(row_text(row, "embedding_id") for row in rows), total

    def write_embedding(self, embedding: StoredEmbedding) -> bool:
        """Create or update one authoritative vector and enqueue its index mutation."""
        with self._connections.transaction() as connection:
            created = (
                connection.execute(
                    "SELECT 1 FROM embeddings WHERE embedding_id = ?",
                    (embedding.embedding_id,),
                ).fetchone()
                is None
            )
            write_embedding(connection, embedding)
        return created

    def read_embedding(self, embedding_id: str) -> StoredEmbedding | None:
        """Return one authoritative FP32 embedding."""
        require_identifier(embedding_id, "embedding_id")
        with self._connections.connection() as connection:
            row = connection.execute(
                """
                SELECT embedding_id, memory_id, object_part, model_id, space_id, task,
                       dimension, normalized, vector, created_at
                FROM embeddings
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            ).fetchone()
        return None if row is None else embedding_from_row(row)

    def read_text_selectors(
        self,
        embedding_ids: Sequence[str],
    ) -> dict[str, tuple[StoredTextSelector, ...]]:
        """Read exact text selectors for a bounded embedding-ID batch."""
        if not embedding_ids:
            return {}
        for embedding_id in embedding_ids:
            require_identifier(embedding_id, "embedding_id")
        selector_rows: list[sqlite3.Row] = []
        piece_rows: list[sqlite3.Row] = []
        with self._connections.connection() as connection:
            for offset in range(0, len(embedding_ids), SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + SQLITE_PARAMETER_BATCH]
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
            key = (row_text(row, "embedding_id"), int(row["selector_position"]))
            pieces.setdefault(key, []).append(
                StoredTextSpanPiece(
                    role=row_text(row, "role"),  # type: ignore[arg-type]
                    start_codepoint=int(row["start_codepoint"]),
                    end_codepoint=int(row["end_codepoint"]),
                    piece_sha256=row_text(row, "piece_sha256"),
                )
            )
        grouped: dict[str, list[StoredTextSelector]] = {}
        for row in selector_rows:
            embedding_id = row_text(row, "embedding_id")
            position = int(row["selector_position"])
            selector_pieces = tuple(pieces.get((embedding_id, position), ()))
            if not selector_pieces:
                continue
            grouped.setdefault(embedding_id, []).append(
                StoredTextSelector(
                    parent_content_sha256=row_text(row, "parent_content_sha256"),
                    embedding_input_sha256=row_text(row, "embedding_input_sha256"),
                    recipe_version=row_text(row, "recipe_version"),
                    pieces=selector_pieces,
                )
            )
        return {
            embedding_id: tuple(grouped.get(embedding_id, ())) for embedding_id in embedding_ids
        }

    def read_index_document(self, embedding_id: str) -> IndexDocument | None:
        """Hydrate the current index payload from authoritative SQLite state."""
        require_identifier(embedding_id, "embedding_id")
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
            require_identifier(embedding_id, "embedding_id")
        rows = []
        with self._connections.connection() as connection:
            for offset in range(0, len(embedding_ids), SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + SQLITE_PARAMETER_BATCH]
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
            identity_ids = identity_projections(
                connection,
                tuple({row_text(row, "memory_id") for row in rows}),
            )
        by_id: dict[str, IndexDocument] = {}
        for row in rows:
            document = index_document_from_row(row, identity_ids)
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
            require_identifier(embedding_id, "embedding_id")
        by_id: dict[str, IndexCandidate] = {}
        with self._connections.connection() as connection:
            for offset in range(0, len(embedding_ids), SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + SQLITE_PARAMETER_BATCH]
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
                        embedding_id=row_text(row, "embedding_id"),
                        memory_id=row_text(row, "memory_id"),
                        occurred_at=optional_datetime_from_row(row, "occurred_at"),
                        occurred_end=optional_datetime_from_row(row, "occurred_end"),
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
            require_identifier(memory_id, "memory_id")
        rows = []
        with self._connections.connection() as connection:
            for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
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
            identity_ids = identity_projections(connection, memory_ids)
        by_memory: dict[str, list[IndexDocument]] = {}
        for row in rows:
            document = index_document_from_row(row, identity_ids)
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
            require_identifier(memory_id, "memory_id")
        require_identifier(space_id, "space_id")
        require_identifier(task, "task")
        with self._connections.connection() as connection:
            for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
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
                    yield row_text(row, "memory_id"), unpack_vector(vector, int(row["dimension"]))

    def pending_index_operations(
        self, *, limit: int = 100, after: int = 0
    ) -> tuple[IndexOperation, ...]:
        """Read queued mutations with an operation id above `after`, without acknowledging them.

        Operation ids are AUTOINCREMENT and rows only ever leave through acknowledgement, so the
        cursor is exact: a caller that has applied every row up to `after` reads exactly the rest.
        """
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a non-negative integer")
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT operation_id, embedding_id, action
                FROM search_index_queue
                WHERE operation_id > ?
                ORDER BY operation_id
                LIMIT ?
                """,
                (after, limit),
            ).fetchall()
        return tuple(
            IndexOperation(
                operation_id=int(row["operation_id"]),
                embedding_id=row_text(row, "embedding_id"),
                action=index_action(row),
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
        with self._connections.transaction() as connection:
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
        with self._connections.transaction() as connection:
            now = datetime_text(datetime.now(timezone.utc))
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
