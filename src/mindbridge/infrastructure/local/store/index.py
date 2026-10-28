"""Embeddings, text selectors, index documents, and the durable search-index outbox."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing
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
    IndexCandidate,
    IndexDocument,
    IndexOperation,
    StoredEmbedding,
    StoredTextSelector,
    StoredTextSpanPiece,
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
        rows: list[sqlite3.Row] = []
        with self._connections.connection() as connection:
            for offset in range(0, len(embedding_ids), SQLITE_PARAMETER_BATCH):
                batch = embedding_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _embedding_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT e.embedding_id, e.memory_id, m.occurred_at, m.occurred_end
                        FROM embeddings AS e
                        JOIN memory_records AS m ON m.memory_id = e.memory_id
                        WHERE e.embedding_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        # Decoding holds no SQLite state; return the exclusive connection before doing it.
        by_id: dict[str, IndexCandidate] = {}
        for row in rows:
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
        with closing(
            self.iter_memory_embedding_parts(memory_ids, space_id=space_id, task=task)
        ) as parts:
            for memory_id, _embedding_id, vector in parts:
                yield memory_id, unpack_vector(vector, len(vector) // 4)

    def iter_memory_embedding_parts(
        self,
        memory_ids: Sequence[str],
        *,
        space_id: str,
        task: str,
    ) -> Generator[tuple[str, str, bytes], None, None]:
        """Stream vector identities and FP32 buffers in deterministic order from one snapshot."""
        if not memory_ids:
            return
        for memory_id in memory_ids:
            require_identifier(memory_id, "memory_id")
        require_identifier(space_id, "space_id")
        require_identifier(task, "task")
        with self._connections.read_transaction() as connection:
            for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows = connection.execute(
                    f"""
                    SELECT memory_id, embedding_id, dimension, vector
                    FROM embeddings
                    WHERE memory_id IN ({placeholders})
                      AND space_id = ? AND task = ?
                    ORDER BY object_part, embedding_id
                    """,
                    (*batch, space_id, task),
                )
                for row in rows:
                    vector = row["vector"]
                    if not isinstance(vector, bytes):
                        raise RuntimeError("stored embedding vector is not a BLOB")
                    dimension = int(row["dimension"])
                    if dimension <= 0 or len(vector) != dimension * 4:
                        raise RuntimeError(
                            "stored embedding dimension does not match its FP32 BLOB"
                        )
                    yield (
                        row_text(row, "memory_id"),
                        row_text(row, "embedding_id"),
                        vector,
                    )

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
