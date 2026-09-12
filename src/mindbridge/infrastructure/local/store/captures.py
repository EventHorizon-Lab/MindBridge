"""The deferred-capture queue."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    parse_datetime,
    prepare_write_batch,
    row_text,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store.records import write_embedding, write_memory
from mindbridge.infrastructure.local.store.rows import (
    StoredEmbedding,
    StoredMemory,
    require_aware,
    require_identifier,
)
from mindbridge.types import PendingCapture


class CaptureQueue:
    """Rows that owe a later `settle()`."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

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
        supplied_memories, _embeddings, supplied_by_memory = prepare_write_batch(memories, ())
        if not supplied_memories:
            return ()
        require_aware(enqueued_at, "enqueued_at")
        enqueued: list[str] = []
        with self._connections.transaction() as connection:
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
                write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                )
                transaction_memory_ids.add(memory.memory_id)
                connection.execute(
                    "INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)",
                    (memory.memory_id, datetime_text(enqueued_at)),
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
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            (memory,),
            embeddings,
        )
        with self._connections.transaction() as connection:
            queued = connection.execute(
                "SELECT 1 FROM capture_queue WHERE memory_id = ?",
                (memory.memory_id,),
            ).fetchone()
            if queued is None:
                return False
            write_memory(
                connection,
                supplied_memories[0],
                supplied_embedding_ids=supplied_by_memory[memory.memory_id],
            )
            for embedding in supplied_embeddings:
                write_embedding(connection, embedding)
        return True

    def complete_captures(self, memory_ids: Sequence[str]) -> int:
        """Remove captures from the queue after every deferred stage succeeded.

        Retention also uses it, to abandon captures whose repeated failures have aged out: the
        queue row is the promise to keep retrying, and that is all it drops.
        """
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            require_identifier(memory_id, "memory_id")
        if not selected:
            return 0
        removed = 0
        with self._connections.transaction() as connection:
            for offset in range(0, len(selected), SQLITE_PARAMETER_BATCH):
                batch = selected[offset : offset + SQLITE_PARAMETER_BATCH]
                removed += connection.execute(
                    f"DELETE FROM capture_queue WHERE memory_id IN "
                    f"({', '.join('?' for _value in batch)})",
                    batch,
                ).rowcount
        return removed

    def record_capture_failure(self, memory_id: str, error: str) -> None:
        """Count one failed settlement and store its reason, leaving the row queued."""
        require_identifier(memory_id, "memory_id")
        with self._connections.transaction() as connection:
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
                require_identifier(memory_id, "memory_id")
        selected = None if memory_ids is None else tuple(dict.fromkeys(memory_ids))
        batches: list[tuple[str, ...] | None] = (
            [None]
            if selected is None
            else [
                selected[offset : offset + SQLITE_PARAMETER_BATCH]
                for offset in range(0, len(selected), SQLITE_PARAMETER_BATCH)
            ]
        )
        queued: list[PendingCapture] = []
        with self._connections.read_transaction() as connection:
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
                        memory_id=row_text(row, "memory_id"),
                        enqueued_at=parse_datetime(row_text(row, "enqueued_at")),
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
