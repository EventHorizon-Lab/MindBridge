"""The derived search projection: applying SQLite truth to Zvec through the outbox."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from threading import local

from opentelemetry.trace import Tracer

from mindbridge.exceptions import StorageError
from mindbridge.infrastructure.local.store import IndexDocument, IndexOperation
from mindbridge.kernel.derived import retrieval_document
from mindbridge.kernel.lifecycle import Lifecycle
from mindbridge.kernel.runtime import (
    Index,
    Storage,
    translate_index_errors,
    translate_storage_errors,
)
from mindbridge.kernel.tracing import Traced

# One drain hydrates and applies this many outbox rows per read. A flush is a fixed ~50 ms
# fsync-class operation whatever it carries -- one document costs the same as a thousand -- and it
# also creates one durable segment, which every later search pays for until an optimize merges it
# away. Both costs are therefore per *flush*, so a bulk write applies rows in the largest batch
# the index writes in a single call, 1024 documents, and flushes once per batch. `ZvecIndex.upsert`
# chunks anything larger, so this is a memory-for-flushes choice and not a safety bound. Raising
# it from 256 quartered both the flush count and the segment count of a bulk `add_many` (8 000
# memories: 32 flushes and 32 segments became 8 and 8) for about 32 MiB of transient hydration at
# 1024 dimensions.
OUTBOX_BATCH_SIZE = 1_024


# Applied-but-unflushed outbox rows that trigger the Zvec flush. Zvec answers dense and full-text
# queries from unflushed writes (and hides unflushed deletes), so a public write only has to
# *apply* its rows to be visible; the ~50 ms flush is taken once per this many rows, when a drain
# applied a deletion, and at `optimize()`, `reindex()` and `close()`. Rows are acknowledged in
# SQLite only after that flush, exactly as before, and a single `add` no longer pays a flush or
# leaves one segment behind.
# ponytail: 256 caps the replay after a crash at 256 idempotent upserts; raise it if segment count
# ever matters more than that.
INDEX_FLUSH_OPERATIONS = 256


_REINDEX_PAGE_SIZE = 256


class Projection(Traced):
    """Apply committed SQLite truth to the Zvec projection through the durable outbox."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        index: Index,
        lifecycle: Lifecycle,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._index = index
        self._lifecycle = lifecycle
        self._unflushed_operations: set[IndexOperation] = set()
        # Whether *this* thread is inside an `add_stream` item write, so a concurrent `add` on
        # another thread keeps flushing before it returns. Thread-local rather than one shared
        # slot: two threads streaming at once saved and restored each other's value, so the second
        # to finish handed the first thread's id back and pinned deferral on a thread that had left
        # the stream -- every later drain on it silently returned without flushing the index.
        self._deferral = local()

    def flush_pending(self) -> None:
        """Flush applied-but-unacknowledged rows, if any; `close()` calls this last."""
        if self._unflushed_operations:
            with self._trace("mindbridge.index.sync", kind="stage"):
                self.flush()

    def drain(self, *, force: bool = False) -> None:
        """Apply current SQLite truth to Zvec; flush and acknowledge once enough has been applied.

        Skipped inside an `add_stream` group, whose own bounds force it. Skipping only leaves
        rows pending: they are still durable in SQLite and the next drain applies them.
        """
        if not force and self._deferring:
            return
        with self._trace("mindbridge.index.sync", kind="stage"):
            self._apply_outbox()

    def _apply_outbox(self) -> None:
        while True:
            with self._trace(
                "mindbridge.index.sync.sqlite.read",
                kind="stage",
                failure_stage="index.sync",
            ):
                # Rows already applied and waiting for the batched flush stay pending in SQLite
                # until that flush acknowledges them; they are always the oldest pending rows, so
                # reading past the newest of them reads exactly what is not applied yet.
                applied_through = max(
                    (operation.operation_id for operation in self._unflushed_operations),
                    default=0,
                )
                with translate_storage_errors("read the search-index outbox"):
                    operations = self._store.index.pending_index_operations(
                        limit=OUTBOX_BATCH_SIZE, after=applied_through
                    )
                if not operations:
                    break
                last_by_embedding = {operation.embedding_id: operation for operation in operations}
                current = sorted(
                    last_by_embedding.values(), key=lambda operation: operation.operation_id
                )
                with translate_storage_errors("hydrate the search-index outbox"):
                    hydrated = self._store.index.read_index_documents(
                        tuple(operation.embedding_id for operation in current)
                    )
                    identity_memory_ids = tuple(
                        dict.fromkeys(
                            document.embedding.memory_id
                            for document in hydrated
                            if "[speech identities:" in document.content
                        )
                    )
                    memories = self._store.records.read_memories(identity_memory_ids)
            by_id = {document.embedding.embedding_id: document for document in hydrated}
            asset_ids = {
                memory.memory_id: frozenset(asset.asset_id for asset in memory.assets)
                for memory in memories
            }
            documents = [
                retrieval_document(
                    by_id[operation.embedding_id],
                    asset_ids.get(
                        by_id[operation.embedding_id].embedding.memory_id,
                        frozenset(),
                    ),
                )
                for operation in current
                if operation.embedding_id in by_id
            ]
            deleted_ids = [
                operation.embedding_id
                for operation in current
                if operation.embedding_id not in by_id
            ]
            with (
                self._trace(
                    "mindbridge.index.sync.zvec.apply",
                    kind="stage",
                    failure_stage="index.sync",
                ),
                translate_index_errors("update the search index"),
            ):
                if deleted_ids:
                    self._index.delete(deleted_ids)
                if documents:
                    self._index.upsert(documents)
            self._unflushed_operations.update(operations)
            # A deletion is made durable before its call returns: an erased record must not stay
            # in the index, nor be named by a pending outbox row, for the sake of write latency.
            if deleted_ids or len(self._unflushed_operations) >= INDEX_FLUSH_OPERATIONS:
                self.flush()
        # A flush that failed in an earlier drain left the set at the bound; retry it here even
        # when no new row arrived, so a read-only workload also heals the index.
        if len(self._unflushed_operations) >= INDEX_FLUSH_OPERATIONS:
            self.flush()

    def flush(self) -> None:
        """Make the applied Zvec changes durable, then acknowledge exactly their outbox rows."""
        if not self._unflushed_operations:
            return
        operations = tuple(self._unflushed_operations)
        with (
            self._trace(
                "mindbridge.index.sync.zvec.flush",
                kind="stage",
                failure_stage="index.sync",
            ),
            translate_index_errors("update the search index"),
        ):
            self._index.flush()
        # A failed flush above keeps the set, so the next drain retries without re-applying. Cleared
        # here, before optimize and acknowledge: if either fails the rows are re-applied (idempotent
        # upsert or delete) and acknowledged by the next drain, and a compaction inside
        # `optimize_if_needed` never runs over unflushed documents.
        self._unflushed_operations.clear()
        with (
            self._trace(
                "mindbridge.index.sync.zvec.optimize",
                kind="stage",
                failure_stage="index.sync",
            ),
            translate_index_errors("update the search index"),
        ):
            self._index.optimize_if_needed()
        with (
            self._trace(
                "mindbridge.index.sync.sqlite.ack",
                kind="stage",
                failure_stage="index.sync",
            ),
            translate_storage_errors("acknowledge the search-index outbox"),
        ):
            acknowledged = self._store.index.acknowledge_index_operations(operations)
        if acknowledged != len(operations):
            raise StorageError(
                "search-index outbox changed while it was being acknowledged",
                reason="flush_failed",
            )

    def _index_documents(self) -> Iterator[IndexDocument]:
        after: tuple[datetime, str] | None = None
        while True:
            with translate_storage_errors("read memories for reindexing"):
                memories = self._store.records.list_memories(limit=_REINDEX_PAGE_SIZE, after=after)
            if not memories:
                return
            asset_ids = {
                memory.memory_id: frozenset(asset.asset_id for asset in memory.assets)
                for memory in memories
            }
            with translate_storage_errors("hydrate memories for reindexing"):
                yield from (
                    retrieval_document(
                        document,
                        asset_ids[document.embedding.memory_id],
                    )
                    for document in self._store.index.read_memory_index_documents(
                        tuple(memory.memory_id for memory in memories)
                    )
                )
            last = memories[-1]
            after = (last.created_at, last.memory_id)

    @contextmanager
    def deferred(self, *, defer: bool) -> Iterator[None]:
        """Defer the calling thread's index commits for the block when `defer`.

        Scoped to the item write rather than the whole stream because `add_stream` is a
        generator: its body runs on whichever thread calls `next`, so a consumer that hands the
        iterator between workers opened the group on one thread and left it on another. Entering
        and leaving on the same thread within one resumption keeps every thread the stream ran on
        clear of a deferral nothing would ever lift, and a consumer's own writes between two
        yields flush the way they would outside the stream.
        """
        if not defer:
            yield
            return
        outer = self._deferring
        self._deferral.active = True
        try:
            yield
        finally:
            self._deferral.active = outer

    @property
    def _deferring(self) -> bool:
        """Whether the calling thread is inside a deferred `add_stream` item write."""
        return bool(getattr(self._deferral, "active", False))

    def reindex(self) -> int:
        with (
            self._trace("mindbridge.reindex", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            # Flushed and acknowledged first: the checkpoint below queues every embedding that has
            # no pending row, so an applied-but-unacknowledged row would otherwise keep its
            # embedding out of the rebuild's replay if the rebuild fails.
            self.drain()
            self.flush()
            with translate_storage_errors("checkpoint a search-index rebuild"):
                self._store.index.queue_all_embeddings()
            memory_count = 0

            def documents() -> Iterator[IndexDocument]:
                nonlocal memory_count
                for document in self._index_documents():
                    if document.embedding.object_part == 0:
                        memory_count += 1
                    yield document

            with translate_index_errors("rebuild the search index"):
                self._index.rebuild(documents(), batch_size=_REINDEX_PAGE_SIZE)
            # Adds may commit SQLite while the rebuild owns the Zvec boundary. Replay instead of
            # blindly acknowledging so records committed after its SQLite scan cannot be lost.
            self.drain()
            self.flush()
            return memory_count

    def optimize(self) -> None:
        with (
            self._trace("mindbridge.optimize", kind="operation"),
            self._lifecycle.operation(),
            self._write_lock,
        ):
            self.drain()
            self.flush()
            with translate_index_errors("optimize the search index"):
                self._index.optimize()
                self._index.flush()
