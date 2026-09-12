"""The write plane: add, batch add, capture, settle, and streamed observations."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from opentelemetry.trace import Tracer

from mindbridge._telemetry import CAPTURE_FAILED, CAPTURE_SETTLED, CAPTURE_TIME_TO_SEARCHABLE
from mindbridge.exceptions import MindBridgeError, StorageError, ValidationError
from mindbridge.infrastructure.local.store import StoredEmbedding, StoredMemory
from mindbridge.kernel.content import (
    PreparedContent,
    PreparedMemory,
    embedding_row_id,
    prepare_memory,
    resolve_prepared_memory_ids,
    stored_memory_context,
    text_selector_map,
)
from mindbridge.kernel.contracts import Backends, fallback_unsupported
from mindbridge.kernel.derived import (
    NAME_BINDING,
    description_sections,
    has_stream_description,
    prepared_from_stored,
    speaker_labels,
    split_description,
    with_stream_description,
    with_stream_transcript,
)
from mindbridge.kernel.embedding import DOCUMENT_TASK, Embedding
from mindbridge.kernel.formation import Formation
from mindbridge.kernel.hydration import Hydrator
from mindbridge.kernel.identity import Identities
from mindbridge.kernel.lifecycle import Lifecycle, OperationAssets
from mindbridge.kernel.materialization import Materializer
from mindbridge.kernel.projection import Projection
from mindbridge.kernel.runtime import Storage, translate_storage_errors
from mindbridge.kernel.settings import Settings
from mindbridge.kernel.speech import Speech
from mindbridge.kernel.tracing import Traced
from mindbridge.kernel.validation import (
    MAX_TEXT_CHARACTERS,
    memory_id_filter,
    strict_bool,
    validate_limit,
    validated_memory_type,
)
from mindbridge.kernel.vision import Vision
from mindbridge.types import (
    AssetRef,
    Blob,
    ContentInput,
    MemoryContext,
    MemoryKind,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    PendingCapture,
    StreamInput,
)

_LOGGER = logging.getLogger(__name__)


STREAM_GROUP_ITEMS = 32


STREAM_GROUP_SECONDS = 0.25


_T = TypeVar("_T")


def _batch_values(
    values: Sequence[_T | None] | None,
    count: int,
    name: str,
) -> tuple[_T | None, ...]:
    if values is None:
        return (None,) * count
    if isinstance(values, (str, bytes, Mapping)):
        raise ValidationError(f"{name} must contain one value per content")
    try:
        batch = tuple(values)
    except TypeError:
        raise ValidationError(f"{name} must contain one value per content") from None
    if len(batch) != count:
        raise ValidationError(f"{name} must contain one value per content")
    return batch


class Ingestion(Traced):
    """Commit observations: blocking adds, deferred captures, settlement, and streams."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        embedding: Embedding,
        projection: Projection,
        vision: Vision,
        formation: Formation,
        identities: Identities,
    ) -> None:
        super().__init__(tracer)
        self._settle_lock = storage.settle_lock
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._embedding = embedding
        self._projection = projection
        self._vision = vision
        self._formation = formation
        self._identities = identities

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return self._add_one(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def _add_one(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
        context: ObservationContext | None,
        transcript: str | None = None,
        description: str | None = None,
    ) -> MemoryRecord:
        with self._trace("mindbridge.add", kind="operation"), self._lifecycle.operation() as assets:
            if self._backends.former is not None and context is None:
                context = ObservationContext()
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared_content = self._materializer.prepare(content, assets)
                if transcript is not None:
                    prepared_content = with_stream_transcript(prepared_content, transcript)
                if description is not None:
                    prepared_content = with_stream_description(prepared_content, description)
                    self._stage_stream_description(prepared_content, description, assets)
            prepared = prepare_memory(
                prepared_content,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
                context=context,
            )
            record = self._add_prepared((prepared,), operation=assets)[0]
            self._formation.form_sources((record,), operation=assets)
            self._complete_formation((record,))
            return record

    def add_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._add_one(
            item.content,
            occurred_at=item.occurred_at,
            occurred_end=item.occurred_end,
            metadata=item.metadata,
            memory_type=item.memory_type,
            context=item.context,
            transcript=item.transcript,
            description=item.description,
        )

    def _stream_record(
        self,
        content: ContentInput | StreamInput,
        *,
        capture: bool,
    ) -> MemoryRecord:
        if isinstance(content, StreamInput):
            if capture:
                return self.capture_stream_input(content)
            return self.add_stream_input(content)
        if capture:
            return self.capture(content)
        return self.add(content)

    def capture_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._capture_one(
            item.content,
            occurred_at=item.occurred_at,
            occurred_end=item.occurred_end,
            metadata=item.metadata,
            memory_type=item.memory_type,
            context=item.context,
            transcript=item.transcript,
            description=item.description,
        )

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]:
        with (
            self._trace("mindbridge.add_many", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            normalized_memory_type = validated_memory_type(memory_type)
            if isinstance(contents, (str, bytes, Path, Blob, AssetRef, Mapping)):
                raise ValidationError("contents must be a sequence of memory inputs")
            try:
                batch = tuple(contents)
            except TypeError:
                raise ValidationError("contents must be a sequence of memory inputs") from None
            occurrences = _batch_values(occurred_at, len(batch), "occurred_at")
            occurrence_ends = _batch_values(occurred_end, len(batch), "occurred_end")
            metadata_values = _batch_values(metadata, len(batch), "metadata")
            context_values = _batch_values(context, len(batch), "context")
            if self._backends.former is not None:
                context_values = tuple(
                    value if value is not None else ObservationContext() for value in context_values
                )
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = tuple(
                    self._materializer.prepare_batch_item(
                        content,
                        index=index,
                        operation=assets,
                        occurred_at=event_time,
                        occurred_end=event_end,
                        metadata=item_metadata,
                        memory_type=normalized_memory_type,
                        context=item_context,
                    )
                    for index, (
                        content,
                        event_time,
                        event_end,
                        item_metadata,
                        item_context,
                    ) in enumerate(
                        zip(
                            batch,
                            occurrences,
                            occurrence_ends,
                            metadata_values,
                            context_values,
                            strict=True,
                        )
                    )
                )
            if not prepared:
                return ()
            records = self._add_prepared(prepared, operation=assets)
            self._formation.form_sources(records, operation=assets)
            self._complete_formation(records)
            return records

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return self._capture_one(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def _capture_one(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
        context: ObservationContext | None,
        transcript: str | None = None,
        description: str | None = None,
    ) -> MemoryRecord:
        with (
            self._trace("mindbridge.capture", kind="operation"),
            self._lifecycle.operation() as assets,
        ):
            if self._backends.former is not None and context is None:
                context = ObservationContext()
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared_content = self._materializer.prepare(content, assets)
                # Folded in here, exactly as `_add_one` folds them, so a streaming FINAL that
                # already carries ASR or caption text keeps the same content-addressed ID on
                # either commit path and `settle()` sees the marker instead of paying again.
                if transcript is not None:
                    prepared_content = with_stream_transcript(prepared_content, transcript)
                if description is not None:
                    prepared_content = with_stream_description(prepared_content, description)
                    self._stage_stream_description(prepared_content, description, assets)
            prepared = prepare_memory(
                prepared_content,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
                context=context,
            )
            if prepared.legacy_memory_id is not None:
                with (
                    self._trace("mindbridge.storage.lookup", kind="stage"),
                    translate_storage_errors("check existing memory"),
                ):
                    candidates = self._store.records.read_memories(
                        (prepared.memory_id, prepared.legacy_memory_id)
                    )
                prepared = resolve_prepared_memory_ids((prepared,), candidates)[0]
            # Reject here what `add()` would reject, before the commit. Committing media no
            # configured model can ever take would make the record durable and then fail every
            # `settle()` forever, which is a queue entry no retry ceiling should have to absorb.
            # Unconditional: `add()` rejects twice, once early and once inside
            # `_embedding_content`, so guarding this on one capability would miss the second.
            # With an embedder that takes everything it is a no-op.
            fallback_unsupported(
                prepared.content,
                self._backends.embedding_capabilities,
                "embedding",
                rescuable=self._vision.deferred_rescue(prepared.content.assets),
            )
            now = datetime.now(timezone.utc)
            # No index lock: a capture enqueues no vectors, so it has no outbox work to drain and
            # never has to wait behind a Zvec flush.
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                translate_storage_errors("capture memory"),
            ):
                self._store.captures.write_captures(
                    (
                        StoredMemory(
                            memory_id=prepared.memory_id,
                            content=prepared.content.text,
                            modality=prepared.content.modality.value,
                            memory_type=prepared.memory_type.value,
                            assets=prepared.content.assets,
                            metadata_json=prepared.metadata_json,
                            occurred_at=prepared.occurred_at,
                            occurred_end=prepared.occurred_end,
                            created_at=now,
                            updated_at=now,
                            place_id=prepared.place_id,
                            context=stored_memory_context(prepared, recorded_at=now),
                        ),
                    ),
                    enqueued_at=now,
                )
            with (
                self._trace("mindbridge.storage.hydrate", kind="stage"),
                translate_storage_errors("hydrate captured memory"),
            ):
                authoritative = self._store.records.read_memories((prepared.memory_id,))
            if not authoritative:
                raise StorageError("captured memory could not be read from SQLite")
            assets.persisted.update(asset.asset_id for asset in authoritative[0].assets)
            # A stream's caption arrives with the capture and its asset now exists, so the cache
            # row is written here rather than waiting for a `settle()` that will see the marker
            # and skip the asset entirely. A no-op for every capture that carries no caption.
            self._vision.persist_descriptions(assets)
            return self._hydrator.memory_record(authoritative[0])

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        with (
            self._trace("mindbridge.settle", kind="operation") as span,
            self._lifecycle.operation() as assets,
        ):
            validate_limit(limit, maximum=100)
            if (
                isinstance(max_attempts, bool)
                or not isinstance(max_attempts, int)
                or max_attempts < 1
            ):
                raise ValidationError("max_attempts must be a positive integer")
            memory_ids = memory_id_filter(memory_ids)
            with self._settle_lock:
                with translate_storage_errors("read the capture queue"):
                    # Same filter `pending_captures()` publishes, so naming records reuses the
                    # queue read rather than adding a second way to select them.
                    queued = self._store.captures.pending_captures(
                        limit=limit,
                        memory_ids=memory_ids,
                        max_attempts=None if memory_ids is not None else max_attempts,
                    )
                span.set_attribute(CAPTURE_SETTLED, 0)
                span.set_attribute(CAPTURE_FAILED, 0)
                if not queued:
                    return 0
                with translate_storage_errors("hydrate captured memories"):
                    rows = self._store.records.read_memories(tuple(row.memory_id for row in queued))
                settled, failures = self._settle_stored(rows, operation=assets)
                span.set_attribute(CAPTURE_SETTLED, len(settled))
                span.set_attribute(CAPTURE_FAILED, len(failures))
            enqueued_at = {row.memory_id: row.enqueued_at for row in queued}
            now = datetime.now(timezone.utc)
            waits = tuple(
                (now - enqueued_at[memory_id]).total_seconds() * 1000.0 for memory_id in settled
            )
            if waits:
                span.set_attribute(CAPTURE_TIME_TO_SEARCHABLE, max(waits))
            if failures:
                # Only the first is raised, and a row that reaches `max_attempts` is then skipped
                # by the queue read rather than retried, so without this the rest are silent.
                _LOGGER.warning(
                    "settle left %d captured record(s) unfinished; first: %s",
                    len(failures),
                    failures[0],
                )
                raise failures[0]
            return len(settled)

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        with (
            self._trace("mindbridge.pending_captures", kind="operation"),
            self._lifecycle.operation(),
        ):
            validate_limit(limit, maximum=100)
            memory_ids = memory_id_filter(memory_ids)
            with translate_storage_errors("read the capture queue"):
                return self._store.captures.pending_captures(limit=limit, memory_ids=memory_ids)

    def add_stream(
        self,
        contents: Iterable[ContentInput | StreamInput],
        *,
        capture: bool = False,
    ) -> Iterator[MemoryRecord]:
        if isinstance(contents, (str, bytes, Path, Blob, AssetRef, Mapping)):
            raise ValidationError("contents must be an iterable of memory inputs")
        capture = strict_bool(capture, "capture")
        try:
            iterator = iter(contents)
        except TypeError:
            raise ValidationError("contents must be an iterable of memory inputs") from None
        index = 0
        grouped = 0
        group_started = perf_counter()
        # A capture-routed item bypasses grouping: `capture()` writes no vectors and enqueues no
        # outbox work, so there is nothing for a group to batch. Deferring index commits for it
        # would only postpone work some other caller left pending, and the empty forced drain at
        # each boundary would put a Zvec lock back on the path capture exists to keep clear.
        while True:
            try:
                content = next(iterator)
            except StopIteration:
                break
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            with self._projection.deferred(defer=not capture):
                record = self._add_stream_item(content, index=index, capture=capture)
            if not capture:
                grouped, group_started = self._advance_stream_group(grouped + 1, group_started)
            yield record
            index += 1
        if grouped:
            self._flush_stream_group()

    def _add_stream_item(
        self,
        content: ContentInput | StreamInput,
        *,
        index: int,
        capture: bool,
    ) -> MemoryRecord:
        try:
            return self._stream_record(content, capture=capture)
        except MindBridgeError as error:
            if error.subject is None:
                error.subject = f"contents[{index}]"
            raise

    def _advance_stream_group(self, grouped: int, group_started: float) -> tuple[int, float]:
        """Close the group once either bound is reached, and report its new count and start."""
        if grouped >= STREAM_GROUP_ITEMS or perf_counter() - group_started >= STREAM_GROUP_SECONDS:
            self._flush_stream_group()
            return 0, perf_counter()
        return grouped, group_started

    def _flush_stream_group(self) -> None:
        """Apply to the index exactly the outbox rows the group's commits left pending."""
        with self._write_lock:
            self._projection.drain(force=True)

    def _add_prepared(
        self,
        prepared: Sequence[PreparedMemory],
        *,
        operation: OperationAssets,
    ) -> tuple[MemoryRecord, ...]:
        lookup_ids = tuple(
            dict.fromkeys(
                candidate
                for memory in prepared
                for candidate in (memory.memory_id, memory.legacy_memory_id)
                if candidate is not None
            )
        )
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            translate_storage_errors("check existing memories"),
        ):
            candidates = self._store.records.read_memories(lookup_ids)
        prepared = resolve_prepared_memory_ids(prepared, candidates)
        unique = {memory.memory_id: memory for memory in prepared}
        ordered_ids = tuple(unique)
        candidates_by_id = {row.memory_id: row for row in candidates}
        existing_rows = tuple(
            candidates_by_id[memory_id]
            for memory_id in ordered_ids
            if memory_id in candidates_by_id
        )
        existing_ids = set(candidates_by_id) & set(ordered_ids)
        missing = [unique[memory_id] for memory_id in ordered_ids if memory_id not in existing_ids]
        # Settled before this write takes its own speech-index guard: the two batches are
        # independent, and nesting the guards would share one rollback list between them.
        self._settle_queued(existing_rows, operation=operation)
        speech_indexed = self._settings.index_speech and any(
            self._speech.answer_speech_assets(memory.content.assets) for memory in missing
        )
        with self._speech_index_guard(operation, enabled=speech_indexed):
            stored_memories: tuple[StoredMemory, ...] = ()
            stored_embeddings: tuple[StoredEmbedding, ...] = ()
            if missing:
                if speech_indexed:
                    self._speech.recognize(
                        self._speech.answer_speech_assets(
                            tuple(asset for memory in missing for asset in memory.content.assets)
                        ),
                        operation,
                        reversible=True,
                    )
                batched = tuple(asset for memory in missing for asset in memory.content.assets)
                # Derived visual text can rescue media this embedder cannot take, so it has to
                # exist before the fallback guard below decides the write is impossible.
                described = self._vision.pending_descriptions(
                    tuple(memory.content for memory in missing),
                    operation,
                )
                self._stage_speaker_names(described, operation)
                missing = [
                    replace(
                        memory,
                        content=self._with_visual_descriptions(
                            memory.content, described, operation
                        ),
                    )
                    for memory in missing
                ]
                fallback = Modality.AUDIO not in self._backends.embedding_capabilities
                if fallback:
                    for memory in missing:
                        fallback_unsupported(
                            memory.content,
                            self._backends.embedding_capabilities,
                            "embedding",
                            rescuable=self._speech.transcript_fallback(memory.content.assets),
                        )
                if fallback:
                    self._speech.cache_audio_transcripts(
                        tuple(
                            asset
                            for memory in missing
                            if not memory.content.audio_transcript
                            for asset in memory.content.assets
                        ),
                        operation,
                    )
                elif self._speech.derives_transcripts(batched):
                    self._speech.cache_audio_transcripts(batched, operation)
                missing = [self._embedding.prepare(memory, operation) for memory in missing]
                selector_maps = {
                    memory.memory_id: text_selector_map(
                        memory,
                        raw_observation=not isinstance(memory.context, MemoryContext),
                    )
                    for memory in missing
                }
                embedding_parts = tuple(
                    (memory, object_part, model_input)
                    for memory in missing
                    for object_part, model_input in enumerate(
                        self._embedding.inputs(memory.content)
                    )
                )
                vectors, embedding_parts = self._embedding.embed_document_parts(embedding_parts)
                now = datetime.now(timezone.utc)
                stored_memories = tuple(
                    StoredMemory(
                        memory_id=memory.memory_id,
                        content=memory.content.text,
                        modality=memory.content.modality.value,
                        memory_type=memory.memory_type.value,
                        assets=memory.content.assets,
                        metadata_json=memory.metadata_json,
                        occurred_at=memory.occurred_at,
                        occurred_end=memory.occurred_end,
                        created_at=now,
                        updated_at=now,
                        place_id=memory.place_id,
                        context=stored_memory_context(memory, recorded_at=now),
                    )
                    for memory in missing
                )
                stored_embeddings = tuple(
                    StoredEmbedding(
                        embedding_id=embedding_row_id(memory.memory_id, object_part),
                        memory_id=memory.memory_id,
                        values=vector,
                        model_id=self._backends.embedding_model,
                        space_id=self._backends.space_id,
                        task=DOCUMENT_TASK,
                        created_at=now,
                        object_part=object_part,
                        normalized=True,
                        text_selectors=selector_maps[memory.memory_id].get(model_input, ()),
                    )
                    for (memory, object_part, model_input), vector in zip(
                        embedding_parts, vectors, strict=True
                    )
                )
            if stored_memories:
                # SQLite is authoritative and uses one WAL connection per transaction. Commit
                # before taking the index lock so ordinary concurrent writers can share one drain.
                with (
                    self._trace("mindbridge.storage.write", kind="stage"),
                    translate_storage_errors("write memories"),
                ):
                    # With a former configured, the same transaction enqueues the formation this
                    # call still owes. `add()` deletes the row once it returns, so the row only
                    # outlives the commit when the process did not: the next `settle()` then
                    # finds an embedded record that owes formation and finishes it.
                    self._store.records.write_memories(
                        stored_memories,
                        stored_embeddings,
                        formation_pending_at=(
                            stored_memories[0].created_at
                            if self._backends.former is not None
                            else None
                        ),
                    )
            with self._write_lock:
                self._projection.drain()
                with (
                    self._trace("mindbridge.storage.hydrate", kind="stage"),
                    translate_storage_errors("hydrate written memories"),
                ):
                    authoritative = self._store.records.read_memories(ordered_ids)
                operation.persisted.update(
                    asset.asset_id for memory in authoritative for asset in memory.assets
                )
                if operation.speech_updates:
                    self._speech.persist_transcripts(operation)
                self._vision.persist_descriptions(operation)
        rows_by_id = {memory.memory_id: memory for memory in authoritative}
        if rows_by_id.keys() != unique.keys():
            raise StorageError("written memories could not be read from SQLite", reason="io_failed")
        # After the commit, and outside the speech-index guard: the naming assertion reindexes
        # every memory that mentions the person, and this write's own memory has to be one of
        # them -- `speaker_memory_ids` joins through `memory_assets`, which exists only now.
        self._identities.bind_speaker_names(operation)
        return tuple(
            self._hydrator.memory_record(rows_by_id[memory.memory_id]) for memory in prepared
        )

    def _complete_formation(self, records: Sequence[MemoryRecord]) -> None:
        """Clear the queue rows `_add_prepared` wrote once formation has actually run.

        A no-op without a former, and harmless for a record that was never enqueued or whose row
        `_settle_queued` already removed: the delete simply matches nothing.
        """
        if self._backends.former is None or not records:
            return
        with translate_storage_errors("complete formation"):
            self._store.captures.complete_captures(tuple(record.id for record in records))

    def _settle_queued(
        self,
        existing: Sequence[StoredMemory],
        *,
        operation: OperationAssets,
    ) -> None:
        """Settle rows `add()` found already stored but still queued from `capture()`.

        Such a row is durable and has no vectors, so `add()` owes it the enrichment `capture()`
        deferred instead of taking the "already exists" shortcut past every model.
        """
        if not existing:
            return
        with translate_storage_errors("check the capture queue"):
            queued = frozenset(
                pending.memory_id
                for pending in self._store.captures.pending_captures(
                    limit=len(existing),
                    memory_ids=tuple(row.memory_id for row in existing),
                )
            )
        # No attempt ceiling here: `add()` promises a searchable return, so a record it is being
        # asked for has to be settled however many times it has already failed.
        _settled, failures = self._settle_stored(
            tuple(row for row in existing if row.memory_id in queued),
            operation=operation,
        )
        if failures:
            raise failures[0]

    def _settle_stored(
        self,
        rows: Sequence[StoredMemory],
        *,
        operation: OperationAssets,
    ) -> tuple[tuple[str, ...], tuple[MindBridgeError, ...]]:
        """Run the stages `capture()` deferred over already-committed rows, and report both sides.

        `settle()` and the `add()` path share this routine, so a captured record reaches exactly
        the state a blocking `add()` would have left it in. Each row commits on its own and a
        failing one is collected rather than raised, so it keeps its queue row, its attempt count,
        and its reason while every other record in the batch still settles. Returns the IDs that
        settled and the failures, and the caller decides which of them to raise.

        Held under `_settle_lock`, which is the whole cross-thread guard: a `settle()` and an
        `add()` of the same captured content would otherwise both find the row unembedded and
        run every model stage over it. Serialized, the second caller reaches `_settle_row`'s
        vector check after the first committed and owes formation only.
        """
        settled: list[str] = []
        failures: list[MindBridgeError] = []
        for row in rows:
            try:
                with self._settle_lock:
                    completed = self._settle_row(row, operation=operation)
            except MindBridgeError as error:
                with translate_storage_errors("record a failed settlement"):
                    self._store.captures.record_capture_failure(row.memory_id, str(error))
                if error.subject is None:
                    error.subject = row.memory_id
                failures.append(error)
                continue
            if completed:
                settled.append(row.memory_id)
        return tuple(settled), tuple(failures)

    def _settle_row(self, row: StoredMemory, *, operation: OperationAssets) -> bool:
        """Settle one captured row, or report `False` if another writer settled it first."""
        # An `add()` that crashed between its commit and formation leaves a queue row over a
        # record that is already enriched, embedded, and indexed. Its vectors are final, so
        # settling it owes formation only; re-running the model stages would buy the same
        # vectors twice.
        with translate_storage_errors("check a captured memory's vectors"):
            embedded = (
                self._store.index.read_embedding(embedding_row_id(row.memory_id, 0)) is not None
            )
        settled = row
        if not embedded:
            enriched = self._enrich_row(row, operation=operation)
            if enriched is None:
                return False
            settled = enriched
        self._formation.form_sources((self._hydrator.memory_record(settled),), operation=operation)
        with translate_storage_errors("complete captured memory"):
            self._store.captures.complete_captures((row.memory_id,))
        return True

    def _enrich_row(
        self,
        row: StoredMemory,
        *,
        operation: OperationAssets,
    ) -> StoredMemory | None:
        """Derive, embed, and commit one captured row, or return `None` if it lost the race."""
        memory = prepared_from_stored(row)
        speech_assets = self._speech.answer_speech_assets(memory.content.assets)
        speech_indexed = self._settings.index_speech and bool(speech_assets)
        with self._speech_index_guard(operation, enabled=speech_indexed):
            if speech_indexed:
                self._speech.recognize(speech_assets, operation, reversible=True)
            # Same order as `_add_prepared`: derived visual text has to exist before the fallback
            # guard decides whether this embedder can take the media at all.
            described = self._vision.pending_descriptions((memory.content,), operation)
            self._stage_speaker_names(described, operation)
            memory = replace(
                memory,
                content=self._with_visual_descriptions(memory.content, described, operation),
            )
            if Modality.AUDIO not in self._backends.embedding_capabilities:
                # Same rescue set `_add_prepared` allows, so a composition `add()` accepts is not
                # one `settle()` refuses after the record is already durable.
                fallback_unsupported(
                    memory.content,
                    self._backends.embedding_capabilities,
                    "embedding",
                    rescuable=self._speech.transcript_fallback(memory.content.assets),
                )
                if not memory.content.audio_transcript:
                    self._speech.cache_audio_transcripts(memory.content.assets, operation)
            elif self._speech.derives_transcripts(memory.content.assets):
                self._speech.cache_audio_transcripts(memory.content.assets, operation)
            memory = self._embedding.prepare(memory, operation)
            vectors, parts = self._embedding.embed_document_parts(
                tuple(
                    (memory, object_part, model_input)
                    for object_part, model_input in enumerate(
                        self._embedding.inputs(memory.content)
                    )
                )
            )
            selector_map = text_selector_map(
                memory,
                raw_observation=(row.context is None or row.context.kind is MemoryKind.OBSERVATION),
            )
            now = datetime.now(timezone.utc)
            enriched = replace(row, content=memory.content.text, updated_at=now)
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                translate_storage_errors("settle captured memory"),
            ):
                # The captured row already carries its assets and its observation context, so
                # this commit replaces only the derived text and adds the queued vectors. The
                # queue row survives it: formation is still owed, and a former failure below must
                # leave the record retryable rather than searchable but silently unformed.
                committed = self._store.captures.settle_capture(
                    replace(enriched, context=None),
                    tuple(
                        StoredEmbedding(
                            embedding_id=embedding_row_id(row.memory_id, object_part),
                            memory_id=row.memory_id,
                            values=vector,
                            model_id=self._backends.embedding_model,
                            space_id=self._backends.space_id,
                            task=DOCUMENT_TASK,
                            created_at=now,
                            object_part=object_part,
                            normalized=True,
                            text_selectors=selector_map.get(model_input, ()),
                        )
                        for (_memory, object_part, model_input), vector in zip(
                            parts, vectors, strict=True
                        )
                    ),
                )
            with self._write_lock:
                self._projection.drain()
                operation.persisted.update(asset.asset_id for asset in row.assets)
                if operation.speech_updates:
                    self._speech.persist_transcripts(operation)
                self._vision.persist_descriptions(operation)
        # Same order as `_add_prepared`: after the commit, so the row this write just settled is
        # one of the memories the naming assertion reindexes.
        self._identities.bind_speaker_names(operation)
        return enriched if committed else None

    def _stage_speaker_names(
        self,
        descriptions: Mapping[str, str],
        operation: OperationAssets,
    ) -> None:
        """Collect the names a caption's facts stated for this write's own diarised speakers.

        A stated name is the one distillation that is more than indexed text: it re-keys the
        person in every document that mentions them, now and for every later clip that resolves
        to the same voice, which is what carries an identity across clips at all.

        Only a label this asset's own recognizer produced can be resolved, so a fact naming a
        speaker who is not in this clip is dropped rather than guessed at -- counted into
        `operation.speaker_names_refused`, the same drop tally `_bind_speaker_names` adds to and
        reports, so a fact naming a label nobody produced is not simply invisible. Cached
        descriptions are read too: re-ingesting a corpus must reach the same named store as the
        first ingest, and re-asserting a standing name is a no-op.
        """
        for asset_id, description in descriptions.items():
            segments = operation.speech_segments.get(asset_id)
            if not segments:
                continue
            identities = {label: identity for identity, label in speaker_labels(segments).items()}
            for fact in split_description(description)[1].splitlines():
                match = NAME_BINDING.fullmatch(fact.strip())
                if match is None:
                    continue
                identity_id = identities.get(f"speaker_{int(match['index'])}")
                if identity_id is None:
                    operation.speaker_names_refused += 1
                    continue
                operation.speaker_names.setdefault(identity_id, match["name"].strip())

    def _known_speaker_names(self, asset_id: str, operation: OperationAssets) -> dict[str, str]:
        """Map this asset's `speaker_N` labels to whatever name is already known for them.

        "Known now" is a name this very batch's own facts just staged (`operation.speaker_names`)
        or a name an already-registered identity carries on its recognized segments; a name a
        later write asserts is not, which is the write-time/retrieval-time split documented on
        `project_fact_labels`.
        """
        segments = operation.speech_segments.get(asset_id)
        if not segments:
            return {}
        names: dict[str, str] = {}
        for identity_id, label in speaker_labels(segments).items():
            name = operation.speaker_names.get(identity_id)
            if name is None:
                name = next(
                    (
                        segment.speaker_name
                        for segment in segments
                        if segment.speaker_id == identity_id and segment.speaker_name
                    ),
                    None,
                )
            if name:
                names[label] = name
        return names

    def _stage_stream_description(
        self,
        prepared: PreparedContent,
        description: str,
        operation: OperationAssets,
    ) -> None:
        """Cache a stream's caption for its frame, the way a described write caches one.

        The inlined marker makes `_pending_visual_descriptions` skip the asset -- the caption is
        already in this document, so re-deriving it would buy the same text twice -- and the row
        it would otherwise have written is exactly what stops the *next* memory citing this frame
        from paying for a differently worded one. Staged like any other derived caption, because
        the row's foreign key is the asset and the asset is not stored yet.
        """
        if self._backends.vision_describer is None:
            return
        for asset in prepared.assets:
            if Modality(asset.modality) in self._backends.vision_capabilities:
                operation.descriptions.setdefault(asset.asset_id, description)

    def _with_visual_descriptions(
        self,
        prepared: PreparedContent,
        descriptions: Mapping[str, str],
        operation: OperationAssets,
    ) -> PreparedContent:
        """Union derived visual text into the indexed document, whatever the embedder can take.

        Routing media to the embedder is a capability decision; what a memory's full-text document
        contains is not. Describing an image only when the embedder could not take one made the
        recommended omni composition the one that stored the empty string as its whole BM25
        document, so a stronger embedder deleted the lexical half of the dense+lexical union.
        Union does not lose here; replacement does.

        One caption becomes up to two sections: what the pixels show, and the durable facts
        distilled from them and from the clip's words. They are separate because they are
        different claims -- one is observation, the other a distillation that outlives the clip --
        and a reader that cannot tell them apart cannot correct the wrong one.
        """
        sections = tuple(
            section
            for asset in prepared.assets
            if asset.asset_id in descriptions
            and not has_stream_description(prepared.text, (asset,))
            for section in description_sections(
                asset.asset_id,
                descriptions[asset.asset_id],
                self._known_speaker_names(asset.asset_id, operation),
            )
        )
        if not sections:
            return prepared
        # A description is derived convenience, not the caller's content, so one that does not fit
        # is omitted rather than allowed to fail a write whose own text is inside the limit. The
        # asset is still stored and still embedded; each description that did land carries its
        # `[visual description:<asset_id>]` marker, so which ones are present is inspectable
        # through `get`. Failing here instead would make a long note plus an image unstorable for
        # no reason the caller could act on.
        kept: list[str] = []
        text = prepared.text
        for section in sections:
            candidate = "\n\n".join(value for value in (text, section) if value)
            if len(candidate) > MAX_TEXT_CHARACTERS:
                continue
            text = candidate
            kept.append(section)
        if not kept:
            return prepared
        sections = tuple(kept)
        return replace(
            prepared,
            text=text,
            canonical_parts=(
                *prepared.canonical_parts,
                *(("visual_description", section) for section in sections),
            ),
            visual_description=True,
        )

    @contextmanager
    def _speech_index_guard(
        self,
        operation: OperationAssets,
        *,
        enabled: bool,
    ) -> Iterator[None]:
        if not enabled:
            yield
            return
        # ponytail: serialize add-time identity matching through the memory commit; replace this
        # with a staged identity plan only if speech-indexed add throughput becomes material.
        with self._write_lock:
            try:
                yield
            except BaseException:
                with translate_storage_errors("roll back speaker recognition"):
                    for rollback in reversed(operation.speech_rollbacks):
                        self._store.media.rollback_speech(rollback)
                raise
            finally:
                operation.speech_rollbacks.clear()
