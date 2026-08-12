"""Vertical tests for the shared observe, remember, and recall path."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    EmbeddingMatch,
    EmbeddingSearch,
    FeedbackWriteResult,
    ForgetPlan,
    GeneratedAnswer,
    MemoryWriteResult,
    ObservationWriteResult,
    PresignedMediaDownload,
    ResolvedQueryMedia,
)
from mindbridge.application.recall import RecallEmbeddingQuery
from mindbridge.contracts import (
    FeedbackRequest,
    IdentityObservationInput,
    MediaObjectInput,
    ObservationStatus,
    ObserveRequest,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    DeletionTombstone,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    EnumerationLimitExceededError,
    EvidenceId,
    EvidenceSpan,
    FeedbackType,
    IdempotencyConflictError,
    IdentityKind,
    JobId,
    JobNotFoundError,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryFeedback,
    MemoryId,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
    ObjectStorageError,
    Observation,
    ObservationId,
    ObservationProcessingJob,
    SensorKind,
    TenantId,
    VerificationStatus,
    apply_memory_feedback,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class InMemoryStore:
    """Small strict test store; production uses PostgreSQL."""

    def __init__(self) -> None:
        self.observations: dict[str, tuple[str, Observation]] = {}
        self.memories: dict[str, tuple[str, MemoryRecord]] = {}
        self.evidence: dict[EvidenceId, EvidenceSpan] = {}
        self.media_objects: dict[MediaObjectId, MediaObject] = {}
        self.embedding_matches: tuple[EmbeddingMatch, ...] = ()
        self.embedding_searches: list[EmbeddingSearch] = []
        self.graph_memories: dict[tuple[EmbeddedObjectType, str], MemoryId] = {}
        self.summary_children: dict[MemoryId, tuple[MemoryId, ...]] = {}
        self.embeddings: dict[str, EmbeddingRecord] = {}
        self.feedback: dict[str, tuple[str, FeedbackWriteResult]] = {}

    async def write_observation(
        self,
        batch: ObservationBatch,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ObservationWriteResult:
        key = f"{batch.observation.tenant_id}:{idempotency_key}"
        existing = self.observations.get(key)
        if existing is not None:
            _require_same_content(existing[0], content_digest)
            return ObservationWriteResult(
                observation=existing[1],
                processing_job_id=JobId(f"job_process_{existing[1].observation_id}"),
                created=False,
            )
        self.observations[key] = (content_digest, batch.observation)
        self.evidence.update((item.evidence_id, item) for item in batch.evidence_spans)
        self.media_objects.update((item.media_object_id, item) for item in batch.media_objects)
        return ObservationWriteResult(
            observation=batch.observation,
            processing_job_id=JobId(f"job_process_{batch.observation.observation_id}"),
            created=True,
        )

    async def write_memory(
        self,
        memory: MemoryRecord,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> MemoryWriteResult:
        key = f"{memory.tenant_id}:{idempotency_key}"
        existing = self.memories.get(key)
        if existing is not None:
            _require_same_content(existing[0], content_digest)
            return MemoryWriteResult(memory=existing[1], created=False)
        if any(evidence_id not in self.evidence for evidence_id in memory.evidence_ids):
            raise DomainInvariantError("memory references unknown evidence")
        self.memories[key] = (content_digest, memory)
        return MemoryWriteResult(memory=memory, created=True)

    async def read_memory(
        self,
        tenant_id: TenantId,
        memory_id: MemoryId,
    ) -> MemoryRecord:
        for _, memory in self.memories.values():
            if memory.tenant_id == tenant_id and memory.memory_id == memory_id:
                return memory
        raise MemoryNotFoundError("memory does not exist")

    async def record_feedback(
        self,
        feedback: MemoryFeedback,
        corrected_memory: MemoryRecord | None,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> FeedbackWriteResult:
        key = f"{feedback.tenant_id}:{idempotency_key}"
        existing = self.feedback.get(key)
        if existing is not None:
            _require_same_content(existing[0], content_digest)
            return replace(existing[1], created=False)
        evolved = None
        if feedback.memory_id is not None:
            original = await self.read_memory(feedback.tenant_id, feedback.memory_id)
            evolved = apply_memory_feedback(original, feedback.feedback_type, feedback.created_at)
            if corrected_memory is not None:
                evolved = replace(evolved, superseded_at=feedback.created_at)
            for memory_key, value in self.memories.items():
                if value[1].memory_id == original.memory_id:
                    self.memories[memory_key] = (value[0], evolved)
                    break
        if corrected_memory is not None:
            self.memories[f"correction:{corrected_memory.memory_id}"] = (
                content_digest,
                corrected_memory,
            )
        result = FeedbackWriteResult(
            feedback_id=feedback.feedback_id,
            feedback_type=feedback.feedback_type,
            memory_id=feedback.memory_id,
            created_at=feedback.created_at,
            resulting_state=evolved.state if evolved is not None else None,
            resulting_strength=evolved.strength if evolved is not None else None,
            corrected_memory=corrected_memory,
            created=True,
        )
        self.feedback[key] = (content_digest, result)
        return result

    async def search_memories(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        query = request.query.text.casefold() if request.query.text is not None else None
        candidates = (
            memory
            for _, memory in self.memories.values()
            if _matches_recall_filters(memory, request)
            and (query is None or query in memory.summary.casefold())
        )
        return tuple(sorted(candidates, key=lambda memory: memory.occurred_at, reverse=True))[
            :limit
        ]

    async def list_memories_for_enumeration(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        candidates = (
            memory
            for _, memory in self.memories.values()
            if memory.memory_type is MemoryType.EPISODIC
            and _matches_recall_filters(memory, request)
        )
        return tuple(sorted(candidates, key=lambda memory: (memory.occurred_at, memory.memory_id)))[
            :limit
        ]

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        rank = {evidence_id: index for index, evidence_id in enumerate(ranked_evidence_ids)}
        candidates = (
            memory
            for _, memory in self.memories.values()
            if memory.tenant_id == request.tenant_id
            and memory.superseded_at is None
            and any(evidence_id in rank for evidence_id in memory.evidence_ids)
        )
        return tuple(
            sorted(
                candidates,
                key=lambda memory: min(rank[item] for item in memory.evidence_ids if item in rank),
            )
        )[:limit]

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        memory_by_id = {memory.memory_id: memory for _, memory in self.memories.values()}
        return tuple(
            memory_by_id[memory_id]
            for memory_id in ranked_memory_ids
            if memory_id in memory_by_id
            and _matches_recall_filters(memory_by_id[memory_id], request)
        )[:limit]

    async def search_memories_by_hierarchy(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        expanded_ids = list(dict.fromkeys(ranked_memory_ids))
        for memory_id in expanded_ids:
            for child_id in self.summary_children.get(memory_id, ()):
                if child_id not in expanded_ids:
                    expanded_ids.append(child_id)
        return await self.search_memories_by_ids(request, tuple(expanded_ids), limit=limit)

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        memory_by_id = {memory.memory_id: memory for _, memory in self.memories.values()}
        ranked_memory_ids = tuple(
            dict.fromkeys(
                self.graph_memories[(match.object_type, match.object_id)]
                for match in ranked_objects
                if (match.object_type, match.object_id) in self.graph_memories
            )
        )
        return tuple(
            memory_by_id[memory_id]
            for memory_id in ranked_memory_ids
            if memory_id in memory_by_id
            and _matches_recall_filters(memory_by_id[memory_id], request)
        )[:limit]

    async def record_memory_accesses(
        self,
        tenant_id: TenantId,
        memory_ids: tuple[MemoryId, ...],
        *,
        accessed_at: datetime,
    ) -> tuple[MemoryRecord, ...]:
        accessed: list[MemoryRecord] = []
        for memory_id in memory_ids:
            for key, (digest, memory) in self.memories.items():
                if memory.tenant_id != tenant_id or memory.memory_id != memory_id:
                    continue
                updated = replace(
                    memory,
                    useful_access_count=memory.useful_access_count + 1,
                    last_accessed_at=max(
                        accessed_at,
                        memory.last_accessed_at or accessed_at,
                    ),
                    state=(
                        MemoryState.ACTIVE if memory.state is MemoryState.COLD else memory.state
                    ),
                )
                self.memories[key] = (digest, updated)
                accessed.append(updated)
                break
        return tuple(accessed)

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        return tuple(
            self.evidence[evidence_id]
            for evidence_id in evidence_ids
            if self.evidence[evidence_id].tenant_id == tenant_id
        )

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        return tuple(
            self.media_objects[media_object_id]
            for media_object_id in media_object_ids
            if media_object_id in self.media_objects
            and self.media_objects[media_object_id].tenant_id == tenant_id
        )

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        for _, observation in self.observations.values():
            if (
                observation.tenant_id == tenant_id
                and job_id == f"job_process_{observation.observation_id}"
            ):
                return ObservationProcessingJob(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    observation_id=observation.observation_id,
                    state=JobState.PENDING,
                    attempt=0,
                    error_code=None,
                    created_at=NOW,
                    updated_at=NOW,
                )
        raise JobNotFoundError("observation processing job does not exist")

    async def prepare_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ForgetPlan:
        raise AssertionError("forget is covered by the PostgreSQL integration store")

    async def complete_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        completed_at: datetime,
    ) -> DeletionTombstone:
        raise AssertionError("forget is covered by the PostgreSQL integration store")

    async def mark_forget_failed(
        self,
        tombstone: DeletionTombstone,
        *,
        error_code: str,
    ) -> DeletionTombstone:
        raise AssertionError("forget is covered by the PostgreSQL integration store")

    async def read_deletion_tombstone(
        self,
        tenant_id: TenantId,
        tombstone_id: str,
    ) -> DeletionTombstone:
        raise AssertionError("forget is covered by the PostgreSQL integration store")

    async def list_deletion_tombstones(
        self,
        tenant_id: TenantId,
        *,
        after_tombstone_id: str | None,
        limit: int,
    ) -> tuple[DeletionTombstone, ...]:
        raise AssertionError("forget is covered by the PostgreSQL integration store")

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool:
        existing = self.embeddings.get(embedding.embedding_id)
        if existing is not None:
            if existing != embedding:
                raise DomainInvariantError("embedding ID stores different content")
            return False
        self.embeddings[embedding.embedding_id] = embedding
        return True

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        self.embedding_searches.append(search)
        return tuple(
            match for match in self.embedding_matches if match.object_type in search.object_types
        )[: search.limit]


class DeterministicMediaUrlSigner:
    """Returns openable-looking URLs without an object-storage dependency."""

    def __init__(self, *, deletion_failures: int = 0) -> None:
        self.deletion_failures = deletion_failures
        self.deleted_media_ids: list[MediaObjectId] = []

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )

    async def delete_media(self, media_object: MediaObject) -> None:
        if self.deletion_failures:
            self.deletion_failures -= 1
            raise ObjectStorageError("temporary object-store failure")
        self.deleted_media_ids.append(media_object.media_object_id)


class RecordingAnswerer:
    """Deterministic answerer proving when the model boundary is invoked."""

    def __init__(
        self,
        *,
        selected_occurrence_ids: tuple[MemoryId, ...] | None = None,
    ) -> None:
        self.calls = 0
        self.last_evidence: tuple[ResolvedEvidence, ...] = ()
        self.last_memories: tuple[MemoryRecord, ...] = ()
        self.last_query_media: tuple[ResolvedQueryMedia, ...] = ()
        self.selected_occurrence_ids = selected_occurrence_ids
        self.occurrence_batches: list[tuple[MemoryRecord, ...]] = []
        self.active_occurrence_calls = 0
        self.maximum_occurrence_concurrency = 0

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> GeneratedAnswer:
        self.calls += 1
        self.last_evidence = evidence
        self.last_memories = memories
        self.last_query_media = query_media
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        self.occurrence_batches.append(memories)
        self.active_occurrence_calls += 1
        self.maximum_occurrence_concurrency = max(
            self.maximum_occurrence_concurrency,
            self.active_occurrence_calls,
        )
        try:
            await asyncio.sleep(0)
            if self.selected_occurrence_ids is not None:
                return self.selected_occurrence_ids
            return tuple(memory.memory_id for memory in memories)
        finally:
            self.active_occurrence_calls -= 1


class RecordingRecallEmbedder:
    """Returns one valid vector while retaining the fused multimodal query."""

    query_model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
    document_model_reference = ModelReference(model_id="jina-text", revision="pinned-revision")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
    dimension = 1_024

    def __init__(self, *, memory_document_failures: int = 0) -> None:
        self.queries: list[RecallEmbeddingQuery] = []
        self.memory_documents: list[str] = []
        self.memory_document_failures = memory_document_failures

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        self.queries.append(query)
        return (1.0,) + (0.0,) * 1_023

    async def encode_memory_document(self, text: str) -> tuple[float, ...]:
        self.memory_documents.append(text)
        if self.memory_document_failures:
            self.memory_document_failures -= 1
            raise ModelUnavailableError("temporary embedding failure")
        return (1.0,) + (0.0,) * 1_023


class RecordingObservationJobPublisher:
    """Records durable job delivery without a test broker."""

    def __init__(self) -> None:
        self.calls: list[tuple[TenantId, ObservationId, JobId]] = []

    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None:
        self.calls.append((tenant_id, observation_id, job_id))


async def test_observe_is_retry_safe() -> None:
    """Retrying one edge sequence stores one observation and returns duplicate status."""
    store = InMemoryStore()
    publisher = RecordingObservationJobPublisher()
    kernel = _kernel(store, RecordingAnswerer(), job_publisher=publisher)

    first = await kernel.observe(_observe_request())
    retry = await kernel.observe(_observe_request())

    assert first.observation_id == retry.observation_id
    assert first.processing_job_id == retry.processing_job_id
    assert first.status is ObservationStatus.ACCEPTED
    assert retry.status is ObservationStatus.DUPLICATE
    assert len(store.observations) == 1
    assert len(store.evidence) == 1
    assert publisher.calls == [
        (TenantId("tenant_01"), first.observation_id, first.processing_job_id),
        (TenantId("tenant_01"), retry.observation_id, retry.processing_job_id),
    ]


async def test_observe_keeps_only_anonymous_edge_identity_metadata() -> None:
    store = InMemoryStore()
    request = _observe_request(
        identity_observations=(
            IdentityObservationInput(
                identity_id="person_device_01",
                kind=IdentityKind.FACE,
                start_ms=500,
                end_ms=3_500,
                confidence=0.91,
                model_id="insightface/buffalo_l",
                model_revision="1.0.1",
            ),
        )
    )

    await _kernel(store, RecordingAnswerer()).observe(request)

    observation = next(iter(store.observations.values()))[1]
    assert observation.identity_observations[0].identity_id == "person_device_01"
    assert observation.identity_observations[0].model_reference.revision == "1.0.1"


async def test_observation_job_status_is_tenant_scoped() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    receipt = await kernel.observe(_observe_request())

    job = await kernel.get_observation_job("tenant_01", receipt.processing_job_id)

    assert job.observation_id == receipt.observation_id
    assert job.state is JobState.PENDING
    assert job.attempt == 0
    with pytest.raises(JobNotFoundError):
        await kernel.get_observation_job("other_tenant", receipt.processing_job_id)


async def test_get_memory_is_tenant_scoped() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    remembered = await kernel.remember(_remember_request())

    found = await kernel.get_memory("tenant_01", remembered.memory_id)

    assert found.memory_id == remembered.memory_id
    assert found.trace_id.startswith("trace_")
    assert remembered.trace_id.startswith("trace_")
    with pytest.raises(MemoryNotFoundError):
        await kernel.get_memory("other_tenant", remembered.memory_id)


async def test_feedback_strengthens_and_correction_supersedes_without_model_training() -> None:
    store = InMemoryStore()
    embedder = RecordingRecallEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), recall_embedder=embedder)
    original = await kernel.remember(_remember_request(idempotency_key="original"))

    useful = await kernel.record_feedback(
        FeedbackRequest(
            tenant_id="tenant_01",
            feedback_type=FeedbackType.USEFUL,
            memory_id=original.memory_id,
            idempotency_key="useful_01",
        )
    )
    correction_request = FeedbackRequest(
        tenant_id="tenant_01",
        feedback_type=FeedbackType.CORRECTION,
        memory_id=original.memory_id,
        correction_summary="The robot put the red screwdriver in the green drawer.",
        idempotency_key="correction_01",
    )
    correction = await kernel.record_feedback(correction_request)
    retry = await kernel.record_feedback(correction_request)
    old = await kernel.get_memory("tenant_01", original.memory_id)
    recalled = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="green drawer"),
        )
    )

    assert useful.resulting_state is MemoryState.STRENGTHENED
    assert useful.resulting_strength == 1.5
    assert correction.corrected_memory_id == retry.corrected_memory_id
    assert old.superseded_at == NOW
    assert old.negative_feedback_count == 1
    assert recalled.answer == "The robot put the red screwdriver in the green drawer."
    assert [item.memory_id for item in recalled.memories] == [correction.corrected_memory_id]
    assert (
        embedder.memory_documents.count("The robot put the red screwdriver in the green drawer.")
        == 2
    )


async def test_missing_feedback_records_trace_without_inventing_a_memory() -> None:
    kernel = _kernel(InMemoryStore(), RecordingAnswerer())

    receipt = await kernel.record_feedback(
        FeedbackRequest(
            tenant_id="tenant_01",
            feedback_type=FeedbackType.MISSING,
            recall_trace_id="trace_missing",
        )
    )

    assert receipt.memory_id is None
    assert receipt.resulting_state is None
    assert receipt.resulting_strength is None


async def test_idempotency_key_rejects_different_observation() -> None:
    """A client key cannot silently alias two physical observations."""
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    await kernel.observe(_observe_request(idempotency_key="edge-request-01"))

    with pytest.raises(IdempotencyConflictError):
        await kernel.observe(_observe_request(sequence=2, idempotency_key="edge-request-01"))


async def test_observe_remember_recall_returns_openable_evidence() -> None:
    """The first full path answers from a memory and its exact media span."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    memory = await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    fetched = await kernel.get_memory("tenant_01", memory.memory_id)

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
        )
    )

    assert memory.verification_status is VerificationStatus.ATTESTED
    assert memory.evidence[0].evidence_id == evidence_id
    assert fetched.evidence == memory.evidence
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.evidence[0].evidence_id == evidence_id
    assert result.evidence[0].end_ms == 4_000
    assert result.evidence[0].media_url == "https://objects.example.test/media_01"
    assert answerer.last_evidence[0].media_object.kind is MediaKind.VIDEO
    assert answerer.calls == 1


async def test_remember_retry_returns_original_record() -> None:
    """A retry cannot replace the system creation time of an existing memory."""
    store = InMemoryStore()
    request = _remember_request(idempotency_key="remember-01")
    first_kernel = _kernel(store, RecordingAnswerer())
    later_kernel = _kernel(store, RecordingAnswerer(), clock=lambda: NOW + timedelta(minutes=5))

    first = await first_kernel.remember(request)
    retry = await later_kernel.remember(request)

    assert retry.memory_id == first.memory_id
    assert retry.created_at == NOW


async def test_remember_indexes_one_pinned_memory_document() -> None:
    store = InMemoryStore()
    recall_embedder = RecordingRecallEmbedder()
    memory = await _kernel(
        store,
        RecordingAnswerer(),
        recall_embedder=recall_embedder,
    ).remember(_remember_request(idempotency_key="remember-index-01"))

    embedding = next(iter(store.embeddings.values()))
    assert recall_embedder.memory_documents == [memory.summary]
    assert embedding.object_type is EmbeddedObjectType.MEMORY_RECORD
    assert embedding.object_id == memory.memory_id
    assert embedding.model_reference == recall_embedder.document_model_reference
    assert embedding.space_reference == recall_embedder.space_reference
    assert embedding.task == "retrieval_document"
    assert embedding.dimension == 1_024
    assert embedding.normalized is True
    assert embedding.created_at == memory.created_at


async def test_remember_retry_repairs_embedding_after_model_failure() -> None:
    store = InMemoryStore()
    recall_embedder = RecordingRecallEmbedder(memory_document_failures=1)
    kernel = _kernel(store, RecordingAnswerer(), recall_embedder=recall_embedder)
    request = _remember_request(idempotency_key="remember-repair-01")

    with pytest.raises(ModelUnavailableError, match="temporary embedding failure"):
        await kernel.remember(request)
    repaired = await kernel.remember(request)

    assert len(store.memories) == 1
    assert len(store.embeddings) == 1
    assert next(iter(store.embeddings.values())).object_id == repaired.memory_id


async def test_hidden_evidence_is_still_used_for_answering() -> None:
    """Response shaping cannot bypass evidence inspection by the answer model."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
            include_evidence=False,
        )
    )

    assert result.evidence == ()
    assert answerer.last_evidence[0].evidence_span.evidence_id == evidence_id


async def test_recall_without_candidates_abstains_without_model_call() -> None:
    """No evidence means no guessed answer and no unnecessary model cost."""
    answerer = RecordingAnswerer()
    kernel = _kernel(InMemoryStore(), answerer)

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="missing"))
    )

    assert result.answer is None
    assert result.confidence == 0.0
    assert answerer.calls == 0


async def test_recall_answers_from_attested_source_memory() -> None:
    """Explicit remembered text is usable without being labeled as sensor-verified."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    await kernel.remember(_remember_request())

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="red screwdriver"))
    )

    assert len(result.memories) == 1
    assert result.memories[0].verification_status is VerificationStatus.ATTESTED
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.confidence == 0.9
    assert answerer.calls == 1


async def test_recall_records_access_and_reactivates_cold_memory() -> None:
    store = InMemoryStore()
    memory = await _kernel(store, RecordingAnswerer()).remember(_remember_request())
    key = next(iter(store.memories))
    digest, stored = store.memories[key]
    store.memories[key] = (digest, replace(stored, state=MemoryState.COLD))
    accessed_at = NOW + timedelta(minutes=1)

    result = await _kernel(
        store,
        RecordingAnswerer(),
        clock=lambda: accessed_at,
    ).recall(RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="red screwdriver")))

    assert result.memories[0].memory_id == memory.memory_id
    assert result.memories[0].useful_access_count == 1
    assert result.memories[0].last_accessed_at == accessed_at
    assert result.memories[0].state is MemoryState.ACTIVE


async def test_recall_answers_and_records_access_only_for_returned_memories() -> None:
    """Hidden fusion candidates cannot support an unverifiable answer or lifecycle signal."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    hidden = await kernel.remember(_remember_request(idempotency_key="hidden"))
    visible = await kernel.remember(
        _remember_request(idempotency_key="visible").model_copy(
            update={"occurred_at": NOW + timedelta(seconds=1)}
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
            limit=1,
        )
    )

    assert [memory.memory_id for memory in result.memories] == [visible.memory_id]
    assert [memory.memory_id for memory in answerer.last_memories] == [visible.memory_id]
    assert (
        await store.read_memory(TenantId("tenant_01"), MemoryId(hidden.memory_id))
    ).useful_access_count == 0


async def test_recall_abstains_from_unverified_derived_summary() -> None:
    """Unsupported derived content remains searchable but cannot ground an answer."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    memory = MemoryRecord(
        memory_id=MemoryId("memory_unverified"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.SEMANTIC,
        summary="An unsupported derived summary.",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    await store.write_memory(memory, idempotency_key="unverified", content_digest="a" * 64)

    result = await _kernel(store, answerer).recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="unsupported derived"))
    )

    assert len(result.memories) == 1
    assert result.answer is None
    assert answerer.calls == 0


async def test_semantic_evidence_finds_memory_without_matching_summary_text() -> None:
    """Jina evidence rank reaches grounded memories that sparse text cannot see."""
    store = InMemoryStore()
    recall_embedder = RecordingRecallEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), recall_embedder=recall_embedder)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_01",
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=evidence_id,
            similarity=0.8,
        ),
    )

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="a hand moves an object"))
    )

    assert result.memories[0].summary.startswith("The robot put")
    assert recall_embedder.queries[0].text == "a hand moves an object"
    assert store.embedding_searches[0].document_task == "retrieval_document"


async def test_semantic_memory_finds_attested_text_without_matching_words() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    memory = await kernel.remember(_remember_request())
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_memory_01",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory.memory_id,
            similarity=0.8,
        ),
    )

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="where is the hand tool?"))
    )

    assert [item.memory_id for item in result.memories] == [memory.memory_id]
    assert result.answer == memory.summary


async def test_semantic_summary_hit_descends_to_attested_source_for_answering() -> None:
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    child = await _kernel(store, answerer).remember(_remember_request())
    parent = MemoryRecord(
        memory_id=MemoryId("summary_parent"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.SEMANTIC,
        summary="A generated navigation summary.",
        evidence_ids=(),
        occurred_at=child.occurred_at,
        ended_at=child.ended_at,
        created_at=NOW,
        verification_status=VerificationStatus.UNVERIFIED,
        model_reference=ModelReference(model_id="omni", revision="summary-revision"),
    )
    store.memories["summary"] = ("a" * 64, parent)
    store.summary_children[parent.memory_id] = (MemoryId(child.memory_id),)
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_summary",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=parent.memory_id,
            similarity=0.9,
        ),
    )

    result = await _kernel(store, answerer).recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="a semantic paraphrase"))
    )

    assert [memory.memory_id for memory in result.memories] == [
        parent.memory_id,
        child.memory_id,
    ]
    assert [memory.memory_id for memory in answerer.last_memories] == [child.memory_id]
    assert result.answer == child.summary


async def test_recall_scopes_followup_to_explicit_memory_ids() -> None:
    store = InMemoryStore()
    recall_embedder = RecordingRecallEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), recall_embedder=recall_embedder)
    await kernel.remember(_remember_request(idempotency_key="first"))
    scoped = await kernel.remember(
        _remember_request(idempotency_key="second").model_copy(
            update={"summary": "The robot later closed the green drawer."}
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="What happened next?"),
            memory_ids=(scoped.memory_id,),
        )
    )

    assert [memory.memory_id for memory in result.memories] == [scoped.memory_id]
    assert result.answer == scoped.summary
    assert recall_embedder.queries == []


async def test_semantic_event_follows_its_memory_representation() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    memory = await kernel.remember(_remember_request())
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_event_01",
            object_type=EmbeddedObjectType.EVENT,
            object_id="event_01",
            similarity=0.82,
        ),
    )
    store.graph_memories[(EmbeddedObjectType.EVENT, "event_01")] = MemoryId(memory.memory_id)

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="an unrelated paraphrase"))
    )

    assert [item.memory_id for item in result.memories] == [memory.memory_id]
    assert any(
        search.object_types == (EmbeddedObjectType.EVENT, EmbeddedObjectType.CLAIM)
        for search in store.embedding_searches
    )


async def test_media_query_is_signed_for_embedding_instead_of_used_as_a_filter() -> None:
    """One resolved AV query reaches both Jina retrieval and Omni inspection."""
    store = InMemoryStore()
    recall_embedder = RecordingRecallEmbedder()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer, recall_embedder=recall_embedder)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_01",
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=evidence_id,
            similarity=0.9,
        ),
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(media_object_ids=("media_01",)),
        )
    )

    resolved = recall_embedder.queries[0].media[0]
    assert resolved.media_object.kind is MediaKind.VIDEO
    assert resolved.media_url == "https://objects.example.test/media_01"
    assert answerer.last_query_media == (resolved,)
    assert result.memories[0].evidence_ids == (evidence_id,)


async def test_media_query_rejects_unknown_tenant_object() -> None:
    """A query cannot turn another tenant's object ID into a signed URL."""
    recall_embedder = RecordingRecallEmbedder()
    kernel = _kernel(InMemoryStore(), RecordingAnswerer(), recall_embedder=recall_embedder)

    with pytest.raises(DomainInvariantError, match="unknown media"):
        await kernel.recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(media_object_ids=("media_missing",)),
            )
        )

    assert recall_embedder.queries == []


async def test_enumerate_verifies_all_filtered_memories_in_bounded_batches() -> None:
    store = InMemoryStore()
    await _write_attested_memories(store, 125)
    answerer = RecordingAnswerer()

    result = await _kernel(store, answerer).recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="words absent from every summary"),
            mode=RecallMode.ENUMERATE,
            limit=1,
        )
    )

    assert result.answer == "125"
    assert len(result.memories) == 125
    assert [memory.occurred_at for memory in result.memories] == sorted(
        memory.occurred_at for memory in result.memories
    )
    assert len(answerer.occurrence_batches) == 8
    assert max(map(len, answerer.occurrence_batches)) == 16
    assert answerer.maximum_occurrence_concurrency == 4
    assert answerer.calls == 0
    assert all(memory.useful_access_count == 1 for memory in result.memories)


async def test_enumerate_rejects_model_ids_outside_the_candidate_batch() -> None:
    store = InMemoryStore()
    await _write_attested_memories(store, 1)
    answerer = RecordingAnswerer(selected_occurrence_ids=(MemoryId("memory_not_a_candidate"),))

    with pytest.raises(ModelOutputError, match="invalid memory IDs"):
        await _kernel(store, answerer).recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="count the events"),
                mode=RecallMode.ENUMERATE,
            )
        )


async def test_enumerate_refuses_to_silently_truncate_a_broad_scope() -> None:
    store = InMemoryStore()
    await _write_attested_memories(store, 1_001)
    answerer = RecordingAnswerer()

    with pytest.raises(EnumerationLimitExceededError, match="narrow"):
        await _kernel(store, answerer).recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="count every event"),
                mode=RecallMode.ENUMERATE,
            )
        )

    assert answerer.occurrence_batches == []


async def test_enumerate_respects_explicit_memory_scope() -> None:
    store = InMemoryStore()
    await _write_attested_memories(store, 3)

    result = await _kernel(store, RecordingAnswerer()).recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="count this context"),
            memory_ids=("memory_0002", "memory_0000"),
            mode=RecallMode.ENUMERATE,
        )
    )

    assert result.answer == "2"
    assert [memory.memory_id for memory in result.memories] == [
        "memory_0000",
        "memory_0002",
    ]


def test_generated_answer_rejects_confidence_without_answer() -> None:
    """Abstention cannot report misleading answer confidence."""
    with pytest.raises(DomainInvariantError, match="zero"):
        GeneratedAnswer(answer=None, confidence=0.9)


def _observe_request(
    *,
    sequence: int = 1,
    idempotency_key: str | None = None,
    identity_observations: tuple[IdentityObservationInput, ...] = (),
) -> ObserveRequest:
    return ObserveRequest(
        tenant_id="tenant_01",
        device_id="device_01",
        boot_id="boot_01",
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id="media_01",
                kind=MediaKind.VIDEO,
                uri="s3://memories/video.mp4",
                sha256="a" * 64,
                size_bytes=100,
                created_at=NOW,
                duration_ms=4_000,
            ),
        ),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=4),
        observed_at=NOW,
        identity_observations=identity_observations,
        idempotency_key=idempotency_key,
    )


async def _write_attested_memories(store: InMemoryStore, count: int) -> None:
    for ordinal in reversed(range(count)):
        occurred_at = NOW + timedelta(seconds=ordinal)
        memory = MemoryRecord(
            memory_id=MemoryId(f"memory_{ordinal:04d}"),
            tenant_id=TenantId("tenant_01"),
            memory_type=MemoryType.EPISODIC,
            summary=f"Unrelated attested occurrence {ordinal}",
            evidence_ids=(),
            occurred_at=occurred_at,
            ended_at=occurred_at,
            created_at=occurred_at,
            verification_status=VerificationStatus.ATTESTED,
        )
        await store.write_memory(
            memory,
            idempotency_key=f"enumeration_{ordinal}",
            content_digest="e" * 64,
        )


def _remember_request(
    *,
    evidence_ids: tuple[str, ...] = (),
    idempotency_key: str | None = None,
) -> RememberRequest:
    return RememberRequest(
        tenant_id="tenant_01",
        summary="The robot put the red screwdriver beside the blue toolbox.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=NOW,
        evidence_ids=evidence_ids,
        idempotency_key=idempotency_key,
    )


def _kernel(
    store: InMemoryStore,
    answerer: RecordingAnswerer,
    *,
    job_publisher: RecordingObservationJobPublisher | None = None,
    recall_embedder: RecordingRecallEmbedder | None = None,
    clock: Callable[[], datetime] = lambda: NOW,
) -> MemoryKernel:
    media_access = DeterministicMediaUrlSigner()
    return MemoryKernel(
        store,
        answerer,
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=job_publisher or RecordingObservationJobPublisher(),
        recall_embedder=recall_embedder or RecordingRecallEmbedder(),
        clock=clock,
    )


def _require_same_content(stored_digest: str, requested_digest: str) -> None:
    if stored_digest != requested_digest:
        raise IdempotencyConflictError("idempotency key already stores different content")


def _matches_recall_filters(memory: MemoryRecord, request: RecallRequest) -> bool:
    return (
        memory.tenant_id == request.tenant_id
        and memory.superseded_at is None
        and (not request.filters.memory_types or memory.memory_type in request.filters.memory_types)
        and (
            request.filters.occurred_after is None
            or memory.occurred_at >= request.filters.occurred_after
        )
        and (
            request.filters.occurred_before is None
            or memory.occurred_at <= request.filters.occurred_before
        )
    )
