"""Vertical tests for the shared observe, remember, and recall path."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import recall as recall_module
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
    EmbeddingId,
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
from mindbridge.models import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    MediaPart,
    Reranker,
    RerankRequest,
    RerankResult,
    TextPart,
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
        roots = tuple(dict.fromkeys(ranked_memory_ids))
        ranks = {memory_id: (0, root_rank) for root_rank, memory_id in enumerate(roots)}
        pending = [
            (memory_id, 0, root_rank, frozenset({memory_id}))
            for root_rank, memory_id in enumerate(roots)
        ]
        while pending:
            memory_id, depth, root_rank, path = pending.pop()
            if depth >= 16:
                continue
            for child_id in self.summary_children.get(memory_id, ()):
                if child_id in path:
                    continue
                ranks[child_id] = min(ranks.get(child_id, (17, len(roots))), (depth + 1, root_rank))
                pending.append((child_id, depth + 1, root_rank, path | {child_id}))
        for root_rank, memory_id in enumerate(roots):
            for parent_id, child_ids in self.summary_children.items():
                if memory_id not in child_ids:
                    continue
                ranks[parent_id] = min(ranks.get(parent_id, (17, len(roots))), (1, root_rank))
                for sibling_id in child_ids[:limit]:
                    if sibling_id != memory_id:
                        ranks[sibling_id] = min(
                            ranks.get(sibling_id, (17, len(roots))), (2, root_rank)
                        )
        expanded_ids = tuple(sorted(ranks, key=lambda memory_id: (*ranks[memory_id], memory_id)))
        return await self.search_memories_by_ids(request, expanded_ids, limit=limit)

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

    async def has_embedding(self, tenant_id: TenantId, embedding_id: EmbeddingId) -> bool:
        embedding = self.embeddings.get(embedding_id)
        return embedding is not None and embedding.tenant_id == tenant_id

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


class SequencedMediaUrlSigner(DeterministicMediaUrlSigner):
    """Makes each model-stage and response signature distinguishable."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.calls += 1
        return PresignedMediaDownload(
            download_url=(
                f"https://objects.example.test/{media_object.media_object_id}"
                f"?signature={self.calls}"
            ),
            expires_at=NOW + timedelta(minutes=5),
        )


class RecordingAnswerer:
    """Deterministic answerer proving when the model boundary is invoked."""

    def __init__(
        self,
        *,
        selected_occurrence_ids: tuple[MemoryId, ...] | None = None,
        answers: tuple[GeneratedAnswer, ...] = (),
    ) -> None:
        self.calls = 0
        self.last_evidence: tuple[ResolvedEvidence, ...] = ()
        self.last_memories: tuple[MemoryRecord, ...] = ()
        self.last_query_media: tuple[ResolvedQueryMedia, ...] = ()
        self.selected_occurrence_ids = selected_occurrence_ids
        self.answers = list(answers)
        self.occurrence_batches: list[tuple[MemoryRecord, ...]] = []
        self.occurrence_evidence: list[tuple[ResolvedEvidence, ...]] = []
        self.occurrence_query_media: list[tuple[ResolvedQueryMedia, ...]] = []
        self.active_occurrence_calls = 0
        self.maximum_occurrence_concurrency = 0

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        self.calls += 1
        self.last_evidence = evidence
        self.last_memories = memories
        self.last_query_media = query_media
        self.attempted_retrieval_queries = attempted_retrieval_queries
        if self.answers:
            return self.answers.pop(0)
        if not memories:
            return GeneratedAnswer(answer=None, confidence=0.0)
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
        self.occurrence_evidence.append(evidence)
        self.occurrence_query_media.append(query_media)
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


@dataclass(frozen=True, slots=True)
class _RecordedMedia:
    media_url: str
    kind: MediaKind


@dataclass(frozen=True, slots=True)
class _RecordedQuery:
    text: str | None
    media: tuple[_RecordedMedia, ...]


class RecordingEmbedder:
    """Returns valid vectors while retaining query and document inputs."""

    model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self, *, memory_document_failures: int = 0) -> None:
        self.queries: list[_RecordedQuery] = []
        self.memory_documents: list[str] = []
        self.memory_document_failures = memory_document_failures

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        if request.task is EmbedTask.QUERY:
            for input_value in request.inputs:
                self.queries.append(
                    _RecordedQuery(
                        text=next(
                            (part.text for part in input_value.parts if isinstance(part, TextPart)),
                            None,
                        ),
                        media=tuple(
                            _RecordedMedia(part.url, part.kind)
                            for part in input_value.parts
                            if isinstance(part, MediaPart)
                        ),
                    )
                )
        else:
            self.memory_documents.extend(
                part.text
                for input_value in request.inputs
                for part in input_value.parts
                if isinstance(part, TextPart)
            )
            if self.memory_document_failures:
                self.memory_document_failures -= 1
                raise ModelUnavailableError("temporary embedding failure")
        return EmbedResult(
            tuple(
                Embedding(
                    (1.0,) + (0.0,) * 1_023,
                    self.model_reference,
                    self.space_reference,
                )
                for _ in request.inputs
            )
        )


class RecordingReranker:
    def __init__(self, result_ids: tuple[str, ...] | None = None) -> None:
        self.result_ids = result_ids
        self.requests: list[RerankRequest] = []

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.requests.append(request)
        return RerankResult(
            self.result_ids
            if self.result_ids is not None
            else tuple(reversed([item.candidate_id for item in request.candidates]))
        )


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
    assert next(iter(store.evidence.values())).end_ms == 4_000
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
    assert job.memory_ids == ()
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
    embedder = RecordingEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), embedder=embedder)
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


async def test_correction_inherits_the_retention_strength_it_supersedes() -> None:
    """The authoritative version must not be colder than the record it replaces."""
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    original = await kernel.remember(_remember_request(idempotency_key="original"))
    strengthened = await kernel.record_feedback(
        FeedbackRequest(
            tenant_id="tenant_01",
            feedback_type=FeedbackType.USEFUL,
            memory_id=original.memory_id,
            idempotency_key="useful_01",
        )
    )

    correction = await kernel.record_feedback(
        FeedbackRequest(
            tenant_id="tenant_01",
            feedback_type=FeedbackType.CORRECTION,
            memory_id=original.memory_id,
            correction_summary="The robot put the red screwdriver in the green drawer.",
            idempotency_key="correction_01",
        )
    )
    assert correction.corrected_memory_id is not None
    corrected = await kernel.get_memory("tenant_01", correction.corrected_memory_id)

    assert strengthened.resulting_strength == 1.5
    assert corrected.strength == 1.5


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


async def test_idempotency_accepts_timezone_equivalent_retry() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    key = "remember-timezone-retry"

    first = await kernel.remember(_remember_request(idempotency_key=key))
    retry = await kernel.remember(
        _remember_request(
            idempotency_key=key,
            occurred_at=NOW.astimezone(timezone(timedelta(hours=8))),
        )
    )

    assert retry.memory_id == first.memory_id


async def test_observe_remember_recall_returns_openable_evidence() -> None:
    """The first full path answers from a memory and its exact media span."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    receipt = await kernel.observe(_observe_request())
    assert receipt.evidence_ids == tuple(store.evidence)
    evidence_id = receipt.evidence_ids[0]
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
    embedder = RecordingEmbedder()
    request = _remember_request(idempotency_key="remember-01")
    first_kernel = _kernel(store, RecordingAnswerer(), embedder=embedder)
    later_kernel = _kernel(
        store,
        RecordingAnswerer(),
        embedder=embedder,
        clock=lambda: NOW + timedelta(minutes=5),
    )

    first = await first_kernel.remember(request)
    retry = await later_kernel.remember(request)

    assert retry.memory_id == first.memory_id
    assert retry.created_at == NOW
    assert embedder.memory_documents == [request.summary, request.summary]


async def test_remember_indexes_one_pinned_memory_document() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    memory = await _kernel(
        store,
        RecordingAnswerer(),
        embedder=embedder,
    ).remember(_remember_request(idempotency_key="remember-index-01"))

    embedding = next(iter(store.embeddings.values()))
    assert embedder.memory_documents == [memory.summary]
    assert embedding.object_type is EmbeddedObjectType.MEMORY_RECORD
    assert embedding.object_id == memory.memory_id
    assert embedding.model_reference == embedder.model_reference
    assert embedding.space_reference == embedder.space_reference
    assert embedding.task == "retrieval_document"
    assert embedding.dimension == 1_024
    assert embedding.normalized is True
    assert embedding.created_at == memory.created_at


async def test_remember_retry_repairs_embedding_after_model_failure() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder(memory_document_failures=1)
    kernel = _kernel(store, RecordingAnswerer(), embedder=embedder)
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


async def test_recall_without_candidates_uses_bounded_reflection_and_abstains() -> None:
    """No evidence permits one bounded query reflection but never a guessed answer."""
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="A model guess without evidence.",
                confidence=0.9,
                retrieval_queries=("missing supporting event",),
            ),
        )
    )
    kernel = _kernel(InMemoryStore(), answerer)

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="missing"))
    )

    assert result.answer is None
    assert result.confidence == 0.0
    assert answerer.calls == 2


async def test_recall_uses_bounded_model_queries_to_reach_missing_evidence() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="Maybe at home.",
                confidence=0.4,
                retrieval_queries=("person_device_01 blue toolbox",),
            ),
            GeneratedAnswer(answer="Beside the blue toolbox.", confidence=0.91),
        )
    )
    kernel = _kernel(store, answerer, embedder=embedder)
    await kernel.remember(
        _remember_request(idempotency_key="initial").model_copy(
            update={"summary": "Where did person_device_01 put the red screwdriver?"}
        )
    )
    missing = await kernel.remember(
        _remember_request(idempotency_key="bridge").model_copy(
            update={"summary": "person_device_01 blue toolbox"}
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="Where did person_device_01 put the red screwdriver?"),
        )
    )

    assert result.answer == "Beside the blue toolbox."
    assert answerer.calls == 2
    assert [query.text for query in embedder.queries] == [
        "Where did person_device_01 put the red screwdriver?",
        "person_device_01 blue toolbox",
    ]
    assert missing.memory_id in {memory.memory_id for memory in result.memories}


async def test_recall_stops_after_two_product_wide_refinement_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_attributes: list[dict[str, str | int | float | bool]] = []
    stages: list[tuple[str, float]] = []
    monkeypatch.setattr(recall_module, "set_current_span_attributes", span_attributes.append)
    monkeypatch.setattr(
        recall_module,
        "record_stage_duration",
        lambda stage, duration: stages.append((stage, duration)),
    )
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(answer=None, confidence=0.0, retrieval_queries=("bridge one",)),
            GeneratedAnswer(answer=None, confidence=0.0, retrieval_queries=("final detail",)),
            GeneratedAnswer(answer="Grounded final answer.", confidence=0.9),
        )
    )
    kernel = _kernel(store, answerer, embedder=embedder)
    for key, summary in (
        ("initial", "start question"),
        ("bridge", "bridge one"),
        ("final", "final detail"),
    ):
        await kernel.remember(
            _remember_request(idempotency_key=key).model_copy(update={"summary": summary})
        )

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="start question"))
    )

    assert result.answer == "Grounded final answer."
    assert answerer.calls == 3
    assert [query.text for query in embedder.queries] == [
        "start question",
        "bridge one",
        "final detail",
    ]
    assert [
        (attributes["mindbridge.recall.answer.phase"], attributes["mindbridge.recall.answer.round"])
        for attributes in span_attributes
        if "mindbridge.recall.answer.phase" in attributes
    ] == [("initial", 1), ("reflection", 2), ("reflection", 3)]
    assert [stage for stage, _ in stages] == ["recall.first_answer", "recall.answer_complete"]
    assert 0 <= stages[0][1] <= stages[1][1]


async def test_search_records_complete_latency_without_calling_the_answerer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[tuple[str, float]] = []
    monkeypatch.setattr(
        recall_module,
        "record_stage_duration",
        lambda stage, duration: stages.append((stage, duration)),
    )
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    remembered = await kernel.remember(_remember_request())

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
            mode=RecallMode.SEARCH,
        )
    )

    assert remembered.memory_id in {memory.memory_id for memory in result.memories}
    assert answerer.calls == 0
    assert [stage for stage, _ in stages] == ["recall.search"]


async def test_recall_switches_query_direction_when_reflection_finds_no_new_candidate() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="Beside the toolbox.",
                confidence=0.7,
                retrieval_queries=("red screwdriver exact location",),
            ),
            GeneratedAnswer(
                answer="Beside the toolbox.",
                confidence=0.7,
                retrieval_queries=("person_device_01 last placement",),
            ),
            GeneratedAnswer(answer="Beside the toolbox.", confidence=0.7),
        )
    )
    kernel = _kernel(store, answerer, embedder=embedder)
    await kernel.remember(_remember_request())

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="red screwdriver"))
    )

    assert result.answer == "Beside the toolbox."
    assert answerer.calls == 3
    assert answerer.attempted_retrieval_queries == (
        "red screwdriver exact location",
        "person_device_01 last placement",
    )
    assert [query.text for query in embedder.queries] == [
        "red screwdriver",
        "red screwdriver exact location",
        "person_device_01 last placement",
    ]


async def test_recall_applies_the_temporal_order_of_the_latest_answer_round() -> None:
    """A reflection round that discovers a latest/first question must still reorder."""
    store = InMemoryStore()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="alpha",
                confidence=0.4,
                retrieval_queries=("beta",),
            ),
            GeneratedAnswer(
                answer="alpha",
                confidence=0.5,
                retrieval_queries=("gamma",),
                temporal_order="newest",
            ),
            GeneratedAnswer(answer="alpha", confidence=0.9, temporal_order="newest"),
        )
    )
    kernel = _kernel(store, answerer)
    old = await kernel.remember(
        _remember_request(idempotency_key="old").model_copy(update={"summary": "old event"})
    )
    new = await kernel.remember(
        _remember_request(idempotency_key="new", occurred_at=NOW + timedelta(days=1)).model_copy(
            update={"summary": "new event"}
        )
    )
    store.embedding_matches = tuple(
        EmbeddingMatch(
            embedding_id=f"embedding_{memory_id}",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory_id,
            similarity=similarity,
        )
        for memory_id, similarity in ((old.memory_id, 0.9), (new.memory_id, 0.8))
    )

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="alpha"))
    )

    assert answerer.calls == 3
    assert [item.memory_id for item in result.memories] == [new.memory_id, old.memory_id]


async def test_recall_reorders_only_visible_candidates_for_an_explicit_latest_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_attributes: list[dict[str, str | int | float | bool]] = []
    monkeypatch.setattr(recall_module, "set_current_span_attributes", span_attributes.append)
    store = InMemoryStore()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="old event",
                confidence=0.5,
                temporal_order="newest",
            ),
            GeneratedAnswer(
                answer="new event",
                confidence=0.9,
                temporal_order="newest",
            ),
        )
    )
    kernel = _kernel(store, answerer)
    old = await kernel.remember(
        _remember_request(idempotency_key="old").model_copy(update={"summary": "old event"})
    )
    new = await kernel.remember(
        _remember_request(idempotency_key="new", occurred_at=NOW + timedelta(days=1)).model_copy(
            update={"summary": "new event"}
        )
    )
    unrelated = await kernel.remember(
        _remember_request(
            idempotency_key="unrelated",
            occurred_at=NOW + timedelta(days=2),
        ).model_copy(update={"summary": "unrelated event"})
    )
    store.embedding_matches = tuple(
        EmbeddingMatch(
            embedding_id=f"embedding_{memory_id}",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory_id,
            similarity=similarity,
        )
        for memory_id, similarity in (
            (old.memory_id, 0.9),
            (new.memory_id, 0.8),
            (unrelated.memory_id, 0.7),
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="What happened most recently?"),
            limit=2,
        )
    )

    assert result.answer == "new event"
    assert [memory.memory_id for memory in result.memories] == [new.memory_id, old.memory_id]
    assert answerer.calls == 2
    assert [
        (attributes["mindbridge.recall.answer.phase"], attributes["mindbridge.recall.answer.round"])
        for attributes in span_attributes
        if "mindbridge.recall.answer.phase" in attributes
    ] == [("initial", 1), ("temporal_reorder", 2)]


async def test_recall_discards_an_answer_when_the_final_visibility_barrier_removes_it() -> None:
    class ForgetBeforeAccessStore(InMemoryStore):
        async def record_memory_accesses(
            self,
            tenant_id: TenantId,
            memory_ids: tuple[MemoryId, ...],
            *,
            accessed_at: datetime,
        ) -> tuple[MemoryRecord, ...]:
            return ()

    store = ForgetBeforeAccessStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="red screwdriver"))
    )

    assert answerer.calls == 1
    assert result.answer is None
    assert result.confidence == 0.0
    assert result.memories == ()
    assert result.evidence == ()


async def test_recall_reassesses_when_reflection_only_adds_hidden_candidates() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="Beside the toolbox.",
                confidence=0.7,
                retrieval_queries=("exact location",),
            ),
            GeneratedAnswer(answer="Beside the toolbox.", confidence=0.7),
        )
    )
    kernel = _kernel(store, answerer, embedder=embedder)
    await kernel.remember(
        _remember_request(idempotency_key="hidden").model_copy(update={"summary": "exact location"})
    )
    visible = await kernel.remember(
        _remember_request(idempotency_key="visible").model_copy(
            update={
                "summary": "red screwdriver exact location",
                "occurred_at": NOW + timedelta(seconds=1),
            }
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
    assert answerer.calls == 2


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
    assert answerer.calls == 1


async def test_semantic_evidence_finds_memory_without_matching_summary_text() -> None:
    """Jina evidence rank reaches grounded memories that sparse text cannot see."""
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), embedder=embedder)
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
    assert embedder.queries[0].text == "a hand moves an object"
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


async def test_recall_uses_optional_reranker_without_changing_candidate_identity() -> None:
    store = InMemoryStore()
    reranker = RecordingReranker()
    first = await _kernel(store, RecordingAnswerer()).remember(
        _remember_request(idempotency_key="rerank-first")
    )
    second = await _kernel(store, RecordingAnswerer()).remember(
        _remember_request(idempotency_key="rerank-second")
    )
    store.embedding_matches = tuple(
        EmbeddingMatch(
            embedding_id=f"embedding_{index}",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory_id,
            similarity=1.0 - index / 10,
        )
        for index, memory_id in enumerate((first.memory_id, second.memory_id))
    )

    result = await _kernel(store, RecordingAnswerer(), reranker=reranker).recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="semantic-only query"))
    )

    assert [item.memory_id for item in result.memories] == [second.memory_id, first.memory_id]
    assert [item.candidate_id for item in reranker.requests[0].candidates] == [
        first.memory_id,
        second.memory_id,
    ]


async def test_recall_rejects_reranker_that_drops_a_candidate() -> None:
    store = InMemoryStore()
    first = await _kernel(store, RecordingAnswerer()).remember(
        _remember_request(idempotency_key="rerank-invalid-first")
    )
    second = await _kernel(store, RecordingAnswerer()).remember(
        _remember_request(idempotency_key="rerank-invalid-second")
    )
    store.embedding_matches = tuple(
        EmbeddingMatch(
            embedding_id=f"embedding_invalid_{index}",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory_id,
            similarity=1.0 - index / 10,
        )
        for index, memory_id in enumerate((first.memory_id, second.memory_id))
    )

    with pytest.raises(ModelOutputError, match="every candidate ID"):
        await _kernel(
            store,
            RecordingAnswerer(),
            reranker=RecordingReranker((first.memory_id,)),
        ).recall(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="semantic-only query"))
        )


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
    middle = MemoryRecord(
        memory_id=MemoryId("summary_middle"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.SEMANTIC,
        summary="A generated intermediate summary.",
        evidence_ids=(),
        occurred_at=child.occurred_at,
        ended_at=child.ended_at,
        created_at=NOW,
        verification_status=VerificationStatus.UNVERIFIED,
        model_reference=ModelReference(model_id="omni", revision="summary-revision"),
    )
    store.memories["summary"] = ("a" * 64, parent)
    store.memories["summary_middle"] = ("b" * 64, middle)
    store.summary_children[parent.memory_id] = (middle.memory_id,)
    store.summary_children[middle.memory_id] = (MemoryId(child.memory_id),)
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
        middle.memory_id,
        child.memory_id,
    ]
    assert [memory.memory_id for memory in answerer.last_memories] == [child.memory_id]
    assert result.answer == child.summary

    expanded_from_child = await store.search_memories_by_hierarchy(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="source detail")),
        (MemoryId(child.memory_id),),
        limit=3,
    )
    assert {memory.memory_id for memory in expanded_from_child} == {
        child.memory_id,
        middle.memory_id,
    }


async def test_semantic_summary_expansion_does_not_displace_direct_hits() -> None:
    store = InMemoryStore()
    kernel = _kernel(store, RecordingAnswerer())
    direct = await kernel.remember(_remember_request(idempotency_key="direct"))
    children = (
        await kernel.remember(_remember_request(idempotency_key="child-0")),
        await kernel.remember(_remember_request(idempotency_key="child-1")),
    )
    parent = MemoryRecord(
        memory_id=MemoryId("summary_parent"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.SEMANTIC,
        summary="A generated session summary.",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    store.memories["summary"] = ("a" * 64, parent)
    store.summary_children[parent.memory_id] = tuple(
        MemoryId(child.memory_id) for child in children
    )
    store.embedding_matches = tuple(
        EmbeddingMatch(
            embedding_id=f"embedding_{index}",
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory_id,
            similarity=similarity,
        )
        for index, (memory_id, similarity) in enumerate(
            ((parent.memory_id, 0.9), (MemoryId(direct.memory_id), 0.8))
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="a semantic paraphrase"),
            limit=3,
        )
    )

    assert [memory.memory_id for memory in result.memories[:2]] == [
        parent.memory_id,
        direct.memory_id,
    ]


async def test_recall_scopes_followup_to_explicit_memory_ids() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), embedder=embedder)
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
    assert embedder.queries == []


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
    embedder = RecordingEmbedder()
    answerer = RecordingAnswerer()
    signer = SequencedMediaUrlSigner()
    kernel = _kernel(
        store,
        answerer,
        embedder=embedder,
        media_access=signer,
    )
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    signer.calls = 0
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

    resolved = embedder.queries[0].media[0]
    assert resolved.kind is MediaKind.VIDEO
    assert resolved.media_url.endswith("?signature=1")
    assert answerer.last_evidence[0].media_url.endswith("?signature=2")
    assert answerer.last_query_media[0].media_url.endswith("?signature=3")
    assert result.evidence[0].media_url.endswith("?signature=4")
    assert signer.calls == 4
    assert result.memories[0].evidence_ids == (evidence_id,)


async def test_media_query_is_resigned_before_reflection_retrieval() -> None:
    store = InMemoryStore()
    embedder = RecordingEmbedder()
    signer = SequencedMediaUrlSigner()
    answerer = RecordingAnswerer(
        answers=(
            GeneratedAnswer(
                answer="Beside the toolbox.",
                confidence=0.7,
                retrieval_queries=("exact tool location",),
            ),
        )
    )
    kernel = _kernel(
        store,
        answerer,
        embedder=embedder,
        media_access=signer,
    )
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    signer.calls = 0
    store.embedding_matches = (
        EmbeddingMatch(
            embedding_id="embedding_01",
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=evidence_id,
            similarity=0.9,
        ),
    )

    await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(media_object_ids=("media_01",)),
        )
    )

    assert len(embedder.queries) == 2
    assert embedder.queries[0].media[0].media_url.endswith("?signature=1")
    assert embedder.queries[1].media[0].media_url.endswith("?signature=4")


async def test_media_query_rejects_unknown_tenant_object() -> None:
    """A query cannot turn another tenant's object ID into a signed URL."""
    embedder = RecordingEmbedder()
    kernel = _kernel(InMemoryStore(), RecordingAnswerer(), embedder=embedder)

    with pytest.raises(DomainInvariantError, match="unknown media"):
        await kernel.recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(media_object_ids=("media_missing",)),
            )
        )

    assert embedder.queries == []


async def test_enumerate_verifies_all_filtered_memories_in_bounded_batches() -> None:
    store = InMemoryStore()
    signer = SequencedMediaUrlSigner()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer, media_access=signer)
    await kernel.observe(_observe_request())
    await _write_attested_memories(store, 125)

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(
                text="words absent from every summary",
                media_object_ids=("media_01",),
            ),
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
    assert signer.calls == 3
    assert all(
        item[0].media_url.endswith("?signature=2") for item in answerer.occurrence_query_media[:4]
    )
    assert all(
        item[0].media_url.endswith("?signature=3") for item in answerer.occurrence_query_media[4:]
    )
    assert answerer.calls == 0
    assert all(memory.useful_access_count == 1 for memory in result.memories)


async def test_enumerate_resigns_returned_evidence_after_verification() -> None:
    store = InMemoryStore()
    signer = SequencedMediaUrlSigner()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer, media_access=signer)
    await kernel.observe(_observe_request())
    evidence_id = next(iter(store.evidence))
    await kernel.remember(_remember_request(evidence_ids=(evidence_id,)))
    signer.calls = 0

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
            mode=RecallMode.ENUMERATE,
        )
    )

    assert answerer.occurrence_evidence[0][0].media_url.endswith("?signature=1")
    assert result.evidence[0].media_url.endswith("?signature=2")


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
    with pytest.raises(DomainInvariantError, match="temporal"):
        GeneratedAnswer(answer="answer", confidence=0.9, temporal_order="sideways")  # type: ignore[arg-type]


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
        occurred_at = NOW - timedelta(seconds=count - ordinal)
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
    occurred_at: datetime = NOW,
) -> RememberRequest:
    return RememberRequest(
        tenant_id="tenant_01",
        summary="The robot put the red screwdriver beside the blue toolbox.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=occurred_at,
        evidence_ids=evidence_ids,
        idempotency_key=idempotency_key,
    )


def _kernel(
    store: InMemoryStore,
    answerer: RecordingAnswerer,
    *,
    job_publisher: RecordingObservationJobPublisher | None = None,
    embedder: RecordingEmbedder | None = None,
    media_access: DeterministicMediaUrlSigner | None = None,
    reranker: Reranker | None = None,
    clock: Callable[[], datetime] = lambda: NOW,
) -> MemoryKernel:
    resolved_media_access = media_access or DeterministicMediaUrlSigner()
    return MemoryKernel(
        store,
        answerer,
        answerer,
        embedding_index=store,
        media_deleter=resolved_media_access,
        media_url_signer=resolved_media_access,
        observation_job_publisher=job_publisher or RecordingObservationJobPublisher(),
        embedder=embedder or RecordingEmbedder(),
        reranker=reranker,
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
            or memory.occurred_at < request.filters.occurred_before
        )
    )
