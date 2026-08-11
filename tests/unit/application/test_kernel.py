"""Vertical tests for the shared observe, remember, and recall path."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import (
    EmbeddingMatch,
    EmbeddingSearch,
    GeneratedAnswer,
    MemoryKernel,
    MemoryWriteResult,
    ObservationBatch,
    ObservationWriteResult,
    PresignedMediaDownload,
    RecallEmbeddingQuery,
    ResolvedEvidence,
)
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationStatus,
    ObserveRequest,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingRecord,
    EvidenceId,
    EvidenceSpan,
    IdempotencyConflictError,
    JobId,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
    VerificationStatus,
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

    async def search_memories(self, request: RecallRequest) -> tuple[MemoryRecord, ...]:
        query = request.query.text.casefold() if request.query.text is not None else None
        candidates = (
            memory
            for _, memory in self.memories.values()
            if memory.tenant_id == request.tenant_id
            and (query is None or query in memory.summary.casefold())
            and (
                not request.filters.memory_types
                or memory.memory_type in request.filters.memory_types
            )
            and (
                request.filters.occurred_after is None
                or memory.occurred_at >= request.filters.occurred_after
            )
            and (
                request.filters.occurred_before is None
                or memory.occurred_at <= request.filters.occurred_before
            )
        )
        return tuple(sorted(candidates, key=lambda memory: memory.occurred_at, reverse=True))[
            : request.limit
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
            and any(evidence_id in rank for evidence_id in memory.evidence_ids)
        )
        return tuple(
            sorted(
                candidates,
                key=lambda memory: min(rank[item] for item in memory.evidence_ids if item in rank),
            )
        )[:limit]

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

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool:
        return True

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        self.embedding_searches.append(search)
        return self.embedding_matches[: search.limit]


class DeterministicMediaUrlSigner:
    """Returns openable-looking URLs without an object-storage dependency."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )


class RecordingAnswerer:
    """Deterministic answerer proving when the model boundary is invoked."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_evidence: tuple[ResolvedEvidence, ...] = ()

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> GeneratedAnswer:
        self.calls += 1
        self.last_evidence = evidence
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)


class RecordingQueryEmbedder:
    """Returns one valid vector while retaining the fused multimodal query."""

    model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
    dimension = 1_024

    def __init__(self) -> None:
        self.queries: list[RecallEmbeddingQuery] = []

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        self.queries.append(query)
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

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_01",
            query=RecallQuery(text="red screwdriver"),
        )
    )

    assert memory.verification_status is VerificationStatus.VERIFIED
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


async def test_recall_abstains_when_candidate_has_no_evidence() -> None:
    """An unverified summary may be returned by search but cannot ground an answer."""
    store = InMemoryStore()
    answerer = RecordingAnswerer()
    kernel = _kernel(store, answerer)
    await kernel.remember(_remember_request())

    result = await kernel.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="red screwdriver"))
    )

    assert len(result.memories) == 1
    assert result.answer is None
    assert result.confidence == 0.0
    assert answerer.calls == 0


async def test_semantic_evidence_finds_memory_without_matching_summary_text() -> None:
    """Jina evidence rank reaches grounded memories that sparse text cannot see."""
    store = InMemoryStore()
    query_embedder = RecordingQueryEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), query_embedder=query_embedder)
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
    assert query_embedder.queries[0].text == "a hand moves an object"
    assert store.embedding_searches[0].document_task == "retrieval_document"


async def test_media_query_is_signed_for_embedding_instead_of_used_as_a_filter() -> None:
    """Original AV reaches Jina through a short-lived tenant-owned URL."""
    store = InMemoryStore()
    query_embedder = RecordingQueryEmbedder()
    kernel = _kernel(store, RecordingAnswerer(), query_embedder=query_embedder)
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

    resolved = query_embedder.queries[0].media[0]
    assert resolved.media_object.kind is MediaKind.VIDEO
    assert resolved.media_url == "https://objects.example.test/media_01"
    assert result.memories[0].evidence_ids == (evidence_id,)


async def test_media_query_rejects_unknown_tenant_object() -> None:
    """A query cannot turn another tenant's object ID into a signed URL."""
    query_embedder = RecordingQueryEmbedder()
    kernel = _kernel(InMemoryStore(), RecordingAnswerer(), query_embedder=query_embedder)

    with pytest.raises(DomainInvariantError, match="unknown media"):
        await kernel.recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(media_object_ids=("media_missing",)),
            )
        )

    assert query_embedder.queries == []


def test_generated_answer_rejects_confidence_without_answer() -> None:
    """Abstention cannot report misleading answer confidence."""
    with pytest.raises(DomainInvariantError, match="zero"):
        GeneratedAnswer(answer=None, confidence=0.9)


def _observe_request(
    *,
    sequence: int = 1,
    idempotency_key: str | None = None,
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
        idempotency_key=idempotency_key,
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
    query_embedder: RecordingQueryEmbedder | None = None,
    clock: Callable[[], datetime] = lambda: NOW,
) -> MemoryKernel:
    return MemoryKernel(
        store,
        answerer,
        embedding_index=store,
        media_url_signer=DeterministicMediaUrlSigner(),
        observation_job_publisher=job_publisher or RecordingObservationJobPublisher(),
        query_embedder=query_embedder or RecordingQueryEmbedder(),
        clock=clock,
    )


def _require_same_content(stored_digest: str, requested_digest: str) -> None:
    if stored_digest != requested_digest:
        raise IdempotencyConflictError("idempotency key already stores different content")
