"""Integration checks for migrations and the real PostgreSQL adapter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from psycopg import AsyncConnection

from mindbridge.application import (
    GeneratedAnswer,
    MemoryKernel,
    PresignedMediaDownload,
    RecallEmbeddingQuery,
    ResolvedEvidence,
    ResolvedQueryMedia,
)
from mindbridge.contracts import (
    FeedbackRequest,
    IdentityObservationInput,
    MediaObjectInput,
    ObservationStatus,
    ObserveRequest,
    RecallFilters,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    EmbeddingSpaceReference,
    FeedbackType,
    IdempotencyConflictError,
    IdentityKind,
    JobId,
    JobNotFoundError,
    JobState,
    MediaKind,
    MediaObject,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelReference,
    ObservationId,
    SensorKind,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure import PostgresMemoryStore

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

pytestmark = pytest.mark.integration


class FirstMemoryAnswerer:
    """Deterministic answerer for persistence-path verification."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        return tuple(memory.memory_id for memory in memories)


class DeterministicMediaUrlSigner:
    """Keeps database integration independent from object storage."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )

    async def delete_media(self, media_object: MediaObject) -> None:
        return None


class DiscardingObservationJobPublisher:
    """Keeps store integration independent from the Redis delivery adapter."""

    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None:
        return None


class FixedRecallEmbedder:
    """Keeps persistence integration independent from the embedding service."""

    query_model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
    document_model_reference = ModelReference(model_id="jina-text", revision="pinned-revision")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
    dimension = 1_024

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        return (1.0,) + (0.0,) * 1_023

    async def encode_memory_document(self, text: str) -> tuple[float, ...]:
        return (1.0,) + (0.0,) * 1_023


async def test_migration_installs_complete_phase_zero_schema(database_url: str) -> None:
    """The initial migration creates pgvector and every Phase 0 system table."""
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        extension = await (
            await connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        tables = await (
            await connection.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                """
            )
        ).fetchall()
        row_secured_tables = await (
            await connection.execute(
                """
                SELECT column_info.table_name
                FROM information_schema.columns AS column_info
                JOIN pg_class AS relation ON relation.relname = column_info.table_name
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE column_info.table_schema = 'public'
                  AND column_info.column_name = 'tenant_id'
                  AND namespace.nspname = 'public'
                  AND relation.relrowsecurity
                  AND relation.relforcerowsecurity
                """
            )
        ).fetchall()
        policy_tables = await (
            await connection.execute(
                """
                SELECT tablename FROM pg_policies
                WHERE schemaname = 'public' AND policyname = 'tenant_isolation'
                """
            )
        ).fetchall()

    assert cast(tuple[str], extension)[0] == "0.8.2"
    assert {
        "claims",
        "embeddings",
        "events",
        "evidence_spans",
        "media_objects",
        "memory_records",
        "observations",
    } <= {cast(tuple[str], row)[0] for row in tables}
    tenant_tables = {
        "claim_evidence",
        "claims",
        "deletion_tombstones",
        "embeddings",
        "entities",
        "entity_mentions",
        "event_evidence",
        "event_observations",
        "events",
        "evidence_spans",
        "idempotency_keys",
        "jobs",
        "media_objects",
        "memory_evidence",
        "memory_feedback",
        "memory_records",
        "observation_media",
        "observations",
        "relations",
    }
    assert {cast(tuple[str], row)[0] for row in row_secured_tables} == tenant_tables
    assert {cast(tuple[str], row)[0] for row in policy_tables} == tenant_tables
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        versions = await (
            await connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ).fetchall()
    assert [cast(tuple[int], row)[0] for row in versions] == [1, 2, 3, 4, 5, 6, 7, 8]


async def test_postgres_vertical_path_is_idempotent_and_evidence_first(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The production store runs observe, remember, and recall without a side path."""
    kernel = _kernel(store)
    request = _observe_request(tenant_id="tenant_roundtrip")

    first = await kernel.observe(request)
    retry = await kernel.observe(request)
    job = await store.read_observation_processing_job(
        TenantId("tenant_roundtrip"),
        JobId(first.processing_job_id),
    )
    stored_batch = await store.read_observation_batch(
        TenantId("tenant_roundtrip"),
        ObservationId(first.observation_id),
    )
    evidence_id = await _first_evidence_id(database_url, "tenant_roundtrip")
    memory = await kernel.remember(
        _remember_request(tenant_id="tenant_roundtrip", evidence_id=evidence_id)
    )
    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_roundtrip",
            query=RecallQuery(text="red screwdriver"),
            filters=RecallFilters(device_ids=("device_01",)),
        )
    )

    assert first.status is ObservationStatus.ACCEPTED
    assert retry.status is ObservationStatus.DUPLICATE
    assert first.processing_job_id == retry.processing_job_id
    assert job.state is JobState.PENDING
    assert job.observation_id == first.observation_id
    assert stored_batch.observation.observation_id == first.observation_id
    assert stored_batch.media_objects[0].media_object_id == "media_01"
    assert stored_batch.evidence_spans[0].end_ms == 4_000
    assert stored_batch.observation.identity_observations[0].identity_id == "person_device_01"
    assert stored_batch.observation.identity_observations[0].model_reference.revision == "1.0.1"
    assert await _processing_job_count(database_url, "tenant_roundtrip") == 1
    assert memory.verification_status is VerificationStatus.VERIFIED
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.evidence[0].evidence_id == evidence_id
    assert result.evidence[0].end_ms == 4_000
    assert result.evidence[0].media_url.startswith("https://objects.example.test/media_")

    with pytest.raises(JobNotFoundError):
        await store.read_observation_processing_job(
            TenantId("other_tenant"),
            JobId(first.processing_job_id),
        )


async def test_postgres_enumeration_scans_filters_without_full_text_truncation(
    store: PostgresMemoryStore,
) -> None:
    kernel = _kernel(store)
    tenant_id = "tenant_enumeration"
    for ordinal in reversed(range(3)):
        await kernel.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=f"Attested occurrence {ordinal}",
                memory_type=MemoryType.EPISODIC,
                occurred_at=NOW + timedelta(minutes=ordinal),
                idempotency_key=f"enumeration_{ordinal}",
            )
        )

    result = await kernel.recall(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="text absent from every stored summary"),
            filters=RecallFilters(occurred_after=NOW + timedelta(minutes=1)),
            mode=RecallMode.ENUMERATE,
            include_evidence=False,
        )
    )

    assert result.answer == "2"
    assert [memory.summary for memory in result.memories] == [
        "Attested occurrence 1",
        "Attested occurrence 2",
    ]


async def test_postgres_round_trips_attested_source_memory(store: PostgresMemoryStore) -> None:
    """Explicit source text survives persistence and can ground a reported answer."""
    kernel = _kernel(store)
    memory = await kernel.remember(
        RememberRequest(
            tenant_id="tenant_attested",
            summary="Caroline said she plans to become a counselor.",
            memory_type=MemoryType.SEMANTIC,
            occurred_at=NOW,
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="plans to become a counselor"),
        )
    )

    assert memory.verification_status is VerificationStatus.ATTESTED
    assert result.answer == "Caroline said she plans to become a counselor."
    assert result.evidence == ()

    ranked_memory_ids = (MemoryId(memory.memory_id),)
    matched = await store.search_memories_by_ids(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="words do not constrain dense candidates"),
            filters=RecallFilters(memory_types=(MemoryType.SEMANTIC,)),
        ),
        ranked_memory_ids,
        limit=10,
    )
    filtered = await store.search_memories_by_ids(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="words do not constrain dense candidates"),
            filters=RecallFilters(memory_types=(MemoryType.EPISODIC,)),
        ),
        ranked_memory_ids,
        limit=10,
    )
    assert [item.memory_id for item in matched] == [memory.memory_id]
    assert filtered == ()

    found = await kernel.get_memory("tenant_attested", memory.memory_id)
    assert found.model_copy(update={"useful_access_count": 0, "last_accessed_at": None}) == memory
    assert found.useful_access_count == 1
    assert found.last_accessed_at == NOW
    with pytest.raises(MemoryNotFoundError):
        await kernel.get_memory("other_tenant", memory.memory_id)


async def test_postgres_rejects_idempotency_key_reuse(store: PostgresMemoryStore) -> None:
    """Conflicting retries roll back rather than aliasing two observations."""
    kernel = _kernel(store)
    await kernel.observe(
        _observe_request(tenant_id="tenant_conflict", idempotency_key="request_01")
    )

    with pytest.raises(IdempotencyConflictError):
        await kernel.observe(
            _observe_request(
                tenant_id="tenant_conflict",
                sequence=2,
                idempotency_key="request_01",
            )
        )


async def test_postgres_feedback_evolves_and_versions_memory_atomically(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    kernel = _kernel(store)
    original = await kernel.remember(
        RememberRequest(
            tenant_id="tenant_feedback",
            summary="The screwdriver is on the blue toolbox.",
            memory_type=MemoryType.EPISODIC,
            occurred_at=NOW,
            idempotency_key="remember_original",
        )
    )
    useful_request = FeedbackRequest(
        tenant_id="tenant_feedback",
        feedback_type=FeedbackType.USEFUL,
        memory_id=original.memory_id,
        idempotency_key="feedback_useful",
    )
    useful = await kernel.record_feedback(useful_request)
    useful_retry = await kernel.record_feedback(useful_request)
    correction_request = FeedbackRequest(
        tenant_id="tenant_feedback",
        feedback_type=FeedbackType.CORRECTION,
        memory_id=original.memory_id,
        correction_summary="The screwdriver is inside the green drawer.",
        idempotency_key="feedback_correction",
    )
    correction = await kernel.record_feedback(correction_request)
    correction_retry = await kernel.record_feedback(correction_request)

    old = await kernel.get_memory("tenant_feedback", original.memory_id)
    corrected = await kernel.get_memory(
        "tenant_feedback", cast(str, correction.corrected_memory_id)
    )
    recalled = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_feedback",
            query=RecallQuery(text="green drawer"),
        )
    )

    assert useful.feedback_id == useful_retry.feedback_id
    assert useful.resulting_state is MemoryState.STRENGTHENED
    assert correction.corrected_memory_id == correction_retry.corrected_memory_id
    assert old.superseded_at == NOW
    assert old.positive_feedback_count == 1
    assert old.negative_feedback_count == 1
    assert corrected.supersedes_memory_id == original.memory_id
    assert corrected.verification_status is VerificationStatus.ATTESTED
    assert [memory.memory_id for memory in recalled.memories] == [corrected.memory_id]

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT count(*) FROM memory_feedback
                WHERE tenant_id = 'tenant_feedback'
                """
            )
        ).fetchone()
    assert cast(tuple[int], row)[0] == 2

    with pytest.raises(IdempotencyConflictError):
        await kernel.record_feedback(
            FeedbackRequest(
                tenant_id="tenant_feedback",
                feedback_type=FeedbackType.WRONG,
                memory_id=corrected.memory_id,
                idempotency_key="feedback_useful",
            )
        )


async def test_postgres_deduplicates_media_by_content_hash(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """Different device aliases for identical bytes share one media row."""
    kernel = _kernel(store)
    await kernel.observe(
        _observe_request(
            tenant_id="tenant_dedup",
            sequence=1,
            media_object_id="media_alias_01",
        )
    )
    await kernel.observe(
        _observe_request(
            tenant_id="tenant_dedup",
            sequence=2,
            media_object_id="media_alias_02",
        )
    )

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                "SELECT count(*) FROM media_objects WHERE tenant_id = 'tenant_dedup'"
            )
        ).fetchone()

    assert cast(tuple[int], row)[0] == 1


async def test_observation_job_state_is_atomic_and_retryable(
    store: PostgresMemoryStore,
) -> None:
    """Concurrent deliveries own one attempt and a failed job can later succeed."""
    receipt = await _kernel(store).observe(_observe_request(tenant_id="tenant_job_state"))
    tenant_id = TenantId("tenant_job_state")
    observation_id = ObservationId(receipt.observation_id)
    job_id = JobId(receipt.processing_job_id)

    claims = await asyncio.gather(
        *(
            store.claim_observation_processing_job(tenant_id, observation_id, job_id)
            for _ in range(5)
        )
    )

    assert sum(claim.acquired for claim in claims) == 1
    assert {claim.job.state for claim in claims} == {JobState.RUNNING}
    assert {claim.job.attempt for claim in claims} == {1}
    failed = await store.mark_observation_processing_failed(
        tenant_id,
        observation_id,
        job_id,
        attempt=1,
        error_code="model_unavailable",
    )
    retry = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
    with pytest.raises(MemoryIntegrityError, match="not running"):
        await store.mark_observation_processing_failed(
            tenant_id,
            observation_id,
            job_id,
            attempt=1,
            error_code="stale_attempt",
        )
    succeeded = await store.mark_observation_processing_succeeded(
        tenant_id, observation_id, job_id, attempt=2
    )
    duplicate = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)

    assert failed.state is JobState.FAILED
    assert failed.error_code == "model_unavailable"
    assert retry.acquired is True
    assert retry.job.attempt == 2
    assert retry.job.error_code is None
    assert succeeded.state is JobState.SUCCEEDED
    assert duplicate.acquired is False
    assert duplicate.job.state is JobState.SUCCEEDED


async def _first_evidence_id(database_url: str, tenant_id: str) -> str:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT evidence_id FROM evidence_spans
                WHERE tenant_id = %s ORDER BY evidence_id LIMIT 1
                """,
                (tenant_id,),
            )
        ).fetchone()
    return cast(tuple[str], row)[0]


async def _processing_job_count(database_url: str, tenant_id: str) -> int:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT count(*) FROM jobs
                WHERE tenant_id = %s AND job_type = 'process_observation'
                """,
                (tenant_id,),
            )
        ).fetchone()
    return cast(tuple[int], row)[0]


def _observe_request(
    *,
    tenant_id: str,
    sequence: int = 1,
    media_object_id: str = "media_01",
    idempotency_key: str | None = None,
) -> ObserveRequest:
    return ObserveRequest(
        tenant_id=tenant_id,
        device_id="device_01",
        boot_id="boot_01",
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id=media_object_id,
                kind=MediaKind.VIDEO,
                uri=f"s3://memories/{media_object_id}.mp4",
                sha256="a" * 64,
                size_bytes=100,
                duration_ms=4_000,
                created_at=NOW,
            ),
        ),
        occurred_at=NOW + timedelta(seconds=sequence),
        ended_at=NOW + timedelta(seconds=sequence + 4),
        observed_at=NOW,
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
        ),
        idempotency_key=idempotency_key,
    )


def _remember_request(*, tenant_id: str, evidence_id: str) -> RememberRequest:
    return RememberRequest(
        tenant_id=tenant_id,
        summary="The robot put the red screwdriver beside the blue toolbox.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=NOW,
        evidence_ids=(evidence_id,),
    )


def _kernel(store: PostgresMemoryStore) -> MemoryKernel:
    media_access = DeterministicMediaUrlSigner()
    return MemoryKernel(
        store,
        FirstMemoryAnswerer(),
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=DiscardingObservationJobPublisher(),
        recall_embedder=FixedRecallEmbedder(),
        clock=lambda: NOW,
    )
