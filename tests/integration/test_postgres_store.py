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
)
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationStatus,
    ObserveRequest,
    RecallFilters,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    IdempotencyConflictError,
    JobId,
    JobNotFoundError,
    JobState,
    MediaKind,
    MediaObject,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
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
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)


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

    model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
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
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        versions = await (
            await connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ).fetchall()
    assert [cast(tuple[int], row)[0] for row in versions] == [1, 2, 3]


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
    assert found == memory
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
    return MemoryKernel(
        store,
        FirstMemoryAnswerer(),
        embedding_index=store,
        media_url_signer=DeterministicMediaUrlSigner(),
        observation_job_publisher=DiscardingObservationJobPublisher(),
        recall_embedder=FixedRecallEmbedder(),
        clock=lambda: NOW,
    )
