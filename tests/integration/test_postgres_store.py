"""Integration checks for migrations and the real PostgreSQL adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
from psycopg import AsyncConnection

from mindbridge.application import (
    EmbeddingSearch,
    GeneratedAnswer,
    MemoryKernel,
    PresignedMediaDownload,
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
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    IdempotencyConflictError,
    MediaKind,
    MediaObject,
    MemoryRecord,
    MemoryType,
    ModelReference,
    SensorKind,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure import PostgresMemoryStore

DATABASE_URL = os.getenv("MINDBRIDGE_TEST_DATABASE_URL")
NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="MINDBRIDGE_TEST_DATABASE_URL is not configured",
    ),
]


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


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def database_url() -> AsyncIterator[str]:
    """Rebuild only an explicitly named disposable test database."""
    assert DATABASE_URL is not None
    connection = await AsyncConnection.connect(DATABASE_URL, autocommit=True)
    async with connection:
        row = await (await connection.execute("SELECT current_database()")).fetchone()
        database_name = cast(tuple[str], row)[0]
        if not database_name.endswith("_test"):
            raise RuntimeError("integration database name must end with _test")
        await connection.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public", prepare=False)
        migration = Path(__file__).parents[2] / "migrations" / "0001_initial.sql"
        await connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
    yield DATABASE_URL


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def store(database_url: str) -> AsyncIterator[PostgresMemoryStore]:
    """Open the production adapter once for the integration suite."""
    postgres_store = PostgresMemoryStore(database_url, max_pool_size=4)
    await postgres_store.open()
    yield postgres_store
    await postgres_store.close()


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


async def test_postgres_vertical_path_is_idempotent_and_evidence_first(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The production store runs observe, remember, and recall without a side path."""
    kernel = _kernel(store)
    request = _observe_request(tenant_id="tenant_roundtrip")

    first = await kernel.observe(request)
    retry = await kernel.observe(request)
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
    assert await _processing_job_count(database_url, "tenant_roundtrip") == 1
    assert memory.verification_status is VerificationStatus.VERIFIED
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.evidence[0].evidence_id == evidence_id
    assert result.evidence[0].end_ms == 4_000
    assert result.evidence[0].media_url.startswith("https://objects.example.test/media_")


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


async def test_pgvector_keeps_model_spaces_separate_and_ranks_by_cosine(
    store: PostgresMemoryStore,
) -> None:
    """The cloud index retrieves only the requested frozen model version."""
    model = ModelReference(
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        revision="abcdef0",
    )
    first = _embedding_record(
        embedding_id="embedding_01",
        object_id="memory_near",
        values=(1.0,) + (0.0,) * 1_023,
        model=model,
    )
    second = _embedding_record(
        embedding_id="embedding_02",
        object_id="memory_far",
        values=(0.0, 1.0) + (0.0,) * 1_022,
        model=model,
    )

    assert await store.write_embedding(first) is True
    assert await store.write_embedding(first) is False
    assert await store.write_embedding(second) is True
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_01",
                object_id="memory_near",
                values=second.values,
                model=model,
            )
        )
    matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=TenantId("tenant_vectors"),
            values=first.values,
            model_reference=model,
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )
    other_revision = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=TenantId("tenant_vectors"),
            values=first.values,
            model_reference=ModelReference(model_id=model.model_id, revision="different"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )

    assert [match.object_id for match in matches] == ["memory_near", "memory_far"]
    assert matches[0].similarity == pytest.approx(1.0)
    assert other_revision == ()


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


def _embedding_record(
    *,
    embedding_id: str,
    object_id: str,
    values: tuple[float, ...],
    model: ModelReference,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(embedding_id),
        tenant_id=TenantId("tenant_vectors"),
        object_type=EmbeddedObjectType.MEMORY_RECORD,
        object_id=object_id,
        values=values,
        model_reference=model,
        task="retrieval_document",
        dimension=1_024,
        normalized=True,
        created_at=NOW,
    )


def _kernel(store: PostgresMemoryStore) -> MemoryKernel:
    return MemoryKernel(
        store,
        FirstMemoryAnswerer(),
        media_url_signer=DeterministicMediaUrlSigner(),
        clock=lambda: NOW,
    )
