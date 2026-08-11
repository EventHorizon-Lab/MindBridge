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

from mindbridge.application import GeneratedAnswer, MemoryKernel
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
    EvidenceSpan,
    IdempotencyConflictError,
    MediaKind,
    MemoryRecord,
    MemoryType,
    SensorKind,
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
        evidence: tuple[EvidenceSpan, ...],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)


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
    kernel = MemoryKernel(store, FirstMemoryAnswerer(), clock=lambda: NOW)
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
    assert memory.verification_status is VerificationStatus.VERIFIED
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.evidence[0].evidence_id == evidence_id
    assert result.evidence[0].end_ms == 4_000


async def test_postgres_rejects_idempotency_key_reuse(store: PostgresMemoryStore) -> None:
    """Conflicting retries roll back rather than aliasing two observations."""
    kernel = MemoryKernel(store, FirstMemoryAnswerer(), clock=lambda: NOW)
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
    kernel = MemoryKernel(store, FirstMemoryAnswerer(), clock=lambda: NOW)
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
