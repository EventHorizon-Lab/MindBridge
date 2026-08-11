"""Integration checks for atomic observation-derived memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypeAlias, cast

import pytest
from psycopg import AsyncConnection

from mindbridge.application import (
    EmbeddingInput,
    EventPerception,
    ObservationBatch,
    ObservationProcessingOutput,
    PerceivedEvent,
    PresignedMediaDownload,
    ProcessObservation,
    ResolvedEvidence,
)
from mindbridge.core import (
    DeviceId,
    DomainInvariantError,
    EvidenceId,
    EvidenceSpan,
    JobId,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    ModelReference,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.infrastructure import PostgresMemoryStore

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
MODEL = ModelReference(model_id="qwen3.8-max", revision="serving-revision-01")
EMBEDDING_MODEL = ModelReference(
    model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
    revision="12949877f0092093f366c6450340011320152a05",
)
DerivedCounts: TypeAlias = tuple[int, int, int, int, int, int, int]

pytestmark = pytest.mark.integration


class RecordingPerceiver:
    """Return one event grounded in the evidence supplied by PostgreSQL."""

    def __init__(self) -> None:
        self.calls = 0

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        self.calls += 1
        return EventPerception(
            events=(
                PerceivedEvent(
                    start_ms=500,
                    end_ms=3_500,
                    description="A person places a red tool beside a blue toolbox.",
                    salience=0.8,
                    evidence_ids=(evidence[0].evidence_span.evidence_id,),
                ),
            ),
            model_reference=MODEL,
            prompt_version="perceive_events_v1",
        )


class FixedEmbedder:
    """Emit a deterministic unit vector in a configured model dimension."""

    model_reference = EMBEDDING_MODEL

    def __init__(self, dimension: int = 1_024) -> None:
        self.dimension = dimension
        self.documents: tuple[EmbeddingInput, ...] = ()

    async def encode_queries(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return self._vectors(len(inputs))

    async def encode_documents(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.documents = inputs
        return self._vectors(len(inputs))

    def _vectors(self, count: int) -> tuple[tuple[float, ...], ...]:
        vector = (1.0,) + (0.0,) * (self.dimension - 1)
        return (vector,) * count


class DeterministicSigner:
    """Keep processing integration independent from an S3 service."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}.mp4",
            expires_at=NOW + timedelta(minutes=5),
        )


async def test_processing_commits_provenance_once(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A completed attempt atomically stores every evidence-derived record."""
    tenant_id, observation_id, job_id = await _write_source_observation(
        store, "tenant_processing_success"
    )
    perceiver = RecordingPerceiver()
    embedder = FixedEmbedder()
    processor = ProcessObservation(
        store,
        perceiver,
        embedder,
        media_url_signer=DeterministicSigner(),
    )

    first = await processor.run(tenant_id, observation_id, job_id)
    duplicate = await processor.run(tenant_id, observation_id, job_id)

    assert first.state is JobState.SUCCEEDED
    assert duplicate.state is JobState.SUCCEEDED
    assert perceiver.calls == 1
    assert embedder.documents == ("https://objects.example.test/media_01.mp4",)
    assert await _derived_counts(database_url, tenant_id) == (1,) * 7
    assert await _job_state(database_url, tenant_id, job_id) == ("succeeded", 1, None)
    assert await _event_provenance(database_url, tenant_id) == (
        "A person places a red tool beside a blue toolbox.",
        "qwen3.8-max",
        "serving-revision-01",
        "perceive_events_v1",
    )


async def test_processing_rolls_back_derived_records_before_retry(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A vector failure leaves no partial memory and a clean retry can succeed."""
    tenant_id, observation_id, job_id = await _write_source_observation(
        store, "tenant_processing_retry"
    )
    failing = ProcessObservation(
        store,
        RecordingPerceiver(),
        FixedEmbedder(dimension=2),
        media_url_signer=DeterministicSigner(),
    )

    with pytest.raises(DomainInvariantError, match="cloud embedding dimension"):
        await failing.run(tenant_id, observation_id, job_id)

    assert await _derived_counts(database_url, tenant_id) == (0,) * 7
    assert await _job_state(database_url, tenant_id, job_id) == (
        "failed",
        1,
        "domain_invariant_failed",
    )

    succeeded = await ProcessObservation(
        store,
        RecordingPerceiver(),
        FixedEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(tenant_id, observation_id, job_id)

    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.attempt == 2
    assert await _derived_counts(database_url, tenant_id) == (1,) * 7


async def test_superseded_attempt_cannot_commit(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A stale worker cannot commit after a reclaimed attempt has completed."""
    tenant_id, observation_id, job_id = await _write_source_observation(
        store, "tenant_processing_stale"
    )
    first = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
    await _age_running_job(database_url, tenant_id, job_id)
    second = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
    await store.mark_observation_processing_succeeded(
        tenant_id,
        observation_id,
        job_id,
        attempt=second.job.attempt,
    )

    with pytest.raises(MemoryIntegrityError, match="attempt was superseded"):
        await store.commit_observation_processing(
            tenant_id,
            observation_id,
            job_id,
            attempt=first.job.attempt,
            output=ObservationProcessingOutput(events=(), memories=(), embeddings=()),
        )

    assert first.job.attempt == 1
    assert second.job.attempt == 2
    assert await _job_state(database_url, tenant_id, job_id) == ("succeeded", 2, None)


async def _write_source_observation(
    store: PostgresMemoryStore,
    tenant: str,
) -> tuple[TenantId, ObservationId, JobId]:
    tenant_id = TenantId(tenant)
    observation_id = ObservationId("observation_01")
    media_object_id = MediaObjectId("media_01")
    observation = Observation(
        observation_id=observation_id,
        tenant_id=tenant_id,
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_object_ids=(media_object_id,),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=4),
        observed_at=NOW,
        clock_offset_ms=0,
    )
    result = await store.write_observation(
        ObservationBatch(
            media_objects=(
                MediaObject(
                    media_object_id=media_object_id,
                    tenant_id=tenant_id,
                    kind=MediaKind.VIDEO,
                    uri=f"s3://memory/{tenant}/clip.mp4",
                    sha256="a" * 64,
                    size_bytes=100,
                    created_at=NOW,
                    duration_ms=4_000,
                ),
            ),
            observation=observation,
            evidence_spans=(
                EvidenceSpan(
                    evidence_id=EvidenceId("evidence_01"),
                    tenant_id=tenant_id,
                    observation_id=observation_id,
                    media_object_id=media_object_id,
                    start_ms=0,
                    end_ms=4_000,
                    created_at=NOW,
                ),
            ),
        ),
        idempotency_key="observe_01",
        content_digest="b" * 64,
    )
    return tenant_id, observation_id, result.processing_job_id


async def _derived_counts(database_url: str, tenant_id: TenantId) -> DerivedCounts:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM events WHERE tenant_id = %s),
                    (SELECT count(*) FROM event_observations WHERE tenant_id = %s),
                    (SELECT count(*) FROM event_evidence WHERE tenant_id = %s),
                    (SELECT count(*) FROM memory_records WHERE tenant_id = %s),
                    (SELECT count(*) FROM memory_evidence WHERE tenant_id = %s),
                    (SELECT count(*) FROM relations WHERE tenant_id = %s),
                    (SELECT count(*) FROM embeddings WHERE tenant_id = %s)
                """,
                (tenant_id,) * 7,
            )
        ).fetchone()
    return cast(DerivedCounts, row)


async def _age_running_job(
    database_url: str,
    tenant_id: TenantId,
    job_id: JobId,
) -> None:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        await connection.execute(
            """
            UPDATE jobs SET updated_at = now() - interval '961 seconds'
            WHERE tenant_id = %s AND job_id = %s
            """,
            (tenant_id, job_id),
        )


async def _job_state(
    database_url: str,
    tenant_id: TenantId,
    job_id: JobId,
) -> tuple[str, int, str | None]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT state, attempt, error_code FROM jobs
                WHERE tenant_id = %s AND job_id = %s
                """,
                (tenant_id, job_id),
            )
        ).fetchone()
    return cast(tuple[str, int, str | None], row)


async def _event_provenance(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[str, str, str, str]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT description, model_id, model_revision, prompt_version
                FROM events WHERE tenant_id = %s
                """,
                (tenant_id,),
            )
        ).fetchone()
    return cast(tuple[str, str, str, str], row)
