"""Integration checks for durable, non-resurrecting explicit forgetting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypeAlias, cast

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
    ForgetRequest,
    MediaObjectInput,
    ObserveRequest,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    DeletionPropagationState,
    ForgetTargetNotFoundError,
    ForgetTargetType,
    JobId,
    MediaKind,
    MediaObject,
    MemoryDeletedError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryType,
    ModelReference,
    ObjectStorageError,
    ObservationId,
    SensorKind,
    TenantId,
    derive_stable_id,
)
from mindbridge.infrastructure import PostgresMemoryStore

NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
Counts: TypeAlias = tuple[int, int, int, int, int, int, int]

pytestmark = pytest.mark.integration


class FirstMemoryAnswerer:
    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)


class FixedRecallEmbedder:
    model_reference = ModelReference(model_id="jina-omni", revision="pinned-revision")
    dimension = 1_024

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]:
        return (1.0,) + (0.0,) * 1_023

    async def encode_memory_document(self, text: str) -> tuple[float, ...]:
        return (1.0,) + (0.0,) * 1_023


class DiscardingJobPublisher:
    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None:
        return None


class RecordingMediaAccess:
    def __init__(self, *, deletion_failures: int = 0) -> None:
        self.deletion_failures = deletion_failures
        self.deleted_media_ids: list[str] = []

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


async def test_memory_forget_is_idempotent_and_blocks_resurrection(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id = "tenant_forget_memory"
    kernel = _kernel(store, RecordingMediaAccess())
    remember_request = RememberRequest(
        tenant_id=tenant_id,
        summary="The red tool is beside the blue toolbox.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=NOW,
        idempotency_key="remember_01",
    )
    memory = await kernel.remember(remember_request)
    request = ForgetRequest(
        tenant_id=tenant_id,
        target_type=ForgetTargetType.MEMORY_RECORD,
        target_id=memory.memory_id,
        idempotency_key="forget_01",
    )

    first = await kernel.forget(request)
    retry = await kernel.forget(request)
    status = await kernel.get_forget_status(tenant_id, first.tombstone_id)

    assert first.tombstone_id == retry.tombstone_id
    assert retry.propagation_state is DeletionPropagationState.COMPLETE
    assert status.propagation_state is DeletionPropagationState.COMPLETE
    assert await _counts(database_url, tenant_id) == (0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(MemoryDeletedError):
        await kernel.get_memory(tenant_id, memory.memory_id)
    with pytest.raises(MemoryDeletedError):
        await kernel.remember(remember_request)
    with pytest.raises(ForgetTargetNotFoundError):
        await kernel.get_forget_status("other_tenant", first.tombstone_id)


async def test_observation_forget_recovers_after_media_failure_and_erases_derivatives(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id = "tenant_forget_observation"
    media_access = RecordingMediaAccess(deletion_failures=1)
    kernel = _kernel(store, media_access)
    observe_request = _observe_request(tenant_id)
    observation = await kernel.observe(observe_request)
    evidence_id = (
        (
            await store.read_observation_batch(
                TenantId(tenant_id),
                ObservationId(observation.observation_id),
            )
        )
        .evidence_spans[0]
        .evidence_id
    )
    memory = await kernel.remember(
        RememberRequest(
            tenant_id=tenant_id,
            summary="The robot saw a red tool.",
            memory_type=MemoryType.EPISODIC,
            occurred_at=NOW,
            evidence_ids=(evidence_id,),
        )
    )
    request = ForgetRequest(
        tenant_id=tenant_id,
        target_type=ForgetTargetType.OBSERVATION,
        target_id=observation.observation_id,
    )
    tombstone_id = derive_stable_id(
        "tombstone",
        tenant_id,
        ForgetTargetType.OBSERVATION.value,
        observation.observation_id,
    )

    with pytest.raises(ObjectStorageError):
        await kernel.forget(request)

    failed = await kernel.get_forget_status(tenant_id, tombstone_id)
    hidden = await kernel.recall(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="red tool"))
    )
    assert failed.propagation_state is DeletionPropagationState.FAILED
    assert failed.error_code == "object_storage_unavailable"
    assert hidden.memories == ()
    assert await _counts(database_url, tenant_id) == (1, 1, 1, 1, 1, 1, 1)

    completed = await kernel.forget(request)

    assert completed.propagation_state is DeletionPropagationState.COMPLETE
    assert media_access.deleted_media_ids == ["media_01"]
    assert await _counts(database_url, tenant_id) == (0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(MemoryDeletedError):
        await kernel.observe(observe_request)
    with pytest.raises(MemoryNotFoundError):
        await kernel.get_memory(tenant_id, memory.memory_id)


def _kernel(store: PostgresMemoryStore, media_access: RecordingMediaAccess) -> MemoryKernel:
    return MemoryKernel(
        store,
        FirstMemoryAnswerer(),
        embedding_index=store,
        media_url_signer=media_access,
        observation_job_publisher=DiscardingJobPublisher(),
        recall_embedder=FixedRecallEmbedder(),
        clock=lambda: NOW,
    )


def _observe_request(tenant_id: str) -> ObserveRequest:
    return ObserveRequest(
        tenant_id=tenant_id,
        device_id="device_01",
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id="media_01",
                kind=MediaKind.VIDEO,
                uri=f"s3://memory/{tenant_id}/media_01.mp4",
                sha256="a" * 64,
                size_bytes=100,
                created_at=NOW,
                duration_ms=4_000,
            ),
        ),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=4),
        observed_at=NOW,
    )


async def _counts(database_url: str, tenant_id: str) -> Counts:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM observations WHERE tenant_id = %s),
                    (SELECT count(*) FROM evidence_spans WHERE tenant_id = %s),
                    (SELECT count(*) FROM media_objects WHERE tenant_id = %s),
                    (SELECT count(*) FROM jobs WHERE tenant_id = %s),
                    (SELECT count(*) FROM memory_records WHERE tenant_id = %s),
                    (SELECT count(*) FROM embeddings WHERE tenant_id = %s),
                    (SELECT count(*) FROM deletion_tombstones WHERE tenant_id = %s)
                """,
                (tenant_id,) * 7,
            )
        ).fetchone()
    return cast(Counts, row)
