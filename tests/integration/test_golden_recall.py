"""Deterministic production-path Golden Recall regression gate."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    GeneratedAnswer,
    PresignedMediaDownload,
    ResolvedQueryMedia,
)
from mindbridge.contracts import (
    MediaObjectInput,
    ObserveRequest,
    RecallFilters,
    RecallQuery,
    RecallRequest,
)
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    EvidenceId,
    JobId,
    MediaKind,
    MediaObject,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    ObservationId,
    SensorKind,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.models import Embedding, EmbedRequest, EmbedResult, EmbedTask, TextPart

TENANT_ID = TenantId("tenant_golden_recall")
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
MODEL_REFERENCE = ModelReference(model_id="golden-jina")
SPACE_REFERENCE = EmbeddingSpaceReference(space_id="golden-space")
VECTOR_DIMENSION = 1_024
GOLDEN_SET_PATH = Path(__file__).parents[1] / "benchmarks" / "golden_recall.json"

pytestmark = pytest.mark.integration


class _GoldenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldenMemory(_GoldenModel):
    memory_id: str
    memory_type: MemoryType
    summary: str
    occurred_at: AwareDatetime
    source: Literal["video", "attested"]


class GoldenCase(_GoldenModel):
    name: str
    text: str
    query_axis: Literal[-1, 0]
    occurred_after: AwareDatetime | None
    expected_memory_ids: tuple[str, ...]
    expected_answer: str | None
    expected_evidence_count: Annotated[int, Field(ge=0)]


class GoldenRecallSet(_GoldenModel):
    version: Literal[1]
    memories: tuple[GoldenMemory, ...]
    cases: tuple[GoldenCase, ...]


class GoldenEmbedder:
    space_reference = SPACE_REFERENCE

    def __init__(self, query_axes: Mapping[str, int]) -> None:
        self._query_axes = query_axes

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        if request.task is not EmbedTask.QUERY:
            raise AssertionError("golden fixture writes immutable documents directly")
        text = next(part.text for part in request.inputs[0].parts if isinstance(part, TextPart))
        return EmbedResult(
            (
                Embedding(
                    values=_axis_vector(self._query_axes[text]),
                    model_reference=MODEL_REFERENCE,
                    space_reference=SPACE_REFERENCE,
                ),
            )
        )


class FirstCandidateAnswerer:
    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        return (
            GeneratedAnswer(answer=memories[0].summary, confidence=1.0)
            if memories
            else GeneratedAnswer(answer=None, confidence=0.0)
        )

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        return tuple(memory.memory_id for memory in memories)


class DeterministicMediaAccess:
    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )

    async def delete_media(self, media_object: MediaObject) -> None:
        raise AssertionError(f"golden recall must not delete {media_object.media_object_id}")


class DiscardingObservationJobPublisher:
    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None:
        return None


async def test_golden_recall_preserves_retrieval_evidence_and_abstention(
    store: PostgresMemoryStore,
) -> None:
    golden_set = GoldenRecallSet.model_validate_json(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    query_axes = {case.text: case.query_axis for case in golden_set.cases}
    media_access = DeterministicMediaAccess()
    answerer = FirstCandidateAnswerer()
    kernel = MemoryKernel(
        store,
        answerer,
        answerer,
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=DiscardingObservationJobPublisher(),
        embedder=GoldenEmbedder(query_axes),
        minimum_embedding_similarity=0.0,
        clock=lambda: NOW,
    )
    await _seed_golden_memories(store, kernel, golden_set.memories)

    for case in golden_set.cases:
        result = await kernel.recall(
            RecallRequest(
                tenant_id=TENANT_ID,
                query=RecallQuery(text=case.text),
                filters=RecallFilters(occurred_after=case.occurred_after),
            )
        )

        assert [memory.memory_id for memory in result.memories] == list(case.expected_memory_ids), (
            case.name
        )
        assert result.answer == case.expected_answer, case.name
        assert len(result.evidence) == case.expected_evidence_count, case.name
        assert (result.confidence > 0.0) is (case.expected_answer is not None), case.name


async def _seed_golden_memories(
    store: PostgresMemoryStore,
    kernel: MemoryKernel,
    memories: tuple[GoldenMemory, ...],
) -> None:
    for ordinal, fixture in enumerate(memories):
        evidence_ids: tuple[EvidenceId, ...] = ()
        verification_status = VerificationStatus.ATTESTED
        if fixture.source == "video":
            receipt = await kernel.observe(_observe_request(fixture, sequence=ordinal))
            batch = await store.read_observation_batch(
                TENANT_ID,
                ObservationId(receipt.observation_id),
            )
            evidence_id = batch.evidence_spans[0].evidence_id
            evidence_ids = (evidence_id,)
            verification_status = VerificationStatus.VERIFIED
            await store.write_embedding(
                EmbeddingRecord(
                    embedding_id=EmbeddingId(f"embedding_{evidence_id}"),
                    tenant_id=TENANT_ID,
                    object_type=EmbeddedObjectType.EVIDENCE_SPAN,
                    object_id=evidence_id,
                    values=_axis_vector(0),
                    model_reference=MODEL_REFERENCE,
                    space_reference=SPACE_REFERENCE,
                    task="retrieval_document",
                    dimension=VECTOR_DIMENSION,
                    normalized=True,
                    created_at=NOW,
                )
            )
        memory = MemoryRecord(
            memory_id=MemoryId(fixture.memory_id),
            tenant_id=TENANT_ID,
            memory_type=fixture.memory_type,
            summary=fixture.summary,
            evidence_ids=evidence_ids,
            occurred_at=fixture.occurred_at,
            ended_at=fixture.occurred_at,
            created_at=NOW,
            verification_status=verification_status,
        )
        await store.write_memory(
            memory,
            idempotency_key=fixture.memory_id,
            content_digest=("a" if ordinal == 0 else "b") * 64,
        )


def _observe_request(memory: GoldenMemory, *, sequence: int) -> ObserveRequest:
    media_object_id = f"media_{memory.memory_id}"
    return ObserveRequest(
        tenant_id=TENANT_ID,
        device_id="robot_camera",
        boot_id="golden_run",
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id=media_object_id,
                kind=MediaKind.VIDEO,
                uri=f"s3://golden/tenants/{TENANT_ID}/{media_object_id}.mp4",
                sha256=("c" if sequence == 0 else "d") * 64,
                size_bytes=1_024,
                duration_ms=4_000,
                created_at=memory.occurred_at,
            ),
        ),
        occurred_at=memory.occurred_at,
        ended_at=memory.occurred_at + timedelta(seconds=4),
        observed_at=memory.occurred_at + timedelta(seconds=4),
    )


def _axis_vector(axis: int) -> tuple[float, ...]:
    return (float(1 if axis == 0 else -1),) + (0.0,) * (VECTOR_DIMENSION - 1)
