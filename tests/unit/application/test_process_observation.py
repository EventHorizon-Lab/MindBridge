"""Vertical unit checks for retry-safe observation processing."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application.observation_processing import (
    ObservationBatch,
    ObservationProcessingOutput,
)
from mindbridge.application.perception import (
    EventPerception,
    PerceivedClaim,
    PerceivedEntity,
    PerceivedEvent,
    ResolvedEvidence,
)
from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.application.process_observation import ProcessObservation
from mindbridge.core import (
    AnonymousIdentityObservation,
    ClaimType,
    DatabaseUnavailableError,
    DeviceId,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    EntityType,
    EvidenceId,
    EvidenceSpan,
    IdentityKind,
    JobId,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    RelationType,
    SensorKind,
    TenantId,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")
OBSERVATION_ID = ObservationId("observation_01")
JOB_ID = JobId("job_process_observation_01")


class RecordingProcessingStore:
    """Strict fake for one durable processing attempt."""

    def __init__(self) -> None:
        self.job = _job(JobState.PENDING, attempt=0)
        self.output: ObservationProcessingOutput | None = None

    async def claim_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationJobClaim:
        if self.job.state is JobState.SUCCEEDED:
            return ObservationJobClaim(job=self.job, acquired=False)
        self.job = _job(JobState.RUNNING, attempt=self.job.attempt + 1)
        return ObservationJobClaim(job=self.job, acquired=True)

    async def read_observation_batch(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
    ) -> ObservationBatch:
        return _batch()

    async def commit_observation_processing(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        output: ObservationProcessingOutput,
    ) -> ObservationProcessingJob:
        assert attempt == self.job.attempt
        self.output = output
        self.job = _job(
            JobState.SUCCEEDED,
            attempt=attempt,
            memory_ids=tuple(memory.memory_id for memory in output.memories),
        )
        return self.job

    async def mark_observation_processing_failed(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        error_code: str,
    ) -> ObservationProcessingJob:
        self.job = _job(JobState.FAILED, attempt=attempt, error_code=error_code)
        return self.job


class RecordingPerceiver:
    """Returns one evidence-grounded event or a configured model failure."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self._error = error

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return EventPerception(
            events=(
                PerceivedEvent(
                    start_ms=500,
                    end_ms=3_500,
                    description="A person places a red tool beside a blue toolbox.",
                    salience=0.8,
                    evidence_ids=(EvidenceId("evidence_01"),),
                    entities=(
                        PerceivedEntity(
                            entity_type=EntityType.OBJECT,
                            canonical_name="red tool",
                            confidence=0.94,
                            evidence_ids=(EvidenceId("evidence_01"),),
                        ),
                        PerceivedEntity(
                            entity_type=EntityType.OBJECT,
                            canonical_name="blue toolbox",
                            confidence=0.91,
                            evidence_ids=(EvidenceId("evidence_01"),),
                        ),
                    ),
                    claims=(
                        PerceivedClaim(
                            claim_type=ClaimType.RELATION,
                            statement="The red tool is beside the blue toolbox.",
                            confidence=0.88,
                            evidence_ids=(EvidenceId("evidence_01"),),
                            valid_from_ms=500,
                            valid_to_ms=3_500,
                            entity_indices=(0, 1),
                        ),
                    ),
                ),
            ),
            model_reference=ModelReference(
                model_id="qwen3.8-max",
                revision="serving-revision-01",
            ),
            prompt_version="perceive_events_v1",
        )


class RecordingEmbedder:
    """Proves raw signed media, not a caption, reaches Jina document encoding."""

    model_reference = ModelReference(model_id="jina-omni", revision="revision-01")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
    dimension = 2

    def __init__(self) -> None:
        self.documents: tuple[str | bytes | tuple[str | bytes, ...], ...] = ()

    async def encode_queries(
        self,
        inputs: tuple[str | bytes | tuple[str | bytes, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0),) * len(inputs)

    async def encode_documents(
        self,
        inputs: tuple[str | bytes | tuple[str | bytes, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.documents = inputs
        return ((1.0, 0.0),) * len(inputs)


class RecordingTextEmbedder:
    model_reference = ModelReference(model_id="jina-text", revision="text-revision-01")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
    dimension = 2

    def __init__(self) -> None:
        self.documents: tuple[str, ...] = ()

    async def encode_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.documents = texts
        return ((1.0, 0.0),) * len(texts)


class DeterministicSigner:
    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url="https://objects.example.test/clip.mp4?signature=test",
            expires_at=NOW + timedelta(minutes=5),
        )


async def test_processor_builds_event_memory_and_raw_media_embedding_once() -> None:
    """One retry-safe path derives memory while keeping AV as the primary index input."""
    store = RecordingProcessingStore()
    perceiver = RecordingPerceiver()
    embedder = RecordingEmbedder()
    text_embedder = RecordingTextEmbedder()
    processor = _processor(store, perceiver, embedder, text_embedder)

    first = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)
    duplicate = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert first.state is JobState.SUCCEEDED
    assert duplicate.state is JobState.SUCCEEDED
    assert perceiver.calls == 1
    assert embedder.documents == ("https://objects.example.test/clip.mp4?signature=test",)
    assert store.output is not None
    assert first.memory_ids == tuple(memory.memory_id for memory in store.output.memories)
    assert store.output.events[0].prompt_version == "perceive_events_v1"
    assert store.output.memories[0].summary == store.output.events[0].description
    assert store.output.embeddings[0].object_type is EmbeddedObjectType.EVIDENCE_SPAN
    assert store.output.embeddings[0].object_id == "evidence_01"
    assert tuple(
        embedding.object_id
        for embedding in store.output.embeddings
        if embedding.object_type is EmbeddedObjectType.EVIDENCE_SPAN
    ) == ("evidence_01", "evidence_unused")
    assert text_embedder.documents == (
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    )
    assert {embedding.object_type for embedding in store.output.embeddings} == {
        EmbeddedObjectType.EVIDENCE_SPAN,
        EmbeddedObjectType.EVENT,
        EmbeddedObjectType.CLAIM,
    }
    assert len(store.output.entities) == 3
    assert len(store.output.entity_mentions) == 3
    assert len(store.output.claims) == 1
    assert len(store.output.memories) == 2
    assert len(store.output.relations) == 8
    identity_mention = next(
        mention
        for mention in store.output.entity_mentions
        if mention.entity_id == "person_robot_01"
    )
    assert identity_mention.event_id == store.output.events[0].event_id
    assert identity_mention.evidence_id == "evidence_01"


async def test_processing_output_rejects_a_graph_without_event_memory_link() -> None:
    store = RecordingProcessingStore()
    await _processor(
        store,
        RecordingPerceiver(),
        RecordingEmbedder(),
        RecordingTextEmbedder(),
    ).run(TENANT_ID, OBSERVATION_ID, JOB_ID)
    assert store.output is not None
    relations = tuple(
        relation
        for relation in store.output.relations
        if not (
            relation.relation_type is RelationType.REPRESENTED_BY
            and relation.source_id == store.output.events[0].event_id
        )
    )

    with pytest.raises(DomainInvariantError, match="one-to-one"):
        replace(store.output, relations=relations)


def test_event_perception_rejects_an_unbounded_detail_fanout() -> None:
    entity = PerceivedEntity(
        entity_type=EntityType.OBJECT,
        canonical_name="tool",
        confidence=0.9,
        evidence_ids=(EvidenceId("evidence_01"),),
    )
    event = PerceivedEvent(
        start_ms=0,
        end_ms=1_000,
        description="A bounded event",
        salience=0.5,
        evidence_ids=(EvidenceId("evidence_01"),),
        entities=(entity,) * 64,
    )

    with pytest.raises(DomainInvariantError, match="entity count"):
        EventPerception(
            events=(event,) * 5,
            model_reference=ModelReference(model_id="omni", revision="revision-01"),
            prompt_version="perceive_events_v3",
        )


@pytest.mark.parametrize(
    ("error", "error_code"),
    [
        (DatabaseUnavailableError("database detail"), "database_unavailable"),
        (ModelUnavailableError("secret provider detail"), "model_unavailable"),
        (ModelRequestError("secret provider detail"), "model_request_failed"),
    ],
)
async def test_processor_records_sanitized_failure_state(
    error: RuntimeError,
    error_code: str,
) -> None:
    """A dependency failure leaves a durable category, never provider details."""
    store = RecordingProcessingStore()
    processor = _processor(
        store,
        RecordingPerceiver(error),
        RecordingEmbedder(),
        RecordingTextEmbedder(),
    )

    with pytest.raises(type(error), match="detail"):
        await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert store.job.state is JobState.FAILED
    assert store.job.error_code == error_code


def _processor(
    store: RecordingProcessingStore,
    perceiver: RecordingPerceiver,
    embedder: RecordingEmbedder,
    text_embedder: RecordingTextEmbedder,
) -> ProcessObservation:
    return ProcessObservation(
        store,
        perceiver,
        embedder,
        text_embedder,
        media_url_signer=DeterministicSigner(),
    )


def _batch() -> ObservationBatch:
    media_id = MediaObjectId("media_01")
    observation = Observation(
        observation_id=OBSERVATION_ID,
        tenant_id=TENANT_ID,
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_object_ids=(media_id,),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=4),
        observed_at=NOW,
        clock_offset_ms=0,
        identity_observations=(
            AnonymousIdentityObservation(
                identity_id="person_robot_01",
                kind=IdentityKind.FACE,
                start_ms=250,
                end_ms=2_000,
                confidence=0.97,
                model_reference=ModelReference(
                    model_id="insightface/buffalo_l",
                    revision="1.0.1",
                ),
            ),
        ),
    )
    return ObservationBatch(
        media_objects=(
            MediaObject(
                media_object_id=media_id,
                tenant_id=TENANT_ID,
                kind=MediaKind.VIDEO,
                uri="s3://memory/tenants/tenant_01/clip.mp4",
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
                tenant_id=TENANT_ID,
                observation_id=OBSERVATION_ID,
                media_object_id=media_id,
                start_ms=0,
                end_ms=4_000,
                created_at=NOW,
            ),
            EvidenceSpan(
                evidence_id=EvidenceId("evidence_unused"),
                tenant_id=TENANT_ID,
                observation_id=OBSERVATION_ID,
                media_object_id=media_id,
                start_ms=0,
                end_ms=4_000,
                created_at=NOW,
            ),
        ),
    )


def _job(
    state: JobState,
    *,
    attempt: int,
    error_code: str | None = None,
    memory_ids: tuple[MemoryId, ...] = (),
) -> ObservationProcessingJob:
    return ObservationProcessingJob(
        job_id=JOB_ID,
        tenant_id=TENANT_ID,
        observation_id=OBSERVATION_ID,
        state=state,
        attempt=attempt,
        error_code=error_code,
        created_at=NOW,
        updated_at=NOW,
        memory_ids=memory_ids,
    )
