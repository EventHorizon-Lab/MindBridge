"""Vertical unit checks for retry-safe observation processing."""

from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import (
    EventPerception,
    ObservationBatch,
    ObservationProcessingOutput,
    PerceivedEvent,
    PresignedMediaDownload,
    ProcessObservation,
    ResolvedEvidence,
)
from mindbridge.core import (
    AnonymousIdentityObservation,
    DeviceId,
    EmbeddedObjectType,
    EvidenceId,
    EvidenceSpan,
    IdentityKind,
    JobId,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelReference,
    ModelUnavailableError,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
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
        self.job = _job(JobState.SUCCEEDED, attempt=attempt)
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
    processor = _processor(store, perceiver, embedder)

    first = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)
    duplicate = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert first.state is JobState.SUCCEEDED
    assert duplicate.state is JobState.SUCCEEDED
    assert perceiver.calls == 1
    assert embedder.documents == ("https://objects.example.test/clip.mp4?signature=test",)
    assert store.output is not None
    assert store.output.events[0].prompt_version == "perceive_events_v1"
    assert store.output.memories[0].summary == store.output.events[0].description
    assert store.output.embeddings[0].object_type is EmbeddedObjectType.EVIDENCE_SPAN
    assert store.output.embeddings[0].object_id == "evidence_01"
    assert len(store.output.identity_mentions) == 1
    mention = store.output.identity_mentions[0]
    assert mention.identity_id == "person_robot_01"
    assert mention.event_id == store.output.events[0].event_id
    assert mention.evidence_id == "evidence_01"


async def test_processor_records_sanitized_failure_state() -> None:
    """A retryable model outage leaves a durable error category, never provider details."""
    store = RecordingProcessingStore()
    processor = _processor(
        store,
        RecordingPerceiver(ModelUnavailableError("secret provider detail")),
        RecordingEmbedder(),
    )

    with pytest.raises(ModelUnavailableError, match="secret provider detail"):
        await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert store.job.state is JobState.FAILED
    assert store.job.error_code == "model_unavailable"


def _processor(
    store: RecordingProcessingStore,
    perceiver: RecordingPerceiver,
    embedder: RecordingEmbedder,
) -> ProcessObservation:
    return ProcessObservation(
        store,
        perceiver,
        embedder,
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
        ),
    )


def _job(
    state: JobState,
    *,
    attempt: int,
    error_code: str | None = None,
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
    )
