"""Vertical unit checks for retry-safe observation processing."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import process_observation as process_observation_module
from mindbridge.application.evidence_clips import ClipSampling
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
from mindbridge.application.process_observation import (
    ProcessObservation,
    _require_grounded_perception,
)
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
from mindbridge.media.clipping import ClipRequest, MediaClip, audio_windows
from mindbridge.models import Embedding, EmbedRequest, EmbedResult, MediaPart, TextPart

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")
OBSERVATION_ID = ObservationId("observation_01")
JOB_ID = JobId("job_process_observation_01")


class RecordingProcessingStore:
    """Strict fake for one durable processing attempt."""

    def __init__(self, *, claimed_at: datetime = NOW) -> None:
        self._claimed_at = claimed_at
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
        self.job = _job(
            JobState.RUNNING,
            attempt=self.job.attempt + 1,
            updated_at=self._claimed_at,
        )
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
        self.evidence_urls: tuple[str, ...] = ()
        self._error = error

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        self.calls += 1
        self.evidence_urls = tuple(dict.fromkeys(item.media_url for item in evidence))
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

    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self) -> None:
        self.documents: tuple[str, ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.documents = tuple(
            part.url
            for input_value in request.inputs
            for part in input_value.parts
            if isinstance(part, MediaPart)
        )
        return EmbedResult(
            tuple(
                Embedding(
                    (1.0, 0.0),
                    ModelReference(model_id="jina-omni", revision="revision-01"),
                    EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
                )
                for _ in request.inputs
            )
        )


class RecordingTextEmbedder:
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self) -> None:
        self.documents: tuple[str, ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.documents = tuple(
            part.text
            for input_value in request.inputs
            for part in input_value.parts
            if isinstance(part, TextPart)
        )
        return EmbedResult(
            tuple(
                Embedding(
                    (1.0, 0.0),
                    ModelReference(model_id="jina-text", revision="text-revision-01"),
                    EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
                )
                for _ in request.inputs
            )
        )


class DeterministicSigner:
    """Object storage double that really round-trips derived clip bytes."""

    def __init__(self) -> None:
        self.calls = 0
        self.uploaded: dict[str, bytes] = {}
        self.deleted: tuple[str, ...] = ()

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.calls += 1
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/clip.mp4?signature={self.calls}",
            expires_at=NOW + timedelta(minutes=5),
        )

    async def read_media(self, media_object: MediaObject) -> bytes:
        return self.uploaded.get(media_object.uri, b"source-media-bytes")

    async def upload_media(self, media_object: MediaObject, content: bytes) -> None:
        self.uploaded[media_object.uri] = content

    async def delete_media(self, media_object: MediaObject) -> None:
        self.deleted = (*self.deleted, media_object.uri)


def stub_proxy_cut(source: bytes, request: ClipRequest) -> MediaClip:
    """Stand in for the real proxy encoder, and come out smaller like the real one does."""
    return MediaClip(
        content=b"px%d-%d" % (request.start_ms, request.end_ms),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


def stub_cut(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
    """Cut deterministically without decoding, so unit tests stay pure."""
    return tuple(
        MediaClip(
            content=b"clip:%d-%d:" % (start, end) + source,
            suffix=".mp4",
            start_ms=start,
            end_ms=end,
        )
        for start, end in audio_windows(request.start_ms, request.end_ms)
    )


async def test_processor_builds_event_memory_and_raw_media_embedding_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One retry-safe path derives memory while keeping AV as the primary index input."""
    stages: list[tuple[str, float]] = []
    monkeypatch.setattr(
        process_observation_module,
        "record_stage_duration",
        lambda stage, duration: stages.append((stage, duration)),
    )
    processing_clock = iter((100.0, 100.5))
    monkeypatch.setattr(process_observation_module, "perf_counter", lambda: next(processing_clock))
    store = RecordingProcessingStore(claimed_at=NOW + timedelta(seconds=2))
    perceiver = RecordingPerceiver()
    embedder = RecordingEmbedder()
    text_embedder = RecordingTextEmbedder()
    signer = DeterministicSigner()
    processor = _processor(store, perceiver, embedder, text_embedder, signer=signer)

    first = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)
    duplicate = await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert first.state is JobState.SUCCEEDED
    assert duplicate.state is JobState.SUCCEEDED
    assert perceiver.calls == 1
    # Two source signings (perception, clip derivation) plus one each for the perception
    # proxy and the stored clip.
    assert signer.calls == 4
    # The encoder receives the derived clip, never the whole source recording.
    assert embedder.documents == ("https://objects.example.test/clip.mp4?signature=4",)
    assert store.output is not None
    # The stored clip plus the perception proxy, which stays transient: it is never registered
    # as an output media object and the orphan-clip sweep reclaims it.
    assert len(signer.uploaded) == 2
    assert len(store.output.media_objects) == 1
    assert store.output.media_objects[0].derived_from_media_object_id == MediaObjectId("media_01")
    # Clips follow the grounded event span, so the unreferenced source span that
    # used to receive a whole-file vector now produces nothing at all.
    assert len(store.output.evidence_clips) == 1
    assert store.output.evidence_clips[0].evidence_id not in {
        EvidenceId("evidence_01"),
        EvidenceId("evidence_unused"),
    }
    assert first.memory_ids == tuple(memory.memory_id for memory in store.output.memories)
    assert store.output.events[0].prompt_version == "perceive_events_v1"
    assert store.output.memories[0].summary == store.output.events[0].description
    assert len(store.output.evidence_spans) == 1
    event_evidence = store.output.evidence_spans[0]
    assert (event_evidence.start_ms, event_evidence.end_ms) == (500, 3_500)
    assert store.output.events[0].evidence_ids == (event_evidence.evidence_id,)
    assert store.output.claims[0].evidence_ids == (event_evidence.evidence_id,)
    assert store.output.embeddings[0].object_type is EmbeddedObjectType.EVIDENCE_SPAN
    assert store.output.embeddings[0].object_id == event_evidence.evidence_id
    # The clip mapping and the vector agree on which span they describe.
    assert store.output.evidence_clips[0].evidence_id == event_evidence.evidence_id
    assert store.output.evidence_clips[0].start_ms == 500
    assert store.output.evidence_clips[0].end_ms == 3_500
    assert tuple(
        embedding.object_id
        for embedding in store.output.embeddings
        if embedding.object_type is EmbeddedObjectType.EVIDENCE_SPAN
    ) == (event_evidence.evidence_id,)
    # Named entities are their own retrieval entry point; the anonymous identity entity has
    # no name to embed and stays reachable only through the events that mention it.
    assert text_embedder.documents == (
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
        "red tool",
        "blue toolbox",
    )
    assert {embedding.object_type for embedding in store.output.embeddings} == {
        EmbeddedObjectType.EVIDENCE_SPAN,
        EmbeddedObjectType.EVENT,
        EmbeddedObjectType.CLAIM,
        EmbeddedObjectType.ENTITY,
    }
    assert len(store.output.entities) == 3
    assert len(store.output.entity_mentions) == 3
    assert len(store.output.claims) == 1
    assert len(store.output.memories) == 2
    assert len(store.output.relations) == 8
    assert stages == [
        ("cloud.job_to_first_claim", 2.0),
        ("cloud.job_to_searchable_ready", 2.5),
    ]
    identity_mention = next(
        mention
        for mention in store.output.entity_mentions
        if mention.entity_id == "person_robot_01"
    )
    assert identity_mention.event_id == store.output.events[0].event_id
    assert identity_mention.evidence_id == event_evidence.evidence_id


class RepeatedEntityPerceiver(RecordingPerceiver):
    """Names the same object in two events, the way a real clip repeats a subject."""

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        perception = await super().perceive_events(observation, evidence)
        first = perception.events[0]
        second = replace(
            first,
            start_ms=3_000,
            end_ms=3_800,
            description="The person picks the red tool back up.",
            entities=(first.entities[0],),
            claims=(),
        )
        return replace(perception, events=(first, second))


class RecasedEntityPerceiver(RecordingPerceiver):
    """Names one object twice with different capitalisation, as a real model does."""

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        perception = await super().perceive_events(observation, evidence)
        first = perception.events[0]
        capitalised = replace(
            first,
            start_ms=3_000,
            end_ms=3_800,
            description="The person picks the Red Tool back up.",
            entities=(replace(first.entities[0], canonical_name="Red Tool"),),
            claims=(),
        )
        lowercase = first
        return replace(perception, events=(lowercase, capitalised))


async def test_a_case_variant_name_stores_the_value_its_entity_id_was_derived_from() -> None:
    """The ID casefolds the name, so a row whose casing depends on which clip arrived first
    would collide with itself across observations and fail the store's identity check."""
    store = RecordingProcessingStore()
    await _processor(
        store,
        RecasedEntityPerceiver(),
        RecordingEmbedder(),
        RecordingTextEmbedder(),
    ).run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert store.output is not None
    red_tool = [
        entity
        for entity in store.output.entities
        if entity.canonical_name is not None and entity.canonical_name.casefold() == "red tool"
    ]
    assert len(red_tool) == 1
    assert red_tool[0].canonical_name == "red tool"


async def test_a_named_entity_is_one_graph_node_across_the_events_that_mention_it() -> None:
    """Retrieval enters at the entity, so one name must not become one node per event."""
    store = RecordingProcessingStore()
    text_embedder = RecordingTextEmbedder()
    await _processor(
        store,
        RepeatedEntityPerceiver(),
        RecordingEmbedder(),
        text_embedder,
    ).run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert store.output is not None
    red_tool = [entity for entity in store.output.entities if entity.canonical_name == "red tool"]
    assert len(red_tool) == 1
    mentioning_events = {
        relation.source_id
        for relation in store.output.relations
        if relation.relation_type is RelationType.MENTIONS
        and relation.target_id == red_tool[0].entity_id
    }
    assert mentioning_events == {event.event_id for event in store.output.events}
    assert text_embedder.documents.count("red tool") == 1
    assert (
        tuple(
            embedding.object_id
            for embedding in store.output.embeddings
            if embedding.object_type is EmbeddedObjectType.ENTITY
        ).count(red_tool[0].entity_id)
        == 1
    )


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


async def test_processing_output_rejects_memory_evidence_from_another_record() -> None:
    store = RecordingProcessingStore()
    await _processor(
        store,
        RecordingPerceiver(),
        RecordingEmbedder(),
        RecordingTextEmbedder(),
    ).run(TENANT_ID, OBSERVATION_ID, JOB_ID)
    assert store.output is not None
    other_evidence = replace(
        store.output.evidence_spans[0],
        evidence_id=EvidenceId("event_evidence_other"),
    )
    memories = (
        replace(store.output.memories[0], evidence_ids=(other_evidence.evidence_id,)),
        *store.output.memories[1:],
    )

    with pytest.raises(DomainInvariantError, match="memory evidence"):
        replace(
            store.output,
            evidence_spans=(*store.output.evidence_spans, other_evidence),
            memories=memories,
        )


def test_grounding_requires_positive_temporal_overlap_but_accepts_point_evidence() -> None:
    batch = _batch()
    source = replace(batch.evidence_spans[0], end_ms=500)
    resolved = ResolvedEvidence(
        evidence_span=source,
        media_object=batch.media_objects[0],
        media_url="https://objects.example.test/clip.mp4",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )
    event = PerceivedEvent(
        start_ms=500,
        end_ms=1_000,
        description="An event after the source interval.",
        salience=0.5,
        evidence_ids=(source.evidence_id,),
    )

    with pytest.raises(DomainInvariantError, match="does not overlap"):
        _require_grounded_perception(batch.observation, (resolved,), (event,))

    point = replace(resolved, evidence_span=replace(source, start_ms=500))
    _require_grounded_perception(batch.observation, (point,), (event,))


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


def test_processing_rejects_embedders_in_different_search_spaces() -> None:
    """Media and text vectors are compared directly, so one drifted space must fail loudly."""

    class DriftedTextEmbedder(RecordingTextEmbedder):
        space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v2")

    with pytest.raises(ValueError, match="one search space"):
        _processor(
            RecordingProcessingStore(),
            RecordingPerceiver(),
            RecordingEmbedder(),
            DriftedTextEmbedder(),
        )


def _processor(
    store: RecordingProcessingStore,
    perceiver: RecordingPerceiver,
    embedder: RecordingEmbedder,
    text_embedder: RecordingTextEmbedder,
    *,
    signer: DeterministicSigner | None = None,
) -> ProcessObservation:
    return ProcessObservation(
        store,
        perceiver,
        embedder,
        text_embedder,
        media_url_signer=signer or DeterministicSigner(),
        clip_cutter=stub_cut,
        proxy_cutter=stub_proxy_cut,
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
            AnonymousIdentityObservation(
                identity_id="person_robot_01",
                kind=IdentityKind.FACE,
                start_ms=1_500,
                end_ms=2_500,
                confidence=0.82,
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
    updated_at: datetime = NOW,
) -> ObservationProcessingJob:
    return ObservationProcessingJob(
        job_id=JOB_ID,
        tenant_id=TENANT_ID,
        observation_id=OBSERVATION_ID,
        state=state,
        attempt=attempt,
        error_code=error_code,
        created_at=NOW,
        updated_at=updated_at,
        memory_ids=memory_ids,
    )


class ObjectNamingSigner(DeterministicSigner):
    """Signs a URL that names the object it signed, so a proxy is distinguishable."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.calls += 1
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}.mp4",
            expires_at=NOW + timedelta(minutes=5),
        )


async def test_perception_reads_a_sampled_proxy_instead_of_the_untouched_source() -> None:
    """The generation request already pins the frame rate and pixel budget, so a remote model
    downloading the full-resolution source moves bytes it throws away on arrival."""
    store = RecordingProcessingStore()
    perceiver = RecordingPerceiver()
    signer = ObjectNamingSigner()

    await _processor(
        store, perceiver, RecordingEmbedder(), RecordingTextEmbedder(), signer=signer
    ).run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    (perceived_url,) = perceiver.evidence_urls
    assert "/media_clip_" in perceived_url
    (proxy_uri,) = [uri for uri in signer.uploaded if "/clips/proxy-" in uri]
    assert signer.uploaded[proxy_uri] == b"px0-4000"
    # The proxy is transient derived media, not a new source the memory now cites, and nothing
    # downstream could reach it to clean it up later. Its key carries the proxy infix so it can
    # never be the object a registered clip of the same bytes owns.
    assert store.output is not None
    assert proxy_uri not in [item.uri for item in store.output.media_objects]
    assert proxy_uri in signer.deleted


async def test_perception_reads_the_source_when_the_proxy_is_switched_off() -> None:
    """A colocated generator must keep the previous behaviour exactly."""
    store = RecordingProcessingStore()
    perceiver = RecordingPerceiver()
    signer = ObjectNamingSigner()
    processor = ProcessObservation(
        store,
        perceiver,
        RecordingEmbedder(),
        RecordingTextEmbedder(),
        media_url_signer=signer,
        clip_sampling=ClipSampling(generation_proxy=False),
        clip_cutter=stub_cut,
        proxy_cutter=stub_proxy_cut,
    )

    await processor.run(TENANT_ID, OBSERVATION_ID, JOB_ID)

    assert perceiver.evidence_urls == ("https://objects.example.test/media_01.mp4",)
