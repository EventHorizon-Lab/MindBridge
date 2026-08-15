"""Integration checks for atomic observation-derived memory."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TypeAlias, cast

import pytest
from psycopg import AsyncConnection

from mindbridge.application.claim_consolidation import (
    ClaimCandidate,
    ClaimCandidateRequest,
)
from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import (
    ConsolidateEpisodes,
    EpisodeCandidateRequest,
)
from mindbridge.application.episodes import (
    EpisodeConsolidation,
    EpisodeProposal,
)
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
from mindbridge.application.ports import (
    EmbeddingMatch,
    EmbeddingSearch,
    PresignedMediaDownload,
)
from mindbridge.application.process_observation import ProcessObservation
from mindbridge.application.semantic_claims import (
    ClaimConsolidation,
    ClaimRelationshipProposal,
    SemanticClaimProposal,
)
from mindbridge.application.summary_consolidation import (
    SummaryCandidate,
    SummaryCandidateRequest,
    SummaryConsolidation,
    SummaryProposal,
    SummaryScope,
)
from mindbridge.contracts import RecallFilters, RecallQuery, RecallRequest
from mindbridge.core import (
    AnonymousIdentityObservation,
    ClaimId,
    ClaimType,
    DeletionPropagationState,
    DeletionTombstone,
    DeviceId,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    EntityType,
    Event,
    EventHierarchyLevel,
    EventStatus,
    EvidenceId,
    EvidenceSpan,
    ForgetTargetType,
    IdentityKind,
    JobId,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryDeletedError,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryType,
    ModelReference,
    Observation,
    ObservationId,
    RelationType,
    SensorKind,
    TenantId,
    TombstoneId,
    VerificationStatus,
    derive_stable_id,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.models import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    MediaPart,
    TextPart,
)

NOW = datetime.now(timezone.utc).replace(microsecond=0)
MODEL = ModelReference(model_id="qwen3.8-max", revision="serving-revision-01")
EMBEDDING_MODEL = ModelReference(
    model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
    revision="12949877f0092093f366c6450340011320152a05",
)
DerivedCounts: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int]

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
                    entities=(
                        PerceivedEntity(
                            entity_type=EntityType.OBJECT,
                            canonical_name="red tool",
                            confidence=0.94,
                            evidence_ids=(evidence[0].evidence_span.evidence_id,),
                        ),
                        PerceivedEntity(
                            entity_type=EntityType.OBJECT,
                            canonical_name="blue toolbox",
                            confidence=0.91,
                            evidence_ids=(evidence[0].evidence_span.evidence_id,),
                        ),
                    ),
                    claims=(
                        PerceivedClaim(
                            claim_type=ClaimType.RELATION,
                            statement="The red tool is beside the blue toolbox.",
                            confidence=0.88,
                            evidence_ids=(evidence[0].evidence_span.evidence_id,),
                            valid_from_ms=500,
                            valid_to_ms=3_500,
                            entity_indices=(0, 1),
                        ),
                    ),
                ),
            ),
            model_reference=MODEL,
            prompt_version="perceive_events_v4",
        )


class FixedEmbedder:
    """Emit a deterministic unit vector in a configured model dimension."""

    model_reference = EMBEDDING_MODEL
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self, dimension: int = 1_024) -> None:
        self.dimension = dimension
        self.documents: tuple[str, ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.documents = tuple(
            next(part.url for part in item.parts if isinstance(part, MediaPart))
            for item in request.inputs
        )
        vector = (1.0,) + (0.0,) * (self.dimension - 1)
        embedding = Embedding(vector, self.model_reference, self.space_reference)
        return EmbedResult((embedding,) * len(request.inputs))


class FixedTextEmbedder:
    model_reference = ModelReference(
        model_id="jinaai/jina-embeddings-v5-text-small-retrieval",
        revision="6856e76bb72982e58de0620458a4e8b3614da340",
    )
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self, dimension: int = 1_024) -> None:
        self.dimension = dimension
        self.documents: tuple[str, ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.documents = tuple(
            next(part.text for part in item.parts if isinstance(part, TextPart))
            for item in request.inputs
        )
        vector = (1.0,) + (0.0,) * (self.dimension - 1)
        embedding = Embedding(vector, self.model_reference, self.space_reference)
        return EmbedResult((embedding,) * len(request.inputs))


class RecordingEpisodeConsolidator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation:
        self.calls += 1
        assert {item.evidence_span.evidence_id for item in evidence} == {
            evidence_id for event in events for evidence_id in event.evidence_ids
        }
        return _episode_consolidation(events)


class CoordinatedEpisodeConsolidator:
    """Hold two stale proposals until both workers are ready to commit."""

    def __init__(self) -> None:
        self.calls = 0
        self._ready = asyncio.Event()

    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation:
        assert len(evidence) == 2
        self.calls += 1
        if self.calls == 2:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=2)
        return _episode_consolidation(events)


class RecordingClaimConsolidator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation:
        self.calls += 1
        assert {item.evidence_span.evidence_id for item in evidence} == {
            evidence_id for candidate in candidates for evidence_id in candidate.claim.evidence_ids
        }
        return _claim_consolidation(candidates)


class CoordinatedClaimConsolidator(RecordingClaimConsolidator):
    """Hold two stale semantic proposals until both workers are ready to commit."""

    def __init__(self) -> None:
        super().__init__()
        self._ready = asyncio.Event()

    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation:
        consolidation = await super().propose_claims(candidates, evidence)
        if self.calls == 2:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=2)
        return consolidation


class RecordingSummaryConsolidator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_summaries(
        self,
        candidates: tuple[SummaryCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> SummaryConsolidation:
        self.calls += 1
        assert {item.evidence_span.evidence_id for item in evidence} == {
            evidence_id for candidate in candidates for evidence_id in candidate.memory.evidence_ids
        }
        return _summary_consolidation(candidates)


class CoordinatedSummaryConsolidator(RecordingSummaryConsolidator):
    """Hold two stale Summary proposals until both workers are ready to commit."""

    def __init__(self) -> None:
        super().__init__()
        self._ready = asyncio.Event()

    async def propose_summaries(
        self,
        candidates: tuple[SummaryCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> SummaryConsolidation:
        consolidation = await super().propose_summaries(candidates, evidence)
        revision = f"summary-serving-revision-{self.calls:02d}"
        if self.calls == 2:
            self._ready.set()
        await asyncio.wait_for(self._ready.wait(), timeout=2)
        return SummaryConsolidation(
            summaries=consolidation.summaries,
            model_reference=ModelReference(model_id="qwen3.8-max", revision=revision),
            prompt_version=consolidation.prompt_version,
        )


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
    text_embedder = FixedTextEmbedder()
    processor = ProcessObservation(
        store,
        perceiver,
        embedder,
        text_embedder,
        media_url_signer=DeterministicSigner(),
    )

    first = await processor.run(tenant_id, observation_id, job_id)
    duplicate = await processor.run(tenant_id, observation_id, job_id)
    repeated_observation = await store.write_observation(
        await store.read_observation_batch(tenant_id, observation_id),
        idempotency_key="observe_01",
        content_digest=f"{101:064x}",
    )

    assert first.state is JobState.SUCCEEDED
    assert duplicate.state is JobState.SUCCEEDED
    assert len(first.memory_ids) == 2
    assert duplicate.memory_ids == first.memory_ids
    assert repeated_observation.created is False
    assert repeated_observation.processing_job_id == job_id
    assert perceiver.calls == 1
    assert embedder.documents == ("https://objects.example.test/media_01.mp4",)
    assert text_embedder.documents == (
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    )
    assert await _derived_counts(database_url, tenant_id) == (
        1,
        1,
        1,
        2,
        2,
        8,
        3,
        3,
        3,
        1,
        1,
    )
    assert await _job_state(database_url, tenant_id, job_id) == ("succeeded", 1, None)
    assert await _event_provenance(database_url, tenant_id) == (
        "A person places a red tool beside a blue toolbox.",
        "qwen3.8-max",
        "serving-revision-01",
        "perceive_events_v4",
    )
    assert await _claim_provenance(database_url, tenant_id) == (
        "relation",
        "The red tool is beside the blue toolbox.",
        "qwen3.8-max",
        "serving-revision-01",
        "perceive_events_v4",
    )
    assert await _relation_counts(database_url, tenant_id) == (
        ("about", 2),
        ("asserts", 1),
        ("mentions", 3),
        ("represented_by", 2),
    )
    stored_evidence = (await store.read_observation_batch(tenant_id, observation_id)).evidence_spans
    event_evidence = next(
        span for span in stored_evidence if (span.start_ms, span.end_ms) == (500, 3_500)
    )
    assert len(stored_evidence) == 2
    request = RecallRequest(
        tenant_id=tenant_id,
        query=RecallQuery(text="words absent from the memory summary"),
    )
    dense_candidates = await store.search_memories_by_evidence(
        request,
        (event_evidence.evidence_id,),
        limit=20,
    )
    source_dense_candidates = await store.search_memories_by_evidence(
        request,
        (EvidenceId("evidence_01"),),
        limit=20,
    )
    graph_matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=tenant_id,
            values=(1.0,) + (0.0,) * 1_023,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.EVENT, EmbeddedObjectType.CLAIM),
            limit=20,
        )
    )
    graph_candidates = await store.search_memories_by_graph_objects(
        request,
        graph_matches,
        limit=20,
    )
    filtered_candidates = await store.search_memories_by_evidence(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="irrelevant"),
            filters=RecallFilters(device_ids=("other_device",)),
        ),
        (event_evidence.evidence_id,),
        limit=20,
    )
    person_candidates = await store.search_memories_by_evidence(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="irrelevant"),
            filters=RecallFilters(person_ids=("person_robot_01",)),
        ),
        (event_evidence.evidence_id,),
        limit=20,
    )

    assert {memory.summary for memory in dense_candidates} == {
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    }
    assert {memory.summary for memory in source_dense_candidates} == {
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    }
    assert {memory.summary for memory in graph_candidates} == {
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    }
    assert filtered_candidates == ()
    assert any(
        memory.summary == "A person places a red tool beside a blue toolbox."
        for memory in person_candidates
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
        FixedTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    )

    with pytest.raises(DomainInvariantError, match="cloud embedding dimension"):
        await failing.run(tenant_id, observation_id, job_id)

    assert await _derived_counts(database_url, tenant_id) == (0,) * 11
    assert await _job_state(database_url, tenant_id, job_id) == (
        "failed",
        1,
        "domain_invariant_failed",
    )

    succeeded = await ProcessObservation(
        store,
        RecordingPerceiver(),
        FixedEmbedder(),
        FixedTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(tenant_id, observation_id, job_id)

    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.attempt == 2
    assert await _derived_counts(database_url, tenant_id) == (
        1,
        1,
        1,
        2,
        2,
        8,
        3,
        3,
        3,
        1,
        1,
    )


async def test_episode_candidates_expand_a_stable_seed_by_event_vector(
    store: PostgresMemoryStore,
) -> None:
    tenant_id, first_observation_id, first_job_id = await _write_source_observation(
        store,
        "tenant_episode_candidates",
        include_identity=False,
    )
    _, second_observation_id, second_job_id = await _write_source_observation(
        store,
        "tenant_episode_candidates",
        ordinal=2,
        occurred_at=NOW + timedelta(seconds=5),
        include_identity=False,
    )
    for observation_id, job_id in (
        (first_observation_id, first_job_id),
        (second_observation_id, second_job_id),
    ):
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    page = await store.list_episode_candidates(
        EpisodeCandidateRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW + timedelta(days=2),
            limit=1,
            maximum_gap_seconds=10,
            minimum_similarity=0.99,
        )
    )

    assert page.scanned_count == 1
    assert page.next_cursor is not None
    assert len(page.events) == 2
    assert {event.hierarchy_level for event in page.events} == {EventHierarchyLevel.EVENT}
    assert {event.status for event in page.events} == {EventStatus.ACTIVE}
    assert all(event.parent_event_id is None for event in page.events)


async def test_claim_candidates_expand_a_stable_seed_by_aligned_vector(
    store: PostgresMemoryStore,
) -> None:
    tenant_id, first_observation_id, first_job_id = await _write_source_observation(
        store,
        "tenant_claim_candidates",
        include_identity=False,
    )
    _, second_observation_id, second_job_id = await _write_source_observation(
        store,
        "tenant_claim_candidates",
        ordinal=2,
        occurred_at=NOW + timedelta(seconds=5),
        include_identity=False,
    )
    for observation_id, job_id in (
        (first_observation_id, first_job_id),
        (second_observation_id, second_job_id),
    ):
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    page = await store.list_claim_candidates(
        ClaimCandidateRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW + timedelta(days=2),
            limit=1,
            maximum_gap_seconds=10,
            minimum_similarity=0.99,
        )
    )

    assert page.scanned_count == 1
    assert page.next_cursor is not None
    assert len(page.candidates) == 2
    assert {candidate.claim.statement for candidate in page.candidates} == {
        "The red tool is beside the blue toolbox."
    }
    assert all(len(candidate.entity_ids) == 2 for candidate in page.candidates)


async def test_summary_candidates_expand_a_stable_seed_across_memory_representations(
    store: PostgresMemoryStore,
) -> None:
    tenant_id, first_observation_id, first_job_id = await _write_source_observation(
        store,
        "tenant_summary_candidates",
        include_identity=False,
    )
    _, second_observation_id, second_job_id = await _write_source_observation(
        store,
        "tenant_summary_candidates",
        ordinal=2,
        occurred_at=NOW + timedelta(days=2),
        include_identity=False,
    )
    for observation_id, job_id in (
        (first_observation_id, first_job_id),
        (second_observation_id, second_job_id),
    ):
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    page = await store.list_summary_candidates(
        SummaryCandidateRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW + timedelta(days=3),
            limit=1,
            maximum_gap_seconds=0,
            minimum_similarity=0.99,
        )
    )

    assert page.scanned_count == 1
    assert page.next_cursor is not None
    assert len(page.candidates) == 4
    assert page.next_cursor.occurred_at == min(
        candidate.memory.occurred_at for candidate in page.candidates
    )
    assert {candidate.memory.memory_type for candidate in page.candidates} == {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    }
    assert all(len(candidate.entity_ids) == 2 for candidate in page.candidates)


async def test_summary_cursor_survives_forgetting_the_previous_seed(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_summary_cursor_forget")
    for ordinal in range(3):
        occurred_at = NOW + timedelta(minutes=ordinal)
        memory = MemoryRecord(
            memory_id=MemoryId(f"memory_{ordinal}"),
            tenant_id=tenant_id,
            memory_type=MemoryType.EPISODIC,
            summary=f"Attested event {ordinal}",
            evidence_ids=(),
            occurred_at=occurred_at,
            ended_at=occurred_at,
            created_at=occurred_at,
            verification_status=VerificationStatus.ATTESTED,
        )
        await store.write_memory(
            memory,
            idempotency_key=f"summary_cursor_{ordinal}",
            content_digest=f"{ordinal + 1:x}" * 64,
        )

    first = await store.list_summary_candidates(
        SummaryCandidateRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW + timedelta(days=1),
            limit=1,
            maximum_gap_seconds=0,
        )
    )
    assert first.next_cursor is not None
    await _forget_memory(
        store,
        tenant_id,
        first.next_cursor.memory_id,
        requested_at=NOW + timedelta(days=2),
        ordinal=10,
    )

    second = await store.list_summary_candidates(
        SummaryCandidateRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW + timedelta(days=1),
            after_cursor=first.next_cursor,
            limit=1,
            maximum_gap_seconds=0,
        )
    )

    assert second.scanned_count == 1
    assert second.next_cursor is not None
    assert second.next_cursor.memory_id == "memory_1"


async def test_episode_consolidation_is_atomic_recallable_and_retry_safe(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id, first_observation_id, first_job_id = await _write_source_observation(
        store,
        "tenant_episode_commit",
    )
    _, second_observation_id, second_job_id = await _write_source_observation(
        store,
        "tenant_episode_commit",
        ordinal=2,
        occurred_at=NOW + timedelta(seconds=5),
    )
    for observation_id, job_id in (
        (first_observation_id, first_job_id),
        (second_observation_id, second_job_id),
    ):
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    request = EpisodeCandidateRequest(
        tenant_id=tenant_id,
        evaluated_at=NOW + timedelta(days=2),
        maximum_gap_seconds=10,
        minimum_similarity=0.99,
    )
    candidate_events = (await store.list_episode_candidates(request)).events
    child_event_id = candidate_events[0].event_id
    identity_expanded_memories = await store.search_memories_by_graph_objects(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="same person")),
        (
            EmbeddingMatch(
                embedding_id="isolated_identity_hit",
                object_type=EmbeddedObjectType.EVENT,
                object_id=child_event_id,
                similarity=1.0,
            ),
        ),
        limit=20,
    )
    assert {memory.memory_id for memory in identity_expanded_memories} >= {
        derive_stable_id("memory", event.event_id) for event in candidate_events
    }
    consolidator = RecordingEpisodeConsolidator()
    with pytest.raises(DomainInvariantError, match="cloud embedding dimension"):
        await ConsolidateEpisodes(
            store,
            consolidator,
            FixedTextEmbedder(dimension=2),
            media_url_signer=DeterministicSigner(),
        ).run(request)

    assert await _episode_counts(database_url, tenant_id) == (0, 0, 0, 0, 0, 0, 0)

    coordinated = CoordinatedEpisodeConsolidator()
    results = await asyncio.gather(
        *(
            ConsolidateEpisodes(
                store,
                coordinated,
                FixedTextEmbedder(),
                media_url_signer=DeterministicSigner(),
            ).run(request)
            for _ in range(2)
        )
    )
    replay_consolidator = RecordingEpisodeConsolidator()
    replay = await ConsolidateEpisodes(
        store,
        replay_consolidator,
        FixedTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(request)

    assert sorted(result.committed_count for result in results) == [0, 1]
    assert replay.candidate_count == 0
    assert replay.committed_count == 0
    assert coordinated.calls == 2
    assert replay_consolidator.calls == 0
    assert await _episode_counts(database_url, tenant_id) == (1, 2, 2, 2, 5, 1, 2)

    expanded_memories = await store.search_memories_by_graph_objects(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="repair episode")),
        (
            EmbeddingMatch(
                embedding_id="isolated_child_hit",
                object_type=EmbeddedObjectType.EVENT,
                object_id=child_event_id,
                similarity=1.0,
            ),
        ),
        limit=20,
    )
    assert {memory.summary for memory in expanded_memories} >= {
        "A person places a red tool beside a blue toolbox.",
        "A person retrieves a tool and explains the repair.",
    }

    graph_matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=tenant_id,
            values=(1.0,) + (0.0,) * 1_023,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.EVENT,),
            limit=20,
        )
    )
    memories = await store.search_memories_by_graph_objects(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="repair episode"),
        ),
        graph_matches,
        limit=20,
    )
    assert any(
        memory.summary == "A person retrieves a tool and explains the repair."
        for memory in memories
    )


async def test_claim_consolidation_is_atomic_versioned_and_forget_safe(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id = TenantId("tenant_claim_commit")
    sources: tuple[tuple[TenantId, ObservationId, JobId], ...] = tuple(
        [
            await _write_source_observation(
                store,
                tenant_id,
                ordinal=ordinal,
                occurred_at=NOW + timedelta(seconds=5 * (ordinal - 1)),
                include_identity=False,
            )
            for ordinal in range(1, 5)
        ]
    )
    for _, observation_id, job_id in sources:
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    request = ClaimCandidateRequest(
        tenant_id=tenant_id,
        evaluated_at=NOW + timedelta(days=2),
        maximum_gap_seconds=30,
        minimum_similarity=0.99,
    )
    candidates = (await store.list_claim_candidates(request)).candidates
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.claim.valid_from))
    supporting_claim_id = ordered[0].claim.claim_id
    source_claim_id = ordered[3].claim.claim_id
    target_claim_id = ordered[2].claim.claim_id
    with pytest.raises(DomainInvariantError, match="cloud embedding dimension"):
        await ConsolidateClaims(
            store,
            RecordingClaimConsolidator(),
            FixedTextEmbedder(dimension=2),
            media_url_signer=DeterministicSigner(),
        ).run(request)

    assert await _semantic_claim_counts(database_url, tenant_id) == (0,) * 5
    assert await _claim_version_state(
        database_url,
        tenant_id,
        source_claim_id,
        target_claim_id,
    ) == (None, None, None)

    coordinated = CoordinatedClaimConsolidator()
    results = await asyncio.gather(
        *(
            ConsolidateClaims(
                store,
                coordinated,
                FixedTextEmbedder(),
                media_url_signer=DeterministicSigner(),
            ).run(request)
            for _ in range(2)
        )
    )
    assert sorted(
        (result.committed_semantic_claim_count, result.committed_relationship_count)
        for result in results
    ) == [(0, 0), (1, 1)]
    assert await _semantic_claim_counts(database_url, tenant_id) == (1, 1, 2, 2, 1)
    source_target, target_superseded_at, memory_superseded_at = await _claim_version_state(
        database_url,
        tenant_id,
        source_claim_id,
        target_claim_id,
    )
    assert source_target == target_claim_id
    assert target_superseded_at == request.evaluated_at
    assert memory_superseded_at == request.evaluated_at
    next_candidate_ids = {
        candidate.claim.claim_id
        for candidate in (await store.list_claim_candidates(request)).candidates
    }
    assert next_candidate_ids == set()

    expanded_memories = await store.search_memories_by_graph_objects(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="red tool")),
        (
            EmbeddingMatch(
                embedding_id="isolated_support_hit",
                object_type=EmbeddedObjectType.CLAIM,
                object_id=supporting_claim_id,
                similarity=1.0,
            ),
        ),
        limit=20,
    )
    assert {memory.summary for memory in expanded_memories} >= {
        "The red tool is beside the blue toolbox.",
        _SEMANTIC_CLAIM_STATEMENT,
    }

    graph_matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=tenant_id,
            values=(1.0,) + (0.0,) * 1_023,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.CLAIM,),
            limit=20,
        )
    )
    memories = await store.search_memories_by_graph_objects(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="red tool")),
        graph_matches,
        limit=20,
    )
    assert any(memory.summary == _SEMANTIC_CLAIM_STATEMENT for memory in memories)

    forgotten_observation_id = sources[3][1]
    tombstone = DeletionTombstone(
        tombstone_id=TombstoneId(
            derive_stable_id(
                "tombstone",
                tenant_id,
                ForgetTargetType.OBSERVATION.value,
                forgotten_observation_id,
            )
        ),
        tenant_id=tenant_id,
        target_type=ForgetTargetType.OBSERVATION,
        target_id=forgotten_observation_id,
        propagation_state=DeletionPropagationState.PENDING,
        requested_at=request.evaluated_at,
    )
    plan = await store.prepare_forget(
        tombstone,
        idempotency_key="forget_superseding_claim",
        content_digest="f" * 64,
    )
    await store.complete_forget(
        plan.tombstone,
        completed_at=request.evaluated_at + timedelta(seconds=1),
    )

    assert await _claim_version_state(
        database_url,
        tenant_id,
        source_claim_id,
        target_claim_id,
    ) == (None, None, None)


async def test_summary_consolidation_is_atomic_recallable_and_retry_safe(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id, first_observation_id, first_job_id = await _write_source_observation(
        store,
        "tenant_summary_commit",
        include_identity=False,
    )
    _, second_observation_id, second_job_id = await _write_source_observation(
        store,
        "tenant_summary_commit",
        ordinal=2,
        occurred_at=NOW + timedelta(seconds=5),
        include_identity=False,
    )
    for observation_id, job_id in (
        (first_observation_id, first_job_id),
        (second_observation_id, second_job_id),
    ):
        await ProcessObservation(
            store,
            RecordingPerceiver(),
            FixedEmbedder(),
            FixedTextEmbedder(),
            media_url_signer=DeterministicSigner(),
        ).run(tenant_id, observation_id, job_id)

    request = SummaryCandidateRequest(
        tenant_id=tenant_id,
        evaluated_at=NOW + timedelta(days=2),
        maximum_gap_seconds=10,
        minimum_similarity=0.99,
    )
    with pytest.raises(DomainInvariantError, match="cloud embedding dimension"):
        await ConsolidateSummaries(
            store,
            RecordingSummaryConsolidator(),
            FixedTextEmbedder(dimension=2),
            media_url_signer=DeterministicSigner(),
        ).run(request)

    assert await _summary_counts(database_url, tenant_id) == (0, 0, 0, 0)

    coordinated = CoordinatedSummaryConsolidator()
    results = await asyncio.gather(
        *(
            ConsolidateSummaries(
                store,
                coordinated,
                FixedTextEmbedder(),
                media_url_signer=DeterministicSigner(),
            ).run(request)
            for _ in range(2)
        )
    )
    replay_consolidator = RecordingSummaryConsolidator()
    replay = await ConsolidateSummaries(
        store,
        replay_consolidator,
        FixedTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(request)

    assert sorted(result.committed_count for result in results) == [0, 1]
    assert replay.candidate_count == 0
    assert replay.committed_count == 0
    assert coordinated.calls == 2
    assert replay_consolidator.calls == 0
    assert await _summary_counts(database_url, tenant_id) == (1, 2, 4, 1)

    matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=tenant_id,
            values=(1.0,) + (0.0,) * 1_023,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=20,
        )
    )
    strict_memories = await store.search_memories_by_ids(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="repair session")),
        tuple(MemoryId(match.object_id) for match in matches),
        limit=20,
    )
    assert [memory.summary for memory in strict_memories] == [
        "Across the session, a person kept a red tool beside a blue toolbox."
    ]
    expanded_memories = await store.search_memories_by_hierarchy(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="repair session")),
        tuple(MemoryId(match.object_id) for match in matches),
        limit=20,
    )
    assert expanded_memories[0].memory_id == strict_memories[0].memory_id
    assert len(expanded_memories) == 5
    assert {memory.summary for memory in expanded_memories[1:]} == {
        "A person places a red tool beside a blue toolbox.",
        "The red tool is beside the blue toolbox.",
    }

    first_summary_id = strict_memories[0].memory_id
    source_memory_ids = tuple(memory.memory_id for memory in expanded_memories[1:])
    bounded_roots = await store.search_memories_by_hierarchy(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="red tool")),
        (source_memory_ids[0], source_memory_ids[-1]),
        limit=2,
    )
    assert [memory.memory_id for memory in bounded_roots] == [
        source_memory_ids[0],
        source_memory_ids[-1],
    ]
    expanded_from_source = await store.search_memories_by_hierarchy(
        RecallRequest(tenant_id=tenant_id, query=RecallQuery(text="red tool")),
        (source_memory_ids[0],),
        limit=20,
    )
    assert first_summary_id in {memory.memory_id for memory in expanded_from_source}
    assert set(source_memory_ids) <= {memory.memory_id for memory in expanded_from_source}
    await _forget_memory(
        store,
        tenant_id,
        first_summary_id,
        requested_at=request.evaluated_at + timedelta(seconds=1),
        ordinal=1,
    )
    with pytest.raises(MemoryDeletedError):
        await store.read_memory(tenant_id, first_summary_id)
    for source_memory_id in source_memory_ids:
        await store.read_memory(tenant_id, source_memory_id)
    assert await _summary_counts(database_url, tenant_id) == (0, 0, 0, 0)

    rebuilt_request = SummaryCandidateRequest(
        tenant_id=tenant_id,
        evaluated_at=request.evaluated_at + timedelta(days=1),
        maximum_gap_seconds=10,
        minimum_similarity=0.99,
    )
    rebuilt = await ConsolidateSummaries(
        store,
        RecordingSummaryConsolidator(),
        FixedTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(rebuilt_request)
    assert rebuilt.committed_count == 1
    rebuilt_matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=tenant_id,
            values=(1.0,) + (0.0,) * 1_023,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=20,
        )
    )
    assert len(rebuilt_matches) == 1
    rebuilt_summary_id = MemoryId(rebuilt_matches[0].object_id)

    await _forget_memory(
        store,
        tenant_id,
        source_memory_ids[0],
        requested_at=rebuilt_request.evaluated_at + timedelta(seconds=1),
        ordinal=2,
    )
    with pytest.raises(MemoryDeletedError):
        await store.read_memory(tenant_id, source_memory_ids[0])
    with pytest.raises(MemoryNotFoundError):
        await store.read_memory(tenant_id, rebuilt_summary_id)
    for source_memory_id in source_memory_ids[1:]:
        await store.read_memory(tenant_id, source_memory_id)
    assert await _summary_counts(database_url, tenant_id) == (0, 0, 0, 0)


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
    await store.commit_observation_processing(
        tenant_id,
        observation_id,
        job_id,
        attempt=second.job.attempt,
        output=ObservationProcessingOutput(
            evidence_spans=(),
            events=(),
            entities=(),
            entity_mentions=(),
            claims=(),
            memories=(),
            relations=(),
            embeddings=(),
        ),
    )

    with pytest.raises(MemoryIntegrityError, match="attempt was superseded"):
        await store.commit_observation_processing(
            tenant_id,
            observation_id,
            job_id,
            attempt=first.job.attempt,
            output=ObservationProcessingOutput(
                evidence_spans=(),
                events=(),
                entities=(),
                entity_mentions=(),
                claims=(),
                memories=(),
                relations=(),
                embeddings=(),
            ),
        )

    assert first.job.attempt == 1
    assert second.job.attempt == 2
    assert await _job_state(database_url, tenant_id, job_id) == ("succeeded", 2, None)


async def _write_source_observation(
    store: PostgresMemoryStore,
    tenant: str,
    *,
    ordinal: int = 1,
    occurred_at: datetime = NOW,
    include_identity: bool = True,
) -> tuple[TenantId, ObservationId, JobId]:
    tenant_id = TenantId(tenant)
    suffix = f"{ordinal:02d}"
    observation_id = ObservationId(f"observation_{suffix}")
    media_object_id = MediaObjectId(f"media_{suffix}")
    observation = Observation(
        observation_id=observation_id,
        tenant_id=tenant_id,
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=ordinal,
        sensor=SensorKind.CAMERA,
        media_object_ids=(media_object_id,),
        occurred_at=occurred_at,
        ended_at=occurred_at + timedelta(seconds=4),
        observed_at=occurred_at,
        clock_offset_ms=0,
        identity_observations=(
            (
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
            )
            if include_identity
            else ()
        ),
    )
    result = await store.write_observation(
        ObservationBatch(
            media_objects=(
                MediaObject(
                    media_object_id=media_object_id,
                    tenant_id=tenant_id,
                    kind=MediaKind.VIDEO,
                    uri=f"s3://memory/{tenant}/clip_{suffix}.mp4",
                    sha256=f"{ordinal:064x}",
                    size_bytes=100,
                    created_at=NOW,
                    duration_ms=4_000,
                ),
            ),
            observation=observation,
            evidence_spans=(
                EvidenceSpan(
                    evidence_id=EvidenceId(f"evidence_{suffix}"),
                    tenant_id=tenant_id,
                    observation_id=observation_id,
                    media_object_id=media_object_id,
                    start_ms=0,
                    end_ms=4_000,
                    created_at=NOW,
                ),
            ),
        ),
        idempotency_key=f"observe_{suffix}",
        content_digest=f"{ordinal + 100:064x}",
    )
    return tenant_id, observation_id, result.processing_job_id


def _episode_consolidation(events: tuple[Event, ...]) -> EpisodeConsolidation:
    return EpisodeConsolidation(
        episodes=(
            EpisodeProposal(
                event_ids=tuple(event.event_id for event in events),
                description="A person retrieves a tool and explains the repair.",
                salience=0.9,
            ),
        ),
        model_reference=ModelReference(
            model_id="qwen3.8-max",
            revision="episode-serving-revision-01",
        ),
        prompt_version="consolidate_episodes_v1",
    )


_SEMANTIC_CLAIM_STATEMENT = "Across two observations, the red tool remained beside the toolbox."


def _claim_consolidation(candidates: tuple[ClaimCandidate, ...]) -> ClaimConsolidation:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.claim.valid_from))
    assert len(ordered) == 4
    return ClaimConsolidation(
        semantic_claims=(
            SemanticClaimProposal(
                source_claim_ids=(ordered[0].claim.claim_id, ordered[1].claim.claim_id),
                statement=_SEMANTIC_CLAIM_STATEMENT,
                confidence=0.94,
            ),
        ),
        relationships=(
            ClaimRelationshipProposal(
                source_claim_id=ordered[3].claim.claim_id,
                relation_type=RelationType.SUPERSEDES,
                target_claim_id=ordered[2].claim.claim_id,
            ),
        ),
        model_reference=ModelReference(
            model_id="qwen3.8-max",
            revision="claim-serving-revision-01",
        ),
        prompt_version="consolidate_claims_v1",
    )


def _summary_consolidation(
    candidates: tuple[SummaryCandidate, ...],
) -> SummaryConsolidation:
    assert len(candidates) == 4
    return SummaryConsolidation(
        summaries=(
            SummaryProposal(
                source_memory_ids=tuple(candidate.memory.memory_id for candidate in candidates),
                scope=SummaryScope.SESSION,
                summary="Across the session, a person kept a red tool beside a blue toolbox.",
                salience=0.9,
            ),
        ),
        model_reference=ModelReference(
            model_id="qwen3.8-max",
            revision="summary-serving-revision-01",
        ),
        prompt_version="consolidate_summaries_v1",
    )


async def _forget_memory(
    store: PostgresMemoryStore,
    tenant_id: TenantId,
    memory_id: MemoryId,
    *,
    requested_at: datetime,
    ordinal: int,
) -> None:
    tombstone = DeletionTombstone(
        tombstone_id=TombstoneId(
            derive_stable_id(
                "tombstone",
                tenant_id,
                ForgetTargetType.MEMORY_RECORD.value,
                memory_id,
            )
        ),
        tenant_id=tenant_id,
        target_type=ForgetTargetType.MEMORY_RECORD,
        target_id=memory_id,
        propagation_state=DeletionPropagationState.PENDING,
        requested_at=requested_at,
    )
    plan = await store.prepare_forget(
        tombstone,
        idempotency_key=f"forget_summary_scope_{ordinal}",
        content_digest=f"{ordinal:x}" * 64,
    )
    await store.complete_forget(
        plan.tombstone,
        completed_at=requested_at + timedelta(seconds=1),
    )


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
                    (SELECT count(*) FROM embeddings WHERE tenant_id = %s),
                    (SELECT count(*) FROM entities WHERE tenant_id = %s),
                    (SELECT count(*) FROM entity_mentions WHERE tenant_id = %s),
                    (SELECT count(*) FROM claims WHERE tenant_id = %s),
                    (SELECT count(*) FROM claim_evidence WHERE tenant_id = %s)
                """,
                (tenant_id,) * 11,
            )
        ).fetchone()
    return cast(DerivedCounts, row)


async def _episode_counts(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[int, int, int, int, int, int, int]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                WITH episode AS (
                    SELECT event_id FROM events
                    WHERE tenant_id = %s AND hierarchy_level = 'episode'
                )
                SELECT
                    (SELECT count(*) FROM episode),
                    (SELECT count(*) FROM events AS child
                     WHERE child.tenant_id = %s
                       AND child.parent_event_id IN (SELECT event_id FROM episode)),
                    (SELECT count(*) FROM event_observations AS link
                     WHERE link.tenant_id = %s
                       AND link.event_id IN (SELECT event_id FROM episode)),
                    (SELECT count(*) FROM event_evidence AS link
                     WHERE link.tenant_id = %s
                       AND link.event_id IN (SELECT event_id FROM episode)),
                    (SELECT count(*) FROM relations AS relation
                     WHERE relation.tenant_id = %s
                       AND (
                           (relation.source_type = 'event'
                            AND relation.source_id IN (SELECT event_id FROM episode))
                           OR
                           (relation.target_type = 'event'
                            AND relation.target_id IN (SELECT event_id FROM episode))
                       )),
                    (SELECT count(*) FROM embeddings AS embedding
                     WHERE embedding.tenant_id = %s
                       AND embedding.object_type = 'event'
                       AND embedding.object_id IN (SELECT event_id FROM episode)),
                    (SELECT count(*) FROM relations AS relation
                     WHERE relation.tenant_id = %s
                       AND relation.relation_type IN ('before', 'after')
                       AND relation.source_id IN (
                           SELECT child.event_id FROM events AS child
                           WHERE child.tenant_id = %s
                             AND child.parent_event_id IN (SELECT event_id FROM episode)
                       ))
                """,
                (tenant_id,) * 8,
            )
        ).fetchone()
    return cast(tuple[int, int, int, int, int, int, int], row)


async def _semantic_claim_counts(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[int, int, int, int, int]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                WITH semantic_claim AS (
                    SELECT claim_id FROM claims
                    WHERE tenant_id = %s AND prompt_version = 'consolidate_claims_v1'
                )
                SELECT
                    (SELECT count(*) FROM semantic_claim),
                    (SELECT count(*) FROM memory_records AS memory
                     WHERE memory.tenant_id = %s
                       AND memory.model_revision = 'claim-serving-revision-01'),
                    (SELECT count(*) FROM claim_evidence AS link
                     WHERE link.tenant_id = %s
                       AND link.claim_id IN (SELECT claim_id FROM semantic_claim)),
                    (SELECT count(*) FROM relations AS relation
                     WHERE relation.tenant_id = %s
                       AND relation.relation_type = 'supports'
                       AND relation.target_id IN (SELECT claim_id FROM semantic_claim)),
                    (SELECT count(*) FROM relations AS relation
                     WHERE relation.tenant_id = %s
                       AND relation.relation_type = 'supersedes')
                """,
                (tenant_id,) * 5,
            )
        ).fetchone()
    return cast(tuple[int, int, int, int, int], row)


async def _summary_counts(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[int, int, int, int]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                WITH summary AS (
                    SELECT memory_id FROM memory_records
                    WHERE tenant_id = %s
                      AND model_revision LIKE 'summary-serving-revision-%%'
                )
                SELECT
                    (SELECT count(*) FROM summary),
                    (SELECT count(*) FROM memory_evidence AS link
                     WHERE link.tenant_id = %s
                       AND link.memory_id IN (SELECT memory_id FROM summary)),
                    (SELECT count(*) FROM relations AS relation
                     WHERE relation.tenant_id = %s
                       AND relation.source_type = 'memory_record'
                       AND relation.source_id IN (SELECT memory_id FROM summary)
                       AND relation.relation_type = 'contains'
                       AND relation.target_type = 'memory_record'),
                    (SELECT count(*) FROM embeddings AS embedding
                     WHERE embedding.tenant_id = %s
                       AND embedding.object_type = 'memory_record'
                       AND embedding.object_id IN (SELECT memory_id FROM summary))
                """,
                (tenant_id,) * 4,
            )
        ).fetchone()
    return cast(tuple[int, int, int, int], row)


async def _claim_version_state(
    database_url: str,
    tenant_id: TenantId,
    source_claim_id: ClaimId,
    target_claim_id: ClaimId,
) -> tuple[str | None, datetime | None, datetime | None]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT source.supersedes_claim_id,
                       target.superseded_at,
                       memory.superseded_at
                FROM claims AS target
                LEFT JOIN claims AS source
                  ON source.tenant_id = target.tenant_id AND source.claim_id = %s
                JOIN relations AS representation
                  ON representation.tenant_id = target.tenant_id
                 AND representation.source_type = 'claim'
                 AND representation.source_id = target.claim_id
                 AND representation.relation_type = 'represented_by'
                 AND representation.target_type = 'memory_record'
                JOIN memory_records AS memory
                  ON memory.tenant_id = representation.tenant_id
                 AND memory.memory_id = representation.target_id
                WHERE target.tenant_id = %s AND target.claim_id = %s
                """,
                (source_claim_id, tenant_id, target_claim_id),
            )
        ).fetchone()
    return cast(tuple[str | None, datetime | None, datetime | None], row)


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


async def _claim_provenance(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[str, str, str, str, str]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT claim_type, statement, model_id, model_revision, prompt_version
                FROM claims WHERE tenant_id = %s
                """,
                (tenant_id,),
            )
        ).fetchone()
    return cast(tuple[str, str, str, str, str], row)


async def _relation_counts(
    database_url: str,
    tenant_id: TenantId,
) -> tuple[tuple[str, int], ...]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        cursor = await connection.execute(
            """
            SELECT relation_type, count(*)
            FROM relations WHERE tenant_id = %s
            GROUP BY relation_type ORDER BY relation_type
            """,
            (tenant_id,),
        )
        rows = tuple([cast(tuple[str, int], row) async for row in cursor])
    return rows
