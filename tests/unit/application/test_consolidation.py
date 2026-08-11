"""Vertical checks for evidence-verified Episode consolidation."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import (
    ConsolidateEpisodes,
    EpisodeCandidatePage,
    EpisodeCandidateRequest,
    EpisodeConsolidation,
    EpisodeProposal,
    EpisodeWrite,
    PresignedMediaDownload,
    ResolvedEvidence,
)
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    Event,
    EventHierarchyLevel,
    EventId,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    MemoryType,
    ModelReference,
    ObservationId,
    RelationType,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")


class RecordingEpisodeStore:
    def __init__(self, events: tuple[Event, ...]) -> None:
        self.page = EpisodeCandidatePage(events=events, scanned_count=2, next_cursor=None)
        self.evidence = tuple(_evidence(index) for index in (1, 2))
        self.media = tuple(_media(index) for index in (1, 2))
        self.writes: tuple[EpisodeWrite, ...] = ()

    async def list_episode_candidates(
        self,
        request: EpisodeCandidateRequest,
    ) -> EpisodeCandidatePage:
        assert request.tenant_id == TENANT_ID
        return self.page

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        assert tenant_id == TENANT_ID
        return tuple(
            item
            for evidence_id in evidence_ids
            for item in self.evidence
            if item.evidence_id == evidence_id
        )

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        assert tenant_id == TENANT_ID
        return tuple(
            item
            for media_id in media_object_ids
            for item in self.media
            if item.media_object_id == media_id
        )

    async def commit_episode_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[EpisodeWrite, ...],
    ) -> int:
        assert tenant_id == TENANT_ID
        self.writes = writes
        return len(writes)


class RecordingConsolidator:
    def __init__(self, episode_event_ids: tuple[EventId, ...]) -> None:
        self._episode_event_ids = episode_event_ids
        self.evidence: tuple[ResolvedEvidence, ...] = ()

    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation:
        assert len(events) == 2
        self.evidence = evidence
        return EpisodeConsolidation(
            episodes=(
                EpisodeProposal(
                    event_ids=self._episode_event_ids,
                    description="A person retrieves a tool and explains the repair.",
                    salience=0.9,
                ),
            ),
            model_reference=ModelReference(model_id="qwen3.8-max", revision="omni-revision"),
            prompt_version="consolidate_episodes_v1",
        )


class RecordingTextEmbedder:
    model_reference = ModelReference(model_id="jina-text", revision="text-revision")
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
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )


async def test_consolidation_builds_one_complete_episode_aggregate() -> None:
    events = (_event(1), _event(2))
    store = RecordingEpisodeStore(events)
    consolidator = RecordingConsolidator(tuple(event.event_id for event in events))
    text_embedder = RecordingTextEmbedder()

    result = await ConsolidateEpisodes(
        store,
        consolidator,
        text_embedder,
        media_url_signer=DeterministicSigner(),
    ).run(
        EpisodeCandidateRequest(
            tenant_id=TENANT_ID,
            evaluated_at=NOW + timedelta(hours=1),
        )
    )

    assert (result.scanned_count, result.candidate_count, result.proposed_count) == (2, 2, 1)
    assert result.committed_count == 1
    assert tuple(item.evidence_span.evidence_id for item in consolidator.evidence) == (
        EvidenceId("evidence_01"),
        EvidenceId("evidence_02"),
    )
    assert text_embedder.documents == ("A person retrieves a tool and explains the repair.",)
    write = store.writes[0]
    assert write.episode.hierarchy_level is EventHierarchyLevel.EPISODE
    assert write.episode.observation_ids == (
        ObservationId("observation_01"),
        ObservationId("observation_02"),
    )
    assert write.episode.evidence_ids == (EvidenceId("evidence_01"), EvidenceId("evidence_02"))
    assert write.episode.occurred_at == events[0].occurred_at
    assert write.episode.ended_at == events[1].ended_at
    assert write.memory.memory_type is MemoryType.EPISODIC
    assert write.memory.verification_status is VerificationStatus.VERIFIED
    assert write.embedding.object_type is EmbeddedObjectType.EVENT
    assert write.embedding.object_id == write.episode.event_id
    assert [relation.relation_type for relation in write.relations].count(
        RelationType.REPRESENTED_BY
    ) == 1
    assert [relation.relation_type for relation in write.relations].count(
        RelationType.CONTAINS
    ) == 2
    assert [relation.relation_type for relation in write.relations].count(
        RelationType.SAME_EPISODE
    ) == 2
    assert write.temporal_event_pairs == ((events[0].event_id, events[1].event_id),)
    assert [relation.relation_type for relation in write.relations].count(RelationType.BEFORE) == 1
    assert [relation.relation_type for relation in write.relations].count(RelationType.AFTER) == 1


async def test_consolidation_rejects_an_unknown_candidate_before_embedding() -> None:
    events = (_event(1), _event(2))
    store = RecordingEpisodeStore(events)
    embedder = RecordingTextEmbedder()

    with pytest.raises(MemoryIntegrityError, match="unknown candidate"):
        await ConsolidateEpisodes(
            store,
            RecordingConsolidator((events[0].event_id, EventId("event_unknown"))),
            embedder,
            media_url_signer=DeterministicSigner(),
        ).run(EpisodeCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW))

    assert embedder.documents == ()
    assert store.writes == ()


async def test_consolidation_does_not_order_overlapping_events() -> None:
    first = _event(1)
    second = replace(
        _event(2),
        occurred_at=first.occurred_at + timedelta(seconds=1),
        ended_at=first.ended_at + timedelta(seconds=1),
    )
    store = RecordingEpisodeStore((first, second))

    await ConsolidateEpisodes(
        store,
        RecordingConsolidator((first.event_id, second.event_id)),
        RecordingTextEmbedder(),
        media_url_signer=DeterministicSigner(),
    ).run(EpisodeCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW + timedelta(hours=1)))

    assert store.writes[0].temporal_event_pairs == ()
    assert all(
        relation.relation_type not in {RelationType.BEFORE, RelationType.AFTER}
        for relation in store.writes[0].relations
    )


def _event(index: int) -> Event:
    suffix = f"{index:02d}"
    occurred_at = NOW + timedelta(seconds=index * 5)
    return Event(
        event_id=EventId(f"event_{suffix}"),
        tenant_id=TENANT_ID,
        observation_ids=(ObservationId(f"observation_{suffix}"),),
        evidence_ids=(EvidenceId(f"evidence_{suffix}"),),
        occurred_at=occurred_at,
        ended_at=occurred_at + timedelta(seconds=4),
        description=f"Observed event {suffix}",
        salience=0.8,
        created_at=NOW,
        model_reference=ModelReference(model_id="omni", revision="perception-revision"),
        prompt_version="perceive_events_v3",
    )


def _evidence(index: int) -> EvidenceSpan:
    suffix = f"{index:02d}"
    return EvidenceSpan(
        evidence_id=EvidenceId(f"evidence_{suffix}"),
        tenant_id=TENANT_ID,
        observation_id=ObservationId(f"observation_{suffix}"),
        media_object_id=MediaObjectId(f"media_{suffix}"),
        start_ms=0,
        end_ms=4_000,
        created_at=NOW,
    )


def _media(index: int) -> MediaObject:
    suffix = f"{index:02d}"
    return MediaObject(
        media_object_id=MediaObjectId(f"media_{suffix}"),
        tenant_id=TENANT_ID,
        kind=MediaKind.VIDEO,
        uri=f"s3://memory/{TENANT_ID}/clip_{suffix}.mp4",
        sha256=f"{index:064x}",
        size_bytes=100,
        created_at=NOW,
        duration_ms=4_000,
    )
