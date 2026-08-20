"""Checks for evidence-aware hierarchical Memory consolidation."""

from datetime import datetime, timedelta, timezone

import pytest
from consolidation_doubles import DeterministicSigner, RecordingTextEmbedder

from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.summary_consolidation import (
    SummaryCandidate,
    SummaryCandidatePage,
    SummaryCandidateRequest,
    SummaryConsolidation,
    SummaryProposal,
    SummaryScope,
    SummaryWrite,
)
from mindbridge.core import (
    EmbeddedObjectType,
    EntityId,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryType,
    ModelReference,
    ObservationId,
    RelationType,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")


class RecordingSummaryStore:
    def __init__(self, candidates: tuple[SummaryCandidate, ...]) -> None:
        self.page = SummaryCandidatePage(candidates=candidates, scanned_count=4, next_cursor=None)
        self.evidence = (_evidence(1), _evidence(2))
        self.media = (_media(1), _media(2))
        self.writes: tuple[SummaryWrite, ...] = ()

    async def list_summary_candidates(
        self,
        request: SummaryCandidateRequest,
    ) -> SummaryCandidatePage:
        assert request.tenant_id == TENANT_ID
        return self.page

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        assert tenant_id == TENANT_ID
        return tuple(item for item in self.evidence if item.evidence_id in evidence_ids)

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        assert tenant_id == TENANT_ID
        return tuple(item for item in self.media if item.media_object_id in media_object_ids)

    async def commit_summary_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[SummaryWrite, ...],
    ) -> int:
        assert tenant_id == TENANT_ID
        self.writes = writes
        return len(writes)


class RecordingSummaryConsolidator:
    def __init__(self, consolidation: SummaryConsolidation) -> None:
        self._consolidation = consolidation
        self.evidence: tuple[ResolvedEvidence, ...] = ()

    async def propose_summaries(
        self,
        candidates: tuple[SummaryCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> SummaryConsolidation:
        assert candidates
        self.evidence = evidence
        return self._consolidation


async def test_summary_consolidation_builds_verified_and_attested_hierarchy() -> None:
    candidates = tuple(_candidate(index) for index in range(1, 5))
    consolidation = SummaryConsolidation(
        summaries=(
            SummaryProposal(
                source_memory_ids=(MemoryId("memory_01"), MemoryId("memory_02")),
                scope=SummaryScope.SESSION,
                summary="The person completed a tool repair.",
                salience=0.9,
            ),
            SummaryProposal(
                source_memory_ids=(MemoryId("memory_03"), MemoryId("memory_04")),
                scope=SummaryScope.TOPIC,
                summary="The user reported two repair preferences.",
                salience=0.7,
            ),
        ),
        model_reference=ModelReference(model_id="qwen3.8-max"),
        prompt_version="consolidate_summaries_v1",
    )
    store = RecordingSummaryStore(candidates)
    consolidator = RecordingSummaryConsolidator(consolidation)
    embedder = RecordingTextEmbedder()

    result = await ConsolidateSummaries(
        store,
        consolidator,
        embedder,
        media_url_signer=DeterministicSigner(NOW),
    ).run(SummaryCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW + timedelta(hours=1)))

    assert result.candidate_count == 4
    assert result.committed_count == 2
    assert tuple(item.evidence_span.evidence_id for item in consolidator.evidence) == (
        EvidenceId("evidence_01"),
        EvidenceId("evidence_02"),
    )
    assert embedder.documents == (
        "The person completed a tool repair.",
        "The user reported two repair preferences.",
    )
    assert [write.memory.verification_status for write in store.writes] == [
        VerificationStatus.VERIFIED,
        VerificationStatus.UNVERIFIED,
    ]
    assert all(write.memory.memory_type is MemoryType.SEMANTIC for write in store.writes)
    assert all(
        write.embedding.object_type is EmbeddedObjectType.MEMORY_RECORD for write in store.writes
    )
    assert all(
        {relation.relation_type for relation in write.relations} == {RelationType.CONTAINS}
        for write in store.writes
    )


async def test_summary_consolidation_rejects_unknown_sources_before_embedding() -> None:
    candidates = tuple(_candidate(index) for index in range(1, 3))
    store = RecordingSummaryStore(candidates)
    embedder = RecordingTextEmbedder()
    consolidation = SummaryConsolidation(
        summaries=(
            SummaryProposal(
                source_memory_ids=(MemoryId("memory_01"), MemoryId("memory_unknown")),
                scope=SummaryScope.DAY,
                summary="Unsupported",
                salience=0.5,
            ),
        ),
        model_reference=ModelReference(model_id="omni"),
        prompt_version="consolidate_summaries_v1",
    )

    with pytest.raises(MemoryIntegrityError, match="unknown candidate"):
        await ConsolidateSummaries(
            store,
            RecordingSummaryConsolidator(consolidation),
            embedder,
            media_url_signer=DeterministicSigner(NOW),
        ).run(SummaryCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW))

    assert embedder.documents == ()
    assert store.writes == ()


def _candidate(index: int) -> SummaryCandidate:
    suffix = f"{index:02d}"
    verified = index <= 2
    return SummaryCandidate(
        memory=MemoryRecord(
            memory_id=MemoryId(f"memory_{suffix}"),
            tenant_id=TENANT_ID,
            memory_type=MemoryType.EPISODIC if verified else MemoryType.SEMANTIC,
            summary=f"Memory {suffix}",
            evidence_ids=(EvidenceId(f"evidence_{suffix}"),) if verified else (),
            occurred_at=NOW + timedelta(minutes=index),
            ended_at=NOW + timedelta(minutes=index, seconds=30),
            created_at=NOW,
            verification_status=(
                VerificationStatus.VERIFIED if verified else VerificationStatus.ATTESTED
            ),
            model_reference=ModelReference(model_id="omni"),
            salience=0.8,
            strength=0.8,
        ),
        entity_ids=(EntityId("red_tool"),),
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
