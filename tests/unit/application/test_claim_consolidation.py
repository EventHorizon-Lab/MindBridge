"""Vertical checks for evidence-verified semantic Claim consolidation."""

from datetime import datetime, timedelta, timezone

import pytest
from consolidation_doubles import DeterministicSigner, RecordingTextEmbedder

from mindbridge.application.claim_consolidation import (
    ClaimCandidate,
    ClaimCandidatePage,
    ClaimCandidateRequest,
)
from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.semantic_claims import (
    ClaimConsolidation,
    ClaimConsolidationCommit,
    ClaimConsolidationWrite,
    ClaimRelationshipProposal,
    SemanticClaimProposal,
)
from mindbridge.core import (
    Claim,
    ClaimId,
    ClaimType,
    EmbeddedObjectType,
    EntityId,
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


class RecordingClaimStore:
    def __init__(self, candidates: tuple[ClaimCandidate, ...]) -> None:
        self.page = ClaimCandidatePage(candidates=candidates, scanned_count=4, next_cursor=None)
        self.evidence = tuple(_evidence(index) for index in range(1, 5))
        self.media = tuple(_media(index) for index in range(1, 5))
        self.write: ClaimConsolidationWrite | None = None

    async def list_claim_candidates(
        self,
        request: ClaimCandidateRequest,
    ) -> ClaimCandidatePage:
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
            for media_object_id in media_object_ids
            for item in self.media
            if item.media_object_id == media_object_id
        )

    async def commit_claim_consolidation(
        self,
        tenant_id: TenantId,
        write: ClaimConsolidationWrite,
    ) -> ClaimConsolidationCommit:
        assert tenant_id == TENANT_ID
        self.write = write
        return ClaimConsolidationCommit(
            semantic_claim_count=len(write.semantic_claims),
            relationship_count=len(write.relationships),
        )


class RecordingClaimConsolidator:
    def __init__(self, consolidation: ClaimConsolidation) -> None:
        self._consolidation = consolidation
        self.evidence: tuple[ResolvedEvidence, ...] = ()

    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation:
        assert len(candidates) == 4
        self.evidence = evidence
        return self._consolidation


async def test_claim_consolidation_builds_semantic_memory_and_version_decision() -> None:
    candidates = tuple(_candidate(index) for index in range(1, 5))
    consolidation = ClaimConsolidation(
        semantic_claims=(
            SemanticClaimProposal(
                source_claim_ids=(ClaimId("claim_01"), ClaimId("claim_02")),
                statement="The red tool is beside the blue toolbox.",
                confidence=0.94,
            ),
        ),
        relationships=(
            ClaimRelationshipProposal(
                source_claim_id=ClaimId("claim_04"),
                relation_type=RelationType.SUPERSEDES,
                target_claim_id=ClaimId("claim_03"),
            ),
        ),
        model_reference=ModelReference(model_id="qwen3.8-max", revision="claim-revision"),
        prompt_version="consolidate_claims_v1",
    )
    store = RecordingClaimStore(candidates)
    consolidator = RecordingClaimConsolidator(consolidation)
    embedder = RecordingTextEmbedder()

    result = await ConsolidateClaims(
        store,
        consolidator,
        embedder,
        media_url_signer=DeterministicSigner(NOW),
    ).run(ClaimCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW + timedelta(hours=1)))

    assert result.candidate_count == 4
    assert result.committed_semantic_claim_count == 1
    assert result.committed_relationship_count == 1
    assert tuple(item.evidence_span.evidence_id for item in consolidator.evidence) == tuple(
        EvidenceId(f"evidence_{index:02d}") for index in range(1, 5)
    )
    assert embedder.documents == ("The red tool is beside the blue toolbox.",)
    assert store.write is not None
    semantic_write = store.write.semantic_claims[0]
    assert semantic_write.claim.evidence_ids == (
        EvidenceId("evidence_01"),
        EvidenceId("evidence_02"),
    )
    assert semantic_write.entity_ids == (EntityId("blue_toolbox"), EntityId("red_tool"))
    assert semantic_write.memory.memory_type is MemoryType.SEMANTIC
    assert semantic_write.embedding.object_type is EmbeddedObjectType.CLAIM
    assert [relation.relation_type for relation in semantic_write.relations].count(
        RelationType.SUPPORTS
    ) == 2
    assert store.write.relationships[0].relation_type is RelationType.SUPERSEDES


async def test_claim_consolidation_rejects_unknown_sources_before_embedding() -> None:
    candidates = tuple(_candidate(index) for index in range(1, 5))
    store = RecordingClaimStore(candidates)
    embedder = RecordingTextEmbedder()
    consolidation = ClaimConsolidation(
        semantic_claims=(
            SemanticClaimProposal(
                source_claim_ids=(ClaimId("claim_01"), ClaimId("claim_unknown")),
                statement="Unsupported",
                confidence=0.5,
            ),
        ),
        relationships=(),
        model_reference=ModelReference(model_id="omni", revision="revision"),
        prompt_version="consolidate_claims_v1",
    )

    with pytest.raises(MemoryIntegrityError, match="unknown candidate"):
        await ConsolidateClaims(
            store,
            RecordingClaimConsolidator(consolidation),
            embedder,
            media_url_signer=DeterministicSigner(NOW),
        ).run(ClaimCandidateRequest(tenant_id=TENANT_ID, evaluated_at=NOW))

    assert embedder.documents == ()
    assert store.write is None


def _candidate(index: int) -> ClaimCandidate:
    suffix = f"{index:02d}"
    return ClaimCandidate(
        claim=Claim(
            claim_id=ClaimId(f"claim_{suffix}"),
            tenant_id=TENANT_ID,
            claim_type=ClaimType.STATE,
            statement=f"Observed state {suffix}",
            evidence_ids=(EvidenceId(f"evidence_{suffix}"),),
            confidence=0.8,
            verification_status=VerificationStatus.VERIFIED,
            valid_from=NOW + timedelta(minutes=index),
            valid_to=None,
            created_at=NOW,
            model_reference=ModelReference(model_id="omni", revision="perception-revision"),
            prompt_version="perceive_events_v3",
        ),
        entity_ids=(EntityId("red_tool"), EntityId("blue_toolbox")),
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
