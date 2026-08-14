"""Evidence-verified semantic Claim consolidation use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mindbridge.application.capabilities import Embedder
from mindbridge.application.claim_consolidation import (
    ClaimCandidate,
    ClaimCandidatePage,
    ClaimCandidateRequest,
)
from mindbridge.application.evidence import EvidenceReader, read_resolved_evidence
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import MediaUrlSigner
from mindbridge.application.semantic_claims import (
    ClaimConsolidation,
    ClaimConsolidationCommit,
    ClaimConsolidationWrite,
    derive_claim_consolidation_write,
)
from mindbridge.core import ClaimId, MemoryIntegrityError, TenantId
from mindbridge.telemetry import set_current_span_attributes, trace_operation


@dataclass(frozen=True, slots=True)
class ClaimConsolidationResult:
    """Content-free progress and outcome for one bounded Claim page."""

    scanned_count: int
    candidate_count: int
    proposed_semantic_claim_count: int
    proposed_relationship_count: int
    committed_semantic_claim_count: int
    committed_relationship_count: int
    next_cursor: ClaimId | None


class ClaimConsolidator(Protocol):
    """Frozen model boundary that verifies semantic relationships against evidence."""

    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation: ...


class ClaimConsolidationStore(EvidenceReader, Protocol):
    """Narrow transactional boundary needed by the semantic Claim use case."""

    async def list_claim_candidates(
        self,
        request: ClaimCandidateRequest,
    ) -> ClaimCandidatePage: ...

    async def commit_claim_consolidation(
        self,
        tenant_id: TenantId,
        write: ClaimConsolidationWrite,
    ) -> ClaimConsolidationCommit: ...


class ConsolidateClaims:
    """Verify and persist one bounded Claim page without changing model weights."""

    def __init__(
        self,
        store: ClaimConsolidationStore,
        consolidator: ClaimConsolidator,
        text_embedder: Embedder,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._consolidator = consolidator
        self._text_embedder = text_embedder
        self._media_url_signer = media_url_signer

    @trace_operation("mindbridge.consolidation.claims")
    async def run(self, request: ClaimCandidateRequest) -> ClaimConsolidationResult:
        """Discover, inspect, and atomically commit one stable Claim page."""
        page = await self._store.list_claim_candidates(request)
        if any(candidate.claim.tenant_id != request.tenant_id for candidate in page.candidates):
            raise MemoryIntegrityError("Claim candidates crossed the requested tenant")
        if len(page.candidates) < 2:
            return _result(page)
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.consolidation.claim_candidate_count": len(page.candidates),
            }
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for candidate in page.candidates
                for evidence_id in candidate.claim.evidence_ids
            )
        )
        evidence = await read_resolved_evidence(
            self._store,
            self._media_url_signer,
            request.tenant_id,
            evidence_ids,
        )
        consolidation = await self._consolidator.propose_claims(page.candidates, evidence)
        _require_candidate_proposals(page.candidates, consolidation)
        write = await derive_claim_consolidation_write(
            request.tenant_id,
            page.candidates,
            consolidation,
            self._text_embedder,
            request.evaluated_at,
        )
        commit = (
            await self._store.commit_claim_consolidation(request.tenant_id, write)
            if write.semantic_claims or write.relationships
            else ClaimConsolidationCommit(semantic_claim_count=0, relationship_count=0)
        )
        set_current_span_attributes(
            {
                "mindbridge.consolidation.semantic_claim_count": commit.semantic_claim_count,
                "mindbridge.consolidation.claim_relationship_count": commit.relationship_count,
            }
        )
        return _result(page, consolidation=consolidation, commit=commit)


def _require_candidate_proposals(
    candidates: tuple[ClaimCandidate, ...],
    consolidation: ClaimConsolidation,
) -> None:
    candidate_by_id = {candidate.claim.claim_id: candidate.claim for candidate in candidates}
    referenced_ids = {
        claim_id
        for proposal in consolidation.semantic_claims
        for claim_id in proposal.source_claim_ids
    } | {
        claim_id
        for relationship in consolidation.relationships
        for claim_id in (relationship.source_claim_id, relationship.target_claim_id)
    }
    if not referenced_ids <= set(candidate_by_id):
        raise MemoryIntegrityError("Claim consolidator returned an unknown candidate")
    if any(
        len({candidate_by_id[claim_id].claim_type for claim_id in proposal.source_claim_ids}) != 1
        for proposal in consolidation.semantic_claims
    ):
        raise MemoryIntegrityError("semantic Claim sources have incompatible types")
    if any(
        (
            candidate_by_id[relationship.source_claim_id].valid_from,
            relationship.source_claim_id,
        )
        <= (
            candidate_by_id[relationship.target_claim_id].valid_from,
            relationship.target_claim_id,
        )
        for relationship in consolidation.relationships
    ):
        raise MemoryIntegrityError("Claim relationship source is not the later candidate")


def _result(
    page: ClaimCandidatePage,
    *,
    consolidation: ClaimConsolidation | None = None,
    commit: ClaimConsolidationCommit | None = None,
) -> ClaimConsolidationResult:
    return ClaimConsolidationResult(
        scanned_count=page.scanned_count,
        candidate_count=len(page.candidates),
        proposed_semantic_claim_count=(
            len(consolidation.semantic_claims) if consolidation is not None else 0
        ),
        proposed_relationship_count=(
            len(consolidation.relationships) if consolidation is not None else 0
        ),
        committed_semantic_claim_count=(commit.semantic_claim_count if commit is not None else 0),
        committed_relationship_count=(commit.relationship_count if commit is not None else 0),
        next_cursor=page.next_cursor,
    )
