"""Bounded candidates for evidence-verified semantic Claim consolidation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.core import (
    Claim,
    ClaimId,
    DomainInvariantError,
    EntityId,
    ModelReference,
    RelationType,
    TenantId,
)


@dataclass(frozen=True, slots=True)
class ClaimCandidateRequest:
    """Stable page and calibrated affinity bounds for one tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_claim_id: ClaimId | None = None
    limit: int = 16
    maximum_gap_seconds: int = 2_592_000
    minimum_similarity: float = 0.8

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise DomainInvariantError("tenant_id must not be empty")
        if self.evaluated_at.utcoffset() is None:
            raise DomainInvariantError("evaluated_at must be timezone-aware")
        if self.after_claim_id is not None and not self.after_claim_id.strip():
            raise DomainInvariantError("after_claim_id must not be empty")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("Claim candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 31_536_000:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 31536000")
        if not math.isfinite(self.minimum_similarity) or not -1.0 <= self.minimum_similarity <= 1.0:
            raise DomainInvariantError("minimum_similarity must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    """One current Claim and its explicit entity graph context."""

    claim: Claim
    entity_ids: tuple[EntityId, ...]

    def __post_init__(self) -> None:
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise DomainInvariantError("Claim candidate entity IDs must be unique")


@dataclass(frozen=True, slots=True)
class ClaimCandidatePage:
    """Related Claims plus cursor progress across all examined seeds."""

    candidates: tuple[ClaimCandidate, ...]
    scanned_count: int
    next_cursor: ClaimId | None

    def __post_init__(self) -> None:
        claims = tuple(candidate.claim for candidate in self.candidates)
        if self.scanned_count < 0:
            raise DomainInvariantError("Claim candidate scanned_count must be non-negative")
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise DomainInvariantError("Claim candidate cursor must not be empty")
        if len({claim.claim_id for claim in claims}) != len(claims):
            raise DomainInvariantError("Claim candidate IDs must be unique")
        if len({claim.tenant_id for claim in claims}) > 1:
            raise DomainInvariantError("Claim candidates must belong to one tenant")


class ClaimCandidateStore(Protocol):
    """Persistence boundary for one stable semantic consolidation page."""

    async def list_claim_candidates(
        self,
        request: ClaimCandidateRequest,
    ) -> ClaimCandidatePage: ...


@dataclass(frozen=True, slots=True)
class SemanticClaimProposal:
    """One stronger Claim grounded in two or more mutually supporting Claims."""

    source_claim_ids: tuple[ClaimId, ...]
    statement: str
    confidence: float

    def __post_init__(self) -> None:
        if not 2 <= len(self.source_claim_ids) <= 32 or len(set(self.source_claim_ids)) != len(
            self.source_claim_ids
        ):
            raise DomainInvariantError("semantic Claim requires 2 to 32 unique source Claims")
        if not self.statement.strip():
            raise DomainInvariantError("semantic Claim statement must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("semantic Claim confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ClaimRelationshipProposal:
    """One evidence-verified contradiction or temporal replacement between Claims."""

    source_claim_id: ClaimId
    relation_type: RelationType
    target_claim_id: ClaimId

    def __post_init__(self) -> None:
        if self.source_claim_id == self.target_claim_id:
            raise DomainInvariantError("Claim relationship cannot point to itself")
        if self.relation_type not in {RelationType.CONTRADICTS, RelationType.SUPERSEDES}:
            raise DomainInvariantError("Claim relationship must contradict or supersede")


@dataclass(frozen=True, slots=True)
class ClaimConsolidation:
    """Validated Claim proposals and frozen Omni provenance."""

    semantic_claims: tuple[SemanticClaimProposal, ...]
    relationships: tuple[ClaimRelationshipProposal, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("Claim consolidation prompt version must not be empty")
        support_ids = [
            claim_id for proposal in self.semantic_claims for claim_id in proposal.source_claim_ids
        ]
        if len(set(support_ids)) != len(support_ids):
            raise DomainInvariantError("one Claim cannot support multiple semantic proposals")
        relationship_pairs = [
            frozenset((proposal.source_claim_id, proposal.target_claim_id))
            for proposal in self.relationships
        ]
        if len(set(relationship_pairs)) != len(relationship_pairs):
            raise DomainInvariantError("a Claim pair can have only one semantic relationship")
        support_groups = [set(proposal.source_claim_ids) for proposal in self.semantic_claims]
        if any(
            {relationship.source_claim_id, relationship.target_claim_id} <= group
            for relationship in self.relationships
            for group in support_groups
        ):
            raise DomainInvariantError("supporting Claims cannot also contradict each other")


class ClaimConsolidator(Protocol):
    """Frozen Omni boundary that verifies semantic relationships against evidence."""

    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation: ...
