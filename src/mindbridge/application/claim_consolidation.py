"""Bounded candidates for evidence-verified semantic Claim consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mindbridge.core import (
    Claim,
    ClaimId,
    DomainInvariantError,
    EntityId,
    TenantId,
    require_aware_datetime,
    require_non_empty,
    require_similarity,
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
        require_non_empty(self.tenant_id, "tenant_id")
        require_aware_datetime(self.evaluated_at, "evaluated_at")
        if self.after_claim_id is not None:
            require_non_empty(self.after_claim_id, "after_claim_id")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("Claim candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 31_536_000:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 31536000")
        require_similarity(self.minimum_similarity, "minimum_similarity")


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
        if self.next_cursor is not None:
            require_non_empty(self.next_cursor, "Claim candidate cursor")
        if len({claim.claim_id for claim in claims}) != len(claims):
            raise DomainInvariantError("Claim candidate IDs must be unique")
        if len({claim.tenant_id for claim in claims}) > 1:
            raise DomainInvariantError("Claim candidates must belong to one tenant")
