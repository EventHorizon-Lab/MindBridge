"""Bounded candidates for hierarchical memory summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from mindbridge.core import (
    DomainInvariantError,
    EntityId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    TenantId,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class SummaryCandidateRequest:
    """Stable page and calibrated affinity bounds for one tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_memory_id: MemoryId | None = None
    limit: int = 16
    maximum_gap_seconds: int = 2_592_000
    minimum_similarity: float = 0.8

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise DomainInvariantError("tenant_id must not be empty")
        if self.evaluated_at.utcoffset() is None:
            raise DomainInvariantError("evaluated_at must be timezone-aware")
        if self.after_memory_id is not None and not self.after_memory_id.strip():
            raise DomainInvariantError("after_memory_id must not be empty")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("Summary candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 31_536_000:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 31536000")
        if not math.isfinite(self.minimum_similarity) or not -1.0 <= self.minimum_similarity <= 1.0:
            raise DomainInvariantError("minimum_similarity must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class SummaryCandidate:
    """One current Memory and its evidence-derived entity context."""

    memory: MemoryRecord
    entity_ids: tuple[EntityId, ...]

    def __post_init__(self) -> None:
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise DomainInvariantError("Summary candidate entity IDs must be unique")
        if (
            self.memory.memory_type not in {MemoryType.EPISODIC, MemoryType.SEMANTIC}
            or self.memory.verification_status
            not in {VerificationStatus.VERIFIED, VerificationStatus.ATTESTED}
            or self.memory.superseded_at is not None
        ):
            raise DomainInvariantError("Summary candidates must be current grounded memories")


@dataclass(frozen=True, slots=True)
class SummaryCandidatePage:
    """Related Memories plus cursor progress across all examined seeds."""

    candidates: tuple[SummaryCandidate, ...]
    scanned_count: int
    next_cursor: MemoryId | None

    def __post_init__(self) -> None:
        memories = tuple(candidate.memory for candidate in self.candidates)
        if self.scanned_count < 0:
            raise DomainInvariantError("Summary candidate scanned_count must be non-negative")
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise DomainInvariantError("Summary candidate cursor must not be empty")
        if len({memory.memory_id for memory in memories}) != len(memories):
            raise DomainInvariantError("Summary candidate Memory IDs must be unique")
        if len({memory.tenant_id for memory in memories}) > 1:
            raise DomainInvariantError("Summary candidates must belong to one tenant")
