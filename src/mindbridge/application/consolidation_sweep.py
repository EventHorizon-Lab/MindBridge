"""Stable tenant sweeps across Episode, Claim, Summary, and entity consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mindbridge.application.claim_consolidation import ClaimCandidateRequest
from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.consolidate_entities import ConsolidateEntities
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import ConsolidateEpisodes, EpisodeCandidateRequest
from mindbridge.application.entity_resolution import EntityCandidateRequest
from mindbridge.application.summary_consolidation import (
    SummaryCandidateCursor,
    SummaryCandidateRequest,
)
from mindbridge.core import (
    ClaimId,
    EntityId,
    EntityType,
    EventId,
    MemoryIntegrityError,
    TenantId,
)


@dataclass(frozen=True, slots=True)
class EpisodeSweepSummary:
    """Content-free operational totals for one complete tenant Episode sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_count: int
    proposed_count: int
    committed_count: int


@dataclass(frozen=True, slots=True)
class ClaimSweepSummary:
    """Content-free operational totals for one complete semantic Claim sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_count: int
    proposed_semantic_claim_count: int
    proposed_relationship_count: int
    committed_semantic_claim_count: int
    committed_relationship_count: int


@dataclass(frozen=True, slots=True)
class SummarySweepSummary:
    """Content-free operational totals for one complete Memory Summary sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_count: int
    proposed_count: int
    committed_count: int


@dataclass(frozen=True, slots=True)
class EntitySweepSummary:
    """Content-free operational totals for one complete entity resolution sweep.

    `skipped_pair_count` and `dropped_pair_count` are the two ways this sweep declines to
    answer, and they mean different things: skipped pairs were reached and left unjudged
    because nothing could be inspected or the judge was unsure, while dropped pairs were never
    looked at because the page hit its budget.
    """

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_pair_count: int
    dropped_pair_count: int
    same_as_count: int
    not_same_as_count: int
    skipped_pair_count: int
    committed_count: int


@dataclass(frozen=True, slots=True)
class ConsolidationSweepSummary:
    """All consolidation outcomes from one scheduled tenant run."""

    episodes: EpisodeSweepSummary
    claims: ClaimSweepSummary
    summaries: SummarySweepSummary
    entities: EntitySweepSummary


async def consolidate_tenant_episodes(
    use_case: ConsolidateEpisodes,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> EpisodeSweepSummary:
    """Consolidate stable Episode pages at one fixed evaluation instant."""
    cursor: EventId | None = None
    page_count = scanned_count = candidate_count = proposed_count = committed_count = 0
    while True:
        result = await use_case.run(
            EpisodeCandidateRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_event_id=cursor,
                limit=page_size,
                maximum_gap_seconds=maximum_gap_seconds,
                minimum_similarity=minimum_similarity,
            )
        )
        page_count += 1
        scanned_count += result.scanned_count
        candidate_count += result.candidate_count
        proposed_count += result.proposed_count
        committed_count += result.committed_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("Episode consolidation cursor did not advance")
        cursor = result.next_cursor
    return EpisodeSweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_count=candidate_count,
        proposed_count=proposed_count,
        committed_count=committed_count,
    )


async def consolidate_tenant_claims(
    use_case: ConsolidateClaims,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> ClaimSweepSummary:
    """Consolidate stable Claim pages at one fixed evaluation instant."""
    cursor: ClaimId | None = None
    page_count = scanned_count = candidate_count = 0
    proposed_semantic_count = proposed_relationship_count = 0
    committed_semantic_count = committed_relationship_count = 0
    while True:
        result = await use_case.run(
            ClaimCandidateRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_claim_id=cursor,
                limit=page_size,
                maximum_gap_seconds=maximum_gap_seconds,
                minimum_similarity=minimum_similarity,
            )
        )
        page_count += 1
        scanned_count += result.scanned_count
        candidate_count += result.candidate_count
        proposed_semantic_count += result.proposed_semantic_claim_count
        proposed_relationship_count += result.proposed_relationship_count
        committed_semantic_count += result.committed_semantic_claim_count
        committed_relationship_count += result.committed_relationship_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("Claim consolidation cursor did not advance")
        cursor = result.next_cursor
    return ClaimSweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_count=candidate_count,
        proposed_semantic_claim_count=proposed_semantic_count,
        proposed_relationship_count=proposed_relationship_count,
        committed_semantic_claim_count=committed_semantic_count,
        committed_relationship_count=committed_relationship_count,
    )


async def consolidate_tenant_summaries(
    use_case: ConsolidateSummaries,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> SummarySweepSummary:
    """Consolidate stable Memory pages at one fixed evaluation instant."""
    cursor: SummaryCandidateCursor | None = None
    page_count = scanned_count = candidate_count = proposed_count = committed_count = 0
    while True:
        result = await use_case.run(
            SummaryCandidateRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_cursor=cursor,
                limit=page_size,
                maximum_gap_seconds=maximum_gap_seconds,
                minimum_similarity=minimum_similarity,
            )
        )
        page_count += 1
        scanned_count += result.scanned_count
        candidate_count += result.candidate_count
        proposed_count += result.proposed_count
        committed_count += result.committed_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("Summary consolidation cursor did not advance")
        cursor = result.next_cursor
    return SummarySweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_count=candidate_count,
        proposed_count=proposed_count,
        committed_count=committed_count,
    )


async def consolidate_tenant_entities(
    use_case: ConsolidateEntities,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    candidate_limit: int,
    minimum_confidence: float,
    evidence_per_side: int,
    maximum_pairs: int,
    entity_types: tuple[EntityType, ...],
    readjudicate: bool,
) -> EntitySweepSummary:
    """Judge stable entity pages at one fixed evaluation instant."""
    cursor: EntityId | None = None
    page_count = scanned_count = candidate_pair_count = dropped_pair_count = 0
    same_as_count = not_same_as_count = skipped_pair_count = committed_count = 0
    while True:
        result = await use_case.run(
            EntityCandidateRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_entity_id=cursor,
                limit=page_size,
                maximum_gap_seconds=maximum_gap_seconds,
                candidate_limit=candidate_limit,
                minimum_confidence=minimum_confidence,
                evidence_per_side=evidence_per_side,
                maximum_pairs=maximum_pairs,
                entity_types=entity_types,
                readjudicate=readjudicate,
            )
        )
        page_count += 1
        scanned_count += result.scanned_count
        candidate_pair_count += result.candidate_pair_count
        dropped_pair_count += result.dropped_pair_count
        same_as_count += result.same_as_count
        not_same_as_count += result.not_same_as_count
        skipped_pair_count += result.skipped_pair_count
        committed_count += result.committed_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("entity consolidation cursor did not advance")
        cursor = result.next_cursor
    return EntitySweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_pair_count=candidate_pair_count,
        dropped_pair_count=dropped_pair_count,
        same_as_count=same_as_count,
        not_same_as_count=not_same_as_count,
        skipped_pair_count=skipped_pair_count,
        committed_count=committed_count,
    )
