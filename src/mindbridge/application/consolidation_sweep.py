"""Stable tenant sweeps across Episode, Claim, and Summary consolidation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Generic, TypeVar

from mindbridge.application.claim_consolidation import ClaimCandidateRequest
from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import ConsolidateEpisodes, EpisodeCandidateRequest
from mindbridge.application.summary_consolidation import (
    SummaryCandidateCursor,
    SummaryCandidateRequest,
)
from mindbridge.core import ClaimId, EventId, MemoryIntegrityError, TenantId

# Each sweep pages with the cursor its own request field accepts; keeping that type on the
# shared loop is what stops a copied sweep from feeding a Claim cursor to an Episode page.
_Cursor = TypeVar("_Cursor")


@dataclass(frozen=True, slots=True)
class SweepSummary:
    """Content-free operational totals for one complete tenant sweep.

    `counts` carries the proposed/committed totals under the names the process prints; they
    differ per memory kind and exist only to be reported, never to be branched on.
    """

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_count: int
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ConsolidationSweepSummary:
    """All consolidation outcomes from one scheduled tenant run."""

    episodes: SweepSummary
    claims: SweepSummary
    summaries: SweepSummary


@dataclass(frozen=True, slots=True)
class _Page(Generic[_Cursor]):
    """One consolidation result reduced to what a sweep accumulates."""

    scanned_count: int
    candidate_count: int
    counts: Mapping[str, int]
    next_cursor: _Cursor | None


async def consolidate_tenant_episodes(
    use_case: ConsolidateEpisodes,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> SweepSummary:
    """Consolidate stable Episode pages at one fixed evaluation instant."""

    async def run_page(cursor: EventId | None) -> _Page[EventId]:
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
        return _Page(
            result.scanned_count,
            result.candidate_count,
            {
                "proposed_count": result.proposed_count,
                "committed_count": result.committed_count,
            },
            result.next_cursor,
        )

    return await _sweep("Episode", tenant_id, evaluated_at, run_page)


async def consolidate_tenant_claims(
    use_case: ConsolidateClaims,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> SweepSummary:
    """Consolidate stable Claim pages at one fixed evaluation instant."""

    async def run_page(cursor: ClaimId | None) -> _Page[ClaimId]:
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
        return _Page(
            result.scanned_count,
            result.candidate_count,
            {
                "proposed_semantic_claim_count": result.proposed_semantic_claim_count,
                "proposed_relationship_count": result.proposed_relationship_count,
                "committed_semantic_claim_count": result.committed_semantic_claim_count,
                "committed_relationship_count": result.committed_relationship_count,
            },
            result.next_cursor,
        )

    return await _sweep("Claim", tenant_id, evaluated_at, run_page)


async def consolidate_tenant_summaries(
    use_case: ConsolidateSummaries,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> SweepSummary:
    """Consolidate stable Memory pages at one fixed evaluation instant."""

    async def run_page(cursor: SummaryCandidateCursor | None) -> _Page[SummaryCandidateCursor]:
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
        return _Page(
            result.scanned_count,
            result.candidate_count,
            {
                "proposed_count": result.proposed_count,
                "committed_count": result.committed_count,
            },
            result.next_cursor,
        )

    return await _sweep("Summary", tenant_id, evaluated_at, run_page)


async def _sweep(
    label: str,
    tenant_id: TenantId,
    evaluated_at: datetime,
    run_page: Callable[[_Cursor | None], Awaitable[_Page[_Cursor]]],
) -> SweepSummary:
    """Page until the cursor stops, refusing a cursor that cannot make progress."""
    cursor: _Cursor | None = None
    page_count = scanned_count = candidate_count = 0
    counts: dict[str, int] = {}
    while True:
        page = await run_page(cursor)
        page_count += 1
        scanned_count += page.scanned_count
        candidate_count += page.candidate_count
        for name, value in page.counts.items():
            counts[name] = counts.get(name, 0) + value
        if page.next_cursor is None:
            break
        if page.next_cursor == cursor:
            raise MemoryIntegrityError(f"{label} consolidation cursor did not advance")
        cursor = page.next_cursor
    return SweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_count=candidate_count,
        counts=MappingProxyType(counts),
    )
