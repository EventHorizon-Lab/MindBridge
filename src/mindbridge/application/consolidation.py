"""Bounded candidate discovery for evidence-verified episode consolidation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from mindbridge.application.capabilities import Embedder
from mindbridge.application.episodes import (
    EpisodeConsolidation,
    EpisodeProposal,
    EpisodeWrite,
    derive_episode_writes,
)
from mindbridge.application.evidence import EvidenceReader, read_resolved_evidence
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import MediaUrlSigner
from mindbridge.core import (
    DomainInvariantError,
    Event,
    EventHierarchyLevel,
    EventId,
    EventStatus,
    MemoryIntegrityError,
    TenantId,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


@dataclass(frozen=True, slots=True)
class EpisodeCandidateRequest:
    """Stable page and calibrated similarity bounds for one tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_event_id: EventId | None = None
    limit: int = 16
    maximum_gap_seconds: int = 900
    minimum_similarity: float = 0.7

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise DomainInvariantError("tenant_id must not be empty")
        if self.evaluated_at.utcoffset() is None:
            raise DomainInvariantError("evaluated_at must be timezone-aware")
        if self.after_event_id is not None and not self.after_event_id.strip():
            raise DomainInvariantError("after_event_id must not be empty")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("episode candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 86_400:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 86400")
        if not math.isfinite(self.minimum_similarity) or not -1.0 <= self.minimum_similarity <= 1.0:
            raise DomainInvariantError("minimum_similarity must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class EpisodeCandidatePage:
    """Related base events plus cursor progress across all examined seeds."""

    events: tuple[Event, ...]
    scanned_count: int
    next_cursor: EventId | None

    def __post_init__(self) -> None:
        if self.scanned_count < 0:
            raise DomainInvariantError("episode candidate scanned_count must be non-negative")
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise DomainInvariantError("episode candidate cursor must not be empty")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise DomainInvariantError("episode candidate event IDs must be unique")
        if len({event.tenant_id for event in self.events}) > 1:
            raise DomainInvariantError("episode candidates must belong to one tenant")
        if any(
            event.hierarchy_level is not EventHierarchyLevel.EVENT
            or event.status is not EventStatus.ACTIVE
            or event.parent_event_id is not None
            for event in self.events
        ):
            raise DomainInvariantError("episode candidates must be active unparented base events")


@dataclass(frozen=True, slots=True)
class EpisodeConsolidationResult:
    """Content-free progress and outcome for one bounded candidate page."""

    scanned_count: int
    candidate_count: int
    proposed_count: int
    committed_count: int
    next_cursor: EventId | None


class EpisodeConsolidator(Protocol):
    """Frozen model boundary that verifies candidate events against original evidence."""

    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation: ...


class EpisodeConsolidationStore(EvidenceReader, Protocol):
    """Narrow transactional boundary needed by the Episode use case."""

    async def list_episode_candidates(
        self,
        request: EpisodeCandidateRequest,
    ) -> EpisodeCandidatePage: ...

    async def commit_episode_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[EpisodeWrite, ...],
    ) -> int: ...


class ConsolidateEpisodes:
    """Verify and persist one bounded page without training or hidden state."""

    def __init__(
        self,
        store: EpisodeConsolidationStore,
        consolidator: EpisodeConsolidator,
        text_embedder: Embedder,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._consolidator = consolidator
        self._text_embedder = text_embedder
        self._media_url_signer = media_url_signer

    @trace_operation("mindbridge.consolidation.episodes")
    async def run(self, request: EpisodeCandidateRequest) -> EpisodeConsolidationResult:
        """Discover, inspect, and atomically commit one stable candidate page."""
        page = await self._store.list_episode_candidates(request)
        if any(event.tenant_id != request.tenant_id for event in page.events):
            raise MemoryIntegrityError("episode candidates crossed the requested tenant")
        if len(page.events) < 2:
            return _result(page)
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.consolidation.candidate_count": len(page.events),
            }
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id for event in page.events for evidence_id in event.evidence_ids
            )
        )
        evidence = await read_resolved_evidence(
            self._store,
            self._media_url_signer,
            request.tenant_id,
            evidence_ids,
        )
        consolidation = await self._consolidator.propose_episodes(page.events, evidence)
        _require_candidate_proposals(page.events, consolidation.episodes)
        writes = await derive_episode_writes(
            request.tenant_id,
            page.events,
            consolidation,
            self._text_embedder,
            request.evaluated_at,
        )
        committed_count = (
            await self._store.commit_episode_consolidation(request.tenant_id, writes)
            if writes
            else 0
        )
        set_current_span_attributes(
            {
                "mindbridge.consolidation.proposed_count": len(writes),
                "mindbridge.consolidation.committed_count": committed_count,
            }
        )
        return _result(
            page,
            proposed_count=len(writes),
            committed_count=committed_count,
        )


def _require_candidate_proposals(
    candidates: tuple[Event, ...],
    proposals: tuple[EpisodeProposal, ...],
) -> None:
    candidate_ids = {event.event_id for event in candidates}
    if any(
        event_id not in candidate_ids for proposal in proposals for event_id in proposal.event_ids
    ):
        raise MemoryIntegrityError("episode consolidator returned an unknown candidate event")


def _result(
    page: EpisodeCandidatePage,
    *,
    proposed_count: int = 0,
    committed_count: int = 0,
) -> EpisodeConsolidationResult:
    return EpisodeConsolidationResult(
        scanned_count=page.scanned_count,
        candidate_count=len(page.events),
        proposed_count=proposed_count,
        committed_count=committed_count,
        next_cursor=page.next_cursor,
    )
