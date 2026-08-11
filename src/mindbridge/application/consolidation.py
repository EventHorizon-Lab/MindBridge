"""Bounded candidate discovery for evidence-verified episode consolidation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.core import (
    DomainInvariantError,
    Event,
    EventHierarchyLevel,
    EventId,
    EventStatus,
    ModelReference,
    TenantId,
)


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
class EpisodeProposal:
    """One Omni-verified group of existing events that form a coherent episode."""

    event_ids: tuple[EventId, ...]
    description: str
    salience: float

    def __post_init__(self) -> None:
        if not 2 <= len(self.event_ids) <= 32 or len(set(self.event_ids)) != len(self.event_ids):
            raise DomainInvariantError("episode proposal requires 2 to 32 unique event IDs")
        if not self.description.strip():
            raise DomainInvariantError("episode proposal description must not be empty")
        if not math.isfinite(self.salience) or not 0.0 <= self.salience <= 1.0:
            raise DomainInvariantError("episode proposal salience must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EpisodeConsolidation:
    """Validated episode proposals and the frozen model provenance that produced them."""

    episodes: tuple[EpisodeProposal, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("episode consolidation prompt version must not be empty")
        event_ids = [event_id for episode in self.episodes for event_id in episode.event_ids]
        if len(set(event_ids)) != len(event_ids):
            raise DomainInvariantError("one event cannot belong to multiple episode proposals")


class EpisodeCandidateStore(Protocol):
    """Persistence boundary for one stable consolidation candidate page."""

    async def list_episode_candidates(
        self,
        request: EpisodeCandidateRequest,
    ) -> EpisodeCandidatePage: ...


class EpisodeConsolidator(Protocol):
    """Frozen Omni boundary that verifies candidate events against original evidence."""

    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation: ...
