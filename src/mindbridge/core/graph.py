"""Typed entity mentions and relation edges for the memory graph."""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import (
    EntityId,
    EventId,
    EvidenceId,
    MentionId,
    RelationId,
    TenantId,
)


class EntityType(str, Enum):
    """Canonical kinds admitted to the shared entity graph."""

    PERSON = "person"
    OBJECT = "object"
    PLACE = "place"
    DEVICE = "device"
    ORGANIZATION = "organization"
    TOPIC = "topic"


class RelationNodeType(str, Enum):
    """Record families that can participate in a graph edge."""

    EVENT = "event"
    ENTITY = "entity"
    CLAIM = "claim"
    MEMORY_RECORD = "memory_record"


class RelationType(str, Enum):
    """Stable relation vocabulary used by persistence and recall expansion."""

    REPRESENTED_BY = "represented_by"
    MENTIONS = "mentions"
    ASSERTS = "asserts"
    ABOUT = "about"
    CONTAINS = "contains"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    SAME_EPISODE = "same_episode"
    BEFORE = "before"
    AFTER = "after"


@dataclass(frozen=True, slots=True)
class Entity:
    """One evidence-linked entity; anonymous people deliberately have no name."""

    entity_id: EntityId
    tenant_id: TenantId
    entity_type: EntityType
    canonical_name: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.entity_id, "entity_id")
        require_non_empty(self.tenant_id, "tenant_id")
        if self.canonical_name is not None:
            require_non_empty(self.canonical_name, "canonical_name")
        require_aware_datetime(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class EntityMention:
    """One entity occurrence grounded in an event and exact evidence span."""

    mention_id: MentionId
    tenant_id: TenantId
    entity_id: EntityId
    event_id: EventId
    evidence_id: EvidenceId
    confidence: float
    created_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.mention_id, "mention_id"),
            (self.tenant_id, "tenant_id"),
            (self.entity_id, "entity_id"),
            (self.event_id, "event_id"),
            (self.evidence_id, "evidence_id"),
        ):
            require_non_empty(value, name)
        require_aware_datetime(self.created_at, "created_at")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("entity mention confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Relation:
    """One deterministic typed edge between persisted memory graph records."""

    relation_id: RelationId
    tenant_id: TenantId
    source_type: RelationNodeType
    source_id: str
    relation_type: RelationType
    target_type: RelationNodeType
    target_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.relation_id, "relation_id"),
            (self.tenant_id, "tenant_id"),
            (self.source_id, "source_id"),
            (self.target_id, "target_id"),
        ):
            require_non_empty(value, name)
        require_aware_datetime(self.created_at, "created_at")
        if self.source_type is self.target_type and self.source_id == self.target_id:
            raise DomainInvariantError("relation cannot point a record to itself")
