"""Bounded candidates and the pure write derivation for cross-clip entity resolution.

Perception names what it sees once per clip, so one person picked up over fifteen clips
becomes as many entities as the clips found ways to describe them. Identical names already
collapse, because `entity_id` is derived from the casefolded name, and an edge identity
signal already collapses the anonymous case. What is left is the same entity described two
different ways, and only evidence can settle that.

Everything here is deliberately pairwise. Adjudications are never composed: A~B and B~C
produce two edges and never A~C. Composition is how a released clustering of this same
video put 152 of 153 observations on one character and merged two visibly different people.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mindbridge.core import (
    DomainInvariantError,
    Entity,
    EntityId,
    EntityType,
    EvidenceId,
    Relation,
    RelationId,
    RelationNodeType,
    RelationType,
    TenantId,
    derive_stable_id,
    require_aware_datetime,
    require_non_empty,
)

_MAXIMUM_PAIRS_CEILING = 512


@dataclass(frozen=True, slots=True)
class EntityCandidateRequest:
    """One stable page plus every bound the sweep is allowed to spend."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_entity_id: EntityId | None = None
    limit: int = 16
    maximum_gap_seconds: int = 2_592_000
    candidate_limit: int = 8
    minimum_confidence: float = 0.75
    evidence_per_side: int = 3
    maximum_pairs: int = 64
    # Person only by default. The observed fragmentation is people, and whether two
    # descriptions of a tissue box merge is worth much less at identical risk.
    entity_types: tuple[EntityType, ...] = (EntityType.PERSON,)
    readjudicate: bool = False

    def __post_init__(self) -> None:
        require_non_empty(self.tenant_id, "tenant_id")
        require_aware_datetime(self.evaluated_at, "evaluated_at")
        if self.after_entity_id is not None:
            require_non_empty(self.after_entity_id, "after_entity_id")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("entity candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 31_536_000:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 31536000")
        if not 1 <= self.candidate_limit <= 32:
            raise DomainInvariantError("candidate_limit must be between 1 and 32")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise DomainInvariantError("minimum_confidence must be between 0 and 1")
        if not 1 <= self.evidence_per_side <= 8:
            raise DomainInvariantError("evidence_per_side must be between 1 and 8")
        if not 1 <= self.maximum_pairs <= _MAXIMUM_PAIRS_CEILING:
            raise DomainInvariantError(
                f"maximum_pairs must be between 1 and {_MAXIMUM_PAIRS_CEILING}"
            )
        if not self.entity_types:
            raise DomainInvariantError("entity_types must not be empty")
        if len(set(self.entity_types)) != len(self.entity_types):
            raise DomainInvariantError("entity_types must be unique")


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """One named entity and the evidence a judge may reopen for it."""

    entity: Entity
    evidence_ids: tuple[EvidenceId, ...]

    def __post_init__(self) -> None:
        if self.entity.canonical_name is None:
            raise DomainInvariantError(
                "identity-backed entities are already stable and are never adjudicated"
            )
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainInvariantError("entity candidate evidence IDs must be unique")


@dataclass(frozen=True, slots=True)
class EntityPair:
    """One unordered pair, held in the single canonical order."""

    left: EntityCandidate
    right: EntityCandidate

    def __post_init__(self) -> None:
        if self.left.entity.entity_id >= self.right.entity.entity_id:
            raise DomainInvariantError("entity pairs must be ordered by ascending entity_id")
        if self.left.entity.entity_type is not self.right.entity.entity_type:
            raise DomainInvariantError("entity pairs must share one entity_type")


@dataclass(frozen=True, slots=True)
class EntityCandidatePage:
    """Adjudicable pairs, cursor progress, and what the bound refused to look at."""

    pairs: tuple[EntityPair, ...]
    scanned_count: int
    dropped_pair_count: int
    next_cursor: EntityId | None

    def __post_init__(self) -> None:
        if self.scanned_count < 0 or self.dropped_pair_count < 0:
            raise DomainInvariantError("entity candidate counts must be non-negative")
        if self.next_cursor is not None:
            require_non_empty(self.next_cursor, "entity candidate cursor")
        keys = tuple(
            (pair.left.entity.entity_id, pair.right.entity.entity_id) for pair in self.pairs
        )
        if len(set(keys)) != len(keys):
            raise DomainInvariantError("entity candidate pairs must be unique")


@dataclass(frozen=True, slots=True)
class EntityAdjudication:
    """One verdict a judge reached after inspecting both sides' original media."""

    same_entity: bool
    confidence: float
    discriminating_cue: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("adjudication confidence must be between 0 and 1")
        require_non_empty(self.discriminating_cue, "discriminating_cue")


@dataclass(frozen=True, slots=True)
class EntityResolutionWrite:
    """Every edge one adjudicated page adds, and nothing else."""

    relations: tuple[Relation, ...]


def derive_entity_resolution_write(
    tenant_id: TenantId,
    decided: tuple[tuple[EntityPair, EntityAdjudication], ...],
    evaluated_at: datetime,
) -> EntityResolutionWrite:
    """Turn verdicts into pairwise edges, one per verdict, with nothing inferred."""
    return EntityResolutionWrite(
        relations=tuple(
            _relation(tenant_id, pair, adjudication, evaluated_at) for pair, adjudication in decided
        )
    )


def _relation(
    tenant_id: TenantId,
    pair: EntityPair,
    adjudication: EntityAdjudication,
    evaluated_at: datetime,
) -> Relation:
    relation_type = RelationType.SAME_AS if adjudication.same_entity else RelationType.NOT_SAME_AS
    source_id = pair.left.entity.entity_id
    target_id = pair.right.entity.entity_id
    return Relation(
        # The verdict is part of the identity: re-judging a pair the other way has to land
        # on its own row rather than silently overwrite the first answer.
        relation_id=RelationId(
            derive_stable_id("relation", tenant_id, relation_type.value, source_id, target_id)
        ),
        tenant_id=TenantId(tenant_id),
        source_type=RelationNodeType.ENTITY,
        source_id=source_id,
        relation_type=relation_type,
        target_type=RelationNodeType.ENTITY,
        target_id=target_id,
        created_at=evaluated_at,
    )
