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
    require_bounded_count,
    require_non_empty,
    require_probability,
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
        require_bounded_count(self.limit, "entity candidate page limit", minimum=1, maximum=32)
        require_bounded_count(
            self.maximum_gap_seconds, "maximum_gap_seconds", minimum=0, maximum=31_536_000
        )
        require_bounded_count(self.candidate_limit, "candidate_limit", minimum=1, maximum=32)
        require_probability(self.minimum_confidence, "minimum_confidence")
        require_bounded_count(self.evidence_per_side, "evidence_per_side", minimum=1, maximum=8)
        require_bounded_count(
            self.maximum_pairs, "maximum_pairs", minimum=1, maximum=_MAXIMUM_PAIRS_CEILING
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
        require_probability(self.confidence, "adjudication confidence")
        require_non_empty(self.discriminating_cue, "discriminating_cue")


@dataclass(frozen=True, slots=True)
class EntityResolutionWrite:
    """Every edge one adjudicated page adds, each bound to the verdict behind it.

    The edge and its adjudication are one value because they are only ever correct together.
    An edge whose cue was dropped is the unauditable merge this pass exists to prevent, and a
    cue whose edge was never committed justifies nothing. Carrying them as one tuple means no
    later caller can persist half of it.
    """

    decided: tuple[tuple[Relation, EntityAdjudication], ...]

    @property
    def relations(self) -> tuple[Relation, ...]:
        """The edges alone, for callers that count verdicts rather than persist them."""
        return tuple(relation for relation, _ in self.decided)


def derive_entity_resolution_write(
    tenant_id: TenantId,
    decided: tuple[tuple[EntityPair, EntityAdjudication], ...],
    evaluated_at: datetime,
) -> EntityResolutionWrite:
    """Turn verdicts into pairwise edges, one per pair, with nothing inferred."""
    keys = tuple((pair.left.entity.entity_id, pair.right.entity.entity_id) for pair, _ in decided)
    if len(set(keys)) != len(keys):
        # One pair, one verdict. Two verdicts for one pair is the worst state this subsystem
        # can produce: the graph would assert the records are and are not the same entity.
        raise DomainInvariantError("one entity pair cannot carry two verdicts")
    return EntityResolutionWrite(
        decided=tuple(
            (_relation(tenant_id, pair, adjudication, evaluated_at), adjudication)
            for pair, adjudication in decided
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
        # Keyed on the pair, deliberately NOT on the verdict, and deliberately not built by
        # core.derive_relation — that helper folds relation_type into the id, which would let
        # a flipped re-judgement insert a second row and leave the pair asserted both same
        # and not-same. One row per pair means a re-judgement can only replace a verdict.
        #
        # The cost of that choice: the shared relation writer in _postgres_graph inserts with
        # ON CONFLICT DO NOTHING and then raises MemoryIntegrityError when the stored row
        # differs from the one offered — created_at included. Since created_at is the sweep's
        # evaluated_at, this id is stable while that column is not, so commit_entity_resolution
        # MUST upsert relation_type and created_at on (tenant_id, relation_id) conflict and
        # cannot reuse that strict writer.
        relation_id=RelationId(
            derive_stable_id("relation", tenant_id, "entity_resolution", source_id, target_id)
        ),
        tenant_id=TenantId(tenant_id),
        source_type=RelationNodeType.ENTITY,
        source_id=source_id,
        relation_type=relation_type,
        target_type=RelationNodeType.ENTITY,
        target_id=target_id,
        created_at=evaluated_at,
    )
