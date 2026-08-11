"""Atomic input and output records for observation processing."""

from dataclasses import dataclass

from mindbridge.core import (
    Claim,
    DomainInvariantError,
    EmbeddingRecord,
    Entity,
    EntityMention,
    Event,
    EvidenceSpan,
    MediaObject,
    MemoryRecord,
    Observation,
    Relation,
    RelationNodeType,
    RelationType,
)

_OBSERVATION_RELATION_ENDPOINTS = frozenset(
    {
        (
            RelationNodeType.EVENT,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
        ),
        (RelationNodeType.EVENT, RelationType.MENTIONS, RelationNodeType.ENTITY),
        (RelationNodeType.EVENT, RelationType.ASSERTS, RelationNodeType.CLAIM),
        (
            RelationNodeType.CLAIM,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
        ),
        (RelationNodeType.CLAIM, RelationType.ABOUT, RelationNodeType.ENTITY),
    }
)


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Atomic evidence payload persisted for one observation."""

    media_objects: tuple[MediaObject, ...]
    observation: Observation
    evidence_spans: tuple[EvidenceSpan, ...]


@dataclass(frozen=True, slots=True)
class ObservationProcessingOutput:
    """Derived records committed together with successful job state."""

    events: tuple[Event, ...]
    entities: tuple[Entity, ...]
    entity_mentions: tuple[EntityMention, ...]
    claims: tuple[Claim, ...]
    memories: tuple[MemoryRecord, ...]
    relations: tuple[Relation, ...]
    embeddings: tuple[EmbeddingRecord, ...]

    def __post_init__(self) -> None:
        identifiers = _output_identifiers(self)
        _require_unique_ids(identifiers)
        _require_single_tenant(self)
        _require_valid_graph(self, identifiers)


def _output_identifiers(
    output: ObservationProcessingOutput,
) -> dict[str, tuple[str, ...]]:
    return {
        "event": tuple(str(item.event_id) for item in output.events),
        "entity": tuple(str(item.entity_id) for item in output.entities),
        "entity mention": tuple(str(item.mention_id) for item in output.entity_mentions),
        "claim": tuple(str(item.claim_id) for item in output.claims),
        "memory": tuple(str(item.memory_id) for item in output.memories),
        "relation": tuple(str(item.relation_id) for item in output.relations),
        "embedding": tuple(str(item.embedding_id) for item in output.embeddings),
    }


def _require_unique_ids(identifiers: dict[str, tuple[str, ...]]) -> None:
    for name, values in identifiers.items():
        if len(set(values)) != len(values):
            raise DomainInvariantError(f"derived {name} IDs must be unique")


def _require_single_tenant(output: ObservationProcessingOutput) -> None:
    tenant_ids = {
        item.tenant_id
        for items in (
            output.events,
            output.entities,
            output.entity_mentions,
            output.claims,
            output.memories,
            output.relations,
            output.embeddings,
        )
        for item in items
    }
    if len(tenant_ids) > 1:
        raise DomainInvariantError("derived records must belong to one tenant")


def _require_valid_graph(
    output: ObservationProcessingOutput,
    identifiers: dict[str, tuple[str, ...]],
) -> None:
    node_ids = {
        RelationNodeType.EVENT: set(identifiers["event"]),
        RelationNodeType.ENTITY: set(identifiers["entity"]),
        RelationNodeType.CLAIM: set(identifiers["claim"]),
        RelationNodeType.MEMORY_RECORD: set(identifiers["memory"]),
    }
    if any(
        str(mention.event_id) not in node_ids[RelationNodeType.EVENT]
        or str(mention.entity_id) not in node_ids[RelationNodeType.ENTITY]
        for mention in output.entity_mentions
    ):
        raise DomainInvariantError("derived entity mention references an unknown record")
    if set(identifiers["entity"]) != {str(mention.entity_id) for mention in output.entity_mentions}:
        raise DomainInvariantError("each derived entity must have an evidence-grounded mention")

    edges = {
        (
            relation.source_type,
            relation.source_id,
            relation.relation_type,
            relation.target_type,
            relation.target_id,
        )
        for relation in output.relations
    }
    if len(edges) != len(output.relations):
        raise DomainInvariantError("derived relation edges must be unique")
    if any(
        relation.source_id not in node_ids[relation.source_type]
        or relation.target_id not in node_ids[relation.target_type]
        for relation in output.relations
    ):
        raise DomainInvariantError("derived relation references an unknown graph record")
    if any(
        (relation.source_type, relation.relation_type, relation.target_type)
        not in _OBSERVATION_RELATION_ENDPOINTS
        for relation in output.relations
    ):
        raise DomainInvariantError("derived observation relation has invalid endpoint types")

    representations = {
        (source_type, source_id, target_id)
        for source_type, source_id, relation_type, target_type, target_id in edges
        if relation_type is RelationType.REPRESENTED_BY
        and target_type is RelationNodeType.MEMORY_RECORD
    }
    expected_sources = {
        *((RelationNodeType.EVENT, event_id) for event_id in identifiers["event"]),
        *((RelationNodeType.CLAIM, claim_id) for claim_id in identifiers["claim"]),
    }
    if (
        {(source_type, source_id) for source_type, source_id, _ in representations}
        != expected_sources
        or {target_id for _, _, target_id in representations}
        != node_ids[RelationNodeType.MEMORY_RECORD]
        or len(representations) != len(expected_sources)
    ):
        raise DomainInvariantError(
            "derived events and claims must map one-to-one to memory representations"
        )

    asserted_claims = {
        target_id
        for source_type, _, relation_type, target_type, target_id in edges
        if source_type is RelationNodeType.EVENT
        and relation_type is RelationType.ASSERTS
        and target_type is RelationNodeType.CLAIM
    }
    if asserted_claims != node_ids[RelationNodeType.CLAIM]:
        raise DomainInvariantError("each derived claim must be asserted by its event")

    mention_edges = {
        (source_id, target_id)
        for source_type, source_id, relation_type, target_type, target_id in edges
        if source_type is RelationNodeType.EVENT
        and relation_type is RelationType.MENTIONS
        and target_type is RelationNodeType.ENTITY
    }
    expected_mentions = {
        (str(mention.event_id), str(mention.entity_id)) for mention in output.entity_mentions
    }
    if mention_edges != expected_mentions:
        raise DomainInvariantError("entity mention rows and graph relations must agree")
