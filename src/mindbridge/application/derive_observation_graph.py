"""Deterministic graph records derived from one validated perception."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from mindbridge.application.capabilities import (
    Embedder,
    EmbedRequest,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.application.perception import (
    EventPerception,
    PerceivedClaim,
    PerceivedEntity,
    time_ranges_overlap,
)
from mindbridge.core import (
    Claim,
    ClaimId,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    Entity,
    EntityId,
    EntityMention,
    EntityType,
    Event,
    EvidenceSpan,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryType,
    MentionId,
    Observation,
    Relation,
    RelationId,
    RelationNodeType,
    RelationType,
    TenantId,
    VerificationStatus,
    derive_relation,
    derive_stable_id,
)
from mindbridge.telemetry import trace_operation


@dataclass(frozen=True, slots=True)
class DerivedObservationGraph:
    """Records committed atomically beside one observation's events."""

    entities: tuple[Entity, ...]
    entity_mentions: tuple[EntityMention, ...]
    claims: tuple[Claim, ...]
    memories: tuple[MemoryRecord, ...]
    relations: tuple[Relation, ...]


def derive_observation_graph(
    observation: Observation,
    perception: EventPerception,
    events: tuple[Event, ...],
    evidence_spans: tuple[EvidenceSpan, ...],
    created_at: datetime,
) -> DerivedObservationGraph:
    """Create retry-stable entities, claims, memories, and typed edges."""
    entities: dict[EntityId, Entity] = {}
    mentions: dict[MentionId, EntityMention] = {}
    claims: list[Claim] = []
    event_memories = tuple(_event_memory(event) for event in events)
    claim_memories: list[MemoryRecord] = []
    relations: dict[RelationId, Relation] = {}

    for event, memory in zip(events, event_memories, strict=True):
        relation = derive_relation(
            observation.tenant_id,
            RelationNodeType.EVENT,
            event.event_id,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
            memory.memory_id,
            created_at,
        )
        relations[relation.relation_id] = relation

    for event, perceived_event in zip(events, perception.events, strict=True):
        event_entities: list[Entity] = []
        for perceived_entity in perceived_event.entities:
            entity = _perceived_entity(observation, perceived_entity, created_at)
            entities[entity.entity_id] = entity
            event_entities.append(entity)
            for mention in _entity_mentions(event, entity, perceived_entity, created_at):
                existing = mentions.get(mention.mention_id)
                if existing is None or mention.confidence > existing.confidence:
                    mentions[mention.mention_id] = mention
            relation = derive_relation(
                observation.tenant_id,
                RelationNodeType.EVENT,
                event.event_id,
                RelationType.MENTIONS,
                RelationNodeType.ENTITY,
                entity.entity_id,
                created_at,
            )
            relations[relation.relation_id] = relation

        for claim_index, perceived_claim in enumerate(perceived_event.claims):
            claim = _claim(
                observation,
                perception,
                event,
                perceived_claim,
                claim_index,
                created_at,
            )
            memory = _claim_memory(claim, event)
            claims.append(claim)
            claim_memories.append(memory)
            for relation in (
                derive_relation(
                    observation.tenant_id,
                    RelationNodeType.EVENT,
                    event.event_id,
                    RelationType.ASSERTS,
                    RelationNodeType.CLAIM,
                    claim.claim_id,
                    created_at,
                ),
                derive_relation(
                    observation.tenant_id,
                    RelationNodeType.CLAIM,
                    claim.claim_id,
                    RelationType.REPRESENTED_BY,
                    RelationNodeType.MEMORY_RECORD,
                    memory.memory_id,
                    created_at,
                ),
            ):
                relations[relation.relation_id] = relation
            for entity_index in perceived_claim.entity_indices:
                relation = derive_relation(
                    observation.tenant_id,
                    RelationNodeType.CLAIM,
                    claim.claim_id,
                    RelationType.ABOUT,
                    RelationNodeType.ENTITY,
                    event_entities[entity_index].entity_id,
                    created_at,
                )
                relations[relation.relation_id] = relation

    identity_entities, identity_mentions = _identity_graph(
        observation,
        events,
        evidence_spans,
        created_at,
    )
    entities.update((entity.entity_id, entity) for entity in identity_entities)
    mentions.update((mention.mention_id, mention) for mention in identity_mentions)
    for mention in identity_mentions:
        relation = derive_relation(
            observation.tenant_id,
            RelationNodeType.EVENT,
            mention.event_id,
            RelationType.MENTIONS,
            RelationNodeType.ENTITY,
            mention.entity_id,
            created_at,
        )
        relations[relation.relation_id] = relation

    return DerivedObservationGraph(
        entities=tuple(entities.values()),
        entity_mentions=tuple(mentions.values()),
        claims=tuple(claims),
        memories=event_memories + tuple(claim_memories),
        relations=tuple(relations.values()),
    )


@trace_operation("mindbridge.process_observation.embed_graph")
async def embed_observation_graph(
    tenant_id: TenantId,
    events: tuple[Event, ...],
    claims: tuple[Claim, ...],
    entities: tuple[Entity, ...],
    embedder: Embedder,
    created_at: datetime,
) -> tuple[EmbeddingRecord, ...]:
    """Batch event, claim, and named entity text through the selected document encoder."""
    objects = (
        tuple((EmbeddedObjectType.EVENT, event.event_id, event.description) for event in events)
        + tuple((EmbeddedObjectType.CLAIM, claim.claim_id, claim.statement) for claim in claims)
        + tuple(
            (EmbeddedObjectType.ENTITY, entity.entity_id, entity.canonical_name)
            for entity in entities
            if entity.canonical_name is not None
        )
    )
    if not objects:
        return ()
    result = await embedder.embed(
        EmbedRequest(
            inputs=tuple(ModelInput((TextPart(text),)) for _, _, text in objects),
            task=EmbedTask.DOCUMENT,
        )
    )
    if len(result.embeddings) != len(objects):
        raise MemoryIntegrityError("embedder returned the wrong graph vector count")
    return tuple(
        EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    tenant_id,
                    object_type.value,
                    object_id,
                    embedding.model_reference.model_id,
                    embedding.model_reference.revision,
                    EmbedTask.DOCUMENT.value,
                )
            ),
            tenant_id=tenant_id,
            object_type=object_type,
            object_id=object_id,
            values=embedding.values,
            model_reference=embedding.model_reference,
            space_reference=embedding.space_reference,
            task=EmbedTask.DOCUMENT.value,
            dimension=embedding.dimension,
            normalized=True,
            created_at=created_at,
        )
        for (object_type, object_id, _), embedding in zip(objects, result.embeddings, strict=True)
    )


def _perceived_entity(
    observation: Observation,
    perceived: PerceivedEntity,
    created_at: datetime,
) -> Entity:
    """Key a named entity per tenant so every mentioning event shares one graph node."""
    # ponytail: exact type plus casefolded name. Two different people with one name merge;
    # split them on evidence only once a real corpus shows the false merges.
    # The stored name is the casefolded one the ID is derived from: keeping the perceived
    # casing would make the row depend on which clip arrived first, and the second casing
    # of one name would then collide with its own entity ID in the store.
    canonical_name = perceived.canonical_name.casefold()
    return Entity(
        entity_id=EntityId(
            derive_stable_id(
                "entity",
                observation.tenant_id,
                perceived.entity_type.value,
                canonical_name,
            )
        ),
        tenant_id=observation.tenant_id,
        entity_type=perceived.entity_type,
        canonical_name=canonical_name,
        created_at=created_at,
    )


def _entity_mentions(
    event: Event,
    entity: Entity,
    perceived: PerceivedEntity,
    created_at: datetime,
) -> tuple[EntityMention, ...]:
    return tuple(
        EntityMention(
            mention_id=MentionId(
                derive_stable_id(
                    "mention", event.tenant_id, entity.entity_id, event.event_id, evidence_id
                )
            ),
            tenant_id=event.tenant_id,
            entity_id=entity.entity_id,
            event_id=event.event_id,
            evidence_id=evidence_id,
            confidence=perceived.confidence,
            created_at=created_at,
        )
        for evidence_id in perceived.evidence_ids
    )


def _claim(
    observation: Observation,
    perception: EventPerception,
    event: Event,
    perceived: PerceivedClaim,
    claim_index: int,
    created_at: datetime,
) -> Claim:
    return Claim(
        claim_id=ClaimId(
            derive_stable_id(
                "claim",
                observation.tenant_id,
                event.event_id,
                claim_index,
                perceived.statement,
            )
        ),
        tenant_id=observation.tenant_id,
        claim_type=perceived.claim_type,
        statement=perceived.statement,
        evidence_ids=perceived.evidence_ids,
        confidence=perceived.confidence,
        verification_status=VerificationStatus.VERIFIED,
        valid_from=observation.occurred_at + timedelta(milliseconds=perceived.valid_from_ms),
        valid_to=(
            observation.occurred_at + timedelta(milliseconds=perceived.valid_to_ms)
            if perceived.valid_to_ms is not None
            else None
        ),
        created_at=created_at,
        model_reference=perception.model_reference,
        prompt_version=perception.prompt_version,
    )


def _event_memory(event: Event) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(derive_stable_id("memory", event.event_id)),
        tenant_id=event.tenant_id,
        memory_type=MemoryType.EPISODIC,
        summary=event.description,
        evidence_ids=event.evidence_ids,
        occurred_at=event.occurred_at,
        ended_at=event.ended_at,
        created_at=event.created_at,
        verification_status=VerificationStatus.VERIFIED,
        model_reference=event.model_reference,
        salience=event.salience,
        strength=event.salience,
    )


def _claim_memory(claim: Claim, event: Event) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(derive_stable_id("memory", claim.claim_id)),
        tenant_id=claim.tenant_id,
        memory_type=MemoryType.SEMANTIC,
        summary=claim.statement,
        evidence_ids=claim.evidence_ids,
        occurred_at=claim.valid_from,
        ended_at=claim.valid_to or event.ended_at,
        created_at=claim.created_at,
        verification_status=claim.verification_status,
        model_reference=claim.model_reference,
        salience=claim.confidence,
        strength=claim.confidence,
    )


def _identity_graph(
    observation: Observation,
    events: tuple[Event, ...],
    evidence_spans: tuple[EvidenceSpan, ...],
    created_at: datetime,
) -> tuple[tuple[Entity, ...], tuple[EntityMention, ...]]:
    entities: dict[EntityId, Entity] = {}
    mentions: dict[MentionId, EntityMention] = {}
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidence_spans}
    for event in events:
        event_start_ms = round(
            (event.occurred_at - observation.occurred_at).total_seconds() * 1_000
        )
        event_end_ms = round((event.ended_at - observation.occurred_at).total_seconds() * 1_000)
        for identity in observation.identity_observations:
            if not time_ranges_overlap(
                identity.start_ms, identity.end_ms, event_start_ms, event_end_ms
            ):
                continue
            matching_evidence_ids = tuple(
                evidence_id
                for evidence_id in event.evidence_ids
                if time_ranges_overlap(
                    identity.start_ms,
                    identity.end_ms,
                    evidence_by_id[evidence_id].start_ms,
                    evidence_by_id[evidence_id].end_ms,
                )
            )
            if not matching_evidence_ids:
                continue
            entity_id = EntityId(identity.identity_id)
            entities.setdefault(
                entity_id,
                Entity(
                    entity_id=entity_id,
                    tenant_id=observation.tenant_id,
                    entity_type=EntityType.PERSON,
                    canonical_name=None,
                    created_at=created_at,
                ),
            )
            for evidence_id in matching_evidence_ids:
                mention = EntityMention(
                    mention_id=MentionId(
                        derive_stable_id(
                            "mention",
                            event.tenant_id,
                            entity_id,
                            event.event_id,
                            evidence_id,
                        )
                    ),
                    tenant_id=event.tenant_id,
                    entity_id=entity_id,
                    event_id=event.event_id,
                    evidence_id=evidence_id,
                    confidence=identity.confidence,
                    created_at=created_at,
                )
                existing = mentions.get(mention.mention_id)
                if existing is None or mention.confidence > existing.confidence:
                    mentions[mention.mention_id] = mention
    return tuple(entities.values()), tuple(mentions.values())
