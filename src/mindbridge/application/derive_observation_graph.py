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
    AnonymousIdentityObservation,
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
    IdentityScope,
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
from mindbridge.telemetry import operation_span


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


@operation_span("mindbridge.process_observation.embed_graph")
async def embed_observation_graph(
    tenant_id: TenantId,
    events: tuple[Event, ...],
    claims: tuple[Claim, ...],
    entities: tuple[Entity, ...],
    memories: tuple[MemoryRecord, ...],
    embedder: Embedder,
    created_at: datetime,
) -> tuple[EmbeddingRecord, ...]:
    """Batch event, claim, named entity, and memory text through the document encoder.

    Every event and claim is indexed twice: once as itself, and once as the memory that
    represents it. The second row is what `observe()` never wrote, and its absence is not a
    lost ranking signal but a dead code path: `recall` searches the MEMORY_RECORD channel on
    every recall and feeds the memory IDs it returns into `search_memories_by_ids` and
    `search_memories_by_hierarchy`, which are ID lookups rather than vector searches. With no
    rows in that channel those two receive an empty ID set and return nothing at all, so two
    of the four fused retrieval paths -- including every hierarchy summary consolidation
    built -- contributed nothing on an audiovisual tenant. Measured: 3 336 memories and zero
    vectors across six audiovisual benchmarks, against ~100% coverage on text tenants where
    `remember()` writes the row.

    The two rows share one vector rather than encoding the text twice, because a derived
    memory's summary *is* its record's own text, so a second round trip per memory buys
    identical numbers at the price of doubling the graph text through the encoder. That is a
    property of how the memory is built, not a law, so it is checked here against the record
    that will actually be committed instead of assumed: if a memory ever starts summarising
    rather than restating, this refuses to label one vector with both meanings.
    """
    memory_by_id = {memory.memory_id: memory for memory in memories}
    inputs: list[tuple[str, tuple[tuple[EmbeddedObjectType, str], ...]]] = []
    for event in events:
        inputs.append(
            (
                event.description,
                _shared_with_memory(
                    (EmbeddedObjectType.EVENT, event.event_id),
                    event.description,
                    memory_by_id,
                    representing_memory_id(event.event_id),
                ),
            )
        )
    for claim in claims:
        inputs.append(
            (
                claim.statement,
                _shared_with_memory(
                    (EmbeddedObjectType.CLAIM, claim.claim_id),
                    claim.statement,
                    memory_by_id,
                    representing_memory_id(claim.claim_id),
                ),
            )
        )
    inputs.extend(
        (entity.canonical_name, ((EmbeddedObjectType.ENTITY, entity.entity_id),))
        for entity in entities
        if entity.canonical_name is not None
    )
    if not inputs:
        return ()
    result = await embedder.embed(
        EmbedRequest(
            inputs=tuple(ModelInput((TextPart(text),)) for text, _ in inputs),
            task=EmbedTask.DOCUMENT,
        )
    )
    if len(result.embeddings) != len(inputs):
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
        for (_, objects), embedding in zip(inputs, result.embeddings, strict=True)
        for object_type, object_id in objects
    )


def _shared_with_memory(
    record: tuple[EmbeddedObjectType, str],
    text: str,
    memory_by_id: dict[MemoryId, MemoryRecord],
    memory_id: MemoryId,
) -> tuple[tuple[EmbeddedObjectType, str], ...]:
    """Name both objects one vector stands for, only while both carry the same text."""
    memory = memory_by_id.get(memory_id)
    if memory is None or memory.summary != text:
        raise MemoryIntegrityError(
            "a memory vector must encode the text of the record it represents"
        )
    return (record, (EmbeddedObjectType.MEMORY_RECORD, memory_id))


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


def _counted_statement(perceived: PerceivedClaim) -> str:
    """Fold a claim's exact count into the one text a reader ever sees.

    `PerceivedCount` is typed where the model produces it, but nothing downstream reads a typed
    number: the answer pipeline is handed `MemoryRecord.summary`, and a claim's memory restates
    its statement verbatim. A count kept beside the statement would therefore be write-only,
    which is the failure this change exists to avoid, so it is written into the statement --
    once, deterministically, rather than hoped for in the sentence the model wrote.

    Rendered as ordinary English on purpose. The statement is also the text that gets embedded,
    and `key=value` syntax in it would be tokens the query side never produces, while "exactly 3
    small monsters" is close to how a question about how many small monsters there were is
    phrased.
    """
    count = perceived.exact_count
    if count is None:
        return perceived.statement
    return f"{perceived.statement} (exactly {count.value} {count.subject})"


def _claim(
    observation: Observation,
    perception: EventPerception,
    event: Event,
    perceived: PerceivedClaim,
    claim_index: int,
    created_at: datetime,
) -> Claim:
    statement = _counted_statement(perceived)
    return Claim(
        claim_id=ClaimId(
            derive_stable_id(
                "claim",
                observation.tenant_id,
                event.event_id,
                claim_index,
                statement,
            )
        ),
        tenant_id=observation.tenant_id,
        claim_type=perceived.claim_type,
        statement=statement,
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


def representing_memory_id(record_id: str) -> MemoryId:
    """Derive in one place the memory that stands for one event or claim.

    Three callers need it: the builder that creates the memory, the encoder batch that indexes
    the memory's summary beside its record, and recall, which has to recognise the two hits one
    shared vector produces. A second copy of this derivation is a silent divergence -- the
    vector would be filed under a memory ID that nothing stores.
    """
    return MemoryId(derive_stable_id("memory", record_id))


def _event_memory(event: Event) -> MemoryRecord:
    return MemoryRecord(
        memory_id=representing_memory_id(event.event_id),
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
        memory_id=representing_memory_id(claim.claim_id),
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


def _identity_entity_id(
    observation: Observation,
    identity: AnonymousIdentityObservation,
) -> EntityId:
    """Make a device identity the same person across clips, and an observation-scoped one not.

    A device-scoped pseudonym is the one durable key an anonymous person has: the edge matched
    this face or voice against its own enrolled gallery, so the same string in next week's clip
    is the same human, and using it verbatim is what lets recall's co-mention expansion reach
    the other clips they appear in.

    `IdentityScope.OBSERVATION` states the opposite -- the pseudonym is only safe to reuse
    inside the observation that produced it, because it is a within-clip diarization or tracking
    label rather than a match. Using it verbatim made every caller-supplied `speaker_0` one
    tenant-wide person, silently merging strangers across the corpus, which is worse than having
    no identity at all: a wrong merge is asserted, and `MENTIONS` edges then join their events.
    Namespacing it by its observation keeps it stable exactly as far as its scope promises.
    """
    if identity.scope is IdentityScope.DEVICE:
        return EntityId(identity.identity_id)
    return EntityId(derive_stable_id("identity", observation.observation_id, identity.identity_id))


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
            entity_id = _identity_entity_id(observation, identity)
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
