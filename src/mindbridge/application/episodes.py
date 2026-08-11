"""Pure derivation of evidence-backed Episode graph aggregates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from mindbridge.application.ports import TextDocumentEmbedder
from mindbridge.application.recall import RETRIEVAL_DOCUMENT_EMBEDDING_TASK
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    Event,
    EventHierarchyLevel,
    EventId,
    EventStatus,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryType,
    ModelReference,
    Relation,
    RelationNodeType,
    RelationType,
    TenantId,
    VerificationStatus,
    derive_relation,
    derive_stable_id,
)


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


@dataclass(frozen=True, slots=True)
class EpisodeWrite:
    """One complete Episode aggregate ready for an atomic persistence attempt."""

    episode: Event
    child_event_ids: tuple[EventId, ...]
    memory: MemoryRecord
    relations: tuple[Relation, ...]
    embedding: EmbeddingRecord

    def __post_init__(self) -> None:
        if (
            self.episode.hierarchy_level is not EventHierarchyLevel.EPISODE
            or self.episode.status is not EventStatus.ACTIVE
            or self.episode.parent_event_id is not None
        ):
            raise DomainInvariantError("episode write requires one active root Episode")
        if not 2 <= len(self.child_event_ids) <= 32 or len(set(self.child_event_ids)) != len(
            self.child_event_ids
        ):
            raise DomainInvariantError("episode write requires 2 to 32 unique child events")
        if self.episode.event_id in self.child_event_ids:
            raise DomainInvariantError("episode cannot contain itself")
        if (
            self.memory.tenant_id != self.episode.tenant_id
            or self.embedding.tenant_id != self.episode.tenant_id
            or any(relation.tenant_id != self.episode.tenant_id for relation in self.relations)
        ):
            raise DomainInvariantError("episode aggregate records must belong to one tenant")
        if (
            self.memory.memory_type is not MemoryType.EPISODIC
            or self.memory.summary != self.episode.description
            or self.memory.evidence_ids != self.episode.evidence_ids
            or self.memory.occurred_at != self.episode.occurred_at
            or self.memory.ended_at != self.episode.ended_at
            or self.memory.created_at != self.episode.created_at
            or self.memory.model_reference != self.episode.model_reference
            or self.memory.salience != self.episode.salience
            or self.memory.verification_status is not VerificationStatus.VERIFIED
        ):
            raise DomainInvariantError("episode memory must represent the complete Episode")
        if (
            self.embedding.object_type is not EmbeddedObjectType.EVENT
            or self.embedding.object_id != self.episode.event_id
            or self.embedding.task != RETRIEVAL_DOCUMENT_EMBEDDING_TASK
            or not self.embedding.normalized
            or self.embedding.created_at != self.episode.created_at
        ):
            raise DomainInvariantError("episode embedding must index its Episode")
        expected_edges = _expected_episode_edges(self)
        if (
            len(self.relations) != len(expected_edges)
            or _relation_edges(self.relations) != expected_edges
        ):
            raise DomainInvariantError(
                "episode relations must exactly link its memory and children"
            )


async def derive_episode_writes(
    tenant_id: TenantId,
    candidates: tuple[Event, ...],
    consolidation: EpisodeConsolidation,
    text_embedder: TextDocumentEmbedder,
    created_at: datetime,
) -> tuple[EpisodeWrite, ...]:
    """Build deterministic Episode graph records and aligned text vectors."""
    if not consolidation.episodes:
        return ()
    candidate_by_id = {event.event_id: event for event in candidates}
    episodes = tuple(
        _episode_event(
            tenant_id,
            proposal,
            candidate_by_id,
            consolidation.model_reference,
            consolidation.prompt_version,
            created_at,
        )
        for proposal in consolidation.episodes
    )
    vectors = await text_embedder.encode_documents(
        tuple(episode.description for episode in episodes)
    )
    if len(vectors) != len(episodes):
        raise MemoryIntegrityError("text embedder returned the wrong Episode vector count")
    return tuple(
        _episode_write(episode, proposal.event_ids, vector, text_embedder, created_at)
        for episode, proposal, vector in zip(
            episodes,
            consolidation.episodes,
            vectors,
            strict=True,
        )
    )


def _episode_event(
    tenant_id: TenantId,
    proposal: EpisodeProposal,
    candidate_by_id: dict[EventId, Event],
    model_reference: ModelReference,
    prompt_version: str,
    created_at: datetime,
) -> Event:
    children = tuple(
        sorted(
            (candidate_by_id[event_id] for event_id in proposal.event_ids),
            key=lambda event: (event.occurred_at, event.event_id),
        )
    )
    return Event(
        event_id=EventId(
            derive_stable_id(
                "episode",
                tenant_id,
                model_reference.model_id,
                model_reference.revision,
                prompt_version,
                created_at.isoformat(),
                *sorted(str(event_id) for event_id in proposal.event_ids),
            )
        ),
        tenant_id=tenant_id,
        observation_ids=tuple(
            dict.fromkeys(
                observation_id for child in children for observation_id in child.observation_ids
            )
        ),
        evidence_ids=tuple(
            dict.fromkeys(evidence_id for child in children for evidence_id in child.evidence_ids)
        ),
        occurred_at=min(child.occurred_at for child in children),
        ended_at=max(child.ended_at for child in children),
        description=proposal.description,
        salience=proposal.salience,
        created_at=created_at,
        model_reference=model_reference,
        prompt_version=prompt_version,
        hierarchy_level=EventHierarchyLevel.EPISODE,
    )


def _episode_write(
    episode: Event,
    child_event_ids: tuple[EventId, ...],
    vector: tuple[float, ...],
    text_embedder: TextDocumentEmbedder,
    created_at: datetime,
) -> EpisodeWrite:
    memory = MemoryRecord(
        memory_id=MemoryId(derive_stable_id("memory", episode.event_id)),
        tenant_id=episode.tenant_id,
        memory_type=MemoryType.EPISODIC,
        summary=episode.description,
        evidence_ids=episode.evidence_ids,
        occurred_at=episode.occurred_at,
        ended_at=episode.ended_at,
        created_at=created_at,
        verification_status=VerificationStatus.VERIFIED,
        model_reference=episode.model_reference,
        salience=episode.salience,
        strength=episode.salience,
    )
    relations = (
        derive_relation(
            episode.tenant_id,
            RelationNodeType.EVENT,
            episode.event_id,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
            memory.memory_id,
            created_at,
        ),
        *(
            relation
            for child_event_id in child_event_ids
            for relation in (
                derive_relation(
                    episode.tenant_id,
                    RelationNodeType.EVENT,
                    episode.event_id,
                    RelationType.CONTAINS,
                    RelationNodeType.EVENT,
                    child_event_id,
                    created_at,
                ),
                derive_relation(
                    episode.tenant_id,
                    RelationNodeType.EVENT,
                    child_event_id,
                    RelationType.SAME_EPISODE,
                    RelationNodeType.EVENT,
                    episode.event_id,
                    created_at,
                ),
            )
        ),
    )
    return EpisodeWrite(
        episode=episode,
        child_event_ids=child_event_ids,
        memory=memory,
        relations=relations,
        embedding=EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    episode.tenant_id,
                    EmbeddedObjectType.EVENT.value,
                    episode.event_id,
                    text_embedder.model_reference.model_id,
                    text_embedder.model_reference.revision,
                    RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                )
            ),
            tenant_id=episode.tenant_id,
            object_type=EmbeddedObjectType.EVENT,
            object_id=episode.event_id,
            values=vector,
            model_reference=text_embedder.model_reference,
            space_reference=text_embedder.space_reference,
            task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
            dimension=text_embedder.dimension,
            normalized=True,
            created_at=created_at,
        ),
    )


def _relation_edges(
    relations: tuple[Relation, ...],
) -> set[tuple[RelationNodeType, str, RelationType, RelationNodeType, str]]:
    return {
        (
            relation.source_type,
            relation.source_id,
            relation.relation_type,
            relation.target_type,
            relation.target_id,
        )
        for relation in relations
    }


def _expected_episode_edges(
    write: EpisodeWrite,
) -> set[tuple[RelationNodeType, str, RelationType, RelationNodeType, str]]:
    return {
        (
            RelationNodeType.EVENT,
            write.episode.event_id,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
            write.memory.memory_id,
        ),
        *(
            (
                RelationNodeType.EVENT,
                write.episode.event_id,
                RelationType.CONTAINS,
                RelationNodeType.EVENT,
                child_event_id,
            )
            for child_event_id in write.child_event_ids
        ),
        *(
            (
                RelationNodeType.EVENT,
                child_event_id,
                RelationType.SAME_EPISODE,
                RelationNodeType.EVENT,
                write.episode.event_id,
            )
            for child_event_id in write.child_event_ids
        ),
    }
