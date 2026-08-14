"""Bounded candidates for hierarchical memory summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.application.capabilities import (
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EntityId,
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


class SummaryScope(str, Enum):
    """Model-selected coherence boundary for one hierarchy node."""

    SESSION = "session"
    DAY = "day"
    PERSON = "person"
    PLACE = "place"
    TOPIC = "topic"


@dataclass(frozen=True, slots=True)
class SummaryCandidateCursor:
    """Self-contained position that survives deletion of the prior seed."""

    occurred_at: datetime
    memory_id: MemoryId

    def __post_init__(self) -> None:
        if self.occurred_at.utcoffset() is None:
            raise DomainInvariantError("Summary candidate cursor time must be timezone-aware")
        if not self.memory_id.strip():
            raise DomainInvariantError("Summary candidate cursor Memory ID must not be empty")


@dataclass(frozen=True, slots=True)
class SummaryCandidateRequest:
    """Stable page and calibrated affinity bounds for one tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_cursor: SummaryCandidateCursor | None = None
    limit: int = 16
    maximum_gap_seconds: int = 2_592_000
    minimum_similarity: float = 0.8

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise DomainInvariantError("tenant_id must not be empty")
        if self.evaluated_at.utcoffset() is None:
            raise DomainInvariantError("evaluated_at must be timezone-aware")
        if not 1 <= self.limit <= 32:
            raise DomainInvariantError("Summary candidate page limit must be between 1 and 32")
        if not 0 <= self.maximum_gap_seconds <= 31_536_000:
            raise DomainInvariantError("maximum_gap_seconds must be between 0 and 31536000")
        if not math.isfinite(self.minimum_similarity) or not -1.0 <= self.minimum_similarity <= 1.0:
            raise DomainInvariantError("minimum_similarity must be between -1 and 1")


@dataclass(frozen=True, slots=True)
class SummaryCandidate:
    """One current Memory and its evidence-derived entity context."""

    memory: MemoryRecord
    entity_ids: tuple[EntityId, ...]

    def __post_init__(self) -> None:
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise DomainInvariantError("Summary candidate entity IDs must be unique")
        if (
            self.memory.memory_type not in {MemoryType.EPISODIC, MemoryType.SEMANTIC}
            or self.memory.verification_status
            not in {VerificationStatus.VERIFIED, VerificationStatus.ATTESTED}
            or self.memory.superseded_at is not None
        ):
            raise DomainInvariantError("Summary candidates must be current grounded memories")


@dataclass(frozen=True, slots=True)
class SummaryCandidatePage:
    """Related Memories plus cursor progress across all examined seeds."""

    candidates: tuple[SummaryCandidate, ...]
    scanned_count: int
    next_cursor: SummaryCandidateCursor | None

    def __post_init__(self) -> None:
        memories = tuple(candidate.memory for candidate in self.candidates)
        if self.scanned_count < 0:
            raise DomainInvariantError("Summary candidate scanned_count must be non-negative")
        if len({memory.memory_id for memory in memories}) != len(memories):
            raise DomainInvariantError("Summary candidate Memory IDs must be unique")
        if len({memory.tenant_id for memory in memories}) > 1:
            raise DomainInvariantError("Summary candidates must belong to one tenant")


@dataclass(frozen=True, slots=True)
class SummaryProposal:
    """One coherent parent proposed over disjoint source Memories."""

    source_memory_ids: tuple[MemoryId, ...]
    scope: SummaryScope
    summary: str
    salience: float

    def __post_init__(self) -> None:
        if not 2 <= len(self.source_memory_ids) <= 32 or len(set(self.source_memory_ids)) != len(
            self.source_memory_ids
        ):
            raise DomainInvariantError("Summary requires 2 to 32 unique source Memories")
        if not self.summary.strip():
            raise DomainInvariantError("Summary text must not be empty")
        if not math.isfinite(self.salience) or not 0.0 <= self.salience <= 1.0:
            raise DomainInvariantError("Summary salience must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SummaryConsolidation:
    """Validated hierarchy proposals and frozen model provenance."""

    summaries: tuple[SummaryProposal, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("Summary consolidation prompt version must not be empty")
        source_ids = [
            memory_id for proposal in self.summaries for memory_id in proposal.source_memory_ids
        ]
        if len(set(source_ids)) != len(source_ids):
            raise DomainInvariantError("one Memory cannot belong to multiple Summary proposals")


@dataclass(frozen=True, slots=True)
class SummaryWrite:
    """One complete parent Memory aggregate ready for atomic persistence."""

    memory: MemoryRecord
    source_memory_ids: tuple[MemoryId, ...]
    relations: tuple[Relation, ...]
    embedding: EmbeddingRecord

    def __post_init__(self) -> None:
        if not 2 <= len(self.source_memory_ids) <= 32 or len(set(self.source_memory_ids)) != len(
            self.source_memory_ids
        ):
            raise DomainInvariantError("Summary write requires unique source Memories")
        if self.memory.memory_id in self.source_memory_ids:
            raise DomainInvariantError("Summary Memory cannot contain itself")
        if self.embedding.tenant_id != self.memory.tenant_id or any(
            relation.tenant_id != self.memory.tenant_id for relation in self.relations
        ):
            raise DomainInvariantError("Summary records must belong to one tenant")
        if self.memory.memory_type is not MemoryType.SEMANTIC:
            raise DomainInvariantError("Summary parent must be a Semantic Memory")
        if (
            self.embedding.object_type is not EmbeddedObjectType.MEMORY_RECORD
            or self.embedding.object_id != self.memory.memory_id
            or self.embedding.task != EmbedTask.DOCUMENT.value
            or not self.embedding.normalized
            or self.embedding.created_at != self.memory.created_at
        ):
            raise DomainInvariantError("Summary embedding must index its parent Memory")
        expected_edges = {
            (
                RelationNodeType.MEMORY_RECORD,
                self.memory.memory_id,
                RelationType.CONTAINS,
                RelationNodeType.MEMORY_RECORD,
                source_memory_id,
            )
            for source_memory_id in self.source_memory_ids
        }
        actual_edges = {
            (
                relation.source_type,
                relation.source_id,
                relation.relation_type,
                relation.target_type,
                relation.target_id,
            )
            for relation in self.relations
        }
        if len(self.relations) != len(expected_edges) or actual_edges != expected_edges:
            raise DomainInvariantError("Summary relations do not match its source Memories")


async def derive_summary_writes(
    tenant_id: TenantId,
    candidates: tuple[SummaryCandidate, ...],
    consolidation: SummaryConsolidation,
    text_embedder: Embedder,
    created_at: datetime,
) -> tuple[SummaryWrite, ...]:
    """Build deterministic hierarchy nodes and aligned Memory vectors."""
    candidate_by_id = {candidate.memory.memory_id: candidate.memory for candidate in candidates}
    memories_and_sources = tuple(
        _summary_memory(
            tenant_id,
            proposal,
            candidate_by_id,
            consolidation.model_reference,
            consolidation.prompt_version,
            created_at,
        )
        for proposal in consolidation.summaries
    )
    result = await text_embedder.embed(
        EmbedRequest(
            inputs=tuple(
                ModelInput((TextPart(memory.summary),)) for memory, _ in memories_and_sources
            ),
            task=EmbedTask.DOCUMENT,
        )
    )
    if len(result.embeddings) != len(memories_and_sources):
        raise MemoryIntegrityError("embedder returned the wrong summary vector count")
    return tuple(
        _summary_write(memory, source_ids, embedding, created_at)
        for (memory, source_ids), embedding in zip(
            memories_and_sources, result.embeddings, strict=True
        )
    )


def _summary_memory(
    tenant_id: TenantId,
    proposal: SummaryProposal,
    candidate_by_id: dict[MemoryId, MemoryRecord],
    model_reference: ModelReference,
    prompt_version: str,
    created_at: datetime,
) -> tuple[MemoryRecord, tuple[MemoryId, ...]]:
    source_ids = tuple(sorted(proposal.source_memory_ids))
    sources = tuple(
        sorted(
            (candidate_by_id[memory_id] for memory_id in source_ids),
            key=lambda memory: (memory.occurred_at, memory.memory_id),
        )
    )
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for source in sources for evidence_id in source.evidence_ids)
    )
    verification_status = (
        VerificationStatus.VERIFIED
        if evidence_ids
        and all(source.verification_status is VerificationStatus.VERIFIED for source in sources)
        else VerificationStatus.UNVERIFIED
    )
    return (
        MemoryRecord(
            memory_id=MemoryId(
                derive_stable_id(
                    "summary-memory",
                    tenant_id,
                    model_reference.model_id,
                    model_reference.revision,
                    prompt_version,
                    proposal.scope.value,
                    created_at.isoformat(),
                    *source_ids,
                )
            ),
            tenant_id=tenant_id,
            memory_type=MemoryType.SEMANTIC,
            summary=proposal.summary,
            evidence_ids=evidence_ids,
            occurred_at=min(source.occurred_at for source in sources),
            ended_at=max(source.ended_at for source in sources),
            created_at=created_at,
            verification_status=verification_status,
            model_reference=model_reference,
            salience=proposal.salience,
            strength=proposal.salience,
        ),
        source_ids,
    )


def _summary_write(
    memory: MemoryRecord,
    source_memory_ids: tuple[MemoryId, ...],
    embedding: Embedding,
    created_at: datetime,
) -> SummaryWrite:
    return SummaryWrite(
        memory=memory,
        source_memory_ids=source_memory_ids,
        relations=tuple(
            derive_relation(
                memory.tenant_id,
                RelationNodeType.MEMORY_RECORD,
                memory.memory_id,
                RelationType.CONTAINS,
                RelationNodeType.MEMORY_RECORD,
                source_memory_id,
                created_at,
            )
            for source_memory_id in source_memory_ids
        ),
        embedding=EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    memory.tenant_id,
                    EmbeddedObjectType.MEMORY_RECORD.value,
                    memory.memory_id,
                    embedding.model_reference.model_id,
                    embedding.model_reference.revision,
                    EmbedTask.DOCUMENT.value,
                )
            ),
            tenant_id=memory.tenant_id,
            object_type=EmbeddedObjectType.MEMORY_RECORD,
            object_id=memory.memory_id,
            values=embedding.values,
            model_reference=embedding.model_reference,
            space_reference=embedding.space_reference,
            task=EmbedTask.DOCUMENT.value,
            dimension=embedding.dimension,
            normalized=True,
            created_at=created_at,
        ),
    )
