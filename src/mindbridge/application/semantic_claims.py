"""Pure derivation of evidence-backed semantic Claim graph aggregates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from mindbridge.application.capabilities import (
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.application.claim_consolidation import ClaimCandidate
from mindbridge.core import (
    Claim,
    ClaimId,
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


@dataclass(frozen=True, slots=True)
class SemanticClaimProposal:
    """One stronger Claim grounded in two or more mutually supporting Claims."""

    source_claim_ids: tuple[ClaimId, ...]
    statement: str
    confidence: float

    def __post_init__(self) -> None:
        if not 2 <= len(self.source_claim_ids) <= 32 or len(set(self.source_claim_ids)) != len(
            self.source_claim_ids
        ):
            raise DomainInvariantError("semantic Claim requires 2 to 32 unique source Claims")
        if not self.statement.strip():
            raise DomainInvariantError("semantic Claim statement must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("semantic Claim confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ClaimRelationshipProposal:
    """One evidence-verified contradiction or temporal replacement between Claims."""

    source_claim_id: ClaimId
    relation_type: RelationType
    target_claim_id: ClaimId

    def __post_init__(self) -> None:
        if self.source_claim_id == self.target_claim_id:
            raise DomainInvariantError("Claim relationship cannot point to itself")
        if self.relation_type not in {RelationType.CONTRADICTS, RelationType.SUPERSEDES}:
            raise DomainInvariantError("Claim relationship must contradict or supersede")


@dataclass(frozen=True, slots=True)
class ClaimConsolidation:
    """Validated Claim proposals and frozen model provenance."""

    semantic_claims: tuple[SemanticClaimProposal, ...]
    relationships: tuple[ClaimRelationshipProposal, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("Claim consolidation prompt version must not be empty")
        support_ids = [
            claim_id for proposal in self.semantic_claims for claim_id in proposal.source_claim_ids
        ]
        if len(set(support_ids)) != len(support_ids):
            raise DomainInvariantError("one Claim cannot support multiple semantic proposals")
        support_id_set = set(support_ids)
        relationship_pairs = [
            frozenset((proposal.source_claim_id, proposal.target_claim_id))
            for proposal in self.relationships
        ]
        if len(set(relationship_pairs)) != len(relationship_pairs):
            raise DomainInvariantError("a Claim pair can have only one semantic relationship")
        if any(
            claim_id in support_id_set
            for relationship in self.relationships
            for claim_id in (relationship.source_claim_id, relationship.target_claim_id)
        ):
            raise DomainInvariantError("supporting Claims cannot also have direct decisions")


@dataclass(frozen=True, slots=True)
class SemanticClaimWrite:
    """One complete Semantic Claim aggregate ready for atomic persistence."""

    claim: Claim
    source_claim_ids: tuple[ClaimId, ...]
    entity_ids: tuple[EntityId, ...]
    memory: MemoryRecord
    relations: tuple[Relation, ...]
    embedding: EmbeddingRecord

    def __post_init__(self) -> None:
        if not 2 <= len(self.source_claim_ids) <= 32 or len(set(self.source_claim_ids)) != len(
            self.source_claim_ids
        ):
            raise DomainInvariantError("Semantic Claim write requires unique source Claims")
        if self.claim.claim_id in self.source_claim_ids:
            raise DomainInvariantError("Semantic Claim cannot support itself")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise DomainInvariantError("Semantic Claim entity IDs must be unique")
        if (
            self.memory.tenant_id != self.claim.tenant_id
            or self.embedding.tenant_id != self.claim.tenant_id
            or any(relation.tenant_id != self.claim.tenant_id for relation in self.relations)
        ):
            raise DomainInvariantError("Semantic Claim records must belong to one tenant")
        if (
            self.claim.verification_status is not VerificationStatus.VERIFIED
            or self.memory.memory_type is not MemoryType.SEMANTIC
            or self.memory.summary != self.claim.statement
            or self.memory.evidence_ids != self.claim.evidence_ids
            or self.memory.occurred_at != self.claim.valid_from
            or self.memory.created_at != self.claim.created_at
            or self.memory.model_reference != self.claim.model_reference
            or self.memory.salience != self.claim.confidence
            or self.memory.verification_status is not VerificationStatus.VERIFIED
        ):
            raise DomainInvariantError("Semantic Claim memory must represent its complete Claim")
        if (
            self.embedding.object_type is not EmbeddedObjectType.CLAIM
            or self.embedding.object_id != self.claim.claim_id
            or self.embedding.task != EmbedTask.DOCUMENT.value
            or not self.embedding.normalized
            or self.embedding.created_at != self.claim.created_at
        ):
            raise DomainInvariantError("Semantic Claim embedding must index its Claim")
        expected_edges = _expected_semantic_claim_edges(self)
        if (
            len(self.relations) != len(expected_edges)
            or _relation_edges(self.relations) != expected_edges
        ):
            raise DomainInvariantError("Semantic Claim relations do not match its aggregate")


@dataclass(frozen=True, slots=True)
class ClaimConsolidationWrite:
    """Semantic Claim aggregates and direct Claim decisions from one model result."""

    semantic_claims: tuple[SemanticClaimWrite, ...]
    relationships: tuple[Relation, ...]

    def __post_init__(self) -> None:
        if any(
            relation.source_type is not RelationNodeType.CLAIM
            or relation.target_type is not RelationNodeType.CLAIM
            or relation.relation_type not in {RelationType.CONTRADICTS, RelationType.SUPERSEDES}
            for relation in self.relationships
        ):
            raise DomainInvariantError("direct Claim decisions must contradict or supersede")
        if len({relation.relation_id for relation in self.relationships}) != len(
            self.relationships
        ):
            raise DomainInvariantError("direct Claim decisions must be unique")


@dataclass(frozen=True, slots=True)
class ClaimConsolidationCommit:
    """Content-free counts from one atomic Claim persistence attempt."""

    semantic_claim_count: int
    relationship_count: int

    def __post_init__(self) -> None:
        if min(self.semantic_claim_count, self.relationship_count) < 0:
            raise DomainInvariantError("Claim commit counts must be non-negative")


async def derive_claim_consolidation_write(
    tenant_id: TenantId,
    candidates: tuple[ClaimCandidate, ...],
    consolidation: ClaimConsolidation,
    text_embedder: Embedder,
    created_at: datetime,
) -> ClaimConsolidationWrite:
    """Build deterministic Claim graph records and aligned text vectors."""
    candidate_by_id = {candidate.claim.claim_id: candidate for candidate in candidates}
    claims_and_entities = tuple(
        _semantic_claim(
            tenant_id,
            proposal,
            candidate_by_id,
            consolidation.model_reference,
            consolidation.prompt_version,
            created_at,
        )
        for proposal in consolidation.semantic_claims
    )
    result = await text_embedder.embed(
        EmbedRequest(
            inputs=tuple(
                ModelInput((TextPart(claim.statement),)) for claim, _, _ in claims_and_entities
            ),
            task=EmbedTask.DOCUMENT,
        )
    )
    if len(result.embeddings) != len(claims_and_entities):
        raise MemoryIntegrityError("embedder returned the wrong semantic claim vector count")
    semantic_claims = tuple(
        _semantic_claim_write(
            claim,
            proposal.source_claim_ids,
            entity_ids,
            memory_ended_at,
            embedding,
            created_at,
        )
        for (claim, entity_ids, memory_ended_at), proposal, embedding in zip(
            claims_and_entities,
            consolidation.semantic_claims,
            result.embeddings,
            strict=True,
        )
    )
    relationships = tuple(
        derive_relation(
            tenant_id,
            RelationNodeType.CLAIM,
            proposal.source_claim_id,
            proposal.relation_type,
            RelationNodeType.CLAIM,
            proposal.target_claim_id,
            created_at,
        )
        for proposal in consolidation.relationships
    )
    return ClaimConsolidationWrite(
        semantic_claims=semantic_claims,
        relationships=relationships,
    )


def _semantic_claim(
    tenant_id: TenantId,
    proposal: SemanticClaimProposal,
    candidate_by_id: dict[ClaimId, ClaimCandidate],
    model_reference: ModelReference,
    prompt_version: str,
    created_at: datetime,
) -> tuple[Claim, tuple[EntityId, ...], datetime]:
    sources = tuple(
        sorted(
            (candidate_by_id[claim_id] for claim_id in proposal.source_claim_ids),
            key=lambda candidate: (candidate.claim.valid_from, candidate.claim.claim_id),
        )
    )
    source_claims = tuple(candidate.claim for candidate in sources)
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_id for source in source_claims for evidence_id in source.evidence_ids
        )
    )
    entity_ids = tuple(sorted({entity_id for source in sources for entity_id in source.entity_ids}))
    memory_ended_at = max(source.valid_to or source.valid_from for source in source_claims)
    return (
        Claim(
            claim_id=ClaimId(
                derive_stable_id(
                    "semantic-claim",
                    tenant_id,
                    model_reference.model_id,
                    prompt_version,
                    created_at.isoformat(),
                    *sorted(str(claim_id) for claim_id in proposal.source_claim_ids),
                )
            ),
            tenant_id=tenant_id,
            claim_type=source_claims[0].claim_type,
            statement=proposal.statement,
            evidence_ids=evidence_ids,
            confidence=proposal.confidence,
            verification_status=VerificationStatus.VERIFIED,
            valid_from=min(source.valid_from for source in source_claims),
            valid_to=(
                None
                if any(source.valid_to is None for source in source_claims)
                else max(source.valid_to for source in source_claims if source.valid_to is not None)
            ),
            created_at=created_at,
            model_reference=model_reference,
            prompt_version=prompt_version,
        ),
        entity_ids,
        memory_ended_at,
    )


def _semantic_claim_write(
    claim: Claim,
    source_claim_ids: tuple[ClaimId, ...],
    entity_ids: tuple[EntityId, ...],
    memory_ended_at: datetime,
    embedding: Embedding,
    created_at: datetime,
) -> SemanticClaimWrite:
    memory = MemoryRecord(
        memory_id=MemoryId(derive_stable_id("memory", claim.claim_id)),
        tenant_id=claim.tenant_id,
        memory_type=MemoryType.SEMANTIC,
        summary=claim.statement,
        evidence_ids=claim.evidence_ids,
        occurred_at=claim.valid_from,
        ended_at=memory_ended_at,
        created_at=created_at,
        verification_status=VerificationStatus.VERIFIED,
        model_reference=claim.model_reference,
        salience=claim.confidence,
        strength=claim.confidence,
    )
    relations = (
        derive_relation(
            claim.tenant_id,
            RelationNodeType.CLAIM,
            claim.claim_id,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
            memory.memory_id,
            created_at,
        ),
        *(
            derive_relation(
                claim.tenant_id,
                RelationNodeType.CLAIM,
                source_claim_id,
                RelationType.SUPPORTS,
                RelationNodeType.CLAIM,
                claim.claim_id,
                created_at,
            )
            for source_claim_id in source_claim_ids
        ),
        *(
            derive_relation(
                claim.tenant_id,
                RelationNodeType.CLAIM,
                claim.claim_id,
                RelationType.ABOUT,
                RelationNodeType.ENTITY,
                entity_id,
                created_at,
            )
            for entity_id in entity_ids
        ),
    )
    return SemanticClaimWrite(
        claim=claim,
        source_claim_ids=source_claim_ids,
        entity_ids=entity_ids,
        memory=memory,
        relations=relations,
        embedding=EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    claim.tenant_id,
                    EmbeddedObjectType.CLAIM.value,
                    claim.claim_id,
                    embedding.model_reference.model_id,
                    EmbedTask.DOCUMENT.value,
                )
            ),
            tenant_id=claim.tenant_id,
            object_type=EmbeddedObjectType.CLAIM,
            object_id=claim.claim_id,
            values=embedding.values,
            model_reference=embedding.model_reference,
            space_reference=embedding.space_reference,
            task=EmbedTask.DOCUMENT.value,
            dimension=embedding.dimension,
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


def _expected_semantic_claim_edges(
    write: SemanticClaimWrite,
) -> set[tuple[RelationNodeType, str, RelationType, RelationNodeType, str]]:
    return {
        (
            RelationNodeType.CLAIM,
            write.claim.claim_id,
            RelationType.REPRESENTED_BY,
            RelationNodeType.MEMORY_RECORD,
            write.memory.memory_id,
        ),
        *(
            (
                RelationNodeType.CLAIM,
                source_claim_id,
                RelationType.SUPPORTS,
                RelationNodeType.CLAIM,
                write.claim.claim_id,
            )
            for source_claim_id in write.source_claim_ids
        ),
        *(
            (
                RelationNodeType.CLAIM,
                write.claim.claim_id,
                RelationType.ABOUT,
                RelationNodeType.ENTITY,
                entity_id,
            )
            for entity_id in write.entity_ids
        ),
    }
