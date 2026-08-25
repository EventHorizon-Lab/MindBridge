"""Checks that what consolidation writes is reachable from recall, not just from the store.

A summary is worth scheduling only if a query can come back with it instead of the individual
moments underneath it. The parent here is deliberately the *worst* vector match in the tenant, so
the only way it can reach the fused ranking is the `contains` edge the Summary sweep wrote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import cast

from mindbridge.application.capabilities import (
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
)
from mindbridge.application.ports import (
    Answerer,
    EmbeddingIndex,
    EmbeddingMatch,
    EmbeddingSearch,
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
)
from mindbridge.application.recall import RecallMemories
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    EvidenceId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
TENANT = TenantId("tenant_hierarchy")
MODEL = ModelReference(model_id="qwen3.8-max")
EMBEDDING_MODEL = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
SPACE = EmbeddingSpaceReference(space_id="jina-v5")
QUERY_VECTOR = (1.0, 0.0)
CHILD_VECTOR = (0.95, math.sqrt(1.0 - 0.9025))
# Orthogonal to the query: the parent's own text matches nothing the query asked for, which is
# the ordinary case for a summary written over a session rather than over one phrasing.
PARENT_VECTOR = (0.0, 1.0)

_SUMMARY = "Across the session, a person kept a red tool beside a blue toolbox."


@dataclass(frozen=True, slots=True)
class _FixedEmbedder:
    """Answer every query with the one vector this test ranks against."""

    model_reference: ModelReference = EMBEDDING_MODEL
    space_reference: EmbeddingSpaceReference = SPACE

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        return EmbedResult(
            embeddings=tuple(
                Embedding(
                    values=QUERY_VECTOR,
                    model_reference=EMBEDDING_MODEL,
                    space_reference=SPACE,
                )
                for _ in request.inputs
            )
        )


class _FakeEmbeddingIndex:
    """Rank the indexed vectors by cosine, the way pgvector does behind the store."""

    def __init__(self, embeddings: tuple[EmbeddingRecord, ...]) -> None:
        self._embeddings = embeddings

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        matches = [
            EmbeddingMatch(
                embedding_id=embedding.embedding_id,
                object_type=embedding.object_type,
                object_id=embedding.object_id,
                similarity=sum(
                    left * right
                    for left, right in zip(search.values, embedding.values, strict=True)
                ),
            )
            for embedding in self._embeddings
            if embedding.object_type in search.object_types
        ]
        ranked = sorted(
            (match for match in matches if match.similarity >= search.minimum_similarity),
            key=lambda match: -match.similarity,
        )
        return tuple(ranked[: search.limit])


class _FakeMemoryStore:
    """Only the channels one text query without media reaches."""

    def __init__(
        self,
        memories: tuple[MemoryRecord, ...],
        contains: dict[MemoryId, tuple[MemoryId, ...]],
    ) -> None:
        self._memories = {memory.memory_id: memory for memory in memories}
        self._contains = contains
        self.hierarchy_requests: list[tuple[MemoryId, ...]] = []

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return ()

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return self._resolve(ranked_memory_ids, limit=limit)

    async def search_memories_by_hierarchy(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        self.hierarchy_requests.append(ranked_memory_ids)
        # The recursive query seeds itself with the matched memories at depth 0 and then adds
        # the parents that `contains` them, which is the only edge the Summary sweep writes.
        parents = tuple(
            parent
            for parent, children in self._contains.items()
            if any(memory_id in children for memory_id in ranked_memory_ids)
        )
        return self._resolve(ranked_memory_ids + parents, limit=limit)

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return ()

    def _resolve(
        self,
        memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        return tuple(
            self._memories[memory_id]
            for memory_id in dict.fromkeys(memory_ids)
            if memory_id in self._memories
        )[:limit]


async def test_recall_returns_the_consolidated_parent_a_query_never_matched() -> None:
    """The layer above a clip is only worth writing if a query can come back with it."""
    children = tuple(_memory(f"memory_child_{index:02d}", index) for index in range(2))
    parent = _memory("memory_summary_01", 0, summary=_SUMMARY, memory_type=MemoryType.SEMANTIC)
    store = _FakeMemoryStore(
        (*children, parent),
        {parent.memory_id: tuple(child.memory_id for child in children)},
    )
    indexed = (
        *(_embedding(memory, CHILD_VECTOR) for memory in children),
        _embedding(parent, PARENT_VECTOR),
    )
    recall = _recall(store, indexed)

    fused = await recall._search_semantic_memories(
        RecallRequest(tenant_id=TENANT, query=RecallQuery(text="what happened with the tool")),
        (),
        limit=10,
    )

    # Asserted before the ranking so the check below cannot pass vacuously: recall has to
    # hand the vector-matched memories to the hierarchy channel for it to expand anything.
    assert store.hierarchy_requests == [tuple(child.memory_id for child in children)]
    # The parent's own vector is orthogonal to the query and the index is asked for a floor of
    # 0.1, so it cannot have arrived through the vector channel, and the evidence and graph
    # channels return nothing: the `contains` edge is the only path left.
    assert parent.memory_id in {memory.memory_id for memory in fused}
    assert {memory.memory_id for memory in fused} >= {child.memory_id for child in children}


def _recall(store: _FakeMemoryStore, embeddings: tuple[EmbeddingRecord, ...]) -> RecallMemories:
    return RecallMemories(
        cast(MemoryStore, store),
        cast(Answerer, None),
        cast(OccurrenceVerifier, None),
        embedding_index=cast(EmbeddingIndex, _FakeEmbeddingIndex(embeddings)),
        media_url_signer=cast(MediaUrlSigner, None),
        embedder=cast(Embedder, _FixedEmbedder()),
        minimum_embedding_similarity=0.1,
    )


def _memory(
    memory_id: str,
    ordinal: int,
    *,
    summary: str = "A person places a red tool beside a blue toolbox.",
    memory_type: MemoryType = MemoryType.EPISODIC,
) -> MemoryRecord:
    occurred_at = NOW + timedelta(seconds=ordinal)
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=TENANT,
        memory_type=memory_type,
        summary=summary,
        evidence_ids=(EvidenceId("evidence_01"),),
        occurred_at=occurred_at,
        ended_at=occurred_at + timedelta(seconds=1),
        created_at=NOW,
        verification_status=VerificationStatus.VERIFIED,
        model_reference=MODEL,
        salience=0.8,
        strength=0.8,
    )


def _embedding(memory: MemoryRecord, values: tuple[float, ...]) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(f"embedding_{memory.memory_id}"),
        tenant_id=TENANT,
        object_type=EmbeddedObjectType.MEMORY_RECORD,
        object_id=memory.memory_id,
        values=values,
        model_reference=EMBEDDING_MODEL,
        space_reference=SPACE,
        task=EmbedTask.DOCUMENT.value,
        dimension=len(values),
        normalized=True,
        created_at=NOW,
    )
