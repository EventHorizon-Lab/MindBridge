"""Typed inputs and boundaries for multimodal candidate recall."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from mindbridge.application.enumeration import EnumerateMemories
from mindbridge.application.evidence import read_resolved_memory_evidence, sign_query_media
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    EmbeddingIndex,
    EmbeddingSearch,
    MediaUrlSigner,
    MemoryAnswerer,
    MemoryStore,
    ResolvedQueryMedia,
)
from mindbridge.application.ranking import fuse_memory_rankings
from mindbridge.contracts import (
    EvidenceView,
    MemoryResult,
    MemoryView,
    RecallMode,
    RecallRequest,
    RecallResult,
)
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    EvidenceId,
    MediaObjectId,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    ModelReference,
    TenantId,
    VerificationStatus,
)
from mindbridge.telemetry import current_trace_id, set_current_span_attributes, trace_operation

RETRIEVAL_DOCUMENT_EMBEDDING_TASK = "retrieval_document"


@dataclass(frozen=True, slots=True)
class RecallEmbeddingQuery:
    """Text and original AV fused into one retrieval-side embedding input."""

    text: str | None
    media: tuple[ResolvedQueryMedia, ...]

    def __post_init__(self) -> None:
        if self.text is not None and not self.text.strip():
            raise DomainInvariantError("embedding query text must not be blank")
        if self.text is None and not self.media:
            raise DomainInvariantError("embedding query requires text or media")
        media_ids = [item.media_object.media_object_id for item in self.media]
        if len(set(media_ids)) != len(media_ids):
            raise DomainInvariantError("embedding query media IDs must be unique")
        if len({item.media_object.tenant_id for item in self.media}) > 1:
            raise DomainInvariantError("embedding query media must belong to one tenant")


class RecallEmbedder(Protocol):
    """Frozen encoder shared by recall queries and explicit memory documents."""

    @property
    def query_model_reference(self) -> ModelReference: ...

    @property
    def document_model_reference(self) -> ModelReference: ...

    @property
    def space_reference(self) -> EmbeddingSpaceReference: ...

    @property
    def dimension(self) -> int: ...

    async def encode_query(self, query: RecallEmbeddingQuery) -> tuple[float, ...]: ...

    async def encode_memory_document(self, text: str) -> tuple[float, ...]: ...


class RecallMemories:
    """Retrieve and inspect evidence through one shared multimodal recall path."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: MemoryAnswerer,
        *,
        embedding_index: EmbeddingIndex,
        media_url_signer: MediaUrlSigner,
        recall_embedder: RecallEmbedder,
        minimum_embedding_similarity: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._answerer = answerer
        self._embedding_index = embedding_index
        self._media_url_signer = media_url_signer
        self._recall_embedder = recall_embedder
        self._minimum_embedding_similarity = minimum_embedding_similarity
        self._clock = clock or _utc_now
        self._enumerate = EnumerateMemories(
            store,
            answerer,
            media_url_signer,
            clock=self._clock,
        )

    @trace_operation("mindbridge.recall")
    async def run(self, request: RecallRequest) -> RecallResult:
        """Retrieve memories, inspect evidence, and answer only when supported."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.recall.mode": request.mode.value,
                "mindbridge.recall.limit": request.limit,
                "mindbridge.query.has_text": request.query.text is not None,
                "mindbridge.query.media_count": len(request.query.media_object_ids),
                "mindbridge.query.memory_count": len(request.memory_ids),
            }
        )
        query_media = await self._resolve_query_media(request)
        if request.mode is RecallMode.ENUMERATE:
            return await self._enumerate_result(request, query_media)
        candidate_limit = min(request.limit * 4, 100)
        memories = await self._retrieve_candidates(
            request,
            query_media,
            limit=candidate_limit,
        )
        visible_candidates = memories[: request.limit]
        visible_memories = await self._store.record_memory_accesses(
            TenantId(request.tenant_id),
            tuple(memory.memory_id for memory in visible_candidates),
            accessed_at=self._clock(),
        )
        answer_memories = tuple(
            memory
            for memory in visible_memories
            if memory.evidence_ids or memory.verification_status is VerificationStatus.ATTESTED
        )
        should_answer = bool(answer_memories) and request.mode is not RecallMode.SEARCH
        evidence_memories = answer_memories if should_answer else visible_memories
        should_read_evidence = request.include_evidence or should_answer
        evidence = (
            await self._read_evidence(request, evidence_memories) if should_read_evidence else ()
        )
        response_evidence = evidence
        answer = None
        confidence = 0.0
        if should_answer:
            answer_query_media = await sign_query_media(
                tuple(item.media_object for item in query_media),
                self._media_url_signer,
            )
            generated = await self._answerer.answer(
                request,
                answer_memories,
                evidence,
                query_media=answer_query_media,
            )
            answer = generated.answer
            confidence = generated.confidence
            if request.include_evidence:
                response_evidence = await self._read_evidence(request, answer_memories)
        set_current_span_attributes(
            {
                "mindbridge.recall.candidate_count": len(memories),
                "mindbridge.recall.memory_count": len(visible_memories),
                "mindbridge.recall.evidence_count": len(evidence),
                "mindbridge.recall.answered": answer is not None,
            }
        )
        visible_evidence_ids = {
            evidence_id for memory in visible_memories for evidence_id in memory.evidence_ids
        }
        return RecallResult(
            answer=answer,
            confidence=confidence,
            memories=tuple(memory_view(memory) for memory in visible_memories),
            evidence=(
                tuple(
                    evidence_view(item)
                    for item in response_evidence
                    if item.evidence_span.evidence_id in visible_evidence_ids
                )
                if request.include_evidence
                else ()
            ),
            trace_id=current_trace_id(),
        )

    async def _enumerate_result(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> RecallResult:
        enumeration = await self._enumerate.run(request, query_media)
        set_current_span_attributes(
            {
                "mindbridge.recall.candidate_count": len(enumeration.memories),
                "mindbridge.recall.memory_count": len(enumeration.memories),
                "mindbridge.recall.evidence_count": len(enumeration.evidence),
                "mindbridge.recall.answered": True,
            }
        )
        return RecallResult(
            answer=str(len(enumeration.memories)),
            confidence=1.0,
            memories=tuple(memory_view(memory) for memory in enumeration.memories),
            evidence=(
                tuple(evidence_view(item) for item in enumeration.evidence)
                if request.include_evidence
                else ()
            ),
            trace_id=current_trace_id(),
        )

    async def _retrieve_candidates(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        if request.memory_ids:
            return await self._store.search_memories_by_ids(
                request,
                tuple(MemoryId(memory_id) for memory_id in request.memory_ids),
                limit=limit,
            )
        semantic_search = self._search_semantic_memories(
            request,
            query_media,
            limit=limit,
        )
        if request.query.text is None:
            memories = await semantic_search
        else:
            sparse, semantic = await asyncio.gather(
                self._store.search_memories(request, limit=limit),
                semantic_search,
            )
            memories = fuse_memory_rankings((semantic, sparse), limit=limit)
        return memories[:limit]

    @trace_operation("mindbridge.recall.semantic_search")
    async def _search_semantic_memories(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        query = RecallEmbeddingQuery(
            text=request.query.text,
            media=query_media,
        )
        values = await self._recall_embedder.encode_query(query)
        searches = {
            object_type: EmbeddingSearch(
                tenant_id=TenantId(request.tenant_id),
                values=values,
                space_reference=self._recall_embedder.space_reference,
                document_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                object_types=(object_type,),
                limit=limit,
                minimum_similarity=self._minimum_embedding_similarity,
            )
            for object_type in (
                EmbeddedObjectType.EVIDENCE_SPAN,
                EmbeddedObjectType.MEMORY_RECORD,
            )
        }
        graph_search = EmbeddingSearch(
            tenant_id=TenantId(request.tenant_id),
            values=values,
            space_reference=self._recall_embedder.space_reference,
            document_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
            object_types=(EmbeddedObjectType.EVENT, EmbeddedObjectType.CLAIM),
            limit=limit,
            minimum_similarity=self._minimum_embedding_similarity,
        )
        evidence_matches, memory_matches, graph_matches = await asyncio.gather(
            self._embedding_index.search_embeddings(searches[EmbeddedObjectType.EVIDENCE_SPAN]),
            self._embedding_index.search_embeddings(searches[EmbeddedObjectType.MEMORY_RECORD]),
            self._embedding_index.search_embeddings(graph_search),
        )
        evidence_ids = tuple(
            dict.fromkeys(EvidenceId(match.object_id) for match in evidence_matches)
        )
        memory_ids = tuple(dict.fromkeys(MemoryId(match.object_id) for match in memory_matches))
        (
            evidence_memories,
            direct_memories,
            hierarchy_memories,
            graph_memories,
        ) = await asyncio.gather(
            self._store.search_memories_by_evidence(request, evidence_ids, limit=limit),
            self._store.search_memories_by_ids(request, memory_ids, limit=limit),
            self._store.search_memories_by_hierarchy(request, memory_ids, limit=limit),
            self._store.search_memories_by_graph_objects(
                request,
                graph_matches,
                limit=limit,
            ),
        )
        set_current_span_attributes(
            {
                "mindbridge.recall.evidence_match_count": len(evidence_matches),
                "mindbridge.recall.memory_match_count": len(memory_matches),
                "mindbridge.recall.graph_match_count": len(graph_matches),
            }
        )
        return fuse_memory_rankings(
            (evidence_memories, graph_memories, direct_memories, hierarchy_memories),
            limit=limit,
        )

    @trace_operation("mindbridge.recall.resolve_query_media")
    async def _resolve_query_media(
        self,
        request: RecallRequest,
    ) -> tuple[ResolvedQueryMedia, ...]:
        requested_ids = tuple(MediaObjectId(value) for value in request.query.media_object_ids)
        if not requested_ids:
            return ()
        tenant_id = TenantId(request.tenant_id)
        media_objects = await self._store.read_media_objects(tenant_id, requested_ids)
        if any(item.tenant_id != tenant_id for item in media_objects):
            raise MemoryIntegrityError("media store returned a cross-tenant query object")
        media_by_id = {item.media_object_id: item for item in media_objects}
        if len(media_by_id) != len(requested_ids) or set(media_by_id) != set(requested_ids):
            raise DomainInvariantError("recall query references unknown media")
        return await sign_query_media(
            tuple(media_by_id[media_object_id] for media_object_id in requested_ids),
            self._media_url_signer,
        )

    @trace_operation("mindbridge.recall.resolve_evidence")
    async def _read_evidence(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[ResolvedEvidence, ...]:
        return await read_resolved_memory_evidence(
            self._store,
            self._media_url_signer,
            TenantId(request.tenant_id),
            memories,
        )


def memory_view(memory: MemoryRecord) -> MemoryView:
    """Convert one domain memory to its stable public representation."""
    return MemoryView(
        memory_id=memory.memory_id,
        memory_type=memory.memory_type,
        summary=memory.summary,
        evidence_ids=memory.evidence_ids,
        occurred_at=memory.occurred_at,
        ended_at=memory.ended_at,
        created_at=memory.created_at,
        verification_status=memory.verification_status,
        state=memory.state,
        salience=memory.salience,
        strength=memory.strength,
        useful_access_count=memory.useful_access_count,
        positive_feedback_count=memory.positive_feedback_count,
        negative_feedback_count=memory.negative_feedback_count,
        last_accessed_at=memory.last_accessed_at,
        supersedes_memory_id=memory.supersedes_memory_id,
        superseded_at=memory.superseded_at,
    )


def memory_result(
    memory: MemoryRecord,
    evidence: tuple[ResolvedEvidence, ...] = (),
) -> MemoryResult:
    """Expose one top-level memory with its inspectable evidence and request trace."""
    return MemoryResult.model_validate(
        memory_view(memory).model_dump()
        | {
            "evidence": tuple(evidence_view(item) for item in evidence),
            "trace_id": current_trace_id(),
        }
    )


def evidence_view(evidence: ResolvedEvidence) -> EvidenceView:
    """Convert resolved evidence without leaking object-store implementation details."""
    return EvidenceView(
        evidence_id=evidence.evidence_span.evidence_id,
        media_object_id=evidence.media_object.media_object_id,
        start_ms=evidence.evidence_span.start_ms,
        end_ms=evidence.evidence_span.end_ms,
        media_url=evidence.media_url,
        media_url_expires_at=evidence.media_url_expires_at,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
