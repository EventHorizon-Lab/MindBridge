"""Typed inputs and boundaries for multimodal candidate recall."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter
from typing import Literal

from mindbridge.application.capabilities import (
    Embedder,
    EmbedRequest,
    EmbedTask,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.application.enumeration import EnumerateMemories
from mindbridge.application.evidence import read_resolved_memory_evidence, sign_query_media
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    Answerer,
    EmbeddingIndex,
    EmbeddingSearch,
    GeneratedAnswer,
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
    ResolvedQueryMedia,
)
from mindbridge.application.ranking import fuse_memory_rankings, order_memory_candidates
from mindbridge.contracts import (
    EvidenceView,
    MemoryResult,
    MemoryView,
    RecallFilters,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RecallResult,
)
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EvidenceId,
    MediaObjectId,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    ModelOutputError,
    TenantId,
    VerificationStatus,
    utc_now,
)
from mindbridge.telemetry import (
    current_trace_id,
    operation_span,
    record_stage_duration,
    set_current_span_attributes,
)

_MAXIMUM_RETRIEVAL_REFINEMENTS = 2
_UNANSWERED = GeneratedAnswer(answer=None, confidence=0.0)


@dataclass(frozen=True, slots=True)
class _AnswerWave:
    """One candidate ranking and the most recent answer produced from it."""

    ranked: tuple[MemoryRecord, ...]
    visible: tuple[MemoryRecord, ...]
    generated: GeneratedAnswer = _UNANSWERED
    evidence: tuple[ResolvedEvidence, ...] = ()
    attempted_queries: tuple[str, ...] = ()
    round_count: int = 0

    @property
    def temporal_order(self) -> Literal["relevance", "newest", "oldest"]:
        """Follow the newest answer, because later rounds see more evidence."""
        return self.generated.temporal_order


class RecallMemories:
    """Retrieve and inspect evidence through one shared multimodal recall path."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: Answerer,
        occurrence_verifier: OccurrenceVerifier,
        *,
        embedding_index: EmbeddingIndex,
        media_url_signer: MediaUrlSigner,
        embedder: Embedder,
        minimum_embedding_similarity: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._answerer = answerer
        self._embedding_index = embedding_index
        self._media_url_signer = media_url_signer
        self._embedder = embedder
        self._minimum_embedding_similarity = minimum_embedding_similarity
        self._clock = clock or utc_now
        self._enumerate = EnumerateMemories(
            store,
            occurrence_verifier,
            media_url_signer,
            clock=self._clock,
        )

    @operation_span("mindbridge.recall")
    async def run(self, request: RecallRequest) -> RecallResult:
        """Retrieve memories, inspect evidence, and answer only when supported."""
        started_at = perf_counter()
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.recall.mode": request.mode.value,
                "mindbridge.recall.limit": request.limit,
                "mindbridge.query.has_text": request.query.text is not None,
                "mindbridge.query.media_count": len(request.query.media_object_ids),
                "mindbridge.query.memory_count": len(request.memory_ids),
                # Completes the query-shape dimensions, so feedback joined on this trace's id can
                # be grouped by shape. Presence only: the filter values are caller content.
                "mindbridge.query.has_filters": request.filters != RecallFilters(),
            }
        )
        query_media = await self._resolve_query_media(request)
        if request.mode is RecallMode.ENUMERATE:
            return await self._enumerate_result(request, query_media)
        candidate_limit = min(request.limit * 4, 100)
        ranked = await self._retrieve_candidates(request, query_media, limit=candidate_limit)
        wave = _AnswerWave(
            ranked=ranked,
            visible=await self._refresh_visible_candidates(request, ranked),
        )
        if request.mode is not RecallMode.SEARCH:
            wave = await self._answer_within_reflection_budget(
                request,
                wave,
                query_media,
                candidate_limit=candidate_limit,
                started_at=started_at,
            )
        return await self._build_result(request, wave, started_at=started_at)

    async def _answer_within_reflection_budget(
        self,
        request: RecallRequest,
        wave: _AnswerWave,
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        candidate_limit: int,
        started_at: float,
    ) -> _AnswerWave:
        """Answer, then spend a bounded budget on temporal and retrieval reflection."""
        wave = await self._answer_round(request, wave, query_media, phase="initial")
        record_stage_duration("recall.first_answer", max(0.0, perf_counter() - started_at))
        reordered = order_memory_candidates(wave.visible, wave.temporal_order)
        if _memory_ids(reordered) != _memory_ids(wave.visible):
            wave = await self._answer_round(
                request,
                replace(wave, visible=reordered),
                query_media,
                phase="temporal_reorder",
            )
        seen_queries = {request.query.text.casefold()} if request.query.text is not None else set()
        for _ in range(_MAXIMUM_RETRIEVAL_REFINEMENTS):
            followup_queries = tuple(
                query
                for query in wave.generated.retrieval_queries
                if query.casefold() not in seen_queries
            )
            if not followup_queries or request.memory_ids:
                break
            seen_queries.update(query.casefold() for query in followup_queries)
            wave = replace(wave, attempted_queries=wave.attempted_queries + followup_queries)
            query_media = await self._resign_query_media(query_media)
            wave = await self._answer_round(
                request,
                await self._merge_followup_candidates(
                    request,
                    wave,
                    query_media,
                    followup_queries,
                    limit=candidate_limit,
                ),
                query_media,
                phase="reflection",
            )
        return wave

    async def _answer_round(
        self,
        request: RecallRequest,
        wave: _AnswerWave,
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        phase: Literal["initial", "temporal_reorder", "reflection"],
    ) -> _AnswerWave:
        """Inspect the current candidates once and adopt that round's temporal intent."""
        generated, evidence = await self._answer_candidates(
            request,
            wave.visible,
            query_media,
            attempted_retrieval_queries=wave.attempted_queries,
            phase=phase,
            round_number=wave.round_count + 1,
        )
        return replace(
            wave,
            generated=generated,
            evidence=evidence,
            round_count=wave.round_count + 1,
        )

    async def _merge_followup_candidates(
        self,
        request: RecallRequest,
        wave: _AnswerWave,
        query_media: tuple[ResolvedQueryMedia, ...],
        followup_queries: tuple[str, ...],
        *,
        limit: int,
    ) -> _AnswerWave:
        """Fuse extra retrieval waves into the current ranking without losing relevance order."""
        followup_rankings = await asyncio.gather(
            *(
                self._retrieve_candidates(
                    request.model_copy(
                        update={
                            "query": RecallQuery(
                                text=query,
                                media_object_ids=request.query.media_object_ids,
                            )
                        }
                    ),
                    query_media,
                    limit=limit,
                )
                for query in followup_queries
            )
        )
        refined = fuse_memory_rankings((*followup_rankings, wave.ranked), limit=limit)
        visible = order_memory_candidates(
            await self._refresh_visible_candidates(request, refined),
            wave.temporal_order,
        )
        if _memory_ids(visible) == _memory_ids(wave.visible):
            return wave
        return replace(wave, ranked=refined, visible=visible)

    async def _build_result(
        self,
        request: RecallRequest,
        wave: _AnswerWave,
        *,
        started_at: float,
    ) -> RecallResult:
        """Charge the recall against memory strength and return only still-visible content."""
        visible_memories = await self._store.record_memory_accesses(
            TenantId(request.tenant_id),
            _memory_ids(wave.visible),
            accessed_at=self._clock(),
        )
        generated = wave.generated
        answer_evidence = wave.evidence
        if _memory_ids(visible_memories) != _memory_ids(wave.visible):
            generated = _UNANSWERED
            answer_evidence = ()
        answer_memories = self._grounded_memories(visible_memories)
        response_evidence = answer_evidence
        if request.include_evidence:
            response_evidence = await self._read_evidence(
                request,
                answer_memories if request.mode is not RecallMode.SEARCH else visible_memories,
            )
        set_current_span_attributes(
            {
                "mindbridge.recall.candidate_count": len(wave.ranked),
                "mindbridge.recall.memory_count": len(visible_memories),
                "mindbridge.recall.evidence_count": len(answer_evidence),
                "mindbridge.recall.retrieval_query_count": len(wave.attempted_queries),
                "mindbridge.recall.retrieval_round_count": wave.round_count,
                "mindbridge.recall.answered": generated.answer is not None,
            }
        )
        visible_evidence_ids = {
            evidence_id for memory in visible_memories for evidence_id in memory.evidence_ids
        }
        result = RecallResult(
            answer=generated.answer,
            confidence=generated.confidence,
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
        record_stage_duration(
            "recall.search" if request.mode is RecallMode.SEARCH else "recall.answer_complete",
            max(0.0, perf_counter() - started_at),
        )
        return result

    @operation_span("mindbridge.recall.answer_round")
    async def _answer_candidates(
        self,
        request: RecallRequest,
        candidates: tuple[MemoryRecord, ...],
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        attempted_retrieval_queries: tuple[str, ...] = (),
        phase: Literal["initial", "temporal_reorder", "reflection"],
        round_number: int,
    ) -> tuple[GeneratedAnswer, tuple[ResolvedEvidence, ...]]:
        set_current_span_attributes(
            {
                "mindbridge.recall.answer.phase": phase,
                "mindbridge.recall.answer.round": round_number,
            }
        )
        grounded = self._grounded_memories(candidates)
        evidence = await self._read_evidence(request, grounded) if grounded else ()
        answer_query_media = await sign_query_media(
            tuple(item.media_object for item in query_media),
            self._media_url_signer,
        )
        generated = await self._answerer.answer(
            request,
            grounded,
            evidence,
            query_media=answer_query_media,
            attempted_retrieval_queries=attempted_retrieval_queries,
        )
        if not grounded and not query_media and generated.answer is not None:
            generated = GeneratedAnswer(
                answer=None,
                confidence=0.0,
                retrieval_queries=generated.retrieval_queries,
                temporal_order=generated.temporal_order,
            )
        return generated, evidence

    async def _refresh_visible_candidates(
        self,
        request: RecallRequest,
        candidates: tuple[MemoryRecord, ...],
    ) -> tuple[MemoryRecord, ...]:
        """Reapply deletion, supersession, and caller filters immediately before answering."""
        return await self._store.search_memories_by_ids(
            request,
            tuple(memory.memory_id for memory in candidates),
            limit=request.limit,
        )

    async def _resign_query_media(
        self,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[ResolvedQueryMedia, ...]:
        """Refresh short-lived media URLs before a delayed retrieval wave."""
        return (
            await sign_query_media(
                tuple(item.media_object for item in query_media),
                self._media_url_signer,
            )
            if query_media
            else ()
        )

    @staticmethod
    def _grounded_memories(memories: tuple[MemoryRecord, ...]) -> tuple[MemoryRecord, ...]:
        return tuple(
            memory
            for memory in memories
            if memory.evidence_ids or memory.verification_status is VerificationStatus.ATTESTED
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

    @operation_span("mindbridge.recall.semantic_search")
    async def _search_semantic_memories(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        embedded = await self._embedder.embed(
            EmbedRequest(
                inputs=(_query_input(request, query_media),),
                task=EmbedTask.QUERY,
            )
        )
        if len(embedded.embeddings) != 1:
            raise ModelOutputError("embedder returned the wrong query vector count")
        query_embedding = embedded.embeddings[0]
        searches = {
            object_type: EmbeddingSearch(
                tenant_id=TenantId(request.tenant_id),
                values=query_embedding.values,
                space_reference=query_embedding.space_reference,
                document_task=EmbedTask.DOCUMENT.value,
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
            values=query_embedding.values,
            space_reference=query_embedding.space_reference,
            document_task=EmbedTask.DOCUMENT.value,
            object_types=(
                EmbeddedObjectType.EVENT,
                EmbeddedObjectType.CLAIM,
                EmbeddedObjectType.ENTITY,
            ),
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

    @operation_span("mindbridge.recall.resolve_query_media")
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

    @operation_span("mindbridge.recall.resolve_evidence")
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


def _memory_ids(memories: tuple[MemoryRecord, ...]) -> tuple[MemoryId, ...]:
    return tuple(memory.memory_id for memory in memories)


def _query_input(
    request: RecallRequest,
    query_media: tuple[ResolvedQueryMedia, ...],
) -> ModelInput:
    parts = ((TextPart(request.query.text),) if request.query.text is not None else ()) + tuple(
        MediaPart(
            kind=item.media_object.kind,
            url=item.media_url,
            source_uri=item.media_object.uri,
        )
        for item in query_media
    )
    return ModelInput(parts)


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
