"""Exhaustive, evidence-verified occurrence enumeration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from mindbridge.application.evidence import read_resolved_memory_evidence, sign_query_media
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    MediaUrlSigner,
    MemoryStore,
    OccurrenceVerifier,
    ResolvedQueryMedia,
)
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    EnumerationLimitExceededError,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelOutputError,
    TenantId,
    VerificationStatus,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation

ENUMERATION_CANDIDATE_LIMIT = 1_000
ENUMERATION_BATCH_SIZE = 16
ENUMERATION_MAX_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    """Chronological verified occurrences and their resolved evidence."""

    memories: tuple[MemoryRecord, ...]
    evidence: tuple[ResolvedEvidence, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedBatch:
    memories: tuple[MemoryRecord, ...]


class EnumerateMemories:
    """Scan a bounded filter scope and ask the selected model to verify every candidate."""

    def __init__(
        self,
        store: MemoryStore,
        verifier: OccurrenceVerifier,
        media_url_signer: MediaUrlSigner,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._verifier = verifier
        self._media_url_signer = media_url_signer
        self._clock = clock

    @trace_operation("mindbridge.recall.enumerate")
    async def run(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> EnumerationResult:
        candidates = (
            await self._store.search_memories_by_ids(
                request,
                tuple(MemoryId(memory_id) for memory_id in request.memory_ids),
                limit=len(request.memory_ids),
            )
            if request.memory_ids
            else await self._store.list_memories_for_enumeration(
                request,
                limit=ENUMERATION_CANDIDATE_LIMIT + 1,
            )
        )
        candidates = tuple(
            sorted(
                (memory for memory in candidates if memory.memory_type is MemoryType.EPISODIC),
                key=lambda memory: (memory.occurred_at, memory.memory_id),
            )
        )
        if len(candidates) > ENUMERATION_CANDIDATE_LIMIT:
            raise EnumerationLimitExceededError(
                "exact enumeration exceeds 1000 candidates; narrow the recall filters"
            )
        verifiable = tuple(
            memory
            for memory in candidates
            if memory.evidence_ids or memory.verification_status is VerificationStatus.ATTESTED
        )
        verified_batches = await self._verify_batches(request, query_media, verifiable)
        selected = tuple(memory for batch in verified_batches for memory in batch.memories)
        accessed = await self._store.record_memory_accesses(
            TenantId(request.tenant_id),
            tuple(memory.memory_id for memory in selected),
            accessed_at=self._clock(),
        )
        evidence = (
            await read_resolved_memory_evidence(
                self._store,
                self._media_url_signer,
                TenantId(request.tenant_id),
                accessed,
            )
            if request.include_evidence
            else ()
        )
        set_current_span_attributes(
            {
                "mindbridge.enumeration.candidate_count": len(candidates),
                "mindbridge.enumeration.verifiable_count": len(verifiable),
                "mindbridge.enumeration.occurrence_count": len(accessed),
            }
        )
        return EnumerationResult(
            memories=accessed,
            evidence=evidence,
        )

    async def _verify_batches(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[_VerifiedBatch, ...]:
        batches = tuple(
            memories[offset : offset + ENUMERATION_BATCH_SIZE]
            for offset in range(0, len(memories), ENUMERATION_BATCH_SIZE)
        )
        verified: list[_VerifiedBatch] = []
        for offset in range(0, len(batches), ENUMERATION_MAX_CONCURRENCY):
            batch_query_media = await sign_query_media(
                tuple(item.media_object for item in query_media),
                self._media_url_signer,
            )
            verified.extend(
                await asyncio.gather(
                    *(
                        self._verify_batch(request, batch_query_media, batch)
                        for batch in batches[offset : offset + ENUMERATION_MAX_CONCURRENCY]
                    )
                )
            )
        return tuple(verified)

    async def _verify_batch(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
        memories: tuple[MemoryRecord, ...],
    ) -> _VerifiedBatch:
        evidence = await read_resolved_memory_evidence(
            self._store,
            self._media_url_signer,
            TenantId(request.tenant_id),
            memories,
        )
        selected_ids = await self._verifier.select_occurrences(
            request,
            memories,
            evidence,
            query_media=query_media,
        )
        candidate_ids = {memory.memory_id for memory in memories}
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= candidate_ids:
            raise ModelOutputError("occurrence selection returned invalid memory IDs")
        selected_set = set(selected_ids)
        selected = tuple(memory for memory in memories if memory.memory_id in selected_set)
        return _VerifiedBatch(memories=selected)
