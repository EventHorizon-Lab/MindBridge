"""Exhaustive, evidence-verified occurrence enumeration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from mindbridge.application.evidence import read_resolved_memory_evidence
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    MediaUrlSigner,
    MemoryAnswerer,
    MemoryStore,
    ResolvedQueryMedia,
)
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    EnumerationLimitExceededError,
    MemoryRecord,
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
    evidence: tuple[ResolvedEvidence, ...]


class EnumerateMemories:
    """Scan a bounded filter scope and ask Omni to verify every candidate."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: MemoryAnswerer,
        media_url_signer: MediaUrlSigner,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._answerer = answerer
        self._media_url_signer = media_url_signer
        self._clock = clock

    @trace_operation("mindbridge.recall.enumerate")
    async def run(
        self,
        request: RecallRequest,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> EnumerationResult:
        candidates = await self._store.list_memories_for_enumeration(
            request,
            limit=ENUMERATION_CANDIDATE_LIMIT + 1,
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
        returned_evidence_ids = {
            evidence_id for memory in accessed for evidence_id in memory.evidence_ids
        }
        evidence_by_id = {
            item.evidence_span.evidence_id: item
            for batch in verified_batches
            for item in batch.evidence
            if item.evidence_span.evidence_id in returned_evidence_ids
        }
        set_current_span_attributes(
            {
                "mindbridge.enumeration.candidate_count": len(candidates),
                "mindbridge.enumeration.verifiable_count": len(verifiable),
                "mindbridge.enumeration.occurrence_count": len(accessed),
            }
        )
        return EnumerationResult(
            memories=accessed,
            evidence=tuple(evidence_by_id.values()),
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
            verified.extend(
                await asyncio.gather(
                    *(
                        self._verify_batch(request, query_media, batch)
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
        selected_ids = await self._answerer.select_occurrences(
            request,
            memories,
            evidence,
            query_media=query_media,
        )
        candidate_ids = {memory.memory_id for memory in memories}
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= candidate_ids:
            raise ModelOutputError("Omni occurrence selection returned invalid memory IDs")
        selected_set = set(selected_ids)
        selected = tuple(memory for memory in memories if memory.memory_id in selected_set)
        selected_evidence_ids = {
            evidence_id for memory in selected for evidence_id in memory.evidence_ids
        }
        return _VerifiedBatch(
            memories=selected,
            evidence=tuple(
                item for item in evidence if item.evidence_span.evidence_id in selected_evidence_ids
            ),
        )
