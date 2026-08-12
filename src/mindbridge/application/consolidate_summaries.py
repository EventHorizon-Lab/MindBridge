"""Evidence-aware hierarchical Memory consolidation use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mindbridge.application.evidence import EvidenceReader, read_resolved_evidence
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import MediaUrlSigner, TextDocumentEmbedder
from mindbridge.application.summary_consolidation import (
    SummaryCandidate,
    SummaryCandidateCursor,
    SummaryCandidatePage,
    SummaryCandidateRequest,
    SummaryConsolidation,
    SummaryWrite,
    derive_summary_writes,
)
from mindbridge.core import MemoryIntegrityError, TenantId
from mindbridge.telemetry import set_current_span_attributes, trace_operation


@dataclass(frozen=True, slots=True)
class SummaryConsolidationResult:
    """Content-free progress and outcome for one bounded Memory page."""

    scanned_count: int
    candidate_count: int
    proposed_count: int
    committed_count: int
    next_cursor: SummaryCandidateCursor | None


class SummaryConsolidator(Protocol):
    """Frozen Omni boundary that verifies hierarchy groups against source evidence."""

    async def propose_summaries(
        self,
        candidates: tuple[SummaryCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> SummaryConsolidation: ...


class SummaryConsolidationStore(EvidenceReader, Protocol):
    """Narrow transactional boundary needed by the Summary use case."""

    async def list_summary_candidates(
        self,
        request: SummaryCandidateRequest,
    ) -> SummaryCandidatePage: ...

    async def commit_summary_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[SummaryWrite, ...],
    ) -> int: ...


class ConsolidateSummaries:
    """Verify and persist one bounded hierarchy page without changing model weights."""

    def __init__(
        self,
        store: SummaryConsolidationStore,
        consolidator: SummaryConsolidator,
        text_embedder: TextDocumentEmbedder,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._consolidator = consolidator
        self._text_embedder = text_embedder
        self._media_url_signer = media_url_signer

    @trace_operation("mindbridge.consolidation.summaries")
    async def run(self, request: SummaryCandidateRequest) -> SummaryConsolidationResult:
        """Discover, inspect, and atomically commit one stable Summary page."""
        page = await self._store.list_summary_candidates(request)
        if any(candidate.memory.tenant_id != request.tenant_id for candidate in page.candidates):
            raise MemoryIntegrityError("Summary candidates crossed the requested tenant")
        if len(page.candidates) < 2:
            return _result(page)
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.consolidation.summary_candidate_count": len(page.candidates),
            }
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for candidate in page.candidates
                for evidence_id in candidate.memory.evidence_ids
            )
        )
        evidence = await read_resolved_evidence(
            self._store,
            self._media_url_signer,
            request.tenant_id,
            evidence_ids,
        )
        consolidation = await self._consolidator.propose_summaries(page.candidates, evidence)
        candidate_ids = {candidate.memory.memory_id for candidate in page.candidates}
        if any(
            memory_id not in candidate_ids
            for proposal in consolidation.summaries
            for memory_id in proposal.source_memory_ids
        ):
            raise MemoryIntegrityError("Summary consolidator returned an unknown candidate")
        writes = await derive_summary_writes(
            request.tenant_id,
            page.candidates,
            consolidation,
            self._text_embedder,
            request.evaluated_at,
        )
        committed_count = (
            await self._store.commit_summary_consolidation(request.tenant_id, writes)
            if writes
            else 0
        )
        set_current_span_attributes({"mindbridge.consolidation.summary_count": committed_count})
        return _result(
            page, proposed_count=len(consolidation.summaries), committed_count=committed_count
        )


def _result(
    page: SummaryCandidatePage,
    *,
    proposed_count: int = 0,
    committed_count: int = 0,
) -> SummaryConsolidationResult:
    return SummaryConsolidationResult(
        scanned_count=page.scanned_count,
        candidate_count=len(page.candidates),
        proposed_count=proposed_count,
        committed_count=committed_count,
        next_cursor=page.next_cursor,
    )
