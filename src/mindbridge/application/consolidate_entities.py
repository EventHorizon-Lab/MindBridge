"""Evidence-verified cross-clip entity resolution use case.

The whole value of this pass is what it refuses to write. A `not_same_as` edge is a durable
claim that two records were inspected and found to be different entities; if a failed
inspection could produce one, the graph would permanently record "we looked and they differ"
for pairs nobody ever looked at. So exactly two outcomes write an edge — a confident yes and
an explicit no — and everything else leaves the pair unjudged for a later sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mindbridge.application.entity_resolution import (
    EntityAdjudication,
    EntityCandidatePage,
    EntityCandidateRequest,
    EntityPair,
    EntityResolutionWrite,
    derive_entity_resolution_write,
)
from mindbridge.application.evidence import EvidenceReader, read_resolved_evidence
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import MediaUrlSigner
from mindbridge.core import (
    EntityId,
    MemoryIntegrityError,
    ModelOutputError,
    ObjectStorageError,
    RelationType,
    TenantId,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


@dataclass(frozen=True, slots=True)
class EntityResolutionResult:
    """Content-free progress and outcome for one bounded entity page."""

    scanned_count: int
    candidate_pair_count: int
    dropped_pair_count: int
    same_as_count: int
    not_same_as_count: int
    skipped_pair_count: int
    committed_count: int
    next_cursor: EntityId | None


class EntityAdjudicator(Protocol):
    """Frozen model boundary that judges one pair against its original media."""

    async def adjudicate(
        self,
        pair: EntityPair,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EntityAdjudication: ...


class EntityResolutionStore(EvidenceReader, Protocol):
    """Narrow transactional boundary needed by the entity resolution use case."""

    async def list_entity_candidates(
        self,
        request: EntityCandidateRequest,
    ) -> EntityCandidatePage: ...

    async def commit_entity_resolution(
        self,
        tenant_id: TenantId,
        write: EntityResolutionWrite,
    ) -> int: ...


class ConsolidateEntities:
    """Judge and persist one bounded page of entity pairs."""

    def __init__(
        self,
        store: EntityResolutionStore,
        adjudicator: EntityAdjudicator,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._adjudicator = adjudicator
        self._media_url_signer = media_url_signer

    @trace_operation("mindbridge.consolidation.entities")
    async def run(self, request: EntityCandidateRequest) -> EntityResolutionResult:
        """Discover, inspect, and atomically commit one stable page of entity pairs."""
        page = await self._store.list_entity_candidates(request)
        self._require_one_tenant(request, page)
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.consolidation.entity_pair_count": len(page.pairs),
                "mindbridge.consolidation.entity_pair_dropped": page.dropped_pair_count,
            }
        )
        decided, skipped = await self._decide(request, page)
        write = derive_entity_resolution_write(request.tenant_id, decided, request.evaluated_at)
        committed = (
            await self._store.commit_entity_resolution(request.tenant_id, write)
            if write.relations
            else 0
        )
        return _result(page, write, skipped=skipped, committed=committed)

    async def _decide(
        self,
        request: EntityCandidateRequest,
        page: EntityCandidatePage,
    ) -> tuple[tuple[tuple[EntityPair, EntityAdjudication], ...], int]:
        decided: list[tuple[EntityPair, EntityAdjudication]] = []
        skipped = 0
        for pair in page.pairs:
            try:
                evidence = await self._pair_evidence(request, pair)
                adjudication = await self._adjudicator.adjudicate(pair, evidence)
            except (ObjectStorageError, ModelOutputError):
                # Could not open the media, or could not read the verdict. Both mean the pair
                # was never actually inspected, and an uninspected pair must stay unjudged.
                # ModelUnavailableError and ModelRequestError are deliberately not caught:
                # they are the sweep's problem to retry, not this pair's verdict.
                skipped += 1
                continue
            if adjudication.same_entity and adjudication.confidence < request.minimum_confidence:
                # A hedged yes is not a yes. It is also not a no, so nothing is written and
                # the pair stays available to a later sweep with better evidence.
                skipped += 1
                continue
            decided.append((pair, adjudication))
        return tuple(decided), skipped

    async def _pair_evidence(
        self,
        request: EntityCandidateRequest,
        pair: EntityPair,
    ) -> tuple[ResolvedEvidence, ...]:
        evidence_ids = tuple(
            dict.fromkeys(
                (
                    *pair.left.evidence_ids[: request.evidence_per_side],
                    *pair.right.evidence_ids[: request.evidence_per_side],
                )
            )
        )
        return await read_resolved_evidence(
            self._store,
            self._media_url_signer,
            request.tenant_id,
            evidence_ids,
        )

    @staticmethod
    def _require_one_tenant(
        request: EntityCandidateRequest,
        page: EntityCandidatePage,
    ) -> None:
        if any(
            candidate.entity.tenant_id != request.tenant_id
            for pair in page.pairs
            for candidate in (pair.left, pair.right)
        ):
            raise MemoryIntegrityError("entity candidates crossed the requested tenant")


def _result(
    page: EntityCandidatePage,
    write: EntityResolutionWrite,
    *,
    skipped: int,
    committed: int,
) -> EntityResolutionResult:
    return EntityResolutionResult(
        scanned_count=page.scanned_count,
        candidate_pair_count=len(page.pairs),
        dropped_pair_count=page.dropped_pair_count,
        # Counted from the write, not from the commit: a re-run legitimately inserts zero
        # rows while still having reached both verdicts.
        same_as_count=sum(item.relation_type is RelationType.SAME_AS for item in write.relations),
        not_same_as_count=sum(
            item.relation_type is RelationType.NOT_SAME_AS for item in write.relations
        ),
        skipped_pair_count=skipped,
        committed_count=committed,
        next_cursor=page.next_cursor,
    )
