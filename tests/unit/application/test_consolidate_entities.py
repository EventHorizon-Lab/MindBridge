"""What the entity resolution pass writes, and — mostly — what it refuses to write."""

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from mindbridge.application.consolidate_entities import (
    ConsolidateEntities,
    EntityResolutionResult,
)
from mindbridge.application.entity_resolution import (
    EntityAdjudication,
    EntityCandidate,
    EntityCandidatePage,
    EntityCandidateRequest,
    EntityPair,
    EntityResolutionWrite,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.core import (
    Entity,
    EntityId,
    EntityType,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    ModelOutputError,
    ModelRequestError,
    ModelUnavailableError,
    ObjectStorageError,
    TenantId,
)

_AT = datetime(2026, 8, 19, tzinfo=timezone.utc)
_TENANT = TenantId("tenant_01")


@dataclass
class _Store:
    page: EntityCandidatePage
    evidence_error: Exception | None = None
    committed: list[EntityResolutionWrite] | None = None

    async def list_entity_candidates(self, request: EntityCandidateRequest) -> EntityCandidatePage:
        return self.page

    async def read_evidence(
        self, tenant_id: TenantId, evidence_ids: tuple[EvidenceId, ...]
    ) -> tuple[EvidenceSpan, ...]:
        if self.evidence_error is not None:
            raise self.evidence_error
        return ()

    async def read_media_objects(
        self, tenant_id: TenantId, media_object_ids: tuple[MediaObjectId, ...]
    ) -> tuple[MediaObject, ...]:
        return ()

    async def commit_entity_resolution(
        self, tenant_id: TenantId, write: EntityResolutionWrite
    ) -> int:
        if self.committed is None:
            self.committed = []
        self.committed.append(write)
        return len(write.relations)


@dataclass
class _Adjudicator:
    verdict: EntityAdjudication | None = None
    error: Exception | None = None

    async def adjudicate(
        self, pair: EntityPair, evidence: tuple[ResolvedEvidence, ...]
    ) -> EntityAdjudication:
        if self.error is not None:
            raise self.error
        assert self.verdict is not None
        return self.verdict


class _Signer:
    async def create_presigned_download(self, media_object: MediaObject) -> PresignedMediaDownload:
        raise AssertionError("no media is resolved in these tests")


async def _run(
    *,
    verdict: EntityAdjudication | None = None,
    evidence_error: Exception | None = None,
    adjudicator_error: Exception | None = None,
    minimum_confidence: float = 0.75,
    tenant_id: TenantId = _TENANT,
) -> EntityResolutionResult:
    store = _Store(
        page=_page(
            tenant_id=tenant_id,
            # Only a pair that actually has evidence can fail to have it read.
            evidence_ids=(EvidenceId("evidence_1"),) if evidence_error else (),
        ),
        evidence_error=evidence_error,
    )
    use_case = ConsolidateEntities(
        store,
        _Adjudicator(verdict=verdict, error=adjudicator_error),
        media_url_signer=_Signer(),
    )
    return await use_case.run(
        EntityCandidateRequest(
            tenant_id=_TENANT,
            evaluated_at=_AT,
            minimum_confidence=minimum_confidence,
        )
    )


def _page(
    *, tenant_id: TenantId = _TENANT, evidence_ids: tuple[EvidenceId, ...] = ()
) -> EntityCandidatePage:
    return EntityCandidatePage(
        pairs=(
            EntityPair(
                left=_candidate("entity_a", tenant_id=tenant_id, evidence_ids=evidence_ids),
                right=_candidate("entity_b", tenant_id=tenant_id, evidence_ids=evidence_ids),
            ),
        ),
        scanned_count=2,
        dropped_pair_count=0,
        next_cursor=None,
    )


def _candidate(
    entity_id: str, *, tenant_id: TenantId = _TENANT, evidence_ids: tuple[EvidenceId, ...] = ()
) -> EntityCandidate:
    return EntityCandidate(
        entity=Entity(
            entity_id=EntityId(entity_id),
            tenant_id=tenant_id,
            entity_type=EntityType.PERSON,
            canonical_name=entity_id,
            created_at=_AT,
        ),
        evidence_ids=evidence_ids,
    )


async def test_a_confident_positive_becomes_one_same_as_edge() -> None:
    result = await _run(verdict=EntityAdjudication(True, 0.9, "cue"))
    assert (result.same_as_count, result.not_same_as_count) == (1, 0)
    assert result.skipped_pair_count == 0


async def test_an_explicit_negative_is_recorded_so_the_pair_is_not_re_paid_for() -> None:
    result = await _run(verdict=EntityAdjudication(False, 0.9, "cue"))
    assert (result.same_as_count, result.not_same_as_count) == (0, 1)


async def test_a_positive_below_the_confidence_floor_writes_nothing() -> None:
    """A hedged yes is not a yes, and it is not a no either."""
    result = await _run(verdict=EntityAdjudication(True, 0.5, "cue"), minimum_confidence=0.75)
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


async def test_a_confident_negative_below_the_floor_is_still_recorded() -> None:
    """The floor guards merging. Refusing to merge needs no confidence bar."""
    result = await _run(verdict=EntityAdjudication(False, 0.1, "cue"), minimum_confidence=0.75)
    assert (result.same_as_count, result.not_same_as_count) == (0, 1)


async def test_unreadable_media_skips_the_pair_and_writes_no_verdict() -> None:
    """A pair nobody could look at must not become a durable 'they differ'."""
    result = await _run(evidence_error=ObjectStorageError("media gone"))
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


async def test_invalid_model_output_skips_the_pair_and_writes_no_verdict() -> None:
    result = await _run(adjudicator_error=ModelOutputError("bad json"))
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


@pytest.mark.parametrize("error", [ModelUnavailableError("down"), ModelRequestError("call failed")])
async def test_infrastructure_failure_propagates_instead_of_recording_a_negative(
    error: Exception,
) -> None:
    with pytest.raises((ModelUnavailableError, ModelRequestError)):
        await _run(adjudicator_error=error)


async def test_a_cross_tenant_candidate_is_refused() -> None:
    with pytest.raises(MemoryIntegrityError):
        await _run(verdict=EntityAdjudication(True, 0.9, "cue"), tenant_id=TenantId("tenant_02"))
