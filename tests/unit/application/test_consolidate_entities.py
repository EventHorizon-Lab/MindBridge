"""What the entity resolution pass writes, and — mostly — what it refuses to write."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    ModelOutputError,
    ModelRequestError,
    ModelUnavailableError,
    ObjectStorageError,
    ObservationId,
    TenantId,
)

_AT = datetime(2026, 8, 19, tzinfo=timezone.utc)
_TENANT = TenantId("tenant_01")
_EVIDENCE = EvidenceId("evidence_1")
_MEDIA = MediaObjectId("media_1")


def _span() -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=_EVIDENCE,
        tenant_id=_TENANT,
        observation_id=ObservationId("observation_1"),
        media_object_id=_MEDIA,
        start_ms=0,
        end_ms=4_000,
        created_at=_AT,
    )


def _media() -> MediaObject:
    return MediaObject(
        media_object_id=_MEDIA,
        tenant_id=_TENANT,
        kind=MediaKind.VIDEO,
        uri="s3://memory/tenants/tenant_01/clip.mp4",
        sha256=f"{1:064x}",
        size_bytes=100,
        created_at=_AT,
        duration_ms=4_000,
    )


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
        return tuple(_span() for _ in evidence_ids)

    async def read_media_objects(
        self, tenant_id: TenantId, media_object_ids: tuple[MediaObjectId, ...]
    ) -> tuple[MediaObject, ...]:
        return tuple(_media() for _ in media_object_ids)

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
        return PresignedMediaDownload(
            download_url="https://objects.example.test/clip.mp4",
            expires_at=_AT + timedelta(minutes=5),
        )


async def _run(
    *,
    verdict: EntityAdjudication | None = None,
    evidence_error: Exception | None = None,
    adjudicator_error: Exception | None = None,
    minimum_confidence: float = 0.75,
    tenant_id: TenantId = _TENANT,
    evidence_ids: tuple[EvidenceId, ...] = (_EVIDENCE,),
) -> EntityResolutionResult:
    store = _Store(
        page=_page(tenant_id=tenant_id, evidence_ids=evidence_ids),
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


async def test_an_unsure_negative_is_not_recorded_as_a_difference() -> None:
    """The prompt answers false when it cannot tell, so a hedged no means unknown."""
    result = await _run(verdict=EntityAdjudication(False, 0.1, "cue"), minimum_confidence=0.75)
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


async def test_a_confident_negative_is_recorded() -> None:
    result = await _run(verdict=EntityAdjudication(False, 0.9, "cue"), minimum_confidence=0.75)
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


async def test_a_pair_with_nothing_to_look_at_is_never_merged() -> None:
    """Answering from the two names alone is the merge this pass exists to prevent."""
    result = await _run(verdict=EntityAdjudication(True, 1.0, "cue"), evidence_ids=())
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


async def test_missing_evidence_skips_the_pair_instead_of_discarding_the_page() -> None:
    """A forget between the page read and the judge must not void the other verdicts."""
    result = await _run(evidence_error=MemoryIntegrityError("evidence is gone"))
    assert (result.same_as_count, result.not_same_as_count) == (0, 0)
    assert result.skipped_pair_count == 1


@dataclass
class _FailsOnePair:
    """Raise for one named pair and answer every other, regardless of arrival order."""

    verdict: EntityAdjudication
    failing_left: str

    async def adjudicate(
        self, pair: EntityPair, evidence: tuple[ResolvedEvidence, ...]
    ) -> EntityAdjudication:
        if pair.left.entity.entity_id == self.failing_left:
            raise ModelRequestError("call failed")
        return self.verdict


def _two_pair_page() -> EntityCandidatePage:
    return EntityCandidatePage(
        pairs=(
            EntityPair(
                left=_candidate("entity_a", evidence_ids=(_EVIDENCE,)),
                right=_candidate("entity_b", evidence_ids=(_EVIDENCE,)),
            ),
            EntityPair(
                left=_candidate("entity_c", evidence_ids=(_EVIDENCE,)),
                right=_candidate("entity_d", evidence_ids=(_EVIDENCE,)),
            ),
        ),
        scanned_count=4,
        dropped_pair_count=0,
        next_cursor=None,
    )


async def test_a_page_fatal_error_still_commits_the_verdicts_the_page_reached() -> None:
    """A page is up to maximum_pairs multimodal calls; one blip must not void the rest."""
    store = _Store(page=_two_pair_page())
    use_case = ConsolidateEntities(
        store,
        _FailsOnePair(EntityAdjudication(True, 0.9, "same scar"), failing_left="entity_c"),
        media_url_signer=_Signer(),
    )

    with pytest.raises(ModelRequestError):
        await use_case.run(EntityCandidateRequest(tenant_id=_TENANT, evaluated_at=_AT))

    assert store.committed is not None, "the reached verdict was discarded with the failure"
    assert [len(write.relations) for write in store.committed] == [1]
