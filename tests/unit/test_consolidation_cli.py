"""Checks for the bounded Episode consolidation command."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge.application.claim_consolidation import ClaimCandidateRequest
from mindbridge.application.consolidate_claims import (
    ClaimConsolidationResult,
    ConsolidateClaims,
)
from mindbridge.application.consolidate_entities import (
    ConsolidateEntities,
    EntityResolutionResult,
)
from mindbridge.application.consolidate_summaries import (
    ConsolidateSummaries,
    SummaryConsolidationResult,
)
from mindbridge.application.consolidation import (
    ConsolidateEpisodes,
    EpisodeCandidateRequest,
    EpisodeConsolidationResult,
)
from mindbridge.application.consolidation_sweep import (
    ConsolidationSweepSummary,
    SweepSummary,
    consolidate_tenant_claims,
    consolidate_tenant_entities,
    consolidate_tenant_episodes,
    consolidate_tenant_summaries,
)
from mindbridge.application.entity_resolution import EntityCandidateRequest
from mindbridge.application.summary_consolidation import (
    SummaryCandidateCursor,
    SummaryCandidateRequest,
)
from mindbridge.consolidation_cli import ConsolidationSettings, _parser, summary_dict
from mindbridge.core import (
    ClaimId,
    EntityId,
    EntityType,
    EventId,
    MemoryId,
    MemoryIntegrityError,
    TenantId,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class ScriptedConsolidation:
    def __init__(self) -> None:
        self.requests: list[EpisodeCandidateRequest] = []

    async def run(self, request: EpisodeCandidateRequest) -> EpisodeConsolidationResult:
        self.requests.append(request)
        if request.after_event_id is None:
            return EpisodeConsolidationResult(
                scanned_count=2,
                candidate_count=3,
                proposed_count=1,
                committed_count=1,
                next_cursor=EventId("event_02"),
            )
        return EpisodeConsolidationResult(
            scanned_count=1,
            candidate_count=1,
            proposed_count=0,
            committed_count=0,
            next_cursor=None,
        )


class ScriptedClaimConsolidation:
    def __init__(self) -> None:
        self.requests: list[ClaimCandidateRequest] = []

    async def run(self, request: ClaimCandidateRequest) -> ClaimConsolidationResult:
        self.requests.append(request)
        if request.after_claim_id is None:
            return ClaimConsolidationResult(
                scanned_count=2,
                candidate_count=4,
                proposed_semantic_claim_count=1,
                proposed_relationship_count=1,
                committed_semantic_claim_count=1,
                committed_relationship_count=1,
                next_cursor=ClaimId("claim_02"),
            )
        return ClaimConsolidationResult(
            scanned_count=1,
            candidate_count=0,
            proposed_semantic_claim_count=0,
            proposed_relationship_count=0,
            committed_semantic_claim_count=0,
            committed_relationship_count=0,
            next_cursor=None,
        )


class ScriptedSummaryConsolidation:
    def __init__(self) -> None:
        self.requests: list[SummaryCandidateRequest] = []

    async def run(self, request: SummaryCandidateRequest) -> SummaryConsolidationResult:
        self.requests.append(request)
        if request.after_cursor is None:
            return SummaryConsolidationResult(
                scanned_count=2,
                candidate_count=4,
                proposed_count=1,
                committed_count=1,
                next_cursor=SummaryCandidateCursor(
                    occurred_at=NOW,
                    memory_id=MemoryId("memory_02"),
                ),
            )
        return SummaryConsolidationResult(
            scanned_count=1,
            candidate_count=0,
            proposed_count=0,
            committed_count=0,
            next_cursor=None,
        )


async def test_episode_sweep_accumulates_stable_pages() -> None:
    scripted = ScriptedConsolidation()

    summary = await consolidate_tenant_episodes(
        cast(ConsolidateEpisodes, scripted),
        TenantId("tenant_01"),
        NOW,
        page_size=8,
        maximum_gap_seconds=600,
        minimum_similarity=0.75,
    )

    assert (summary.page_count, summary.scanned_count) == (2, 3)
    assert summary.counts["committed_count"] == 1
    assert scripted.requests[1].after_event_id == "event_02"
    assert all(request.evaluated_at == NOW for request in scripted.requests)


async def test_claim_sweep_accumulates_semantic_and_relationship_counts() -> None:
    scripted = ScriptedClaimConsolidation()

    summary = await consolidate_tenant_claims(
        cast(ConsolidateClaims, scripted),
        TenantId("tenant_01"),
        NOW,
        page_size=8,
        maximum_gap_seconds=2_592_000,
        minimum_similarity=0.8,
    )

    assert (summary.page_count, summary.scanned_count, summary.candidate_count) == (2, 3, 4)
    assert summary.counts["committed_semantic_claim_count"] == 1
    assert summary.counts["committed_relationship_count"] == 1
    assert scripted.requests[1].after_claim_id == "claim_02"
    assert all(request.evaluated_at == NOW for request in scripted.requests)


async def test_summary_sweep_accumulates_stable_memory_pages() -> None:
    scripted = ScriptedSummaryConsolidation()

    summary = await consolidate_tenant_summaries(
        cast(ConsolidateSummaries, scripted),
        TenantId("tenant_01"),
        NOW,
        page_size=8,
        maximum_gap_seconds=2_592_000,
        minimum_similarity=0.8,
    )

    assert (summary.page_count, summary.scanned_count) == (2, 3)
    assert summary.counts["committed_count"] == 1
    assert scripted.requests[1].after_cursor == SummaryCandidateCursor(
        occurred_at=NOW,
        memory_id=MemoryId("memory_02"),
    )
    assert all(request.evaluated_at == NOW for request in scripted.requests)


def test_consolidation_settings_require_and_redact_credentials() -> None:
    environment = {
        "MINDBRIDGE_DATABASE_URL": "postgresql://user:database-secret@postgres/mindbridge",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
        "MINDBRIDGE_EMBEDDER_API_KEY": "embedding-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://embedding.example.test/v1",
    }

    settings = ConsolidationSettings.from_environment(environment)

    assert settings.object_storage.bucket == "memory"
    assert "database-secret" not in repr(settings)
    assert "generator-secret" not in repr(settings)
    assert "embedding-secret" not in repr(settings)


class ScriptedEntityResolution:
    def __init__(self, *, stuck: bool = False) -> None:
        self.requests: list[EntityCandidateRequest] = []
        self._stuck = stuck

    async def run(self, request: EntityCandidateRequest) -> EntityResolutionResult:
        self.requests.append(request)
        if self._stuck:
            return _entity_result(next_cursor=EntityId("entity_02"))
        if request.after_entity_id is None:
            return _entity_result(
                scanned_count=2,
                candidate_pair_count=3,
                dropped_pair_count=1,
                same_as_count=1,
                not_same_as_count=1,
                skipped_pair_count=1,
                committed_count=2,
                next_cursor=EntityId("entity_02"),
            )
        return _entity_result(
            scanned_count=1, candidate_pair_count=1, not_same_as_count=1, committed_count=1
        )


def _entity_result(
    *,
    scanned_count: int = 0,
    candidate_pair_count: int = 0,
    dropped_pair_count: int = 0,
    same_as_count: int = 0,
    not_same_as_count: int = 0,
    skipped_pair_count: int = 0,
    committed_count: int = 0,
    next_cursor: EntityId | None = None,
) -> EntityResolutionResult:
    return EntityResolutionResult(
        scanned_count=scanned_count,
        candidate_pair_count=candidate_pair_count,
        dropped_pair_count=dropped_pair_count,
        same_as_count=same_as_count,
        not_same_as_count=not_same_as_count,
        skipped_pair_count=skipped_pair_count,
        committed_count=committed_count,
        next_cursor=next_cursor,
    )


async def _entity_sweep(scripted: ScriptedEntityResolution) -> SweepSummary:
    return await consolidate_tenant_entities(
        cast(ConsolidateEntities, scripted),
        TenantId("tenant_01"),
        NOW,
        page_size=8,
        maximum_gap_seconds=2_592_000,
        candidate_limit=8,
        minimum_confidence=0.75,
        evidence_per_side=3,
        maximum_pairs=64,
        entity_types=(EntityType.PERSON,),
        readjudicate=False,
    )


async def test_entity_sweep_accumulates_both_verdicts_and_both_kinds_of_refusal() -> None:
    scripted = ScriptedEntityResolution()

    summary = await _entity_sweep(scripted)

    assert (summary.page_count, summary.scanned_count, summary.candidate_count) == (2, 3, 4)
    assert (summary.counts["same_as_count"], summary.counts["not_same_as_count"]) == (1, 2)
    # The two ways the sweep declines to answer stay separate: one pair was reached and left
    # unjudged, one was never looked at because the page hit its budget.
    assert summary.counts["skipped_pair_count"] == 1
    assert summary.counts["dropped_pair_count"] == 1
    assert summary.counts["committed_count"] == 3
    assert scripted.requests[1].after_entity_id == "entity_02"
    assert all(request.evaluated_at == NOW for request in scripted.requests)


async def test_entity_sweep_refuses_a_cursor_that_does_not_advance() -> None:
    with pytest.raises(MemoryIntegrityError):
        await _entity_sweep(ScriptedEntityResolution(stuck=True))


def test_entity_options_default_to_person_only_and_bounded() -> None:
    """Widening to other types and re-judging settled pairs are both opt-in."""
    options = _parser().parse_args(["--tenant-id", "tenant_01"])

    assert options.entity_types is None
    assert options.entity_readjudicate is False
    assert (options.entity_page_size, options.entity_maximum_pairs) == (16, 64)
    assert options.entity_minimum_confidence == 0.75


def test_entity_type_is_repeatable_and_validated() -> None:
    options = _parser().parse_args(
        ["--tenant-id", "tenant_01", "--entity-type", "person", "--entity-type", "object"]
    )
    assert options.entity_types == ["person", "object"]
    with pytest.raises(SystemExit):
        _parser().parse_args(["--tenant-id", "tenant_01", "--entity-type", "not-a-type"])


def test_entity_resolution_is_the_one_sweep_an_operator_can_turn_off() -> None:
    """It is the only sweep that opens media and spends a generator call per pair."""
    assert _parser().parse_args(["--tenant-id", "tenant_01"]).skip_entity_resolution is False
    options = _parser().parse_args(["--tenant-id", "tenant_01", "--skip-entity-resolution"])

    assert options.skip_entity_resolution is True


def test_a_skipped_entity_sweep_is_reported_apart_from_one_that_found_nothing() -> None:
    """Zeroing the counts would read as "ran, paired nothing" in anything summing them."""
    empty = SweepSummary(
        tenant_id=TenantId("tenant_01"),
        evaluated_at=NOW,
        page_count=1,
        scanned_count=0,
        candidate_count=0,
        counts={},
    )
    ran = summary_dict(
        ConsolidationSweepSummary(episodes=empty, claims=empty, summaries=empty, entities=empty)
    )
    skipped = summary_dict(
        ConsolidationSweepSummary(episodes=empty, claims=empty, summaries=empty, entities=None)
    )

    assert skipped["entities"] is None
    assert ran["entities"] == {"candidate_count": 0, "page_count": 1, "scanned_count": 0}
