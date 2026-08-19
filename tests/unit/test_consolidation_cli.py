"""Checks for the bounded Episode consolidation command."""

from datetime import datetime, timezone
from typing import cast

from mindbridge.application.claim_consolidation import ClaimCandidateRequest
from mindbridge.application.consolidate_claims import (
    ClaimConsolidationResult,
    ConsolidateClaims,
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
    consolidate_tenant_claims,
    consolidate_tenant_episodes,
    consolidate_tenant_summaries,
)
from mindbridge.application.summary_consolidation import (
    SummaryCandidateCursor,
    SummaryCandidateRequest,
)
from mindbridge.consolidation_cli import ConsolidationSettings
from mindbridge.core import ClaimId, EventId, MemoryId, TenantId

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
        "MINDBRIDGE_GENERATOR_MODEL_REVISION": "deployment-revision",
        "MINDBRIDGE_EMBEDDER_API_KEY": "embedding-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://embedding.example.test/v1",
    }

    settings = ConsolidationSettings.from_environment(environment)

    assert settings.object_storage.bucket == "memory"
    assert "database-secret" not in repr(settings)
    assert "generator-secret" not in repr(settings)
    assert "embedding-secret" not in repr(settings)
