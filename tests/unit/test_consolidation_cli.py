"""Checks for the bounded Episode consolidation command."""

from datetime import datetime, timezone
from typing import cast

from mindbridge.application import (
    ClaimCandidateRequest,
    ClaimConsolidationResult,
    ConsolidateClaims,
    ConsolidateEpisodes,
    EpisodeCandidateRequest,
    EpisodeConsolidationResult,
)
from mindbridge.consolidation_cli import (
    ConsolidationSettings,
    consolidate_tenant_claims,
    consolidate_tenant_episodes,
)
from mindbridge.core import ClaimId, EventId, TenantId

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

    assert (summary.page_count, summary.scanned_count, summary.committed_count) == (2, 3, 1)
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
    assert (summary.committed_semantic_claim_count, summary.committed_relationship_count) == (1, 1)
    assert scripted.requests[1].after_claim_id == "claim_02"
    assert all(request.evaluated_at == NOW for request in scripted.requests)


def test_consolidation_settings_require_and_redact_credentials() -> None:
    environment = {
        "MINDBRIDGE_DATABASE_URL": "postgresql://user:database-secret@postgres/mindbridge",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_VLM_API_KEY": "vlm-secret",
        "MINDBRIDGE_VLM_ENDPOINT": "https://vlm.example.test/v1/chat/completions",
        "MINDBRIDGE_VLM_MODEL_REVISION": "deployment-revision",
        "MINDBRIDGE_TEXT_EMBEDDING_API_KEY": "embedding-secret",
        "MINDBRIDGE_TEXT_EMBEDDING_ENDPOINT": "https://embedding.example.test/v1/embeddings",
    }

    settings = ConsolidationSettings.from_environment(environment)

    assert settings.object_storage_bucket == "memory"
    assert "database-secret" not in repr(settings)
    assert "vlm-secret" not in repr(settings)
    assert "embedding-secret" not in repr(settings)
