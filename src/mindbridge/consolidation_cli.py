"""Scheduled evidence-verified Episode, Claim, and Summary consolidation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import ConsolidateEpisodes
from mindbridge.application.consolidation_sweep import (
    ConsolidationSweepSummary,
    consolidate_tenant_claims,
    consolidate_tenant_episodes,
    consolidate_tenant_summaries,
)
from mindbridge.configuration import (
    optional_environment_value,
    parse_aware_datetime,
    require_environment_value,
)
from mindbridge.core import EmbeddingSpaceReference, TenantId
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import S3MediaAccess
from mindbridge.models.jina import (
    DEFAULT_JINA_RETRIEVAL_SPACE,
    DEFAULT_JINA_TEXT_MODEL_ID,
    DEFAULT_JINA_TEXT_REVISION,
)
from mindbridge.models.openai_claim_consolidation import OpenAIOmniClaimConsolidator
from mindbridge.models.openai_consolidation import OpenAIOmniEpisodeConsolidator
from mindbridge.models.openai_embeddings import OpenAIJinaTextEmbedder
from mindbridge.models.openai_omni import DEFAULT_OMNI_MODEL_ID
from mindbridge.models.openai_summary_consolidation import (
    OpenAIOmniSummaryConsolidator,
)
from mindbridge.telemetry import configure_telemetry


@dataclass(frozen=True, slots=True)
class ConsolidationSettings:
    """Validated consolidation process configuration with redacted credentials."""

    database_url: str = field(repr=False)
    object_storage_bucket: str
    vlm_api_key: str = field(repr=False)
    vlm_endpoint: str
    vlm_model_revision: str
    text_embedding_api_key: str = field(repr=False)
    text_embedding_endpoint: str
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "us-east-1"
    vlm_model_id: str = DEFAULT_OMNI_MODEL_ID
    text_embedding_model_id: str = DEFAULT_JINA_TEXT_MODEL_ID
    text_embedding_model_revision: str = DEFAULT_JINA_TEXT_REVISION
    embedding_space_id: str = DEFAULT_JINA_RETRIEVAL_SPACE.space_id
    embedding_space_revision: str = DEFAULT_JINA_RETRIEVAL_SPACE.revision

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("object_storage_bucket", self.object_storage_bucket),
            ("object_storage_region", self.object_storage_region),
            ("vlm_api_key", self.vlm_api_key),
            ("vlm_endpoint", self.vlm_endpoint),
            ("vlm_model_id", self.vlm_model_id),
            ("vlm_model_revision", self.vlm_model_revision),
            ("text_embedding_api_key", self.text_embedding_api_key),
            ("text_embedding_endpoint", self.text_embedding_endpoint),
            ("text_embedding_model_id", self.text_embedding_model_id),
            ("text_embedding_model_revision", self.text_embedding_model_revision),
            ("embedding_space_id", self.embedding_space_id),
            ("embedding_space_revision", self.embedding_space_revision),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            self.object_storage_endpoint_url is not None
            and not self.object_storage_endpoint_url.strip()
        ):
            raise ValueError("object_storage_endpoint_url must not be empty when provided")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ConsolidationSettings:
        """Read the documented process contract without requiring API or broker settings."""
        source = os.environ if environ is None else environ
        return cls(
            database_url=require_environment_value(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage_bucket=require_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"
            ),
            object_storage_endpoint_url=optional_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"
            ),
            object_storage_region=source.get("MINDBRIDGE_OBJECT_STORAGE_REGION", "us-east-1"),
            vlm_api_key=require_environment_value(source, "MINDBRIDGE_VLM_API_KEY"),
            vlm_endpoint=require_environment_value(source, "MINDBRIDGE_VLM_ENDPOINT"),
            vlm_model_id=source.get("MINDBRIDGE_VLM_MODEL_ID", DEFAULT_OMNI_MODEL_ID),
            vlm_model_revision=require_environment_value(source, "MINDBRIDGE_VLM_MODEL_REVISION"),
            text_embedding_api_key=require_environment_value(
                source, "MINDBRIDGE_TEXT_EMBEDDING_API_KEY"
            ),
            text_embedding_endpoint=require_environment_value(
                source, "MINDBRIDGE_TEXT_EMBEDDING_ENDPOINT"
            ),
            text_embedding_model_id=source.get(
                "MINDBRIDGE_TEXT_EMBEDDING_MODEL_ID", DEFAULT_JINA_TEXT_MODEL_ID
            ),
            text_embedding_model_revision=source.get(
                "MINDBRIDGE_TEXT_EMBEDDING_MODEL_REVISION", DEFAULT_JINA_TEXT_REVISION
            ),
            embedding_space_id=source.get(
                "MINDBRIDGE_EMBEDDING_SPACE_ID", DEFAULT_JINA_RETRIEVAL_SPACE.space_id
            ),
            embedding_space_revision=source.get(
                "MINDBRIDGE_EMBEDDING_SPACE_REVISION",
                DEFAULT_JINA_RETRIEVAL_SPACE.revision,
            ),
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one tenant sweep using only explicit process configuration."""
    configure_telemetry("mindbridge-consolidation")
    options = _parser().parse_args(argv)
    summary = asyncio.run(
        _run_postgres_sweep(
            ConsolidationSettings.from_environment(),
            TenantId(options.tenant_id),
            options.evaluated_at,
            page_size=options.page_size,
            maximum_gap_seconds=options.maximum_gap_seconds,
            minimum_similarity=options.minimum_similarity,
            claim_page_size=options.claim_page_size,
            claim_maximum_gap_seconds=options.claim_maximum_gap_seconds,
            claim_minimum_similarity=options.claim_minimum_similarity,
            summary_page_size=options.summary_page_size,
            summary_maximum_gap_seconds=options.summary_maximum_gap_seconds,
            summary_minimum_similarity=options.summary_minimum_similarity,
        )
    )
    print(json.dumps(_summary_dict(summary), sort_keys=True))


async def _run_postgres_sweep(
    settings: ConsolidationSettings,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
    claim_page_size: int,
    claim_maximum_gap_seconds: int,
    claim_minimum_similarity: float,
    summary_page_size: int,
    summary_maximum_gap_seconds: int,
    summary_minimum_similarity: float,
) -> ConsolidationSweepSummary:
    store = PostgresMemoryStore(settings.database_url)
    media_access = S3MediaAccess(
        settings.object_storage_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        region_name=settings.object_storage_region,
    )
    consolidator = OpenAIOmniEpisodeConsolidator.connect(
        api_key=settings.vlm_api_key,
        endpoint=settings.vlm_endpoint,
        model_id=settings.vlm_model_id,
        model_revision=settings.vlm_model_revision,
    )
    claim_consolidator = OpenAIOmniClaimConsolidator.connect(
        api_key=settings.vlm_api_key,
        endpoint=settings.vlm_endpoint,
        model_id=settings.vlm_model_id,
        model_revision=settings.vlm_model_revision,
    )
    summary_consolidator = OpenAIOmniSummaryConsolidator.connect(
        api_key=settings.vlm_api_key,
        endpoint=settings.vlm_endpoint,
        model_id=settings.vlm_model_id,
        model_revision=settings.vlm_model_revision,
    )
    text_embedder = OpenAIJinaTextEmbedder.connect(
        api_key=settings.text_embedding_api_key,
        endpoint=settings.text_embedding_endpoint,
        model_id=settings.text_embedding_model_id,
        model_revision=settings.text_embedding_model_revision,
        space_reference=EmbeddingSpaceReference(
            space_id=settings.embedding_space_id,
            revision=settings.embedding_space_revision,
        ),
    )
    async with AsyncExitStack() as resources:
        resources.push_async_callback(consolidator.close)
        resources.push_async_callback(claim_consolidator.close)
        resources.push_async_callback(summary_consolidator.close)
        resources.push_async_callback(text_embedder.close)
        await store.open()
        resources.push_async_callback(store.close)
        episodes = await consolidate_tenant_episodes(
            ConsolidateEpisodes(
                store,
                consolidator,
                text_embedder,
                media_url_signer=media_access,
            ),
            tenant_id,
            evaluated_at,
            page_size=page_size,
            maximum_gap_seconds=maximum_gap_seconds,
            minimum_similarity=minimum_similarity,
        )
        claims = await consolidate_tenant_claims(
            ConsolidateClaims(
                store,
                claim_consolidator,
                text_embedder,
                media_url_signer=media_access,
            ),
            tenant_id,
            evaluated_at,
            page_size=claim_page_size,
            maximum_gap_seconds=claim_maximum_gap_seconds,
            minimum_similarity=claim_minimum_similarity,
        )
        summaries = await consolidate_tenant_summaries(
            ConsolidateSummaries(
                store,
                summary_consolidator,
                text_embedder,
                media_url_signer=media_access,
            ),
            tenant_id,
            evaluated_at,
            page_size=summary_page_size,
            maximum_gap_seconds=summary_maximum_gap_seconds,
            minimum_similarity=summary_minimum_similarity,
        )
        return ConsolidationSweepSummary(
            episodes=episodes,
            claims=claims,
            summaries=summaries,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--evaluated-at", type=parse_aware_datetime, default=_utc_now())
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--maximum-gap-seconds", type=int, default=900)
    parser.add_argument("--minimum-similarity", type=float, default=0.7)
    parser.add_argument("--claim-page-size", type=int, default=16)
    parser.add_argument("--claim-maximum-gap-seconds", type=int, default=2_592_000)
    parser.add_argument("--claim-minimum-similarity", type=float, default=0.8)
    parser.add_argument("--summary-page-size", type=int, default=16)
    parser.add_argument("--summary-maximum-gap-seconds", type=int, default=2_592_000)
    parser.add_argument("--summary-minimum-similarity", type=float, default=0.8)
    return parser


def _summary_dict(summary: ConsolidationSweepSummary) -> dict[str, object]:
    return {
        "claims": {
            "candidate_count": summary.claims.candidate_count,
            "committed_relationship_count": summary.claims.committed_relationship_count,
            "committed_semantic_claim_count": summary.claims.committed_semantic_claim_count,
            "page_count": summary.claims.page_count,
            "proposed_relationship_count": summary.claims.proposed_relationship_count,
            "proposed_semantic_claim_count": summary.claims.proposed_semantic_claim_count,
            "scanned_count": summary.claims.scanned_count,
        },
        "episodes": {
            "candidate_count": summary.episodes.candidate_count,
            "committed_count": summary.episodes.committed_count,
            "page_count": summary.episodes.page_count,
            "proposed_count": summary.episodes.proposed_count,
            "scanned_count": summary.episodes.scanned_count,
        },
        "summaries": {
            "candidate_count": summary.summaries.candidate_count,
            "committed_count": summary.summaries.committed_count,
            "page_count": summary.summaries.page_count,
            "proposed_count": summary.summaries.proposed_count,
            "scanned_count": summary.summaries.scanned_count,
        },
        "evaluated_at": summary.episodes.evaluated_at.isoformat(),
        "tenant_id": summary.episodes.tenant_id,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
