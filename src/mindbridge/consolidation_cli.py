"""Scheduled evidence-verified Episode, Claim, and Summary consolidation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

from mindbridge.application.consolidate_claims import ConsolidateClaims
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import ConsolidateEpisodes
from mindbridge.application.consolidation_sweep import (
    ConsolidationSweepSummary,
    consolidate_tenant_claims,
    consolidate_tenant_episodes,
    consolidate_tenant_summaries,
)
from mindbridge.application.pipelines import ClaimPipeline, EpisodePipeline, SummaryPipeline
from mindbridge.configuration import (
    copy_plugin_configuration,
    optional_environment_value,
    parse_aware_datetime,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
)
from mindbridge.core import TenantId, utc_now
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import S3MediaAccess
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
    DEFAULT_GENERATOR_MODEL_ID,
    embedding_dimension_from_environment,
    require_matryoshka_dimension,
)
from mindbridge.models.plugins import close_model, load_embedder, load_generator
from mindbridge.telemetry import configure_telemetry


@dataclass(frozen=True, slots=True)
class ConsolidationSettings:
    """Validated consolidation process configuration with redacted credentials."""

    database_url: str = field(repr=False)
    object_storage_bucket: str
    generator_config: Mapping[str, object] = field(repr=False)
    embedder_config: Mapping[str, object] = field(repr=False)
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "us-east-1"
    generator_plugin: str = "openai"
    embedder_plugin: str = "openai"

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("object_storage_bucket", self.object_storage_bucket),
            ("object_storage_region", self.object_storage_region),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name, value in (
            ("generator_plugin", self.generator_plugin),
            ("embedder_plugin", self.embedder_plugin),
        ):
            validate_plugin_name(value, name)
        for name, config in (
            ("generator_config", self.generator_config),
            ("embedder_config", self.embedder_config),
        ):
            object.__setattr__(self, name, copy_plugin_configuration(config, name))
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
        generator_plugin = source.get("MINDBRIDGE_GENERATOR_PLUGIN", "openai")
        embedder_plugin = source.get("MINDBRIDGE_EMBEDDER_PLUGIN", "openai")
        return cls(
            database_url=require_environment_value(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage_bucket=require_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"
            ),
            object_storage_endpoint_url=optional_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"
            ),
            object_storage_region=source.get("MINDBRIDGE_OBJECT_STORAGE_REGION", "us-east-1"),
            generator_plugin=generator_plugin,
            generator_config=plugin_configuration(
                source,
                "MINDBRIDGE_GENERATOR_CONFIG_JSON",
                (lambda: _generator_config(source)) if generator_plugin == "openai" else None,
            ),
            embedder_plugin=embedder_plugin,
            embedder_config=plugin_configuration(
                source,
                "MINDBRIDGE_EMBEDDER_CONFIG_JSON",
                (lambda: _document_embedder_config(source))
                if embedder_plugin == "openai"
                else None,
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
    store = PostgresMemoryStore(
        settings.database_url,
        embedding_dimension=require_matryoshka_dimension(
            int(cast(int, settings.embedder_config.get("dimension", DEFAULT_EMBEDDING_DIMENSION)))
        ),
    )
    media_access = S3MediaAccess(
        settings.object_storage_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        region_name=settings.object_storage_region,
    )
    async with AsyncExitStack() as resources:
        generator = load_generator(settings.generator_plugin, settings.generator_config)
        resources.push_async_callback(close_model, generator)
        embedder = load_embedder(settings.embedder_plugin, settings.embedder_config)
        resources.push_async_callback(close_model, embedder)
        await store.open()
        resources.push_async_callback(store.close)
        episodes = await consolidate_tenant_episodes(
            ConsolidateEpisodes(
                store,
                EpisodePipeline(generator),
                embedder,
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
                ClaimPipeline(generator),
                embedder,
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
                SummaryPipeline(generator),
                embedder,
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
    parser.add_argument("--evaluated-at", type=parse_aware_datetime, default=utc_now())
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


def _generator_config(source: Mapping[str, str]) -> Mapping[str, object]:
    return {
        "api_key": require_environment_value(source, "MINDBRIDGE_GENERATOR_API_KEY"),
        "endpoint": require_environment_value(source, "MINDBRIDGE_GENERATOR_ENDPOINT"),
        "model_id": source.get("MINDBRIDGE_GENERATOR_MODEL_ID", DEFAULT_GENERATOR_MODEL_ID),
        "model_revision": require_environment_value(source, "MINDBRIDGE_GENERATOR_MODEL_REVISION"),
    }


def _document_embedder_config(source: Mapping[str, str]) -> Mapping[str, object]:
    return {
        "api_key": require_environment_value(source, "MINDBRIDGE_EMBEDDER_API_KEY"),
        "endpoint": require_environment_value(source, "MINDBRIDGE_EMBEDDER_ENDPOINT"),
        "model_id": source.get("MINDBRIDGE_EMBEDDER_MODEL_ID", DEFAULT_EMBEDDER_MODEL_ID),
        "model_revision": source.get(
            "MINDBRIDGE_EMBEDDER_MODEL_REVISION", DEFAULT_EMBEDDER_REVISION
        ),
        "space_id": source.get("MINDBRIDGE_EMBEDDING_SPACE_ID", DEFAULT_EMBEDDING_SPACE.space_id),
        "space_revision": source.get(
            "MINDBRIDGE_EMBEDDING_SPACE_REVISION", DEFAULT_EMBEDDING_SPACE.revision
        ),
        "dimension": embedding_dimension_from_environment(source),
    }


if __name__ == "__main__":
    main()
