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
from mindbridge.application.consolidate_entities import ConsolidateEntities
from mindbridge.application.consolidate_summaries import ConsolidateSummaries
from mindbridge.application.consolidation import ConsolidateEpisodes
from mindbridge.application.consolidation_sweep import (
    ConsolidationSweepSummary,
    SweepSummary,
    consolidate_tenant_claims,
    consolidate_tenant_entities,
    consolidate_tenant_episodes,
    consolidate_tenant_summaries,
)
from mindbridge.application.pipelines import (
    ClaimPipeline,
    EntityResolutionPipeline,
    EpisodePipeline,
    SummaryPipeline,
)
from mindbridge.cli import parser as build_parser
from mindbridge.configuration import (
    copy_plugin_configuration,
    parse_aware_datetime,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
)
from mindbridge.core import EntityType, TenantId, utc_now
from mindbridge.infrastructure.postgres import (
    PostgresMemoryStore,
    resolve_database_max_pool_size,
)
from mindbridge.infrastructure.s3 import (
    ObjectStorageEnvironment,
    S3MediaAccess,
    object_storage_from_environment,
)
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    openai_embedder_config,
    openai_generator_config,
    require_matryoshka_dimension,
)
from mindbridge.models.plugins import close_model, load_embedder, load_generator
from mindbridge.telemetry import configure_telemetry

CONSOLIDATION_ENVIRONMENT = """environment:
  MINDBRIDGE_DATABASE_URL           PostgreSQL DSN (required). Read from the environment
                                    rather than a flag so the DSN never reaches a process
                                    list or this shell's history.
  MINDBRIDGE_OBJECT_STORAGE_BUCKET, MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL,
  MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL
                                    object storage holding the source audio and video
                                    this sweep lets the Generator inspect
  MINDBRIDGE_GENERATOR_PLUGIN, MINDBRIDGE_EMBEDDER_PLUGIN
                                    model plugins to load (default: openai)
  MINDBRIDGE_GENERATOR_API_KEY, MINDBRIDGE_GENERATOR_ENDPOINT
                                    required by the default openai generator plugin;
                                    MINDBRIDGE_GENERATOR_MODEL_ID is optional
  MINDBRIDGE_EMBEDDER_API_KEY, MINDBRIDGE_EMBEDDER_ENDPOINT
                                    required by the default openai embedder plugin;
                                    MINDBRIDGE_EMBEDDER_MODEL_ID is optional
  MINDBRIDGE_GENERATOR_CONFIG_JSON, MINDBRIDGE_EMBEDDER_CONFIG_JSON
                                    explicit plugin configuration; an object here
                                    replaces the per-field variables above"""


@dataclass(frozen=True, slots=True)
class ConsolidationSettings:
    """Validated consolidation process configuration with redacted credentials."""

    database_url: str = field(repr=False)
    object_storage: ObjectStorageEnvironment
    generator_config: Mapping[str, object] = field(repr=False)
    embedder_config: Mapping[str, object] = field(repr=False)
    generator_plugin: str = "openai"
    embedder_plugin: str = "openai"

    def __post_init__(self) -> None:
        if not self.database_url.strip():
            raise ValueError("database_url must not be empty")
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
            object_storage=object_storage_from_environment(source),
            generator_plugin=generator_plugin,
            generator_config=plugin_configuration(
                source,
                "MINDBRIDGE_GENERATOR_CONFIG_JSON",
                (lambda: openai_generator_config(source)) if generator_plugin == "openai" else None,
            ),
            embedder_plugin=embedder_plugin,
            embedder_config=plugin_configuration(
                source,
                "MINDBRIDGE_EMBEDDER_CONFIG_JSON",
                (lambda: openai_embedder_config(source)) if embedder_plugin == "openai" else None,
            ),
        )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run one tenant sweep using only explicit process configuration."""
    options = _parser(prog).parse_args(argv)
    # Configured after parsing so --help and a rejected flag stay side-effect free.
    configure_telemetry("mindbridge-consolidation")
    summary = asyncio.run(
        _run_postgres_sweep(
            ConsolidationSettings.from_environment(),
            TenantId(options.tenant_id),
            options.evaluated_at or utc_now(),
            page_size=options.page_size,
            maximum_gap_seconds=options.maximum_gap_seconds,
            minimum_similarity=options.minimum_similarity,
            claim_page_size=options.claim_page_size,
            claim_maximum_gap_seconds=options.claim_maximum_gap_seconds,
            claim_minimum_similarity=options.claim_minimum_similarity,
            summary_page_size=options.summary_page_size,
            summary_maximum_gap_seconds=options.summary_maximum_gap_seconds,
            summary_minimum_similarity=options.summary_minimum_similarity,
            entity_page_size=options.entity_page_size,
            entity_maximum_gap_seconds=options.entity_maximum_gap_seconds,
            entity_candidate_limit=options.entity_candidate_limit,
            entity_minimum_confidence=options.entity_minimum_confidence,
            entity_evidence_per_side=options.entity_evidence_per_side,
            entity_maximum_pairs=options.entity_maximum_pairs,
            # argparse append leaves None when the flag is absent, and the default is a
            # deliberate choice rather than "all types", so it is spelled out here.
            entity_types=tuple(
                EntityType(value) for value in (options.entity_types or [EntityType.PERSON.value])
            ),
            entity_readjudicate=options.entity_readjudicate,
            skip_entity_resolution=options.skip_entity_resolution,
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
    entity_page_size: int,
    entity_maximum_gap_seconds: int,
    entity_candidate_limit: int,
    entity_minimum_confidence: float,
    entity_evidence_per_side: int,
    entity_maximum_pairs: int,
    entity_types: tuple[EntityType, ...],
    entity_readjudicate: bool,
    skip_entity_resolution: bool,
) -> ConsolidationSweepSummary:
    store = PostgresMemoryStore(
        settings.database_url,
        embedding_dimension=require_matryoshka_dimension(
            int(cast(int, settings.embedder_config.get("dimension", DEFAULT_EMBEDDING_DIMENSION)))
        ),
        max_pool_size=resolve_database_max_pool_size(),
    )
    media_access = S3MediaAccess(settings.object_storage)
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
        # The only sweep here that opens media and spends a generator call per candidate
        # pair, so it is also the only one worth being able to turn off without dropping the
        # rest of the run.
        entities = (
            None
            if skip_entity_resolution
            else await consolidate_tenant_entities(
                ConsolidateEntities(
                    store,
                    EntityResolutionPipeline(generator),
                    media_url_signer=media_access,
                ),
                tenant_id,
                evaluated_at,
                page_size=entity_page_size,
                maximum_gap_seconds=entity_maximum_gap_seconds,
                candidate_limit=entity_candidate_limit,
                minimum_confidence=entity_minimum_confidence,
                evidence_per_side=entity_evidence_per_side,
                maximum_pairs=entity_maximum_pairs,
                entity_types=entity_types,
                readjudicate=entity_readjudicate,
            )
        )
        return ConsolidationSweepSummary(
            episodes=episodes,
            claims=claims,
            summaries=summaries,
            entities=entities,
        )


def _parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = build_parser(prog=prog, description=__doc__, epilog=CONSOLIDATION_ENVIRONMENT)
    parser.add_argument("--tenant-id", required=True, help="tenant whose memories are swept")
    parser.add_argument(
        "--evaluated-at",
        type=parse_aware_datetime,
        metavar="TIMESTAMP",
        help="the one aware instant this whole sweep evaluates at (default: now)",
    )
    parser.add_argument(
        "--page-size", type=int, default=16, help="Episode candidates per bounded page"
    )
    parser.add_argument(
        "--maximum-gap-seconds",
        type=int,
        default=900,
        help="longest silence two Events may span and still join one Episode",
    )
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.7,
        help="lowest similarity two Events may have and still join one Episode",
    )
    parser.add_argument(
        "--claim-page-size", type=int, default=16, help="Claim candidates per bounded page"
    )
    parser.add_argument(
        "--claim-maximum-gap-seconds",
        type=int,
        default=2_592_000,
        help="longest span two memories may cover and still support one Claim",
    )
    parser.add_argument(
        "--claim-minimum-similarity",
        type=float,
        default=0.8,
        help="lowest similarity two memories may have and still support one Claim",
    )
    parser.add_argument(
        "--summary-page-size", type=int, default=16, help="Summary candidates per bounded page"
    )
    parser.add_argument(
        "--summary-maximum-gap-seconds",
        type=int,
        default=2_592_000,
        help="longest span two memories may cover and still join one Summary",
    )
    parser.add_argument(
        "--summary-minimum-similarity",
        type=float,
        default=0.8,
        help="lowest similarity two memories may have and still join one Summary",
    )
    parser.add_argument(
        "--entity-page-size", type=int, default=16, help="entity seeds per bounded page"
    )
    parser.add_argument(
        "--entity-maximum-gap-seconds",
        type=int,
        default=2_592_000,
        help="longest span two entities may cover and still be judged the same entity",
    )
    parser.add_argument(
        "--entity-candidate-limit",
        type=int,
        default=8,
        help="peers per entity seed, taken in order of vector affinity",
    )
    parser.add_argument(
        "--entity-minimum-confidence",
        type=float,
        default=0.75,
        help="lowest confidence a verdict may carry and still be recorded either way",
    )
    parser.add_argument(
        "--entity-evidence-per-side",
        type=int,
        default=3,
        help="evidence spans reopened per entity when judging one pair",
    )
    parser.add_argument(
        "--entity-maximum-pairs",
        type=int,
        default=64,
        help="pairs judged per page; the rest are reported as dropped, not hidden",
    )
    parser.add_argument(
        "--entity-type",
        dest="entity_types",
        action="append",
        choices=[item.value for item in EntityType],
        help="entity type to adjudicate; repeatable, default person only",
    )
    parser.add_argument(
        "--entity-readjudicate",
        action="store_true",
        help="re-judge pairs that already carry a verdict, replacing it",
    )
    parser.add_argument(
        "--skip-entity-resolution",
        action="store_true",
        help=(
            "skip the entity-resolution sweep; the other three still run. It is the only "
            "sweep that opens media and spends a generator call per candidate pair"
        ),
    )
    return parser


def _summary_dict(summary: ConsolidationSweepSummary) -> dict[str, object]:
    return {
        "claims": _sweep_dict(summary.claims),
        "episodes": _sweep_dict(summary.episodes),
        "entities": None if summary.entities is None else _sweep_dict(summary.entities),
        "summaries": _sweep_dict(summary.summaries),
        "evaluated_at": summary.episodes.evaluated_at.isoformat(),
        "tenant_id": summary.episodes.tenant_id,
    }


def _sweep_dict(sweep: SweepSummary) -> dict[str, int]:
    return {
        "candidate_count": sweep.candidate_count,
        "page_count": sweep.page_count,
        "scanned_count": sweep.scanned_count,
        **sweep.counts,
    }


if __name__ == "__main__":
    main()
