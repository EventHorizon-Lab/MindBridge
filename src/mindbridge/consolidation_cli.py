"""Scheduled evidence-verified Episode consolidation for one tenant."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mindbridge.application import (
    ConsolidateEpisodes,
    EpisodeCandidateRequest,
)
from mindbridge.configuration import (
    optional_environment_value,
    parse_aware_datetime,
    require_environment_value,
)
from mindbridge.core import EmbeddingSpaceReference, EventId, MemoryIntegrityError, TenantId
from mindbridge.infrastructure import PostgresMemoryStore, S3MediaAccess
from mindbridge.models import (
    DEFAULT_JINA_RETRIEVAL_SPACE,
    DEFAULT_JINA_TEXT_MODEL_ID,
    DEFAULT_JINA_TEXT_REVISION,
    DEFAULT_OMNI_MODEL_ID,
    OpenAIJinaTextEmbedder,
    OpenAIOmniEpisodeConsolidator,
)
from mindbridge.telemetry import configure_telemetry


@dataclass(frozen=True, slots=True)
class ConsolidationSettings:
    """Validated Episode process configuration with redacted credentials."""

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


@dataclass(frozen=True, slots=True)
class EpisodeSweepSummary:
    """Content-free operational totals for one complete tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    scanned_count: int
    candidate_count: int
    proposed_count: int
    committed_count: int


async def consolidate_tenant_episodes(
    use_case: ConsolidateEpisodes,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    maximum_gap_seconds: int,
    minimum_similarity: float,
) -> EpisodeSweepSummary:
    """Consolidate stable candidate pages at one fixed evaluation instant."""
    cursor: EventId | None = None
    page_count = scanned_count = candidate_count = proposed_count = committed_count = 0
    while True:
        result = await use_case.run(
            EpisodeCandidateRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_event_id=cursor,
                limit=page_size,
                maximum_gap_seconds=maximum_gap_seconds,
                minimum_similarity=minimum_similarity,
            )
        )
        page_count += 1
        scanned_count += result.scanned_count
        candidate_count += result.candidate_count
        proposed_count += result.proposed_count
        committed_count += result.committed_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("Episode consolidation cursor did not advance")
        cursor = result.next_cursor
    return EpisodeSweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        scanned_count=scanned_count,
        candidate_count=candidate_count,
        proposed_count=proposed_count,
        committed_count=committed_count,
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
) -> EpisodeSweepSummary:
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
        resources.push_async_callback(text_embedder.close)
        await store.open()
        resources.push_async_callback(store.close)
        return await consolidate_tenant_episodes(
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--evaluated-at", type=parse_aware_datetime, default=_utc_now())
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--maximum-gap-seconds", type=int, default=900)
    parser.add_argument("--minimum-similarity", type=float, default=0.7)
    return parser


def _summary_dict(summary: EpisodeSweepSummary) -> dict[str, object]:
    return {
        "candidate_count": summary.candidate_count,
        "committed_count": summary.committed_count,
        "evaluated_at": summary.evaluated_at.isoformat(),
        "page_count": summary.page_count,
        "proposed_count": summary.proposed_count,
        "scanned_count": summary.scanned_count,
        "tenant_id": summary.tenant_id,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
