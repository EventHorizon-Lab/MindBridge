"""Scheduled command for explainable memory decay and state transitions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from mindbridge.application.evidence_clips import (
    reclaim_orphan_clips as reclaim_orphan_clips_use_case,
)
from mindbridge.application.lifecycle import (
    EvolveMemoryLifecycle,
    LifecycleSweepRequest,
    MemoryLifecycleStore,
)
from mindbridge.configuration import (
    parse_aware_datetime,
    require_environment_value,
)
from mindbridge.core import (
    DEFAULT_MEMORY_STRENGTH_POLICY,
    MemoryId,
    MemoryIntegrityError,
    MemoryStrengthPolicy,
    TenantId,
    utc_now,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import S3MediaAccess
from mindbridge.telemetry import configure_telemetry


@dataclass(frozen=True, slots=True)
class LifecycleSweepSummary:
    """Content-free operational totals for one complete tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    evaluated_count: int
    updated_count: int
    reclaimed_clip_count: int | None = None
    purged_clip_count: int | None = None


async def purge_tenant_compressed_clips(
    store: MemoryLifecycleStore,
    tenant_id: TenantId,
    *,
    page_size: int,
) -> int:
    """Drop clips behind compressed memories until no purgeable page remains."""
    purged = 0
    while page := await store.purge_compressed_clips(tenant_id, limit=page_size):
        purged += page
    return purged


async def sweep_tenant_lifecycle(
    store: MemoryLifecycleStore,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    policy: MemoryStrengthPolicy = DEFAULT_MEMORY_STRENGTH_POLICY,
) -> LifecycleSweepSummary:
    """Evaluate stable pages at one instant until the tenant scan is complete."""
    use_case = EvolveMemoryLifecycle(store, policy)
    cursor: MemoryId | None = None
    page_count = evaluated_count = updated_count = 0
    while True:
        result = await use_case.run(
            LifecycleSweepRequest(
                tenant_id=tenant_id,
                evaluated_at=evaluated_at,
                after_memory_id=cursor,
                limit=page_size,
            )
        )
        page_count += 1
        evaluated_count += result.evaluated_count
        updated_count += result.updated_count
        if result.next_cursor is None:
            break
        if result.next_cursor == cursor:
            raise MemoryIntegrityError("lifecycle cursor did not advance")
        cursor = result.next_cursor
    return LifecycleSweepSummary(
        tenant_id=tenant_id,
        evaluated_at=evaluated_at,
        page_count=page_count,
        evaluated_count=evaluated_count,
        updated_count=updated_count,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one tenant sweep using PostgreSQL configured by the process environment."""
    configure_telemetry("mindbridge-lifecycle")
    options = _parser().parse_args(argv)
    policy = MemoryStrengthPolicy(
        access_weight=options.access_weight,
        positive_feedback_weight=options.positive_feedback_weight,
        negative_feedback_weight=options.negative_feedback_weight,
        age_decay_weight=options.age_decay_weight,
        strengthen_at=options.strengthen_at,
        cold_below=options.cold_below,
        compress_below=options.compress_below,
    )
    summary = asyncio.run(
        _run_postgres_sweep(
            require_environment_value(os.environ, "MINDBRIDGE_DATABASE_URL"),
            TenantId(options.tenant_id),
            options.evaluated_at,
            page_size=options.page_size,
            policy=policy,
            reclaim_orphan_clips=options.reclaim_orphan_clips,
        )
    )
    print(
        json.dumps(
            {
                "evaluated_at": summary.evaluated_at.isoformat(),
                "evaluated_count": summary.evaluated_count,
                "page_count": summary.page_count,
                "tenant_id": summary.tenant_id,
                "updated_count": summary.updated_count,
                "reclaimed_clip_count": summary.reclaimed_clip_count,
                "purged_clip_count": summary.purged_clip_count,
            },
            sort_keys=True,
        )
    )


async def _run_postgres_sweep(
    database_url: str,
    tenant_id: TenantId,
    evaluated_at: datetime,
    *,
    page_size: int,
    policy: MemoryStrengthPolicy,
    reclaim_orphan_clips: bool = False,
) -> LifecycleSweepSummary:
    # Build object storage before the sweep so a missing variable fails fast
    # instead of after every memory in the tenant has already been evaluated.
    media_access = S3MediaAccess.from_environment() if reclaim_orphan_clips else None
    store = PostgresMemoryStore(database_url)
    await store.open()
    try:
        summary = await sweep_tenant_lifecycle(
            store,
            tenant_id,
            evaluated_at,
            page_size=page_size,
            policy=policy,
        )
        if policy.compress_below is not None:
            # Purge the rows first: that leaves each clip's content-addressed key an orphan, which
            # the reclaim pass below then deletes from object storage.
            summary = replace(
                summary,
                purged_clip_count=await purge_tenant_compressed_clips(
                    store, tenant_id, page_size=page_size
                ),
            )
        if media_access is None:
            return summary
        reclaimed = await reclaim_orphan_clips_use_case(
            tenant_id, janitor=media_access, digests=store
        )
        return replace(summary, reclaimed_clip_count=reclaimed.reclaimed_count)
    finally:
        await store.close()


def _parser() -> argparse.ArgumentParser:
    policy = DEFAULT_MEMORY_STRENGTH_POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--evaluated-at", type=parse_aware_datetime, default=utc_now())
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--reclaim-orphan-clips",
        action="store_true",
        help=(
            "also delete derived clip objects no committed record references; "
            "requires the object storage variables"
        ),
    )
    parser.add_argument("--access-weight", type=float, default=policy.access_weight)
    parser.add_argument(
        "--positive-feedback-weight", type=float, default=policy.positive_feedback_weight
    )
    parser.add_argument(
        "--negative-feedback-weight", type=float, default=policy.negative_feedback_weight
    )
    parser.add_argument(
        "--age-decay-weight",
        type=float,
        default=policy.age_decay_weight,
        help=(
            "strength lost per idle day; idle days to cold are salience divided by this, "
            "so the default cools a 0.5-salience memory after 100 unused days"
        ),
    )
    parser.add_argument("--strengthen-at", type=float, default=policy.strengthen_at)
    parser.add_argument("--cold-below", type=float, default=policy.cold_below)
    parser.add_argument(
        "--compress-below",
        type=float,
        default=policy.compress_below,
        help=(
            "strength at or under which a cold memory also drops its rebuildable clips "
            "(must not exceed --cold-below); omit to leave compression off. Pair with "
            "--reclaim-orphan-clips to also delete the clip objects the purge orphans"
        ),
    )
    return parser


if __name__ == "__main__":
    main()
