"""Scheduled command for explainable memory decay and state transitions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from mindbridge.application import (
    EvolveMemoryLifecycle,
    LifecycleSweepRequest,
    MemoryLifecycleStore,
)
from mindbridge.configuration import require_environment_value
from mindbridge.core import (
    DEFAULT_MEMORY_STRENGTH_POLICY,
    MemoryId,
    MemoryIntegrityError,
    MemoryStrengthPolicy,
    TenantId,
)
from mindbridge.infrastructure import PostgresMemoryStore


@dataclass(frozen=True, slots=True)
class LifecycleSweepSummary:
    """Content-free operational totals for one complete tenant sweep."""

    tenant_id: TenantId
    evaluated_at: datetime
    page_count: int
    evaluated_count: int
    updated_count: int


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
    options = _parser().parse_args(argv)
    policy = MemoryStrengthPolicy(
        access_weight=options.access_weight,
        positive_feedback_weight=options.positive_feedback_weight,
        negative_feedback_weight=options.negative_feedback_weight,
        age_decay_weight=options.age_decay_weight,
        strengthen_at=options.strengthen_at,
        cold_below=options.cold_below,
    )
    summary = asyncio.run(
        _run_postgres_sweep(
            require_environment_value(os.environ, "MINDBRIDGE_DATABASE_URL"),
            TenantId(options.tenant_id),
            options.evaluated_at,
            page_size=options.page_size,
            policy=policy,
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
) -> LifecycleSweepSummary:
    store = PostgresMemoryStore(database_url)
    await store.open()
    try:
        return await sweep_tenant_lifecycle(
            store,
            tenant_id,
            evaluated_at,
            page_size=page_size,
            policy=policy,
        )
    finally:
        await store.close()


def _parser() -> argparse.ArgumentParser:
    policy = DEFAULT_MEMORY_STRENGTH_POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--evaluated-at", type=_aware_datetime, default=_utc_now())
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--access-weight", type=float, default=policy.access_weight)
    parser.add_argument(
        "--positive-feedback-weight", type=float, default=policy.positive_feedback_weight
    )
    parser.add_argument(
        "--negative-feedback-weight", type=float, default=policy.negative_feedback_weight
    )
    parser.add_argument("--age-decay-weight", type=float, default=policy.age_decay_weight)
    parser.add_argument("--strengthen-at", type=float, default=policy.strengthen_at)
    parser.add_argument("--cold-below", type=float, default=policy.cold_below)
    return parser


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 datetime") from error
    if parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("must include a timezone offset")
    return parsed


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
