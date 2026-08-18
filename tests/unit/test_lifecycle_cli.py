"""Checks for the scheduled lifecycle command loop."""

import argparse
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge import lifecycle_cli
from mindbridge.application.evidence_clips import ClipReclaimSummary
from mindbridge.application.lifecycle import MemoryLifecycleChange
from mindbridge.configuration import parse_aware_datetime
from mindbridge.core import (
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryStrengthPolicy,
    MemoryType,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure.s3 import S3MediaAccess
from mindbridge.lifecycle_cli import sweep_tenant_lifecycle

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class InMemoryLifecycleStore:
    def __init__(self) -> None:
        self.memories = (
            _memory("memory_01", NOW - timedelta(days=30)),
            _memory("memory_02", NOW),
        )

    async def list_memories_for_lifecycle(
        self,
        tenant_id: TenantId,
        *,
        evaluated_at: datetime,
        after_memory_id: MemoryId | None,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        assert tenant_id == "tenant_01"
        assert evaluated_at == NOW
        return tuple(
            memory
            for memory in self.memories
            if after_memory_id is None or memory.memory_id > after_memory_id
        )[:limit]

    async def update_memory_lifecycles(
        self,
        changes: tuple[MemoryLifecycleChange, ...],
        *,
        evaluated_at: datetime,
    ) -> int:
        assert evaluated_at == NOW
        return len(changes)


async def test_sweep_tenant_lifecycle_accumulates_bounded_pages() -> None:
    summary = await sweep_tenant_lifecycle(
        InMemoryLifecycleStore(),
        TenantId("tenant_01"),
        NOW,
        page_size=1,
        policy=MemoryStrengthPolicy(age_decay_weight=1.0, cold_below=0.0),
    )

    assert summary.page_count == 2
    assert summary.evaluated_count == 2
    assert summary.updated_count == 1


async def test_a_dry_run_never_reaches_the_strength_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`-n` writes nothing, so the sweep that persists strengths must not run at all."""

    async def fail_if_swept(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("the strength sweep ran during a dry run")

    async def reclaim(*_arguments: object, **_keywords: object) -> ClipReclaimSummary:
        return ClipReclaimSummary(
            tenant_id=TenantId("tenant_01"),
            scanned_count=7,
            skipped_recent_count=0,
            reclaimed_count=3,
        )

    monkeypatch.setattr(lifecycle_cli, "sweep_tenant_lifecycle", fail_if_swept)
    monkeypatch.setattr(lifecycle_cli, "PostgresMemoryStore", lambda _url: _OpenableStore())
    monkeypatch.setattr(S3MediaAccess, "from_environment", classmethod(lambda _cls: object()))
    monkeypatch.setattr(lifecycle_cli, "reclaim_orphan_clips_use_case", reclaim)

    summary = await lifecycle_cli._run_postgres_sweep(
        "postgresql://unused",
        TenantId("tenant_01"),
        NOW,
        page_size=100,
        policy=MemoryStrengthPolicy(),
        reclaim_orphan_clips=True,
        dry_run=True,
    )

    assert (summary.page_count, summary.evaluated_count, summary.updated_count) == (0, 0, 0)
    assert summary.reclaimed_clip_count == 3


class _OpenableStore:
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_lifecycle_datetime_requires_an_explicit_timezone() -> None:
    assert parse_aware_datetime("2026-08-12T12:00:00Z") == NOW
    with pytest.raises(argparse.ArgumentTypeError, match="timezone"):
        parse_aware_datetime("2026-08-12T12:00:00")


def _memory(memory_id: str, created_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.EPISODIC,
        summary="A retained event",
        evidence_ids=(),
        occurred_at=created_at,
        ended_at=created_at,
        created_at=created_at,
        verification_status=VerificationStatus.ATTESTED,
        state=MemoryState.ACTIVE,
    )
