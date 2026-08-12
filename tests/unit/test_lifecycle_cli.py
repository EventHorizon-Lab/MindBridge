"""Checks for the scheduled lifecycle command loop."""

import argparse
from datetime import datetime, timedelta, timezone

import pytest

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
        after_memory_id: MemoryId | None,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        assert tenant_id == "tenant_01"
        return tuple(
            memory
            for memory in self.memories
            if after_memory_id is None or memory.memory_id > after_memory_id
        )[:limit]

    async def update_memory_lifecycles(
        self,
        changes: tuple[MemoryLifecycleChange, ...],
    ) -> int:
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
