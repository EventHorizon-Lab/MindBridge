"""PostgreSQL checks for automatic lifecycle pages and concurrent feedback."""

from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application import (
    EvolveMemoryLifecycle,
    LifecycleSweepRequest,
    MemoryLifecycleChange,
)
from mindbridge.core import (
    FeedbackId,
    FeedbackType,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryStrengthPolicy,
    MemoryType,
    TenantId,
    VerificationStatus,
    evolve_memory_strength,
)
from mindbridge.infrastructure import PostgresMemoryStore

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
POLICY = MemoryStrengthPolicy(age_decay_weight=1.0, cold_below=0.0)
pytestmark = pytest.mark.integration


async def test_postgres_lifecycle_sweep_pages_and_cools_unused_memory(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_page")
    old = _memory(tenant_id, "memory_01", NOW - timedelta(days=30))
    recent = _memory(tenant_id, "memory_02", NOW)
    await _write_memory(store, old)
    await _write_memory(store, recent)
    use_case = EvolveMemoryLifecycle(store, POLICY)

    first = await use_case.run(
        LifecycleSweepRequest(tenant_id=tenant_id, evaluated_at=NOW, limit=1)
    )
    second = await use_case.run(
        LifecycleSweepRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW,
            after_memory_id=first.next_cursor,
            limit=1,
        )
    )

    assert first.evaluated_count == 1
    assert first.updated_count == 1
    assert first.next_cursor == "memory_01"
    assert second.evaluated_count == 1
    assert second.updated_count == 0
    assert second.next_cursor is None
    assert (await store.read_memory(tenant_id, old.memory_id)).state is MemoryState.COLD
    assert (await store.read_memory(tenant_id, recent.memory_id)).state is MemoryState.ACTIVE


async def test_postgres_lifecycle_does_not_overwrite_concurrent_feedback(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_concurrent")
    memory = _memory(tenant_id, "memory_01", NOW - timedelta(days=30))
    await _write_memory(store, memory)
    stale_change = MemoryLifecycleChange(
        previous=memory,
        evolved=evolve_memory_strength(memory, NOW, POLICY),
    )
    await store.record_feedback(
        MemoryFeedback(
            feedback_id=FeedbackId("feedback_01"),
            tenant_id=tenant_id,
            feedback_type=FeedbackType.USEFUL,
            memory_id=memory.memory_id,
            created_at=NOW,
        ),
        None,
        idempotency_key="feedback_01",
        content_digest="f" * 64,
    )

    updated_count = await store.update_memory_lifecycles((stale_change,))
    stored = await store.read_memory(tenant_id, memory.memory_id)

    assert updated_count == 0
    assert stored.positive_feedback_count == 1
    assert stored.state is MemoryState.STRENGTHENED


async def _write_memory(store: PostgresMemoryStore, memory: MemoryRecord) -> None:
    await store.write_memory(
        memory,
        idempotency_key=memory.memory_id,
        content_digest=("a" if memory.memory_id == "memory_01" else "b") * 64,
    )


def _memory(tenant_id: TenantId, memory_id: str, created_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=tenant_id,
        memory_type=MemoryType.EPISODIC,
        summary=f"Retained event {memory_id}",
        evidence_ids=(),
        occurred_at=created_at,
        ended_at=created_at,
        created_at=created_at,
        verification_status=VerificationStatus.ATTESTED,
    )
