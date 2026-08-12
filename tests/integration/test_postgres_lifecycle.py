"""PostgreSQL checks for automatic lifecycle pages and concurrent feedback."""

from datetime import datetime, timedelta, timezone

import pytest
from psycopg import AsyncConnection
from psycopg.errors import InsufficientPrivilege

from mindbridge.application.lifecycle import (
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
from mindbridge.infrastructure.postgres import PostgresMemoryStore

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


async def test_postgres_recall_access_reactivates_cold_memory_without_rewinding_time(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_access")
    memory = _memory(tenant_id, "memory_access", NOW - timedelta(days=30))
    await _write_memory(store, memory)
    await EvolveMemoryLifecycle(store, POLICY).run(
        LifecycleSweepRequest(tenant_id=tenant_id, evaluated_at=NOW, limit=10)
    )

    first = await store.record_memory_accesses(
        tenant_id,
        (memory.memory_id,),
        accessed_at=NOW + timedelta(minutes=1),
    )
    second = await store.record_memory_accesses(
        tenant_id,
        (memory.memory_id,),
        accessed_at=NOW,
    )

    assert first[0].state is MemoryState.ACTIVE
    assert first[0].useful_access_count == 1
    assert second[0].useful_access_count == 2
    assert second[0].last_accessed_at == NOW + timedelta(minutes=1)


async def test_postgres_runtime_role_enforces_tenant_row_security(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_a = TenantId("tenant_rls_a")
    tenant_b = TenantId("tenant_rls_b")
    memory_a = _memory(tenant_a, "memory_rls_a", NOW)
    memory_b = _memory(tenant_b, "memory_rls_b", NOW)
    await _write_memory(store, memory_a)
    await _write_memory(store, memory_b)

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        await connection.execute("SET ROLE mindbridge_runtime")
        await connection.execute(
            "SELECT set_config('mindbridge.tenant_id', %s, true)",
            (tenant_a,),
        )
        rows = await (
            await connection.execute(
                "SELECT tenant_id FROM memory_records ORDER BY tenant_id, memory_id"
            )
        ).fetchall()

        assert {row[0] for row in rows} == {tenant_a}
        with pytest.raises(InsufficientPrivilege):
            await connection.execute(
                "UPDATE memory_records SET tenant_id = %s WHERE tenant_id = %s AND memory_id = %s",
                (tenant_b, tenant_a, memory_a.memory_id),
            )


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
