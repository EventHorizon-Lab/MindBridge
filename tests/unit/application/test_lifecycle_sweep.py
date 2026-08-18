"""Tests for bounded automatic memory lifecycle evolution."""

from datetime import datetime, timedelta, timezone

from mindbridge.application.lifecycle import (
    EvolveMemoryLifecycle,
    LifecycleSweepRequest,
    MemoryLifecycleChange,
)
from mindbridge.core import (
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryStrengthPolicy,
    MemoryType,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class RecordingLifecycleStore:
    def __init__(self, memories: tuple[MemoryRecord, ...], *, reject_first: bool = False) -> None:
        self.memories = memories
        self.reject_first = reject_first
        self.changes: tuple[MemoryLifecycleChange, ...] = ()
        self.update_calls = 0

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
        self.update_calls += 1
        self.changes = changes
        return len(changes) - int(self.reject_first and bool(changes))

    async def purge_compressed_clips(self, tenant_id: TenantId, *, limit: int) -> int:
        raise AssertionError("the scoring sweep must not purge derived artifacts")


async def test_lifecycle_sweep_is_bounded_scored_and_cursor_stable() -> None:
    store = RecordingLifecycleStore(
        (
            _memory("memory_01", created_at=NOW - timedelta(days=30)),
            _memory("memory_02", created_at=NOW),
        ),
        reject_first=True,
    )
    use_case = EvolveMemoryLifecycle(
        store,
        MemoryStrengthPolicy(age_decay_weight=1.0, cold_below=0.0),
    )

    first = await use_case.run(
        LifecycleSweepRequest(tenant_id=TenantId("tenant_01"), evaluated_at=NOW, limit=1)
    )
    second = await use_case.run(
        LifecycleSweepRequest(
            tenant_id=TenantId("tenant_01"),
            evaluated_at=NOW,
            after_memory_id=first.next_cursor,
            limit=1,
        )
    )

    assert first.evaluated_count == 1
    assert first.updated_count == 0
    assert first.next_cursor == "memory_01"
    assert second.evaluated_count == 1
    assert second.updated_count == 0
    assert second.next_cursor is None
    assert store.update_calls == 1
    assert store.changes[0].evolved.state is MemoryState.COLD


async def test_lifecycle_sweep_moves_an_old_unused_memory_to_cold() -> None:
    store = RecordingLifecycleStore((_memory("memory_01", created_at=NOW - timedelta(days=30)),))

    result = await EvolveMemoryLifecycle(
        store,
        MemoryStrengthPolicy(age_decay_weight=1.0, cold_below=0.0),
    ).run(LifecycleSweepRequest(tenant_id=TenantId("tenant_01"), evaluated_at=NOW))

    assert result.updated_count == 1
    assert store.changes[0].evolved.state is MemoryState.COLD
    assert store.changes[0].evolved.strength < 0.0


def _memory(memory_id: str, *, created_at: datetime) -> MemoryRecord:
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
