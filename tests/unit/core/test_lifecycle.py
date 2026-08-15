"""Transparent memory-strength policy checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from mindbridge.core import (
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryType,
    TenantId,
    VerificationStatus,
    calculate_memory_strength,
    evolve_memory_strength,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_feedback_and_access_signals_drive_explicit_lifecycle_states() -> None:
    memory = _memory()

    strengthened = evolve_memory_strength(
        replace(memory, positive_feedback_count=1),
        NOW,
    )
    cold = evolve_memory_strength(
        replace(memory, negative_feedback_count=1),
        NOW,
    )
    revived = evolve_memory_strength(
        replace(cold, useful_access_count=2),
        NOW,
    )

    assert strengthened.state is MemoryState.STRENGTHENED
    assert strengthened.strength == 1.5
    assert cold.state is MemoryState.COLD
    assert cold.strength == -0.5
    assert revived.state is MemoryState.ACTIVE
    assert calculate_memory_strength(memory, NOW + timedelta(days=365)) < memory.salience


def test_recent_access_resets_age_decay() -> None:
    old_memory = replace(
        _memory(),
        created_at=NOW - timedelta(days=365),
        last_accessed_at=NOW,
    )

    assert calculate_memory_strength(old_memory, NOW) == old_memory.salience


def _memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId("memory_01"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.EPISODIC,
        summary="The robot moved the tool.",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.ATTESTED,
    )
