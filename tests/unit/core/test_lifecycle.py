"""Transparent memory-strength policy checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from mindbridge.core import (
    DEFAULT_MEMORY_STRENGTH_POLICY,
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


def test_idle_memories_cool_within_a_usable_window_across_the_salience_range() -> None:
    """A logarithmic age term put salience 0.5 roughly 60 years from COLD, and 0.9 far beyond.

    The horizon has to land inside a retention window an operator can actually name, for every
    salience the perceiver produces — not just for the low tail.
    """
    for salience in (0.2, 0.5, 0.9):
        memory = replace(_memory(), salience=salience)

        assert evolve_memory_strength(memory, NOW + timedelta(days=7)).state is not MemoryState.COLD
        assert evolve_memory_strength(memory, NOW + timedelta(days=365)).state is MemoryState.COLD


def test_a_useful_access_buys_back_idle_time() -> None:
    """Decay must not cool a memory the system keeps successfully retrieving."""
    used = replace(_memory(), useful_access_count=1)

    assert evolve_memory_strength(used, NOW + timedelta(days=120)).state is not MemoryState.COLD


def test_compression_is_reachable_only_when_configured_and_never_reverses() -> None:
    """COMPRESSED was a state nothing in the codebase could ever write."""
    idle = replace(_memory(), created_at=NOW)
    policy = replace(DEFAULT_MEMORY_STRENGTH_POLICY, compress_below=-0.5)

    assert evolve_memory_strength(idle, NOW + timedelta(days=400)).state is MemoryState.COLD
    compressed = evolve_memory_strength(idle, NOW + timedelta(days=400), policy)

    assert compressed.state is MemoryState.COMPRESSED
    # Cold at 150 days, but not yet decayed past the compression threshold.
    assert evolve_memory_strength(idle, NOW + timedelta(days=150), policy).state is MemoryState.COLD
    # Recall revives cold; nothing re-derives discarded clips, so compression stays put.
    revived = replace(compressed, useful_access_count=5, positive_feedback_count=3)
    assert evolve_memory_strength(revived, NOW + timedelta(days=400), policy).state is (
        MemoryState.COMPRESSED
    )


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
