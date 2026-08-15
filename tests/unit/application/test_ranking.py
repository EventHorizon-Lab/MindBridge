"""Checks for score-independent memory rank fusion."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application.ranking import fuse_memory_rankings, order_memory_candidates
from mindbridge.core import (
    DomainInvariantError,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_rrf_promotes_memory_supported_by_dense_and_sparse_search() -> None:
    """Agreement across retrievers outranks a candidate strong in only one list."""
    first = _memory("memory_01")
    shared = _memory("memory_shared")
    last = _memory("memory_03")

    fused = fuse_memory_rankings(
        ((first, shared), (last, shared)),
        limit=3,
    )

    assert [memory.memory_id for memory in fused] == ["memory_shared", "memory_01", "memory_03"]


def test_rrf_rejects_conflicting_identity() -> None:
    """Fusion cannot hide contradictory records returned by two indexes."""
    memory = _memory("memory_01")

    with pytest.raises(MemoryIntegrityError, match="conflicting candidate content"):
        fuse_memory_rankings(
            ((memory,), (replace(memory, summary="different"),)),
            limit=1,
        )


def test_rrf_accepts_concurrent_lifecycle_updates() -> None:
    """Concurrent recalls may update lifecycle counters between candidate queries."""
    memory = _memory("memory_01")
    accessed = replace(
        memory,
        state=MemoryState.STRENGTHENED,
        strength=1.5,
        useful_access_count=1,
        positive_feedback_count=1,
        negative_feedback_count=1,
        last_accessed_at=NOW,
    )

    assert fuse_memory_rankings(((memory,), (accessed,)), limit=1) == (memory,)


def test_rrf_rejects_invalid_budget() -> None:
    """A caller must make its bounded candidate budget explicit."""
    with pytest.raises(DomainInvariantError, match="must be positive"):
        fuse_memory_rankings((), limit=0)


def test_temporal_order_only_reorders_the_bounded_relevance_set() -> None:
    oldest = _memory("memory_oldest")
    newest = replace(
        _memory("memory_newest"),
        occurred_at=NOW + timedelta(days=1),
        ended_at=NOW + timedelta(days=1),
    )

    assert order_memory_candidates((oldest, newest), "newest") == (newest, oldest)
    assert order_memory_candidates((newest, oldest), "oldest") == (oldest, newest)
    assert order_memory_candidates((oldest, newest), "relevance") == (oldest, newest)


def _memory(memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.EPISODIC,
        summary=f"summary for {memory_id}",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.UNVERIFIED,
    )
