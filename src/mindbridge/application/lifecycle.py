"""Bounded, explainable lifecycle evolution for retained memories."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from mindbridge.core import (
    DEFAULT_MEMORY_STRENGTH_POLICY,
    DomainInvariantError,
    MemoryId,
    MemoryRecord,
    MemoryStrengthPolicy,
    TenantId,
    evolve_memory_strength,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


@dataclass(frozen=True, slots=True)
class LifecycleSweepRequest:
    """One stable tenant page evaluated against a single wall-clock instant."""

    tenant_id: TenantId
    evaluated_at: datetime
    after_memory_id: MemoryId | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise DomainInvariantError("tenant_id must not be empty")
        if self.evaluated_at.utcoffset() is None:
            raise DomainInvariantError("evaluated_at must be timezone-aware")
        if self.after_memory_id is not None and not self.after_memory_id.strip():
            raise DomainInvariantError("after_memory_id must not be empty")
        if not 1 <= self.limit <= 1_000:
            raise DomainInvariantError("lifecycle page limit must be between 1 and 1000")


@dataclass(frozen=True, slots=True)
class MemoryLifecycleChange:
    """A score/state-only mutation guarded by the complete prior lifecycle snapshot."""

    previous: MemoryRecord
    evolved: MemoryRecord

    def __post_init__(self) -> None:
        expected = replace(
            self.previous,
            state=self.evolved.state,
            strength=self.evolved.strength,
        )
        if self.evolved != expected:
            raise DomainInvariantError("lifecycle sweep may only change state and strength")


@dataclass(frozen=True, slots=True)
class LifecycleSweepResult:
    """Observable page progress without exposing memory content."""

    evaluated_count: int
    updated_count: int
    next_cursor: MemoryId | None


class MemoryLifecycleStore(Protocol):
    """Persistence boundary for stable lifecycle pages and optimistic updates."""

    async def list_memories_for_lifecycle(
        self,
        tenant_id: TenantId,
        *,
        after_memory_id: MemoryId | None,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def update_memory_lifecycles(
        self,
        changes: tuple[MemoryLifecycleChange, ...],
    ) -> int: ...


class EvolveMemoryLifecycle:
    """Recalculate one bounded page without overwriting concurrent feedback or deletion."""

    def __init__(
        self,
        store: MemoryLifecycleStore,
        policy: MemoryStrengthPolicy = DEFAULT_MEMORY_STRENGTH_POLICY,
    ) -> None:
        self._store = store
        self._policy = policy

    @trace_operation("mindbridge.lifecycle.evolve_page")
    async def run(self, request: LifecycleSweepRequest) -> LifecycleSweepResult:
        """Evaluate one page and return the cursor needed for the next page."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.page.limit": request.limit,
            }
        )
        candidates = await self._store.list_memories_for_lifecycle(
            request.tenant_id,
            after_memory_id=request.after_memory_id,
            limit=request.limit + 1,
        )
        page = candidates[: request.limit]
        changes: list[MemoryLifecycleChange] = []
        for memory in page:
            evolved = evolve_memory_strength(memory, request.evaluated_at, self._policy)
            if evolved == memory:
                continue
            changes.append(
                MemoryLifecycleChange(
                    previous=memory,
                    evolved=evolved,
                )
            )
        updated_count = await self._store.update_memory_lifecycles(tuple(changes)) if changes else 0
        set_current_span_attributes(
            {
                "mindbridge.lifecycle.evaluated_count": len(page),
                "mindbridge.lifecycle.updated_count": updated_count,
            }
        )
        return LifecycleSweepResult(
            evaluated_count=len(page),
            updated_count=updated_count,
            next_cursor=(page[-1].memory_id if len(candidates) > request.limit else None),
        )
