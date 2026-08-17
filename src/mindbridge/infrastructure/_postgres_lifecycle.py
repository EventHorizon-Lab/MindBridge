"""Concurrent-safe PostgreSQL access and automatic memory evolution."""

from datetime import datetime
from typing import cast

from mindbridge.application.lifecycle import MemoryLifecycleChange
from mindbridge.core import (
    DomainInvariantError,
    MemoryId,
    MemoryRecord,
    TenantId,
    require_aware_datetime,
)
from mindbridge.infrastructure._postgres_memory_rows import (
    MEMORY_NOT_TOMBSTONED_SQL,
    MEMORY_SELECT_SQL,
    MemoryRow,
    memory_from_row,
)
from mindbridge.infrastructure._postgres_types import (
    PostgresStoreOperations,
    tenant_connection,
)


class LifecycleOperations(PostgresStoreOperations):
    async def record_memory_accesses(
        self,
        tenant_id: TenantId,
        memory_ids: tuple[MemoryId, ...],
        *,
        accessed_at: datetime,
    ) -> tuple[MemoryRecord, ...]:
        """Record one recall per selected memory and reactivate cold rows atomically."""
        if not memory_ids:
            return ()
        if len(set(memory_ids)) != len(memory_ids):
            raise DomainInvariantError("accessed memory IDs must be unique")
        require_aware_datetime(accessed_at, "accessed_at")
        async with tenant_connection(self._pool, tenant_id) as connection:
            await connection.execute(
                f"""
                WITH locked AS MATERIALIZED (
                    SELECT memory.memory_id
                    FROM memory_records AS memory
                    WHERE memory.tenant_id = %(tenant_id)s
                      AND memory.memory_id = ANY(%(memory_ids)s::text[])
                      AND memory.superseded_at IS NULL
                      AND {MEMORY_NOT_TOMBSTONED_SQL}
                    ORDER BY memory.memory_id
                    FOR UPDATE
                )
                UPDATE memory_records AS memory
                SET useful_access_count = memory.useful_access_count + 1,
                    last_accessed_at = GREATEST(
                        memory.created_at, memory.last_accessed_at, %(accessed_at)s
                    ),
                    lifecycle_changed_at = GREATEST(
                        memory.lifecycle_changed_at, now(), %(accessed_at)s
                    ),
                    state = CASE WHEN memory.state = 'cold' THEN 'active' ELSE memory.state END
                FROM locked
                WHERE memory.tenant_id = %(tenant_id)s
                  AND memory.memory_id = locked.memory_id
                """,
                {
                    "accessed_at": accessed_at,
                    "memory_ids": list(memory_ids),
                    "tenant_id": tenant_id,
                },
            )
            cursor = await connection.execute(
                f"""
                WITH requested AS (
                    SELECT memory_id, rank
                    FROM unnest(%(memory_ids)s::text[]) WITH ORDINALITY AS item(memory_id, rank)
                )
                {MEMORY_SELECT_SQL}
                JOIN requested ON requested.memory_id = memory.memory_id
                WHERE memory.tenant_id = %(tenant_id)s
                  AND memory.superseded_at IS NULL
                  AND {MEMORY_NOT_TOMBSTONED_SQL}
                ORDER BY requested.rank
                """,
                {"memory_ids": list(memory_ids), "tenant_id": tenant_id},
            )
            return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])

    async def list_memories_for_lifecycle(
        self,
        tenant_id: TenantId,
        *,
        evaluated_at: datetime,
        after_memory_id: MemoryId | None,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Read one stable ID-ordered page without deleted or superseded records."""
        if not 1 <= limit <= 1_001:
            raise DomainInvariantError("lifecycle storage limit must be between 1 and 1001")
        require_aware_datetime(evaluated_at, "lifecycle evaluated_at")
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                f"""
                {MEMORY_SELECT_SQL}
                WHERE memory.tenant_id = %s
                  AND memory.superseded_at IS NULL
                  AND memory.lifecycle_changed_at <= %s
                  AND (%s::text IS NULL OR memory.memory_id > %s)
                  AND {MEMORY_NOT_TOMBSTONED_SQL}
                ORDER BY memory.memory_id
                LIMIT %s
                """,
                (tenant_id, evaluated_at, after_memory_id, after_memory_id, limit),
            )
            return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])

    async def update_memory_lifecycles(
        self,
        changes: tuple[MemoryLifecycleChange, ...],
        *,
        evaluated_at: datetime,
    ) -> int:
        """Apply score/state changes only while every scoring input remains unchanged."""
        if not changes:
            return 0
        require_aware_datetime(evaluated_at, "lifecycle evaluated_at")
        tenant_id = changes[0].previous.tenant_id
        if any(change.previous.tenant_id != tenant_id for change in changes):
            raise DomainInvariantError("one lifecycle update batch cannot cross tenants")
        updated_count = 0
        async with tenant_connection(self._pool, tenant_id) as connection:
            for change in changes:
                previous = change.previous
                cursor = await connection.execute(
                    f"""
                    UPDATE memory_records AS memory
                    SET state = %s, strength = %s,
                        lifecycle_changed_at = GREATEST(
                            memory.lifecycle_changed_at, now(), %s
                        )
                    WHERE memory.tenant_id = %s
                      AND memory.memory_id = %s
                      AND memory.state = %s
                      AND memory.strength = %s
                      AND memory.useful_access_count = %s
                      AND memory.positive_feedback_count = %s
                      AND memory.negative_feedback_count = %s
                      AND memory.last_accessed_at IS NOT DISTINCT FROM %s
                      AND memory.lifecycle_changed_at <= %s
                      AND memory.superseded_at IS NULL
                      AND {MEMORY_NOT_TOMBSTONED_SQL}
                    RETURNING memory.memory_id
                    """,
                    (
                        change.evolved.state.value,
                        change.evolved.strength,
                        evaluated_at,
                        previous.tenant_id,
                        previous.memory_id,
                        previous.state.value,
                        previous.strength,
                        previous.useful_access_count,
                        previous.positive_feedback_count,
                        previous.negative_feedback_count,
                        previous.last_accessed_at,
                        evaluated_at,
                    ),
                )
                updated_count += int(await cursor.fetchone() is not None)
        return updated_count
