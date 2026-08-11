"""Concurrent-safe PostgreSQL pages for automatic memory evolution."""

from typing import cast

from mindbridge.application import MemoryLifecycleChange
from mindbridge.core import DomainInvariantError, MemoryId, MemoryRecord, TenantId
from mindbridge.infrastructure._postgres_memory_rows import (
    MEMORY_NOT_TOMBSTONED_SQL,
    MEMORY_SELECT_SQL,
    MemoryRow,
    memory_from_row,
)
from mindbridge.infrastructure._postgres_types import DatabasePool


async def list_memories_for_lifecycle(
    pool: DatabasePool,
    tenant_id: TenantId,
    *,
    after_memory_id: MemoryId | None,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Read one stable ID-ordered page without deleted or superseded records."""
    if not 1 <= limit <= 1_001:
        raise DomainInvariantError("lifecycle storage limit must be between 1 and 1001")
    async with pool.connection() as connection:
        cursor = await connection.execute(
            f"""
            {MEMORY_SELECT_SQL}
            WHERE memory.tenant_id = %s
              AND memory.superseded_at IS NULL
              AND (%s::text IS NULL OR memory.memory_id > %s)
              AND {MEMORY_NOT_TOMBSTONED_SQL}
            ORDER BY memory.memory_id
            LIMIT %s
            """,
            (tenant_id, after_memory_id, after_memory_id, limit),
        )
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def update_memory_lifecycles(
    pool: DatabasePool,
    changes: tuple[MemoryLifecycleChange, ...],
) -> int:
    """Apply score/state changes only while every scoring input remains unchanged."""
    updated_count = 0
    async with pool.connection() as connection:
        for change in changes:
            previous = change.previous
            cursor = await connection.execute(
                f"""
                UPDATE memory_records AS memory
                SET state = %s, strength = %s
                WHERE memory.tenant_id = %s
                  AND memory.memory_id = %s
                  AND memory.state = %s
                  AND memory.strength = %s
                  AND memory.useful_access_count = %s
                  AND memory.positive_feedback_count = %s
                  AND memory.negative_feedback_count = %s
                  AND memory.superseded_at IS NULL
                  AND {MEMORY_NOT_TOMBSTONED_SQL}
                RETURNING memory.memory_id
                """,
                (
                    change.evolved.state.value,
                    change.evolved.strength,
                    previous.tenant_id,
                    previous.memory_id,
                    previous.state.value,
                    previous.strength,
                    previous.useful_access_count,
                    previous.positive_feedback_count,
                    previous.negative_feedback_count,
                ),
            )
            updated_count += int(await cursor.fetchone() is not None)
    return updated_count
