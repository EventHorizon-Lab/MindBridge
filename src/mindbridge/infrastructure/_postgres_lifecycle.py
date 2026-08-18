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

    async def purge_compressed_clips(self, tenant_id: TenantId, *, limit: int) -> int:
        """Drop the rebuildable clips behind fully compressed memories, one bounded page.

        Re-runnable: the persisted `COMPRESSED` state is the intent, so a crash mid-purge heals on
        the next sweep. Deleting a clip's content-addressed `media_objects` row orphans its storage
        key, which `--reclaim-orphan-clips` then deletes. Source media, evidence spans, and evidence
        vectors stay: recall signs the source object, never the clip. Returns clip rows deleted, so
        the caller's page loop terminates.
        """
        if limit <= 0:
            raise DomainInvariantError("clip purge limit must be positive")
        async with tenant_connection(self._pool, tenant_id) as connection:
            # Two statements rather than one chain of data-modifying CTEs: those all read the same
            # snapshot, so the orphan check below would never see this delete and would never drop
            # a media row. Both run in the one transaction tenant_connection already opens.
            cursor = await connection.execute(
                """
                DELETE FROM evidence_clips AS clip
                USING (
                    SELECT candidate.evidence_id, candidate.ordinal
                    FROM evidence_clips AS candidate
                    WHERE candidate.tenant_id = %(tenant_id)s
                      -- Some memory must already cite this evidence. Without this an evidence
                      -- span that no memory references yet satisfies the NOT EXISTS below
                      -- vacuously, and a freshly observed span loses its clips.
                      AND EXISTS (
                          SELECT 1 FROM memory_evidence AS link
                          WHERE link.tenant_id = candidate.tenant_id
                            AND link.evidence_id = candidate.evidence_id
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM memory_evidence AS link
                          JOIN memory_records AS memory
                            ON memory.tenant_id = link.tenant_id
                           AND memory.memory_id = link.memory_id
                          WHERE link.tenant_id = candidate.tenant_id
                            AND link.evidence_id = candidate.evidence_id
                            AND memory.state <> 'compressed'
                      )
                    ORDER BY candidate.evidence_id, candidate.ordinal
                    LIMIT %(limit)s
                ) AS purgeable
                WHERE clip.tenant_id = %(tenant_id)s
                  AND clip.evidence_id = purgeable.evidence_id
                  AND clip.ordinal = purgeable.ordinal
                RETURNING clip.media_object_id
                """,
                {"tenant_id": tenant_id, "limit": limit},
            )
            media_object_ids = [cast(tuple[str], row)[0] async for row in cursor]
            if media_object_ids:
                await connection.execute(
                    """
                    DELETE FROM media_objects AS media
                    WHERE media.tenant_id = %(tenant_id)s
                      AND media.media_object_id = ANY(%(media_object_ids)s::text[])
                      -- Identical clip content deduplicates onto one media object, and the clip
                      -- foreign key is ON DELETE RESTRICT, so only drop it once nothing cites it.
                      AND NOT EXISTS (
                          SELECT 1 FROM evidence_clips AS remaining
                          WHERE remaining.tenant_id = media.tenant_id
                            AND remaining.media_object_id = media.media_object_id
                      )
                    """,
                    {
                        "tenant_id": tenant_id,
                        "media_object_ids": list(dict.fromkeys(media_object_ids)),
                    },
                )
        return len(media_object_ids)
