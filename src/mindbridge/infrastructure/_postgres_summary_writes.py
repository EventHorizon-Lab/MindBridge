"""Atomic PostgreSQL commit for hierarchical Memory summaries."""

from __future__ import annotations

from typing import cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application.summary_consolidation import SummaryWrite
from mindbridge.core import DomainInvariantError, MemoryId, MemoryIntegrityError, TenantId
from mindbridge.infrastructure._postgres_derived_records import derived_memory_content_digest
from mindbridge.infrastructure._postgres_embeddings import write_embedding_on_connection
from mindbridge.infrastructure._postgres_graph import write_relations
from mindbridge.infrastructure._postgres_memories import write_memory_on_connection
from mindbridge.infrastructure._postgres_memory_rows import MEMORY_NOT_TOMBSTONED_SQL
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)


async def commit_summary_consolidation(
    pool: DatabasePool,
    tenant_id: TenantId,
    writes: tuple[SummaryWrite, ...],
) -> int:
    """Lock source Memories and commit each complete hierarchy node at most once."""
    source_ids = [memory_id for write in writes for memory_id in write.source_memory_ids]
    if any(write.memory.tenant_id != tenant_id for write in writes):
        raise DomainInvariantError("Summary consolidation must remain in the requested tenant")
    if len(set(source_ids)) != len(source_ids):
        raise DomainInvariantError("Summary writes cannot share source Memories")
    if {write.memory.memory_id for write in writes} & set(source_ids):
        raise DomainInvariantError("Summary writes cannot create an in-transaction cycle")
    if not writes:
        return 0

    try:
        async with tenant_connection(pool, tenant_id) as connection:
            current_source_ids = await _lock_current_sources(
                connection,
                tenant_id,
                sorted(source_ids),
            )
            parent_ids = await _read_parent_ids(connection, tenant_id, source_ids)
            committed_count = 0
            for write in writes:
                if not set(write.source_memory_ids) <= current_source_ids:
                    continue
                expected_parent_id = write.memory.memory_id
                existing_parent_ids = tuple(
                    parent_ids.get(source_id) for source_id in write.source_memory_ids
                )
                if any(
                    parent_id not in {None, expected_parent_id} for parent_id in existing_parent_ids
                ):
                    continue
                created = all(parent_id is None for parent_id in existing_parent_ids)
                await _write_summary(connection, write)
                parent_ids.update(
                    (source_id, expected_parent_id) for source_id in write.source_memory_ids
                )
                committed_count += int(created)
            return committed_count
    except ForeignKeyViolation as error:
        raise DomainInvariantError("Summary references missing source evidence") from error


async def _lock_current_sources(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    source_ids: list[MemoryId],
) -> set[MemoryId]:
    cursor = await connection.execute(
        f"""
        SELECT memory.memory_id
        FROM memory_records AS memory
        WHERE memory.tenant_id = %s
          AND memory.memory_id = ANY(%s)
          AND memory.superseded_at IS NULL
          AND memory.memory_type IN ('episodic', 'semantic')
          AND memory.verification_status IN ('verified', 'attested')
          AND {MEMORY_NOT_TOMBSTONED_SQL}
        ORDER BY memory.memory_id
        FOR UPDATE OF memory
        """,
        (tenant_id, list(source_ids)),
    )
    return {MemoryId(cast(tuple[str], row)[0]) async for row in cursor}


async def _read_parent_ids(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    source_ids: list[MemoryId],
) -> dict[MemoryId, MemoryId]:
    cursor = await connection.execute(
        """
        SELECT target_id, source_id
        FROM relations
        WHERE tenant_id = %s
          AND source_type = 'memory_record'
          AND relation_type = 'contains'
          AND target_type = 'memory_record'
          AND target_id = ANY(%s)
        """,
        (tenant_id, list(source_ids)),
    )
    parents: dict[MemoryId, MemoryId] = {}
    async for row in cursor:
        child_id, parent_id = cast(tuple[str, str], row)
        child = MemoryId(child_id)
        parent = MemoryId(parent_id)
        if child in parents and parents[child] != parent:
            raise MemoryIntegrityError("one Memory belongs to multiple Summary parents")
        parents[child] = parent
    return parents


async def _write_summary(
    connection: DatabaseConnection,
    write: SummaryWrite,
) -> None:
    await write_memory_on_connection(
        connection,
        write.memory,
        derived_memory_content_digest(write.memory),
    )
    await write_relations(connection, write.relations)
    await write_embedding_on_connection(connection, write.embedding)
