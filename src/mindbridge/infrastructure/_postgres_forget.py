"""Recoverable PostgreSQL coordination for explicit forgetting."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application.ports import ForgetPlan
from mindbridge.core import (
    DeletionPropagationState,
    DeletionTombstone,
    DomainInvariantError,
    ForgetTargetNotFoundError,
    ForgetTargetType,
    MemoryDeletedError,
    MemoryIntegrityError,
    TenantId,
    TombstoneId,
)
from mindbridge.infrastructure._postgres_forget_cleanup import (
    delete_memory_scope,
    delete_observation_scope,
    lock_exclusive_observation_media,
)
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

TombstoneRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    datetime,
    datetime | None,
    str | None,
]


async def prepare_forget(
    pool: DatabasePool,
    tombstone: DeletionTombstone,
    *,
    idempotency_key: str,
    content_digest: str,
) -> ForgetPlan:
    """Persist the deletion barrier and plan remaining object-store erasure."""
    async with tenant_connection(pool, tombstone.tenant_id) as connection:
        existing_id = await claim_idempotency_key(
            connection,
            tenant_id=tombstone.tenant_id,
            operation="forget",
            idempotency_key=idempotency_key,
            content_digest=content_digest,
            resource_id=tombstone.tombstone_id,
        )
        if existing_id is not None and existing_id != tombstone.tombstone_id:
            raise MemoryIntegrityError("forget idempotency key references another tombstone")

        existing = await _find_target_tombstone(
            connection,
            tombstone.tenant_id,
            tombstone.target_type,
            tombstone.target_id,
            for_update=True,
        )
        if existing is None:
            await _lock_forget_target(connection, tombstone)
            existing = await _find_target_tombstone(
                connection,
                tombstone.tenant_id,
                tombstone.target_type,
                tombstone.target_id,
                for_update=True,
            )
            if existing is None:
                existing = await _insert_tombstone(connection, tombstone)
        elif existing.tombstone_id != tombstone.tombstone_id:
            raise MemoryIntegrityError("forget target has conflicting tombstone identity")

        propagating = await _begin_propagation(connection, existing)
        media_objects = (
            await lock_exclusive_observation_media(connection, propagating)
            if propagating.target_type is ForgetTargetType.OBSERVATION
            and propagating.propagation_state is not DeletionPropagationState.COMPLETE
            else ()
        )
        return ForgetPlan(tombstone=propagating, media_objects=media_objects)


async def read_deletion_tombstone(
    pool: DatabasePool,
    tenant_id: TenantId,
    tombstone_id: TombstoneId,
) -> DeletionTombstone:
    """Read content-free deletion progress without crossing tenants."""
    async with tenant_connection(pool, tenant_id) as connection:
        tombstone = await _find_tombstone(connection, tenant_id, tombstone_id)
    if tombstone is None:
        raise ForgetTargetNotFoundError("deletion tombstone does not exist")
    return tombstone


async def list_deletion_tombstones(
    pool: DatabasePool,
    tenant_id: TenantId,
    *,
    after_tombstone_id: str | None,
    limit: int,
) -> tuple[DeletionTombstone, ...]:
    """List one stable tenant page after an optional tombstone cursor."""
    async with tenant_connection(pool, tenant_id) as connection:
        after_sequence = 0
        if after_tombstone_id is not None:
            boundary = await connection.execute(
                """
                SELECT cursor_sequence FROM deletion_tombstones
                WHERE tenant_id = %s AND tombstone_id = %s
                """,
                (tenant_id, after_tombstone_id),
            )
            row = await boundary.fetchone()
            if row is None:
                raise DomainInvariantError("deletion cursor does not exist")
            after_sequence = cast(tuple[int], row)[0]
        cursor = await connection.execute(
            """
            SELECT tombstone_id, tenant_id, target_type, target_id,
                   propagation_state, requested_at, completed_at, error_code
            FROM deletion_tombstones AS tombstone
            WHERE tombstone.tenant_id = %s
              AND tombstone.cursor_sequence > %s
            ORDER BY tombstone.cursor_sequence
            LIMIT %s
            """,
            (tenant_id, after_sequence, limit),
        )
        return tuple([_tombstone_from_row(cast(TombstoneRow, row)) async for row in cursor])


async def complete_forget(
    pool: DatabasePool,
    tombstone: DeletionTombstone,
    *,
    completed_at: datetime,
) -> DeletionTombstone:
    """Erase database derivatives and atomically mark propagation complete."""
    async with tenant_connection(pool, tombstone.tenant_id) as connection:
        stored = await _require_matching_tombstone(connection, tombstone)
        if stored.propagation_state is DeletionPropagationState.COMPLETE:
            return stored
        if stored.propagation_state is not DeletionPropagationState.PROPAGATING:
            raise MemoryIntegrityError("forget completion requires propagating state")
        if stored.target_type is ForgetTargetType.MEMORY_RECORD:
            await delete_memory_scope(connection, stored.tenant_id, (stored.target_id,))
        else:
            await delete_observation_scope(
                connection,
                stored,
                completed_at=completed_at,
            )
        return await _set_tombstone_complete(connection, stored, completed_at)


async def mark_forget_failed(
    pool: DatabasePool,
    tombstone: DeletionTombstone,
    *,
    error_code: str,
) -> DeletionTombstone:
    """Retain a sanitized recoverable failure while the barrier stays active."""
    async with tenant_connection(pool, tombstone.tenant_id) as connection:
        stored = await _require_matching_tombstone(connection, tombstone)
        if stored.propagation_state is DeletionPropagationState.COMPLETE:
            return stored
        cursor = await connection.execute(
            """
            UPDATE deletion_tombstones
            SET propagation_state = 'failed', completed_at = NULL, error_code = %s
            WHERE tenant_id = %s AND tombstone_id = %s
            RETURNING tombstone_id, tenant_id, target_type, target_id,
                      propagation_state, requested_at, completed_at, error_code
            """,
            (error_code, stored.tenant_id, stored.tombstone_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise MemoryIntegrityError("deletion tombstone disappeared during failure update")
        return _tombstone_from_row(cast(TombstoneRow, row))


async def ensure_target_not_tombstoned(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    target_type: ForgetTargetType,
    target_id: str,
) -> None:
    """Reject writes and worker commits that would resurrect deleted content."""
    tombstone = await _find_target_tombstone(
        connection,
        tenant_id,
        target_type,
        target_id,
    )
    if tombstone is not None:
        raise MemoryDeletedError("target has been explicitly forgotten")


async def ensure_memory_not_tombstoned(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_id: str,
) -> None:
    """Reject direct or source-observation deletion of a memory."""
    cursor = await connection.execute(
        """
        SELECT 1
        FROM deletion_tombstones AS tombstone
        WHERE tombstone.tenant_id = %s
          AND (
              (tombstone.target_type = 'memory_record' AND tombstone.target_id = %s)
              OR (
                  tombstone.target_type = 'observation'
                  AND EXISTS (
                      SELECT 1
                      FROM memory_evidence AS link
                      JOIN evidence_spans AS evidence
                        ON evidence.tenant_id = link.tenant_id
                       AND evidence.evidence_id = link.evidence_id
                      WHERE link.tenant_id = tombstone.tenant_id
                        AND link.memory_id = %s
                        AND evidence.observation_id = tombstone.target_id
                  )
              )
          )
        LIMIT 1
        """,
        (tenant_id, memory_id, memory_id),
    )
    if await cursor.fetchone() is not None:
        raise MemoryDeletedError("memory has been explicitly forgotten")


async def _lock_forget_target(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
) -> None:
    if tombstone.target_type is ForgetTargetType.MEMORY_RECORD:
        table_name, id_column = "memory_records", "memory_id"
    else:
        table_name, id_column = "observations", "observation_id"
    cursor = await connection.execute(
        f"""
        SELECT 1 FROM {table_name}
        WHERE tenant_id = %s AND {id_column} = %s
        FOR UPDATE
        """,
        (tombstone.tenant_id, tombstone.target_id),
    )
    if await cursor.fetchone() is None:
        raise ForgetTargetNotFoundError("forget target does not exist")


async def _insert_tombstone(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
) -> DeletionTombstone:
    await connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended('mindbridge:deletion:' || %s, 0))",
        (tombstone.tenant_id,),
    )
    cursor = await connection.execute(
        """
        INSERT INTO deletion_tombstones (
            tenant_id, tombstone_id, target_type, target_id,
            propagation_state, requested_at, completed_at, error_code
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING tombstone_id, tenant_id, target_type, target_id,
                  propagation_state, requested_at, completed_at, error_code
        """,
        (
            tombstone.tenant_id,
            tombstone.tombstone_id,
            tombstone.target_type.value,
            tombstone.target_id,
            tombstone.propagation_state.value,
            tombstone.requested_at,
            tombstone.completed_at,
            tombstone.error_code,
        ),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("deletion tombstone was not inserted")
    return _tombstone_from_row(cast(TombstoneRow, row))


async def _begin_propagation(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
) -> DeletionTombstone:
    if tombstone.propagation_state is DeletionPropagationState.COMPLETE:
        return tombstone
    cursor = await connection.execute(
        """
        UPDATE deletion_tombstones
        SET propagation_state = 'propagating', completed_at = NULL, error_code = NULL
        WHERE tenant_id = %s AND tombstone_id = %s
        RETURNING tombstone_id, tenant_id, target_type, target_id,
                  propagation_state, requested_at, completed_at, error_code
        """,
        (tombstone.tenant_id, tombstone.tombstone_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("deletion tombstone disappeared during propagation")
    return _tombstone_from_row(cast(TombstoneRow, row))


async def _set_tombstone_complete(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
    completed_at: datetime,
) -> DeletionTombstone:
    cursor = await connection.execute(
        """
        UPDATE deletion_tombstones
        SET propagation_state = 'complete', completed_at = %s, error_code = NULL
        WHERE tenant_id = %s AND tombstone_id = %s
        RETURNING tombstone_id, tenant_id, target_type, target_id,
                  propagation_state, requested_at, completed_at, error_code
        """,
        (completed_at, tombstone.tenant_id, tombstone.tombstone_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("deletion tombstone disappeared during completion")
    return _tombstone_from_row(cast(TombstoneRow, row))


async def _require_matching_tombstone(
    connection: DatabaseConnection,
    expected: DeletionTombstone,
) -> DeletionTombstone:
    stored = await _find_tombstone(
        connection,
        expected.tenant_id,
        expected.tombstone_id,
        for_update=True,
    )
    if stored is None:
        raise MemoryIntegrityError("deletion tombstone does not exist")
    if stored.target_type is not expected.target_type or stored.target_id != expected.target_id:
        raise MemoryIntegrityError("deletion tombstone target changed")
    return stored


async def _find_target_tombstone(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    target_type: ForgetTargetType,
    target_id: str,
    *,
    for_update: bool = False,
) -> DeletionTombstone | None:
    lock = " FOR UPDATE" if for_update else ""
    cursor = await connection.execute(
        f"""
        SELECT tombstone_id, tenant_id, target_type, target_id,
               propagation_state, requested_at, completed_at, error_code
        FROM deletion_tombstones
        WHERE tenant_id = %s AND target_type = %s AND target_id = %s{lock}
        """,
        (tenant_id, target_type.value, target_id),
    )
    row = await cursor.fetchone()
    return _tombstone_from_row(cast(TombstoneRow, row)) if row is not None else None


async def _find_tombstone(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    tombstone_id: TombstoneId,
    *,
    for_update: bool = False,
) -> DeletionTombstone | None:
    lock = " FOR UPDATE" if for_update else ""
    cursor = await connection.execute(
        f"""
        SELECT tombstone_id, tenant_id, target_type, target_id,
               propagation_state, requested_at, completed_at, error_code
        FROM deletion_tombstones
        WHERE tenant_id = %s AND tombstone_id = %s{lock}
        """,
        (tenant_id, tombstone_id),
    )
    row = await cursor.fetchone()
    return _tombstone_from_row(cast(TombstoneRow, row)) if row is not None else None


def _tombstone_from_row(row: TombstoneRow) -> DeletionTombstone:
    (
        tombstone_id,
        tenant_id,
        target_type,
        target_id,
        propagation_state,
        requested_at,
        completed_at,
        error_code,
    ) = row
    return DeletionTombstone(
        tombstone_id=TombstoneId(tombstone_id),
        tenant_id=TenantId(tenant_id),
        target_type=ForgetTargetType(target_type),
        target_id=target_id,
        propagation_state=DeletionPropagationState(propagation_state),
        requested_at=requested_at,
        completed_at=completed_at,
        error_code=error_code,
    )
