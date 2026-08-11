"""Transactional PostgreSQL feedback and correction persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application import FeedbackWriteResult
from mindbridge.core import (
    DomainInvariantError,
    FeedbackId,
    FeedbackType,
    MemoryFeedback,
    MemoryId,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryState,
    TenantId,
    apply_memory_feedback,
)
from mindbridge.infrastructure._postgres_forget import ensure_memory_not_tombstoned
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_memories import (
    find_memory_on_connection,
    write_memory_on_connection,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

FeedbackRow: TypeAlias = tuple[
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | None,
    datetime,
]


async def record_feedback(
    pool: DatabasePool,
    feedback: MemoryFeedback,
    corrected_memory: MemoryRecord | None,
    *,
    idempotency_key: str,
    content_digest: str,
) -> FeedbackWriteResult:
    """Atomically record feedback, lifecycle counters, and any correction version."""
    async with tenant_connection(pool, feedback.tenant_id) as connection:
        existing_id = await claim_idempotency_key(
            connection,
            tenant_id=feedback.tenant_id,
            operation="feedback",
            idempotency_key=idempotency_key,
            content_digest=content_digest,
            resource_id=feedback.feedback_id,
        )
        if existing_id is not None:
            return await _read_feedback_result(
                connection,
                feedback.tenant_id,
                FeedbackId(existing_id),
                created=False,
            )

        evolved_memory = await _evolve_target_memory(connection, feedback)
        if feedback.feedback_type is FeedbackType.CORRECTION:
            _require_corrected_memory(feedback, corrected_memory)
            assert corrected_memory is not None
            await write_memory_on_connection(connection, corrected_memory, content_digest)
            assert evolved_memory is not None
            evolved_memory = replace(evolved_memory, superseded_at=feedback.created_at)
        elif corrected_memory is not None:
            raise DomainInvariantError("only correction feedback may create a memory version")

        if evolved_memory is not None:
            await _update_memory_lifecycle(connection, evolved_memory)
        await connection.execute(
            """
            INSERT INTO memory_feedback (
                tenant_id, feedback_id, memory_id, feedback_type, recall_trace_id,
                corrected_memory_id, resulting_state, resulting_strength, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                feedback.tenant_id,
                feedback.feedback_id,
                feedback.memory_id,
                feedback.feedback_type.value,
                feedback.recall_trace_id,
                corrected_memory.memory_id if corrected_memory is not None else None,
                evolved_memory.state.value if evolved_memory is not None else None,
                evolved_memory.strength if evolved_memory is not None else None,
                feedback.created_at,
            ),
        )
        return FeedbackWriteResult(
            feedback_id=feedback.feedback_id,
            feedback_type=feedback.feedback_type,
            memory_id=feedback.memory_id,
            created_at=feedback.created_at,
            resulting_state=evolved_memory.state if evolved_memory is not None else None,
            resulting_strength=evolved_memory.strength if evolved_memory is not None else None,
            corrected_memory=corrected_memory,
            created=True,
        )


async def _evolve_target_memory(
    connection: DatabaseConnection,
    feedback: MemoryFeedback,
) -> MemoryRecord | None:
    if feedback.memory_id is None:
        return None
    await ensure_memory_not_tombstoned(connection, feedback.tenant_id, feedback.memory_id)
    memory = await find_memory_on_connection(
        connection,
        feedback.tenant_id,
        feedback.memory_id,
        for_update=True,
    )
    if memory is None:
        raise MemoryNotFoundError("memory does not exist")
    return apply_memory_feedback(memory, feedback.feedback_type, feedback.created_at)


def _require_corrected_memory(
    feedback: MemoryFeedback,
    corrected_memory: MemoryRecord | None,
) -> None:
    if (
        corrected_memory is None
        or feedback.memory_id is None
        or corrected_memory.tenant_id != feedback.tenant_id
        or corrected_memory.supersedes_memory_id != feedback.memory_id
        or corrected_memory.summary != feedback.correction_summary
        or corrected_memory.created_at != feedback.created_at
    ):
        raise DomainInvariantError("corrected memory does not match its feedback event")


async def _update_memory_lifecycle(
    connection: DatabaseConnection,
    memory: MemoryRecord,
) -> None:
    await connection.execute(
        """
        UPDATE memory_records
        SET state = %s,
            strength = %s,
            useful_access_count = %s,
            positive_feedback_count = %s,
            negative_feedback_count = %s,
            last_accessed_at = %s,
            superseded_at = %s
        WHERE tenant_id = %s AND memory_id = %s
        """,
        (
            memory.state.value,
            memory.strength,
            memory.useful_access_count,
            memory.positive_feedback_count,
            memory.negative_feedback_count,
            memory.last_accessed_at,
            memory.superseded_at,
            memory.tenant_id,
            memory.memory_id,
        ),
    )


async def _read_feedback_result(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    feedback_id: FeedbackId,
    *,
    created: bool,
) -> FeedbackWriteResult:
    cursor = await connection.execute(
        """
        SELECT feedback_id, feedback_type, memory_id, corrected_memory_id,
               resulting_state, resulting_strength, created_at
        FROM memory_feedback
        WHERE tenant_id = %s AND feedback_id = %s
        """,
        (tenant_id, feedback_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise DomainInvariantError("feedback idempotency key references a missing event")
    (
        stored_feedback_id,
        feedback_type,
        memory_id,
        corrected_memory_id,
        resulting_state,
        resulting_strength,
        created_at,
    ) = cast(FeedbackRow, row)
    corrected_memory = (
        await find_memory_on_connection(
            connection,
            tenant_id,
            MemoryId(corrected_memory_id),
        )
        if corrected_memory_id is not None
        else None
    )
    return FeedbackWriteResult(
        feedback_id=FeedbackId(stored_feedback_id),
        feedback_type=FeedbackType(feedback_type),
        memory_id=MemoryId(memory_id) if memory_id is not None else None,
        created_at=created_at,
        resulting_state=MemoryState(resulting_state) if resulting_state is not None else None,
        resulting_strength=resulting_strength,
        corrected_memory=corrected_memory,
        created=created,
    )
