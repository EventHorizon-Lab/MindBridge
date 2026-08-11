"""PostgreSQL persistence and candidate search for memory records."""

from datetime import datetime
from typing import Any, TypeAlias, cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application import MemoryWriteResult
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DomainInvariantError,
    EvidenceId,
    EvidenceSpan,
    IdempotencyConflictError,
    MediaObjectId,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelReference,
    ObservationId,
    PixelRegion,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool

MemoryRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    datetime,
    datetime,
    datetime,
    str | None,
    str | None,
    list[str],
]
EvidenceRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    int,
    int,
    datetime,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]


async def write_memory(
    pool: DatabasePool,
    memory: MemoryRecord,
    *,
    idempotency_key: str,
    content_digest: str,
) -> MemoryWriteResult:
    """Write a memory atomically or return its idempotent predecessor."""
    async with pool.connection() as connection:
        existing_id = await claim_idempotency_key(
            connection,
            tenant_id=memory.tenant_id,
            operation="remember",
            idempotency_key=idempotency_key,
            content_digest=content_digest,
            resource_id=memory.memory_id,
        )
        if existing_id is not None:
            existing = await _read_memory(
                connection,
                memory.tenant_id,
                MemoryId(existing_id),
            )
            return MemoryWriteResult(memory=existing, created=False)

        created = await _insert_memory(connection, memory, content_digest)
        if not created:
            existing = await _read_memory(connection, memory.tenant_id, memory.memory_id)
            if await _memory_digest(connection, memory) != content_digest:
                raise IdempotencyConflictError("memory identifier already stores different content")
            return MemoryWriteResult(memory=existing, created=False)
        try:
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO memory_evidence (tenant_id, memory_id, evidence_id)
                    VALUES (%s, %s, %s)
                    """,
                    (
                        (memory.tenant_id, memory.memory_id, evidence_id)
                        for evidence_id in memory.evidence_ids
                    ),
                )
        except ForeignKeyViolation as error:
            raise DomainInvariantError("memory references unknown evidence") from error
        return MemoryWriteResult(memory=memory, created=True)


async def search_memories(
    pool: DatabasePool,
    request: RecallRequest,
) -> tuple[MemoryRecord, ...]:
    """Apply exact filters and PostgreSQL full-text candidate retrieval."""
    parameters: dict[str, Any] = {
        "tenant_id": request.tenant_id,
        "query": request.query.text,
        "media_object_ids": list(request.query.media_object_ids),
        "person_ids": list(request.filters.person_ids),
        "device_ids": list(request.filters.device_ids),
        "memory_types": [memory_type.value for memory_type in request.filters.memory_types],
        "occurred_after": request.filters.occurred_after,
        "occurred_before": request.filters.occurred_before,
        "limit": request.limit,
    }
    async with pool.connection() as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_SQL, parameters)
        return tuple([_memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def read_evidence(
    pool: DatabasePool,
    tenant_id: TenantId,
    evidence_ids: tuple[EvidenceId, ...],
) -> tuple[EvidenceSpan, ...]:
    """Read evidence spans in caller order without crossing tenants."""
    if not evidence_ids:
        return ()
    async with pool.connection() as connection:
        cursor = await connection.execute(
            f"{_EVIDENCE_SELECT_SQL} WHERE tenant_id = %s AND evidence_id = ANY(%s)",
            (tenant_id, list(evidence_ids)),
        )
        evidence_by_id = {
            evidence.evidence_id: evidence
            async for row in cursor
            for evidence in (_evidence_from_row(cast(EvidenceRow, row)),)
        }
    return tuple(
        evidence_by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_by_id
    )


async def _insert_memory(
    connection: DatabaseConnection,
    memory: MemoryRecord,
    content_digest: str,
) -> bool:
    cursor = await connection.execute(
        """
        INSERT INTO memory_records (
            tenant_id, memory_id, memory_type, summary, verification_status, state,
            occurred_at, ended_at, model_id, model_revision, content_digest, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING memory_id
        """,
        (
            memory.tenant_id,
            memory.memory_id,
            memory.memory_type.value,
            memory.summary,
            memory.verification_status.value,
            memory.state.value,
            memory.occurred_at,
            memory.ended_at,
            memory.model_reference.model_id if memory.model_reference is not None else None,
            memory.model_reference.revision if memory.model_reference is not None else None,
            content_digest,
            memory.created_at,
        ),
    )
    return await cursor.fetchone() is not None


async def _memory_digest(connection: DatabaseConnection, memory: MemoryRecord) -> str:
    cursor = await connection.execute(
        """
        SELECT content_digest FROM memory_records
        WHERE tenant_id = %s AND memory_id = %s
        """,
        (memory.tenant_id, memory.memory_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("memory disappeared during transaction")
    return cast(tuple[str], row)[0]


async def _read_memory(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_id: MemoryId,
) -> MemoryRecord:
    cursor = await connection.execute(
        f"{_MEMORY_SELECT_SQL} WHERE memory.tenant_id = %s AND memory.memory_id = %s",
        (tenant_id, memory_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("idempotency key references a missing memory")
    return _memory_from_row(cast(MemoryRow, row))


def _memory_from_row(row: MemoryRow) -> MemoryRecord:
    (
        memory_id,
        tenant_id,
        memory_type,
        summary,
        verification_status,
        state,
        occurred_at,
        ended_at,
        created_at,
        model_id,
        model_revision,
        evidence_ids,
    ) = row
    model_reference = (
        ModelReference(model_id=model_id, revision=cast(str, model_revision))
        if model_id is not None
        else None
    )
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=TenantId(tenant_id),
        memory_type=MemoryType(memory_type),
        summary=summary,
        evidence_ids=tuple(EvidenceId(value) for value in evidence_ids),
        occurred_at=occurred_at,
        ended_at=ended_at,
        created_at=created_at,
        verification_status=VerificationStatus(verification_status),
        state=MemoryState(state),
        model_reference=model_reference,
    )


def _evidence_from_row(row: EvidenceRow) -> EvidenceSpan:
    (
        evidence_id,
        tenant_id,
        observation_id,
        media_object_id,
        start_ms,
        end_ms,
        created_at,
        frame_start,
        frame_end,
        x_min,
        y_min,
        x_max,
        y_max,
        audio_track,
    ) = row
    region = (
        PixelRegion(
            x_min=x_min,
            y_min=cast(int, y_min),
            x_max=cast(int, x_max),
            y_max=cast(int, y_max),
        )
        if x_min is not None
        else None
    )
    return EvidenceSpan(
        evidence_id=EvidenceId(evidence_id),
        tenant_id=TenantId(tenant_id),
        observation_id=ObservationId(observation_id),
        media_object_id=MediaObjectId(media_object_id),
        start_ms=start_ms,
        end_ms=end_ms,
        created_at=created_at,
        frame_start=frame_start,
        frame_end=frame_end,
        region=region,
        audio_track=audio_track,
    )


_MEMORY_SELECT_SQL = """
SELECT memory.memory_id,
       memory.tenant_id,
       memory.memory_type,
       memory.summary,
       memory.verification_status,
       memory.state,
       memory.occurred_at,
       memory.ended_at,
       memory.created_at,
       memory.model_id,
       memory.model_revision,
       ARRAY(
           SELECT link.evidence_id
           FROM memory_evidence AS link
           WHERE link.tenant_id = memory.tenant_id AND link.memory_id = memory.memory_id
           ORDER BY link.evidence_id
       ) AS evidence_ids
FROM memory_records AS memory
"""

_EVIDENCE_SELECT_SQL = """
SELECT evidence_id, tenant_id, observation_id, media_object_id,
       start_ms, end_ms, created_at, frame_start, frame_end,
       x_min, y_min, x_max, y_max, audio_track
FROM evidence_spans
"""

_SEARCH_MEMORIES_SQL = f"""
{_MEMORY_SELECT_SQL}
WHERE memory.tenant_id = %(tenant_id)s
  AND (
      %(query)s::text IS NULL
      OR to_tsvector('simple', memory.summary) @@ websearch_to_tsquery('simple', %(query)s)
      OR memory.summary ILIKE '%%' || %(query)s || '%%'
  )
  AND (
      cardinality(%(memory_types)s::text[]) = 0
      OR memory.memory_type = ANY(%(memory_types)s::text[])
  )
  AND (%(occurred_after)s::timestamptz IS NULL OR memory.occurred_at >= %(occurred_after)s)
  AND (%(occurred_before)s::timestamptz IS NULL OR memory.occurred_at <= %(occurred_before)s)
  AND (
      cardinality(%(media_object_ids)s::text[]) = 0
      OR EXISTS (
          SELECT 1
          FROM memory_evidence AS media_link
          JOIN evidence_spans AS media_evidence
            ON media_evidence.tenant_id = media_link.tenant_id
           AND media_evidence.evidence_id = media_link.evidence_id
          WHERE media_link.tenant_id = memory.tenant_id
            AND media_link.memory_id = memory.memory_id
            AND media_evidence.media_object_id = ANY(%(media_object_ids)s::text[])
      )
  )
  AND (
      cardinality(%(device_ids)s::text[]) = 0
      OR EXISTS (
          SELECT 1
          FROM memory_evidence AS device_link
          JOIN evidence_spans AS device_evidence
            ON device_evidence.tenant_id = device_link.tenant_id
           AND device_evidence.evidence_id = device_link.evidence_id
          JOIN observations AS source_observation
            ON source_observation.tenant_id = device_evidence.tenant_id
           AND source_observation.observation_id = device_evidence.observation_id
          WHERE device_link.tenant_id = memory.tenant_id
            AND device_link.memory_id = memory.memory_id
            AND source_observation.device_id = ANY(%(device_ids)s::text[])
      )
  )
  AND (
      cardinality(%(person_ids)s::text[]) = 0
      OR EXISTS (
          SELECT 1
          FROM memory_evidence AS person_link
          JOIN entity_mentions AS mention
            ON mention.tenant_id = person_link.tenant_id
           AND mention.evidence_id = person_link.evidence_id
          WHERE person_link.tenant_id = memory.tenant_id
            AND person_link.memory_id = memory.memory_id
            AND mention.entity_id = ANY(%(person_ids)s::text[])
      )
  )
ORDER BY
    CASE
        WHEN %(query)s::text IS NULL THEN 0
        ELSE ts_rank_cd(
            to_tsvector('simple', memory.summary),
            websearch_to_tsquery('simple', %(query)s)
        )
    END DESC,
    memory.occurred_at DESC,
    memory.memory_id
LIMIT %(limit)s
"""
