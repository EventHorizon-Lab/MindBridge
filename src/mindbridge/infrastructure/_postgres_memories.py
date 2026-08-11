"""PostgreSQL persistence and candidate search for memory records."""

from datetime import datetime
from typing import Any, TypeAlias, cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application import MemoryWriteResult
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DomainInvariantError,
    EvidenceId,
    ForgetTargetType,
    IdempotencyConflictError,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelReference,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure._postgres_forget import (
    ensure_memory_not_tombstoned,
    ensure_target_not_tombstoned,
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
    float,
    float,
    int,
    int,
    int,
    datetime | None,
    str | None,
    datetime | None,
    list[str],
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
            existing = await _require_memory_on_connection(
                connection,
                memory.tenant_id,
                MemoryId(existing_id),
            )
            return MemoryWriteResult(memory=existing, created=False)

        created = await write_memory_on_connection(connection, memory, content_digest)
        if not created:
            existing = await _require_memory_on_connection(
                connection, memory.tenant_id, memory.memory_id
            )
            return MemoryWriteResult(memory=existing, created=False)
        return MemoryWriteResult(memory=memory, created=True)


async def read_memory(
    pool: DatabasePool,
    tenant_id: TenantId,
    memory_id: MemoryId,
) -> MemoryRecord:
    """Read one memory without revealing whether its ID exists in another tenant."""
    async with pool.connection() as connection:
        await ensure_memory_not_tombstoned(connection, tenant_id, memory_id)
        memory = await find_memory_on_connection(connection, tenant_id, memory_id)
    if memory is None:
        raise MemoryNotFoundError("memory does not exist")
    return memory


async def write_memory_on_connection(
    connection: DatabaseConnection,
    memory: MemoryRecord,
    content_digest: str,
) -> bool:
    """Write one memory and evidence links inside a caller-owned transaction."""
    await ensure_target_not_tombstoned(
        connection,
        memory.tenant_id,
        ForgetTargetType.MEMORY_RECORD,
        memory.memory_id,
    )
    created = await _insert_memory(connection, memory, content_digest)
    if not created and await _memory_digest(connection, memory) != content_digest:
        raise IdempotencyConflictError("memory identifier already stores different content")
    try:
        async with connection.cursor() as cursor:
            await cursor.executemany(
                """
                INSERT INTO memory_evidence (tenant_id, memory_id, evidence_id)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    (memory.tenant_id, memory.memory_id, evidence_id)
                    for evidence_id in memory.evidence_ids
                ),
            )
    except ForeignKeyViolation as error:
        raise DomainInvariantError("memory references unknown evidence") from error
    return created


async def search_memories(
    pool: DatabasePool,
    request: RecallRequest,
) -> tuple[MemoryRecord, ...]:
    """Apply exact filters and PostgreSQL full-text candidate retrieval."""
    async with pool.connection() as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_SQL, _recall_parameters(request))
        return tuple([_memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def search_memories_by_evidence(
    pool: DatabasePool,
    request: RecallRequest,
    ranked_evidence_ids: tuple[EvidenceId, ...],
    *,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Map semantic evidence rank to memories while retaining exact recall filters."""
    if not ranked_evidence_ids:
        return ()
    if limit <= 0:
        raise DomainInvariantError("semantic memory candidate limit must be positive")
    parameters = _recall_parameters(request)
    parameters.update(evidence_ids=list(ranked_evidence_ids), limit=limit)
    async with pool.connection() as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_BY_EVIDENCE_SQL, parameters)
        return tuple([_memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def search_memories_by_ids(
    pool: DatabasePool,
    request: RecallRequest,
    ranked_memory_ids: tuple[MemoryId, ...],
    *,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Resolve semantic memory hits while retaining exact recall filters."""
    if not ranked_memory_ids:
        return ()
    if limit <= 0:
        raise DomainInvariantError("semantic memory candidate limit must be positive")
    parameters = _recall_parameters(request)
    parameters.update(memory_ids=list(ranked_memory_ids), limit=limit)
    async with pool.connection() as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_BY_IDS_SQL, parameters)
        return tuple([_memory_from_row(cast(MemoryRow, row)) async for row in cursor])


def _recall_parameters(request: RecallRequest) -> dict[str, Any]:
    return {
        "tenant_id": request.tenant_id,
        "query": request.query.text,
        "person_ids": list(request.filters.person_ids),
        "device_ids": list(request.filters.device_ids),
        "memory_types": [memory_type.value for memory_type in request.filters.memory_types],
        "occurred_after": request.filters.occurred_after,
        "occurred_before": request.filters.occurred_before,
        "limit": request.limit,
    }


async def _insert_memory(
    connection: DatabaseConnection,
    memory: MemoryRecord,
    content_digest: str,
) -> bool:
    cursor = await connection.execute(
        """
        INSERT INTO memory_records (
            tenant_id, memory_id, memory_type, summary, verification_status, state,
            occurred_at, ended_at, model_id, model_revision, content_digest, created_at,
            salience, strength, useful_access_count, positive_feedback_count,
            negative_feedback_count, last_accessed_at, supersedes_memory_id, superseded_at
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s
        )
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
            memory.salience,
            memory.strength,
            memory.useful_access_count,
            memory.positive_feedback_count,
            memory.negative_feedback_count,
            memory.last_accessed_at,
            memory.supersedes_memory_id,
            memory.superseded_at,
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


async def find_memory_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_id: MemoryId,
    *,
    for_update: bool = False,
) -> MemoryRecord | None:
    """Find one memory inside a caller transaction, optionally locking its lifecycle row."""
    lock = " FOR UPDATE OF memory" if for_update else ""
    cursor = await connection.execute(
        f"{_MEMORY_SELECT_SQL} WHERE memory.tenant_id = %s AND memory.memory_id = %s{lock}",
        (tenant_id, memory_id),
    )
    row = await cursor.fetchone()
    return _memory_from_row(cast(MemoryRow, row)) if row is not None else None


async def _require_memory_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_id: MemoryId,
) -> MemoryRecord:
    memory = await find_memory_on_connection(connection, tenant_id, memory_id)
    if memory is None:
        raise MemoryIntegrityError("idempotency key references a missing memory")
    return memory


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
        salience,
        strength,
        useful_access_count,
        positive_feedback_count,
        negative_feedback_count,
        last_accessed_at,
        supersedes_memory_id,
        superseded_at,
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
        salience=salience,
        strength=strength,
        useful_access_count=useful_access_count,
        positive_feedback_count=positive_feedback_count,
        negative_feedback_count=negative_feedback_count,
        last_accessed_at=last_accessed_at,
        supersedes_memory_id=(
            MemoryId(supersedes_memory_id) if supersedes_memory_id is not None else None
        ),
        superseded_at=superseded_at,
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
       memory.salience,
       memory.strength,
       memory.useful_access_count,
       memory.positive_feedback_count,
       memory.negative_feedback_count,
       memory.last_accessed_at,
       memory.supersedes_memory_id,
       memory.superseded_at,
       ARRAY(
           SELECT link.evidence_id
           FROM memory_evidence AS link
           WHERE link.tenant_id = memory.tenant_id AND link.memory_id = memory.memory_id
           ORDER BY link.evidence_id
       ) AS evidence_ids
FROM memory_records AS memory
"""

_STRUCTURED_RECALL_FILTER_SQL = """
  AND memory.superseded_at IS NULL
  AND NOT EXISTS (
      SELECT 1
      FROM deletion_tombstones AS tombstone
      WHERE tombstone.tenant_id = memory.tenant_id
        AND (
            (tombstone.target_type = 'memory_record' AND tombstone.target_id = memory.memory_id)
            OR (
                tombstone.target_type = 'observation'
                AND EXISTS (
                    SELECT 1
                    FROM memory_evidence AS deleted_link
                    JOIN evidence_spans AS deleted_evidence
                      ON deleted_evidence.tenant_id = deleted_link.tenant_id
                     AND deleted_evidence.evidence_id = deleted_link.evidence_id
                    WHERE deleted_link.tenant_id = memory.tenant_id
                      AND deleted_link.memory_id = memory.memory_id
                      AND deleted_evidence.observation_id = tombstone.target_id
                )
            )
        )
  )
  AND (
      cardinality(%(memory_types)s::text[]) = 0
      OR memory.memory_type = ANY(%(memory_types)s::text[])
  )
  AND (%(occurred_after)s::timestamptz IS NULL OR memory.occurred_at >= %(occurred_after)s)
  AND (%(occurred_before)s::timestamptz IS NULL OR memory.occurred_at <= %(occurred_before)s)
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
"""

_SEARCH_MEMORIES_SQL = f"""
{_MEMORY_SELECT_SQL}
WHERE memory.tenant_id = %(tenant_id)s
  AND (
      %(query)s::text IS NULL
      OR to_tsvector('simple', memory.summary) @@ websearch_to_tsquery('simple', %(query)s)
      OR memory.summary ILIKE '%%' || %(query)s || '%%'
  )
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY
    CASE
        WHEN %(query)s::text IS NULL THEN 0
        ELSE ts_rank_cd(
            to_tsvector('simple', memory.summary),
            websearch_to_tsquery('simple', %(query)s)
        )
    END DESC,
    memory.strength DESC,
    memory.occurred_at DESC,
    memory.memory_id
LIMIT %(limit)s
"""

_SEARCH_MEMORIES_BY_EVIDENCE_SQL = f"""
WITH ranked_evidence AS (
    SELECT evidence_id, rank
    FROM unnest(%(evidence_ids)s::text[]) WITH ORDINALITY AS hit(evidence_id, rank)
),
ranked_memories AS (
    SELECT link.tenant_id, link.memory_id, min(hit.rank) AS dense_rank
    FROM memory_evidence AS link
    JOIN ranked_evidence AS hit ON hit.evidence_id = link.evidence_id
    WHERE link.tenant_id = %(tenant_id)s
    GROUP BY link.tenant_id, link.memory_id
)
{_MEMORY_SELECT_SQL}
JOIN ranked_memories AS dense
  ON dense.tenant_id = memory.tenant_id AND dense.memory_id = memory.memory_id
WHERE memory.tenant_id = %(tenant_id)s
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY dense.dense_rank, memory.strength DESC, memory.occurred_at DESC, memory.memory_id
LIMIT %(limit)s
"""

_SEARCH_MEMORIES_BY_IDS_SQL = f"""
WITH ranked_memories AS (
    SELECT memory_id, rank
    FROM unnest(%(memory_ids)s::text[]) WITH ORDINALITY AS hit(memory_id, rank)
)
{_MEMORY_SELECT_SQL}
JOIN ranked_memories AS dense ON dense.memory_id = memory.memory_id
WHERE memory.tenant_id = %(tenant_id)s
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY dense.rank, memory.strength DESC, memory.occurred_at DESC, memory.memory_id
LIMIT %(limit)s
"""
