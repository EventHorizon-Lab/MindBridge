"""PostgreSQL persistence and candidate search for memory records."""

from typing import Any, cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application import EmbeddingMatch, MemoryWriteResult
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EvidenceId,
    ForgetTargetType,
    IdempotencyConflictError,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
    TenantId,
)
from mindbridge.infrastructure._postgres_forget import (
    ensure_memory_not_tombstoned,
    ensure_target_not_tombstoned,
)
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_memory_rows import (
    MEMORY_NOT_TOMBSTONED_SQL,
    MEMORY_SELECT_SQL,
    MemoryRow,
    memory_from_row,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)


async def write_memory(
    pool: DatabasePool,
    memory: MemoryRecord,
    *,
    idempotency_key: str,
    content_digest: str,
) -> MemoryWriteResult:
    """Write a memory atomically or return its idempotent predecessor."""
    async with tenant_connection(pool, memory.tenant_id) as connection:
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
    async with tenant_connection(pool, tenant_id) as connection:
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
    *,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Apply exact filters and PostgreSQL full-text candidate retrieval."""
    if not 1 <= limit <= 1_000:
        raise DomainInvariantError("memory candidate limit must be between 1 and 1000")
    async with tenant_connection(pool, request.tenant_id) as connection:
        parameters = _recall_parameters(request)
        parameters["limit"] = limit
        cursor = await connection.execute(_SEARCH_MEMORIES_SQL, parameters)
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def list_memories_for_enumeration(
    pool: DatabasePool,
    request: RecallRequest,
    *,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Scan the complete structured-filter scope in chronological order."""
    if not 1 <= limit <= 1_001:
        raise DomainInvariantError("enumeration candidate limit must be between 1 and 1001")
    async with tenant_connection(pool, request.tenant_id) as connection:
        parameters = _recall_parameters(request)
        parameters["limit"] = limit
        cursor = await connection.execute(_LIST_MEMORIES_FOR_ENUMERATION_SQL, parameters)
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


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
    async with tenant_connection(pool, request.tenant_id) as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_BY_EVIDENCE_SQL, parameters)
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


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
    async with tenant_connection(pool, request.tenant_id) as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_BY_IDS_SQL, parameters)
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


async def search_memories_by_graph_objects(
    pool: DatabasePool,
    request: RecallRequest,
    ranked_objects: tuple[EmbeddingMatch, ...],
    *,
    limit: int,
) -> tuple[MemoryRecord, ...]:
    """Follow ranked Event/Claim representation edges to filtered memories."""
    if not ranked_objects:
        return ()
    if limit <= 0:
        raise DomainInvariantError("semantic graph candidate limit must be positive")
    if any(
        match.object_type not in {EmbeddedObjectType.EVENT, EmbeddedObjectType.CLAIM}
        for match in ranked_objects
    ):
        raise DomainInvariantError("semantic graph candidates must be events or claims")
    parameters = _recall_parameters(request)
    parameters.update(
        object_types=[match.object_type.value for match in ranked_objects],
        object_ids=[match.object_id for match in ranked_objects],
        graph_neighbor_limit=min(limit, 16),
        limit=limit,
    )
    async with tenant_connection(pool, request.tenant_id) as connection:
        cursor = await connection.execute(_SEARCH_MEMORIES_BY_GRAPH_OBJECTS_SQL, parameters)
        return tuple([memory_from_row(cast(MemoryRow, row)) async for row in cursor])


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
        f"{MEMORY_SELECT_SQL} WHERE memory.tenant_id = %s AND memory.memory_id = %s{lock}",
        (tenant_id, memory_id),
    )
    row = await cursor.fetchone()
    return memory_from_row(cast(MemoryRow, row)) if row is not None else None


async def _require_memory_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_id: MemoryId,
) -> MemoryRecord:
    memory = await find_memory_on_connection(connection, tenant_id, memory_id)
    if memory is None:
        raise MemoryIntegrityError("idempotency key references a missing memory")
    return memory


_STRUCTURED_RECALL_FILTER_SQL = f"""
  AND memory.superseded_at IS NULL
  AND {MEMORY_NOT_TOMBSTONED_SQL}
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
{MEMORY_SELECT_SQL}
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

_LIST_MEMORIES_FOR_ENUMERATION_SQL = f"""
{MEMORY_SELECT_SQL}
WHERE memory.tenant_id = %(tenant_id)s
  AND memory.memory_type = 'episodic'
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY memory.occurred_at, memory.memory_id
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
{MEMORY_SELECT_SQL}
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
{MEMORY_SELECT_SQL}
JOIN ranked_memories AS dense ON dense.memory_id = memory.memory_id
WHERE memory.tenant_id = %(tenant_id)s
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY dense.rank, memory.strength DESC, memory.occurred_at DESC, memory.memory_id
LIMIT %(limit)s
"""

_SEARCH_MEMORIES_BY_GRAPH_OBJECTS_SQL = f"""
WITH ranked_objects AS (
    SELECT object_type, object_id, rank
    FROM unnest(
        %(object_types)s::text[], %(object_ids)s::text[]
    ) WITH ORDINALITY AS hit(object_type, object_id, rank)
),
related_objects AS (
    SELECT hit.object_type, hit.object_id, hit.rank, 0 AS hop
    FROM ranked_objects AS hit
    UNION ALL
    SELECT relation.target_type, relation.target_id, hit.rank, 1 AS hop
    FROM ranked_objects AS hit
    JOIN relations AS relation
      ON relation.tenant_id = %(tenant_id)s
     AND relation.source_type = hit.object_type
     AND relation.source_id = hit.object_id
    WHERE relation.target_type IN ('event', 'claim')
      AND relation.relation_type IN (
          'asserts', 'contains', 'same_episode', 'supports',
          'contradicts', 'supersedes', 'before', 'after'
      )
    UNION ALL
    SELECT relation.source_type, relation.source_id, hit.rank, 1 AS hop
    FROM ranked_objects AS hit
    JOIN relations AS relation
      ON relation.tenant_id = %(tenant_id)s
     AND relation.target_type = hit.object_type
     AND relation.target_id = hit.object_id
    WHERE relation.source_type IN ('event', 'claim')
      AND relation.relation_type IN (
          'asserts', 'contains', 'same_episode', 'supports',
          'contradicts', 'supersedes', 'before', 'after'
      )
    UNION ALL
    SELECT neighbor.object_type, neighbor.object_id, hit.rank, 2 AS hop
    FROM ranked_objects AS hit
    JOIN LATERAL (
        SELECT peer.source_type AS object_type, peer.source_id AS object_id
        FROM relations AS anchor
        JOIN relations AS peer
          ON peer.tenant_id = anchor.tenant_id
         AND peer.target_type = 'entity'
         AND peer.target_id = anchor.target_id
         AND peer.relation_type IN ('mentions', 'about')
         AND peer.source_type IN ('event', 'claim')
        WHERE anchor.tenant_id = %(tenant_id)s
          AND anchor.source_type = hit.object_type
          AND anchor.source_id = hit.object_id
          AND anchor.relation_type IN ('mentions', 'about')
          AND anchor.target_type = 'entity'
          AND (peer.source_type, peer.source_id) <> (hit.object_type, hit.object_id)
        ORDER BY peer.created_at DESC, peer.source_type, peer.source_id
        LIMIT %(graph_neighbor_limit)s
    ) AS neighbor ON true
),
ranked_memories AS (
    SELECT relation.tenant_id,
           relation.target_id AS memory_id,
           min(hit.rank + hit.hop * cardinality(%(object_ids)s::text[])) AS graph_rank
    FROM relations AS relation
    JOIN related_objects AS hit
      ON hit.object_type = relation.source_type
     AND hit.object_id = relation.source_id
    WHERE relation.tenant_id = %(tenant_id)s
      AND relation.relation_type = 'represented_by'
      AND relation.target_type = 'memory_record'
    GROUP BY relation.tenant_id, relation.target_id
)
{MEMORY_SELECT_SQL}
JOIN ranked_memories AS dense
  ON dense.tenant_id = memory.tenant_id AND dense.memory_id = memory.memory_id
WHERE memory.tenant_id = %(tenant_id)s
{_STRUCTURED_RECALL_FILTER_SQL}
ORDER BY dense.graph_rank, memory.strength DESC, memory.occurred_at DESC, memory.memory_id
LIMIT %(limit)s
"""
