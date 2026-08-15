"""PostgreSQL candidates for bounded hierarchical memory summaries."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from mindbridge.application.summary_consolidation import (
    SummaryCandidate,
    SummaryCandidateCursor,
    SummaryCandidatePage,
    SummaryCandidateRequest,
)
from mindbridge.core import EntityId, MemoryId
from mindbridge.infrastructure._postgres_memory_rows import (
    MEMORY_NOT_TOMBSTONED_SQL,
    MEMORY_SELECT_SQL,
    MemoryRow,
    memory_from_row,
)
from mindbridge.infrastructure._postgres_types import DatabasePool, tenant_connection


async def list_summary_candidates(
    pool: DatabasePool,
    request: SummaryCandidateRequest,
) -> SummaryCandidatePage:
    """Return one stable seed page expanded by time, entity, or aligned-vector affinity."""
    async with tenant_connection(pool, request.tenant_id) as connection:
        cursor = await connection.execute(
            _SUMMARY_SEEDS_SQL,
            {
                "tenant_id": request.tenant_id,
                "evaluated_at": request.evaluated_at,
                "after_occurred_at": (
                    request.after_cursor.occurred_at if request.after_cursor is not None else None
                ),
                "after_memory_id": (
                    request.after_cursor.memory_id if request.after_cursor is not None else None
                ),
                "limit": request.limit + 1,
            },
        )
        seed_cursors = tuple(
            [_summary_cursor_from_row(cast(tuple[object, ...], row)) async for row in cursor]
        )
        page_seed_cursors = seed_cursors[: request.limit]
        page_seed_ids = tuple(cursor.memory_id for cursor in page_seed_cursors)
        if not page_seed_ids:
            return SummaryCandidatePage(candidates=(), scanned_count=0, next_cursor=None)

        cursor = await connection.execute(
            _RELATED_MEMORY_IDS_SQL,
            {
                "tenant_id": request.tenant_id,
                "evaluated_at": request.evaluated_at,
                "seed_ids": list(page_seed_ids),
                "maximum_gap_seconds": request.maximum_gap_seconds,
                "minimum_similarity": request.minimum_similarity,
                "candidate_limit": min(request.limit * 4, 64),
            },
        )
        memory_ids = tuple([MemoryId(cast(tuple[str], row)[0]) async for row in cursor])
        cursor = await connection.execute(
            _READ_SUMMARY_CANDIDATES_SQL,
            {
                "tenant_id": request.tenant_id,
                "evaluated_at": request.evaluated_at,
                "memory_ids": list(memory_ids),
            },
        )
        candidates = tuple(
            [_summary_candidate_from_row(cast(tuple[object, ...], row)) async for row in cursor]
        )
    return SummaryCandidatePage(
        candidates=candidates,
        scanned_count=len(page_seed_ids),
        next_cursor=(page_seed_cursors[-1] if len(seed_cursors) > request.limit else None),
    )


def _summary_cursor_from_row(row: tuple[object, ...]) -> SummaryCandidateCursor:
    return SummaryCandidateCursor(
        memory_id=MemoryId(cast(str, row[0])),
        occurred_at=cast(datetime, row[1]),
    )


def _summary_candidate_from_row(row: tuple[object, ...]) -> SummaryCandidate:
    return SummaryCandidate(
        memory=memory_from_row(cast(MemoryRow, row[:-1])),
        entity_ids=tuple(EntityId(value) for value in cast(list[str], row[-1])),
    )


_CURRENT_SUMMARY_SOURCE_SQL = f"""
memory.superseded_at IS NULL
AND memory.created_at < %(evaluated_at)s
AND memory.memory_type IN ('episodic', 'semantic')
AND memory.verification_status IN ('verified', 'attested')
AND {MEMORY_NOT_TOMBSTONED_SQL}
AND NOT EXISTS (
    SELECT 1 FROM relations AS parent
    WHERE parent.tenant_id = memory.tenant_id
      AND parent.source_type = 'memory_record'
      AND parent.relation_type = 'contains'
      AND parent.target_type = 'memory_record'
      AND parent.target_id = memory.memory_id
)
"""

_SUMMARY_SEEDS_SQL = f"""
SELECT memory.memory_id, memory.occurred_at
FROM memory_records AS memory
WHERE memory.tenant_id = %(tenant_id)s
  AND {_CURRENT_SUMMARY_SOURCE_SQL}
  AND (
      %(after_occurred_at)s::timestamptz IS NULL
      OR (memory.occurred_at, memory.memory_id) >
         (%(after_occurred_at)s::timestamptz, %(after_memory_id)s::text)
  )
ORDER BY memory.occurred_at, memory.memory_id
LIMIT %(limit)s
"""

_RELATED_MEMORY_IDS_SQL = f"""
WITH seed_ids AS (
    SELECT memory_id FROM unnest(%(seed_ids)s::text[]) AS seed(memory_id)
),
memory_embeddings AS (
    SELECT embedding.tenant_id,
           embedding.object_id AS memory_id,
           embedding.space_id,
           embedding.space_revision,
           embedding.task,
           embedding.embedding
    FROM embeddings AS embedding
    WHERE embedding.tenant_id = %(tenant_id)s
      AND embedding.object_type = 'memory_record'
    UNION ALL
    SELECT relation.tenant_id,
           relation.target_id AS memory_id,
           embedding.space_id,
           embedding.space_revision,
           embedding.task,
           embedding.embedding
    FROM relations AS relation
    JOIN embeddings AS embedding
      ON embedding.tenant_id = relation.tenant_id
     AND embedding.object_type = relation.source_type
     AND embedding.object_id = relation.source_id
    WHERE relation.tenant_id = %(tenant_id)s
      AND relation.source_type IN ('event', 'claim')
      AND relation.relation_type = 'represented_by'
      AND relation.target_type = 'memory_record'
),
related_pairs AS (
    SELECT seed.memory_id AS seed_id, peer.memory_id AS peer_id
    FROM seed_ids
    JOIN memory_records AS seed
      ON seed.tenant_id = %(tenant_id)s AND seed.memory_id = seed_ids.memory_id
    JOIN memory_records AS peer
      ON peer.tenant_id = seed.tenant_id
     AND (peer.occurred_at, peer.memory_id) > (seed.occurred_at, seed.memory_id)
    WHERE {_CURRENT_SUMMARY_SOURCE_SQL.replace("memory.", "peer.")}
      AND (
          (
              peer.occurred_at
                  <= seed.ended_at + make_interval(secs => %(maximum_gap_seconds)s)
              AND seed.occurred_at
                  <= peer.ended_at + make_interval(secs => %(maximum_gap_seconds)s)
          )
          OR EXISTS (
              SELECT 1
              FROM memory_evidence AS seed_link
              JOIN entity_mentions AS seed_mention
                ON seed_mention.tenant_id = seed_link.tenant_id
               AND seed_mention.evidence_id = seed_link.evidence_id
              JOIN entity_mentions AS peer_mention
                ON peer_mention.tenant_id = seed_mention.tenant_id
               AND peer_mention.entity_id = seed_mention.entity_id
              JOIN memory_evidence AS peer_link
                ON peer_link.tenant_id = peer_mention.tenant_id
               AND peer_link.evidence_id = peer_mention.evidence_id
              WHERE seed_link.tenant_id = seed.tenant_id
                AND seed_link.memory_id = seed.memory_id
                AND peer_link.memory_id = peer.memory_id
          )
          OR EXISTS (
              SELECT 1
              FROM memory_embeddings AS seed_embedding
              JOIN memory_embeddings AS peer_embedding
                ON peer_embedding.tenant_id = seed_embedding.tenant_id
               AND peer_embedding.space_id = seed_embedding.space_id
               AND peer_embedding.space_revision = seed_embedding.space_revision
               AND peer_embedding.task = seed_embedding.task
              WHERE seed_embedding.tenant_id = seed.tenant_id
                AND seed_embedding.memory_id = seed.memory_id
                AND peer_embedding.memory_id = peer.memory_id
                AND 1 - (seed_embedding.embedding <=> peer_embedding.embedding)
                    >= %(minimum_similarity)s
          )
      )
),
related_ids AS (
    SELECT seed_id AS memory_id FROM related_pairs
    UNION
    SELECT peer_id AS memory_id FROM related_pairs
)
SELECT related.memory_id
FROM related_ids AS related
JOIN memory_records AS memory
  ON memory.tenant_id = %(tenant_id)s
 AND memory.memory_id = related.memory_id
ORDER BY memory.occurred_at, memory.memory_id
LIMIT %(candidate_limit)s
"""

_READ_SUMMARY_CANDIDATES_SQL = f"""
SELECT selected.*,
       ARRAY(
           SELECT DISTINCT mention.entity_id
           FROM memory_evidence AS link
           JOIN entity_mentions AS mention
             ON mention.tenant_id = link.tenant_id
            AND mention.evidence_id = link.evidence_id
           WHERE link.tenant_id = selected.tenant_id
             AND link.memory_id = selected.memory_id
           ORDER BY mention.entity_id
       ) AS entity_ids
FROM (
    {MEMORY_SELECT_SQL}
    WHERE memory.tenant_id = %(tenant_id)s
      AND memory.memory_id = ANY(%(memory_ids)s)
      AND {_CURRENT_SUMMARY_SOURCE_SQL}
) AS selected
ORDER BY selected.occurred_at, selected.memory_id
"""
