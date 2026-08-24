"""PostgreSQL candidates and the verdict write for cross-clip entity resolution.

Two things here are deliberately different from the sibling consolidation adapters.

Candidacy excludes entities whose `canonical_name` is null: those are keyed by an edge
identity signal and are already stable across clips, so guessing about them would undo a
stronger answer.

The verdict write upserts instead of inserting. `relation_id` is keyed on the pair and not on
the verdict, so one pair owns exactly one row and a re-judgement replaces it — which the
shared relation writer cannot express, because it raises when a conflicting row differs in
any column and `created_at` moves every sweep.

The cue the judge was required to name is written beside the edge, into
`entity_resolution_verdicts`, under that same pair-keyed id. It is stored apart from the edge
because it justifies the adjudication rather than describing the edge; every other relation
kind would carry such a column empty. Both writes share one transaction, so an edge is never
committed without the reason an operator would need to audit it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application.entity_resolution import (
    EntityCandidate,
    EntityCandidatePage,
    EntityCandidateRequest,
    EntityPair,
    EntityResolutionWrite,
)
from mindbridge.core import (
    Entity,
    EntityId,
    EntityType,
    EvidenceId,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import (
    PostgresStoreOperations,
    tenant_connection,
)

EntityCandidateRow: TypeAlias = tuple[str, str, str, str, datetime, list[str]]
_PairRow: TypeAlias = tuple[str, str, int]

# Both verdicts settle a pair. The question the skip asks is "has this been judged", not
# "which way", so re-judging is opt-in rather than a consequence of having answered no.
_SETTLED_SQL = """
NOT EXISTS (
    SELECT 1
    FROM relations AS decided
    WHERE decided.tenant_id = seed.tenant_id
      AND decided.source_type = 'entity'
      AND decided.target_type = 'entity'
      AND decided.relation_type IN ('same_as', 'not_same_as')
      AND (
          (decided.source_id = seed.entity_id AND decided.target_id = peer.entity_id)
          OR
          (decided.source_id = peer.entity_id AND decided.target_id = seed.entity_id)
      )
)
"""

_ENTITY_SEEDS_SQL = """
SELECT entity.entity_id
FROM entities AS entity
WHERE entity.tenant_id = %(tenant_id)s
  AND entity.canonical_name IS NOT NULL
  AND entity.entity_type = ANY(%(entity_types)s)
  AND (%(after_entity_id)s::text IS NULL OR entity.entity_id > %(after_entity_id)s)
ORDER BY entity.entity_id
LIMIT %(limit)s
"""

# An entity's temporal extent is the span of the events that mention it. Two entities are
# candidates when those extents are within the configured gap of each other.
_ENTITY_PAIRS_SQL = f"""
WITH seed_ids AS (
    SELECT entity_id FROM unnest(%(seed_ids)s::text[]) AS seed(entity_id)
),
extent AS (
    SELECT mention.entity_id,
           min(event.occurred_at) AS first_at,
           max(event.ended_at) AS last_at
    FROM entity_mentions AS mention
    JOIN events AS event
      ON event.tenant_id = mention.tenant_id
     AND event.event_id = mention.event_id
    WHERE mention.tenant_id = %(tenant_id)s
    GROUP BY mention.entity_id
),
candidate AS (
    SELECT seed.entity_id AS left_id,
           peer.entity_id AS right_id,
           row_number() OVER (
               PARTITION BY seed.entity_id
               ORDER BY coalesce(
                            1 - (seed_vector.embedding <=> peer_vector.embedding), 0
                        ) DESC,
                        peer.entity_id
           ) AS peer_rank
    FROM seed_ids
    JOIN entities AS seed
      ON seed.tenant_id = %(tenant_id)s AND seed.entity_id = seed_ids.entity_id
    JOIN entities AS peer
      ON peer.tenant_id = seed.tenant_id
     AND peer.entity_id > seed.entity_id
     AND peer.entity_type = seed.entity_type
     AND peer.canonical_name IS NOT NULL
    JOIN extent AS seed_extent ON seed_extent.entity_id = seed.entity_id
    JOIN extent AS peer_extent ON peer_extent.entity_id = peer.entity_id
    LEFT JOIN embeddings AS seed_vector
      ON seed_vector.tenant_id = seed.tenant_id
     AND seed_vector.object_type = 'entity'
     AND seed_vector.object_id = seed.entity_id
    LEFT JOIN embeddings AS peer_vector
      ON peer_vector.tenant_id = seed.tenant_id
     AND peer_vector.object_type = 'entity'
     AND peer_vector.object_id = peer.entity_id
     AND peer_vector.space_id = seed_vector.space_id
     AND peer_vector.task = seed_vector.task
    WHERE seed_extent.first_at
              <= peer_extent.last_at + make_interval(secs => %(maximum_gap_seconds)s)
      AND peer_extent.first_at
              <= seed_extent.last_at + make_interval(secs => %(maximum_gap_seconds)s)
      AND (%(readjudicate)s OR {_SETTLED_SQL})
),
bounded AS (
    SELECT left_id,
           right_id,
           row_number() OVER (ORDER BY left_id, right_id) AS pair_rank,
           count(*) OVER () AS pair_total
    FROM candidate
    WHERE peer_rank <= %(candidate_limit)s
)
SELECT left_id, right_id, pair_total
FROM bounded
WHERE pair_rank <= %(maximum_pairs)s
ORDER BY left_id, right_id
"""

_READ_ENTITY_ROWS_SQL = """
SELECT entity.entity_id,
       entity.tenant_id,
       entity.entity_type,
       entity.canonical_name,
       entity.created_at,
       ARRAY(
           SELECT DISTINCT mention.evidence_id
           FROM entity_mentions AS mention
           WHERE mention.tenant_id = entity.tenant_id
             AND mention.entity_id = entity.entity_id
             AND mention.evidence_id IS NOT NULL
           ORDER BY mention.evidence_id
       ) AS evidence_ids
FROM entities AS entity
WHERE entity.tenant_id = %s AND entity.entity_id = ANY(%s)
"""

# DO UPDATE, not DO NOTHING: a re-judgement has to be able to replace a verdict. The WHERE
# keeps an unchanged re-run from touching the row, so the reported count means "verdicts that
# actually changed" rather than "rows offered".
_UPSERT_VERDICT_SQL = """
INSERT INTO relations (
    tenant_id, relation_id, source_type, source_id,
    relation_type, target_type, target_id, created_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (tenant_id, relation_id) DO UPDATE
SET relation_type = EXCLUDED.relation_type,
    created_at = EXCLUDED.created_at
WHERE relations.relation_type <> EXCLUDED.relation_type
"""

# Unconditional DO UPDATE, unlike the edge above. A re-judgement that reaches the same
# answer for a different reason leaves the edge untouched but still supersedes the cue: the
# stored justification has to be the one the current verdict actually rested on. Keyed on
# relation_id, so the pair's single verdict row is replaced rather than appended to.
_UPSERT_CUE_SQL = """
INSERT INTO entity_resolution_verdicts (
    tenant_id, relation_id, confidence, discriminating_cue, decided_at
) VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (tenant_id, relation_id) DO UPDATE
SET confidence = EXCLUDED.confidence,
    discriminating_cue = EXCLUDED.discriminating_cue,
    decided_at = EXCLUDED.decided_at
"""


def _candidate_from_row(row: EntityCandidateRow) -> EntityCandidate:
    entity_id, tenant_id, entity_type, canonical_name, created_at, evidence_ids = row
    return EntityCandidate(
        entity=Entity(
            entity_id=EntityId(entity_id),
            tenant_id=TenantId(tenant_id),
            entity_type=EntityType(entity_type),
            canonical_name=canonical_name,
            created_at=created_at,
        ),
        evidence_ids=tuple(EvidenceId(value) for value in evidence_ids),
    )


class EntityCandidateOperations(PostgresStoreOperations):
    async def list_entity_candidates(
        self,
        request: EntityCandidateRequest,
    ) -> EntityCandidatePage:
        """Return one stable seed page paired within its type, time window, and budget."""
        async with tenant_connection(self._pool, request.tenant_id) as connection:
            cursor = await connection.execute(
                _ENTITY_SEEDS_SQL,
                {
                    "tenant_id": request.tenant_id,
                    "entity_types": [item.value for item in request.entity_types],
                    "after_entity_id": request.after_entity_id,
                    "limit": request.limit + 1,
                },
            )
            seed_ids = tuple([EntityId(cast(tuple[str], row)[0]) async for row in cursor])
            page_seed_ids = seed_ids[: request.limit]
            next_cursor = page_seed_ids[-1] if len(seed_ids) > request.limit else None
            if not page_seed_ids:
                return EntityCandidatePage(
                    pairs=(), scanned_count=0, dropped_pair_count=0, next_cursor=None
                )

            cursor = await connection.execute(
                _ENTITY_PAIRS_SQL,
                {
                    "tenant_id": request.tenant_id,
                    "seed_ids": list(page_seed_ids),
                    "maximum_gap_seconds": request.maximum_gap_seconds,
                    "candidate_limit": request.candidate_limit,
                    "maximum_pairs": request.maximum_pairs,
                    "readjudicate": request.readjudicate,
                },
            )
            pair_rows = [cast(_PairRow, row) async for row in cursor]
            if not pair_rows:
                return EntityCandidatePage(
                    pairs=(),
                    scanned_count=len(page_seed_ids),
                    dropped_pair_count=0,
                    next_cursor=next_cursor,
                )
            # Every row carries the same window total, so the maximum_pairs bound is reported
            # rather than silently truncating: a sweep that looked at less than it found says
            # so. candidate_limit is deliberately not counted here — it is the shortlist
            # strategy (the top few peers per seed by vector affinity), not a budget cut to
            # work the sweep meant to do.
            dropped = max(0, pair_rows[0][2] - len(pair_rows))

            entity_ids = tuple(dict.fromkeys(value for row in pair_rows for value in row[:2]))
            cursor = await connection.execute(
                _READ_ENTITY_ROWS_SQL,
                (request.tenant_id, list(entity_ids)),
            )
            by_id = {
                candidate.entity.entity_id: candidate
                for candidate in [
                    _candidate_from_row(cast(EntityCandidateRow, row)) async for row in cursor
                ]
            }
        return EntityCandidatePage(
            pairs=tuple(
                EntityPair(left=by_id[EntityId(left)], right=by_id[EntityId(right)])
                for left, right, _ in pair_rows
            ),
            scanned_count=len(page_seed_ids),
            dropped_pair_count=dropped,
            next_cursor=next_cursor,
        )

    async def commit_entity_resolution(
        self,
        tenant_id: TenantId,
        write: EntityResolutionWrite,
    ) -> int:
        """Record each pair's verdict and the cue behind it, replacing any it already had."""
        if not write.decided:
            return 0
        changed = 0
        async with tenant_connection(self._pool, tenant_id) as connection:
            for relation, adjudication in write.decided:
                cursor = await connection.execute(
                    _UPSERT_VERDICT_SQL,
                    (
                        relation.tenant_id,
                        relation.relation_id,
                        relation.source_type.value,
                        relation.source_id,
                        relation.relation_type.value,
                        relation.target_type.value,
                        relation.target_id,
                        relation.created_at,
                    ),
                )
                # Counted from the edge alone. A re-run that reaches the same verdict changed
                # nothing an operator would call a decision, even when it reworded its cue,
                # and the caller reports this as "verdicts that actually changed".
                changed += cursor.rowcount
                # Never gated on that rowcount: the edge row is left alone when the direction
                # holds, and gating here would strand the superseded cue beside a fresh
                # verdict. The foreign key is why this runs second.
                await connection.execute(
                    _UPSERT_CUE_SQL,
                    (
                        relation.tenant_id,
                        relation.relation_id,
                        adjudication.confidence,
                        adjudication.discriminating_cue,
                        relation.created_at,
                    ),
                )
        return changed
