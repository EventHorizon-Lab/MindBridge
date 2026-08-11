"""PostgreSQL candidate discovery for bounded episode consolidation."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application import EpisodeCandidatePage, EpisodeCandidateRequest
from mindbridge.core import (
    Event,
    EventHierarchyLevel,
    EventId,
    EventStatus,
    EvidenceId,
    ModelReference,
    ObservationId,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

EventRow: TypeAlias = tuple[
    str,
    str,
    list[str],
    list[str],
    datetime,
    datetime,
    str,
    float,
    datetime,
    str,
    str,
    str,
    str | None,
    str,
    str,
]


async def list_episode_candidates(
    pool: DatabasePool,
    request: EpisodeCandidateRequest,
) -> EpisodeCandidatePage:
    """Return one stable seed page expanded by time, entity, or vector affinity."""
    async with tenant_connection(pool, request.tenant_id) as connection:
        cursor = await connection.execute(
            _EPISODE_SEEDS_SQL,
            {
                "tenant_id": request.tenant_id,
                "evaluated_at": request.evaluated_at,
                "after_event_id": request.after_event_id,
                "limit": request.limit + 1,
            },
        )
        seed_ids = tuple([EventId(cast(tuple[str], row)[0]) async for row in cursor])
        page_seed_ids = seed_ids[: request.limit]
        if not page_seed_ids:
            return EpisodeCandidatePage(events=(), scanned_count=0, next_cursor=None)

        cursor = await connection.execute(
            _RELATED_EVENT_IDS_SQL,
            {
                "tenant_id": request.tenant_id,
                "evaluated_at": request.evaluated_at,
                "seed_ids": list(page_seed_ids),
                "maximum_gap_seconds": request.maximum_gap_seconds,
                "minimum_similarity": request.minimum_similarity,
                "candidate_limit": min(request.limit * 4, 64),
            },
        )
        event_ids = tuple([EventId(cast(tuple[str], row)[0]) async for row in cursor])
        events = await _read_events(connection, request.tenant_id, event_ids)
    return EpisodeCandidatePage(
        events=events,
        scanned_count=len(page_seed_ids),
        next_cursor=(page_seed_ids[-1] if len(seed_ids) > request.limit else None),
    )


async def _read_events(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    event_ids: tuple[EventId, ...],
) -> tuple[Event, ...]:
    if not event_ids:
        return ()
    cursor = await connection.execute(_READ_EVENTS_SQL, (tenant_id, list(event_ids)))
    return tuple([_event_from_row(cast(EventRow, row)) async for row in cursor])


def _event_from_row(row: EventRow) -> Event:
    (
        event_id,
        tenant_id,
        observation_ids,
        evidence_ids,
        occurred_at,
        ended_at,
        description,
        salience,
        created_at,
        model_id,
        model_revision,
        prompt_version,
        parent_event_id,
        hierarchy_level,
        status,
    ) = row
    return Event(
        event_id=EventId(event_id),
        tenant_id=TenantId(tenant_id),
        observation_ids=tuple(ObservationId(value) for value in observation_ids),
        evidence_ids=tuple(EvidenceId(value) for value in evidence_ids),
        occurred_at=occurred_at,
        ended_at=ended_at,
        description=description,
        salience=salience,
        created_at=created_at,
        model_reference=ModelReference(model_id=model_id, revision=model_revision),
        prompt_version=prompt_version,
        parent_event_id=EventId(parent_event_id) if parent_event_id is not None else None,
        hierarchy_level=EventHierarchyLevel(hierarchy_level),
        status=EventStatus(status),
    )


_EPISODE_SEEDS_SQL = """
SELECT event_id
FROM events
WHERE tenant_id = %(tenant_id)s
  AND hierarchy_level = 'event'
  AND status = 'active'
  AND parent_event_id IS NULL
  AND created_at <= %(evaluated_at)s
  AND (%(after_event_id)s::text IS NULL OR event_id > %(after_event_id)s)
ORDER BY event_id
LIMIT %(limit)s
"""

_RELATED_EVENT_IDS_SQL = """
WITH seed_ids AS (
    SELECT event_id FROM unnest(%(seed_ids)s::text[]) AS seed(event_id)
),
related_pairs AS (
    SELECT seed.event_id AS seed_id, peer.event_id AS peer_id
    FROM seed_ids
    JOIN events AS seed
      ON seed.tenant_id = %(tenant_id)s AND seed.event_id = seed_ids.event_id
    JOIN events AS peer
      ON peer.tenant_id = seed.tenant_id
     AND peer.event_id > seed.event_id
     AND peer.hierarchy_level = 'event'
     AND peer.status = 'active'
     AND peer.parent_event_id IS NULL
     AND peer.created_at <= %(evaluated_at)s
     AND peer.occurred_at <= seed.ended_at + make_interval(secs => %(maximum_gap_seconds)s)
     AND seed.occurred_at <= peer.ended_at + make_interval(secs => %(maximum_gap_seconds)s)
    WHERE (
        (peer.occurred_at <= seed.ended_at AND seed.occurred_at <= peer.ended_at)
        OR EXISTS (
            SELECT 1
            FROM entity_mentions AS seed_mention
            JOIN entity_mentions AS peer_mention
              ON peer_mention.tenant_id = seed_mention.tenant_id
             AND peer_mention.entity_id = seed_mention.entity_id
            WHERE seed_mention.tenant_id = seed.tenant_id
              AND seed_mention.event_id = seed.event_id
              AND peer_mention.event_id = peer.event_id
        )
        OR EXISTS (
            SELECT 1
            FROM embeddings AS seed_embedding
            JOIN embeddings AS peer_embedding
              ON peer_embedding.tenant_id = seed_embedding.tenant_id
             AND peer_embedding.space_id = seed_embedding.space_id
             AND peer_embedding.space_revision = seed_embedding.space_revision
             AND peer_embedding.task = seed_embedding.task
            WHERE seed_embedding.tenant_id = seed.tenant_id
              AND seed_embedding.object_type = 'event'
              AND seed_embedding.object_id = seed.event_id
              AND peer_embedding.object_type = 'event'
              AND peer_embedding.object_id = peer.event_id
              AND 1 - (seed_embedding.embedding <=> peer_embedding.embedding)
                  >= %(minimum_similarity)s
        )
    )
),
related_ids AS (
    SELECT seed_id AS event_id FROM related_pairs
    UNION
    SELECT peer_id AS event_id FROM related_pairs
)
SELECT event_id FROM related_ids ORDER BY event_id LIMIT %(candidate_limit)s
"""

_READ_EVENTS_SQL = """
SELECT event.event_id,
       event.tenant_id,
       ARRAY(
           SELECT link.observation_id
           FROM event_observations AS link
           WHERE link.tenant_id = event.tenant_id AND link.event_id = event.event_id
           ORDER BY link.observation_id
       ) AS observation_ids,
       ARRAY(
           SELECT link.evidence_id
           FROM event_evidence AS link
           WHERE link.tenant_id = event.tenant_id AND link.event_id = event.event_id
           ORDER BY link.evidence_id
       ) AS evidence_ids,
       event.occurred_at,
       event.ended_at,
       event.description,
       event.salience,
       event.created_at,
       event.model_id,
       event.model_revision,
       event.prompt_version,
       event.parent_event_id,
       event.hierarchy_level,
       event.status
FROM events AS event
WHERE event.tenant_id = %s AND event.event_id = ANY(%s)
ORDER BY event.occurred_at, event.event_id
"""
