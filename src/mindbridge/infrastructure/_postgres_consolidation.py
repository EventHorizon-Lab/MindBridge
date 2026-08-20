"""PostgreSQL candidate discovery for bounded episode consolidation."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application.consolidation import EpisodeCandidatePage, EpisodeCandidateRequest
from mindbridge.application.episodes import EpisodeWrite
from mindbridge.core import (
    DomainInvariantError,
    Event,
    EventHierarchyLevel,
    EventId,
    EventStatus,
    EvidenceId,
    MemoryIntegrityError,
    ModelReference,
    ObservationId,
    TenantId,
)
from mindbridge.infrastructure._postgres_derived_records import (
    derived_memory_content_digest,
    write_event,
)
from mindbridge.infrastructure._postgres_embeddings import write_embedding_on_connection
from mindbridge.infrastructure._postgres_graph import write_relations
from mindbridge.infrastructure._postgres_memories import write_memory_on_connection
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    PostgresStoreOperations,
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
    str | None,
    str,
    str,
]
ChildEventState: TypeAlias = tuple[str | None, str, str]


async def commit_episode_consolidation(
    pool: DatabasePool,
    tenant_id: TenantId,
    writes: tuple[EpisodeWrite, ...],
) -> int:
    """Claim child Events and persist complete Episode aggregates atomically."""
    child_event_ids = [event_id for write in writes for event_id in write.child_event_ids]
    if any(write.episode.tenant_id != tenant_id for write in writes):
        raise DomainInvariantError("episode writes must remain in the requested tenant")
    if len(set(child_event_ids)) != len(child_event_ids):
        raise DomainInvariantError("episode writes cannot share child events")
    if not writes:
        return 0

    try:
        async with tenant_connection(pool, tenant_id) as connection:
            child_states = await _lock_child_events(connection, tenant_id, child_event_ids)
            committed_count = 0
            for write in writes:
                states = tuple(child_states.get(event_id) for event_id in write.child_event_ids)
                if any(state is None for state in states):
                    continue
                known_states = cast(tuple[ChildEventState, ...], states)
                if all(state[0] == write.episode.event_id for state in known_states):
                    await _write_episode_aggregate(connection, write)
                    continue
                if any(
                    parent_id is not None
                    or hierarchy_level != EventHierarchyLevel.EVENT.value
                    or status != EventStatus.ACTIVE.value
                    for parent_id, hierarchy_level, status in known_states
                ):
                    continue
                await _write_episode_aggregate(connection, write)
                cursor = await connection.execute(
                    """
                    UPDATE events SET parent_event_id = %s
                    WHERE tenant_id = %s
                      AND event_id = ANY(%s)
                      AND parent_event_id IS NULL
                      AND hierarchy_level = 'event'
                      AND status = 'active'
                    """,
                    (write.episode.event_id, tenant_id, list(write.child_event_ids)),
                )
                if cursor.rowcount != len(write.child_event_ids):
                    raise MemoryIntegrityError("locked Episode children changed unexpectedly")
                committed_count += 1
            return committed_count
    except ForeignKeyViolation as error:
        raise DomainInvariantError("episode references missing source data") from error


async def _lock_child_events(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    child_event_ids: list[EventId],
) -> dict[EventId, ChildEventState]:
    cursor = await connection.execute(
        """
        SELECT event_id, parent_event_id, hierarchy_level, status
        FROM events
        WHERE tenant_id = %s AND event_id = ANY(%s)
        ORDER BY event_id
        FOR UPDATE
        """,
        (tenant_id, list(child_event_ids)),
    )
    return {
        EventId(event_id): (parent_event_id, hierarchy_level, status)
        async for row in cursor
        for event_id, parent_event_id, hierarchy_level, status in (
            cast(tuple[str, str | None, str, str], row),
        )
    }


async def _write_episode_aggregate(
    connection: DatabaseConnection,
    write: EpisodeWrite,
) -> None:
    await write_event(connection, write.episode)
    await write_memory_on_connection(
        connection,
        write.memory,
        derived_memory_content_digest(write.memory),
    )
    await write_relations(connection, write.relations)
    await write_embedding_on_connection(connection, write.embedding)


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
        model_reference=ModelReference(model_id=model_id),
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
  AND created_at < %(evaluated_at)s
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
     AND peer.created_at < %(evaluated_at)s
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
       event.prompt_version,
       event.parent_event_id,
       event.hierarchy_level,
       event.status
FROM events AS event
WHERE event.tenant_id = %s AND event.event_id = ANY(%s)
ORDER BY event.occurred_at, event.event_id
"""


class EpisodeCandidateOperations(PostgresStoreOperations):
    async def list_episode_candidates(
        self,
        request: EpisodeCandidateRequest,
    ) -> EpisodeCandidatePage:
        """Return one stable seed page expanded by time, entity, or vector affinity."""
        async with tenant_connection(self._pool, request.tenant_id) as connection:
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
