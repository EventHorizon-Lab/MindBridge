"""Database-local cascade cleanup for explicit forgetting."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from mindbridge.core import DeletionTombstone, MediaObject, MediaObjectId, TenantId
from mindbridge.infrastructure._postgres_media import read_media_objects_on_connection
from mindbridge.infrastructure._postgres_types import DatabaseConnection


async def lock_exclusive_observation_media(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
) -> tuple[MediaObject, ...]:
    lock_cursor = await connection.execute(
        """
        SELECT media.media_object_id
        FROM media_objects AS media
        JOIN observation_media AS link
          ON link.tenant_id = media.tenant_id
         AND link.media_object_id = media.media_object_id
        WHERE link.tenant_id = %s AND link.observation_id = %s
        FOR UPDATE OF media
        """,
        (tombstone.tenant_id, tombstone.target_id),
    )
    await lock_cursor.fetchall()
    cursor = await connection.execute(
        """
        SELECT link.media_object_id
        FROM observation_media AS link
        WHERE link.tenant_id = %s AND link.observation_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM observation_media AS other
              WHERE other.tenant_id = link.tenant_id
                AND other.media_object_id = link.media_object_id
                AND other.observation_id <> link.observation_id
          )
        ORDER BY link.ordinal
        """,
        (tombstone.tenant_id, tombstone.target_id),
    )
    media_object_ids = tuple([MediaObjectId(cast(tuple[str], row)[0]) async for row in cursor])
    return await read_media_objects_on_connection(
        connection,
        tombstone.tenant_id,
        media_object_ids,
    )


async def delete_memory_scope(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    memory_ids: tuple[str, ...],
) -> None:
    if not memory_ids:
        return
    scoped_memory_ids = await _select_ids(
        connection,
        """
        WITH RECURSIVE memory_scope(memory_id) AS (
            SELECT memory_id FROM unnest(%s::text[]) AS initial(memory_id)
            UNION
            SELECT relation.source_id
            FROM memory_scope AS child
            JOIN relations AS relation
              ON relation.tenant_id = %s
             AND relation.source_type = 'memory_record'
             AND relation.relation_type = 'contains'
             AND relation.target_type = 'memory_record'
             AND relation.target_id = child.memory_id
        )
        SELECT memory_id FROM memory_scope
        """,
        (list(memory_ids), tenant_id),
    )
    feedback_cursor = await connection.execute(
        """
        SELECT feedback_id FROM memory_feedback
        WHERE tenant_id = %s
          AND (memory_id = ANY(%s) OR corrected_memory_id = ANY(%s))
        """,
        (tenant_id, list(scoped_memory_ids), list(scoped_memory_ids)),
    )
    feedback_ids = tuple([cast(tuple[str], row)[0] async for row in feedback_cursor])
    await connection.execute(
        """
        DELETE FROM embeddings
        WHERE tenant_id = %s AND object_type = 'memory_record' AND object_id = ANY(%s)
        """,
        (tenant_id, list(scoped_memory_ids)),
    )
    await connection.execute(
        """
        DELETE FROM relations
        WHERE tenant_id = %s
          AND ((source_type = 'memory_record' AND source_id = ANY(%s))
            OR (target_type = 'memory_record' AND target_id = ANY(%s)))
        """,
        (tenant_id, list(scoped_memory_ids), list(scoped_memory_ids)),
    )
    if feedback_ids:
        await connection.execute(
            """
            DELETE FROM idempotency_keys
            WHERE tenant_id = %s AND operation = 'feedback' AND resource_id = ANY(%s)
            """,
            (tenant_id, list(feedback_ids)),
        )
        await connection.execute(
            "DELETE FROM memory_feedback WHERE tenant_id = %s AND feedback_id = ANY(%s)",
            (tenant_id, list(feedback_ids)),
        )
    await connection.execute(
        """
        DELETE FROM idempotency_keys
        WHERE tenant_id = %s AND operation = 'remember' AND resource_id = ANY(%s)
        """,
        (tenant_id, list(scoped_memory_ids)),
    )
    await connection.execute(
        "DELETE FROM memory_records WHERE tenant_id = %s AND memory_id = ANY(%s)",
        (tenant_id, list(scoped_memory_ids)),
    )


async def delete_observation_scope(
    connection: DatabaseConnection,
    tombstone: DeletionTombstone,
    *,
    completed_at: datetime,
) -> None:
    tenant_id = tombstone.tenant_id
    observation_id = tombstone.target_id
    evidence_ids = await _select_ids(
        connection,
        "SELECT evidence_id FROM evidence_spans WHERE tenant_id = %s AND observation_id = %s",
        (tenant_id, observation_id),
    )
    event_ids = await _select_ids(
        connection,
        """
        SELECT event_id FROM event_observations
        WHERE tenant_id = %s AND observation_id = %s
        UNION
        SELECT event_id FROM event_evidence
        WHERE tenant_id = %s AND evidence_id = ANY(%s)
        """,
        (tenant_id, observation_id, tenant_id, list(evidence_ids)),
    )
    memory_ids = await _select_ids(
        connection,
        """
        SELECT memory_id FROM memory_evidence
        WHERE tenant_id = %s AND evidence_id = ANY(%s)
        UNION
        SELECT target_id FROM relations
        WHERE tenant_id = %s AND source_type = 'event' AND source_id = ANY(%s)
          AND target_type = 'memory_record'
        """,
        (tenant_id, list(evidence_ids), tenant_id, list(event_ids)),
    )
    claim_ids = await _select_ids(
        connection,
        """
        SELECT claim_id FROM claim_evidence
        WHERE tenant_id = %s AND evidence_id = ANY(%s)
        """,
        (tenant_id, list(evidence_ids)),
    )
    restorable_claim_ids = await _select_ids(
        connection,
        """
        SELECT target_id FROM relations
        WHERE tenant_id = %s
          AND source_type = 'claim'
          AND source_id = ANY(%s)
          AND relation_type = 'supersedes'
          AND target_type = 'claim'
          AND NOT (target_id = ANY(%s))
        """,
        (tenant_id, list(claim_ids), list(claim_ids)),
    )
    entity_ids = await _select_ids(
        connection,
        """
        SELECT DISTINCT entity_id FROM entity_mentions
        WHERE tenant_id = %s
          AND (event_id = ANY(%s) OR evidence_id = ANY(%s))
        """,
        (tenant_id, list(event_ids), list(evidence_ids)),
    )
    exclusive_media_ids = tuple(
        media.media_object_id
        for media in await lock_exclusive_observation_media(connection, tombstone)
    )

    await delete_memory_scope(connection, tenant_id, memory_ids)
    await _delete_typed_derivatives(connection, tenant_id, "claim", claim_ids)
    await _delete_typed_derivatives(connection, tenant_id, "event", event_ids)
    await _delete_typed_derivatives(connection, tenant_id, "evidence_span", evidence_ids)
    if claim_ids:
        await connection.execute(
            "DELETE FROM claims WHERE tenant_id = %s AND claim_id = ANY(%s)",
            (tenant_id, list(claim_ids)),
        )
    await _restore_claim_versions(
        connection,
        tenant_id,
        restorable_claim_ids,
        restored_at=completed_at,
    )
    if event_ids:
        await connection.execute(
            "DELETE FROM events WHERE tenant_id = %s AND event_id = ANY(%s)",
            (tenant_id, list(event_ids)),
        )
    await connection.execute(
        "DELETE FROM evidence_spans WHERE tenant_id = %s AND observation_id = %s",
        (tenant_id, observation_id),
    )
    await connection.execute(
        """
        DELETE FROM jobs
        WHERE tenant_id = %s AND job_type = 'process_observation'
          AND payload->>'observation_id' = %s
        """,
        (tenant_id, observation_id),
    )
    await connection.execute(
        """
        DELETE FROM idempotency_keys
        WHERE tenant_id = %s AND operation = 'observe' AND resource_id = %s
        """,
        (tenant_id, observation_id),
    )
    await connection.execute(
        "DELETE FROM observations WHERE tenant_id = %s AND observation_id = %s",
        (tenant_id, observation_id),
    )
    if exclusive_media_ids:
        await connection.execute(
            "DELETE FROM media_objects WHERE tenant_id = %s AND media_object_id = ANY(%s)",
            (tenant_id, list(exclusive_media_ids)),
        )
    if entity_ids:
        orphan_entity_ids = await _select_ids(
            connection,
            """
            SELECT entity.entity_id
            FROM entities AS entity
            WHERE entity.tenant_id = %s AND entity.entity_id = ANY(%s)
              AND NOT EXISTS (
                  SELECT 1 FROM entity_mentions AS mention
                  WHERE mention.tenant_id = entity.tenant_id
                    AND mention.entity_id = entity.entity_id
              )
            """,
            (tenant_id, list(entity_ids)),
        )
        await _delete_relations(connection, tenant_id, "entity", orphan_entity_ids)
        await connection.execute(
            """
            DELETE FROM entities
            WHERE tenant_id = %s AND entity_id = ANY(%s)
            """,
            (tenant_id, list(orphan_entity_ids)),
        )


async def _delete_typed_derivatives(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    object_type: str,
    object_ids: tuple[str, ...],
) -> None:
    if not object_ids:
        return
    await connection.execute(
        "DELETE FROM embeddings WHERE tenant_id = %s AND object_type = %s AND object_id = ANY(%s)",
        (tenant_id, object_type, list(object_ids)),
    )
    await _delete_relations(connection, tenant_id, object_type, object_ids)


async def _restore_claim_versions(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    claim_ids: tuple[str, ...],
    *,
    restored_at: datetime,
) -> None:
    if not claim_ids:
        return
    await connection.execute(
        """
        WITH restored AS (
            UPDATE claims AS claim SET superseded_at = NULL
            WHERE claim.tenant_id = %s
              AND claim.claim_id = ANY(%s)
              AND NOT EXISTS (
                  SELECT 1 FROM relations AS replacement
                  WHERE replacement.tenant_id = claim.tenant_id
                    AND replacement.source_type = 'claim'
                    AND replacement.relation_type = 'supersedes'
                    AND replacement.target_type = 'claim'
                    AND replacement.target_id = claim.claim_id
              )
            RETURNING claim.claim_id
        )
        UPDATE memory_records AS memory
        SET superseded_at = NULL,
            lifecycle_changed_at = GREATEST(memory.lifecycle_changed_at, %s)
        FROM restored
        JOIN relations AS representation
          ON representation.tenant_id = %s
         AND representation.source_type = 'claim'
         AND representation.source_id = restored.claim_id
         AND representation.relation_type = 'represented_by'
         AND representation.target_type = 'memory_record'
        WHERE memory.tenant_id = representation.tenant_id
          AND memory.memory_id = representation.target_id
          AND NOT EXISTS (
              SELECT 1 FROM memory_feedback AS feedback
              WHERE feedback.tenant_id = memory.tenant_id
                AND feedback.memory_id = memory.memory_id
                AND feedback.corrected_memory_id IS NOT NULL
          )
        """,
        (tenant_id, list(claim_ids), restored_at, tenant_id),
    )


async def _delete_relations(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    object_type: str,
    object_ids: tuple[str, ...],
) -> None:
    if not object_ids:
        return
    await connection.execute(
        """
        DELETE FROM relations
        WHERE tenant_id = %s
          AND ((source_type = %s AND source_id = ANY(%s))
            OR (target_type = %s AND target_id = ANY(%s)))
        """,
        (tenant_id, object_type, list(object_ids), object_type, list(object_ids)),
    )


async def _select_ids(
    connection: DatabaseConnection,
    query: str,
    parameters: tuple[object, ...],
) -> tuple[str, ...]:
    cursor = await connection.execute(query, parameters)
    return tuple([cast(tuple[str], row)[0] async for row in cursor])
