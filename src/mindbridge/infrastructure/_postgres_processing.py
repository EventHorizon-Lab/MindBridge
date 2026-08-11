"""Atomic PostgreSQL commit for derived observation memory."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application import ObservationProcessingOutput
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    Event,
    JobId,
    MemoryIntegrityError,
    MemoryRecord,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
    derive_stable_id,
)
from mindbridge.infrastructure._postgres_embeddings import write_embedding_on_connection
from mindbridge.infrastructure._postgres_jobs import (
    mark_observation_processing_succeeded_on_connection,
)
from mindbridge.infrastructure._postgres_memories import write_memory_on_connection
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool


async def commit_observation_processing(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
    output: ObservationProcessingOutput,
) -> ObservationProcessingJob:
    """Commit all derived records and successful job state in one transaction."""
    _require_output_identity(tenant_id, observation_id, output)
    try:
        async with pool.connection() as connection:
            await _require_source_evidence(connection, tenant_id, observation_id, output)
            for event, memory in zip(output.events, output.memories, strict=True):
                await _write_event(connection, event)
                await write_memory_on_connection(
                    connection,
                    memory,
                    _memory_digest(memory),
                )
                await _write_event_memory_relation(connection, event, memory)
            for embedding in output.embeddings:
                await write_embedding_on_connection(connection, embedding)
            return await mark_observation_processing_succeeded_on_connection(
                connection,
                tenant_id,
                observation_id,
                job_id,
                attempt=attempt,
            )
    except ForeignKeyViolation as error:
        raise DomainInvariantError("derived observation references missing source data") from error


def _require_output_identity(
    tenant_id: TenantId,
    observation_id: ObservationId,
    output: ObservationProcessingOutput,
) -> None:
    for event, memory in zip(output.events, output.memories, strict=True):
        if event.tenant_id != tenant_id or memory.tenant_id != tenant_id:
            raise DomainInvariantError("derived records must remain in the source tenant")
        if event.observation_ids != (observation_id,):
            raise DomainInvariantError("derived event must reference its source observation")
    if any(embedding.tenant_id != tenant_id for embedding in output.embeddings):
        raise DomainInvariantError("derived embedding must remain in the source tenant")
    if any(
        embedding.object_type is not EmbeddedObjectType.EVIDENCE_SPAN
        for embedding in output.embeddings
    ):
        raise DomainInvariantError("observation processing only accepts evidence embeddings")


async def _require_source_evidence(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    output: ObservationProcessingOutput,
) -> None:
    cursor = await connection.execute(
        """
        SELECT evidence_id FROM evidence_spans
        WHERE tenant_id = %s AND observation_id = %s
        """,
        (tenant_id, observation_id),
    )
    source_evidence_ids = {cast(tuple[str], row)[0] async for row in cursor}
    referenced_evidence_ids = {
        str(evidence_id) for event in output.events for evidence_id in event.evidence_ids
    } | {embedding.object_id for embedding in output.embeddings}
    if not referenced_evidence_ids <= source_evidence_ids:
        raise DomainInvariantError("derived records reference evidence outside the observation")


async def _write_event(connection: DatabaseConnection, event: Event) -> None:
    content_digest = _event_digest(event)
    cursor = await connection.execute(
        """
        INSERT INTO events (
            tenant_id, event_id, description, salience, occurred_at, ended_at,
            model_id, model_revision, prompt_version, content_digest, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING event_id
        """,
        (
            event.tenant_id,
            event.event_id,
            event.description,
            event.salience,
            event.occurred_at,
            event.ended_at,
            event.model_reference.model_id,
            event.model_reference.revision,
            event.prompt_version,
            content_digest,
            event.created_at,
        ),
    )
    if await cursor.fetchone() is None:
        cursor = await connection.execute(
            """
            SELECT content_digest FROM events
            WHERE tenant_id = %s AND event_id = %s
            """,
            (event.tenant_id, event.event_id),
        )
        row = await cursor.fetchone()
        if row is None or cast(tuple[str], row)[0] != content_digest:
            raise DomainInvariantError("event identifier already stores different content")
    await _write_event_links(connection, event)


async def _write_event_links(connection: DatabaseConnection, event: Event) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO event_observations (tenant_id, event_id, observation_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (
                (event.tenant_id, event.event_id, observation_id)
                for observation_id in event.observation_ids
            ),
        )
        await cursor.executemany(
            """
            INSERT INTO event_evidence (tenant_id, event_id, evidence_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
            """,
            ((event.tenant_id, event.event_id, evidence_id) for evidence_id in event.evidence_ids),
        )


async def _write_event_memory_relation(
    connection: DatabaseConnection,
    event: Event,
    memory: MemoryRecord,
) -> None:
    relation_type = "represented_by"
    relation_id = derive_stable_id(
        "relation",
        event.event_id,
        relation_type,
        memory.memory_id,
    )
    values = (
        event.tenant_id,
        relation_id,
        "event",
        event.event_id,
        relation_type,
        "memory_record",
        memory.memory_id,
        event.created_at,
    )
    cursor = await connection.execute(
        """
        INSERT INTO relations (
            tenant_id, relation_id, source_type, source_id, relation_type,
            target_type, target_id, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING relation_id
        """,
        values,
    )
    if await cursor.fetchone() is not None:
        return
    cursor = await connection.execute(
        """
        SELECT tenant_id, relation_id, source_type, source_id, relation_type,
               target_type, target_id, created_at
        FROM relations WHERE tenant_id = %s AND relation_id = %s
        """,
        (event.tenant_id, relation_id),
    )
    row = await cursor.fetchone()
    if row is None or tuple(row) != values:
        raise MemoryIntegrityError("event-memory relation has conflicting identity")


def _event_digest(event: Event) -> str:
    return _digest(
        {
            "event_id": event.event_id,
            "tenant_id": event.tenant_id,
            "observation_ids": event.observation_ids,
            "evidence_ids": event.evidence_ids,
            "occurred_at": event.occurred_at.isoformat(),
            "ended_at": event.ended_at.isoformat(),
            "description": event.description,
            "salience": event.salience,
            "model_id": event.model_reference.model_id,
            "model_revision": event.model_reference.revision,
            "prompt_version": event.prompt_version,
            "created_at": event.created_at.isoformat(),
        }
    )


def _memory_digest(memory: MemoryRecord) -> str:
    return _digest(
        {
            "memory_id": memory.memory_id,
            "tenant_id": memory.tenant_id,
            "memory_type": memory.memory_type.value,
            "summary": memory.summary,
            "evidence_ids": memory.evidence_ids,
            "occurred_at": memory.occurred_at.isoformat(),
            "ended_at": memory.ended_at.isoformat(),
            "created_at": memory.created_at.isoformat(),
            "verification_status": memory.verification_status.value,
            "state": memory.state.value,
            "salience": memory.salience,
            "supersedes_memory_id": memory.supersedes_memory_id,
            "model_id": (
                memory.model_reference.model_id if memory.model_reference is not None else None
            ),
            "model_revision": (
                memory.model_reference.revision if memory.model_reference is not None else None
            ),
        }
    )


def _digest(value: dict[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
