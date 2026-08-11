"""Atomic PostgreSQL commit for derived observation memory."""

from __future__ import annotations

from typing import cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application import ObservationProcessingOutput
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    ForgetTargetType,
    JobId,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
)
from mindbridge.infrastructure._postgres_derived_records import (
    derived_memory_content_digest,
    write_event,
)
from mindbridge.infrastructure._postgres_embeddings import write_embedding_on_connection
from mindbridge.infrastructure._postgres_forget import ensure_target_not_tombstoned
from mindbridge.infrastructure._postgres_graph import (
    write_claims,
    write_entities,
    write_entity_mentions,
    write_relations,
)
from mindbridge.infrastructure._postgres_jobs import (
    mark_observation_processing_succeeded_on_connection,
)
from mindbridge.infrastructure._postgres_memories import write_memory_on_connection
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)


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
        async with tenant_connection(pool, tenant_id) as connection:
            await ensure_target_not_tombstoned(
                connection,
                tenant_id,
                ForgetTargetType.OBSERVATION,
                observation_id,
            )
            await _require_source_evidence(connection, tenant_id, observation_id, output)
            for event in output.events:
                await write_event(connection, event)
            await write_entities(connection, output.entities)
            await write_claims(connection, output.claims)
            for memory in output.memories:
                await write_memory_on_connection(
                    connection,
                    memory,
                    derived_memory_content_digest(memory),
                )
            await write_entity_mentions(connection, output.entity_mentions)
            await write_relations(connection, output.relations)
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
    for event in output.events:
        if event.tenant_id != tenant_id:
            raise DomainInvariantError("derived records must remain in the source tenant")
        if event.observation_ids != (observation_id,):
            raise DomainInvariantError("derived event must reference its source observation")
    if any(
        item.tenant_id != tenant_id
        for items in (
            output.entities,
            output.entity_mentions,
            output.claims,
            output.memories,
            output.relations,
            output.embeddings,
        )
        for item in items
    ):
        raise DomainInvariantError("derived records must remain in the source tenant")
    valid_graph_embedding_objects = {
        *((EmbeddedObjectType.EVENT, str(event.event_id)) for event in output.events),
        *((EmbeddedObjectType.CLAIM, str(claim.claim_id)) for claim in output.claims),
    }
    if any(
        embedding.object_type is not EmbeddedObjectType.EVIDENCE_SPAN
        and (embedding.object_type, embedding.object_id) not in valid_graph_embedding_objects
        for embedding in output.embeddings
    ):
        raise DomainInvariantError("derived embedding references an unknown graph record")


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
    referenced_evidence_ids = (
        {str(evidence_id) for event in output.events for evidence_id in event.evidence_ids}
        | {str(evidence_id) for claim in output.claims for evidence_id in claim.evidence_ids}
        | {
            embedding.object_id
            for embedding in output.embeddings
            if embedding.object_type is EmbeddedObjectType.EVIDENCE_SPAN
        }
        | {str(mention.evidence_id) for mention in output.entity_mentions}
    )
    if not referenced_evidence_ids <= source_evidence_ids:
        raise DomainInvariantError("derived records reference evidence outside the observation")
