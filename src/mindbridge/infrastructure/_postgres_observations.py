"""PostgreSQL persistence for raw observations and evidence."""

from dataclasses import replace
from typing import cast

from psycopg.types.json import Jsonb

from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.application.ports import ObservationWriteResult
from mindbridge.core import (
    DomainInvariantError,
    ForgetTargetType,
    IdempotencyConflictError,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    Observation,
    ObservationId,
)
from mindbridge.infrastructure._postgres_evidence import write_evidence_spans
from mindbridge.infrastructure._postgres_forget import (
    ensure_target_not_tombstoned,
)
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_jobs import ensure_observation_processing_job
from mindbridge.infrastructure._postgres_media import write_media_object
from mindbridge.infrastructure._postgres_observation_reads import read_observation
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    PostgresStoreOperations,
    tenant_connection,
)


async def _insert_observation(
    connection: DatabaseConnection,
    observation: Observation,
    content_digest: str,
) -> bool:
    cursor = await connection.execute(
        """
        INSERT INTO observations (
            tenant_id, observation_id, device_id, boot_id, sequence, sensor,
            occurred_at, ended_at, observed_at, clock_offset_ms,
            identity_observations, content_digest
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING observation_id
        """,
        (
            observation.tenant_id,
            observation.observation_id,
            observation.device_id,
            observation.boot_id,
            observation.sequence,
            observation.sensor.value,
            observation.occurred_at,
            observation.ended_at,
            observation.observed_at,
            observation.clock_offset_ms,
            Jsonb(
                [
                    {
                        "identity_id": identity.identity_id,
                        "kind": identity.kind.value,
                        "start_ms": identity.start_ms,
                        "end_ms": identity.end_ms,
                        "confidence": identity.confidence,
                        "model_id": identity.model_reference.model_id,
                        "scope": identity.scope.value,
                        "transcript": identity.transcript,
                        "visual_bbox_xyxy": identity.visual_bbox_xyxy,
                    }
                    for identity in observation.identity_observations
                ]
            ),
            content_digest,
        ),
    )
    return await cursor.fetchone() is not None


async def _observation_digest(
    connection: DatabaseConnection,
    observation: Observation,
) -> str | None:
    """Return the stored digest, or None when it was written by a retired recipe.

    Migration 0025 nulls this column for observations carrying identity spans, because
    migration 0021 removed a field from `ObserveRequest` and `_request_digest` hashes the whole
    request: the digest of a request whose bytes did not change moved. None means "this cannot
    be compared", which is the one thing a stale digest could not say for itself.
    """
    cursor = await connection.execute(
        """
        SELECT content_digest FROM observations
        WHERE tenant_id = %s AND observation_id = %s
        """,
        (observation.tenant_id, observation.observation_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("observation disappeared during transaction")
    return cast(tuple[str | None], row)[0]


async def _adopt_observation_digest(
    connection: DatabaseConnection,
    observation: Observation,
    content_digest: str,
) -> None:
    """Record what this observation digests under the current recipe.

    Accepting the resend is only half of it. Writing the digest back restores the guard for
    every later retry of this same device sequence, so the window in which a genuinely
    different body would be accepted is one resend wide rather than permanent.
    """
    await connection.execute(
        """
        UPDATE observations SET content_digest = %s
        WHERE tenant_id = %s AND observation_id = %s AND content_digest IS NULL
        """,
        (content_digest, observation.tenant_id, observation.observation_id),
    )


async def _write_media_objects(
    connection: DatabaseConnection,
    media_objects: tuple[MediaObject, ...],
) -> dict[MediaObjectId, MediaObjectId]:
    canonical_ids: dict[MediaObjectId, MediaObjectId] = {}
    for media_object in media_objects:
        canonical_id = await write_media_object(connection, media_object)
        if canonical_id in canonical_ids.values():
            raise DomainInvariantError("observation media objects must have unique content")
        canonical_ids[media_object.media_object_id] = canonical_id
    return canonical_ids


async def _write_observation_media(
    connection: DatabaseConnection,
    observation: Observation,
    canonical_media_ids: dict[MediaObjectId, MediaObjectId],
) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO observation_media (tenant_id, observation_id, media_object_id, ordinal)
            VALUES (%s, %s, %s, %s)
            """,
            (
                (
                    observation.tenant_id,
                    observation.observation_id,
                    canonical_media_ids[media_object_id],
                    ordinal,
                )
                for ordinal, media_object_id in enumerate(observation.media_object_ids)
            ),
        )


class ObservationWriteOperations(PostgresStoreOperations):
    async def write_observation(
        self,
        batch: ObservationBatch,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ObservationWriteResult:
        """Write an observation atomically or return its idempotent predecessor."""
        observation = batch.observation
        async with tenant_connection(self._pool, observation.tenant_id) as connection:
            await ensure_target_not_tombstoned(
                connection,
                observation.tenant_id,
                ForgetTargetType.OBSERVATION,
                observation.observation_id,
            )
            existing_id = await claim_idempotency_key(
                connection,
                tenant_id=observation.tenant_id,
                operation="observe",
                idempotency_key=idempotency_key,
                content_digest=content_digest,
                resource_id=observation.observation_id,
            )
            if existing_id is not None:
                existing = await read_observation(
                    connection,
                    observation.tenant_id,
                    ObservationId(existing_id),
                )
                job_id = await ensure_observation_processing_job(connection, existing)
                return ObservationWriteResult(
                    observation=existing,
                    processing_job_id=job_id,
                    created=False,
                )

            created = await _insert_observation(connection, observation, content_digest)
            if not created:
                existing = await read_observation(
                    connection,
                    observation.tenant_id,
                    observation.observation_id,
                )
                stored_digest = await _observation_digest(connection, observation)
                if stored_digest is None:
                    await _adopt_observation_digest(connection, observation, content_digest)
                elif stored_digest != content_digest:
                    raise IdempotencyConflictError(
                        "device sequence already stores different observation content"
                    )
                job_id = await ensure_observation_processing_job(connection, existing)
                return ObservationWriteResult(
                    observation=existing,
                    processing_job_id=job_id,
                    created=False,
                )

            canonical_media_ids = await _write_media_objects(connection, batch.media_objects)
            await _write_observation_media(connection, observation, canonical_media_ids)
            await write_evidence_spans(
                connection,
                tuple(
                    replace(
                        evidence,
                        media_object_id=canonical_media_ids[evidence.media_object_id],
                    )
                    for evidence in batch.evidence_spans
                ),
            )
            job_id = await ensure_observation_processing_job(connection, observation)
            return ObservationWriteResult(
                observation=observation,
                processing_job_id=job_id,
                created=True,
            )
