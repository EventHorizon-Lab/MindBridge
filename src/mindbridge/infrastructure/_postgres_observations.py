"""PostgreSQL persistence for raw observations and evidence."""

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application import ObservationBatch, ObservationWriteResult
from mindbridge.core import (
    DeviceId,
    DomainInvariantError,
    EvidenceSpan,
    IdempotencyConflictError,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.infrastructure._postgres_idempotency import claim_idempotency_key
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool

ObservationRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    int,
    str,
    datetime,
    datetime,
    datetime,
    int,
]
MediaObjectRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    int,
    datetime,
    int | None,
]


async def write_observation(
    pool: DatabasePool,
    batch: ObservationBatch,
    *,
    idempotency_key: str,
    content_digest: str,
) -> ObservationWriteResult:
    """Write an observation atomically or return its idempotent predecessor."""
    observation = batch.observation
    async with pool.connection() as connection:
        existing_id = await claim_idempotency_key(
            connection,
            tenant_id=observation.tenant_id,
            operation="observe",
            idempotency_key=idempotency_key,
            content_digest=content_digest,
            resource_id=observation.observation_id,
        )
        if existing_id is not None:
            existing = await _read_observation(
                connection,
                observation.tenant_id,
                ObservationId(existing_id),
            )
            return ObservationWriteResult(observation=existing, created=False)

        created = await _insert_observation(connection, observation, content_digest)
        if not created:
            existing = await _read_observation(
                connection,
                observation.tenant_id,
                observation.observation_id,
            )
            if await _observation_digest(connection, observation) != content_digest:
                raise IdempotencyConflictError(
                    "device sequence already stores different observation content"
                )
            return ObservationWriteResult(observation=existing, created=False)

        canonical_media_ids = await _write_media_objects(connection, batch.media_objects)
        await _write_observation_media(connection, observation, canonical_media_ids)
        await _write_evidence_spans(connection, batch.evidence_spans, canonical_media_ids)
        return ObservationWriteResult(observation=observation, created=True)


async def read_media_objects(
    pool: DatabasePool,
    tenant_id: TenantId,
    media_object_ids: tuple[MediaObjectId, ...],
) -> tuple[MediaObject, ...]:
    """Read immutable media metadata in caller order without crossing tenants."""
    if not media_object_ids:
        return ()
    async with pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT media_object_id, tenant_id, kind, uri, sha256,
                   size_bytes, created_at, duration_ms
            FROM media_objects
            WHERE tenant_id = %s AND media_object_id = ANY(%s)
            """,
            (tenant_id, list(media_object_ids)),
        )
        media_by_id = {
            media_object.media_object_id: media_object
            async for row in cursor
            for media_object in (_media_object_from_row(cast(MediaObjectRow, row)),)
        }
    return tuple(
        media_by_id[media_object_id]
        for media_object_id in media_object_ids
        if media_object_id in media_by_id
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
            occurred_at, ended_at, observed_at, clock_offset_ms, content_digest
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            content_digest,
        ),
    )
    return await cursor.fetchone() is not None


async def _observation_digest(
    connection: DatabaseConnection,
    observation: Observation,
) -> str:
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
    return cast(tuple[str], row)[0]


async def _write_media_objects(
    connection: DatabaseConnection,
    media_objects: tuple[MediaObject, ...],
) -> dict[MediaObjectId, MediaObjectId]:
    canonical_ids: dict[MediaObjectId, MediaObjectId] = {}
    for media_object in media_objects:
        canonical_id = await _write_media_object(connection, media_object)
        if canonical_id in canonical_ids.values():
            raise DomainInvariantError("observation media objects must have unique content")
        canonical_ids[media_object.media_object_id] = canonical_id
    return canonical_ids


async def _write_media_object(
    connection: DatabaseConnection,
    media_object: MediaObject,
) -> MediaObjectId:
    cursor = await connection.execute(
        """
        INSERT INTO media_objects (
            tenant_id, media_object_id, kind, uri, sha256, size_bytes, duration_ms, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING media_object_id
        """,
        (
            media_object.tenant_id,
            media_object.media_object_id,
            media_object.kind.value,
            media_object.uri,
            media_object.sha256,
            media_object.size_bytes,
            media_object.duration_ms,
            media_object.created_at,
        ),
    )
    row = await cursor.fetchone()
    if row is not None:
        return MediaObjectId(cast(tuple[str], row)[0])
    existing = await _find_media_object(connection, media_object, by_content_hash=False)
    if existing is None:
        existing = await _find_media_object(connection, media_object, by_content_hash=True)
    if existing is None:
        raise MemoryIntegrityError("media object conflict could not be resolved")
    existing_id, kind, sha256, size_bytes, duration_ms = existing
    if (
        kind != media_object.kind.value
        or sha256 != media_object.sha256
        or size_bytes != media_object.size_bytes
        or duration_ms != media_object.duration_ms
    ):
        raise DomainInvariantError("media identity conflicts with immutable metadata")
    return MediaObjectId(existing_id)


async def _find_media_object(
    connection: DatabaseConnection,
    media_object: MediaObject,
    *,
    by_content_hash: bool,
) -> tuple[str, str, str, int, int | None] | None:
    field_name = "sha256" if by_content_hash else "media_object_id"
    field_value = media_object.sha256 if by_content_hash else media_object.media_object_id
    cursor = await connection.execute(
        f"""
        SELECT media_object_id, kind, sha256, size_bytes, duration_ms
        FROM media_objects
        WHERE tenant_id = %s AND {field_name} = %s
        """,
        (media_object.tenant_id, field_value),
    )
    row = await cursor.fetchone()
    return None if row is None else cast(tuple[str, str, str, int, int | None], row)


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


async def _write_evidence_spans(
    connection: DatabaseConnection,
    evidence_spans: tuple[EvidenceSpan, ...],
    canonical_media_ids: dict[MediaObjectId, MediaObjectId],
) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO evidence_spans (
                tenant_id, evidence_id, observation_id, media_object_id,
                start_ms, end_ms, frame_start, frame_end,
                x_min, y_min, x_max, y_max, audio_track, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                (
                    evidence.tenant_id,
                    evidence.evidence_id,
                    evidence.observation_id,
                    canonical_media_ids[evidence.media_object_id],
                    evidence.start_ms,
                    evidence.end_ms,
                    evidence.frame_start,
                    evidence.frame_end,
                    evidence.region.x_min if evidence.region is not None else None,
                    evidence.region.y_min if evidence.region is not None else None,
                    evidence.region.x_max if evidence.region is not None else None,
                    evidence.region.y_max if evidence.region is not None else None,
                    evidence.audio_track,
                    evidence.created_at,
                )
                for evidence in evidence_spans
            ),
        )


async def _read_observation(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> Observation:
    cursor = await connection.execute(
        """
        SELECT observation_id, tenant_id, device_id, boot_id, sequence, sensor,
               occurred_at, ended_at, observed_at, clock_offset_ms
        FROM observations
        WHERE tenant_id = %s AND observation_id = %s
        """,
        (tenant_id, observation_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("idempotency key references a missing observation")
    cursor = await connection.execute(
        """
        SELECT media_object_id FROM observation_media
        WHERE tenant_id = %s AND observation_id = %s
        ORDER BY ordinal
        """,
        (tenant_id, observation_id),
    )
    media_object_ids: tuple[MediaObjectId, ...] = tuple(
        [MediaObjectId(cast(tuple[str], item)[0]) async for item in cursor]
    )
    return _observation_from_row(cast(ObservationRow, row), media_object_ids)


def _observation_from_row(
    row: ObservationRow,
    media_object_ids: tuple[MediaObjectId, ...],
) -> Observation:
    (
        observation_id,
        tenant_id,
        device_id,
        boot_id,
        sequence,
        sensor,
        occurred_at,
        ended_at,
        observed_at,
        clock_offset_ms,
    ) = row
    return Observation(
        observation_id=ObservationId(observation_id),
        tenant_id=TenantId(tenant_id),
        device_id=DeviceId(device_id),
        boot_id=boot_id,
        sequence=sequence,
        sensor=SensorKind(sensor),
        media_object_ids=media_object_ids,
        occurred_at=occurred_at,
        ended_at=ended_at,
        observed_at=observed_at,
        clock_offset_ms=clock_offset_ms,
    )


def _media_object_from_row(row: MediaObjectRow) -> MediaObject:
    (
        media_object_id,
        tenant_id,
        kind,
        uri,
        sha256,
        size_bytes,
        created_at,
        duration_ms,
    ) = row
    return MediaObject(
        media_object_id=MediaObjectId(media_object_id),
        tenant_id=TenantId(tenant_id),
        kind=MediaKind(kind),
        uri=uri,
        sha256=sha256,
        size_bytes=size_bytes,
        created_at=created_at,
        duration_ms=duration_ms,
    )
