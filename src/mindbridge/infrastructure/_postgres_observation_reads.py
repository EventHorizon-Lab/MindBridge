"""PostgreSQL reads and row mapping for immutable observations."""

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application import ObservationBatch
from mindbridge.core import (
    DeviceId,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.infrastructure._postgres_evidence import read_observation_evidence
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


async def read_observation_batch(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> ObservationBatch:
    """Read an immutable observation, media, and evidence from one snapshot."""
    async with pool.connection() as connection:
        observation = await read_observation(connection, tenant_id, observation_id)
        media_objects = await _read_media_objects(
            connection,
            tenant_id,
            observation.media_object_ids,
        )
        evidence_spans = await read_observation_evidence(
            connection,
            tenant_id,
            observation_id,
        )
    return ObservationBatch(
        media_objects=media_objects,
        observation=observation,
        evidence_spans=evidence_spans,
    )


async def read_media_objects(
    pool: DatabasePool,
    tenant_id: TenantId,
    media_object_ids: tuple[MediaObjectId, ...],
) -> tuple[MediaObject, ...]:
    """Read immutable media metadata in caller order without crossing tenants."""
    async with pool.connection() as connection:
        return await _read_media_objects(connection, tenant_id, media_object_ids)


async def _read_media_objects(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    media_object_ids: tuple[MediaObjectId, ...],
) -> tuple[MediaObject, ...]:
    if not media_object_ids:
        return ()
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


async def read_observation(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> Observation:
    """Read one observation and its canonical media order."""
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
        raise MemoryIntegrityError("observation does not exist")
    cursor = await connection.execute(
        """
        SELECT media_object_id FROM observation_media
        WHERE tenant_id = %s AND observation_id = %s
        ORDER BY ordinal
        """,
        (tenant_id, observation_id),
    )
    media_object_ids = tuple([MediaObjectId(cast(tuple[str], item)[0]) async for item in cursor])
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
