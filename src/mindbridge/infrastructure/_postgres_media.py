"""Shared PostgreSQL reads and writes for immutable media metadata."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.core import (
    DomainInvariantError,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryDeletedError,
    MemoryIntegrityError,
    ObservationId,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import DatabaseConnection

MediaObjectRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    int,
    datetime,
    int | None,
    str | None,
]


async def read_media_objects_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    media_object_ids: tuple[MediaObjectId, ...],
) -> tuple[MediaObject, ...]:
    """Read immutable media metadata in caller order inside one transaction."""
    if not media_object_ids:
        return ()
    cursor = await connection.execute(
        """
        SELECT media_object_id, tenant_id, kind, uri, sha256,
               size_bytes, created_at, duration_ms, derived_from_media_object_id
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
        derived_from_media_object_id,
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
        derived_from_media_object_id=(
            None
            if derived_from_media_object_id is None
            else MediaObjectId(derived_from_media_object_id)
        ),
    )


async def ensure_media_not_scheduled_for_deletion(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    media_object_id: str,
) -> None:
    """Prevent content deduplication from attaching to a forgetting observation."""
    cursor = await connection.execute(
        """
        SELECT 1
        FROM observation_media AS link
        JOIN deletion_tombstones AS tombstone
          ON tombstone.tenant_id = link.tenant_id
         AND tombstone.target_type = 'observation'
         AND tombstone.target_id = link.observation_id
        WHERE link.tenant_id = %s AND link.media_object_id = %s
        LIMIT 1
        """,
        (tenant_id, media_object_id),
    )
    if await cursor.fetchone() is not None:
        raise MemoryDeletedError("media belongs to an explicitly forgotten observation")


async def write_media_object(
    connection: DatabaseConnection,
    media_object: MediaObject,
) -> MediaObjectId:
    cursor = await connection.execute(
        """
        INSERT INTO media_objects (
            tenant_id, media_object_id, kind, uri, sha256, size_bytes, duration_ms,
            created_at, derived_from_media_object_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            media_object.derived_from_media_object_id,
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
    existing_id, kind, sha256, size_bytes, duration_ms, derived_from = existing
    await ensure_media_not_scheduled_for_deletion(
        connection,
        media_object.tenant_id,
        existing_id,
    )
    if (
        kind != media_object.kind.value
        or sha256 != media_object.sha256
        or size_bytes != media_object.size_bytes
        or duration_ms != media_object.duration_ms
    ):
        raise DomainInvariantError("media identity conflicts with immutable metadata")
    incoming_source = media_object.derived_from_media_object_id
    if incoming_source is not None and derived_from is None:
        # Identical bytes previously stored without provenance: record the
        # derivation instead of silently losing the link. A different existing
        # source is left alone, because byte-identical clips can legitimately
        # come from more than one recording.
        await connection.execute(
            """
            UPDATE media_objects SET derived_from_media_object_id = %s
            WHERE tenant_id = %s AND media_object_id = %s
              AND derived_from_media_object_id IS NULL
            """,
            (incoming_source, media_object.tenant_id, existing_id),
        )
    return MediaObjectId(existing_id)


async def _find_media_object(
    connection: DatabaseConnection,
    media_object: MediaObject,
    *,
    by_content_hash: bool,
) -> tuple[str, str, str, int, int | None, str | None] | None:
    field_name = "sha256" if by_content_hash else "media_object_id"
    field_value = media_object.sha256 if by_content_hash else media_object.media_object_id
    cursor = await connection.execute(
        f"""
        SELECT media.media_object_id, media.kind, media.sha256,
               media.size_bytes, media.duration_ms, media.derived_from_media_object_id
        FROM media_objects AS media
        WHERE media.tenant_id = %s AND media.{field_name} = %s
        FOR KEY SHARE OF media
        """,
        (media_object.tenant_id, field_value),
    )
    row = await cursor.fetchone()
    return None if row is None else cast(tuple[str, str, str, int, int | None, str | None], row)


async def link_observation_media(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    media_object_ids: tuple[MediaObjectId, ...],
) -> None:
    """Attach media to an observation after the ordinals it already owns."""
    if not media_object_ids:
        return
    cursor = await connection.execute(
        """
        SELECT COALESCE(MAX(ordinal), -1) FROM observation_media
        WHERE tenant_id = %s AND observation_id = %s
        """,
        (tenant_id, observation_id),
    )
    row = await cursor.fetchone()
    next_ordinal = cast(tuple[int], row)[0] + 1 if row is not None else 0
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO observation_media (tenant_id, observation_id, media_object_id, ordinal)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                (tenant_id, observation_id, media_object_id, next_ordinal + offset)
                for offset, media_object_id in enumerate(media_object_ids)
            ),
        )
