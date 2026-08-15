"""Shared PostgreSQL reads for immutable media metadata."""

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.core import MediaKind, MediaObject, MediaObjectId, TenantId
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
