"""PostgreSQL reads and row mapping for exact evidence spans."""

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.core import (
    EvidenceClip,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    ObservationId,
    PixelRegion,
    TenantId,
)
from mindbridge.infrastructure._postgres_media import (
    link_observation_media,
    write_media_object,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    PostgresStoreOperations,
    tenant_connection,
)

EvidenceRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    int,
    int,
    datetime,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]

EVIDENCE_SELECT_SQL = """
SELECT evidence_id, tenant_id, observation_id, media_object_id,
       start_ms, end_ms, created_at, frame_start, frame_end,
       x_min, y_min, x_max, y_max, audio_track
FROM evidence_spans
"""


async def write_evidence_spans(
    connection: DatabaseConnection,
    evidence_spans: tuple[EvidenceSpan, ...],
) -> None:
    """Write already validated spans on the caller's transaction."""
    if not evidence_spans:
        return
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
                    evidence.media_object_id,
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


async def read_observation_evidence(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> tuple[EvidenceSpan, ...]:
    """Read one observation's evidence in stable temporal order."""
    cursor = await connection.execute(
        f"""{EVIDENCE_SELECT_SQL}
        WHERE tenant_id = %s AND observation_id = %s
        ORDER BY start_ms, end_ms, evidence_id
        """,
        (tenant_id, observation_id),
    )
    return tuple([evidence_from_row(cast(EvidenceRow, row)) async for row in cursor])


def evidence_from_row(row: EvidenceRow) -> EvidenceSpan:
    """Restore a validated evidence domain value from a database row."""
    (
        evidence_id,
        tenant_id,
        observation_id,
        media_object_id,
        start_ms,
        end_ms,
        created_at,
        frame_start,
        frame_end,
        x_min,
        y_min,
        x_max,
        y_max,
        audio_track,
    ) = row
    region = (
        PixelRegion(
            x_min=x_min,
            y_min=cast(int, y_min),
            x_max=cast(int, x_max),
            y_max=cast(int, y_max),
        )
        if x_min is not None
        else None
    )
    return EvidenceSpan(
        evidence_id=EvidenceId(evidence_id),
        tenant_id=TenantId(tenant_id),
        observation_id=ObservationId(observation_id),
        media_object_id=MediaObjectId(media_object_id),
        start_ms=start_ms,
        end_ms=end_ms,
        created_at=created_at,
        frame_start=frame_start,
        frame_end=frame_end,
        region=region,
        audio_track=audio_track,
    )


async def write_evidence_clips(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    media_objects: tuple[MediaObject, ...],
    clips: tuple[EvidenceClip, ...],
) -> None:
    """Persist derived clips and link them to the observation that owns them.

    The observation link is what makes the existing forget sweep reclaim clip
    objects: it already deletes media that no surviving observation references.
    """
    if not clips:
        return
    canonical_ids = {
        media_object.media_object_id: await write_media_object(connection, media_object)
        for media_object in media_objects
    }
    await link_observation_media(
        connection,
        tenant_id,
        observation_id,
        tuple(dict.fromkeys(canonical_ids.values())),
    )
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO evidence_clips (
                tenant_id, evidence_id, ordinal, media_object_id,
                start_ms, end_ms, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                (
                    clip.tenant_id,
                    clip.evidence_id,
                    clip.ordinal,
                    canonical_ids[clip.media_object_id],
                    clip.start_ms,
                    clip.end_ms,
                    clip.created_at,
                )
                for clip in clips
            ),
        )


class EvidenceReadOperations(PostgresStoreOperations):
    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        """Read evidence spans in caller order without crossing tenants."""
        if not evidence_ids:
            return ()
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                f"{EVIDENCE_SELECT_SQL} WHERE tenant_id = %s AND evidence_id = ANY(%s)",
                (tenant_id, list(evidence_ids)),
            )
            evidence_by_id = {
                evidence.evidence_id: evidence
                async for row in cursor
                for evidence in (evidence_from_row(cast(EvidenceRow, row)),)
            }
        return tuple(
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        )

    async def list_known_clip_digests(
        self,
        tenant_id: TenantId,
        digests: tuple[str, ...],
    ) -> frozenset[str]:
        """Return which content digests the database still accounts for.

        Clip bytes are uploaded before the transaction that registers them, so a
        rolled-back attempt leaves an object with no row at all. Object keys carry
        the digest, which makes reclaiming them a set difference against this.
        """
        if not digests:
            return frozenset()
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                """
                SELECT sha256 FROM media_objects
                WHERE tenant_id = %s AND sha256 = ANY(%s)
                """,
                (tenant_id, list(digests)),
            )
            rows = await cursor.fetchall()
        return frozenset(cast(str, row[0]) for row in rows)
