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
    read_media_objects_on_connection,
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

    async def read_evidence_clip_media(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> dict[EvidenceId, MediaObject]:
        """Map each evidence span to the derived clip already cut for a model to read.

        The write path stores one clip per span at the deployment's sampling and embeds
        that, not the source. Generation asked for the source instead, so a populated
        recall attached full-resolution originals: measured 12.3k prompt tokens per clip
        against 1.65k for the stored clip, and four originals exceed this endpoint's
        60 s gateway limit outright. Spans with no clip fall back to their source.

        A span is only substituted when one clip covers all of it. A span is not always one
        clip: `cut_clips` splits audio at `AUDIO_WINDOW_MS`, so a 70 s audio span is stored
        as three ordinals. Returning the lowest of those -- which is what an earlier
        `DISTINCT ON (evidence_id) ... ORDER BY ordinal` did -- hands the answer model the
        first 30 seconds while the recall that retrieved the span may well have matched a
        later window's embedding, and nothing about the result says a tail is missing.
        Falling back to the source is a bigger request but a complete one, and a
        `MediaObject` per evidence id cannot express more than one window anyway.
        """
        if not evidence_ids:
            return {}
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                """
                SELECT evidence_id, min(media_object_id)
                FROM evidence_clips
                WHERE tenant_id = %s AND evidence_id = ANY(%s)
                GROUP BY evidence_id
                -- The whole point: a span cut into several windows is skipped, not truncated.
                -- With exactly one row per group, min() is that row's clip.
                HAVING count(*) = 1
                """,
                (tenant_id, list(evidence_ids)),
            )
            rows = await cursor.fetchall()
            if not rows:
                return {}
            clip_id_by_evidence = {
                EvidenceId(cast(str, row[0])): MediaObjectId(cast(str, row[1])) for row in rows
            }
            media_objects = await read_media_objects_on_connection(
                connection,
                tenant_id,
                tuple(dict.fromkeys(clip_id_by_evidence.values())),
            )
        media_by_id = {item.media_object_id: item for item in media_objects}
        return {
            evidence_id: media_by_id[clip_id]
            for evidence_id, clip_id in clip_id_by_evidence.items()
            if clip_id in media_by_id
        }

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
