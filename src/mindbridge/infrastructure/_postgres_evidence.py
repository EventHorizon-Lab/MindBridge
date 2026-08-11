"""PostgreSQL reads and row mapping for exact evidence spans."""

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaObjectId,
    ObservationId,
    PixelRegion,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool

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


async def read_evidence(
    pool: DatabasePool,
    tenant_id: TenantId,
    evidence_ids: tuple[EvidenceId, ...],
) -> tuple[EvidenceSpan, ...]:
    """Read evidence spans in caller order without crossing tenants."""
    if not evidence_ids:
        return ()
    async with pool.connection() as connection:
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
        evidence_by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_by_id
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
