"""PostgreSQL reads and row mapping for immutable observations."""

from datetime import datetime
from typing import TypeAlias, cast

from pydantic import TypeAdapter, ValidationError

from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    AnonymousIdentityObservation,
    DeviceId,
    ForgetTargetType,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    ModelReference,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.infrastructure._postgres_evidence import read_observation_evidence
from mindbridge.infrastructure._postgres_forget import ensure_target_not_tombstoned
from mindbridge.infrastructure._postgres_media import read_media_objects_on_connection
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

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
    object,
]

_IDENTITY_OBSERVATIONS = TypeAdapter(tuple[IdentityObservationInput, ...])


async def read_observation_batch(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> ObservationBatch:
    """Read an immutable observation, media, and evidence from one snapshot."""
    async with tenant_connection(pool, tenant_id) as connection:
        observation = await read_observation(connection, tenant_id, observation_id)
        media_objects = await read_media_objects_on_connection(
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
    async with tenant_connection(pool, tenant_id) as connection:
        return await read_media_objects_on_connection(connection, tenant_id, media_object_ids)


async def read_observation(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
) -> Observation:
    """Read one observation and its canonical media order."""
    await ensure_target_not_tombstoned(
        connection,
        tenant_id,
        ForgetTargetType.OBSERVATION,
        observation_id,
    )
    cursor = await connection.execute(
        """
        SELECT observation_id, tenant_id, device_id, boot_id, sequence, sensor,
               occurred_at, ended_at, observed_at, clock_offset_ms, identity_observations
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
        identity_observations,
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
        identity_observations=_identity_observations_from_json(identity_observations),
    )


def _identity_observations_from_json(value: object) -> tuple[AnonymousIdentityObservation, ...]:
    try:
        inputs = _IDENTITY_OBSERVATIONS.validate_python(value)
    except ValidationError as error:
        raise MemoryIntegrityError("stored identity observations are invalid") from error
    return tuple(
        AnonymousIdentityObservation(
            identity_id=item.identity_id,
            kind=item.kind,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            confidence=item.confidence,
            model_reference=ModelReference(
                model_id=item.model_id,
                revision=item.model_revision,
            ),
        )
        for item in inputs
    )
