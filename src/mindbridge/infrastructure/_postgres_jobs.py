"""PostgreSQL persistence for observation processing jobs."""

from __future__ import annotations

from typing import cast

from psycopg.types.json import Jsonb

from mindbridge.core import JobId, MemoryIntegrityError, Observation
from mindbridge.infrastructure._postgres_types import DatabaseConnection

PROCESS_OBSERVATION_JOB_TYPE = "process_observation"


def observation_processing_job_id(observation: Observation) -> JobId:
    """Return the stable task identity exposed by observation receipts."""
    return JobId(f"job_process_{observation.observation_id}")


async def ensure_observation_processing_job(
    connection: DatabaseConnection,
    observation: Observation,
) -> JobId:
    """Create the durable job once, inside the observation transaction."""
    job_id = observation_processing_job_id(observation)
    payload = {"observation_id": observation.observation_id}
    cursor = await connection.execute(
        """
        INSERT INTO jobs (
            tenant_id, job_id, job_type, state, payload, created_at, updated_at
        )
        VALUES (%s, %s, %s, 'pending', %s, now(), now())
        ON CONFLICT DO NOTHING
        RETURNING job_id
        """,
        (observation.tenant_id, job_id, PROCESS_OBSERVATION_JOB_TYPE, Jsonb(payload)),
    )
    if await cursor.fetchone() is not None:
        return job_id

    cursor = await connection.execute(
        """
        SELECT job_type, payload FROM jobs
        WHERE tenant_id = %s AND job_id = %s
        """,
        (observation.tenant_id, job_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("observation processing job disappeared during transaction")
    job_type, stored_payload = cast(tuple[str, dict[str, str]], row)
    if job_type != PROCESS_OBSERVATION_JOB_TYPE or stored_payload != payload:
        raise MemoryIntegrityError("observation processing job has conflicting identity")
    return job_id
