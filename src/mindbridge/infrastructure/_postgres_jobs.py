"""PostgreSQL persistence for observation processing jobs."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TypeAlias, cast

from psycopg.types.json import Jsonb

from mindbridge.core import (
    JobId,
    JobState,
    MemoryIntegrityError,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool

PROCESS_OBSERVATION_JOB_TYPE = "process_observation"
OBSERVATION_JOB_STALE_AFTER_SECONDS = 960
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
JobRow: TypeAlias = tuple[
    str,
    str,
    str,
    int,
    str | None,
    datetime,
    datetime,
    str,
]


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


async def claim_observation_processing_job(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
) -> ObservationJobClaim:
    """Atomically claim a ready or stale job without concurrent ownership."""
    _require_expected_job_id(observation_id, job_id)
    async with pool.connection() as connection:
        cursor = await connection.execute(
            """
            UPDATE jobs
            SET state = 'running', attempt = attempt + 1,
                error_code = NULL, updated_at = now()
            WHERE tenant_id = %s AND job_id = %s
              AND job_type = %s AND payload ->> 'observation_id' = %s
              AND (
                  state IN ('pending', 'failed')
                  OR (
                      state = 'running'
                      AND updated_at <= now() - make_interval(secs => %s)
                  )
              )
            RETURNING job_id, tenant_id, state, attempt, error_code,
                      created_at, updated_at, payload ->> 'observation_id'
            """,
            (
                tenant_id,
                job_id,
                PROCESS_OBSERVATION_JOB_TYPE,
                observation_id,
                OBSERVATION_JOB_STALE_AFTER_SECONDS,
            ),
        )
        row = await cursor.fetchone()
        if row is not None:
            return ObservationJobClaim(job=_job_from_row(cast(JobRow, row)), acquired=True)
        job = await _read_expected_job(connection, tenant_id, observation_id, job_id)
        return ObservationJobClaim(job=job, acquired=False)


async def mark_observation_processing_succeeded(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
) -> ObservationProcessingJob:
    """Complete a running job; duplicate completion remains a no-op."""
    return await _finish_observation_processing_job(
        pool,
        tenant_id,
        observation_id,
        job_id,
        attempt=attempt,
        state=JobState.SUCCEEDED,
        error_code=None,
    )


async def mark_observation_processing_failed(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
    error_code: str,
) -> ObservationProcessingJob:
    """Record a sanitized retryable failure without exception details."""
    if _ERROR_CODE.fullmatch(error_code) is None:
        raise ValueError("error_code must be a lowercase machine identifier")
    return await _finish_observation_processing_job(
        pool,
        tenant_id,
        observation_id,
        job_id,
        attempt=attempt,
        state=JobState.FAILED,
        error_code=error_code,
    )


async def _finish_observation_processing_job(
    pool: DatabasePool,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
    state: JobState,
    error_code: str | None,
) -> ObservationProcessingJob:
    _require_expected_job_id(observation_id, job_id)
    async with pool.connection() as connection:
        cursor = await connection.execute(
            """
            UPDATE jobs
            SET state = %s, error_code = %s, updated_at = now()
            WHERE tenant_id = %s AND job_id = %s
              AND state = 'running' AND attempt = %s
              AND job_type = %s AND payload ->> 'observation_id' = %s
            RETURNING job_id, tenant_id, state, attempt, error_code,
                      created_at, updated_at, payload ->> 'observation_id'
            """,
            (
                state.value,
                error_code,
                tenant_id,
                job_id,
                attempt,
                PROCESS_OBSERVATION_JOB_TYPE,
                observation_id,
            ),
        )
        row = await cursor.fetchone()
        if row is not None:
            return _job_from_row(cast(JobRow, row))
        existing = await _read_expected_job(connection, tenant_id, observation_id, job_id)
        if existing.state in {state, JobState.SUCCEEDED}:
            return existing
        raise MemoryIntegrityError("observation processing job is not running")


async def _read_expected_job(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
) -> ObservationProcessingJob:
    cursor = await connection.execute(
        """
        SELECT job_id, tenant_id, state, attempt, error_code,
               created_at, updated_at, payload ->> 'observation_id'
        FROM jobs
        WHERE tenant_id = %s AND job_id = %s AND job_type = %s
        """,
        (tenant_id, job_id, PROCESS_OBSERVATION_JOB_TYPE),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("observation processing job does not exist")
    job = _job_from_row(cast(JobRow, row))
    if job.observation_id != observation_id:
        raise MemoryIntegrityError("observation processing job payload conflicts with task")
    return job


def _job_from_row(row: JobRow) -> ObservationProcessingJob:
    job_id, tenant_id, state, attempt, error_code, created_at, updated_at, observation_id = row
    return ObservationProcessingJob(
        job_id=JobId(job_id),
        tenant_id=TenantId(tenant_id),
        observation_id=ObservationId(observation_id),
        state=JobState(state),
        attempt=attempt,
        error_code=error_code,
        created_at=created_at,
        updated_at=updated_at,
    )


def _require_expected_job_id(observation_id: ObservationId, job_id: JobId) -> None:
    if job_id != f"job_process_{observation_id}":
        raise MemoryIntegrityError("observation processing job ID conflicts with task")
