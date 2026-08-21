"""PostgreSQL persistence for observation processing jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TypeAlias, cast

from psycopg.types.json import Jsonb

from mindbridge.core import (
    JobId,
    JobNotFoundError,
    JobState,
    MemoryId,
    MemoryIntegrityError,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    PostgresStoreOperations,
    tenant_connection,
)
from mindbridge.telemetry import model_token_usage

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
    list[str],
]
_AccountingRow: TypeAlias = tuple[
    str,
    int,
    int,
    int,
    int,
    int,
    Decimal,
    Decimal,
    Decimal,
    Decimal,
]


@dataclass(frozen=True, slots=True)
class ObservationJobAccounting:
    """One tenant's share of the ledger: how many jobs, how long they waited, what they cost."""

    tenant_id: TenantId
    jobs: int
    pending: int
    running: int
    failed: int
    succeeded: int
    queue_wait_seconds: float
    work_seconds: float
    input_tokens: int
    output_tokens: int


async def tenant_scope_required(connection: DatabaseConnection) -> bool:
    """Report whether row-level security confines this role to one tenant at a time.

    `jobs` is FORCE ROW LEVEL SECURITY, so a cross-tenant scan by the runtime role returns no
    rows rather than failing. A repair tool that reads that as "nothing to repair" is worse than
    one that refuses, which is why both scans below are only offered to a role that can see
    every tenant, or to a caller that names the one tenant it means.
    """
    cursor = await connection.execute(
        """
        SELECT current_setting('is_superuser') = 'on'
               OR EXISTS (
                   SELECT 1 FROM pg_roles
                   WHERE rolname = current_user AND rolbypassrls
               )
        """
    )
    row = await cursor.fetchone()
    return not cast(tuple[bool], row)[0]


async def unreachable_observation_jobs(
    connection: DatabaseConnection,
    *,
    tenant_id: TenantId | None = None,
    include_failed: bool = False,
) -> tuple[tuple[TenantId, ObservationId, JobId], ...]:
    """List the jobs a worker would still accept a claim for, in the order they arrived.

    This is the claim predicate of `claim_observation_processing_job`, read instead of written:
    a row matching it is work the ledger still owes, whether or not the broker has a message
    left for it. A stale `running` row is included because that is what a lost worker leaves
    behind, and `failed` only on request because a deterministic failure republished on a timer
    is a paid-for loop.

    The join keeps a job whose observation has since been deleted out of the result: the worker
    would only fail it again.
    """
    cursor = await connection.execute(
        """
        SELECT job.tenant_id, job.payload ->> 'observation_id', job.job_id
        FROM jobs AS job
        JOIN observations AS observation
          ON observation.tenant_id = job.tenant_id
         AND observation.observation_id = job.payload ->> 'observation_id'
        WHERE job.job_type = %s
          AND (%s::text IS NULL OR job.tenant_id = %s)
          AND (
              job.state = 'pending'
              OR (%s AND job.state = 'failed')
              OR (
                  job.state = 'running'
                  AND job.updated_at <= now() - make_interval(secs => %s)
              )
          )
        ORDER BY job.created_at, job.job_id
        """,
        (
            PROCESS_OBSERVATION_JOB_TYPE,
            tenant_id,
            tenant_id,
            include_failed,
            OBSERVATION_JOB_STALE_AFTER_SECONDS,
        ),
    )
    return tuple(
        (TenantId(tenant), ObservationId(observation), JobId(job))
        for tenant, observation, job in cast(list[tuple[str, str, str]], await cursor.fetchall())
    )


async def observation_job_accounting(
    connection: DatabaseConnection,
    *,
    tenant_id: TenantId | None = None,
) -> tuple[ObservationJobAccounting, ...]:
    """Summarize the ledger per tenant, busiest first, from the columns migration 0022 added."""
    cursor = await connection.execute(
        """
        SELECT tenant_id,
               count(*),
               count(*) FILTER (WHERE state = 'pending'),
               count(*) FILTER (WHERE state = 'running'),
               count(*) FILTER (WHERE state = 'failed'),
               count(*) FILTER (WHERE state = 'succeeded'),
               COALESCE(SUM(EXTRACT(epoch FROM started_at - created_at)), 0),
               COALESCE(SUM(EXTRACT(epoch FROM updated_at - started_at)), 0),
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0)
        FROM jobs
        WHERE job_type = %s AND (%s::text IS NULL OR tenant_id = %s)
        GROUP BY tenant_id
        -- 8 is the work time, which is the answer to "who is consuming the worker".
        ORDER BY 8 DESC, tenant_id
        """,
        (PROCESS_OBSERVATION_JOB_TYPE, tenant_id, tenant_id),
    )
    return tuple(
        ObservationJobAccounting(
            tenant_id=TenantId(row[0]),
            jobs=row[1],
            pending=row[2],
            running=row[3],
            failed=row[4],
            succeeded=row[5],
            queue_wait_seconds=float(row[6]),
            work_seconds=float(row[7]),
            input_tokens=int(row[8]),
            output_tokens=int(row[9]),
        )
        for row in cast(list[_AccountingRow], await cursor.fetchall())
    )


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
    job_type, stored_payload = cast(tuple[str, dict[str, object]], row)
    if (
        job_type != PROCESS_OBSERVATION_JOB_TYPE
        or stored_payload.get("observation_id") != observation.observation_id
    ):
        raise MemoryIntegrityError("observation processing job has conflicting identity")
    return job_id


async def mark_observation_processing_succeeded_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
    memory_ids: tuple[MemoryId, ...],
) -> ObservationProcessingJob:
    """Complete a job inside the caller's derived-record transaction."""
    completed = await _finish_observation_processing_job_on_connection(
        connection,
        tenant_id,
        observation_id,
        job_id,
        attempt=attempt,
        state=JobState.SUCCEEDED,
        error_code=None,
        memory_ids=memory_ids,
    )
    if completed.attempt != attempt:
        raise MemoryIntegrityError("observation processing attempt was superseded")
    return completed


async def lock_observation_processing_attempt(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
) -> None:
    """Lock the active attempt before any derived record can be written."""
    _require_expected_job_id(observation_id, job_id)
    cursor = await connection.execute(
        """
        SELECT job_id, tenant_id, state, attempt, error_code,
               created_at, updated_at, payload ->> 'observation_id',
               COALESCE(payload -> 'memory_ids', '[]'::jsonb)
        FROM jobs
        WHERE tenant_id = %s AND job_id = %s
        FOR UPDATE
        """,
        (tenant_id, job_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("observation processing job does not exist")
    job = _job_from_row(cast(JobRow, row))
    if job.observation_id != observation_id:
        raise MemoryIntegrityError("observation processing job payload conflicts with task")
    if job.attempt != attempt:
        raise MemoryIntegrityError("observation processing attempt was superseded")
    if job.state is not JobState.RUNNING:
        raise MemoryIntegrityError("observation processing job is not running")


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
    async with tenant_connection(pool, tenant_id) as connection:
        return await _finish_observation_processing_job_on_connection(
            connection,
            tenant_id,
            observation_id,
            job_id,
            attempt=attempt,
            state=state,
            error_code=error_code,
        )


async def _finish_observation_processing_job_on_connection(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    *,
    attempt: int,
    state: JobState,
    error_code: str | None,
    memory_ids: tuple[MemoryId, ...] | None = None,
) -> ObservationProcessingJob:
    _require_expected_job_id(observation_id, job_id)
    # Whatever the models charged while this attempt ran. The account belongs to the operation
    # that spent it -- `mindbridge.process_observation` here -- and this is the last moment it
    # is still open, so it is read rather than passed down through every pipeline in between.
    # Success and failure both land here, which is the point: a failed attempt was paid for.
    charged = model_token_usage()
    cursor = await connection.execute(
        """
        UPDATE jobs
        SET state = %s, error_code = %s,
            payload = COALESCE(
                jsonb_set(payload, '{memory_ids}', %s::jsonb),
                payload
            ),
            input_tokens = COALESCE(input_tokens, 0) + %s,
            output_tokens = COALESCE(output_tokens, 0) + %s,
            updated_at = now()
        WHERE tenant_id = %s AND job_id = %s
          AND state = 'running' AND attempt = %s
          AND job_type = %s AND payload ->> 'observation_id' = %s
        RETURNING job_id, tenant_id, state, attempt, error_code,
                  created_at, updated_at, payload ->> 'observation_id',
                  COALESCE(payload -> 'memory_ids', '[]'::jsonb)
        """,
        (
            state.value,
            error_code,
            Jsonb(list(memory_ids)) if memory_ids is not None else None,
            charged.get("input", 0),
            charged.get("output", 0),
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
    job = await _find_observation_processing_job(connection, tenant_id, job_id)
    if job is None:
        raise MemoryIntegrityError("observation processing job does not exist")
    if job.observation_id != observation_id:
        raise MemoryIntegrityError("observation processing job payload conflicts with task")
    return job


async def _find_observation_processing_job(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    job_id: JobId,
) -> ObservationProcessingJob | None:
    cursor = await connection.execute(
        """
        SELECT job_id, tenant_id, state, attempt, error_code,
               created_at, updated_at, payload ->> 'observation_id',
               COALESCE(payload -> 'memory_ids', '[]'::jsonb)
        FROM jobs
        WHERE tenant_id = %s AND job_id = %s AND job_type = %s
        """,
        (tenant_id, job_id, PROCESS_OBSERVATION_JOB_TYPE),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _job_from_row(cast(JobRow, row))


def _job_from_row(row: JobRow) -> ObservationProcessingJob:
    (
        job_id,
        tenant_id,
        state,
        attempt,
        error_code,
        created_at,
        updated_at,
        observation_id,
        memory_ids,
    ) = row
    return ObservationProcessingJob(
        job_id=JobId(job_id),
        tenant_id=TenantId(tenant_id),
        observation_id=ObservationId(observation_id),
        state=JobState(state),
        attempt=attempt,
        error_code=error_code,
        created_at=created_at,
        updated_at=updated_at,
        memory_ids=tuple(MemoryId(memory_id) for memory_id in memory_ids),
    )


def _require_expected_job_id(observation_id: ObservationId, job_id: JobId) -> None:
    if job_id != f"job_process_{observation_id}":
        raise MemoryIntegrityError("observation processing job ID conflicts with task")


class ObservationJobOperations(PostgresStoreOperations):
    async def claim_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationJobClaim:
        """Atomically claim a ready or stale job without concurrent ownership.

        The claim is also what stamps `started_at`, so the row always separates the wait from
        the attempt it is currently reporting. A re-claim moves it, because the previous
        attempt's start describes an attempt the row no longer reports.
        """
        _require_expected_job_id(observation_id, job_id)
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                """
                UPDATE jobs
                SET state = 'running', attempt = attempt + 1,
                    error_code = NULL, started_at = now(), updated_at = now()
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
                          created_at, updated_at, payload ->> 'observation_id',
                          COALESCE(payload -> 'memory_ids', '[]'::jsonb)
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

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        """Read one tenant-owned processing job without exposing its payload."""
        async with tenant_connection(self._pool, tenant_id) as connection:
            job = await _find_observation_processing_job(connection, tenant_id, job_id)
        if job is None:
            raise JobNotFoundError("observation processing job does not exist")
        return job

    async def mark_observation_processing_failed(
        self,
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
            self._pool,
            tenant_id,
            observation_id,
            job_id,
            attempt=attempt,
            state=JobState.FAILED,
            error_code=error_code,
        )
