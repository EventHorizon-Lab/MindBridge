"""Domain checks for durable observation processing state."""

from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.core import (
    DomainInvariantError,
    JobId,
    JobState,
    MemoryId,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_only_failed_jobs_carry_sanitized_failure_state() -> None:
    """A success cannot accidentally retain a previous failure marker."""
    with pytest.raises(DomainInvariantError, match="only failed"):
        _job(state=JobState.SUCCEEDED, attempt=1, error_code="model_unavailable")


def test_job_timestamps_and_attempts_are_monotonic() -> None:
    """Impossible persisted lifecycle values are rejected at the core boundary."""
    with pytest.raises(DomainInvariantError, match="negative"):
        _job(attempt=-1)
    with pytest.raises(DomainInvariantError, match="precede"):
        _job(updated_at=NOW - timedelta(seconds=1))
    with pytest.raises(DomainInvariantError, match="zero attempts"):
        _job(state=JobState.RUNNING)


def test_only_succeeded_jobs_carry_unique_memory_results() -> None:
    with pytest.raises(DomainInvariantError, match="only succeeded"):
        _job(memory_ids=(MemoryId("memory_01"),))
    with pytest.raises(DomainInvariantError, match="unique"):
        _job(
            state=JobState.SUCCEEDED,
            attempt=1,
            memory_ids=(MemoryId("memory_01"), MemoryId("memory_01")),
        )


def _job(
    *,
    state: JobState = JobState.PENDING,
    attempt: int = 0,
    error_code: str | None = None,
    updated_at: datetime = NOW,
    memory_ids: tuple[MemoryId, ...] = (),
) -> ObservationProcessingJob:
    return ObservationProcessingJob(
        job_id=JobId("job_process_observation_01"),
        tenant_id=TenantId("tenant_01"),
        observation_id=ObservationId("observation_01"),
        state=state,
        attempt=attempt,
        error_code=error_code,
        created_at=NOW,
        updated_at=updated_at,
        memory_ids=memory_ids,
    )
