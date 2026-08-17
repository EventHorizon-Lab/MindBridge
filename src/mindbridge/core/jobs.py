"""Durable processing state exposed consistently by API and workers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import JobId, MemoryId, ObservationId, TenantId


class JobState(str, Enum):
    """Persisted observation processing states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_settled(self) -> bool:
        """Whether this processing attempt already reached an outcome.

        Deliberately not called terminal: `SUCCEEDED` is irreversible, but the stale-job sweep
        reclaims a `FAILED` job for a later attempt, so a settled job can still change state.
        """
        return self in {JobState.SUCCEEDED, JobState.FAILED}


@dataclass(frozen=True, slots=True)
class ObservationProcessingJob:
    """Typed view of one durable observation processing job."""

    job_id: JobId
    tenant_id: TenantId
    observation_id: ObservationId
    state: JobState
    attempt: int
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    memory_ids: tuple[MemoryId, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.job_id, "job_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.observation_id, "observation_id")
        require_aware_datetime(self.created_at, "created_at")
        require_aware_datetime(self.updated_at, "updated_at")
        if self.attempt < 0:
            raise DomainInvariantError("job attempt must not be negative")
        if (self.state is JobState.PENDING) != (self.attempt == 0):
            raise DomainInvariantError("only pending jobs must have zero attempts")
        if self.updated_at < self.created_at:
            raise DomainInvariantError("job updated_at must not precede created_at")
        if (self.state is JobState.FAILED) != (self.error_code is not None):
            raise DomainInvariantError("only failed jobs must carry an error code")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise DomainInvariantError("job memory IDs must be unique")
        if self.state is not JobState.SUCCEEDED and self.memory_ids:
            raise DomainInvariantError("only succeeded jobs may carry memory IDs")


@dataclass(frozen=True, slots=True)
class ObservationJobClaim:
    """Result of atomically attempting to own one processing attempt."""

    job: ObservationProcessingJob
    acquired: bool
