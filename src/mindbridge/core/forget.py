"""Explicit deletion tombstones that outlive forgotten content."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import TenantId, TombstoneId


class ForgetTargetType(str, Enum):
    """Exact deletion scopes supported by the first production path."""

    MEMORY_RECORD = "memory_record"
    OBSERVATION = "observation"


class DeletionPropagationState(str, Enum):
    """Recoverable progress across PostgreSQL, S3, and offline edge devices."""

    PENDING = "pending"
    PROPAGATING = "propagating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeletionTombstone:
    """Content-free audit record preventing a forgotten target from returning."""

    tombstone_id: TombstoneId
    tenant_id: TenantId
    target_type: ForgetTargetType
    target_id: str
    propagation_state: DeletionPropagationState
    requested_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.tombstone_id, "tombstone_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.target_id, "target_id")
        require_aware_datetime(self.requested_at, "requested_at")
        if self.completed_at is not None:
            require_aware_datetime(self.completed_at, "completed_at")
            if self.completed_at < self.requested_at:
                raise DomainInvariantError("deletion cannot complete before it was requested")
        if (self.propagation_state is DeletionPropagationState.COMPLETE) != (
            self.completed_at is not None
        ):
            raise DomainInvariantError("only complete tombstones require completed_at")
        if (self.propagation_state is DeletionPropagationState.FAILED) != (
            self.error_code is not None
        ):
            raise DomainInvariantError("only failed tombstones carry an error code")
