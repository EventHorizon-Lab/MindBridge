"""Typed memory feedback that evolves records without changing model weights."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import FeedbackId, MemoryId, TenantId


class FeedbackType(str, Enum):
    """Supported user or Agent signals about a recall result."""

    USEFUL = "useful"
    WRONG = "wrong"
    MISSING = "missing"
    CORRECTION = "correction"


@dataclass(frozen=True, slots=True)
class MemoryFeedback:
    """One immutable feedback event and optional corrected statement."""

    feedback_id: FeedbackId
    tenant_id: TenantId
    feedback_type: FeedbackType
    created_at: datetime
    memory_id: MemoryId | None = None
    recall_trace_id: str | None = None
    correction_summary: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.feedback_id, "feedback_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_aware_datetime(self.created_at, "created_at")
        if self.recall_trace_id is not None:
            require_non_empty(self.recall_trace_id, "recall_trace_id")
        if self.feedback_type is FeedbackType.MISSING:
            if self.recall_trace_id is None:
                raise DomainInvariantError("missing feedback must reference a recall trace")
            if self.memory_id is not None:
                raise DomainInvariantError("missing feedback must not target a memory")
        elif self.memory_id is None:
            raise DomainInvariantError("memory feedback must reference a memory")
        if self.feedback_type is FeedbackType.CORRECTION:
            if self.correction_summary is None:
                raise DomainInvariantError("correction feedback must provide a summary")
            require_non_empty(self.correction_summary, "correction_summary")
        elif self.correction_summary is not None:
            raise DomainInvariantError("only correction feedback may provide a summary")
