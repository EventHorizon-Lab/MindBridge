"""Shared Python, REST, and MCP contracts for MindBridge use cases."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from mindbridge.core import (
    DeletionPropagationState,
    FeedbackType,
    ForgetTargetType,
    JobState,
    MediaKind,
    MemoryState,
    MemoryType,
    SensorKind,
    VerificationStatus,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


class ContractModel(BaseModel):
    """Strict immutable base shared by every external contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaObjectInput(ContractModel):
    """Metadata for an already addressable evidence object."""

    media_object_id: Identifier
    kind: MediaKind
    uri: NonEmptyString
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    created_at: AwareDatetime
    duration_ms: Annotated[int, Field(ge=0)] | None = None


class ObserveRequest(ContractModel):
    """Submit one timestamped device observation and its media metadata."""

    tenant_id: Identifier
    device_id: Identifier
    boot_id: Identifier
    sequence: Annotated[int, Field(ge=0)]
    sensor: SensorKind
    media_objects: Annotated[tuple[MediaObjectInput, ...], Field(min_length=1)]
    occurred_at: AwareDatetime
    ended_at: AwareDatetime
    observed_at: AwareDatetime
    clock_offset_ms: int = 0
    idempotency_key: Identifier | None = None


class ObservationStatus(str, Enum):
    """Ingestion outcome returned to a retrying edge device."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


class ObservationReceipt(ContractModel):
    """Retry-safe acknowledgement for an observation."""

    observation_id: Identifier
    processing_job_id: Identifier
    idempotency_key: Identifier
    status: ObservationStatus
    trace_id: Identifier


class ObservationProcessingJobView(ContractModel):
    """Public state of one durable observation processing job."""

    job_id: Identifier
    observation_id: Identifier
    state: JobState
    attempt: Annotated[int, Field(ge=0)]
    error_code: Identifier | None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    trace_id: Identifier


class RememberRequest(ContractModel):
    """Explicitly retain user- or agent-supplied content."""

    tenant_id: Identifier
    summary: NonEmptyString
    memory_type: MemoryType
    occurred_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    evidence_ids: tuple[Identifier, ...] = ()
    idempotency_key: Identifier | None = None


class FeedbackRequest(ContractModel):
    """Record useful, wrong, missing, or corrected recall feedback."""

    tenant_id: Identifier
    feedback_type: FeedbackType
    memory_id: Identifier | None = None
    recall_trace_id: Identifier | None = None
    correction_summary: NonEmptyString | None = None
    idempotency_key: Identifier | None = None

    @model_validator(mode="after")
    def require_feedback_target(self) -> FeedbackRequest:
        """Make each feedback kind actionable without an untyped details bag."""
        if self.feedback_type is FeedbackType.MISSING:
            if self.recall_trace_id is None:
                raise ValueError("missing feedback requires recall_trace_id")
            if self.memory_id is not None:
                raise ValueError("missing feedback must not provide memory_id")
        elif self.memory_id is None:
            raise ValueError("memory feedback requires memory_id")
        if self.feedback_type is FeedbackType.CORRECTION:
            if self.correction_summary is None:
                raise ValueError("correction feedback requires correction_summary")
        elif self.correction_summary is not None:
            raise ValueError("correction_summary is only valid for correction feedback")
        return self


class FeedbackReceipt(ContractModel):
    """Durable feedback result and resulting transparent lifecycle state."""

    feedback_id: Identifier
    feedback_type: FeedbackType
    memory_id: Identifier | None
    corrected_memory_id: Identifier | None
    resulting_state: MemoryState | None
    resulting_strength: float | None = Field(allow_inf_nan=False)
    created_at: AwareDatetime
    trace_id: Identifier

    @model_validator(mode="after")
    def require_result_pair(self) -> FeedbackReceipt:
        """Keep lifecycle state and its score from drifting apart."""
        if (self.resulting_state is None) != (self.resulting_strength is None):
            raise ValueError("resulting_state and resulting_strength must be provided together")
        return self


class ForgetRequest(ContractModel):
    """Explicitly delete one exact memory or its complete source observation."""

    tenant_id: Identifier
    target_type: ForgetTargetType
    target_id: Identifier
    idempotency_key: Identifier | None = None


class DeletionTombstoneView(ContractModel):
    """Content-free deletion state safe to retain after physical erasure."""

    tombstone_id: Identifier
    target_type: ForgetTargetType
    target_id: Identifier
    propagation_state: DeletionPropagationState
    requested_at: AwareDatetime
    completed_at: AwareDatetime | None
    error_code: Identifier | None


class ForgetReceipt(DeletionTombstoneView):
    """Deletion result returned by one command or status lookup."""

    trace_id: Identifier


class DeletionListRequest(ContractModel):
    """Tenant-scoped cursor request used by reconnecting edge devices."""

    tenant_id: Identifier
    cursor: Identifier | None = None
    limit: Annotated[int, Field(ge=1, le=100)] = 100


class DeletionPage(ContractModel):
    """Stable ordered tombstones and the next cursor when another page exists."""

    items: tuple[DeletionTombstoneView, ...]
    next_cursor: Identifier | None
    trace_id: Identifier


class RecallQuery(ContractModel):
    """Text, media, or both used to find relevant experience."""

    text: NonEmptyString | None = None
    media_object_ids: Annotated[tuple[Identifier, ...], Field(max_length=8)] = ()

    @model_validator(mode="after")
    def require_content(self) -> RecallQuery:
        """Require at least one modality without privileging text."""
        if self.text is None and not self.media_object_ids:
            raise ValueError("recall query requires text or media_object_ids")
        if len(set(self.media_object_ids)) != len(self.media_object_ids):
            raise ValueError("recall query media_object_ids must be unique")
        return self


class RecallFilters(ContractModel):
    """Structured constraints applied before semantic ranking."""

    person_ids: tuple[Identifier, ...] = ()
    device_ids: tuple[Identifier, ...] = ()
    memory_types: tuple[MemoryType, ...] = ()
    occurred_after: AwareDatetime | None = None
    occurred_before: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_ordered_time_range(self) -> RecallFilters:
        """Reject an impossible temporal query at the trust boundary."""
        if (
            self.occurred_after is not None
            and self.occurred_before is not None
            and self.occurred_before < self.occurred_after
        ):
            raise ValueError("occurred_before must not precede occurred_after")
        return self


class RecallMode(str, Enum):
    """Supported recall result shapes."""

    ANSWER = "answer"
    SEARCH = "search"
    ENUMERATE = "enumerate"


class RecallRequest(ContractModel):
    """Recall relevant memories and, by default, their evidence."""

    tenant_id: Identifier
    query: RecallQuery
    filters: RecallFilters = Field(default_factory=RecallFilters)
    mode: RecallMode = RecallMode.ANSWER
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    include_evidence: bool = True


class MemoryView(ContractModel):
    """Serializable stable view of a retained memory."""

    memory_id: Identifier
    memory_type: MemoryType
    summary: NonEmptyString
    evidence_ids: tuple[Identifier, ...]
    occurred_at: AwareDatetime
    ended_at: AwareDatetime
    created_at: AwareDatetime
    verification_status: VerificationStatus
    state: MemoryState
    salience: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    strength: float = Field(default=0.5, allow_inf_nan=False)
    useful_access_count: Annotated[int, Field(ge=0)] = 0
    positive_feedback_count: Annotated[int, Field(ge=0)] = 0
    negative_feedback_count: Annotated[int, Field(ge=0)] = 0
    last_accessed_at: AwareDatetime | None = None
    supersedes_memory_id: Identifier | None = None
    superseded_at: AwareDatetime | None = None


class EvidenceView(ContractModel):
    """Precise evidence location safe to expose to a caller."""

    evidence_id: Identifier
    media_object_id: Identifier
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=0)]
    media_url: NonEmptyString
    media_url_expires_at: AwareDatetime


class RecallResult(ContractModel):
    """Answer plus the memory and evidence needed to verify it."""

    answer: str | None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    memories: tuple[MemoryView, ...]
    evidence: tuple[EvidenceView, ...]
    trace_id: Identifier


class ValidationIssue(ContractModel):
    """One sanitized request validation failure."""

    location: tuple[str, ...]
    message: NonEmptyString
    code: Identifier


class ErrorResponse(ContractModel):
    """Stable error envelope with a reproducible trace identifier."""

    code: Identifier
    message: NonEmptyString
    trace_id: Identifier
    issues: tuple[ValidationIssue, ...] = ()


class HealthResponse(ContractModel):
    """Liveness response that does not claim dependency readiness."""

    status: str = "ok"
    trace_id: Identifier
