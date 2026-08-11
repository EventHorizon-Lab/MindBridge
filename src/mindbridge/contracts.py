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
    idempotency_key: Identifier
    status: ObservationStatus
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


class RecallQuery(ContractModel):
    """Text, media, or both used to find relevant experience."""

    text: NonEmptyString | None = None
    media_object_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def require_content(self) -> RecallQuery:
        """Require at least one modality without privileging text."""
        if self.text is None and not self.media_object_ids:
            raise ValueError("recall query requires text or media_object_ids")
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


class EvidenceView(ContractModel):
    """Precise evidence location safe to expose to a caller."""

    evidence_id: Identifier
    media_object_id: Identifier
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=0)]
    thumbnail_url: NonEmptyString | None = None


class RecallResult(ContractModel):
    """Answer plus the memory and evidence needed to verify it."""

    answer: str | None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    memories: tuple[MemoryView, ...]
    evidence: tuple[EvidenceView, ...]
    trace_id: Identifier
