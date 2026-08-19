"""Shared Python, REST, and MCP contracts for MindBridge use cases.

Every field carries a `description`. These are the only documentation an Agent reading the
generated JSON Schema receives, and the cross-field rules the `model_validator`s enforce are
invisible there, so each field that participates in one says so: an Agent that has to
discover a rule by failing a call is an Agent that fails the call.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
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
    IdentityKind,
    IdentityScope,
    JobState,
    MediaKind,
    MemoryState,
    MemoryType,
    SensorKind,
    VerificationStatus,
    media_kind_for_suffix,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(to_lower=True, pattern=r"^[0-9a-fA-F]{64}$"),
]
_MAXIMUM_IDENTITY_TRANSCRIPT_CHARACTERS = 65_536


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_as_utc)]


class ContractModel(BaseModel):
    """Strict immutable base shared by every external contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaObjectInput(ContractModel):
    """Metadata for an already addressable evidence object."""

    media_object_id: Identifier = Field(
        description="Caller-assigned ID for this object, unique within the observation.",
    )
    kind: MediaKind = Field(
        description="Modality routing trusts; must not contradict the URI's extension.",
    )
    uri: NonEmptyString = Field(
        description=(
            "Where the bytes already live, as `s3://<bucket>/tenants/<tenant_id>/<key>`. "
            "MindBridge reads this object; it does not accept an upload."
        ),
    )
    sha256: Sha256Hex = Field(
        description="Hex digest of the exact bytes at `uri`, used to detect a mismatch.",
    )
    size_bytes: Annotated[
        int,
        Field(
            ge=0,
            le=9_223_372_036_854_775_807,
            description="Byte length of the object at `uri`.",
        ),
    ]
    created_at: UtcDatetime = Field(
        description="When this media was captured, not when it was uploaded.",
    )
    # The description belongs on the outer Field: inside the Annotated branch of an optional
    # it lands under `anyOf` instead of on the property, where no schema reader looks for it.
    duration_ms: Annotated[int, Field(ge=0, le=9_223_372_036_854_775_807)] | None = Field(
        default=None,
        description=(
            "Playable length for time-based media; must not exceed the observation's own "
            "`occurred_at`-to-`ended_at` span. Omit for a still image."
        ),
    )

    @model_validator(mode="after")
    def require_kind_matching_uri(self) -> MediaObjectInput:
        """Reject a declared kind the URI contradicts.

        Routing trusts this kind and the server cannot sniff an object it has not fetched, so a
        wrong declaration otherwise surfaces as a decode failure in a worker. Unrecognized
        suffixes are left alone: extensionless object keys are normal.
        """
        implied = media_kind_for_suffix(PurePosixPath(urlsplit(self.uri).path).suffix)
        if implied is not None and implied is not self.kind:
            raise ValueError(
                f"media kind {self.kind.value} contradicts its {implied.value} URI extension"
            )
        return self


class IdentityObservationInput(ContractModel):
    """Anonymous edge identity metadata with no face or voice embedding."""

    identity_id: Identifier = Field(
        description="Device-local identity this span belongs to; never a face or voice template.",
    )
    kind: IdentityKind = Field(
        description="Signal that produced the span; gates `transcript` and `visual_bbox_xyxy`.",
    )
    start_ms: Annotated[
        int,
        Field(ge=0, description="Span start, in milliseconds from the observation's start."),
    ]
    end_ms: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Span end, in milliseconds from the observation's start. Must not precede "
                "`start_ms`, and must fall inside the observation's own span."
            ),
        ),
    ]
    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            allow_inf_nan=False,
            description="The edge detector's own confidence in this span.",
        ),
    ]
    model_id: Identifier = Field(
        description="Edge model that produced the span, recorded for provenance.",
    )
    model_revision: Identifier = Field(
        description="Exact revision of `model_id`, so a re-identification can be reproduced.",
    )
    scope: IdentityScope = Field(
        default=IdentityScope.DEVICE,
        description="How far `identity_id` is meaningful; device-local unless promoted.",
    )
    transcript: NonEmptyString | None = Field(
        default=None,
        description=(
            "What was said in this span. Valid only when `kind` is `voice`; all transcripts in "
            "one observation together may not exceed 65,536 characters."
        ),
    )
    visual_bbox_xyxy: (
        tuple[
            Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)],
            Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)],
            Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)],
            Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)],
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Where the face sits, as 0..1 normalized `(left, top, right, bottom)` — not pixels. "
            "Valid only when `kind` is `face`, and must have positive width and height."
        ),
    )

    @model_validator(mode="after")
    def require_ordered_time_range(self) -> IdentityObservationInput:
        if self.end_ms < self.start_ms:
            raise ValueError("identity end_ms must not precede start_ms")
        if self.visual_bbox_xyxy is not None:
            if self.kind is not IdentityKind.FACE:
                raise ValueError("visual_bbox_xyxy is only valid for face identities")
            left, top, right, bottom = self.visual_bbox_xyxy
            if right <= left or bottom <= top:
                raise ValueError("visual_bbox_xyxy must have positive width and height")
        if self.transcript is not None and self.kind is not IdentityKind.VOICE:
            raise ValueError("transcript is only valid for voice identities")
        return self


class ObserveRequest(ContractModel):
    """Submit one timestamped device observation and its media metadata."""

    tenant_id: Identifier = Field(
        description="Tenant that owns the resulting memory; must be one this key authorizes.",
    )
    device_id: Identifier = Field(description="Device that captured this observation.")
    boot_id: Identifier = Field(
        description="Changes on every device restart, so `sequence` need not survive one.",
    )
    sequence: Annotated[
        int,
        Field(
            ge=0,
            le=9_223_372_036_854_775_807,
            description="Monotonic counter within one `boot_id`, used to order and deduplicate.",
        ),
    ]
    sensor: SensorKind = Field(description="Which sensor produced this observation.")
    media_objects: Annotated[
        tuple[MediaObjectInput, ...],
        Field(
            min_length=1,
            max_length=8,
            description=(
                "The already-stored evidence this observation refers to, at most 8. "
                "`media_object_id` must not repeat."
            ),
        ),
    ]
    occurred_at: UtcDatetime = Field(description="When the observed events began.")
    ended_at: UtcDatetime = Field(
        description="When the observed events ended; must not precede `occurred_at`.",
    )
    observed_at: UtcDatetime = Field(
        description="When the device recorded them, which may trail `ended_at`.",
    )
    clock_offset_ms: Annotated[
        int,
        Field(
            ge=-2_147_483_648,
            le=2_147_483_647,
            description="Known device clock skew, so a drifting edge clock stays reconcilable.",
        ),
    ] = 0
    identity_observations: Annotated[
        tuple[IdentityObservationInput, ...],
        Field(
            max_length=512,
            description=(
                "Anonymous face and voice spans found on the edge, at most 512. The same "
                "identity, kind, span, and model must not appear twice."
            ),
        ),
    ] = ()
    idempotency_key: Identifier | None = Field(
        default=None,
        description=(
            "Makes a retry safe. Omit it and one is derived from the content, so an identical "
            "resend answers `duplicate` instead of storing a second copy."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_observation(self) -> ObserveRequest:
        if self.ended_at < self.occurred_at:
            raise ValueError("ended_at must not precede occurred_at")
        media_object_ids = [media.media_object_id for media in self.media_objects]
        if len(set(media_object_ids)) != len(media_object_ids):
            raise ValueError("media_objects must not contain duplicate IDs")
        duration_ms = round((self.ended_at - self.occurred_at).total_seconds() * 1_000)
        if any(
            media.duration_ms is not None and media.duration_ms > duration_ms
            for media in self.media_objects
        ):
            raise ValueError("media duration exceeds source observation")
        if any(identity.end_ms > duration_ms for identity in self.identity_observations):
            raise ValueError("identity observation exceeds source duration")
        if (
            sum(len(identity.transcript or "") for identity in self.identity_observations)
            > _MAXIMUM_IDENTITY_TRANSCRIPT_CHARACTERS
        ):
            raise ValueError("identity transcripts exceed the per-observation character limit")
        keys = [
            (
                identity.kind,
                identity.identity_id,
                identity.start_ms,
                identity.end_ms,
                identity.model_id,
                identity.model_revision,
            )
            for identity in self.identity_observations
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("identity observations must not contain duplicates")
        return self


class ObservationStatus(str, Enum):
    """Ingestion outcome returned to a retrying edge device."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


class ObservationReceipt(ContractModel):
    """Retry-safe acknowledgement for an observation."""

    observation_id: Identifier = Field(description="Stable ID of the stored observation.")
    processing_job_id: Identifier = Field(
        description=(
            "The durable job deriving memory from this observation. Read it until it reaches "
            "`succeeded`; memory does not exist yet when this receipt returns."
        ),
    )
    evidence_ids: tuple[Identifier, ...] = Field(
        default=(),
        description="Evidence spans registered synchronously, before any derivation ran.",
    )
    idempotency_key: Identifier = Field(
        description="The key this write was recorded under, supplied or derived.",
    )
    status: ObservationStatus = Field(
        description="`accepted` for a first write, `duplicate` when a retry matched an earlier one.",
    )
    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")


class ObservationProcessingJobView(ContractModel):
    """Public state of one durable observation processing job."""

    job_id: Identifier = Field(description="The job this state belongs to.")
    observation_id: Identifier = Field(description="Observation this job derives memory from.")
    state: JobState = Field(
        description=(
            "`pending`, `running`, `succeeded`, or `failed`. A `failed` attempt can still be "
            "retried by the stale-job sweep, so it settles the attempt, not the job."
        ),
    )
    attempt: Annotated[
        int,
        Field(ge=0, description="How many times this job has been claimed."),
    ]
    error_code: Identifier | None = Field(
        description="Why the last attempt failed, or null when none has.",
    )
    memory_ids: tuple[Identifier, ...] = Field(
        default=(),
        description=(
            "Memories this job derived, unique and present only once `state` is `succeeded`. "
            "Read them directly instead of searching for what was just written."
        ),
    )
    created_at: UtcDatetime = Field(description="When the job was enqueued.")
    updated_at: UtcDatetime = Field(
        description="When this state was written; the only value on the row that always rises.",
    )
    trace_id: Identifier = Field(description="Correlates this read with its telemetry.")

    @model_validator(mode="after")
    def require_consistent_memory_ids(self) -> ObservationProcessingJobView:
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("job memory_ids must be unique")
        if self.state is not JobState.SUCCEEDED and self.memory_ids:
            raise ValueError("only succeeded jobs may carry memory_ids")
        return self


class GetObservationJobRequest(ContractModel):
    """Identify one tenant-owned processing job without relying on ambient tenancy."""

    tenant_id: Identifier = Field(description="Tenant that owns the job; must match the key.")
    job_id: Identifier = Field(
        description="The `processing_job_id` an observation receipt returned.",
    )


class RememberRequest(ContractModel):
    """Explicitly retain user- or agent-supplied content."""

    tenant_id: Identifier = Field(
        description="Tenant that will own the memory; must be one this key authorizes.",
    )
    summary: NonEmptyString = Field(
        description="The content to retain, written so it stays useful out of context.",
    )
    memory_type: MemoryType = Field(
        description=(
            "The role this content will serve: `episodic` for something that happened at a "
            "time, `semantic` for a durable fact, `procedural` for how to do something, "
            "`prospective` for a future intention, `working` for short-lived task state, "
            "`perceptual` for a raw sensory detail."
        ),
    )
    occurred_at: UtcDatetime = Field(
        description="When the content is about, which is not necessarily now.",
    )
    ended_at: UtcDatetime | None = Field(
        default=None,
        description="End of the span; defaults to `occurred_at` and must not precede it.",
    )
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(
            max_length=100,
            description=(
                "Existing evidence spans grounding this memory, at most 100 and without "
                "repeats. Leave empty for content with no stored media behind it."
            ),
        ),
    ] = ()
    idempotency_key: Identifier | None = Field(
        default=None,
        description=(
            "Makes a retry safe. Omit it and one is derived from the content, so an identical "
            "resend returns `duplicate` with the same memory rather than a second copy."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_memory(self) -> RememberRequest:
        if self.ended_at is not None and self.ended_at < self.occurred_at:
            raise ValueError("ended_at must not precede occurred_at")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
        return self


class FeedbackRequest(ContractModel):
    """Record useful, wrong, missing, or corrected recall feedback."""

    tenant_id: Identifier = Field(
        description="Tenant that owns the memory or recall being judged.",
    )
    feedback_type: FeedbackType = Field(
        description=(
            "Which signal this is, and therefore which other fields are required. `useful`, "
            "`wrong`, and `correction` each judge one memory and need `memory_id`; `missing` "
            "reports that a recall found nothing usable and needs `recall_trace_id` instead."
        ),
    )
    memory_id: Identifier | None = Field(
        default=None,
        description=(
            "The memory being judged. Required for `useful`, `wrong`, and `correction`; must "
            "be omitted for `missing`, which is about a recall rather than a memory."
        ),
    )
    recall_trace_id: Identifier | None = Field(
        default=None,
        description=(
            "The `trace_id` of the recall this judges. Required for `missing`, which is how a "
            "retrieval failure is tied back to what was asked."
        ),
    )
    correction_summary: NonEmptyString | None = Field(
        default=None,
        description=(
            "What the memory should have said. Required for `correction` and rejected for "
            "every other type; it supersedes the original with a new version."
        ),
    )
    idempotency_key: Identifier | None = Field(
        default=None,
        description="Makes a retry safe; derived from the content when omitted.",
    )

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

    feedback_id: Identifier = Field(description="Stable ID of the recorded signal.")
    feedback_type: FeedbackType = Field(description="The signal that was recorded.")
    memory_id: Identifier | None = Field(description="Memory that was judged, when one was.")
    corrected_memory_id: Identifier | None = Field(
        description="The new version a `correction` created, which supersedes `memory_id`.",
    )
    resulting_state: MemoryState | None = Field(
        description="Lifecycle state the judged memory now holds; null when none was judged.",
    )
    resulting_strength: float | None = Field(
        allow_inf_nan=False,
        description="Retention score the memory now holds, always paired with `resulting_state`.",
    )
    created_at: UtcDatetime = Field(description="When the signal was recorded.")
    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")

    @model_validator(mode="after")
    def require_result_pair(self) -> FeedbackReceipt:
        """Keep lifecycle state and its score from drifting apart."""
        if (self.resulting_state is None) != (self.resulting_strength is None):
            raise ValueError("resulting_state and resulting_strength must be provided together")
        return self


class ForgetRequest(ContractModel):
    """Explicitly delete one exact memory or its complete source observation."""

    tenant_id: Identifier = Field(description="Tenant that owns the target.")
    target_type: ForgetTargetType = Field(
        description=(
            "What `target_id` names: `memory_record` erases that one memory, `observation` "
            "erases a source observation and everything derived from it."
        ),
    )
    target_id: Identifier = Field(description="Exact memory or observation ID to erase.")
    idempotency_key: Identifier | None = Field(
        default=None,
        description="Makes a retry safe; forget is idempotent either way.",
    )


class DeletionTombstoneView(ContractModel):
    """Content-free deletion state safe to retain after physical erasure."""

    tombstone_id: Identifier = Field(
        description="Stable ID of this deletion barrier, usable as a cursor.",
    )
    target_type: ForgetTargetType = Field(
        description=(
            "What `target_id` named: `memory_record` erased that one memory, `observation` "
            "erased a source observation and everything derived from it."
        ),
    )
    target_id: Identifier = Field(description="Which memory or observation was erased.")
    propagation_state: DeletionPropagationState = Field(
        description=(
            "How far erasure has reached across storage and offline edge devices. Only "
            "`complete` means every copy is gone."
        ),
    )
    requested_at: UtcDatetime = Field(description="When erasure was requested.")
    completed_at: UtcDatetime | None = Field(
        description="When erasure finished propagating, or null while it has not.",
    )
    error_code: Identifier | None = Field(
        description="Why propagation stalled, or null when it has not.",
    )


class ForgetReceipt(DeletionTombstoneView):
    """Deletion result returned by one command or status lookup."""

    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")


class DeletionListRequest(ContractModel):
    """Tenant-scoped cursor request used by reconnecting edge devices."""

    tenant_id: Identifier = Field(description="Tenant whose deletion barriers to list.")
    cursor: Identifier | None = Field(
        default=None,
        description=(
            "`next_cursor` from the previous page. It must belong to this tenant and still "
            "exist; a stale one is rejected rather than answered with an empty page."
        ),
    )
    limit: Annotated[
        int,
        Field(ge=1, le=100, description="Maximum tombstones to return on this page."),
    ] = 100


class DeletionPage(ContractModel):
    """Stable ordered tombstones and the next cursor when another page exists."""

    items: tuple[DeletionTombstoneView, ...] = Field(
        description="This page of deletion barriers, in a stable order.",
    )
    next_cursor: Identifier | None = Field(
        description="Pass as `cursor` for the next page, or null when this page is the last.",
    )
    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")


class RecallQuery(ContractModel):
    """Text, media, or both used to find relevant experience."""

    text: NonEmptyString | None = Field(
        default=None,
        description="What to look for, in words. Required unless `media_object_ids` is given.",
    )
    media_object_ids: Annotated[
        tuple[Identifier, ...],
        Field(
            max_length=8,
            description=(
                "Stored media to search by, at most 8 and without repeats — an image or clip "
                "stands in for the query itself. Required unless `text` is given."
            ),
        ),
    ] = ()

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

    person_ids: Annotated[
        tuple[Identifier, ...],
        Field(max_length=100, description="Keep only memories involving these people."),
    ] = ()
    device_ids: Annotated[
        tuple[Identifier, ...],
        Field(max_length=100, description="Keep only memories captured by these devices."),
    ] = ()
    memory_types: Annotated[
        tuple[MemoryType, ...],
        Field(max_length=len(MemoryType), description="Keep only these memory types."),
    ] = ()
    occurred_after: UtcDatetime | None = Field(
        default=None,
        description="Keep only memories occurring at or after this moment; the bound is inclusive.",
    )
    occurred_before: UtcDatetime | None = Field(
        default=None,
        description=(
            "Keep only memories occurring strictly before this moment -- unlike "
            "`occurred_after`, this bound is exclusive, so a memory exactly at it is left "
            "out. Must not precede `occurred_after`."
        ),
    )

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

    tenant_id: Identifier = Field(description="Tenant whose memory to search.")
    query: RecallQuery = Field(description="What to look for, as text, media, or both.")
    memory_ids: Annotated[
        tuple[Identifier, ...],
        Field(
            max_length=100,
            description=(
                "Restrict the search to exactly these memories, at most 100 and without "
                "repeats — this is how a grounded follow-up reuses IDs from a previous "
                "result. It is a strict scope, not a ranking hint."
            ),
        ),
    ] = ()
    filters: RecallFilters = Field(
        default_factory=RecallFilters,
        description="Structured constraints applied before ranking, not after it.",
    )
    mode: RecallMode = Field(
        default=RecallMode.ANSWER,
        description=(
            "What work to do. `answer` reasons over the retrieved memories and fills `answer`. "
            "`search` ranks and returns memories, leaving `answer` null. `enumerate` scans the "
            "complete filter scope for count and timeline questions and fails rather than "
            "silently truncating an oversized one."
        ),
    )
    limit: Annotated[
        int,
        Field(ge=1, le=100, description="Maximum memories to return."),
    ] = 20
    include_evidence: bool = Field(
        default=True,
        description="Return the evidence needed to verify the answer; on by default.",
    )

    @model_validator(mode="after")
    def require_unique_memory_scope(self) -> RecallRequest:
        """Keep explicit follow-up context ordered and unambiguous."""
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("recall memory_ids must be unique")
        return self


class MemoryView(ContractModel):
    """Serializable stable view of a retained memory."""

    memory_id: Identifier = Field(description="Stable ID; pass it to a read or a follow-up.")
    memory_type: MemoryType = Field(description="The role this memory serves.")
    summary: NonEmptyString = Field(description="What is remembered, in words.")
    evidence_ids: tuple[Identifier, ...] = Field(
        description="Evidence spans grounding this memory.",
    )
    occurred_at: UtcDatetime = Field(description="When the remembered events began.")
    ended_at: UtcDatetime = Field(description="When the remembered events ended.")
    created_at: UtcDatetime = Field(description="When MindBridge retained this version.")
    verification_status: VerificationStatus = Field(
        description=(
            "How the content was established: whether original media was inspected, or the "
            "content was only attested by its writer."
        ),
    )
    state: MemoryState = Field(
        description="Where this memory sits in its lifecycle, from active through compressed.",
    )
    salience: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="How important this memory is judged to be."),
    ] = 0.5
    strength: float = Field(
        default=0.5,
        allow_inf_nan=False,
        description="Retention score, raised by useful access and lowered by decay.",
    )
    useful_access_count: Annotated[
        int,
        Field(ge=0, description="How often recalling this memory proved useful."),
    ] = 0
    positive_feedback_count: Annotated[
        int,
        Field(ge=0, description="How often it was reported correct."),
    ] = 0
    negative_feedback_count: Annotated[
        int,
        Field(ge=0, description="How often it was reported wrong."),
    ] = 0
    last_accessed_at: UtcDatetime | None = Field(
        default=None,
        description="When it was last recalled, or null if never.",
    )
    supersedes_memory_id: Identifier | None = Field(
        default=None,
        description="The earlier version this one replaced, when it is a correction.",
    )
    superseded_at: UtcDatetime | None = Field(
        default=None,
        description="When a later version replaced this one; null while it is current.",
    )


class GetMemoryRequest(ContractModel):
    """Identify one tenant-owned memory without relying on ambient tenancy."""

    tenant_id: Identifier = Field(description="Tenant that owns the memory; must match the key.")
    memory_id: Identifier = Field(description="Exact memory to read.")


class EvidenceView(ContractModel):
    """Precise evidence location safe to expose to a caller."""

    evidence_id: Identifier = Field(description="Stable ID of this evidence span.")
    media_object_id: Identifier = Field(description="Media object the span was cut from.")
    start_ms: Annotated[
        int,
        Field(ge=0, description="Span start within that media, in milliseconds."),
    ]
    end_ms: Annotated[
        int,
        Field(ge=0, description="Span end within that media, in milliseconds."),
    ]
    media_url: NonEmptyString = Field(
        description="Signed URL for the bytes, so verifying needs no separate storage call.",
    )
    media_url_expires_at: UtcDatetime = Field(
        description="When `media_url` stops working; re-read the memory for a fresh one.",
    )


class MemoryResult(MemoryView):
    """Top-level memory response with directly inspectable evidence."""

    evidence: tuple[EvidenceView, ...] = Field(
        default=(),
        description="Signed evidence for this memory, attached rather than referenced.",
    )
    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")


class MemoryWriteStatus(str, Enum):
    """Whether an explicit write stored new content or matched an earlier one."""

    CREATED = "created"
    DUPLICATE = "duplicate"


class RememberResult(MemoryResult):
    """One retained memory and whether this request is what created it."""

    status: MemoryWriteStatus = Field(
        description=(
            "`created` when this request stored the memory, `duplicate` when an earlier write "
            "under the same idempotency key already had. The memory returned is the same "
            "either way, so a retry is safe without being silent."
        ),
    )


class RecallResult(ContractModel):
    """Answer plus the memory and evidence needed to verify it."""

    answer: str | None = Field(
        description="The answer, or null in `search` mode and when nothing supports one.",
    )
    confidence: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="How well the retrieved evidence supports `answer`."),
    ]
    memories: tuple[MemoryView, ...] = Field(
        description="Memories the answer rests on; pass their IDs back for a follow-up.",
    )
    evidence: tuple[EvidenceView, ...] = Field(
        description="Signed original media behind those memories, so the answer is checkable.",
    )
    trace_id: Identifier = Field(
        description="Correlates this recall with its telemetry, and names it to `missing` feedback.",
    )


class ValidationIssue(ContractModel):
    """One sanitized request validation failure."""

    location: tuple[str, ...] = Field(description="Path to the rejected field within the request.")
    message: NonEmptyString = Field(description="Why that field was rejected.")
    code: Identifier = Field(description="Machine-readable kind of validation failure.")


class ErrorResponse(ContractModel):
    """Stable error envelope with a reproducible trace identifier."""

    code: Identifier = Field(description="Stable error code; branch on this, not on the message.")
    message: NonEmptyString = Field(description="One human-readable sentence about the failure.")
    trace_id: Identifier = Field(description="Correlates this failure with its telemetry.")
    issues: tuple[ValidationIssue, ...] = Field(
        default=(),
        description="Which fields were rejected, present when the request failed validation.",
    )


class HealthResponse(ContractModel):
    """Liveness response that does not claim dependency readiness."""

    status: str = Field(
        default="ok",
        description="Liveness only; it makes no claim about dependency readiness.",
    )
    trace_id: Identifier = Field(description="Correlates this request with its telemetry.")
