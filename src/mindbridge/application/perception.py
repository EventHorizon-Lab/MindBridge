"""Validated multimodal perception records shared by model adapters and use cases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from mindbridge.core import (
    AnonymousIdentityObservation,
    ClaimType,
    DomainInvariantError,
    EntityType,
    EvidenceId,
    EvidenceSpan,
    IdentityKind,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelReference,
    Observation,
    require_aware_datetime,
    require_non_empty,
)

MAX_PERCEPTION_EVENTS = 64
MAX_PERCEPTION_ENTITIES = 256
MAX_PERCEPTION_CLAIMS = 256
MAX_PERCEIVED_ENTITIES_PER_EVENT = 64
MAX_PERCEIVED_CLAIMS_PER_EVENT = 64
MAX_PERCEIVED_COUNT = 10_000
"""A bound, not a budget: it exists so a placeholder or runaway integer cannot become a memory.

Far above anything one observation window can be counted to, and far below the transcribed
on-screen numbers -- viewer counts, experience points -- that a claim states as text instead.
"""


def time_ranges_overlap(
    left_start_ms: int,
    left_end_ms: int,
    right_start_ms: int,
    right_end_ms: int,
) -> bool:
    """Return whether intervals overlap, preserving instantaneous point evidence."""
    if left_start_ms == left_end_ms or right_start_ms == right_end_ms:
        return left_start_ms <= right_end_ms and right_start_ms <= left_end_ms
    return left_start_ms < right_end_ms and right_start_ms < left_end_ms


def resolve_transcript_segments(
    observation: Observation,
    media_objects: tuple[MediaObject, ...],
) -> dict[MediaObjectId, tuple[AnonymousIdentityObservation, ...]]:
    """Resolve each transcript to one source, inferring only an unambiguous legacy audio."""
    audio_ids = {item.media_object_id for item in media_objects if item.kind is MediaKind.AUDIO}
    inferred_audio_id = next(iter(audio_ids)) if len(audio_ids) == 1 else None
    grouped: dict[MediaObjectId, list[AnonymousIdentityObservation]] = {}
    for identity in observation.identity_observations:
        if identity.kind is not IdentityKind.VOICE or identity.transcript is None:
            continue
        source_id = identity.transcript_media_object_id or inferred_audio_id
        if source_id is not None:
            grouped.setdefault(source_id, []).append(identity)
    return {
        source_id: tuple(
            sorted(
                segments,
                key=lambda item: (
                    item.start_ms,
                    item.end_ms,
                    item.identity_id,
                    item.model_reference.model_id,
                ),
            )
        )
        for source_id, segments in grouped.items()
    }


def overlapping_transcript_segments(
    segments_by_media: dict[MediaObjectId, tuple[AnonymousIdentityObservation, ...]],
    media_object_id: MediaObjectId,
    start_ms: int,
    end_ms: int,
) -> tuple[AnonymousIdentityObservation, ...]:
    """Return source-owned transcript segments intersecting one exact evidence window."""
    return tuple(
        item
        for item in segments_by_media.get(media_object_id, ())
        if time_ranges_overlap(item.start_ms, item.end_ms, start_ms, end_ms)
    )


@dataclass(frozen=True, slots=True)
class ResolvedEvidence:
    """An exact evidence span joined to openable media.

    `media_url` is not always the source object: a deployment may substitute a copy cut from it,
    and `attached_media_object` is that copy when the resolver knows which object it signed. It
    is None when the request carries `media_object`'s own bytes, and also when a caller swapped
    in bytes it did not describe -- so a consumer deciding what it is about to send may only draw
    conclusions from a value that is present.

    `sampled_frames_per_second` and `sampled_max_pixels` state the budget a copy was cut at where
    the caller knows it, and are None otherwise. They are a description, not a lever: the local
    embedder path sends nothing but the URL, and the deployment's generation endpoint measurably
    ignores both (fps 1.0 and 0.5 produced the same 12,282 prompt tokens for one clip). Sending
    physically smaller bytes is what changes cost, which is what the substitution is for.
    """

    evidence_span: EvidenceSpan
    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime
    sampled_frames_per_second: float | None = None
    sampled_max_pixels: int | None = None
    attached_media_object: MediaObject | None = None

    def __post_init__(self) -> None:
        if self.evidence_span.tenant_id != self.media_object.tenant_id:
            raise DomainInvariantError("evidence and media tenants must match")
        if self.evidence_span.media_object_id != self.media_object.media_object_id:
            raise DomainInvariantError("evidence must resolve to its referenced media object")
        if (
            self.attached_media_object is not None
            and self.attached_media_object.tenant_id != self.evidence_span.tenant_id
        ):
            raise DomainInvariantError("attached media must belong to the evidence tenant")
        require_non_empty(self.media_url, "media_url")
        require_aware_datetime(self.media_url_expires_at, "media_url_expires_at")


@dataclass(frozen=True, slots=True)
class PerceivedCount:
    """How many distinct instances of one thing a claim's own window contains.

    Typed, because the prose version does not happen. Across the 7 937 claims and 9 189 events
    the 2026-08-24 evaluation wrote from six video corpora, not one claim said "exactly N": 175
    claims and 635 event descriptions instead used "multiple", "several", "various" or "a group
    of" in the exact position a number would answer a counting question, and the 301 claims that
    do carry a digit carry transcribed on-screen text (a viewer count, an item's name) rather
    than anything the model counted. The prompt already asked for exact counts in prose over
    that whole run.

    `subject` is what was counted, not the entity it belongs to: "small monsters" is a group of
    instances that never becomes a graph entity, and it is exactly what the questions ask about.
    The interval the count covers is the claim's own `valid_from_ms`/`valid_to_ms`, so an
    enumeration is a claim about a window rather than a caption with a number bolted on.

    Optional everywhere, and normally absent. A fabricated integer is strictly worse than none,
    so nothing here rewards filling it in.
    """

    subject: str
    value: int

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise DomainInvariantError("perceived count subject must not be empty")
        if not 0 <= self.value <= MAX_PERCEIVED_COUNT:
            raise DomainInvariantError("perceived count value is out of range")


@dataclass(frozen=True, slots=True)
class PerceivedEntity:
    """One named semantic entity grounded by source evidence."""

    entity_type: EntityType
    canonical_name: str
    confidence: float
    evidence_ids: tuple[EvidenceId, ...]

    def __post_init__(self) -> None:
        if not self.canonical_name.strip():
            raise DomainInvariantError("perceived entity name must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("perceived entity confidence must be between 0 and 1")
        if not self.evidence_ids or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainInvariantError("perceived entity evidence IDs must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class PerceivedClaim:
    """One evidence-grounded fact, state, intent, or relation proposed by a model."""

    claim_type: ClaimType
    statement: str
    confidence: float
    evidence_ids: tuple[EvidenceId, ...]
    valid_from_ms: int
    valid_to_ms: int | None
    entity_indices: tuple[int, ...] = ()
    exact_count: PerceivedCount | None = None

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise DomainInvariantError("perceived claim statement must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("perceived claim confidence must be between 0 and 1")
        if not self.evidence_ids or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainInvariantError("perceived claim evidence IDs must be non-empty and unique")
        if self.valid_from_ms < 0 or (
            self.valid_to_ms is not None and self.valid_to_ms < self.valid_from_ms
        ):
            raise DomainInvariantError("perceived claim validity range is invalid")
        if any(index < 0 for index in self.entity_indices) or len(set(self.entity_indices)) != len(
            self.entity_indices
        ):
            raise DomainInvariantError(
                "perceived claim entity indices must be unique and non-negative"
            )


@dataclass(frozen=True, slots=True)
class PerceivedEvent:
    """One schema-validated semantic interval proposed by a model."""

    start_ms: int
    end_ms: int
    description: str
    salience: float
    evidence_ids: tuple[EvidenceId, ...]
    entities: tuple[PerceivedEntity, ...] = ()
    claims: tuple[PerceivedClaim, ...] = ()

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("perceived event time range is invalid")
        if not self.description.strip():
            raise DomainInvariantError("perceived event description must not be empty")
        if not math.isfinite(self.salience) or not 0.0 <= self.salience <= 1.0:
            raise DomainInvariantError("perceived event salience must be between 0 and 1")
        if not self.evidence_ids or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainInvariantError("perceived event evidence IDs must be non-empty and unique")
        _require_bounded_event_details(self)
        _require_grounded_event_details(self)


def _require_bounded_event_details(event: PerceivedEvent) -> None:
    if len(event.entities) > MAX_PERCEIVED_ENTITIES_PER_EVENT:
        raise DomainInvariantError("perceived event entity count exceeds the processing limit")
    if len(event.claims) > MAX_PERCEIVED_CLAIMS_PER_EVENT:
        raise DomainInvariantError("perceived event claim count exceeds the processing limit")


def _require_grounded_event_details(event: PerceivedEvent) -> None:
    evidence_ids = set(event.evidence_ids)
    if any(not set(entity.evidence_ids) <= evidence_ids for entity in event.entities):
        raise DomainInvariantError("perceived entity evidence must belong to its event")
    if any(not set(claim.evidence_ids) <= evidence_ids for claim in event.claims):
        raise DomainInvariantError("perceived claim evidence must belong to its event")
    if any(
        index >= len(event.entities) for claim in event.claims for index in claim.entity_indices
    ):
        raise DomainInvariantError("perceived claim references an unknown event entity")
    if any(
        claim.valid_from_ms < event.start_ms
        or claim.valid_from_ms > event.end_ms
        or (claim.valid_to_ms is not None and claim.valid_to_ms > event.end_ms)
        for claim in event.claims
    ):
        raise DomainInvariantError("perceived claim validity must remain inside its event")


@dataclass(frozen=True, slots=True)
class EventPerception:
    """Reproducible event proposals and the frozen model that produced them."""

    events: tuple[PerceivedEvent, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("perception prompt version must not be empty")
        if len(self.events) > MAX_PERCEPTION_EVENTS:
            raise DomainInvariantError("perception event count exceeds the processing limit")
        if sum(len(event.entities) for event in self.events) > MAX_PERCEPTION_ENTITIES:
            raise DomainInvariantError("perception entity count exceeds the processing limit")
        if sum(len(event.claims) for event in self.events) > MAX_PERCEPTION_CLAIMS:
            raise DomainInvariantError("perception claim count exceeds the processing limit")
