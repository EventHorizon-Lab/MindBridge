"""Validated multimodal perception records shared by model adapters and use cases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from mindbridge.core import (
    ClaimType,
    DomainInvariantError,
    EntityType,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    ModelReference,
)

MAX_PERCEPTION_EVENTS = 64
MAX_PERCEPTION_ENTITIES = 256
MAX_PERCEPTION_CLAIMS = 256
MAX_PERCEIVED_ENTITIES_PER_EVENT = 64
MAX_PERCEIVED_CLAIMS_PER_EVENT = 64


@dataclass(frozen=True, slots=True)
class ResolvedEvidence:
    """An exact evidence span joined to openable source media."""

    evidence_span: EvidenceSpan
    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime

    def __post_init__(self) -> None:
        if self.evidence_span.tenant_id != self.media_object.tenant_id:
            raise DomainInvariantError("evidence and media tenants must match")
        if self.evidence_span.media_object_id != self.media_object.media_object_id:
            raise DomainInvariantError("evidence must resolve to its referenced media object")
        if not self.media_url.strip():
            raise DomainInvariantError("media_url must not be empty")
        if self.media_url_expires_at.utcoffset() is None:
            raise DomainInvariantError("media_url_expires_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PerceivedEntity:
    """One named semantic entity grounded by Omni evidence."""

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
    """One evidence-grounded fact, state, intent, or relation proposed by Omni."""

    claim_type: ClaimType
    statement: str
    confidence: float
    evidence_ids: tuple[EvidenceId, ...]
    valid_from_ms: int
    valid_to_ms: int | None
    entity_indices: tuple[int, ...] = ()

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
    """One schema-validated semantic interval proposed by an Omni model."""

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
