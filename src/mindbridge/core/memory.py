"""Derived memory records that remain traceable to captured evidence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from mindbridge.core._validation import (
    require_aware_datetime,
    require_non_empty,
    require_probability,
)
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import (
    ClaimId,
    EmbeddingId,
    EventId,
    EvidenceId,
    MemoryId,
    ObservationId,
    TenantId,
)


class VerificationStatus(str, Enum):
    """How a record is supported without conflating reports and observations."""

    VERIFIED = "verified"
    ATTESTED = "attested"
    UNVERIFIED = "unverified"


class ClaimType(str, Enum):
    """Semantic role of a versioned assertion."""

    FACT = "fact"
    STATE = "state"
    INTENT = "intent"
    RELATION = "relation"

    @classmethod
    def _missing_(cls, value: object) -> ClaimType | None:
        """Resolve a listed alias for a role this enum names differently.

        Perception asked for `claim_type='action'` 186 times across 134 observations in the
        2026-08-21 evaluation, the only enum violation in the whole run, and deterministic
        rather than stochastic: three clips each failed all three attempts with the identical
        rejection. Every rejection discarded an entire observation's events, entities, and
        claims after paying a slow generator for them.

        An observed action is a perceptible fact at a time, and a claim already carries its
        temporal extent in `valid_from`/`valid_to`, so this mapping loses the type label
        rather than any content. Resolving it here instead of at one parse site covers every
        caller that builds a `ClaimType` from a string -- the perception pipeline, the claim
        consolidation reader -- without either of them holding a copy of the alias table.

        Only listed aliases resolve. Anything else still raises, because the sole reason
        `action` is known at all is that an unrecognised value surfaced instead of being
        quietly coerced into something plausible.

        Giving `action` its own member is the better model of a robotics memory and stays
        open, but it is not a one-line change, and doing it carelessly would move the same
        loss later and make it more expensive. Three things have to be decided first:
        `migrations/0008_semantic_graph.sql:5` constrains the column to exactly these four
        values, so a fifth member without a migration converts these parse rejections into
        commit-time constraint violations; and `consolidate_claims.py` and
        `_postgres_claim_consolidation.py` both partition merge candidates by exact
        `claim_type`, so a fifth type is a fifth partition in each.
        """
        if not isinstance(value, str):
            return None
        alias = value.strip().casefold()
        resolved = _CLAIM_TYPE_ALIASES.get(alias)
        if resolved is None:
            return None
        _CLAIM_TYPE_ALIAS_USES[alias] += 1
        return resolved

    @classmethod
    def alias_uses(cls) -> dict[str, int]:
        """How often each alias resolved in this process, so drift is seen and not inferred."""
        return dict(_CLAIM_TYPE_ALIAS_USES)


_CLAIM_TYPE_ALIASES: dict[str, ClaimType] = {"action": ClaimType.FACT}
_CLAIM_TYPE_ALIAS_USES: Counter[str] = Counter()


class EmbeddedObjectType(str, Enum):
    """Domain records admitted to the primary semantic index."""

    EVIDENCE_SPAN = "evidence_span"
    EVENT = "event"
    CLAIM = "claim"
    MEMORY_RECORD = "memory_record"
    ENTITY = "entity"


class MemoryType(str, Enum):
    """Long-term role a memory serves."""

    PERCEPTUAL = "perceptual"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROSPECTIVE = "prospective"


class MemoryState(str, Enum):
    """Retained states; explicit deletion is represented by a tombstone."""

    ACTIVE = "active"
    STRENGTHENED = "strengthened"
    COLD = "cold"
    COMPRESSED = "compressed"


class EventHierarchyLevel(str, Enum):
    """Whether an event is a directly perceived event or a consolidated episode."""

    EVENT = "event"
    EPISODE = "episode"


class EventStatus(str, Enum):
    """Lifecycle of one derived event hierarchy node."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class ModelReference:
    """The model that produced a derived record."""

    model_id: str

    def __post_init__(self) -> None:
        require_non_empty(self.model_id, "model_id")


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceReference:
    """Compatibility space shared by independently served encoders."""

    space_id: str

    def __post_init__(self) -> None:
        require_non_empty(self.space_id, "space_id")


@dataclass(frozen=True, slots=True)
class Event:
    """A semantic event derived from one or more observations."""

    event_id: EventId
    tenant_id: TenantId
    observation_ids: tuple[ObservationId, ...]
    evidence_ids: tuple[EvidenceId, ...]
    occurred_at: datetime
    ended_at: datetime
    description: str
    salience: float
    created_at: datetime
    model_reference: ModelReference
    prompt_version: str
    parent_event_id: EventId | None = None
    hierarchy_level: EventHierarchyLevel = EventHierarchyLevel.EVENT
    status: EventStatus = EventStatus.ACTIVE

    def __post_init__(self) -> None:
        require_non_empty(self.event_id, "event_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.description, "description")
        require_non_empty(self.prompt_version, "prompt_version")
        require_aware_datetime(self.occurred_at, "occurred_at")
        require_aware_datetime(self.ended_at, "ended_at")
        require_aware_datetime(self.created_at, "created_at")
        _require_identifiers(self.observation_ids, "observation_ids")
        _require_identifiers(self.evidence_ids, "evidence_ids")
        if self.ended_at < self.occurred_at:
            raise DomainInvariantError("ended_at must not precede occurred_at")
        if self.parent_event_id == self.event_id:
            raise DomainInvariantError("event cannot be its own parent")
        _require_probability(self.salience, "salience")


@dataclass(frozen=True, slots=True)
class Claim:
    """A versioned assertion derived from evidence or marked unverified."""

    claim_id: ClaimId
    tenant_id: TenantId
    claim_type: ClaimType
    statement: str
    evidence_ids: tuple[EvidenceId, ...]
    confidence: float
    verification_status: VerificationStatus
    valid_from: datetime
    valid_to: datetime | None
    created_at: datetime
    model_reference: ModelReference
    prompt_version: str
    supersedes_claim_id: ClaimId | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.claim_id, "claim_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.statement, "statement")
        require_non_empty(self.prompt_version, "prompt_version")
        require_aware_datetime(self.valid_from, "valid_from")
        require_aware_datetime(self.created_at, "created_at")
        if self.valid_to is not None:
            require_aware_datetime(self.valid_to, "valid_to")
            if self.valid_to < self.valid_from:
                raise DomainInvariantError("valid_to must not precede valid_from")
        if self.evidence_ids:
            _require_identifiers(self.evidence_ids, "evidence_ids")
        elif self.verification_status is VerificationStatus.VERIFIED:
            raise DomainInvariantError("a verified claim must reference evidence")
        _require_probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Stable external view of an explicit or derived memory."""

    memory_id: MemoryId
    tenant_id: TenantId
    memory_type: MemoryType
    summary: str
    evidence_ids: tuple[EvidenceId, ...]
    occurred_at: datetime
    ended_at: datetime
    created_at: datetime
    verification_status: VerificationStatus
    state: MemoryState = MemoryState.ACTIVE
    model_reference: ModelReference | None = None
    salience: float = 0.5
    strength: float = 0.5
    useful_access_count: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    last_accessed_at: datetime | None = None
    supersedes_memory_id: MemoryId | None = None
    superseded_at: datetime | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.memory_id, "memory_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.summary, "summary")
        require_aware_datetime(self.occurred_at, "occurred_at")
        require_aware_datetime(self.ended_at, "ended_at")
        require_aware_datetime(self.created_at, "created_at")
        if self.ended_at < self.occurred_at:
            raise DomainInvariantError("ended_at must not precede occurred_at")
        if self.evidence_ids:
            _require_identifiers(self.evidence_ids, "evidence_ids")
        elif self.verification_status is VerificationStatus.VERIFIED:
            raise DomainInvariantError("a verified memory must reference evidence")
        _require_probability(self.salience, "salience")
        if not math.isfinite(self.strength):
            raise DomainInvariantError("strength must be finite")
        if (
            min(
                self.useful_access_count,
                self.positive_feedback_count,
                self.negative_feedback_count,
            )
            < 0
        ):
            raise DomainInvariantError("memory lifecycle counters must be non-negative")
        if self.last_accessed_at is not None:
            require_aware_datetime(self.last_accessed_at, "last_accessed_at")
            if self.last_accessed_at < self.created_at:
                raise DomainInvariantError("last_accessed_at must not precede created_at")
        if self.superseded_at is not None:
            require_aware_datetime(self.superseded_at, "superseded_at")
        if self.supersedes_memory_id == self.memory_id:
            raise DomainInvariantError("memory cannot supersede itself")


DEFAULT_EMBEDDING_DIMENSION = 1_024


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """A versioned semantic vector that never mixes incompatible spaces."""

    embedding_id: EmbeddingId
    tenant_id: TenantId
    object_type: EmbeddedObjectType
    object_id: str
    values: tuple[float, ...]
    model_reference: ModelReference
    space_reference: EmbeddingSpaceReference
    task: str
    dimension: int
    normalized: bool
    created_at: datetime
    object_part: int = 0
    """Which part of `object_id` this vector encodes, when one object is embedded in pieces.

    Zero for an object embedded whole, which is every type but one. An evidence span is cut into
    encoder-sized clips before it is embedded -- `audio_windows` splits anything longer than
    `AUDIO_WINDOW_MS`, so a 70-second span becomes three -- and each clip is a different sound
    with a different vector. Without this the vectors table held one row per span, so the second
    clip conflicted with the first, was read as content drift, and raised inside the
    single-transaction commit that writes an observation's derived records: one long audio span
    cost the whole observation, not just its own vector.
    """

    def __post_init__(self) -> None:
        require_non_empty(self.embedding_id, "embedding_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.object_id, "object_id")
        if self.object_part < 0:
            raise DomainInvariantError("object_part must not be negative")
        require_non_empty(self.task, "task")
        require_aware_datetime(self.created_at, "created_at")
        if self.dimension <= 0:
            raise DomainInvariantError("dimension must be positive")
        if len(self.values) != self.dimension:
            raise DomainInvariantError("dimension must match the number of vector values")
        if not all(math.isfinite(value) for value in self.values):
            raise DomainInvariantError("embedding values must be finite")
        if self.normalized and not math.isclose(
            math.hypot(*self.values),
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-6,
        ):
            raise DomainInvariantError("normalized embedding must have unit length")


def _require_identifiers(identifiers: tuple[str, ...], field_name: str) -> None:
    if not identifiers:
        raise DomainInvariantError(f"{field_name} must not be empty")
    if any(not identifier.strip() for identifier in identifiers):
        raise DomainInvariantError(f"{field_name} must not contain blank identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise DomainInvariantError(f"{field_name} must not contain duplicates")


_require_probability = require_probability
