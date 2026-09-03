"""Small values accepted and returned by the public MindBridge API."""

from __future__ import annotations

import math
import re
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

from mindbridge.exceptions import ValidationError

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")


class Modality(str, Enum):
    """A native MindBridge input modality; omni means multiple media kinds."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OMNI = "omni"


class MemoryType(str, Enum):
    """The cognitive role a memory serves for an agent."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class EvidenceBasis(str, Enum):
    """How a source or derived memory became known."""

    OBSERVATION = "observation"
    USER_STATEMENT = "user_statement"
    MODEL_INFERENCE = "model_inference"
    RESPONSE_FEEDBACK = "response_feedback"


class MemoryKind(str, Enum):
    """The typed semantic role maintained for a memory record."""

    OBSERVATION = "observation"
    ENTITY = "entity"
    EVENT = "event"
    STATE = "state"
    RELATION = "relation"
    AFFECT = "affect"
    TRAIT = "trait"
    RESPONSE_POLICY = "response_policy"


# The `MemoryType` a validated formation proposal of each kind is stored as. Kernel policy, kept
# beside the two enums because two readers depend on it and must not drift: `memory.py` applies it
# when it writes a formed record, and the context compiler reads it back to know which bundle
# sections a `memory_types` filter can empty. A kind absent here is stored as `SEMANTIC`.
KIND_MEMORY_TYPES: Mapping[MemoryKind, MemoryType] = MappingProxyType(
    {
        MemoryKind.EVENT: MemoryType.EPISODIC,
        MemoryKind.AFFECT: MemoryType.EPISODIC,
        MemoryKind.RESPONSE_POLICY: MemoryType.PROCEDURAL,
    }
)


class ContextUnknownKind(str, Enum):
    """Why a compiled bundle is missing something the request implied it might contain."""

    SCOPE_EMPTY = "scope_empty"
    SECTION_EMPTY = "section_empty"
    BUDGET_EXCLUDED = "budget_excluded"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"
    MODALITY_UNSUPPORTED = "modality_unsupported"
    STAGE_SKIPPED = "stage_skipped"


class SpatialAnchor(str, Enum):
    """Whether a pose locates the observer or the observed subject."""

    OBSERVER = "observer"
    SUBJECT = "subject"


class AbstentionReason(str, Enum):
    """Why an answer backend declined to answer from retrieved evidence."""

    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class IndexQuantization(str, Enum):
    """Explicit compression applied only to the rebuildable vector index."""

    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    RABITQ = "rabitq"


@dataclass(frozen=True, slots=True)
class Blob:
    """Immutable inline media bytes with an explicit IANA media type."""

    data: bytes = field(repr=False)
    media_type: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValidationError("Blob data must be non-empty bytes")
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        object.__setattr__(self, "name", _optional_text(self.name, "Blob name"))


@dataclass(frozen=True, slots=True)
class AssetRef:
    """A local asset, or an opaque ID for Memory to resolve from SQLite."""

    id: str
    modality: Modality | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    name: str | None = None
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "asset id"))
        modality = self.modality
        if modality is not None and not isinstance(modality, Modality):
            try:
                modality = Modality(modality)
            except (TypeError, ValueError):
                raise ValidationError("asset modality is invalid") from None
        if modality in {Modality.TEXT, Modality.OMNI}:
            raise ValidationError("asset modality must be image, video, or audio")
        media_type = _optional_media_type(self.media_type)
        if modality is not None and media_type is not None:
            _require_matching_media_type(modality, media_type)
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValidationError("asset size_bytes must be a non-negative integer")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValidationError("asset sha256 must be 64 lowercase hexadecimal characters")
        path = self.path
        if path is not None:
            try:
                path = Path(path)
            except TypeError:
                raise ValidationError("asset path must be a filesystem path") from None
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "name", _optional_text(self.name, "asset name"))
        object.__setattr__(self, "path", path)

    @property
    def is_resolved(self) -> bool:
        """Whether this reference carries all metadata required by a model adapter."""
        return (
            self.modality is not None
            and self.media_type is not None
            and self.size_bytes is not None
            and self.sha256 is not None
            and self.path is not None
        )


ContentAtom: TypeAlias = str | Path | Blob | AssetRef
ContentInput: TypeAlias = ContentAtom | Sequence[ContentAtom]


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialContext:
    """One metric pose in an application-selected Cartesian coordinate frame."""

    frame_id: str
    anchor: SpatialAnchor
    x: float
    y: float
    z: float = 0.0
    orientation_xyzw: tuple[float, float, float, float] | None = None
    position_uncertainty_m: float | None = None

    def __post_init__(self) -> None:  # noqa: C901 - validates one physical value
        object.__setattr__(self, "frame_id", _text(self.frame_id, "spatial frame_id"))
        if not isinstance(self.anchor, SpatialAnchor):
            try:
                object.__setattr__(self, "anchor", SpatialAnchor(self.anchor))
            except (TypeError, ValueError):
                raise ValidationError("spatial anchor is invalid") from None
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise ValidationError(f"spatial {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        uncertainty = self.position_uncertainty_m
        if uncertainty is not None:
            if (
                isinstance(uncertainty, bool)
                or not isinstance(uncertainty, int | float)
                or not math.isfinite(float(uncertainty))
                or uncertainty < 0
            ):
                raise ValidationError(
                    "spatial position_uncertainty_m must be a non-negative finite number"
                )
            object.__setattr__(self, "position_uncertainty_m", float(uncertainty))
        supplied_orientation = self.orientation_xyzw
        if supplied_orientation is None:
            return
        try:
            values = tuple(float(value) for value in supplied_orientation)
        except (TypeError, ValueError):
            raise ValidationError("spatial orientation_xyzw must contain four numbers") from None
        if len(values) != 4 or any(not math.isfinite(value) for value in values):
            raise ValidationError("spatial orientation_xyzw must contain four finite numbers")
        x, y, z, w = values
        norm = math.hypot(*values)
        if norm == 0:
            raise ValidationError("spatial orientation_xyzw must not be zero")
        normalized: tuple[float, float, float, float] = (
            x / norm,
            y / norm,
            z / norm,
            w / norm,
        )
        sign_value = normalized[3]
        if sign_value == 0:
            sign_value = next((value for value in normalized[:3] if value != 0), 0)
        if sign_value < 0:
            normalized = (
                -normalized[0],
                -normalized[1],
                -normalized[2],
                -normalized[3],
            )
        object.__setattr__(
            self,
            "orientation_xyzw",
            tuple(0.0 if value == 0 else value for value in normalized),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationContext:
    """Typed provenance, validity, and optional pose supplied with an observation."""

    basis: EvidenceBasis = EvidenceBasis.OBSERVATION
    source_id: str | None = None
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    spatial: SpatialContext | None = None
    # The symbolic room-level place, as distinct from `spatial`'s metric pose: a robot can label
    # "kitchen" when it cannot localise, and "in the kitchen" is the query a household asks.
    # Equality-matched, so a producer must label consistently -- "kitchen" and "the kitchen" would
    # partition the store, and nothing normalises it beyond rejecting untrimmed text.
    place_id: str | None = None

    def __post_init__(self) -> None:
        if self.place_id is not None:
            place = self.place_id
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("place_id must be non-empty and trimmed")
        if not isinstance(self.basis, EvidenceBasis):
            try:
                object.__setattr__(self, "basis", EvidenceBasis(self.basis))
            except (TypeError, ValueError):
                raise ValidationError("observation basis is invalid") from None
        object.__setattr__(
            self,
            "source_id",
            _optional_text(self.source_id, "observation source_id"),
        )
        object.__setattr__(
            self,
            "confidence",
            _unit_interval(self.confidence, "observation confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "valid_from",
            "valid_until",
        )
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("observation spatial context is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalScope:
    """Optional world-time, knowledge-time, and spatial retrieval scope.

    Two spatial axes, because a household asks in both. `near`/`radius_m` is metric and answers
    "within two metres of here". `place_id` is symbolic and answers "in the kitchen" — the query a
    person actually asks, and the one a robot can label when it cannot localise metrically. They
    are independent: a symbolic equality is the one spatial predicate SQLite indexes cheaply,
    while the metric radius is a filter over retrieved candidates.
    """

    valid_at: datetime | None = None
    known_at: datetime | None = None
    near: SpatialContext | None = None
    radius_m: float | None = None
    # `None` means "do not scope by place", never "memories that have no place".
    place_id: str | None = None

    def __post_init__(self) -> None:
        _require_aware(self.valid_at, "scope valid_at")
        _require_aware(self.known_at, "scope known_at")
        if self.place_id is not None:
            place = self.place_id
            # Matches the store's own rule and its SQLite CHECK, so a value that would be
            # rejected on the write path cannot be silently accepted on the read path.
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("scope place_id must be non-empty and trimmed")
        if (self.near is None) != (self.radius_m is None):
            raise ValidationError("scope near and radius_m must be supplied together")
        if self.near is not None and not isinstance(self.near, SpatialContext):
            raise ValidationError("scope near must be a SpatialContext")
        if self.radius_m is not None:
            radius = self.radius_m
            if (
                isinstance(radius, bool)
                or not isinstance(radius, int | float)
                or not math.isfinite(float(radius))
                or radius < 0
            ):
                raise ValidationError("scope radius_m must be a non-negative finite number")
            object.__setattr__(self, "radius_m", float(radius))


@dataclass(frozen=True, slots=True, kw_only=True)
class FormationProposal:
    """One typed, source-grounded semantic memory proposed by a model adapter."""

    kind: MemoryKind
    content: str
    basis: EvidenceBasis = EvidenceBasis.MODEL_INFERENCE
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    spatial: SpatialContext | None = None
    cue_modality: Modality | None = None
    valence: float | None = None
    arousal: float | None = None

    def __post_init__(self) -> None:  # noqa: C901 - kind-specific boundary validation
        if not isinstance(self.kind, MemoryKind):
            try:
                object.__setattr__(self, "kind", MemoryKind(self.kind))
            except (TypeError, ValueError):
                raise ValidationError("formation kind is invalid") from None
        if self.kind is MemoryKind.OBSERVATION:
            raise ValidationError("formation cannot propose an observation")
        if not isinstance(self.basis, EvidenceBasis):
            try:
                object.__setattr__(self, "basis", EvidenceBasis(self.basis))
            except (TypeError, ValueError):
                raise ValidationError("formation basis is invalid") from None
        if self.basis is EvidenceBasis.OBSERVATION:
            raise ValidationError("formation basis cannot be observation")
        object.__setattr__(self, "content", _text(self.content, "formation content"))
        for name in ("subject", "predicate", "value"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"formation {name}"),
            )
        needs_subject = {
            MemoryKind.ENTITY,
            MemoryKind.STATE,
            MemoryKind.RELATION,
            MemoryKind.AFFECT,
            MemoryKind.TRAIT,
            MemoryKind.RESPONSE_POLICY,
        }
        needs_predicate = {
            MemoryKind.STATE,
            MemoryKind.RELATION,
            MemoryKind.TRAIT,
            MemoryKind.RESPONSE_POLICY,
        }
        needs_value = needs_predicate | {MemoryKind.AFFECT}
        if self.kind in needs_subject and self.subject is None:
            raise ValidationError(f"{self.kind.value} formation requires a subject")
        if self.kind in needs_predicate and self.predicate is None:
            raise ValidationError(f"{self.kind.value} formation requires a predicate")
        if self.kind in needs_value and self.value is None:
            raise ValidationError(f"{self.kind.value} formation requires a value")
        object.__setattr__(
            self,
            "confidence",
            _unit_interval(self.confidence, "formation confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "formation valid_from",
            "formation valid_until",
        )
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("formation spatial context is invalid")
        if self.cue_modality is not None:
            try:
                cue_modality = Modality(self.cue_modality)
            except (TypeError, ValueError):
                raise ValidationError("formation cue_modality is invalid") from None
            if cue_modality is Modality.OMNI:
                raise ValidationError("formation cue_modality must be atomic")
            object.__setattr__(self, "cue_modality", cue_modality)
        object.__setattr__(self, "valence", _bounded_optional(self.valence, "valence", -1, 1))
        object.__setattr__(self, "arousal", _bounded_optional(self.arousal, "arousal", 0, 1))
        if self.kind is not MemoryKind.AFFECT and any(
            value is not None for value in (self.cue_modality, self.valence, self.arousal)
        ):
            raise ValidationError("affect fields require kind='affect'")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryContext:
    """Authoritative typed semantics attached to a stored memory."""

    kind: MemoryKind
    basis: EvidenceBasis
    confidence: float
    valid_from: datetime | None
    valid_until: datetime | None
    recorded_at: datetime
    visible: bool = True
    retired_at: datetime | None = None
    lineage_id: str | None = None
    source_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    evidence_ids: tuple[str, ...] = ()
    supersedes_id: str | None = None
    model_id: str | None = None
    recipe: str | None = None
    identity_id: str | None = None
    spatial: SpatialContext | None = None
    cue_modality: Modality | None = None
    valence: float | None = None
    arousal: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MemoryKind) or not isinstance(self.basis, EvidenceBasis):
            raise ValidationError("memory context kind or basis is invalid")
        object.__setattr__(
            self,
            "confidence",
            _unit_interval(self.confidence, "memory context confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "memory context valid_from",
            "memory context valid_until",
        )
        _require_aware(self.recorded_at, "memory context recorded_at")
        if not isinstance(self.visible, bool):
            raise ValidationError("memory context visible must be a boolean")
        _require_aware(self.retired_at, "memory context retired_at")
        if self.retired_at is not None and self.retired_at <= self.recorded_at:
            raise ValidationError("memory context retired_at must follow recorded_at")
        for name in (
            "source_id",
            "lineage_id",
            "subject",
            "predicate",
            "value",
            "supersedes_id",
            "model_id",
            "recipe",
            "identity_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"memory context {name}"),
            )
        evidence_ids = tuple(dict.fromkeys(self.evidence_ids))
        if any(not isinstance(value, str) or not value.strip() for value in evidence_ids):
            raise ValidationError("memory context evidence_ids are invalid")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("memory context spatial value is invalid")
        if self.cue_modality is not None and not isinstance(self.cue_modality, Modality):
            raise ValidationError("memory context cue_modality is invalid")
        object.__setattr__(self, "valence", _bounded_optional(self.valence, "valence", -1, 1))
        object.__setattr__(self, "arousal", _bounded_optional(self.arousal, "arousal", 0, 1))


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityClaim:
    """Who one recognized person is, as an `IDENTIFY` operation claims it.

    The kernel, not the backend, turns this into the typed ENTITY assertion `identities.name`
    projects, so a proposal only has to name the identity and cite its evidence.
    """

    identity_id: str
    name: str
    relationship: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        for label, value in (("name", self.name), ("relationship", self.relationship)):
            if value is None and label == "relationship":
                continue
            text = _text(value, f"identity {label}")
            if len(text) > 255 or not text.isprintable():
                raise ValidationError(f"identity {label} must be at most 255 printable characters")
            object.__setattr__(self, label, text)


class MemoryIntent(str, Enum):
    """One memory-management operation the agentic control plane may propose."""

    REINFORCE = "reinforce"
    CONSOLIDATE = "consolidate"
    CORRECT = "correct"
    FORGET = "forget"
    IDENTIFY = "identify"


class MemoryTrigger(str, Enum):
    """Why one bounded memory-management deliberation ran."""

    MANUAL = "manual"
    EVIDENCE = "evidence"
    FEEDBACK = "feedback"
    CONTRADICTION = "contradiction"
    QUERY_FAILURE = "query_failure"
    PRESSURE = "pressure"
    IDLE = "idle"


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryOperation:
    """One proposed memory operation whose required fields follow from its intent."""

    intent: MemoryIntent
    evidence_ids: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    proposal: FormationProposal | None = None
    claim: IdentityClaim | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:  # noqa: C901 - intent-specific boundary validation
        if not isinstance(self.intent, MemoryIntent):
            try:
                object.__setattr__(self, "intent", MemoryIntent(self.intent))
            except (TypeError, ValueError):
                raise ValidationError("memory operation intent is invalid") from None
        object.__setattr__(self, "evidence_ids", _memory_ids(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "target_ids", _memory_ids(self.target_ids, "target_ids"))
        object.__setattr__(
            self,
            "rationale",
            _optional_text(self.rationale, "operation rationale"),
        )
        if self.proposal is not None and not isinstance(self.proposal, FormationProposal):
            raise ValidationError("memory operation proposal is invalid")
        if self.claim is not None and not isinstance(self.claim, IdentityClaim):
            raise ValidationError("memory operation claim is invalid")
        if self.intent is MemoryIntent.IDENTIFY:
            # A host naming somebody cites nothing, so an empty evidence set is well formed here
            # and it is the kernel that requires cited evidence of an agent's proposal.
            if self.claim is None or self.target_ids or self.proposal is not None:
                raise ValidationError(
                    "identify requires a claim and cites evidence, and names no target"
                )
            return
        if self.claim is not None:
            raise ValidationError(f"{self.intent.value} must not carry a claim")
        if self.intent is MemoryIntent.CONSOLIDATE:
            # `target_ids` on a consolidation is consolidation forgetting: the sources to retire
            # from ordinary recall once the derived record exists. The kernel enforces the
            # containment rule (a target must be one of this proposal's own evidence IDs); the
            # value type only knows that naming a target requires evidence to name it from.
            if self.proposal is None or not self.evidence_ids:
                raise ValidationError("consolidate requires a proposal and cited evidence")
        elif self.proposal is not None:
            raise ValidationError(f"{self.intent.value} must not carry a proposal")
        elif self.intent is MemoryIntent.REINFORCE:
            if len(self.target_ids) != 1 or not self.evidence_ids:
                raise ValidationError("reinforce requires exactly one target and cited evidence")
        elif not self.target_ids or self.evidence_ids:
            raise ValidationError(
                f"{self.intent.value} requires at least one target and cites no evidence"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryOperationRecord:
    """One applied control-plane operation as the append-only log holds it."""

    operation_id: int
    operation: MemoryOperation
    trigger: MemoryTrigger
    applied_at: datetime
    model_id: str | None = None
    recipe: str | None = None
    created_ids: tuple[str, ...] = ()
    changed_ids: tuple[str, ...] = ()
    # Records this operation moved out of ordinary recall. On a FORGET row that is cognitive
    # forgetting; on a CONSOLIDATE row it is consolidation forgetting, the sources the derived
    # record replaced. `rollback()` clears exactly these.
    forgotten_ids: tuple[str, ...] = ()
    # `(memory_id, version)` pairs the kernel's own lineage rule superseded while applying this
    # operation: the records a new `STATE` or user-stated `TRAIT` replaced in its lineage, which
    # the backend never named and may never have been shown. `rollback()` restores exactly these.
    superseded: tuple[tuple[str, int], ...] = ()
    rolled_back_at: datetime | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.operation_id, bool)
            or not isinstance(self.operation_id, int)
            or self.operation_id <= 0
        ):
            raise ValidationError("operation_id must be a positive integer")
        if not isinstance(self.operation, MemoryOperation):
            raise ValidationError("operation must be a MemoryOperation")
        if not isinstance(self.trigger, MemoryTrigger):
            raise ValidationError("operation trigger is invalid")
        _require_aware(self.applied_at, "applied_at")
        _require_aware(self.rolled_back_at, "rolled_back_at")
        for name in ("model_id", "recipe"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"operation {name}"),
            )
        object.__setattr__(self, "created_ids", _memory_ids(self.created_ids, "created_ids"))
        object.__setattr__(self, "changed_ids", _memory_ids(self.changed_ids, "changed_ids"))
        object.__setattr__(self, "forgotten_ids", _memory_ids(self.forgotten_ids, "forgotten_ids"))
        superseded = tuple(self.superseded)
        for memory_id, version in superseded:
            _memory_ids((memory_id,), "superseded")
            if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
                raise ValidationError("superseded version must be a positive integer")
        object.__setattr__(self, "superseded", superseded)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidationCandidate:
    """One piece of deliberation work the store's own state says is due.

    `memory_ids` is the set to hand straight to `consolidate(evidence_ids=...)`. `evidence_count`
    is what the trigger counted: new evidence links for `EVIDENCE`, distinct conflicting values
    for `CONTRADICTION`, and recorded confirmations for `FEEDBACK`.
    """

    trigger: MemoryTrigger
    memory_ids: tuple[str, ...]
    evidence_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.trigger, MemoryTrigger):
            raise ValidationError("candidate trigger is invalid")
        object.__setattr__(self, "memory_ids", _memory_ids(self.memory_ids, "memory_ids"))
        if not self.memory_ids:
            raise ValidationError("a consolidation candidate must name at least one memory")
        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(self.evidence_count, int)
            or self.evidence_count <= 0
        ):
            raise ValidationError("evidence_count must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsolidationReport:
    """What one consolidation pass applied and which proposals the kernel refused."""

    operations: tuple[MemoryOperationRecord, ...] = ()
    rejected: tuple[tuple[MemoryOperation, str], ...] = ()

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(not isinstance(value, MemoryOperationRecord) for value in operations):
            raise ValidationError("consolidation operations are invalid")
        try:
            rejected = tuple((value, reason) for value, reason in self.rejected)
        except (TypeError, ValueError):
            raise ValidationError("consolidation rejections are invalid") from None
        if any(
            not isinstance(value, MemoryOperation)
            or not isinstance(reason, str)
            or not reason.strip()
            for value, reason in rejected
        ):
            raise ValidationError("consolidation rejections are invalid")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "rejected", rejected)


@dataclass(frozen=True, slots=True)
class StreamInput:
    """One independently durable observation from an omni input stream."""

    content: ContentInput
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] | None = field(default=None, hash=False)
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: ObservationContext | None = None
    transcript: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        content = _stream_content(self.content)
        _require_interval(self.occurred_at, self.occurred_end)
        if not isinstance(self.memory_type, MemoryType):
            raise ValidationError("memory_type is invalid")
        metadata = self.metadata
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValidationError("metadata must be a mapping")
            metadata = dict(metadata)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", metadata)
        if self.context is not None and not isinstance(self.context, ObservationContext):
            raise ValidationError("stream context must be an ObservationContext")
        object.__setattr__(
            self,
            "transcript",
            _optional_text(self.transcript, "stream transcript"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "stream description"),
        )


class AudioBoundary(str, Enum):
    """A canonical acoustic segment boundary."""

    START = "start"
    END = "end"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class PCMChunk:
    """One immutable chunk of interleaved WAV-compatible linear PCM."""

    data: bytes = field(repr=False)
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValidationError("PCM chunk data must be non-empty bytes")
        for name in ("sample_rate_hz", "channels", "sample_width_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"PCM {name} must be a positive integer")
        if self.sample_width_bytes > 4:
            raise ValidationError("PCM sample_width_bytes must be between one and four")
        frame_bytes = self.channels * self.sample_width_bytes
        if frame_bytes > 0xFFFF or self.sample_rate_hz * frame_bytes > 0xFFFFFFFF:
            raise ValidationError("PCM format exceeds WAV limits")
        if len(self.data) % frame_bytes:
            raise ValidationError("PCM chunk data must contain complete sample frames")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "PCM occurred_at")


@dataclass(frozen=True, slots=True)
class VADPacket:
    """One canonical voice-activity state transition."""

    active: bool
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise ValidationError("VAD active must be a boolean")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "VAD occurred_at")


@dataclass(frozen=True, slots=True)
class ASRPartial:
    """The complete current ASR hypothesis for one audio stream."""

    text: str
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("ASR partial text must be text")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "ASR occurred_at")


@dataclass(frozen=True, slots=True)
class AcousticBoundary:
    """Start, finish, or cancel one canonical acoustic segment."""

    boundary: AudioBoundary
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, AudioBoundary):
            try:
                object.__setattr__(self, "boundary", AudioBoundary(self.boundary))
            except (TypeError, ValueError):
                raise ValidationError("audio boundary is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "audio boundary occurred_at")


AudioStreamPacket: TypeAlias = PCMChunk | VADPacket | ASRPartial | AcousticBoundary


class VisionBoundary(str, Enum):
    """A canonical visual scene boundary."""

    START = "start"
    END = "end"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class VisionFrame:
    """One immutable encoded image frame from a visual stream."""

    image: Blob
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, Blob) or not self.image.media_type.startswith("image/"):
            raise ValidationError("vision frame must contain an image Blob")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision frame occurred_at")


@dataclass(frozen=True, slots=True)
class VisionPartial:
    """The complete current caption, OCR, or detector description for one scene."""

    text: str
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("vision partial text must be text")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision partial occurred_at")


@dataclass(frozen=True, slots=True)
class SceneBoundary:
    """Start, finish, or cancel one canonical visual scene."""

    boundary: VisionBoundary
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, VisionBoundary):
            try:
                object.__setattr__(self, "boundary", VisionBoundary(self.boundary))
            except (TypeError, ValueError):
                raise ValidationError("vision boundary is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision boundary occurred_at")


VisionStreamPacket: TypeAlias = VisionFrame | VisionPartial | SceneBoundary


class StreamPhase(str, Enum):
    """Lifecycle boundary emitted by a modality-specific capture adapter."""

    UPDATE = "update"
    FINAL = "final"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A complete current snapshot, final durable observation, or cancellation."""

    phase: StreamPhase
    item: ContentInput | StreamInput | None = None
    stream_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.phase, StreamPhase):
            try:
                object.__setattr__(self, "phase", StreamPhase(self.phase))
            except (TypeError, ValueError):
                raise ValidationError("stream phase is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        if self.phase is StreamPhase.CANCEL:
            if self.item is not None:
                raise ValidationError("cancel stream events must not contain an item")
            return
        if self.item is None:
            raise ValidationError("update and final stream events require an item")
        if self.phase is StreamPhase.UPDATE and isinstance(self.item, StreamInput):
            raise ValidationError("update stream events require a content snapshot")
        content = (
            self.item.content if isinstance(self.item, StreamInput) else _stream_content(self.item)
        )
        if not isinstance(self.item, StreamInput):
            object.__setattr__(self, "item", content)
        atoms = (content,) if isinstance(content, (str, Path, Blob, AssetRef)) else tuple(content)
        if self.phase is StreamPhase.UPDATE and any(isinstance(atom, Path) for atom in atoms):
            raise ValidationError("update stream paths are mutable; use Blob or AssetRef")


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One timed transcript turn linked to a stable local speaker identity."""

    asset_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    identity_score: float | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("speaker segment asset_id must be a SHA-256 identifier")
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speaker segment must have a positive time range")
        object.__setattr__(self, "text", _text(self.text, "speaker segment text"))
        if self.speaker_id is not None:
            object.__setattr__(
                self,
                "speaker_id",
                _text(self.speaker_id, "speaker_id"),
            )
        if self.speaker_name is not None:
            name = _text(self.speaker_name, "speaker_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("speaker_name must be at most 255 printable characters")
            object.__setattr__(self, "speaker_name", name)
        if self.identity_score is not None and (
            isinstance(self.identity_score, bool)
            or not isinstance(self.identity_score, int | float)
            or not math.isfinite(float(self.identity_score))
            or not 0.0 <= self.identity_score <= 1.0
        ):
            raise ValidationError("identity_score must be between zero and one")


@dataclass(frozen=True, slots=True)
class IdentityProfile:
    """What a caller has registered about one recognized person.

    `confirmed` and `evidence_ids` are derived, never stored: a person is confirmed exactly
    while a visible naming assertion names them, and the evidence is what that assertion
    cites. An unconfirmed person is provisional -- present, recognized, not yet named -- which
    is a different thing from an unknown ID, and the difference decides what an agent may say.
    """

    identity_id: str
    name: str | None = None
    relationship: str | None = None
    confirmed: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        object.__setattr__(self, "confirmed", bool(self.confirmed))
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(_text(value, "evidence id") for value in self.evidence_ids),
        )
        for label in ("name", "relationship"):
            value = getattr(self, label)
            if value is None:
                continue
            text = _text(value, f"identity {label}")
            if len(text) > 255 or not text.isprintable():
                raise ValidationError(f"identity {label} must be at most 255 printable characters")
            object.__setattr__(self, label, text)


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face linked to a stable local multimodal identity."""

    asset_id: str
    bounding_box: tuple[float, float, float, float]
    identity_id: str
    identity_name: str | None = None
    identity_score: float | None = None
    observed_at_ms: int | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("face observation asset_id must be a SHA-256 identifier")
        box = tuple(self.bounding_box)
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            raise ValidationError("face bounding_box must contain four finite values")
        x, y, width, height = box
        if (
            x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.0
            or y + height > 1.0
        ):
            raise ValidationError("face bounding_box must be normalized within the frame")
        object.__setattr__(self, "bounding_box", box)
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if self.identity_name is not None:
            name = _text(self.identity_name, "identity_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("identity_name must be at most 255 printable characters")
            object.__setattr__(self, "identity_name", name)
        if self.identity_score is not None and (
            isinstance(self.identity_score, bool)
            or not isinstance(self.identity_score, int | float)
            or not math.isfinite(float(self.identity_score))
            or not 0.0 <= self.identity_score <= 1.0
        ):
            raise ValidationError("identity_score must be between zero and one")
        if self.observed_at_ms is not None and (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ValidationError("observed_at_ms must be a non-negative integer")


def _metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    return dict(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must not be blank")
    return value.strip()


def _stream_content(value: object) -> ContentInput:
    if isinstance(value, str):
        return _text(value, "stream content")
    if isinstance(value, (Path, Blob, AssetRef)):
        return value
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise ValidationError("stream content must be text, media, or an ordered sequence of them")
    atoms = tuple(value)
    if not atoms:
        raise ValidationError("stream content must not be empty")
    normalized: list[ContentAtom] = []
    for atom in atoms:
        if isinstance(atom, str):
            normalized.append(_text(atom, "stream text"))
        elif isinstance(atom, (Path, Blob, AssetRef)):
            normalized.append(atom)
        else:
            raise ValidationError("stream content contains an unsupported value")
    return tuple(normalized)


def _optional_text(value: object | None, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _media_type(value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError("media_type must be an image, video, or audio media type")
    normalized = value.strip().lower()
    if _MEDIA_TYPE.fullmatch(normalized) is None or normalized.split("/", 1)[0] not in {
        "image",
        "video",
        "audio",
    }:
        raise ValidationError("media_type must be an image, video, or audio media type")
    return normalized


def _optional_media_type(value: object | None) -> str | None:
    return None if value is None else _media_type(value)


def _require_matching_media_type(modality: Modality, media_type: str) -> None:
    if media_type.split("/", 1)[0] != modality.value:
        raise ValidationError("asset modality does not match media_type")


def _memory_ids(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError(f"{name} must be a sequence of memory IDs")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValidationError(f"{name} must contain non-blank memory IDs")
    return tuple(dict.fromkeys(values))


def _require_aware(value: datetime | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValidationError(f"{name} must include a timezone")


def _require_interval(start: datetime | None, end: datetime | None) -> None:
    _require_aware(start, "occurred_at")
    _require_aware(end, "occurred_end")
    if end is not None and (start is None or end <= start):
        raise ValidationError("occurred_end must be later than occurred_at")


def _require_named_interval(
    start: datetime | None,
    end: datetime | None,
    start_name: str,
    end_name: str,
) -> None:
    _require_aware(start, start_name)
    _require_aware(end, end_name)
    if end is not None and (start is None or end <= start):
        raise ValidationError(f"{end_name} must be later than {start_name}")


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _bounded_optional(
    value: object | None,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not minimum <= value <= maximum
    ):
        raise ValidationError(f"{name} must be between {minimum:g} and {maximum:g}")
    return float(value)


def _trace_number(value: object, name: str, *, maximum: float | None = None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValidationError(f"trace {name} must be a non-negative finite number")
    if maximum is not None and value > maximum:
        raise ValidationError(f"trace {name} must not exceed one")


def _assets(value: Sequence[AssetRef], modality: Modality) -> tuple[AssetRef, ...]:
    assets = tuple(value)
    if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
        raise ValidationError("record assets must contain resolved AssetRef values")
    kinds = {asset.modality for asset in assets}
    if modality is Modality.TEXT and assets:
        raise ValidationError("text memories must not contain media assets")
    if modality in {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO} and (
        not assets or kinds != {modality}
    ):
        raise ValidationError("memory modality does not match its assets")
    if modality is Modality.OMNI and len(kinds) < 2:
        raise ValidationError("omni memories must contain at least two media modalities")
    return assets


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One stored memory.

    `content` is the caller's text followed by any text the configured models derived from the
    media. Derived sections are appended, never substituted: what the caller supplied stays
    byte-identical at the front, and each derived section is introduced by its own marker line --
    `[transcript:<asset_id>]`, `[visual description:<asset_id>]`, or
    `[speech identities:<asset_id>]` -- so a reader can tell interpretation from evidence and
    which asset it came from. `add` derives before its first write and `settle` derives after
    `capture` already committed, so the same record shape results either way; the raw media is
    never rewritten and stays in `assets` under its content address.
    """

    id: str
    content: str
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: MemoryContext | None = None
    # The symbolic place this was captured in, as supplied on `ObservationContext`. Read-back
    # matters because the label is equality-matched: an application cannot notice it wrote
    # "the kitchen" where "kitchen" was meant unless it can see what was stored.
    place_id: str | None = None
    forgotten_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_aware(self.forgotten_at, "forgotten_at")
        if self.place_id is not None:
            place = self.place_id
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("place_id must be non-empty and trimmed")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryType):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)
        if self.context is not None and not isinstance(self.context, MemoryContext):
            raise ValidationError("memory context is invalid")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked memory, flattened for direct use."""

    id: str
    content: str
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: MemoryContext | None = None
    # A hit is a memory, so a place-scoped search reports which place each came from.
    place_id: str | None = None
    forgotten_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_aware(self.forgotten_at, "forgotten_at")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between zero and one")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryType):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)
        if self.context is not None and not isinstance(self.context, MemoryContext):
            raise ValidationError("memory context is invalid")


@dataclass(frozen=True, slots=True)
class PendingCapture:
    """One record whose deferred work is not finished, and why it is still waiting.

    `Memory.pending_captures` returns these. `awaiting` is the stage the record is stopped at:
    `"enrichment"` has no vectors and is invisible to `search`, while `"formation"` is already
    embedded, indexed, and searchable and owes only the formation `add` holds a row for between
    its commit and its model call. A memory ID that is absent from the result is not pending: it
    is either settled or unknown, and `get` distinguishes the two. `attempts` and `last_error` are
    the failure state a poisoned record accumulates, so an operator can see the reason without
    opening SQLite.
    """

    memory_id: str
    enqueued_at: datetime
    attempts: int = 0
    last_error: str | None = None
    awaiting: Literal["enrichment", "formation"] = "enrichment"

    def __post_init__(self) -> None:
        _text(self.memory_id, "memory_id")
        _require_aware(self.enqueued_at, "enqueued_at")
        if self.enqueued_at is None:
            raise ValidationError("enqueued_at must include a timezone")
        if self.awaiting not in {"enrichment", "formation"}:
            raise ValidationError("awaiting must be enrichment or formation")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise ValidationError("attempts must be an integer")
        if self.attempts < 0:
            raise ValidationError("attempts must not be negative")
        if self.last_error is not None and not isinstance(self.last_error, str):
            raise ValidationError("last_error must be text")


@dataclass(frozen=True, slots=True)
class PrefetchResult:
    """The newest completed speculative search for one streaming turn."""

    revision: int
    hits: tuple[SearchHit, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision <= 0
        ):
            raise ValidationError("prefetch revision must be a positive integer")
        hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValidationError("prefetch hits are invalid")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class StreamCommit:
    """One durable final observation plus retrieval success or a visible retrieval failure.

    `pending_settlement` is true when the reducer committed through `capture` rather than `add`:
    the record is durable and readable but has no vectors yet, so it stays invisible to `search`
    until the host calls `settle`.
    """

    record: MemoryRecord
    prefetch: PrefetchResult | None
    retrieval_error: str | None = None
    stream_id: str = "default"
    pending_settlement: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise ValidationError("stream commit values are invalid")
        if not isinstance(self.pending_settlement, bool):
            raise ValidationError("stream commit pending_settlement must be a boolean")
        if self.prefetch is not None and not isinstance(self.prefetch, PrefetchResult):
            raise ValidationError("stream prefetch result is invalid")
        error = _optional_text(self.retrieval_error, "stream retrieval_error")
        if (self.prefetch is None) == (error is None):
            raise ValidationError("stream commit requires either prefetch or retrieval_error")
        object.__setattr__(self, "retrieval_error", error)
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))


class RetrievalRejection(str, Enum):
    """Why one bounded retrieval candidate did not become a search hit."""

    STALE_INDEX = "stale_index"
    OCCURRENCE_RANGE = "occurrence_range"
    MISSING_MEMORY = "missing_memory"
    MEMORY_TYPE = "memory_type"
    MINIMUM_RELEVANCE = "minimum_relevance"
    AMBIGUITY = "ambiguity"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class IdentityErasure:
    """What one `forget_identity` call actually destroyed.

    The counts are the audit record for a privacy operation: a caller that erased a person can
    state how many biometric templates and annotations were removed instead of assuming. It lives
    here rather than in the storage layer because it is the return type of a public `Memory`
    operation, and the supported import path is `mindbridge`.
    """

    identity_id: str
    alias_ids: tuple[str, ...]
    face_exemplars: int
    voice_exemplars: int
    face_observations: int
    speech_segments: int

    def __post_init__(self) -> None:
        _text(self.identity_id, "identity_id")
        for name in ("face_exemplars", "voice_exemplars", "face_observations", "speech_segments"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """What one composition can actually do, declared by its backends rather than inferred.

    Routing is by declared capability, so this is the same information `Memory` routes on. It is
    published because a caller and an agent both need to know which modalities an operation will
    accept before sending one, and because the embedding identity is what decides whether stored
    vectors and a new backend belong to the same space.
    """

    embedding: frozenset[Modality]
    embedding_model: str
    embedding_space: str
    embedding_dimension: int
    generation: frozenset[Modality] = frozenset()
    transcription: frozenset[Modality] = frozenset()
    vision: frozenset[Modality] = frozenset()
    face: frozenset[Modality] = frozenset()
    formation: frozenset[Modality] = frozenset()
    generation_model: str | None = None
    transcription_space: str | None = None
    vision_model: str | None = None
    face_model: str | None = None
    formation_model: str | None = None
    consolidation_model: str | None = None
    # A `TranscriptionBackend` yields text; only a `SpeechBackend` resolves speakers, and the two
    # occupy the same slot, so whether `speech()` will work is not visible from the modalities.
    speaker_recognition: bool = False
    streaming_generation: bool = False

    def __post_init__(self) -> None:
        _text(self.embedding_model, "embedding_model")
        _text(self.embedding_space, "embedding_space")
        if not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValidationError("embedding_dimension must be a positive integer")
        for name in ("embedding", "generation", "transcription", "vision", "face", "formation"):
            values = getattr(self, name)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, Modality) for value in values
            ):
                raise ValidationError(f"{name} capabilities must be a frozenset of modalities")

    @property
    def operations(self) -> frozenset[str]:
        """Optional operations this composition can serve, named by the backend each needs.

        The mapping from an operation to its backend was prose in three reference pages, so a
        caller had to know that `ask` needs generation and `consolidate` needs a consolidation
        backend. It is derived, never declared: the backends above are the only source.
        """
        available = {
            "ask": bool(self.generation),
            "speech": self.speaker_recognition and bool(self.transcription),
            "transcribe": bool(self.transcription),
            "faces": bool(self.face),
            "describe_vision": bool(self.vision),
            "formation": bool(self.formation),
            "consolidate": self.consolidation_model is not None,
        }
        return frozenset(name for name, ready in available.items() if ready)

    def document(self) -> dict[str, object]:
        """Return the JSON-ready capability document every surface publishes.

        REST serves it from `/healthz`, the MCP server embeds it in its instructions, and
        `mindbridge doctor` prints it, so the three cannot describe the same composition
        differently. Modality sets and operation names are sorted so the document is stable.
        """
        values: dict[str, object] = {
            declared.name: (
                sorted(item.value for item in value)
                if isinstance(value := getattr(self, declared.name), frozenset)
                else value
            )
            for declared in fields(self)
        }
        values["operations"] = sorted(self.operations)
        return values


@dataclass(frozen=True, slots=True)
class RetrievalCandidateTrace:
    """Effective score components and final disposition for one considered parent memory."""

    memory_id: str | None
    index_ids: tuple[str, ...]
    dense_relevance: float | None = None
    dense_confidence: float | None = None
    lexical_relevance: float | None = None
    lexical_rerank_bonus: float | None = None
    lexical_match: bool = False
    gate_relevance: float | None = None
    base_relevance: float | None = None
    reinforcement_factor: float | None = None
    temporal_factor: float | None = None
    retention_factor: float | None = None
    final_score: float | None = None
    rank: int | None = None
    rejected_by: RetrievalRejection | None = None

    def __post_init__(self) -> None:
        if self.memory_id is not None:
            _text(self.memory_id, "trace memory_id")
        index_ids = tuple(self.index_ids)
        if not index_ids or len(set(index_ids)) != len(index_ids):
            raise ValidationError("trace index_ids must be non-empty and unique")
        for index_id in index_ids:
            _text(index_id, "trace index_id")
        for name in (
            "dense_relevance",
            "dense_confidence",
            "lexical_relevance",
            "lexical_rerank_bonus",
            "gate_relevance",
            "base_relevance",
            "reinforcement_factor",
            "temporal_factor",
            "retention_factor",
            "final_score",
        ):
            _trace_number(getattr(self, name), name)
        for name in (
            "dense_relevance",
            "dense_confidence",
            "lexical_relevance",
            "lexical_rerank_bonus",
            "gate_relevance",
            "base_relevance",
            "final_score",
        ):
            _trace_number(getattr(self, name), name, maximum=1.0)
        if not isinstance(self.lexical_match, bool):
            raise ValidationError("trace lexical_match must be a boolean")
        if self.rank is not None and (
            isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0
        ):
            raise ValidationError("trace rank must be a positive integer")
        if self.rejected_by is not None and not isinstance(self.rejected_by, RetrievalRejection):
            raise ValidationError("trace rejected_by is invalid")
        object.__setattr__(self, "index_ids", index_ids)


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Bounded candidate trace for one explicit traced search."""

    candidates: tuple[RetrievalCandidateTrace, ...]
    candidate_limit: int
    exhaustive: bool
    ambiguous: bool = False

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, RetrievalCandidateTrace) for candidate in candidates):
            raise ValidationError("trace candidates are invalid")
        if (
            isinstance(self.candidate_limit, bool)
            or not isinstance(self.candidate_limit, int)
            or self.candidate_limit <= 0
        ):
            raise ValidationError("trace candidate_limit must be a positive integer")
        if not isinstance(self.exhaustive, bool) or not isinstance(self.ambiguous, bool):
            raise ValidationError("trace exhaustive and ambiguous must be booleans")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class TracedSearchResult:
    """Search hits paired with their opt-in retrieval trace."""

    hits: tuple[SearchHit, ...]
    trace: RetrievalTrace

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValidationError("traced search hits are invalid")
        if not isinstance(self.trace, RetrievalTrace):
            raise ValidationError("traced search trace is invalid")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A grounded answer and the memories used to produce it."""

    answer: str
    hits: tuple[SearchHit, ...] = ()
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None

    def __post_init__(self) -> None:
        _text(self.answer, "answer")
        if not isinstance(self.abstained, bool):
            raise ValidationError("abstained must be a boolean")
        if self.abstention_reason is not None and not isinstance(
            self.abstention_reason, AbstentionReason
        ):
            raise ValidationError("abstention_reason is invalid")
        if self.abstained != (self.abstention_reason is not None):
            raise ValidationError("abstained and abstention_reason must agree")


@dataclass(frozen=True, slots=True)
class Page:
    """One stable page of memories."""

    items: tuple[MemoryRecord, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None:
            _text(self.next_cursor, "next_cursor")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBudget:
    """The size, type, confidence, and freshness bounds one context compilation may spend.

    `max_chars` bounds the rendered evidence, the same quantity `ContextBundle.chars` reports:
    the four-line header, each section heading, each memory's rendered line with its `- [id]`
    frame and confidence suffix, and a text-equivalent price per grounded media part. A bundle
    that carries no compilation diagnostics therefore satisfies `len(bundle.render()) <=
    max_chars`; the `## Conflicts`, `## Unknowns`, `Omitted:` and provisional-actor lines are
    appended outside it, because they explain the bundle rather than ground it and suppressing
    them to fit a budget would make a thin bundle look like an empty store.

    `max_media_items` bounds grounded media parts instead of their price: `0` compiles a
    text-only bundle, and `None` lets `max_chars` alone decide. The default `max_chars` buys the
    most expensive single grounded part -- a video part is priced at 12 000 characters -- and
    still leaves room for its record text and a cheaper second part, so a default compilation
    can reach a media memory at all.
    """

    max_chars: int = 16000
    max_items: int = 24
    max_media_items: int | None = None
    memory_types: frozenset[MemoryType] | None = None
    min_confidence: float = 0.0
    freshness: timedelta | None = None
    # A deadline, not a timeout: the compiler checks the clock between stages and stops adding
    # optional ones. It never interrupts work already in flight, so an exceeded deadline is
    # reported on the bundle rather than raised.
    max_latency_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_chars", _positive_int(self.max_chars, "budget max_chars"))
        object.__setattr__(self, "max_items", _positive_int(self.max_items, "budget max_items"))
        if self.max_media_items is not None:
            if isinstance(self.max_media_items, bool) or not isinstance(self.max_media_items, int):
                raise ValidationError("budget max_media_items must be a non-negative integer")
            if self.max_media_items < 0:
                raise ValidationError("budget max_media_items must be a non-negative integer")
        if self.max_latency_ms is not None:
            object.__setattr__(
                self,
                "max_latency_ms",
                _positive_int(self.max_latency_ms, "budget max_latency_ms"),
            )
        if self.memory_types is not None:
            memory_types = frozenset(self.memory_types)
            if not memory_types or any(not isinstance(value, MemoryType) for value in memory_types):
                raise ValidationError("budget memory_types must name at least one MemoryType")
            object.__setattr__(self, "memory_types", memory_types)
        object.__setattr__(
            self,
            "min_confidence",
            _unit_interval(self.min_confidence, "budget min_confidence"),
        )
        if self.freshness is not None and (
            not isinstance(self.freshness, timedelta) or self.freshness <= timedelta(0)
        ):
            raise ValidationError("budget freshness must be a positive timedelta")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextConflict:
    """Two or more candidates in one lineage that disagree on the same value.

    The compiler reports a disagreement and never resolves it. `values` and `memory_ids` are
    aligned: each value is paired with the highest-ranked candidate that asserts it, whether or
    not the budget included that memory; `render()` marks the ones it did not.
    """

    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 2 or len(self.values) != len(self.memory_ids):
            raise ValidationError("a conflict pairs at least two values with one memory each")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProvisionalActor:
    """One recognized person in the compiled evidence whom no visible assertion names.

    A stranger in the room is not a stranger missing from context. This carries the identity
    and the included memories that observed them, and nothing else: there is no name to carry,
    which is the whole point. It is not a hit, so it costs no budget and occupies no item slot.
    """

    identity_id: str
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if not self.memory_ids:
            raise ValidationError("a provisional actor names at least one observing memory")
        object.__setattr__(
            self,
            "memory_ids",
            tuple(_text(value, "memory id") for value in self.memory_ids),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextUnknown:
    """One thing the request implied a bundle might contain that this bundle does not.

    Every unknown is a deterministic statement about the compilation itself -- a scope that
    matched nothing, a requested type nothing was included for, evidence the budget could not
    buy, a modality this composition cannot search, a stage a deadline skipped. The compiler
    calls no model to produce one, so an unknown is never a guess about the world.
    """

    kind: ContextUnknownKind
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ContextUnknownKind):
            raise ValidationError("unknown kind is invalid")
        object.__setattr__(self, "detail", _text(self.detail, "unknown detail"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBundle:
    """One bounded, structured context view compiled for a goal."""

    goal: str
    reference_at: datetime
    budget: ContextBudget
    actors: tuple[SearchHit | ProvisionalActor, ...]
    relationships: tuple[SearchHit, ...]
    scene: tuple[SearchHit, ...]
    episodes: tuple[SearchHit, ...]
    facts: tuple[SearchHit, ...]
    procedures: tuple[SearchHit, ...]
    affect: tuple[SearchHit, ...]
    traits: tuple[SearchHit, ...]
    conflicts: tuple[ContextConflict, ...]
    unknowns: tuple[ContextUnknown, ...]
    occurred_from: datetime | None
    occurred_until: datetime | None
    frames: tuple[str, ...]
    places: tuple[str, ...]
    omitted: int
    chars: int
    elapsed_ms: int
    deadline_exceeded: bool

    @property
    def hits(self) -> tuple[SearchHit, ...]:
        """Every included hit in rank order, without duplicates.

        A provisional actor is not a hit: it is a person the evidence observed, not evidence.
        """
        merged = {
            entry.id: entry
            for _heading, section in self._sections()
            for entry in section
            if isinstance(entry, SearchHit)
        }
        return tuple(sorted(merged.values(), key=lambda hit: (-hit.score, hit.id)))

    def render(self) -> str:
        """Return the bundle as deterministic sectioned text carrying `[id]` provenance."""
        included = len(self.hits)
        lines = [
            f"# Context: {self.goal}",
            "Each line is one memory: [id] content (confidence; validity).",
            f"Reference time: {self.reference_at.isoformat()}",
            f"Budget: {self.chars}/{self.budget.max_chars} chars, "
            f"{included}/{self.budget.max_items} items",
        ]
        for heading, section in self._sections():
            if not section:
                continue
            lines.extend(("", f"## {heading}"))
            lines.extend(_section_line(entry) for entry in section)
        if self.conflicts:
            lines.extend(("", "## Conflicts"))
            included_ids = {hit.id for hit in self.hits}
            lines.extend(_conflict_line(conflict, included_ids) for conflict in self.conflicts)
        if self.unknowns:
            lines.extend(("", "## Unknowns"))
            lines.extend(f"- {item.kind.value}: {item.detail}" for item in self.unknowns)
        if self.omitted > 0:
            lines.extend(("", f"Omitted: {self.omitted} lower-ranked candidates"))
        return "\n".join(lines)

    def _sections(self) -> tuple[tuple[str, tuple[SearchHit | ProvisionalActor, ...]], ...]:
        """Return the sections in their fixed rendering order."""
        return (
            ("Actors", self.actors),
            ("Relationships", self.relationships),
            ("Scene", self.scene),
            ("Facts", self.facts),
            ("Episodes", self.episodes),
            ("Procedures", self.procedures),
            ("Affect", self.affect),
            ("Traits", self.traits),
        )


def _section_line(entry: SearchHit | ProvisionalActor) -> str:
    if isinstance(entry, SearchHit):
        return render_hit_line(entry)
    seen = ", ".join(f"[{memory_id}]" for memory_id in entry.memory_ids)
    return f"- [{entry.identity_id}] unnamed person present (provisional identity; seen in {seen})"


def render_hit_line(hit: SearchHit) -> str:
    """Return the one line `render()` writes for this hit.

    The compiler charges its length against `ContextBudget.max_chars`, so the price and the text
    come from the same function and cannot drift.
    """
    confidence = 1.0 if hit.context is None else hit.context.confidence
    marks = f"confidence {confidence:.2f}{_validity(hit)}"
    # One hit is one line, so stored newlines collapse rather than break the section shape.
    return f"- [{hit.id}] {' '.join(hit.content.split())} ({marks})"


def _validity(hit: SearchHit) -> str:
    """Render a typed memory's world-validity bounds inline, so a stale fact reads as stale.

    `MemoryContext` refuses an end without a start, so the only open form is an open-ended one.
    """
    context = hit.context
    if context is None or context.valid_from is None:
        return ""
    start = context.valid_from.isoformat()
    if context.valid_until is None:
        return f"; valid from {start}"
    return f"; valid {start} → {context.valid_until.isoformat()}"


def _conflict_line(conflict: ContextConflict, included: Container[str]) -> str:
    label = " ".join(part for part in (conflict.subject, conflict.predicate) if part)
    # A conflict may name a candidate the budget could not buy. Marking it keeps the `[id]`
    # provenance honest: that memory has no line of its own anywhere above.
    values = " vs ".join(
        f'"{value}" [{memory_id}]'
        if memory_id in included
        else f'"{value}" [{memory_id}, not included]'
        for value, memory_id in zip(conflict.values, conflict.memory_ids, strict=True)
    )
    return f"- {label or conflict.lineage_id}: {values}"
