"""Small values accepted and returned by the public MindBridge API."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeAlias

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

    def __post_init__(self) -> None:
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
    """Optional world-time, knowledge-time, and same-frame spatial retrieval scope."""

    valid_at: datetime | None = None
    known_at: datetime | None = None
    near: SpatialContext | None = None
    radius_m: float | None = None

    def __post_init__(self) -> None:
        _require_aware(self.valid_at, "scope valid_at")
        _require_aware(self.known_at, "scope known_at")
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
    """One stored memory."""

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

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
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

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
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
    """One durable final observation plus retrieval success or a visible retrieval failure."""

    record: MemoryRecord
    prefetch: PrefetchResult | None
    retrieval_error: str | None = None
    stream_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise ValidationError("stream commit values are invalid")
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
class RetrievalCandidateTrace:
    """Effective score components and final disposition for one considered parent memory."""

    memory_id: str | None
    index_ids: tuple[str, ...]
    dense_relevance: float | None = None
    dense_confidence: float | None = None
    lexical_relevance: float | None = None
    lexical_rerank_bonus: float | None = None
    lexical_match: bool = False
    gate_confidence: float | None = None
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
            "gate_confidence",
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
            "gate_confidence",
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
