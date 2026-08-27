"""Minimal model values shared by local and remote backends."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from mindbridge.exceptions import ValidationError
from mindbridge.types import AnswerResult, AssetRef, Modality, SearchHit


class EmbedTask(str, Enum):
    """Retrieval semantics for asymmetric embedding models."""

    QUERY = "retrieval.query"
    DOCUMENT = "retrieval.document"


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Aggregated text and ordered, resolved media assets for one model call."""

    text: str = ""
    assets: tuple[AssetRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("model input text must be text")
        assets = tuple(self.assets)
        if not self.text.strip() and not assets:
            raise ValidationError("model input must contain text or media")
        if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
            raise ValidationError("model input assets must be resolved")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "assets", assets)

    @property
    def modality(self) -> Modality:
        """Return text, one media kind, or omni for two or more media kinds."""
        kinds = {asset.modality for asset in self.assets}
        if not kinds:
            return Modality.TEXT
        if len(kinds) == 1:
            kind = next(iter(kinds))
            if kind is not None:
                return kind
        return Modality.OMNI

    @property
    def modalities(self) -> frozenset[Modality]:
        """Return every atomic modality present in this input."""
        values = {asset.modality for asset in self.assets}
        if self.text:
            values.add(Modality.TEXT)
        return frozenset(value for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Explicit input modalities supported by each model operation."""

    embedding: frozenset[Modality]
    generation: frozenset[Modality]
    transcription: frozenset[Modality]

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _modalities(self.embedding, "embedding"))
        object.__setattr__(self, "generation", _modalities(self.generation, "generation"))
        object.__setattr__(
            self,
            "transcription",
            _modalities(self.transcription, "transcription"),
        )


def _modalities(
    values: Iterable[Modality],
    name: str,
) -> frozenset[Modality]:
    try:
        normalized = frozenset(
            value if isinstance(value, Modality) else Modality(value) for value in values
        )
    except (TypeError, ValueError):
        raise ValidationError(f"{name} capabilities contain an invalid modality") from None
    if Modality.OMNI in normalized:
        raise ValidationError(f"{name} capabilities must contain only atomic modalities")
    return normalized


@runtime_checkable
class EmbeddingBackend(Protocol):
    """One thread-safe embedding model with a stable vector-space recipe."""

    @property
    def capabilities(self) -> frozenset[Modality]: ...

    @property
    def model_id(self) -> str: ...

    @property
    def space_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SpeechTurn:
    """One timed local-speaker turn returned by a speech backend."""

    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speech turn must have a positive time range")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("speech turn text must not be blank")
        object.__setattr__(self, "text", self.text.strip())
        if self.speaker_label is not None:
            if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
                raise ValidationError("speech turn speaker_label must not be blank")
            object.__setattr__(self, "speaker_label", self.speaker_label.strip())


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """One backend-local speaker label and its voiceprint centroid."""

    speaker_label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
            raise ValidationError("speaker embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("speaker embedding must contain finite values")
        object.__setattr__(self, "speaker_label", self.speaker_label.strip())
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    """Timed transcript and voiceprint centroids for one media asset."""

    turns: tuple[SpeechTurn, ...]
    speakers: tuple[SpeakerEmbedding, ...]

    def __post_init__(self) -> None:
        turns, speakers = tuple(self.turns), tuple(self.speakers)
        if any(not isinstance(turn, SpeechTurn) for turn in turns) or any(
            not isinstance(speaker, SpeakerEmbedding) for speaker in speakers
        ):
            raise ValidationError("speech analysis contains invalid values")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "speakers", speakers)


@runtime_checkable
class SpeechBackend(Protocol):
    """One thread-safe speech analyzer with a stable recognition recipe."""

    @property
    def capabilities(self) -> frozenset[Modality]: ...

    @property
    def model_id(self) -> str: ...

    @property
    def space_id(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One normalized face box emitted by a face backend."""

    face_label: str
    bbox_xyxy: tuple[float, float, float, float]
    detection_score: float
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.face_label, str) or not self.face_label.strip():
            raise ValidationError("face detection label must not be blank")
        bbox = tuple(self.bbox_xyxy)
        if len(bbox) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= value <= 1.0
            for value in bbox
        ):
            raise ValidationError("face detection bbox_xyxy must contain normalized coordinates")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValidationError("face detection bbox_xyxy must have positive area")
        if (
            isinstance(self.detection_score, bool)
            or not isinstance(self.detection_score, int | float)
            or not math.isfinite(float(self.detection_score))
            or not 0.0 <= self.detection_score <= 1.0
        ):
            raise ValidationError("face detection score must be between zero and one")
        if (self.start_ms is None) != (self.end_ms is None) or (
            self.start_ms is not None
            and (
                isinstance(self.start_ms, bool)
                or not isinstance(self.start_ms, int)
                or isinstance(self.end_ms, bool)
                or not isinstance(self.end_ms, int)
                or self.start_ms < 0
                or self.end_ms <= self.start_ms
            )
        ):
            raise ValidationError("face detection time range must be absent or positive")
        object.__setattr__(self, "face_label", self.face_label.strip())
        object.__setattr__(self, "bbox_xyxy", bbox)


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """One backend-local face label and its recognition embedding."""

    face_label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.face_label, str) or not self.face_label.strip():
            raise ValidationError("face embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("face embedding must contain finite values")
        object.__setattr__(self, "face_label", self.face_label.strip())
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """Face detections and recognition embeddings for one visual asset."""

    detections: tuple[FaceDetection, ...]
    faces: tuple[FaceEmbedding, ...]

    def __post_init__(self) -> None:
        detections, faces = tuple(self.detections), tuple(self.faces)
        if any(not isinstance(value, FaceDetection) for value in detections) or any(
            not isinstance(value, FaceEmbedding) for value in faces
        ):
            raise ValidationError("face analysis contains invalid values")
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "faces", faces)


@runtime_checkable
class FaceBackend(Protocol):
    """One thread-safe face analyzer with a stable recognition recipe."""

    @property
    def capabilities(self) -> frozenset[Modality]: ...

    @property
    def model_id(self) -> str: ...

    @property
    def space_id(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class ModelBackend(Protocol):
    """The complete model surface consumed by Memory.

    Calls may overlap across threads; implementations must be thread-safe until ``close``.
    """

    @property
    def capabilities(self) -> ModelCapabilities: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_space(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult: ...

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...
