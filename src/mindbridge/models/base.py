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
    def embedding_capabilities(self) -> frozenset[Modality]: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_space(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

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
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """One thread-safe grounded-answer adapter over a provider SDK."""

    @property
    def generation_capabilities(self) -> frozenset[Modality]: ...

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult: ...

    def close(self) -> None: ...


@runtime_checkable
class TranscriptionBackend(Protocol):
    """One thread-safe plain-transcription adapter over a provider SDK."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...
