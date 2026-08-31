"""Explicit capability composition for one local memory instance."""

from __future__ import annotations

from dataclasses import dataclass

from mindbridge.exceptions import ValidationError
from mindbridge.models.base import (
    EmbeddingBackend,
    FaceBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
)
from mindbridge.types import IndexQuantization


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPlugins:
    """Already-constructed capability adapters owned and closed by ``Memory``."""

    embedder: EmbeddingBackend
    answerer: GenerationBackend | None = None
    transcriber: SpeechBackend | TranscriptionBackend | None = None
    face_analyzer: FaceBackend | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedder, EmbeddingBackend):
            raise ValidationError("plugins.embedder must implement EmbeddingBackend")
        if self.answerer is not None and not isinstance(self.answerer, GenerationBackend):
            raise ValidationError("plugins.answerer must implement GenerationBackend")
        if self.transcriber is not None and not isinstance(
            self.transcriber,
            (SpeechBackend, TranscriptionBackend),
        ):
            raise ValidationError(
                "plugins.transcriber must implement SpeechBackend or TranscriptionBackend"
            )
        if self.face_analyzer is not None and not isinstance(self.face_analyzer, FaceBackend):
            raise ValidationError("plugins.face_analyzer must implement FaceBackend")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryConfig:
    """Value-only local policy kept separate from runtime capability objects."""

    index_speech: bool = False
    index_quantization: IndexQuantization = IndexQuantization.NONE
    minimum_relevance: float = 0.55
    ambiguity_margin: float = 0.01
    decay_half_life_days: float | None = None
    speaker_similarity: float = 0.78
    speaker_margin: float = 0.05
    face_similarity: float = 0.363
    face_margin: float = 0.05
