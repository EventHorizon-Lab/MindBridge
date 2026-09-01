"""Explicit capability composition for one local memory instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from mindbridge.exceptions import ValidationError
from mindbridge.models.base import (
    EmbeddingBackend,
    FaceBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
)
from mindbridge.types import IndexQuantization

_StrictBool = Annotated[bool, Field(strict=True)]
_UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1)]
_PositiveFloat = Annotated[float, Field(strict=True, gt=0)]


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

    index_speech: _StrictBool = False
    index_quantization: IndexQuantization = IndexQuantization.NONE
    minimum_relevance: _UnitInterval = 0.55
    ambiguity_margin: _UnitInterval = 0.01
    decay_half_life_days: _PositiveFloat | None = None
    speaker_similarity: _UnitInterval = 0.78
    speaker_margin: _UnitInterval = 0.05
    face_similarity: _UnitInterval = 0.363
    face_margin: _UnitInterval = 0.05


# Clearer name for new code; keep the original public value intact for compatibility.
MemorySettings = MemoryConfig
