"""Atomic model capability ports owned by the application layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

from mindbridge.core import (
    DomainInvariantError,
    EmbeddingSpaceReference,
    MediaKind,
    ModelReference,
)


@dataclass(frozen=True, slots=True)
class TextPart:
    """One ordered text segment."""

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise DomainInvariantError("model text must not be blank")


@dataclass(frozen=True, slots=True)
class MediaPart:
    """One model-readable media reference."""

    kind: MediaKind
    url: str
    source_uri: str | None = None
    frames_per_second: float | None = None
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise DomainInvariantError("model media URL must not be blank")
        if self.source_uri is not None and not self.source_uri.strip():
            raise DomainInvariantError("model media source URI must not be blank")
        if self.frames_per_second is not None and (
            self.kind is not MediaKind.VIDEO
            or not math.isfinite(self.frames_per_second)
            or self.frames_per_second <= 0
        ):
            raise DomainInvariantError("video frame rate must be finite and positive")
        if self.max_pixels is not None and (
            self.kind is not MediaKind.VIDEO or self.max_pixels <= 0
        ):
            raise DomainInvariantError("video pixel limit must be positive")


InputPart: TypeAlias = TextPart | MediaPart


@dataclass(frozen=True, slots=True)
class ModelInput:
    """An ordered combination of whichever modalities a model needs."""

    parts: tuple[InputPart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise DomainInvariantError("model input must contain at least one part")


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """One deterministic generation request."""

    system_prompt: str
    input: ModelInput
    max_output_tokens: int
    json_mode: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise DomainInvariantError("generation system prompt must not be blank")
        if self.max_output_tokens <= 0:
            raise DomainInvariantError("generation output token limit must be positive")


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Generated text with the exact producing model identity."""

    text: str
    model_reference: ModelReference


@runtime_checkable
class Generator(Protocol):
    """A replaceable provider or local generation model."""

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...


class EmbedTask(str, Enum):
    """Retrieval-side semantics required by asymmetric encoders."""

    QUERY = "retrieval_query"
    DOCUMENT = "retrieval_document"


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    """A homogeneous embedding batch."""

    inputs: tuple[ModelInput, ...]
    task: EmbedTask


@dataclass(frozen=True, slots=True)
class Embedding:
    """One normalized vector and its compatibility provenance."""

    values: tuple[float, ...]
    model_reference: ModelReference
    space_reference: EmbeddingSpaceReference

    def __post_init__(self) -> None:
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise DomainInvariantError("embedding values must be finite and non-empty")
        norm = math.sqrt(sum(value * value for value in self.values))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
            raise DomainInvariantError("embedding values must be L2-normalized")

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """Vectors in the same order as the request inputs."""

    embeddings: tuple[Embedding, ...]


@runtime_checkable
class Embedder(Protocol):
    """A replaceable provider or local embedding model."""

    async def embed(self, request: EmbedRequest) -> EmbedResult: ...
