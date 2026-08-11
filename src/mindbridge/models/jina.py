"""Jina v5 Omni embedding through the official Sentence Transformers API."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from typing import Protocol, cast

from mindbridge.application import EmbeddingInput
from mindbridge.core import ModelOutputError, ModelReference, ModelUnavailableError

DEFAULT_JINA_OMNI_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_JINA_OMNI_REVISION = "12949877f0092093f366c6450340011320152a05"
DEFAULT_JINA_OMNI_DIMENSION = 1_024


class JinaModality(str, Enum):
    """Selective towers supported by the upstream Jina model."""

    OMNI = "omni"
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"


class _EmbeddingMatrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _SentenceEncoder(Protocol):
    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> _EmbeddingMatrix: ...

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> _EmbeddingMatrix: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        trust_remote_code: bool,
        device: str | None,
        model_kwargs: dict[str, str],
    ) -> _SentenceEncoder: ...


class JinaOmniEmbedder:
    """Async-safe query/document encoder for text, image, video, and audio."""

    def __init__(
        self,
        encoder: _SentenceEncoder,
        model_reference: ModelReference,
        *,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        max_concurrency: int = 1,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._encoder = encoder
        self._model_reference = model_reference
        self._dimension = dimension
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def load(
        cls,
        *,
        revision: str,
        model_id: str = DEFAULT_JINA_OMNI_MODEL_ID,
        device: str | None = None,
        modality: JinaModality = JinaModality.OMNI,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        max_concurrency: int = 1,
    ) -> JinaOmniEmbedder:
        """Load a pinned upstream model without exposing training operations."""
        try:
            module = import_module("sentence_transformers")
        except ImportError as error:
            raise ModelUnavailableError(
                "install MindBridge with the cloud-models extra to load Jina Omni"
            ) from error
        sentence_transformer = cast(
            _SentenceTransformerFactory,
            module.SentenceTransformer,
        )
        encoder = sentence_transformer(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device=device,
            model_kwargs={"modality": modality.value},
        )
        return cls(
            encoder,
            ModelReference(model_id=model_id, revision=revision),
            dimension=dimension,
            max_concurrency=max_concurrency,
        )

    @property
    def model_reference(self) -> ModelReference:
        """Return the exact model identity stored beside every vector."""
        return self._model_reference

    @property
    def dimension(self) -> int:
        """Return the configured Matryoshka output dimension."""
        return self._dimension

    async def encode_queries(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode retrieval queries with the upstream query prompt semantics."""
        return await self._encode(inputs, self._encoder.encode_query)

    async def encode_documents(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode index documents with the upstream document prompt semantics."""
        return await self._encode(inputs, self._encoder.encode_document)

    async def _encode(
        self,
        inputs: tuple[EmbeddingInput, ...],
        encode: Callable[..., _EmbeddingMatrix],
    ) -> tuple[tuple[float, ...], ...]:
        if not inputs:
            return ()
        async with self._semaphore:
            matrix = await asyncio.to_thread(
                encode,
                cast(list[object], list(inputs)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self._dimension,
            )
        vectors = tuple(tuple(float(value) for value in row) for row in matrix.tolist())
        if len(vectors) != len(inputs):
            raise ModelOutputError("embedding batch size does not match its inputs")
        for vector in vectors:
            validate_jina_embedding(vector, self._dimension)
        return vectors


def validate_jina_embedding(values: tuple[float, ...], dimension: int) -> None:
    """Reject malformed vectors before they cross into a versioned index."""
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("embedding vector has invalid dimension or values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelOutputError("embedding vector is not L2-normalized")
