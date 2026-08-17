"""Local Jina adapter for MindBridge's atomic embedding capability."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Protocol, cast

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.configuration import (
    config_integer,
    config_optional_string,
    config_string,
)
from mindbridge.core import (
    EmbeddingSpaceReference,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
)
from mindbridge.models.compute import select_torch_device
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
    require_matryoshka_dimension,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


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
        config_kwargs: dict[str, str],
    ) -> _SentenceEncoder: ...


class JinaEmbedder:
    """Async-safe query/document encoder for text, image, video, and audio."""

    def __init__(
        self,
        encoder: _SentenceEncoder,
        model_reference: ModelReference,
        *,
        space_reference: EmbeddingSpaceReference = DEFAULT_EMBEDDING_SPACE,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        max_concurrency: int = 1,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._encoder = encoder
        self._model_reference = model_reference
        self._space_reference = space_reference
        self._dimension = dimension
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def load(
        cls,
        *,
        revision: str,
        model_id: str = DEFAULT_EMBEDDER_MODEL_ID,
        device: str | None = None,
        space_reference: EmbeddingSpaceReference = DEFAULT_EMBEDDING_SPACE,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        max_concurrency: int = 1,
    ) -> JinaEmbedder:
        """Load a pinned upstream model without exposing training operations."""
        try:
            module = import_module("sentence_transformers")
            hub_module = import_module("huggingface_hub")
        except ImportError as error:
            raise ModelUnavailableError(
                "install MindBridge with the cloud-models extra to load Jina Omni"
            ) from error
        sentence_transformer = cast(
            _SentenceTransformerFactory,
            module.SentenceTransformer,
        )
        snapshot_download = cast(Callable[..., str], hub_module.snapshot_download)
        model_path = snapshot_download(repo_id=model_id, revision=revision)
        selected_device = select_torch_device(device)
        encoder = sentence_transformer(
            model_path,
            revision=revision,
            trust_remote_code=True,
            device=selected_device,
            model_kwargs={"modality": "omni", "code_revision": revision},
            config_kwargs={"code_revision": revision},
        )
        return cls(
            encoder,
            ModelReference(model_id=model_id, revision=revision),
            space_reference=space_reference,
            dimension=dimension,
            max_concurrency=max_concurrency,
        )

    @trace_operation("mindbridge.model.embed")
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        """Encode a homogeneous query or document batch."""
        encode = (
            self._encoder.encode_query
            if request.task is EmbedTask.QUERY
            else self._encoder.encode_document
        )
        vectors = await self._encode(request.inputs, encode)
        return EmbedResult(
            embeddings=tuple(
                Embedding(
                    values=vector,
                    model_reference=self._model_reference,
                    space_reference=self._space_reference,
                )
                for vector in vectors
            )
        )

    async def _encode(
        self,
        inputs: tuple[ModelInput, ...],
        encode: Callable[..., _EmbeddingMatrix],
    ) -> tuple[tuple[float, ...], ...]:
        if not inputs:
            return ()
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.model.revision": self._model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(inputs),
            }
        )
        async with self._semaphore:
            matrix = await asyncio.to_thread(
                encode,
                cast(list[object], [_jina_input(item) for item in inputs]),
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


def create_embedder(config: Mapping[str, object]) -> JinaEmbedder:
    """Entry-point factory for the bundled local Jina model."""
    allowed = {
        "model_id",
        "revision",
        "device",
        "space_id",
        "space_revision",
        "dimension",
        "max_concurrency",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown plugin configuration: {', '.join(sorted(unknown))}")
    return JinaEmbedder.load(
        model_id=config_string(config, "model_id", DEFAULT_EMBEDDER_MODEL_ID),
        revision=config_string(config, "revision", DEFAULT_EMBEDDER_REVISION),
        device=config_optional_string(config, "device"),
        space_reference=EmbeddingSpaceReference(
            space_id=config_string(config, "space_id", DEFAULT_EMBEDDING_SPACE.space_id),
            revision=config_string(
                config,
                "space_revision",
                DEFAULT_EMBEDDING_SPACE.revision,
            ),
        ),
        dimension=require_matryoshka_dimension(
            config_integer(config, "dimension", DEFAULT_EMBEDDING_DIMENSION)
        ),
        max_concurrency=config_integer(config, "max_concurrency", 1),
    )


def _jina_input(input_value: ModelInput) -> str | tuple[str, ...]:
    values = tuple(
        part.text if isinstance(part, TextPart) else part.url for part in input_value.parts
    )
    return values[0] if len(values) == 1 else values


def validate_jina_embedding(values: tuple[float, ...], dimension: int) -> None:
    """Reject malformed vectors before they cross into a versioned index."""
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("embedding vector has invalid dimension or values")
    norm = math.hypot(*values)
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelOutputError("embedding vector is not L2-normalized")
