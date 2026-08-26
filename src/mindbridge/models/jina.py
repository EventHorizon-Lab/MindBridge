"""Local SentenceTransformers adapter for MindBridge's embedding capability."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Protocol, cast

from pydantic import StrictBool

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.configuration import PluginConfigModel, PluginInteger, PluginText
from mindbridge.core import (
    EmbeddingSpaceReference,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.models._vectors import validate_embedding_vector
from mindbridge.models.compute import select_torch_device
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
    EmbeddingDimension,
    embedder_revision_for,
    require_distinct_embedding_space,
)
from mindbridge.telemetry import operation_span, set_current_span_attributes


class _EmbeddingMatrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _SentenceEncoder(Protocol):
    def supports(self, modality: str | tuple[str, ...]) -> bool: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        trust_remote_code: bool,
        device: str | None,
        model_kwargs: dict[str, str | None] | None,
        config_kwargs: dict[str, str | None] | None,
    ) -> _SentenceEncoder: ...


_ABSENT = object()
_JINA_MODALITIES = frozenset({"text", "image", "video", "audio", "message"})
_MISSING_PROCESSOR = (
    "the model's image and video processor did not load, so it can only embed text; "
    "install MindBridge with the cloud-models extra"
)


def _media_processor_missing(encoder: object) -> bool:
    """Catch Jina's swallowed optional-dependency failure during readiness."""
    try:
        modules = list(iter(encoder))  # type: ignore[call-overload]
    except TypeError:
        return False
    return any(getattr(module, "processor", _ABSENT) is None for module in modules)


class SentenceTransformersEmbedder:
    """Async-safe SentenceTransformers query/document encoder."""

    def __init__(
        self,
        encoder: _SentenceEncoder,
        model_reference: ModelReference,
        *,
        space_reference: EmbeddingSpaceReference = DEFAULT_EMBEDDING_SPACE,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        truncate_dim: int | None = None,
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
        self._truncate_dim = truncate_dim
        self._legacy_jina_input = model_reference.model_id == DEFAULT_EMBEDDER_MODEL_ID
        self._document_semaphore = asyncio.Semaphore(max_concurrency)
        # Interactive recall must not queue behind bulk document encoding. The extra query
        # slot costs one encode's activation memory beyond the resident-model worker guard.
        self._query_semaphore = asyncio.Semaphore(1)

    @classmethod
    def load(
        cls,
        *,
        model_id: str = DEFAULT_EMBEDDER_MODEL_ID,
        revision: str | None = None,
        trust_remote_code: bool | None = None,
        device: str | None = None,
        space_reference: EmbeddingSpaceReference = DEFAULT_EMBEDDING_SPACE,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        max_concurrency: int = 1,
    ) -> SentenceTransformersEmbedder:
        """Load one model and validate its vector and modality metadata."""
        revision = embedder_revision_for(model_id, revision)
        require_distinct_embedding_space(
            model_id,
            space_reference.space_id,
            model_revision=revision,
        )
        trusted = (
            model_id == DEFAULT_EMBEDDER_MODEL_ID
            if trust_remote_code is None
            else trust_remote_code
        )
        try:
            module = import_module("sentence_transformers")
        except ImportError as error:
            raise ModelUnavailableError(
                "install MindBridge with the cloud-models extra to load SentenceTransformers"
            ) from error
        sentence_transformer = cast(_SentenceTransformerFactory, module.SentenceTransformer)
        code_kwargs: dict[str, str | None] | None = (
            {"code_revision": revision} if trusted and revision is not None else None
        )
        encoder = sentence_transformer(
            model_id,
            revision=revision,
            trust_remote_code=trusted,
            device=select_torch_device(device),
            model_kwargs=code_kwargs,
            config_kwargs=code_kwargs,
        )
        if model_id == DEFAULT_EMBEDDER_MODEL_ID and _media_processor_missing(encoder):
            raise ModelUnavailableError(_MISSING_PROCESSOR)
        truncate_dim = _validated_truncate_dim(encoder, dimension)
        return cls(
            encoder,
            ModelReference(model_id=model_id),
            space_reference=space_reference,
            dimension=dimension,
            truncate_dim=truncate_dim,
            max_concurrency=max_concurrency,
        )

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        """Declare the search space this model's vectors belong to."""
        return self._space_reference

    @operation_span("mindbridge.model.embed")
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        """Encode a query or document batch through the official task methods."""
        is_query = request.task is EmbedTask.QUERY
        encode = self._encoder.encode_query if is_query else self._encoder.encode_document
        vectors = await self._encode(
            request.inputs,
            encode,
            self._query_semaphore if is_query else self._document_semaphore,
        )
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
        semaphore: asyncio.Semaphore,
    ) -> tuple[tuple[float, ...], ...]:
        if not inputs:
            return ()
        prepared = [self._model_input(item) for item in inputs]
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.embedding.space_id": self._space_reference.space_id,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(inputs),
            }
        )
        async with semaphore:
            matrix = await asyncio.to_thread(
                encode,
                prepared,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self._truncate_dim,
            )
        vectors = tuple(tuple(float(value) for value in row) for row in matrix.tolist())
        if len(vectors) != len(inputs):
            raise ModelOutputError("embedding batch size does not match its inputs")
        for vector in vectors:
            validate_embedding_vector(vector, self._dimension)
        return vectors

    def _model_input(self, input_value: ModelInput) -> object:
        modalities = tuple(_part_modality(part) for part in input_value.parts)
        for modality in set(modalities):
            if not self._supports(modality):
                raise ModelRequestError(
                    f"model {self._model_reference.model_id!r} does not support {modality} input"
                )
        values = tuple(_part_value(part) for part in input_value.parts)
        if len(values) == 1:
            return (
                values[0]
                if self._legacy_jina_input or modalities[0] == "text"
                else {modalities[0]: values[0]}
            )
        if self._legacy_jina_input:
            return values
        compound = tuple(sorted(set(modalities)))
        if len(compound) == len(modalities) and self._supports(compound):
            return dict(zip(modalities, values, strict=True))
        if not self._supports("message"):
            raise ModelRequestError(
                f"model {self._model_reference.model_id!r} cannot combine these input parts"
            )
        return [{"role": "user", "content": [_message_part(part) for part in input_value.parts]}]

    def _supports(self, modality: str | tuple[str, ...]) -> bool:
        if self._legacy_jina_input:
            requested = (modality,) if isinstance(modality, str) else modality
            return all(item in _JINA_MODALITIES for item in requested)
        return self._encoder.supports(modality)


# Compatibility for code that imported the original provider-specific class directly.
JinaEmbedder = SentenceTransformersEmbedder


class _EmbedderConfig(PluginConfigModel):
    model_id: PluginText = DEFAULT_EMBEDDER_MODEL_ID
    model_revision: PluginText | None = None
    trust_remote_code: StrictBool | None = None
    device: PluginText | None = None
    space_id: PluginText = DEFAULT_EMBEDDING_SPACE.space_id
    dimension: EmbeddingDimension = DEFAULT_EMBEDDING_DIMENSION
    max_concurrency: PluginInteger = 1


def create_embedder(config: Mapping[str, object]) -> SentenceTransformersEmbedder:
    """Entry-point factory for local SentenceTransformers models."""
    validated = _EmbedderConfig.model_validate(config)
    return SentenceTransformersEmbedder.load(
        model_id=validated.model_id,
        revision=validated.model_revision,
        trust_remote_code=validated.trust_remote_code,
        device=validated.device,
        space_reference=EmbeddingSpaceReference(space_id=validated.space_id),
        dimension=validated.dimension,
        max_concurrency=validated.max_concurrency,
    )


def _validated_truncate_dim(encoder: _SentenceEncoder, requested: int) -> int | None:
    native = encoder.get_sentence_embedding_dimension()
    if native is None or native <= 0:
        raise ModelUnavailableError("the model does not declare its native embedding dimension")
    if requested == native:
        return None
    trained = _model_config_value(encoder, "matryoshka_dimensions")
    is_matryoshka = _model_config_value(encoder, "is_matryoshka")
    if (
        requested > native
        or is_matryoshka is not True
        or not isinstance(trained, (list, tuple))
        or requested not in trained
    ):
        raise ModelUnavailableError(
            f"model native dimension is {native}; requested dimension {requested} is not an "
            "advertised Matryoshka dimension"
        )
    return requested


def _model_config_value(encoder: _SentenceEncoder, name: str) -> object:
    try:
        first = encoder[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        first = None
    candidates = (
        getattr(encoder, "config", None),
        getattr(first, "config", None),
        getattr(getattr(first, "auto_model", None), "config", None),
        getattr(getattr(first, "model", None), "config", None),
    )
    for config in candidates:
        for source in (config, getattr(config, "text_config", None)):
            value = getattr(source, name, _ABSENT)
            if value is not _ABSENT:
                return value
    return None


def _part_modality(part: TextPart | MediaPart) -> str:
    return "text" if isinstance(part, TextPart) else part.kind.value


def _part_value(part: TextPart | MediaPart) -> str:
    return part.text if isinstance(part, TextPart) else part.url


def _message_part(part: TextPart | MediaPart) -> dict[str, str]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    return {"type": part.kind.value, part.kind.value: part.url}
