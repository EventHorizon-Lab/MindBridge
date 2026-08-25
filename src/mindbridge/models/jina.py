"""Local Jina adapter for MindBridge's atomic embedding capability."""

from __future__ import annotations

import asyncio
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
from mindbridge.configuration import PluginConfigModel, PluginInteger, PluginText
from mindbridge.core import (
    EmbeddingSpaceReference,
    ModelOutputError,
    ModelReference,
    ModelUnavailableError,
)
from mindbridge.models._vectors import validate_embedding_vector
from mindbridge.models.compute import select_torch_device
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDER_REVISION,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
    MatryoshkaDimension,
)
from mindbridge.telemetry import operation_span, set_current_span_attributes


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
        revision: str | None,
        trust_remote_code: bool,
        device: str | None,
        model_kwargs: dict[str, str | None],
        config_kwargs: dict[str, str | None],
    ) -> _SentenceEncoder: ...


_ABSENT = object()
"""Distinguishes an upstream that never had a processor slot from one whose slot is empty."""

_MISSING_PROCESSOR = (
    "the Jina Omni image and video processor did not load, so this model can only embed text; "
    "install MindBridge with the cloud-models extra so Torchvision is present"
)


def _media_processor_missing(encoder: object) -> bool:
    """Report an upstream module whose processor slot exists but was left empty.

    Jina builds that processor inside a bare `except Exception` and assigns `None` on failure, so
    a missing Torchvision produces a model that loads, embeds text, and then raises
    `TypeError: 'NoneType' object is not callable` on the first frame -- after a perception call
    has already been paid for. An absent attribute is not a failure: a future upstream may hold
    the processor somewhere else, and guessing wrong must not refuse a working model.
    """
    try:
        modules = list(iter(encoder))  # type: ignore[call-overload]
    except TypeError:
        return False
    return any(getattr(module, "processor", _ABSENT) is None for module in modules)


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
        self._document_semaphore = asyncio.Semaphore(max_concurrency)
        # A query is one short string a caller is waiting on; a document batch is bulk work
        # nobody is watching. Sharing one semaphore made an interactive recall queue behind
        # whatever ingest was running: an empty recall, with no memories and no generation at
        # all, took 57-71 s against a 12.4 s idle baseline during the 2026-08-21 evaluation.
        # The query lane is its own single slot rather than a reservation out of
        # `max_concurrency`, because that value defaults to 1 and carving a slot out of it
        # would leave documents none. Its price is a ceiling of `max_concurrency + 1`
        # concurrent encodes wherever one process both queries and ingests -- one encode's
        # worth of activation memory beyond what the Worker's VRAM guard estimates, which
        # counts resident weights only and says so.
        self._query_semaphore = asyncio.Semaphore(1)

    @classmethod
    def load(
        cls,
        *,
        model_id: str = DEFAULT_EMBEDDER_MODEL_ID,
        revision: str | None = DEFAULT_EMBEDDER_REVISION,
        device: str | None = None,
        space_reference: EmbeddingSpaceReference = DEFAULT_EMBEDDING_SPACE,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        max_concurrency: int = 1,
    ) -> JinaEmbedder:
        """Load a pinned upstream model without exposing training operations.

        `revision` defaults to the pin for `DEFAULT_EMBEDDER_MODEL_ID`, so the safe thing is
        what a caller gets for free. `None` means "resolve the repository's default branch"
        and exists for the one caller that sweeps arbitrary repositories, where a pin for a
        different repository could not resolve. Anything loading the bundled model in a
        deployment goes through `create_embedder`, which always supplies a pin.
        """
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
        # `code_revision` pins the remote code separately from the weights, and it has to be
        # set on both: `sentence_transformers` forwards one to the module implementation and
        # the other to the config. Under `trust_remote_code=True` this is what decides which
        # third-party Python this process executes.
        encoder = sentence_transformer(
            model_path,
            revision=revision,
            trust_remote_code=True,
            device=selected_device,
            model_kwargs={"modality": "omni", "code_revision": revision},
            config_kwargs={"code_revision": revision},
        )
        if _media_processor_missing(encoder):
            raise ModelUnavailableError(_MISSING_PROCESSOR)
        return cls(
            encoder,
            ModelReference(model_id=model_id),
            space_reference=space_reference,
            dimension=dimension,
            max_concurrency=max_concurrency,
        )

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        """Declare the search space this model's vectors belong to."""
        return self._space_reference

    @operation_span("mindbridge.model.embed")
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        """Encode a homogeneous query or document batch."""
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
                cast(list[object], [_jina_input(item) for item in inputs]),
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self._dimension,
            )
        vectors = tuple(tuple(float(value) for value in row) for row in matrix.tolist())
        if len(vectors) != len(inputs):
            raise ModelOutputError("embedding batch size does not match its inputs")
        for vector in vectors:
            validate_embedding_vector(vector, self._dimension)
        return vectors


class _EmbedderConfig(PluginConfigModel):
    model_id: PluginText = DEFAULT_EMBEDDER_MODEL_ID
    # Spelled `model_revision` because that is the name a deployment configured before
    # migration 0021 and the value still means the same thing here. `PluginConfigModel`
    # ignores that name where nothing declares it, so an operator's existing
    # MINDBRIDGE_MEDIA_EMBEDDER_CONFIG_JSON keeps pinning what it always pinned instead of
    # being silently replaced by this default.
    model_revision: PluginText = DEFAULT_EMBEDDER_REVISION
    device: PluginText | None = None
    space_id: PluginText = DEFAULT_EMBEDDING_SPACE.space_id
    dimension: MatryoshkaDimension = DEFAULT_EMBEDDING_DIMENSION
    max_concurrency: PluginInteger = 1


def create_embedder(config: Mapping[str, object]) -> JinaEmbedder:
    """Entry-point factory for the bundled local Jina model."""
    validated = _EmbedderConfig.model_validate(config)
    return JinaEmbedder.load(
        model_id=validated.model_id,
        revision=validated.model_revision,
        device=validated.device,
        space_reference=EmbeddingSpaceReference(space_id=validated.space_id),
        dimension=validated.dimension,
        max_concurrency=validated.max_concurrency,
    )


def _jina_input(input_value: ModelInput) -> str | tuple[str, ...]:
    values = tuple(
        part.text if isinstance(part, TextPart) else part.url for part in input_value.parts
    )
    return values[0] if len(values) == 1 else values
