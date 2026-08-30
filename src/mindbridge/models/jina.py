"""Pinned adapter for Jina v5 Omni's official Sentence Transformers contract."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from importlib import import_module
from importlib.util import find_spec
from types import MethodType
from typing import cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.models.sentence_transformers import (
    _ST_LOCK,
    SentenceTransformersEmbedder,
    _optional_text,
    _parts,
    _positive_integer,
    _recipe_space,
    _SentenceEncoder,
    _SentenceTransformerFactory,
)
from mindbridge.types import Modality

DEFAULT_JINA_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_JINA_REVISION = "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4"
DEFAULT_JINA_DIMENSION = 1024
_JINA_RECIPE = "jina-v5-omni-official-sentence-transformers-v4"
_JINA_LEGACY_RECIPES = frozenset({"jina-v5-omni-official-sentence-transformers-v3"})
_JINA_CAPABILITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
_JINA_DIMENSIONS = frozenset({32, 64, 128, 256, 512, 1024})
_ENCODE_METHODS = ("encode", "encode_query", "encode_document")
_jina_methods: dict[str, Callable[..., object]] | None = None


class _Text:
    """Keep application text out of Jina's URL/path media autodetection."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class JinaOmniEmbedder:
    """Deferred default backend for the pinned Jina Omni embedding model."""

    __slots__ = (
        "_backend",
        "_batch_size",
        "_closed",
        "_device",
        "_dimension",
        "_lock",
        "_space_id",
    )

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_JINA_DIMENSION,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._dimension = _jina_dimension(dimension)
        self._device = _optional_text(device, "device")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        _require_local_extra()
        self._space_id = _recipe_space(
            _JINA_RECIPE,
            DEFAULT_JINA_MODEL_ID,
            DEFAULT_JINA_REVISION,
            self._dimension,
        )
        self._lock = threading.Lock()
        self._backend: _LoadedJinaOmniEmbedder | None = None
        self._closed = False

    @classmethod
    def load(
        cls,
        *,
        dimension: int = DEFAULT_JINA_DIMENSION,
        device: str | None = None,
        batch_size: int = 32,
    ) -> JinaOmniEmbedder:
        """Construct the public backend and load its pinned weights immediately."""
        backend = cls(dimension=dimension, device=device, batch_size=batch_size)
        with backend._lock:
            backend._ensure_loaded()
        return backend

    @property
    def embedding_capabilities(self) -> frozenset[Modality]:
        return _JINA_CAPABILITIES

    @property
    def embedding_model(self) -> str:
        return DEFAULT_JINA_MODEL_ID

    @property
    def embedding_space(self) -> str:
        return self._space_id

    @property
    def _legacy_embedding_spaces(self) -> frozenset[str]:
        return frozenset(
            _recipe_space(
                recipe,
                DEFAULT_JINA_MODEL_ID,
                DEFAULT_JINA_REVISION,
                self._dimension,
            )
            for recipe in _JINA_LEGACY_RECIPES
        )

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch, selected_task = _request(inputs, task)
        with self._lock:
            if self._closed:
                raise ModelError("embedding backend is closed")
            if not batch:
                return ()
            backend = self._ensure_loaded()
        return backend.embed(batch, selected_task)

    def close(self) -> None:
        """Close loaded weights without turning an unused backend into a model load."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._backend is not None:
                self._backend.close()

    def _ensure_loaded(self) -> _LoadedJinaOmniEmbedder:
        if self._backend is None:
            self._backend = _load_jina(
                dimension=self._dimension,
                device=self._device,
                batch_size=self._batch_size,
            )
        return self._backend


class _LoadedJinaOmniEmbedder(SentenceTransformersEmbedder):
    """Loaded Jina encoder using its official Sentence Transformers methods."""

    _input_recipe = _JINA_RECIPE

    def __init__(
        self,
        encoder: _SentenceEncoder,
        *,
        dimension: int,
        batch_size: int = 32,
    ) -> None:
        if _media_processor_missing(encoder):
            raise ModelError(
                "Jina Omni's media processor is unavailable; install MindBridge with the "
                "local extra"
            )
        video_processor = getattr(encoder[0].processor, "video_processor", None)  # type: ignore[index]
        if video_processor is not None and hasattr(video_processor, "cap_pixels_per_frame"):
            video_processor.cap_pixels_per_frame = True
        super().__init__(
            encoder,
            model_id=DEFAULT_JINA_MODEL_ID,
            revision=DEFAULT_JINA_REVISION,
            dimension=dimension,
            batch_size=batch_size,
        )

    def _discover_capabilities(self) -> frozenset[Modality]:
        return _JINA_CAPABILITIES

    def _prepare(self, value: ModelInput) -> object:
        values = tuple(
            _Text(part) if modality == Modality.TEXT.value else part
            for modality, part in _parts(value)
        )
        return values[0] if len(values) == 1 else values


def _load_jina(
    *,
    dimension: int,
    device: str | None,
    batch_size: int,
) -> _LoadedJinaOmniEmbedder:
    try:
        module = import_module("sentence_transformers")
        import_module("librosa")
        factory = cast(_SentenceTransformerFactory, module.SentenceTransformer)
    except (AttributeError, ImportError):
        raise ModelError(
            "Jina Omni is unavailable; install MindBridge with the local extra"
        ) from None
    try:
        code_kwargs = {"code_revision": DEFAULT_JINA_REVISION}
        sentence_transformer = module.SentenceTransformer
        with _ST_LOCK:
            original = {
                name: cast(Callable[..., object], getattr(sentence_transformer, name))
                for name in _ENCODE_METHODS
            }
            if getattr(original["encode"], "_omni_audio_patched", False):
                raise ModelError("Sentence Transformers was modified before Jina loaded")
            try:
                encoder = factory(
                    DEFAULT_JINA_MODEL_ID,
                    revision=DEFAULT_JINA_REVISION,
                    trust_remote_code=True,
                    device=device,
                    model_kwargs=code_kwargs,
                    config_kwargs=code_kwargs,
                )
                _bind_jina_methods(encoder, sentence_transformer, original)
            finally:
                for name, method in original.items():
                    setattr(sentence_transformer, name, method)
    except ModelError:
        raise
    except Exception:
        raise ModelError("failed to load the pinned Jina Omni model") from None
    return _LoadedJinaOmniEmbedder(
        encoder,
        dimension=dimension,
        batch_size=batch_size,
    )


def _media_processor_missing(encoder: object) -> bool:
    try:
        first = encoder[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return True
    return getattr(first, "processor", None) is None


def _request(
    inputs: Sequence[ModelInput],
    task: EmbedTask,
) -> tuple[tuple[ModelInput, ...], EmbedTask]:
    if isinstance(inputs, (str, bytes)):
        raise ValidationError("inputs must be a sequence of ModelInput values")
    try:
        batch = tuple(inputs)
    except TypeError:
        raise ValidationError("inputs must be a sequence of ModelInput values") from None
    if any(not isinstance(value, ModelInput) for value in batch):
        raise ValidationError("inputs must be a sequence of ModelInput values")
    try:
        selected_task = EmbedTask(task)
    except (TypeError, ValueError):
        raise ValidationError("embedding task is invalid") from None
    return batch, selected_task


def _jina_dimension(value: object) -> int:
    dimension = _positive_integer(value, "dimension")
    if dimension not in _JINA_DIMENSIONS:
        choices = ", ".join(str(item) for item in sorted(_JINA_DIMENSIONS))
        raise ValidationError(f"dimension must be one of: {choices}")
    return dimension


def _require_local_extra() -> None:
    try:
        missing = next(
            (
                module
                for module in ("sentence_transformers", "librosa")
                if find_spec(module) is None
            ),
            None,
        )
    except (ImportError, ValueError):
        missing = "local dependencies"
    if missing is not None:
        raise ModelError("Jina Omni is unavailable; install MindBridge with the local extra")


def _bind_jina_methods(
    encoder: _SentenceEncoder,
    sentence_transformer: object,
    original: dict[str, Callable[..., object]],
) -> None:
    global _jina_methods
    patched = {
        name: cast(Callable[..., object], getattr(sentence_transformer, name))
        for name in _ENCODE_METHODS
    }
    if any(patched[name] is not original[name] for name in _ENCODE_METHODS):
        _jina_methods = patched
    if _jina_methods is None:
        raise ModelError("the pinned Jina model did not install its embedding methods")
    for name, method in _jina_methods.items():
        setattr(encoder, name, MethodType(method, encoder))
