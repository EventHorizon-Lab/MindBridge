"""Pinned adapter for Jina v5 Omni's legacy Sentence Transformers input contract."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from types import MethodType
from typing import Protocol, cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.models.sentence_transformers import (
    _ST_LOCK,
    SentenceTransformersEmbedder,
    _EmbeddingMatrix,
    _optional_text,
    _parts,
    _positive_integer,
    _recipe_space,
    _SentenceEncoder,
    _SentenceTransformerFactory,
    _vectors,
)
from mindbridge.types import Modality

DEFAULT_JINA_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_JINA_REVISION = "1f9ba7a04283c80cecdfe7a98f9d9c6f09796ffb"
DEFAULT_JINA_DIMENSION = 1024
_JINA_RECIPE = "jina-v5-omni-typed-text-tuple-v2"
_JINA_CAPABILITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
_JINA_DIMENSIONS = frozenset({32, 64, 128, 256, 512, 1024})
_ENCODE_METHODS = ("encode", "encode_query", "encode_document")
_DEFAULT_BATCH_WAIT_MS = 2.0
_jina_methods: dict[str, Callable[..., object]] | None = None


class _Text:
    """Keep application text out of Jina's URL/path media autodetection."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _TextEncoder(Protocol):
    def __call__(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...


@dataclass(slots=True)
class _EmbeddingRequest:
    inputs: tuple[ModelInput, ...]
    task: EmbedTask
    done: threading.Event = field(default_factory=threading.Event)
    result: tuple[tuple[float, ...], ...] | None = None
    error: BaseException | None = None


class JinaOmniEmbedder:
    """Deferred default backend for the pinned Jina Omni embedding model."""

    __slots__ = (
        "_backend",
        "_batch_size",
        "_batch_wait_seconds",
        "_closed",
        "_closing",
        "_condition",
        "_device",
        "_dimension",
        "_pending",
        "_space_id",
        "_worker",
    )

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_JINA_DIMENSION,
        device: str | None = None,
        batch_size: int = 32,
        batch_wait_ms: float = _DEFAULT_BATCH_WAIT_MS,
    ) -> None:
        self._dimension = _jina_dimension(dimension)
        self._device = _optional_text(device, "device")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._batch_wait_seconds = _batch_wait_ms(batch_wait_ms) / 1_000
        _require_local_extra()
        self._space_id = _recipe_space(
            _JINA_RECIPE,
            DEFAULT_JINA_MODEL_ID,
            DEFAULT_JINA_REVISION,
            self._dimension,
        )
        self._condition = threading.Condition()
        self._pending: list[_EmbeddingRequest] = []
        self._worker: threading.Thread | None = None
        self._backend: _LoadedJinaOmniEmbedder | None = None
        self._closing = False
        self._closed = False

    @classmethod
    def load(
        cls,
        *,
        dimension: int = DEFAULT_JINA_DIMENSION,
        device: str | None = None,
        batch_size: int = 32,
        batch_wait_ms: float = _DEFAULT_BATCH_WAIT_MS,
    ) -> JinaOmniEmbedder:
        """Construct the public backend and load its pinned weights immediately."""
        backend = cls(
            dimension=dimension,
            device=device,
            batch_size=batch_size,
            batch_wait_ms=batch_wait_ms,
        )
        with backend._condition:
            backend._ensure_loaded()
        return backend

    @property
    def capabilities(self) -> frozenset[Modality]:
        return _JINA_CAPABILITIES

    @property
    def model_id(self) -> str:
        return DEFAULT_JINA_MODEL_ID

    @property
    def space_id(self) -> str:
        return self._space_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch, selected_task = _request(inputs, task)
        with self._condition:
            if self._closing or self._closed:
                raise ModelError("embedding backend is closed")
            if not batch:
                return ()
            request = _EmbeddingRequest(batch, selected_task)
            self._pending.append(request)
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run,
                    name="mindbridge-jina-batcher",
                    daemon=True,
                )
                self._worker.start()
            self._condition.notify()
        request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise ModelError("embedding model failed")
        return request.result

    def close(self) -> None:
        """Close loaded weights without turning an unused backend into a model load."""
        with self._condition:
            if self._closed:
                return
            if self._closing:
                while self._closing:
                    self._condition.wait()
                return
            self._closing = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            with self._condition:
                self._closed = True
                self._closing = False
                self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if not self._pending:
                    return
                task = self._pending[0].task
                deadline = time.monotonic() + self._batch_wait_seconds
                while not self._closing and self._pending_size(task) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                requests = self._take_batch(task)
            try:
                backend = self._ensure_loaded()
                vectors = backend.embed(
                    tuple(value for request in requests for value in request.inputs),
                    task,
                )
                offset = 0
                for request in requests:
                    end = offset + len(request.inputs)
                    request.result = vectors[offset:end]
                    offset = end
            except BaseException as error:
                for request in requests:
                    request.error = error
            finally:
                for request in requests:
                    request.done.set()

    def _pending_size(self, task: EmbedTask) -> int:
        return sum(len(request.inputs) for request in self._pending if request.task is task)

    def _take_batch(self, task: EmbedTask) -> list[_EmbeddingRequest]:
        selected: list[_EmbeddingRequest] = []
        remaining: list[_EmbeddingRequest] = []
        size = 0
        for request in self._pending:
            request_size = len(request.inputs)
            if request.task is task and (not selected or size + request_size <= self._batch_size):
                selected.append(request)
                size += request_size
            else:
                remaining.append(request)
        self._pending = remaining
        return selected

    def _ensure_loaded(self) -> _LoadedJinaOmniEmbedder:
        if self._backend is None:
            self._backend = _load_jina(
                dimension=self._dimension,
                device=self._device,
                batch_size=self._batch_size,
            )
        return self._backend


class _LoadedJinaOmniEmbedder(SentenceTransformersEmbedder):
    """Loaded Jina encoder with its legacy tuple methods isolated to one instance."""

    _input_recipe = _JINA_RECIPE

    def __init__(
        self,
        encoder: _SentenceEncoder,
        *,
        text_encode: _TextEncoder,
        dimension: int,
        batch_size: int = 32,
    ) -> None:
        if _media_processor_missing(encoder):
            raise ModelError(
                "Jina Omni's media processor is unavailable; install MindBridge with the "
                "local extra"
            )
        super().__init__(
            encoder,
            model_id=DEFAULT_JINA_MODEL_ID,
            revision=DEFAULT_JINA_REVISION,
            dimension=dimension,
            batch_size=batch_size,
        )
        self._text_encode = text_encode

    def _discover_capabilities(self) -> frozenset[Modality]:
        return _JINA_CAPABILITIES

    def _prepare(self, value: ModelInput) -> object:
        values = tuple(
            _Text(part) if modality == Modality.TEXT.value else part
            for modality, part in _parts(value)
        )
        return values[0] if len(values) == 1 else values

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch, selected_task = _request(inputs, task)
        if not batch:
            return super().embed(batch, selected_task)
        text = tuple((index, value) for index, value in enumerate(batch) if not value.assets)
        media = tuple((index, value) for index, value in enumerate(batch) if value.assets)
        vectors: dict[int, tuple[float, ...]] = {}
        if text:
            prefix = "Query: " if selected_task is EmbedTask.QUERY else "Document: "
            prepared: list[object] = [{"text": prefix + value.text} for _, value in text]
            with self._lock:
                if self._closed:
                    raise ModelError("embedding backend is closed")
                try:
                    matrix = self._text_encode(
                        prepared,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        truncate_dim=self._truncate_dim,
                        batch_size=self._batch_size,
                    )
                except Exception:
                    raise ModelError("embedding model failed") from None
            vectors.update(
                zip(
                    (index for index, _ in text),
                    _vectors(matrix, len(text), self._dimension),
                    strict=True,
                )
            )
        if media:
            media_vectors = super().embed(
                tuple(value for _, value in media),
                selected_task,
            )
            vectors.update(zip((index for index, _ in media), media_vectors, strict=True))
        return tuple(vectors[index] for index in range(len(batch)))


def _load_jina(
    *,
    dimension: int,
    device: str | None,
    batch_size: int,
) -> _LoadedJinaOmniEmbedder:
    try:
        module = import_module("sentence_transformers")
        hub = import_module("huggingface_hub")
        import_module("librosa")
        factory = cast(_SentenceTransformerFactory, module.SentenceTransformer)
        download = cast(Callable[..., str], hub.snapshot_download)
    except (AttributeError, ImportError):
        raise ModelError(
            "Jina Omni is unavailable; install MindBridge with the local extra"
        ) from None
    try:
        model_path = download(
            repo_id=DEFAULT_JINA_MODEL_ID,
            revision=DEFAULT_JINA_REVISION,
        )
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
                    model_path,
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
    text_encode = cast(_TextEncoder, MethodType(original["encode"], encoder))
    return _LoadedJinaOmniEmbedder(
        encoder,
        text_encode=text_encode,
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


def _batch_wait_ms(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 100
    ):
        raise ValidationError("batch_wait_ms must be between zero and 100")
    return float(value)


def _require_local_extra() -> None:
    try:
        missing = next(
            (
                module
                for module in ("sentence_transformers", "huggingface_hub", "librosa")
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
