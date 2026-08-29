"""Dependency-light boundary around Sentence Transformers embedding models."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Protocol, cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import AssetRef, Modality

_ATOMIC_MODALITIES = (Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
_STANDARD_RECIPE = "sentence-transformers-standard-input-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
# Some trust-remote-code models mutate shared Sentence Transformers state while importing.
_ST_LOCK = threading.RLock()


class _EmbeddingMatrix(Protocol):
    def tolist(self) -> object: ...


class _SentenceEncoder(Protocol):
    def supports(self, modality: str | tuple[str, ...]) -> bool: ...

    def get_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...

    def encode_document(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(self, model_name_or_path: str, **kwargs: object) -> _SentenceEncoder: ...


class SentenceTransformersEmbedder:
    """Synchronous query/document encoder using standard ST dict and message inputs."""

    _input_recipe = _STANDARD_RECIPE

    def __init__(
        self,
        encoder: _SentenceEncoder,
        *,
        model_id: str,
        revision: str,
        dimension: int | None = None,
        batch_size: int = 32,
    ) -> None:
        with _ST_LOCK:
            self._encoder = encoder
            encode = getattr(encoder, "encode", None)
            if callable(encode):
                object.__setattr__(encoder, "encode", encode)
            self._encode_query = encoder.encode_query
            self._encode_document = encoder.encode_document
            self._model_id = _text(model_id, "model_id")
            self._revision = _revision(revision)
            self._dimension, self._truncate_dim = _dimensions(encoder, dimension)
            self._batch_size = _positive_integer(batch_size, "batch_size")
            self._capabilities = self._discover_capabilities()
        self._space_id = _recipe_space(
            self._input_recipe,
            self._model_id,
            self._revision,
            self._dimension,
        )
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str,
        dimension: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> SentenceTransformersEmbedder:
        """Load a standard model at an immutable revision."""
        model_id = _text(model_id, "model_id")
        revision = _revision(revision)
        device = _optional_text(device, "device")
        batch_size = _positive_integer(batch_size, "batch_size")
        if dimension is not None:
            _positive_integer(dimension, "dimension")
        with _ST_LOCK:
            encoder = _load_encoder(
                model_id,
                revision=revision,
                device=device,
            )
            return cls(
                encoder,
                model_id=model_id,
                revision=revision,
                dimension=dimension,
                batch_size=batch_size,
            )

    @property
    def embedding_capabilities(self) -> frozenset[Modality]:
        return self._capabilities

    @property
    def embedding_model(self) -> str:
        return self._model_id

    @property
    def embedding_space(self) -> str:
        return self._space_id

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode a batch through the retrieval-side method selected by ``task``."""
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

        prepared = [self._prepare(value) for value in batch]
        with self._lock:
            if self._closed:
                raise ModelError("embedding backend is closed")
            if not batch:
                return ()
            encode = (
                self._encode_query if selected_task is EmbedTask.QUERY else self._encode_document
            )
            try:
                matrix = encode(
                    prepared,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    truncate_dim=self._truncate_dim,
                    batch_size=self._batch_size,
                )
            except Exception:
                raise ModelError("embedding model failed") from None
        return _vectors(matrix, len(batch), self._dimension)

    def close(self) -> None:
        """Stop new calls and release an encoder-owned resource when it exposes one."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._encoder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    raise ModelError("failed to close embedding model") from None

    def _discover_capabilities(self) -> frozenset[Modality]:
        capabilities = frozenset(
            modality for modality in _ATOMIC_MODALITIES if self._supports(modality.value)
        )
        if not capabilities:
            raise ModelError("embedding model does not advertise a supported modality")
        return capabilities

    def _prepare(self, value: ModelInput) -> object:
        parts = _parts(value)
        modalities = tuple(modality for modality, _ in parts)
        for modality in set(modalities):
            if Modality(modality) not in self._capabilities:
                raise ModelError(f"embedding model does not support {modality} input")
        if len(parts) == 1:
            modality, part = parts[0]
            return part if modality == Modality.TEXT.value else {modality: part}

        compound = tuple(sorted(set(modalities)))
        if len(compound) == len(modalities) and self._supports(compound):
            return dict(parts)
        if not self._supports("message"):
            raise ModelError("embedding model cannot combine these input parts")
        return [
            {
                "role": "user",
                "content": [_message_part(modality, part) for modality, part in parts],
            }
        ]

    def _supports(self, modality: str | tuple[str, ...]) -> bool:
        try:
            supported = self._encoder.supports(modality)
        except Exception:
            return False
        return supported is True


def _load_encoder(
    model_id: str,
    *,
    revision: str,
    device: str | None,
) -> _SentenceEncoder:
    try:
        module = import_module("sentence_transformers")
        factory = cast(_SentenceTransformerFactory, module.SentenceTransformer)
    except (AttributeError, ImportError):
        raise ModelError(
            "Sentence Transformers is unavailable; install MindBridge with the local extra"
        ) from None

    try:
        return factory(
            model_id,
            revision=revision,
            trust_remote_code=False,
            device=device,
        )
    except Exception:
        raise ModelError("failed to load the embedding model") from None


def _dimensions(encoder: _SentenceEncoder, requested: int | None) -> tuple[int, int | None]:
    try:
        native = encoder.get_embedding_dimension()
    except Exception:
        raise ModelError("embedding model does not declare its native dimension") from None
    if isinstance(native, bool) or not isinstance(native, int) or native <= 0:
        raise ModelError("embedding model does not declare its native dimension")
    if requested is None:
        return native, None
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValidationError("dimension must be a positive integer")
    if requested == native:
        return requested, None
    advertised = _config_value(encoder, "matryoshka_dimensions")
    matryoshka = _config_value(encoder, "is_matryoshka")
    if (
        requested > native
        or matryoshka is not True
        or not isinstance(advertised, (list, tuple))
        or any(isinstance(value, bool) or not isinstance(value, int) for value in advertised)
        or requested not in advertised
    ):
        raise ModelError(
            f"model native dimension is {native}; requested dimension {requested} is not an "
            "advertised Matryoshka dimension"
        )
    return requested, requested


def _config_value(encoder: object, name: str) -> object:
    first = _first_module(encoder)
    candidates = (
        getattr(encoder, "config", None),
        getattr(first, "config", None),
        getattr(getattr(first, "auto_model", None), "config", None),
        getattr(getattr(first, "model", None), "config", None),
    )
    for config in candidates:
        for source in (config, getattr(config, "text_config", None)):
            if source is not None and hasattr(source, name):
                return getattr(source, name)
    return None


def _first_module(encoder: object) -> object | None:
    try:
        return cast(object, encoder[0])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None


def _parts(value: ModelInput) -> tuple[tuple[str, str], ...]:
    parts: list[tuple[str, str]] = []
    if value.text:
        parts.append((Modality.TEXT.value, value.text))
    parts.extend((_asset_modality(asset), _asset_path(asset)) for asset in value.assets)
    return tuple(parts)


def _asset_modality(asset: AssetRef) -> str:
    modality = asset.modality
    if modality is None:  # ModelInput already enforces resolved assets.
        raise ValidationError("model input assets must be resolved")
    return modality.value


def _asset_path(asset: AssetRef) -> str:
    path = asset.path
    if path is None:  # ModelInput already enforces resolved assets.
        raise ValidationError("model input assets must be resolved")
    return str(path)


def _message_part(modality: str, value: str) -> dict[str, str]:
    return {"type": modality, "text" if modality == Modality.TEXT.value else modality: value}


def _vectors(
    matrix: _EmbeddingMatrix,
    expected_count: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        rows = matrix.tolist()
    except Exception:
        raise ModelError("embedding model returned an invalid response") from None
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ModelError("embedding model returned an invalid response")
    vectors = tuple(_normalized(row, dimension) for row in rows)
    return vectors


def _normalized(row: object, dimension: int) -> tuple[float, ...]:
    if isinstance(row, (str, bytes)):
        raise ModelError("embedding model returned an invalid response")
    if not isinstance(row, Iterable):
        raise ModelError("embedding model returned an invalid response") from None
    raw: tuple[object, ...] = tuple(cast(Iterable[object], row))
    if len(raw) != dimension or any(
        isinstance(value, bool) or not isinstance(value, (float, int)) for value in raw
    ):
        raise ModelError("embedding model returned an invalid response")
    try:
        values = tuple(float(cast(float | int, value)) for value in raw)
    except OverflowError:
        raise ModelError("embedding model returned an invalid response") from None
    if any(not math.isfinite(value) for value in values):
        raise ModelError("embedding model returned an invalid response")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ModelError("embedding model returned an invalid response")
    return tuple(value / norm for value in values)


def _recipe_space(recipe: str, model_id: str, revision: str, dimension: int) -> str:
    payload = json.dumps(
        {
            "dimension": dimension,
            "document": "encode_document",
            "model": model_id,
            "normalize": True,
            "query": "encode_query",
            "recipe": recipe,
            "revision": revision,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sentence-transformers:{digest}"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()


def _optional_text(value: object | None, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _revision(value: object) -> str:
    normalized = _text(value, "revision").lower()
    if _COMMIT.fullmatch(normalized) is None:
        raise ValidationError("revision must be an immutable 40-character commit hash")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value
