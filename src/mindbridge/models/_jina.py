"""Internal input and loading adaptations for the pinned Jina Omni checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, cast

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models._media import fetch_videos as _fetch_videos
from mindbridge.types import Modality

if TYPE_CHECKING:
    from mindbridge.models.sentence_transformers import (
        _SentenceEncoder,
        _SentenceTransformerFactory,
    )

DEFAULT_JINA_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_JINA_REVISION = "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4"
DEFAULT_JINA_DIMENSION = 1024
_JINA_RECIPE = "jina-v5-omni-official-sentence-transformers-v6"
_JINA_LEGACY_RECIPES = frozenset(
    {
        "jina-v5-omni-official-sentence-transformers-v3",
        "jina-v5-omni-official-sentence-transformers-v4",
        "jina-v5-omni-official-sentence-transformers-v5",
    }
)
_JINA_CAPABILITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
_JINA_DIMENSIONS = frozenset({32, 64, 128, 256, 512, 1024})
_JINA_VIDEO_FRAMES = 32
_ENCODE_METHODS = ("encode", "encode_query", "encode_document")
_jina_methods: dict[str, Callable[..., object]] | None = None


class _Text:
    """Keep application text out of Jina's URL/path media autodetection."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


def _load_jina(
    *,
    device: str | None,
) -> _SentenceEncoder:
    try:
        module = import_module("sentence_transformers")
        import_module("librosa")
        factory = cast("_SentenceTransformerFactory", module.SentenceTransformer)
    except (AttributeError, ImportError):
        raise ModelError(
            "Jina Omni is unavailable; install MindBridge with the local extra"
        ) from None
    try:
        code_kwargs = {"code_revision": DEFAULT_JINA_REVISION}
        sentence_transformer = module.SentenceTransformer
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
    return encoder


def _media_processor_missing(encoder: object) -> bool:
    try:
        first = encoder[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return True
    return getattr(first, "processor", None) is None


def _configure_jina_video(module: object) -> None:
    processor = getattr(module, "processor", None)
    video_processor = getattr(processor, "video_processor", None)
    composite = getattr(module, "_encode_composite_parts", None)
    required = ("cap_pixels_per_frame", "do_sample_frames", "fps", "max_frames", "num_frames")
    if (
        video_processor is None
        or any(not hasattr(video_processor, name) for name in required)
        or not isinstance(composite, MethodType)
        or composite.__self__ is not module
        or "_eval_video_frames" not in composite.__func__.__code__.co_names
        or not callable(composite.__func__.__globals__.get("_eval_video_frames"))
    ):
        raise ModelError("the pinned Jina video integration is incompatible")

    provider_sample_frames = getattr(video_processor, "sample_frames", None)
    fetch_videos = getattr(video_processor, "fetch_videos", None)
    if (
        not isinstance(provider_sample_frames, MethodType)
        or provider_sample_frames.__self__ is not video_processor
        or not isinstance(fetch_videos, MethodType)
        or fetch_videos.__self__ is not video_processor
    ):
        raise ModelError("the pinned Jina video integration is incompatible")
    load_video = fetch_videos.__func__.__globals__.get("load_video")
    if not callable(load_video):
        raise ModelError("the pinned Jina video integration is incompatible")

    source = composite.__func__
    isolated_globals = dict(source.__globals__)
    isolated_globals["_eval_video_frames"] = _video_path
    isolated = FunctionType(
        source.__code__,
        isolated_globals,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    isolated.__kwdefaults__ = source.__kwdefaults__

    def bounded_sample_frames(
        _processor: object,
        metadata: object,
        **kwargs: object,
    ) -> object:
        del kwargs
        total = getattr(metadata, "total_num_frames", None)
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ModelError("video decoder returned invalid frame metadata")
        import numpy as np

        return np.linspace(0, total - 1, min(total, _JINA_VIDEO_FRAMES), dtype=int)

    def pyav_fetch_videos(
        _processor: object,
        videos: object,
        sample_indices_fn: Callable[..., object] | None = None,
    ) -> object:
        return _fetch_videos(load_video, videos, sample_indices_fn)

    try:
        setattr(module, "_encode_composite_parts", MethodType(isolated, module))  # noqa: B010
        settings = {
            "cap_pixels_per_frame": True,
            "do_sample_frames": True,
            "fps": None,
            "max_frames": _JINA_VIDEO_FRAMES,
            "num_frames": _JINA_VIDEO_FRAMES,
            "sample_frames": MethodType(bounded_sample_frames, video_processor),
            "fetch_videos": MethodType(pyav_fetch_videos, video_processor),
        }
        for name, value in settings.items():
            setattr(video_processor, name, value)
    except (AttributeError, TypeError, ValueError):
        raise ModelError("the pinned Jina video integration is incompatible") from None


def _video_path(value: object) -> object:
    return value


def _jina_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("dimension must be a positive integer")
    dimension = value
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
