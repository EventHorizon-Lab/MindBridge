"""Jina v5 Omni embedding through the official Sentence Transformers API."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol, cast

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import EmbeddingInput
from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
    PixelRegion,
)
from mindbridge.models.compute import select_torch_device
from mindbridge.telemetry import operation_span, set_current_span_attributes, trace_operation

DEFAULT_JINA_OMNI_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_JINA_OMNI_REVISION = "12949877f0092093f366c6450340011320152a05"
DEFAULT_JINA_TEXT_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"
DEFAULT_JINA_TEXT_REVISION = "6856e76bb72982e58de0620458a4e8b3614da340"
DEFAULT_JINA_OMNI_DIMENSION = 1_024
JINA_EVENT_MEDIA_PREPROCESSING_REVISION = "event-media-v1"
_EVENT_VIDEO_FPS = 2.0
_EVENT_VIDEO_MIN_FRAMES = 4
_EVENT_VIDEO_MAX_FRAMES = 32
_EVENT_VIDEO_MAX_TOTAL_PIXELS = 6_422_528
_EVENT_AUDIO_SAMPLE_RATE = 16_000
_EVENT_VIDEO_METADATA_KEY = "mindbridge_event_video"
DEFAULT_JINA_RETRIEVAL_SPACE = EmbeddingSpaceReference(
    space_id="jinaai/jina-embeddings-v5-small-retrieval-1024",
    revision=(
        "omni@12949877f0092093f366c6450340011320152a05+"
        "text@6856e76bb72982e58de0620458a4e8b3614da340"
    ),
)


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


class JinaOmniEmbedder:
    """Async-safe query/document encoder for text, image, video, and audio."""

    def __init__(
        self,
        encoder: _SentenceEncoder,
        model_reference: ModelReference,
        *,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        max_concurrency: int = 1,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        _install_event_video_metadata(encoder)
        self._encoder = encoder
        self._model_reference = ModelReference(
            model_id=model_reference.model_id,
            revision=(f"{model_reference.revision}+{JINA_EVENT_MEDIA_PREPROCESSING_REVISION}"),
        )
        self._space_reference = space_reference
        self._dimension = dimension
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def load(
        cls,
        *,
        revision: str,
        model_id: str = DEFAULT_JINA_OMNI_MODEL_ID,
        device: str | None = None,
        space_reference: EmbeddingSpaceReference = DEFAULT_JINA_RETRIEVAL_SPACE,
        dimension: int = DEFAULT_JINA_OMNI_DIMENSION,
        max_concurrency: int = 1,
    ) -> JinaOmniEmbedder:
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

    @property
    def model_reference(self) -> ModelReference:
        """Return the exact model identity stored beside every vector."""
        return self._model_reference

    @property
    def dimension(self) -> int:
        """Return the configured Matryoshka output dimension."""
        return self._dimension

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        """Return the exact aligned space used for cross-model search."""
        return self._space_reference

    @trace_operation("mindbridge.model.encode_queries")
    async def encode_queries(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode retrieval queries with the upstream query prompt semantics."""
        return await self._encode(inputs, self._encoder.encode_query)

    @trace_operation("mindbridge.model.encode_documents")
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
        event_evidence = tuple(item for item in inputs if isinstance(item, ResolvedEvidence))
        set_current_span_attributes(
            {
                "mindbridge.model.id": self._model_reference.model_id,
                "mindbridge.model.revision": self._model_reference.revision,
                "mindbridge.embedding.dimension": self._dimension,
                "mindbridge.embedding.input_count": len(inputs),
                "mindbridge.embedding.event_media_count": len(event_evidence),
                "mindbridge.embedding.preprocessing_revision": (
                    JINA_EVENT_MEDIA_PREPROCESSING_REVISION
                ),
                "mindbridge.embedding.input_media_seconds": sum(
                    (item.evidence_span.end_ms - item.evidence_span.start_ms) / 1_000
                    for item in event_evidence
                ),
            }
        )
        with operation_span("mindbridge.model.prepare_embedding_inputs"):
            prepared_inputs = await asyncio.to_thread(_materialize_embedding_inputs, inputs)
        with operation_span("mindbridge.model.embedding_forward"):
            set_current_span_attributes(_prepared_media_attributes(prepared_inputs))
            async with self._semaphore:
                matrix = await asyncio.to_thread(
                    encode,
                    prepared_inputs,
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


def _materialize_embedding_inputs(inputs: tuple[EmbeddingInput, ...]) -> list[object]:
    prepared: list[object] = list(inputs)
    groups: dict[tuple[str, int | None], list[tuple[int, ResolvedEvidence]]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, ResolvedEvidence):
            continue
        if item.media_object.kind is MediaKind.IMAGE:
            prepared[index] = (
                item.media_url if item.evidence_span.region is None else _decode_event_image(item)
            )
            continue
        groups.setdefault(
            (item.media_url, item.evidence_span.audio_track),
            [],
        ).append((index, item))

    for group in groups.values():
        try:
            materialized = _decode_event_media(tuple(item for _, item in group))
        except (ModelRequestError, ModelUnavailableError):
            raise
        except Exception as error:
            raise ModelRequestError("event evidence media could not be decoded") from error
        for (index, _), value in zip(group, materialized, strict=True):
            prepared[index] = value
    return prepared


def _decode_event_image(evidence: ResolvedEvidence) -> object:
    try:
        av = cast(Any, import_module("av"))
        with av.open(evidence.media_url) as container:
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise ModelRequestError("event image contains no decodable frame")
            region = cast(PixelRegion, evidence.evidence_span.region)
            x_max = min(region.x_max, frame.width)
            y_max = min(region.y_max, frame.height)
            if region.x_min >= x_max or region.y_min >= y_max:
                raise ModelRequestError("event pixel region falls outside image")
            pixels = frame.to_ndarray(format="rgb24")
            return pixels[region.y_min : y_max, region.x_min : x_max].copy()
    except (ModelRequestError, ModelUnavailableError):
        raise
    except ImportError as error:
        raise ModelUnavailableError(
            "install MindBridge with the cloud-models extra to decode event images"
        ) from error
    except Exception as error:
        raise ModelRequestError("event evidence image could not be decoded") from error


def _prepared_media_attributes(inputs: list[object]) -> dict[str, int | float]:
    multimodal_items = 0
    video_frames = 0
    video_pixels = 0
    audio_samples = 0
    for item in inputs:
        has_media = False
        for part in item if isinstance(item, tuple) else (item,):
            shape = cast(tuple[int, ...], getattr(part, "shape", ()))
            if len(shape) == 4 and shape[-1] in {3, 4}:
                has_media = True
                video_frames += shape[0]
                video_pixels += shape[0] * shape[1] * shape[2]
            elif len(shape) == 3 and shape[-1] in {3, 4}:
                has_media = True
            elif len(shape) == 1:
                has_media = True
                audio_samples += shape[0]
        multimodal_items += has_media
    return {
        "mindbridge.embedding.multimodal_item_count": multimodal_items,
        "mindbridge.embedding.video_frame_count": video_frames,
        "mindbridge.embedding.video_pixel_count": video_pixels,
        "mindbridge.embedding.audio_seconds": audio_samples / _EVENT_AUDIO_SAMPLE_RATE,
    }


def _install_event_video_metadata(encoder: object) -> None:
    try:
        video_processor = cast(Any, encoder)[0].processor.video_processor
        original_preprocess = cast(Callable[..., object], video_processor.preprocess)
    except (AttributeError, KeyError, TypeError):
        return
    if getattr(video_processor, "_mindbridge_event_metadata", False):
        return

    def preprocess(videos: object, **kwargs: object) -> object:
        if kwargs.get("video_metadata") is None:
            metadata = _event_video_metadata(videos)
            if metadata is not None:
                kwargs["video_metadata"] = metadata
        return original_preprocess(videos, **kwargs)

    video_processor.preprocess = preprocess
    video_processor._mindbridge_event_metadata = True


def _event_video_metadata(videos: object) -> list[dict[str, object]] | None:
    items = videos if isinstance(videos, (list, tuple)) else (videos,)
    output: list[dict[str, object]] = []
    for item in items:
        dtype_metadata = getattr(getattr(item, "dtype", None), "metadata", None) or {}
        metadata = dtype_metadata.get(_EVENT_VIDEO_METADATA_KEY)
        shape = cast(tuple[int, ...], getattr(item, "shape", ()))
        if not isinstance(metadata, dict) or len(shape) != 4:
            return None
        output.append(
            {
                "total_num_frames": shape[0],
                "fps": metadata["fps"],
                "duration": metadata["duration"],
                "frames_indices": list(range(shape[0])),
                "height": shape[1],
                "width": shape[2],
            }
        )
    return output


def _decode_event_media(evidence: tuple[ResolvedEvidence, ...]) -> tuple[object, ...]:
    try:
        av = cast(Any, import_module("av"))
        numpy = cast(Any, import_module("numpy"))
        audio_module = cast(Any, import_module("av.audio.resampler"))
    except ImportError as error:
        raise ModelUnavailableError(
            "install MindBridge with the cloud-models extra to decode event media"
        ) from error

    video_indices = tuple(
        index for index, item in enumerate(evidence) if item.media_object.kind is MediaKind.VIDEO
    )
    audio_indices = tuple(
        index
        for index, item in enumerate(evidence)
        if item.media_object.kind in {MediaKind.VIDEO, MediaKind.AUDIO}
    )
    frame_targets_by_index = {
        index: _event_frame_targets(evidence[index])
        for index in video_indices
        if evidence[index].evidence_span.frame_start is not None
    }
    targets_by_index = {
        index: _event_video_targets(
            evidence[index].evidence_span.start_ms / 1_000,
            evidence[index].evidence_span.end_ms / 1_000,
        )
        for index in video_indices
        if index not in frame_targets_by_index
    }
    selected_video: dict[int, list[object | None]] = {
        index: [None]
        * len(
            frame_targets_by_index[index]
            if index in frame_targets_by_index
            else targets_by_index[index]
        )
        for index in video_indices
    }
    flat_targets = sorted(
        (
            target,
            index,
            position,
            evidence[index].evidence_span.start_ms / 1_000,
            evidence[index].evidence_span.end_ms / 1_000,
        )
        for index in video_indices
        if index in targets_by_index
        for position, target in enumerate(targets_by_index[index])
    )
    flat_frame_targets = sorted(
        (target, index, position)
        for index, targets in frame_targets_by_index.items()
        for position, target in enumerate(targets)
    )
    audio_parts: dict[int, list[object]] = {index: [] for index in audio_indices}
    target_position = 0
    frame_target_position = 0
    decoded_video_frames = 0
    previous_video: tuple[float, object] | None = None
    frame_cache: dict[tuple[object, object, int, PixelRegion | None], object] = {}
    decode_windows = tuple(_event_audio_window(item) for item in evidence)
    # ponytail: exact frame ordinals require a linear decode; use indexed derivatives for long media.
    decode_start_seconds = 0.0 if flat_frame_targets else min(start for start, _ in decode_windows)
    decode_end_seconds = max(end for _, end in decode_windows)

    with av.open(evidence[0].media_url) as container:
        origin_seconds = (container.start_time or 0) / av.time_base
        decode_streams, audio_resampler = _configure_event_decoder(
            container,
            evidence,
            video_indices,
            audio_indices,
            audio_module,
        )
        audio_cursor_seconds = _seek_event_media(
            container,
            av.time_base,
            origin_seconds,
            decode_start_seconds,
        )
        for frame in container.decode(**decode_streams):
            if isinstance(frame, av.VideoFrame):
                target_position, previous_video = _select_event_video_targets(
                    frame,
                    origin_seconds,
                    flat_targets,
                    target_position,
                    previous_video,
                    selected_video,
                    frame_cache,
                    evidence,
                )
                frame_target_position = _select_event_frame_targets(
                    frame,
                    decoded_video_frames,
                    flat_frame_targets,
                    frame_target_position,
                    selected_video,
                    frame_cache,
                    evidence,
                )
                decoded_video_frames += 1
            elif audio_resampler is not None:
                for resampled in cast(Any, audio_resampler).resample(frame):
                    audio_cursor_seconds = _append_event_audio(
                        resampled,
                        origin_seconds,
                        audio_cursor_seconds,
                        evidence,
                        audio_indices,
                        audio_parts,
                        numpy,
                    )
            if (
                target_position == len(flat_targets)
                and frame_target_position == len(flat_frame_targets)
                and (audio_resampler is None or audio_cursor_seconds >= decode_end_seconds)
            ):
                break

        if audio_resampler is not None:
            for resampled in cast(Any, audio_resampler).resample(None):
                audio_cursor_seconds = _append_event_audio(
                    resampled,
                    origin_seconds,
                    audio_cursor_seconds,
                    evidence,
                    audio_indices,
                    audio_parts,
                    numpy,
                )

    _fill_trailing_video_targets(
        flat_targets,
        target_position,
        previous_video,
        selected_video,
        frame_cache,
        evidence,
    )
    if frame_target_position != len(flat_frame_targets):
        raise ModelRequestError("event frame range exceeds decodable video frames")
    return _assemble_event_media(evidence, selected_video, audio_parts, numpy)


def _seek_event_media(
    container: object,
    time_base: int,
    origin_seconds: float,
    start_seconds: float,
) -> float:
    if start_seconds <= 0:
        return 0.0
    try:
        cast(Any, container).seek(
            round((origin_seconds + start_seconds) * time_base),
            backward=True,
            any_frame=False,
        )
    except (OSError, ValueError):
        return 0.0
    return start_seconds


def _configure_event_decoder(
    container: object,
    evidence: tuple[ResolvedEvidence, ...],
    video_indices: tuple[int, ...],
    audio_indices: tuple[int, ...],
    audio_module: object,
) -> tuple[dict[str, int], object | None]:
    source = cast(Any, container)
    decode_streams: dict[str, int] = {}
    if video_indices:
        if not source.streams.video:
            raise ModelRequestError("event video evidence has no video stream")
        source.streams.video[0].thread_type = "AUTO"
        decode_streams["video"] = 0

    audio_resampler = None
    if audio_indices and source.streams.audio:
        audio_track = evidence[audio_indices[0]].evidence_span.audio_track or 0
        if audio_track >= len(source.streams.audio):
            raise ModelRequestError("event evidence selects an unavailable audio track")
        decode_streams["audio"] = audio_track
        audio_resampler = cast(Any, audio_module).AudioResampler(
            format="flt",
            layout="mono",
            rate=_EVENT_AUDIO_SAMPLE_RATE,
        )
    elif any(evidence[index].media_object.kind is MediaKind.AUDIO for index in audio_indices):
        raise ModelRequestError("event audio evidence has no audio stream")
    return decode_streams, audio_resampler


def _select_event_video_targets(
    frame: object,
    origin_seconds: float,
    flat_targets: list[tuple[float, int, int, float, float]],
    target_position: int,
    previous_video: tuple[float, object] | None,
    selected_video: dict[int, list[object | None]],
    frame_cache: dict[tuple[object, object, int, PixelRegion | None], object],
    evidence: tuple[ResolvedEvidence, ...],
) -> tuple[int, tuple[float, object]]:
    video_frame = cast(Any, frame)
    frame_seconds = (
        float(video_frame.time) - origin_seconds if video_frame.time is not None else 0.0
    )
    while target_position < len(flat_targets) and flat_targets[target_position][0] <= frame_seconds:
        target, index, position, start_seconds, end_seconds = flat_targets[target_position]
        candidates = [item for item in (previous_video, (frame_seconds, frame)) if item is not None]
        inside = [
            item for item in candidates if _inside_window(item[0], start_seconds, end_seconds)
        ]
        selected = min(inside or candidates, key=lambda item: abs(item[0] - target))[1]
        selected_video[index][position] = _cached_video_frame(
            selected,
            len(selected_video[index]),
            frame_cache,
            evidence[index].evidence_span.region,
        )
        target_position += 1
    return target_position, (frame_seconds, frame)


def _select_event_frame_targets(
    frame: object,
    frame_number: int,
    flat_targets: list[tuple[int, int, int]],
    target_position: int,
    selected_video: dict[int, list[object | None]],
    frame_cache: dict[tuple[object, object, int, PixelRegion | None], object],
    evidence: tuple[ResolvedEvidence, ...],
) -> int:
    while target_position < len(flat_targets) and flat_targets[target_position][0] <= frame_number:
        _, index, position = flat_targets[target_position]
        selected_video[index][position] = _cached_video_frame(
            frame,
            len(selected_video[index]),
            frame_cache,
            evidence[index].evidence_span.region,
        )
        target_position += 1
    return target_position


def _append_event_audio(
    frame: object,
    origin_seconds: float,
    cursor_seconds: float,
    evidence: tuple[ResolvedEvidence, ...],
    audio_indices: tuple[int, ...],
    audio_parts: dict[int, list[object]],
    numpy: object,
) -> float:
    audio_frame = cast(Any, frame)
    np = cast(Any, numpy)
    samples = audio_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
    frame_start = (
        float(audio_frame.time) - origin_seconds if audio_frame.time is not None else cursor_seconds
    )
    frame_end = frame_start + len(samples) / _EVENT_AUDIO_SAMPLE_RATE
    for index in audio_indices:
        start_seconds, end_seconds = _event_audio_window(evidence[index])
        if frame_end <= start_seconds or frame_start >= end_seconds:
            continue
        sample_start = max(
            0,
            math.floor((max(start_seconds, frame_start) - frame_start) * _EVENT_AUDIO_SAMPLE_RATE),
        )
        sample_end = min(
            len(samples),
            math.ceil((min(end_seconds, frame_end) - frame_start) * _EVENT_AUDIO_SAMPLE_RATE),
        )
        if sample_end > sample_start:
            audio_parts[index].append(samples[sample_start:sample_end])
    return frame_end


def _fill_trailing_video_targets(
    flat_targets: list[tuple[float, int, int, float, float]],
    target_position: int,
    previous_video: tuple[float, object] | None,
    selected_video: dict[int, list[object | None]],
    frame_cache: dict[tuple[object, object, int, PixelRegion | None], object],
    evidence: tuple[ResolvedEvidence, ...],
) -> None:
    while target_position < len(flat_targets):
        _, index, position, _, _ = flat_targets[target_position]
        if previous_video is not None:
            selected_video[index][position] = _cached_video_frame(
                previous_video[1],
                len(selected_video[index]),
                frame_cache,
                evidence[index].evidence_span.region,
            )
        target_position += 1


def _assemble_event_media(
    evidence: tuple[ResolvedEvidence, ...],
    selected_video: dict[int, list[object | None]],
    audio_parts: dict[int, list[object]],
    numpy: object,
) -> tuple[object, ...]:
    np = cast(Any, numpy)
    output: list[object] = []
    for index, item in enumerate(evidence):
        video_frames = tuple(frame for frame in selected_video.get(index, ()) if frame is not None)
        video = None
        if video_frames:
            if item.evidence_span.duration_ms == 0 or len(video_frames) == 1:
                video = video_frames[0]
            else:
                duration_seconds = item.evidence_span.duration_ms / 1_000
                video = np.stack(video_frames)
                video = video.view(
                    np.dtype(
                        video.dtype,
                        metadata={
                            _EVENT_VIDEO_METADATA_KEY: {
                                "fps": (len(video_frames) - 1) / duration_seconds,
                                "duration": duration_seconds,
                            }
                        },
                    )
                )
        parts = audio_parts.get(index, ())
        audio = np.concatenate(parts).astype(np.float32, copy=False) if parts else None
        if item.media_object.kind is MediaKind.VIDEO and video is None:
            raise ModelRequestError("event video window contains no decodable frames")
        if item.media_object.kind is MediaKind.AUDIO and audio is None:
            raise ModelRequestError("event audio window contains no decodable samples")
        if video is not None and audio is not None:
            output.append((audio, video))
        else:
            output.append(video if video is not None else audio)
    return tuple(output)


def _event_video_targets(start_seconds: float, end_seconds: float) -> tuple[float, ...]:
    if end_seconds == start_seconds:
        return (start_seconds,)
    count = _event_video_target_count(start_seconds, end_seconds)
    step = (end_seconds - start_seconds) / (count - 1)
    return tuple(start_seconds + position * step for position in range(count))


def _event_frame_targets(item: ResolvedEvidence) -> tuple[int, ...]:
    span = item.evidence_span
    if span.frame_start is None or span.frame_end is None:
        raise ValueError("event frame targets require a complete frame range")
    available_frames = span.frame_end - span.frame_start + 1
    if available_frames == 1:
        return (span.frame_start,)
    count = min(
        available_frames,
        _event_video_target_count(span.start_ms / 1_000, span.end_ms / 1_000),
    )
    step = (span.frame_end - span.frame_start) / (count - 1)
    return tuple(round(span.frame_start + position * step) for position in range(count))


def _event_video_target_count(start_seconds: float, end_seconds: float) -> int:
    return min(
        _EVENT_VIDEO_MAX_FRAMES,
        max(
            _EVENT_VIDEO_MIN_FRAMES,
            math.ceil(max(0.0, end_seconds - start_seconds) * _EVENT_VIDEO_FPS),
        ),
    )


def _event_audio_window(item: ResolvedEvidence) -> tuple[float, float]:
    start = item.evidence_span.start_ms / 1_000
    end = item.evidence_span.end_ms / 1_000
    sample_duration = 1 / _EVENT_AUDIO_SAMPLE_RATE
    if end - start >= sample_duration:
        return start, end
    midpoint = (start + end) / 2
    start = max(0.0, midpoint - sample_duration / 2)
    end = start + sample_duration
    if item.media_object.duration_ms is not None:
        duration = item.media_object.duration_ms / 1_000
        if end > duration:
            end = duration
            start = max(0.0, end - sample_duration)
    return start, end


def _inside_window(value: float, start: float, end: float) -> bool:
    return start <= value <= end


def _cached_video_frame(
    frame: object,
    frame_count: int,
    frame_cache: dict[tuple[object, object, int, PixelRegion | None], object],
    region: PixelRegion | None,
) -> object:
    video_frame = cast(Any, frame)
    cache_key = (
        video_frame.pts if video_frame.pts is not None else id(frame),
        video_frame.time_base,
        frame_count,
        region,
    )
    if cache_key not in frame_cache:
        frame_cache[cache_key] = _video_frame_array(frame, frame_count, region)
    return frame_cache[cache_key]


def _video_frame_array(
    frame: object,
    frame_count: int,
    region: PixelRegion | None = None,
) -> object:
    video_frame = cast(Any, frame)
    if region is not None:
        x_max = min(region.x_max, video_frame.width)
        y_max = min(region.y_max, video_frame.height)
        if region.x_min >= x_max or region.y_min >= y_max:
            raise ModelRequestError("event pixel region falls outside video frame")
        pixels = video_frame.to_ndarray(format="rgb24")
        cropped = pixels[region.y_min : y_max, region.x_min : x_max]
        video_frame = cast(Any, type(video_frame)).from_ndarray(cropped, format="rgb24")
    max_pixels = _EVENT_VIDEO_MAX_TOTAL_PIXELS // frame_count
    scale = min(1.0, math.sqrt(max_pixels / (video_frame.width * video_frame.height)))
    width = max(2, math.floor(video_frame.width * scale))
    height = max(2, math.floor(video_frame.height * scale))
    width -= width % 2
    height -= height % 2
    return video_frame.reformat(width=width, height=height, format="rgb24").to_ndarray(
        format="rgb24"
    )


def validate_jina_embedding(values: tuple[float, ...], dimension: int) -> None:
    """Reject malformed vectors before they cross into a versioned index."""
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ModelOutputError("embedding vector has invalid dimension or values")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-6):
        raise ModelOutputError("embedding vector is not L2-normalized")
