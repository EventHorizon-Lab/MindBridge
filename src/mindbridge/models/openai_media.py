"""Qwen-compatible multimodal content parts for the official OpenAI SDK."""

from __future__ import annotations

import math
from collections.abc import Collection
from pathlib import PurePosixPath
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.core import MediaKind, MediaObject


class UrlValue(TypedDict):
    url: str


class AudioValue(TypedDict):
    data: str
    format: str


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: UrlValue


class VideoPart(TypedDict):
    type: Literal["video_url"]
    video_url: UrlValue
    fps: float
    max_pixels: int


class AudioPart(TypedDict):
    type: Literal["input_audio"]
    input_audio: AudioValue


class AudioUrlPart(TypedDict):
    type: Literal["audio_url"]
    audio_url: UrlValue


OpenAIContentPart = TextPart | ImagePart | VideoPart | AudioPart | AudioUrlPart


def media_input_span_attributes(
    media_objects: Collection[MediaObject],
    *,
    video_frames_per_second: float,
) -> dict[str, int | float]:
    """Summarize distinct model media inputs without recording IDs, URLs, or content."""
    if not math.isfinite(video_frames_per_second) or video_frames_per_second <= 0:
        raise ValueError("video_frames_per_second must be finite and positive")
    distinct = {item.media_object_id: item for item in media_objects}.values()
    videos = tuple(item for item in distinct if item.kind is MediaKind.VIDEO)
    audios = tuple(item for item in distinct if item.kind is MediaKind.AUDIO)
    video_seconds = sum((item.duration_ms or 0) / 1_000 for item in videos)
    audio_seconds = sum((item.duration_ms or 0) / 1_000 for item in audios)
    return {
        "mindbridge.model.input.media_count": len(distinct),
        "mindbridge.model.input.duration_known_count": sum(
            item.duration_ms is not None for item in distinct
        ),
        "mindbridge.model.input.video_seconds": video_seconds,
        "mindbridge.model.input.audio_seconds": audio_seconds,
        "mindbridge.model.input.estimated_video_frames": (video_seconds * video_frames_per_second),
    }


def evidence_media_content_parts(
    evidence: tuple[ResolvedEvidence, ...],
    *,
    video_frames_per_second: float,
    video_max_pixels: int,
    excluded_media_object_ids: Collection[str] = (),
) -> list[OpenAIContentPart]:
    """Label and attach each distinct source media object exactly once."""
    parts: list[OpenAIContentPart] = []
    seen_media_object_ids = set(excluded_media_object_ids)
    for item in evidence:
        media_object_id = item.media_object.media_object_id
        if media_object_id in seen_media_object_ids:
            continue
        seen_media_object_ids.add(media_object_id)
        parts.append({"type": "text", "text": f"Source media_object_id={media_object_id} follows."})
        parts.append(
            media_content_part(
                item,
                video_frames_per_second=video_frames_per_second,
                video_max_pixels=video_max_pixels,
            )
        )
    return parts


def media_content_part(
    evidence: ResolvedEvidence,
    *,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> ImagePart | VideoPart | AudioPart:
    """Convert openable evidence to the provider's native AV content shape."""
    return qwen_media_url_content_part(
        evidence.media_object.kind,
        evidence.media_url,
        source_uri=evidence.media_object.uri,
        video_frames_per_second=video_frames_per_second,
        video_max_pixels=video_max_pixels,
    )


def qwen_media_url_content_part(
    media_kind: MediaKind,
    media_url: str,
    *,
    source_uri: str,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> ImagePart | VideoPart | AudioPart:
    """Build one Qwen AV part from a validated short-lived media URL."""
    if media_kind is MediaKind.IMAGE:
        return {"type": "image_url", "image_url": {"url": media_url}}
    if media_kind is MediaKind.VIDEO:
        return {
            "type": "video_url",
            "video_url": {"url": media_url},
            "fps": video_frames_per_second,
            "max_pixels": video_max_pixels,
        }
    suffix = PurePosixPath(urlsplit(source_uri).path).suffix
    return {
        "type": "input_audio",
        "input_audio": {
            "data": media_url,
            "format": suffix.removeprefix(".").lower() or "wav",
        },
    }


def vllm_media_url_content_part(
    media_kind: MediaKind,
    media_url: str,
    *,
    source_uri: str,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> OpenAIContentPart:
    """Build one vLLM pooling content part without Qwen's audio extension."""
    if media_kind is MediaKind.AUDIO:
        return {"type": "audio_url", "audio_url": {"url": media_url}}
    return qwen_media_url_content_part(
        media_kind,
        media_url,
        source_uri=source_uri,
        video_frames_per_second=video_frames_per_second,
        video_max_pixels=video_max_pixels,
    )
