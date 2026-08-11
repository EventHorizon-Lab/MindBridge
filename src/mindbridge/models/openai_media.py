"""Qwen-compatible multimodal content parts for the official OpenAI SDK."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import PurePosixPath
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from mindbridge.application import ResolvedEvidence
from mindbridge.core import MediaKind


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


OpenAIContentPart = TextPart | ImagePart | VideoPart | AudioPart


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
    return media_url_content_part(
        evidence.media_object.kind,
        evidence.media_url,
        source_uri=evidence.media_object.uri,
        video_frames_per_second=video_frames_per_second,
        video_max_pixels=video_max_pixels,
    )


def media_url_content_part(
    media_kind: MediaKind,
    media_url: str,
    *,
    source_uri: str,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> ImagePart | VideoPart | AudioPart:
    """Build one native AV part from a validated short-lived media URL."""
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
