"""Qwen-compatible multimodal content parts for the official OpenAI SDK."""

from __future__ import annotations

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


def media_content_part(
    evidence: ResolvedEvidence,
    *,
    video_frames_per_second: float,
    video_max_pixels: int,
) -> ImagePart | VideoPart | AudioPart:
    """Convert openable evidence to the provider's native AV content shape."""
    media_kind = evidence.media_object.kind
    if media_kind is MediaKind.IMAGE:
        return {"type": "image_url", "image_url": {"url": evidence.media_url}}
    if media_kind is MediaKind.VIDEO:
        return {
            "type": "video_url",
            "video_url": {"url": evidence.media_url},
            "fps": video_frames_per_second,
            "max_pixels": video_max_pixels,
        }
    suffix = PurePosixPath(urlsplit(evidence.media_object.uri).path).suffix
    return {
        "type": "input_audio",
        "input_audio": {
            "data": evidence.media_url,
            "format": suffix.removeprefix(".").lower() or "wav",
        },
    }
