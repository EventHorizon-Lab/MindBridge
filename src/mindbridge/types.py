"""Small values accepted and returned by the public MindBridge API."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeAlias
from urllib.parse import urlsplit

from mindbridge.exceptions import ValidationError

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")


class Modality(str, Enum):
    """A native MindBridge input modality; omni means multiple media kinds."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OMNI = "omni"


class MemoryType(str, Enum):
    """The cognitive role a memory serves for an agent."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass(frozen=True, slots=True)
class URL:
    """One HTTPS source with an optional expected media type or top-level range."""

    value: str
    media_type: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        value = _text(self.value, "URL value")
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError:
            raise ValidationError("URL value must be an HTTPS URL") from None
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValidationError("URL value must be an HTTPS URL without credentials")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "media_type", _optional_url_media_type(self.media_type))
        object.__setattr__(self, "name", _optional_text(self.name, "URL name"))


@dataclass(frozen=True, slots=True)
class Blob:
    """Immutable inline media bytes with an explicit IANA media type."""

    data: bytes = field(repr=False)
    media_type: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValidationError("Blob data must be non-empty bytes")
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        object.__setattr__(self, "name", _optional_text(self.name, "Blob name"))


@dataclass(frozen=True, slots=True)
class AssetRef:
    """A local asset, or an opaque ID for Memory to resolve from SQLite."""

    id: str
    modality: Modality | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    name: str | None = None
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, "asset id"))
        modality = self.modality
        if modality is not None and not isinstance(modality, Modality):
            try:
                modality = Modality(modality)
            except (TypeError, ValueError):
                raise ValidationError("asset modality is invalid") from None
        if modality in {Modality.TEXT, Modality.OMNI}:
            raise ValidationError("asset modality must be image, video, or audio")
        media_type = _optional_media_type(self.media_type)
        if modality is not None and media_type is not None:
            _require_matching_media_type(modality, media_type)
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValidationError("asset size_bytes must be a non-negative integer")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValidationError("asset sha256 must be 64 lowercase hexadecimal characters")
        path = self.path
        if path is not None:
            try:
                path = Path(path)
            except TypeError:
                raise ValidationError("asset path must be a filesystem path") from None
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "name", _optional_text(self.name, "asset name"))
        object.__setattr__(self, "path", path)

    @property
    def is_resolved(self) -> bool:
        """Whether this reference carries all metadata required by a model adapter."""
        return (
            self.modality is not None
            and self.media_type is not None
            and self.size_bytes is not None
            and self.sha256 is not None
            and self.path is not None
        )


ContentAtom: TypeAlias = str | Path | URL | Blob | AssetRef
ContentInput: TypeAlias = ContentAtom | Sequence[ContentAtom]


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One timed transcript turn linked to a stable local speaker identity."""

    asset_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    identity_score: float | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("speaker segment asset_id must be a SHA-256 identifier")
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speaker segment must have a positive time range")
        object.__setattr__(self, "text", _text(self.text, "speaker segment text"))
        if self.speaker_id is not None:
            object.__setattr__(
                self,
                "speaker_id",
                _text(self.speaker_id, "speaker_id"),
            )
        if self.speaker_name is not None:
            name = _text(self.speaker_name, "speaker_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("speaker_name must be at most 255 printable characters")
            object.__setattr__(self, "speaker_name", name)
        _optional_score(self.identity_score, "identity_score")

    @property
    def identity_id(self) -> str | None:
        """Return the unified identity ID; ``speaker_id`` remains as a compatibility alias."""
        return self.speaker_id

    @property
    def identity_name(self) -> str | None:
        """Return the unified identity name; ``speaker_name`` remains as a compatibility alias."""
        return self.speaker_name


@dataclass(frozen=True, slots=True)
class FaceMatch:
    """One detected face linked to a stable local biometric identity."""

    asset_id: str
    bbox_xyxy: tuple[float, float, float, float]
    detection_score: float
    identity_id: str
    identity_name: str | None = None
    identity_score: float | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("face match asset_id must be a SHA-256 identifier")
        bbox = tuple(self.bbox_xyxy)
        if len(bbox) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not 0.0 <= value <= 1.0
            for value in bbox
        ):
            raise ValidationError("face bbox_xyxy must contain four normalized coordinates")
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            raise ValidationError("face bbox_xyxy must have positive area")
        if (self.start_ms is None) != (self.end_ms is None) or (
            self.start_ms is not None
            and (
                isinstance(self.start_ms, bool)
                or not isinstance(self.start_ms, int)
                or isinstance(self.end_ms, bool)
                or not isinstance(self.end_ms, int)
                or self.start_ms < 0
                or self.end_ms <= self.start_ms
            )
        ):
            raise ValidationError("face time range must be absent or positive")
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if self.identity_name is not None:
            name = _text(self.identity_name, "identity_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("identity_name must be at most 255 printable characters")
            object.__setattr__(self, "identity_name", name)
        _optional_score(self.detection_score, "detection_score", required=True)
        _optional_score(self.identity_score, "identity_score")


def _metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    return dict(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must not be blank")
    return value.strip()


def _optional_text(value: object | None, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _optional_score(value: object | None, name: str, *, required: bool = False) -> None:
    if value is None and not required:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")


def _media_type(value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError("media_type must be an image, video, or audio media type")
    normalized = value.strip().lower()
    if _MEDIA_TYPE.fullmatch(normalized) is None or normalized.split("/", 1)[0] not in {
        "image",
        "video",
        "audio",
    }:
        raise ValidationError("media_type must be an image, video, or audio media type")
    return normalized


def _optional_media_type(value: object | None) -> str | None:
    return None if value is None else _media_type(value)


def _optional_url_media_type(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"image/*", "video/*", "audio/*"}:
        return value.strip().lower()
    return _media_type(value)


def _require_matching_media_type(modality: Modality, media_type: str) -> None:
    if media_type.split("/", 1)[0] != modality.value:
        raise ValidationError("asset modality does not match media_type")


def _require_aware(value: datetime | None, name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValidationError(f"{name} must include a timezone")


def _assets(value: Sequence[AssetRef], modality: Modality) -> tuple[AssetRef, ...]:
    assets = tuple(value)
    if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
        raise ValidationError("record assets must contain resolved AssetRef values")
    kinds = {asset.modality for asset in assets}
    if modality is Modality.TEXT and assets:
        raise ValidationError("text memories must not contain media assets")
    if modality in {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO} and (
        not assets or kinds != {modality}
    ):
        raise ValidationError("memory modality does not match its assets")
    if modality is Modality.OMNI and len(kinds) < 2:
        raise ValidationError("omni memories must contain at least two media modalities")
    return assets


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One stored memory."""

    id: str
    content: str
    created_at: datetime
    occurred_at: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryType = MemoryType.SEMANTIC

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_aware(self.occurred_at, "occurred_at")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryType):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked memory, flattened for direct use."""

    id: str
    content: str
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryType = MemoryType.SEMANTIC

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_aware(self.occurred_at, "occurred_at")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between zero and one")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryType):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A grounded answer and the memories used to produce it."""

    answer: str
    hits: tuple[SearchHit, ...] = ()

    def __post_init__(self) -> None:
        _text(self.answer, "answer")


@dataclass(frozen=True, slots=True)
class Page:
    """One stable page of memories."""

    items: tuple[MemoryRecord, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None:
            _text(self.next_cursor, "next_cursor")
