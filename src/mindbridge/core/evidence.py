"""Evidence-first domain types for captured multimodal observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from string import hexdigits

from mindbridge.core._validation import require_aware_datetime, require_non_empty
from mindbridge.core.errors import DomainInvariantError
from mindbridge.core.identifiers import (
    DeviceId,
    EvidenceId,
    MediaObjectId,
    ObservationId,
    TenantId,
)
from mindbridge.core.identity import AnonymousIdentityObservation

_SHA256_HEX_LENGTH = 64
_SIGNED_INT64_MAX = 2**63 - 1
_SIGNED_INT32_MIN = -(2**31)
_SIGNED_INT32_MAX = 2**31 - 1


class MediaKind(str, Enum):
    """Supported raw or derived media kinds."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class SensorKind(str, Enum):
    """Sensor that produced an observation; each one must carry MediaKind evidence."""

    CAMERA = "camera"
    MICROPHONE = "microphone"


# Explicit rather than `mimetypes.guess_type`, whose answers come from the host's /etc/mime.types:
# a validator that accepts a URI on one machine and rejects it on another is worse than none.
_MEDIA_KIND_BY_SUFFIX = {
    ".aac": MediaKind.AUDIO,
    ".flac": MediaKind.AUDIO,
    ".m4a": MediaKind.AUDIO,
    ".mp3": MediaKind.AUDIO,
    ".ogg": MediaKind.AUDIO,
    ".opus": MediaKind.AUDIO,
    ".wav": MediaKind.AUDIO,
    ".bmp": MediaKind.IMAGE,
    ".gif": MediaKind.IMAGE,
    ".jpeg": MediaKind.IMAGE,
    ".jpg": MediaKind.IMAGE,
    ".png": MediaKind.IMAGE,
    ".webp": MediaKind.IMAGE,
    ".avi": MediaKind.VIDEO,
    ".mkv": MediaKind.VIDEO,
    ".mov": MediaKind.VIDEO,
    ".mp4": MediaKind.VIDEO,
    ".webm": MediaKind.VIDEO,
}


def media_kind_for_suffix(suffix: str) -> MediaKind | None:
    """Return the kind a container extension implies, or None when it implies nothing.

    None means "no opinion", not "invalid": object keys are frequently extensionless, so an
    unrecognized suffix must never be treated as a mismatch.
    """
    return _MEDIA_KIND_BY_SUFFIX.get(suffix.lower())


@dataclass(frozen=True, slots=True)
class PixelRegion:
    """Axis-aligned image region in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if min(self.x_min, self.y_min) < 0:
            raise DomainInvariantError("pixel region coordinates must be non-negative")
        if max(self.x_min, self.y_min, self.x_max, self.y_max) > _SIGNED_INT32_MAX:
            raise DomainInvariantError("pixel region coordinates must fit signed 32-bit integers")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise DomainInvariantError("pixel region maximums must exceed minimums")


@dataclass(frozen=True, slots=True)
class MediaObject:
    """Immutable reference to raw or derived media content."""

    media_object_id: MediaObjectId
    tenant_id: TenantId
    kind: MediaKind
    uri: str
    sha256: str
    size_bytes: int
    created_at: datetime
    duration_ms: int | None = None
    derived_from_media_object_id: MediaObjectId | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.media_object_id, "media_object_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.uri, "uri")
        require_aware_datetime(self.created_at, "created_at")
        if not _is_sha256(self.sha256):
            raise DomainInvariantError("sha256 must contain exactly 64 hexadecimal characters")
        if not 0 <= self.size_bytes <= _SIGNED_INT64_MAX:
            raise DomainInvariantError("size_bytes must fit a non-negative signed 64-bit integer")
        if self.duration_ms is not None and not 0 <= self.duration_ms <= _SIGNED_INT64_MAX:
            raise DomainInvariantError("duration_ms must fit a non-negative signed 64-bit integer")
        if self.derived_from_media_object_id is not None:
            require_non_empty(self.derived_from_media_object_id, "derived_from_media_object_id")
            if self.derived_from_media_object_id == self.media_object_id:
                raise DomainInvariantError("media object cannot derive from itself")


@dataclass(frozen=True, slots=True)
class Observation:
    """A timestamped sensor observation produced by one device."""

    observation_id: ObservationId
    tenant_id: TenantId
    device_id: DeviceId
    boot_id: str
    sequence: int
    sensor: SensorKind
    media_object_ids: tuple[MediaObjectId, ...]
    occurred_at: datetime
    ended_at: datetime
    observed_at: datetime
    clock_offset_ms: int = 0
    identity_observations: tuple[AnonymousIdentityObservation, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.observation_id, "observation_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.device_id, "device_id")
        require_non_empty(self.boot_id, "boot_id")
        require_aware_datetime(self.occurred_at, "occurred_at")
        require_aware_datetime(self.ended_at, "ended_at")
        require_aware_datetime(self.observed_at, "observed_at")
        if not 0 <= self.sequence <= _SIGNED_INT64_MAX:
            raise DomainInvariantError("sequence must fit a non-negative signed 64-bit integer")
        if not _SIGNED_INT32_MIN <= self.clock_offset_ms <= _SIGNED_INT32_MAX:
            raise DomainInvariantError("clock_offset_ms must fit a signed 32-bit integer")
        if not self.media_object_ids:
            raise DomainInvariantError("an observation must reference at least one media object")
        if len(set(self.media_object_ids)) != len(self.media_object_ids):
            raise DomainInvariantError("media_object_ids must not contain duplicates")
        if self.ended_at < self.occurred_at:
            raise DomainInvariantError("ended_at must not precede occurred_at")
        duration_ms = round((self.ended_at - self.occurred_at).total_seconds() * 1_000)
        if any(identity.end_ms > duration_ms for identity in self.identity_observations):
            raise DomainInvariantError("identity observation exceeds its source observation")
        identity_keys = [
            (
                identity.kind,
                identity.identity_id,
                identity.start_ms,
                identity.end_ms,
                identity.model_reference,
            )
            for identity in self.identity_observations
        ]
        if len(set(identity_keys)) != len(identity_keys):
            raise DomainInvariantError("identity observations must not contain duplicates")

    @property
    def idempotency_key(self) -> str:
        """Return a stable key for retry-safe edge-to-cloud ingestion."""
        components = [self.tenant_id, self.device_id, self.boot_id, str(self.sequence)]
        canonical = json.dumps(components, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"observation_{digest}"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Precise time, frame, and optional image region within captured media."""

    evidence_id: EvidenceId
    tenant_id: TenantId
    observation_id: ObservationId
    media_object_id: MediaObjectId
    start_ms: int
    end_ms: int
    created_at: datetime
    frame_start: int | None = None
    frame_end: int | None = None
    region: PixelRegion | None = None
    audio_track: int | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.evidence_id, "evidence_id")
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.observation_id, "observation_id")
        require_non_empty(self.media_object_id, "media_object_id")
        require_aware_datetime(self.created_at, "created_at")
        if not 0 <= self.start_ms <= _SIGNED_INT64_MAX:
            raise DomainInvariantError("start_ms must fit a non-negative signed 64-bit integer")
        if self.end_ms < self.start_ms:
            raise DomainInvariantError("end_ms must not precede start_ms")
        if self.end_ms > _SIGNED_INT64_MAX:
            raise DomainInvariantError("end_ms must fit a non-negative signed 64-bit integer")
        if (self.frame_start is None) != (self.frame_end is None):
            raise DomainInvariantError("frame_start and frame_end must be provided together")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and (self.frame_start < 0 or self.frame_end < self.frame_start)
        ):
            raise DomainInvariantError("frame range must be non-negative and ordered")
        if self.frame_end is not None and self.frame_end > _SIGNED_INT64_MAX:
            raise DomainInvariantError("frame range must fit signed 64-bit integers")
        if self.audio_track is not None and not 0 <= self.audio_track <= _SIGNED_INT32_MAX:
            raise DomainInvariantError("audio_track must fit a non-negative signed 32-bit integer")

    @property
    def duration_ms(self) -> int:
        """Return the evidence window duration in milliseconds."""
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class EvidenceClip:
    """One encoder-sized window of an evidence span, stored as derived media.

    A span longer than the encoder's window becomes several ordered clips, so
    the mapping is one-to-many and cannot be recomputed from the evidence id
    alone once identical clip content deduplicates onto one media object.
    """

    tenant_id: TenantId
    evidence_id: EvidenceId
    ordinal: int
    media_object_id: MediaObjectId
    start_ms: int
    end_ms: int
    created_at: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.tenant_id, "tenant_id")
        require_non_empty(self.evidence_id, "evidence_id")
        require_non_empty(self.media_object_id, "media_object_id")
        require_aware_datetime(self.created_at, "created_at")
        if self.ordinal < 0:
            raise DomainInvariantError("evidence clip ordinal must be non-negative")
        if not 0 <= self.start_ms <= _SIGNED_INT64_MAX:
            raise DomainInvariantError("start_ms must fit a non-negative signed 64-bit integer")
        if self.end_ms < self.start_ms or self.end_ms > _SIGNED_INT64_MAX:
            raise DomainInvariantError("end_ms must not precede start_ms")


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(char in hexdigits for char in value)
