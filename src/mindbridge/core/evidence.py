"""Evidence-first domain types for captured multimodal observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from string import hexdigits
from typing import NewType

from mindbridge.core.errors import DomainInvariantError

TenantId = NewType("TenantId", str)
DeviceId = NewType("DeviceId", str)
MediaObjectId = NewType("MediaObjectId", str)
ObservationId = NewType("ObservationId", str)
EvidenceId = NewType("EvidenceId", str)

_SHA256_HEX_LENGTH = 64


class MediaKind(str, Enum):
    """Supported raw or derived media kinds."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class SensorKind(str, Enum):
    """Sensor that produced an observation."""

    CAMERA = "camera"
    MICROPHONE = "microphone"
    GAZE = "gaze"
    IMU = "imu"
    ROBOT_STATE = "robot_state"


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
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise DomainInvariantError("pixel region maximums must exceed minimums")


@dataclass(frozen=True, slots=True)
class MediaObject:
    """Immutable reference to raw or derived media content."""

    id: MediaObjectId
    tenant_id: TenantId
    kind: MediaKind
    uri: str
    sha256: str
    size_bytes: int
    created_at: datetime
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "id")
        _require_non_empty(self.tenant_id, "tenant_id")
        _require_non_empty(self.uri, "uri")
        _require_aware_datetime(self.created_at, "created_at")
        if not _is_sha256(self.sha256):
            raise DomainInvariantError("sha256 must contain exactly 64 hexadecimal characters")
        if self.size_bytes < 0:
            raise DomainInvariantError("size_bytes must be non-negative")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise DomainInvariantError("duration_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class Observation:
    """A timestamped sensor observation produced by one device."""

    id: ObservationId
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

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "id")
        _require_non_empty(self.tenant_id, "tenant_id")
        _require_non_empty(self.device_id, "device_id")
        _require_non_empty(self.boot_id, "boot_id")
        _require_aware_datetime(self.occurred_at, "occurred_at")
        _require_aware_datetime(self.ended_at, "ended_at")
        _require_aware_datetime(self.observed_at, "observed_at")
        if self.sequence < 0:
            raise DomainInvariantError("sequence must be non-negative")
        if not self.media_object_ids:
            raise DomainInvariantError("an observation must reference at least one media object")
        if len(set(self.media_object_ids)) != len(self.media_object_ids):
            raise DomainInvariantError("media_object_ids must not contain duplicates")
        if self.ended_at < self.occurred_at:
            raise DomainInvariantError("ended_at must not precede occurred_at")

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

    id: EvidenceId
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
        _require_non_empty(self.id, "id")
        _require_non_empty(self.tenant_id, "tenant_id")
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.media_object_id, "media_object_id")
        _require_aware_datetime(self.created_at, "created_at")
        if self.start_ms < 0:
            raise DomainInvariantError("start_ms must be non-negative")
        if self.end_ms < self.start_ms:
            raise DomainInvariantError("end_ms must not precede start_ms")
        if (self.frame_start is None) != (self.frame_end is None):
            raise DomainInvariantError("frame_start and frame_end must be provided together")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and (self.frame_start < 0 or self.frame_end < self.frame_start)
        ):
            raise DomainInvariantError("frame range must be non-negative and ordered")
        if self.audio_track is not None and self.audio_track < 0:
            raise DomainInvariantError("audio_track must be non-negative")

    @property
    def duration_ms(self) -> int:
        """Return the evidence window duration in milliseconds."""
        return self.end_ms - self.start_ms


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(char in hexdigits for char in value)


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainInvariantError(f"{field_name} must include timezone information")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise DomainInvariantError(f"{field_name} must not be empty")
