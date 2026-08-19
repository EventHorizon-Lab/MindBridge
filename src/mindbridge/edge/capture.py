"""Handoff from a completed GStreamer/DeepStream segment to the durable outbox."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from mindbridge.contracts import IdentityObservationInput, MediaObjectInput, ObserveRequest
from mindbridge.core import MediaKind, SensorKind, derive_stable_id
from mindbridge.edge.outbox import EdgeMediaFile, SQLiteObservationOutbox
from mindbridge.file_integrity import sha256_file
from mindbridge.infrastructure.s3 import tenant_s3_object_key

_DEFAULT_CONTENT_TYPES = {
    MediaKind.VIDEO: "video/mp4",
    MediaKind.AUDIO: "audio/wav",
    MediaKind.IMAGE: "image/png",
}


def enqueue_captured_media(
    outbox: SQLiteObservationOutbox,
    media_path: Path,
    *,
    kind: MediaKind = MediaKind.VIDEO,
    audio_path: Path | None = None,
    tenant_id: str,
    device_id: str,
    boot_id: str,
    sequence: int,
    bucket: str,
    occurred_at: datetime,
    ended_at: datetime,
    observed_at: datetime,
    clock_offset_ms: int = 0,
    identity_observations: tuple[IdentityObservationInput, ...] = (),
    content_type: str | None = None,
    audio_content_type: str = "audio/wav",
) -> ObserveRequest:
    """Queue one captured segment, plus a synchronized native audio sidecar for video.

    `kind` is what makes a microphone-only or still-image observation reachable from the edge:
    the cloud contract has always accepted both, but this was the only capture entry and it
    hardcoded video with a camera sensor.
    """
    if not bucket.strip():
        raise ValueError("bucket must not be empty")
    if audio_path is not None and kind is not MediaKind.VIDEO:
        raise ValueError("an audio sidecar only accompanies captured video")
    duration_ms = round((ended_at - occurred_at).total_seconds() * 1_000)
    if duration_ms <= 0:
        raise ValueError("captured media duration must be positive")
    resolved_path = media_path.resolve(strict=True)
    resolved_audio_path = audio_path.resolve(strict=True) if audio_path is not None else None
    if not resolved_path.is_file() or (
        resolved_audio_path is not None and not resolved_audio_path.is_file()
    ):
        raise ValueError("captured media path must identify a regular file")
    media_inputs = []
    media_files = []
    for media_kind, path, media_content_type in (
        (kind, resolved_path, content_type or _DEFAULT_CONTENT_TYPES[kind]),
        (MediaKind.AUDIO, resolved_audio_path, audio_content_type),
    ):
        if path is None:
            continue
        checksum = sha256_file(path)
        media_object_id = derive_stable_id(
            # Video keeps the bare "media" prefix and audio keeps "audio_media" so already-queued
            # outbox rows keep deriving the same IDs across this change.
            "media" if media_kind is MediaKind.VIDEO else f"{media_kind.value}_media",
            tenant_id,
            device_id,
            boot_id,
            str(sequence),
            checksum,
        )
        object_key = f"tenants/{quote(tenant_id, safe='')}/media/{checksum}{path.suffix.lower()}"
        uri = f"s3://{bucket}/{object_key}"
        tenant_s3_object_key(bucket, tenant_id, uri)
        media_inputs.append(
            MediaObjectInput(
                media_object_id=media_object_id,
                kind=media_kind,
                uri=uri,
                sha256=checksum,
                size_bytes=path.stat().st_size,
                created_at=observed_at,
                # A still image has no duration; claiming the observation's span would be a lie.
                duration_ms=None if media_kind is MediaKind.IMAGE else duration_ms,
            )
        )
        media_files.append(
            EdgeMediaFile(
                media_object_id=media_object_id,
                local_path=path,
                content_type=media_content_type,
            )
        )
    request = ObserveRequest(
        tenant_id=tenant_id,
        device_id=device_id,
        boot_id=boot_id,
        sequence=sequence,
        sensor=(
            SensorKind.MICROPHONE
            if all(media.kind is MediaKind.AUDIO for media in media_inputs)
            else SensorKind.CAMERA
        ),
        media_objects=tuple(media_inputs),
        occurred_at=occurred_at,
        ended_at=ended_at,
        observed_at=observed_at,
        clock_offset_ms=clock_offset_ms,
        identity_observations=identity_observations,
        idempotency_key=derive_stable_id(
            "edge_observation",
            tenant_id,
            device_id,
            boot_id,
            str(sequence),
        ),
    )
    outbox.enqueue(
        request,
        tuple(media_files),
    )
    return request
