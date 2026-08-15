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


def enqueue_captured_video(
    outbox: SQLiteObservationOutbox,
    media_path: Path,
    *,
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
    content_type: str = "video/mp4",
    audio_content_type: str = "audio/wav",
) -> ObserveRequest:
    """Queue one synchronized video segment and its optional native audio sidecar."""
    if not bucket.strip():
        raise ValueError("bucket must not be empty")
    duration_ms = round((ended_at - occurred_at).total_seconds() * 1_000)
    if duration_ms <= 0:
        raise ValueError("captured video duration must be positive")
    resolved_path = media_path.resolve(strict=True)
    resolved_audio_path = audio_path.resolve(strict=True) if audio_path is not None else None
    if not resolved_path.is_file() or (
        resolved_audio_path is not None and not resolved_audio_path.is_file()
    ):
        raise ValueError("captured video path must identify a regular file")
    media_inputs = []
    media_files = []
    for kind, path, media_content_type in (
        (MediaKind.VIDEO, resolved_path, content_type),
        (MediaKind.AUDIO, resolved_audio_path, audio_content_type),
    ):
        if path is None:
            continue
        checksum = sha256_file(path)
        media_object_id = derive_stable_id(
            "media" if kind is MediaKind.VIDEO else "audio_media",
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
                kind=kind,
                uri=uri,
                sha256=checksum,
                size_bytes=path.stat().st_size,
                created_at=observed_at,
                duration_ms=duration_ms,
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
        sensor=SensorKind.CAMERA,
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
