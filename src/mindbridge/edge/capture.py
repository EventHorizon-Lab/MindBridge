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
) -> ObserveRequest:
    """Validate and queue one immutable video segment produced by the native capture stack."""
    if not bucket.strip():
        raise ValueError("bucket must not be empty")
    duration_ms = round((ended_at - occurred_at).total_seconds() * 1_000)
    if duration_ms <= 0:
        raise ValueError("captured video duration must be positive")
    resolved_path = media_path.resolve(strict=True)
    if not resolved_path.is_file():
        raise ValueError("captured video path must identify a regular file")
    checksum = sha256_file(resolved_path)
    media_object_id = derive_stable_id(
        "media",
        tenant_id,
        device_id,
        boot_id,
        str(sequence),
        checksum,
    )
    object_key = (
        f"tenants/{quote(tenant_id, safe='')}/devices/{quote(device_id, safe='')}/"
        f"{quote(boot_id, safe='')}/{sequence:020d}-{checksum}{resolved_path.suffix.lower()}"
    )
    uri = f"s3://{bucket}/{object_key}"
    tenant_s3_object_key(bucket, tenant_id, uri)
    request = ObserveRequest(
        tenant_id=tenant_id,
        device_id=device_id,
        boot_id=boot_id,
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id=media_object_id,
                kind=MediaKind.VIDEO,
                uri=uri,
                sha256=checksum,
                size_bytes=resolved_path.stat().st_size,
                created_at=observed_at,
                duration_ms=duration_ms,
            ),
        ),
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
        (
            EdgeMediaFile(
                media_object_id=media_object_id,
                local_path=resolved_path,
                content_type=content_type,
            ),
        ),
    )
    return request
