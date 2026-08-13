"""Native-capture handoff checks for stable edge observations."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import IdentityKind
from mindbridge.edge import SQLiteObservationOutbox, enqueue_captured_video

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_completed_video_becomes_one_retry_safe_outbox_item(tmp_path: Path) -> None:
    media_path = tmp_path / "capture.mp4"
    media_path.write_bytes(b"video")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)
    identities = (
        IdentityObservationInput(
            identity_id="person_device_01",
            kind=IdentityKind.FACE,
            start_ms=100,
            end_ms=900,
            confidence=0.91,
            model_id="insightface/buffalo_l",
            model_revision="1.0.1",
        ),
    )

    first = enqueue_captured_video(
        outbox,
        media_path,
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=7,
        bucket="memory",
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=31),
        clock_offset_ms=12,
        identity_observations=identities,
    )
    duplicate = enqueue_captured_video(
        outbox,
        media_path,
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=7,
        bucket="memory",
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=31),
        clock_offset_ms=12,
        identity_observations=identities,
    )

    checksum = hashlib.sha256(b"video").hexdigest()
    assert first == duplicate
    assert first.media_objects[0].sha256 == checksum
    assert first.media_objects[0].duration_ms == 30_000
    assert first.media_objects[0].uri == f"s3://memory/tenants/tenant_01/media/{checksum}.mp4"
    assert first.idempotency_key is not None
    assert first.identity_observations == identities
    assert outbox.pending_count() == 1


def test_synchronized_audio_sidecar_shares_one_observation(tmp_path: Path) -> None:
    video_path = tmp_path / "capture.mp4"
    audio_path = tmp_path / "capture.wav"
    video_path.write_bytes(b"same bytes")
    audio_path.write_bytes(b"same bytes")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)

    request = enqueue_captured_video(
        outbox,
        video_path,
        audio_path=audio_path,
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=8,
        bucket="memory",
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=31),
    )

    assert [item.kind.value for item in request.media_objects] == ["video", "audio"]
    assert len({item.media_object_id for item in request.media_objects}) == 2
    assert {item.duration_ms for item in request.media_objects} == {30_000}
    queued = outbox.next_pending()
    assert queued is not None
    assert [item.local_path for item in queued.media_files] == [
        video_path.resolve(),
        audio_path.resolve(),
    ]
