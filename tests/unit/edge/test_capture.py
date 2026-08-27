"""Native-capture handoff checks for stable edge observations."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge.contracts import IdentityObservationInput, MediaObjectInput
from mindbridge.core import IdentityKind, MediaKind, SensorKind
from mindbridge.edge import SQLiteObservationOutbox, enqueue_captured_media

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_microphone_only_capture_reaches_the_cloud_contract(tmp_path: Path) -> None:
    """The cloud has always accepted a microphone sensor; the edge could not produce one."""
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"audio")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)

    request = enqueue_captured_media(
        outbox,
        audio_path,
        kind=MediaKind.AUDIO,
        tenant_id="tenant_01",
        device_id="mic_01",
        boot_id="boot_01",
        sequence=1,
        bucket="memory",
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=31),
    )

    assert request.sensor is SensorKind.MICROPHONE
    assert [item.kind for item in request.media_objects] == [MediaKind.AUDIO]
    assert request.media_objects[0].duration_ms == 30_000


def test_image_only_capture_claims_no_duration(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"image")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)

    request = enqueue_captured_media(
        outbox,
        image_path,
        kind=MediaKind.IMAGE,
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=1,
        bucket="memory",
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=31),
    )

    assert request.sensor is SensorKind.CAMERA
    assert [item.kind for item in request.media_objects] == [MediaKind.IMAGE]
    # A still frame has no duration, so it must not borrow the observation's span.
    assert request.media_objects[0].duration_ms is None


def test_audio_sidecar_is_rejected_for_non_video_capture(tmp_path: Path) -> None:
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"audio")
    sidecar_path = tmp_path / "sidecar.wav"
    sidecar_path.write_bytes(b"sidecar")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)

    with pytest.raises(ValueError, match="sidecar only accompanies captured video"):
        enqueue_captured_media(
            outbox,
            audio_path,
            kind=MediaKind.AUDIO,
            audio_path=sidecar_path,
            tenant_id="tenant_01",
            device_id="mic_01",
            boot_id="boot_01",
            sequence=1,
            bucket="memory",
            occurred_at=NOW,
            ended_at=NOW + timedelta(seconds=30),
            observed_at=NOW + timedelta(seconds=31),
        )


def test_declared_media_kind_must_match_its_uri_extension() -> None:
    """Routing trusts the declared kind, and the server cannot fetch the URI to check it."""
    with pytest.raises(ValueError, match="contradicts its video URI extension"):
        MediaObjectInput(
            media_object_id="media_01",
            kind=MediaKind.IMAGE,
            uri="s3://memory/tenants/tenant_01/media/abc.mp4",
            sha256="a" * 64,
            size_bytes=1_024,
            created_at=NOW,
        )
    # An extensionless object key carries no claim, so it must still be accepted.
    assert (
        MediaObjectInput(
            media_object_id="media_02",
            kind=MediaKind.IMAGE,
            uri="s3://memory/tenants/tenant_01/media/abc",
            sha256="a" * 64,
            size_bytes=1_024,
            created_at=NOW,
        ).kind
        is MediaKind.IMAGE
    )


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
        ),
    )

    first = enqueue_captured_media(
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
    duplicate = enqueue_captured_media(
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
    transcript = IdentityObservationInput(
        identity_id="speaker_01",
        kind=IdentityKind.VOICE,
        start_ms=0,
        end_ms=1_000,
        confidence=0.9,
        model_id="funasr/sensevoice",
        transcript="pass the tool",
    )

    request = enqueue_captured_media(
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
        identity_observations=(transcript,),
    )

    assert [item.kind.value for item in request.media_objects] == ["video", "audio"]
    assert len({item.media_object_id for item in request.media_objects}) == 2
    assert {item.duration_ms for item in request.media_objects} == {30_000}
    audio = next(item for item in request.media_objects if item.kind is MediaKind.AUDIO)
    assert request.identity_observations[0].transcript_media_object_id == audio.media_object_id
    queued = outbox.next_pending()
    assert queued is not None
    assert [item.local_path for item in queued.media_files] == [
        video_path.resolve(),
        audio_path.resolve(),
    ]
