"""Tests for evidence-first domain invariants."""

from datetime import datetime, timezone

import pytest

from mindbridge.core import (
    DeviceId,
    DomainInvariantError,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    Observation,
    ObservationId,
    PixelRegion,
    SensorKind,
    TenantId,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
MEDIA_ID = MediaObjectId("media_01")


def test_media_object_rejects_naive_created_at() -> None:
    """Media timestamps must be timezone-aware at the domain boundary."""
    with pytest.raises(DomainInvariantError, match="created_at"):
        MediaObject(
            media_object_id=MEDIA_ID,
            tenant_id=TenantId("tenant_01"),
            kind=MediaKind.VIDEO,
            uri="s3://memories/video.mp4",
            sha256="a" * 64,
            size_bytes=1,
            created_at=datetime(2026, 8, 11, 12, 0),  # noqa: DTZ001 - invalid input
        )


def test_media_object_rejects_invalid_content_hash() -> None:
    """Evidence cannot be stored without a valid content digest."""
    with pytest.raises(DomainInvariantError, match="sha256"):
        MediaObject(
            media_object_id=MEDIA_ID,
            tenant_id=TenantId("tenant_01"),
            kind=MediaKind.VIDEO,
            uri="s3://memories/video.mp4",
            sha256="not-a-sha256",
            size_bytes=1,
            created_at=NOW,
        )


def test_observation_builds_stable_idempotency_key() -> None:
    """Retries of the same device sequence produce the same ingestion key."""
    observation = _observation()

    assert observation.idempotency_key == _observation().idempotency_key
    assert observation.idempotency_key.startswith("observation_")


def test_observation_idempotency_key_changes_with_sequence() -> None:
    """A later device sequence cannot collide with an earlier observation."""
    first = _observation(sequence=1)
    second = _observation(sequence=2)

    assert first.idempotency_key != second.idempotency_key


def test_observation_rejects_values_outside_storage_integer_ranges() -> None:
    with pytest.raises(DomainInvariantError, match="sequence"):
        _observation(sequence=2**63)
    with pytest.raises(DomainInvariantError, match="clock_offset_ms"):
        _observation(clock_offset_ms=2**31)


def test_observation_rejects_reversed_time_range() -> None:
    """An observation cannot end before it occurred."""
    with pytest.raises(DomainInvariantError, match="ended_at"):
        _observation(
            occurred_at=datetime(2026, 8, 11, 12, 1, tzinfo=timezone.utc),
            ended_at=NOW,
        )


def test_observation_rejects_duplicate_media_references() -> None:
    """Duplicate media references must not inflate an observation."""
    with pytest.raises(DomainInvariantError, match="duplicates"):
        _observation(media_object_ids=(MEDIA_ID, MEDIA_ID))


def test_pixel_region_rejects_inverted_coordinates() -> None:
    """A region must have positive width and height."""
    with pytest.raises(DomainInvariantError, match="maximums"):
        PixelRegion(x_min=10, y_min=20, x_max=5, y_max=30)


def test_evidence_span_requires_complete_frame_range() -> None:
    """Frame evidence cannot have only one range boundary."""
    with pytest.raises(DomainInvariantError, match="provided together"):
        _evidence_span(frame_start=10)


def test_evidence_span_reports_duration() -> None:
    """Evidence duration is derived from its source offsets."""
    evidence = _evidence_span(start_ms=1_000, end_ms=2_500)

    assert evidence.duration_ms == 1_500


def _observation(
    *,
    sequence: int = 1,
    media_object_ids: tuple[MediaObjectId, ...] = (MEDIA_ID,),
    occurred_at: datetime = NOW,
    ended_at: datetime = NOW,
    clock_offset_ms: int = 0,
) -> Observation:
    return Observation(
        observation_id=ObservationId("observation_01"),
        tenant_id=TenantId("tenant_01"),
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_object_ids=media_object_ids,
        occurred_at=occurred_at,
        ended_at=ended_at,
        observed_at=NOW,
        clock_offset_ms=clock_offset_ms,
    )


def _evidence_span(
    *,
    start_ms: int = 0,
    end_ms: int = 1_000,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=EvidenceId("evidence_01"),
        tenant_id=TenantId("tenant_01"),
        observation_id=ObservationId("observation_01"),
        media_object_id=MEDIA_ID,
        start_ms=start_ms,
        end_ms=end_ms,
        created_at=NOW,
        frame_start=frame_start,
        frame_end=frame_end,
    )
