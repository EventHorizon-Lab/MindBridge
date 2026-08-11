"""Durability and idempotency checks for the Jetson SQLite outbox."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge.contracts import (
    MediaObjectInput,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
)
from mindbridge.core import IdempotencyConflictError, MediaKind, SensorKind
from mindbridge.edge import EdgeMediaFile, SQLiteObservationOutbox

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_outbox_survives_restart_and_rejects_sequence_content_conflicts(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    database_path = tmp_path / "edge.db"
    outbox = SQLiteObservationOutbox(database_path, clock=lambda: NOW)
    request = _request()
    files = (_file(media_path),)

    assert outbox.enqueue(request, files) is True
    assert outbox.enqueue(request, files) is False
    restarted = SQLiteObservationOutbox(database_path, clock=lambda: NOW)
    item = restarted.next_pending()
    assert item is not None
    assert item.request == request
    assert item.media_files == files
    assert database_path.stat().st_mode & 0o777 == 0o600

    changed = request.model_copy(update={"ended_at": request.ended_at + timedelta(seconds=1)})
    with pytest.raises(IdempotencyConflictError, match="different content"):
        restarted.enqueue(changed, files)


def test_acknowledgement_advances_watermark_and_prevents_requeue(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)
    request = _request()
    files = (_file(media_path),)
    outbox.enqueue(request, files)
    item = outbox.next_pending()
    assert item is not None

    outbox.mark_media_uploaded(item.outbox_id)
    outbox.record_failure(item.outbox_id, "transport_error")
    failed = outbox.next_pending()
    assert failed is not None
    assert failed.media_uploaded is True
    assert failed.attempts == 1
    assert failed.last_error_code == "transport_error"

    receipt = ObservationReceipt(
        observation_id="observation_01",
        processing_job_id="job_01",
        idempotency_key="edge_observation_01",
        status=ObservationStatus.ACCEPTED,
        trace_id="trace_01",
    )
    outbox.acknowledge(failed, receipt)
    outbox.acknowledge(failed, receipt)

    watermark = outbox.read_watermark("tenant_01", "camera_01", "boot_01")
    assert watermark is not None
    assert watermark.sequence == 7
    assert watermark.observation_id == "observation_01"
    assert outbox.pending_count() == 0
    assert outbox.enqueue(request, files) is False


def _request() -> ObserveRequest:
    return ObserveRequest(
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=7,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id="media_01",
                kind=MediaKind.VIDEO,
                uri="s3://memory/tenants/tenant_01/media_01.mp4",
                sha256="00" * 32,
                size_bytes=5,
                created_at=NOW,
                duration_ms=30_000,
            ),
        ),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=30),
    )


def _file(path: Path) -> EdgeMediaFile:
    return EdgeMediaFile(
        media_object_id="media_01",
        local_path=path,
        content_type="video/mp4",
    )
