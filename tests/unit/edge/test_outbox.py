"""Durability and idempotency checks for the Jetson SQLite outbox."""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge.contracts import (
    MediaObjectInput,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
)
from mindbridge.core import (
    IdempotencyConflictError,
    MediaKind,
    MemoryIntegrityError,
    SensorKind,
    derive_observation_id,
)
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


def test_acknowledgement_keeps_watermark_without_dropping_late_work(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)
    request = _request().model_copy(update={"sequence": 8})
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
        observation_id=derive_observation_id("tenant_01", "camera_01", "boot_01", 8),
        processing_job_id="job_01",
        idempotency_key="edge_observation_01",
        status=ObservationStatus.ACCEPTED,
        trace_id="trace_01",
    )
    with pytest.raises(MemoryIntegrityError, match="observation ID"):
        outbox.acknowledge(
            failed,
            receipt.model_copy(update={"observation_id": "unexpected_observation"}),
        )
    outbox.acknowledge(failed, receipt)
    outbox.acknowledge(failed, receipt)

    watermark = outbox.read_watermark("tenant_01", "camera_01", "boot_01")
    assert watermark is not None
    assert watermark.sequence == 8
    assert watermark.observation_id == receipt.observation_id
    jobs = SQLiteObservationOutbox(
        tmp_path / "edge.db", clock=lambda: NOW
    ).pending_processing_jobs()
    assert len(jobs) == 1
    assert jobs[0].tenant_id == "tenant_01"
    assert jobs[0].observation_id == receipt.observation_id
    assert jobs[0].processing_job_id == "job_01"
    assert outbox.pending_count() == 0
    assert outbox.enqueue(_request(), files) is True
    late = outbox.next_pending()
    assert late is not None
    assert late.request.sequence == 7


def test_outbox_upgrades_existing_processing_jobs_for_fair_polling(tmp_path: Path) -> None:
    database_path = tmp_path / "edge.db"
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE edge_processing_jobs (
                tenant_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                processing_job_id TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, processing_job_id),
                UNIQUE (tenant_id, observation_id)
            );
            PRAGMA user_version = 3;
            """
        )

    SQLiteObservationOutbox(database_path, clock=lambda: NOW)

    with closing(sqlite3.connect(database_path)) as connection, connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(edge_processing_jobs)")}
        version = connection.execute("PRAGMA user_version").fetchone()
    assert "last_polled_at" in columns
    assert version == (4,)


def test_polled_processing_job_yields_to_unpolled_work(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)
    request = _request()
    for sequence, job_id in ((7, "job_01"), (8, "job_02")):
        queued = request.model_copy(update={"sequence": sequence})
        outbox.enqueue(queued, (_file(media_path),))
        item = outbox.next_pending()
        assert item is not None
        outbox.acknowledge(
            item,
            ObservationReceipt(
                observation_id=derive_observation_id(
                    queued.tenant_id,
                    queued.device_id,
                    queued.boot_id,
                    queued.sequence,
                ),
                processing_job_id=job_id,
                idempotency_key=f"edge_observation_{sequence}",
                status=ObservationStatus.ACCEPTED,
                trace_id=f"trace_{job_id}",
            ),
        )

    first = outbox.pending_processing_jobs(limit=1)[0]
    outbox.mark_processing_job_polled(first)

    assert first.processing_job_id == "job_01"
    assert outbox.pending_processing_jobs(limit=1)[0].processing_job_id == "job_02"


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
