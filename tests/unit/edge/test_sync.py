"""Retry and integrity checks for the edge-to-cloud synchronizer."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from mypy_boto3_s3 import S3Client

from mindbridge import AsyncMindBridge, MindBridgeClientError
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
)
from mindbridge.core import MediaKind, ObjectStorageError, SensorKind
from mindbridge.edge import (
    EdgeMediaFile,
    EdgeObservationSynchronizer,
    S3EdgeMediaUploader,
    SQLiteObservationOutbox,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class FailOnceMemoryApi:
    def __init__(self) -> None:
        self.calls = 0

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls += 1
        if self.calls == 1:
            raise MindBridgeClientError("offline", code="transport_error")
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_01",
            idempotency_key=request.idempotency_key or "server_derived_key",
            status=ObservationStatus.DUPLICATE,
            trace_id="trace_01",
        )


class RecordingS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, dict[str, str]]] = []

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, str],
    ) -> None:
        self.calls.append((filename, bucket, key, ExtraArgs))


async def test_sync_retries_only_metadata_after_media_upload(tmp_path: Path) -> None:
    request, media_file = _capture(tmp_path)
    outbox = SQLiteObservationOutbox(tmp_path / "edge.db", clock=lambda: NOW)
    outbox.enqueue(request, (media_file,))
    api = FailOnceMemoryApi()
    upload_calls: list[str] = []

    async def upload(_request: ObserveRequest, file: EdgeMediaFile) -> None:
        upload_calls.append(file.media_object_id)

    synchronizer = EdgeObservationSynchronizer(
        outbox,
        cast(AsyncMindBridge, api),
        upload,
    )
    with pytest.raises(MindBridgeClientError, match="offline"):
        await synchronizer.sync_next()

    pending = outbox.next_pending()
    assert pending is not None
    assert pending.media_uploaded is True
    assert pending.attempts == 1
    assert pending.last_error_code == "transport_error"

    receipt = await synchronizer.sync_next()
    assert receipt is not None
    assert receipt.status is ObservationStatus.DUPLICATE
    assert upload_calls == ["media_01"]
    assert outbox.pending_count() == 0


async def test_s3_uploader_checks_bytes_and_tenant_before_upload(tmp_path: Path) -> None:
    request, media_file = _capture(tmp_path)
    client = RecordingS3Client()
    uploader = S3EdgeMediaUploader("memory", client=cast(S3Client, client))

    await uploader.upload(request, media_file)

    assert client.calls == [
        (
            str(media_file.local_path),
            "memory",
            "tenants/tenant_01/media_01.mp4",
            {"ContentType": "video/mp4", "ChecksumAlgorithm": "SHA256"},
        )
    ]
    changed = request.model_copy(
        update={
            "media_objects": (request.media_objects[0].model_copy(update={"sha256": "0" * 64}),)
        }
    )
    with pytest.raises(ObjectStorageError, match="checksum"):
        await uploader.upload(changed, media_file)


def _capture(tmp_path: Path) -> tuple[ObserveRequest, EdgeMediaFile]:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    media = MediaObjectInput(
        media_object_id="media_01",
        kind=MediaKind.VIDEO,
        uri="s3://memory/tenants/tenant_01/media_01.mp4",
        sha256=hashlib.sha256(b"video").hexdigest(),
        size_bytes=5,
        created_at=NOW,
        duration_ms=30_000,
    )
    request = ObserveRequest(
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=7,
        sensor=SensorKind.CAMERA,
        media_objects=(media,),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=30),
        idempotency_key="edge_observation_01",
    )
    return request, EdgeMediaFile(
        media_object_id=media.media_object_id,
        local_path=media_path,
        content_type="video/mp4",
    )
