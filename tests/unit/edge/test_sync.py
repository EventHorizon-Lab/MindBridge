"""Retry and integrity checks for the edge-to-cloud synchronizer."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from mypy_boto3_s3 import S3Client

from mindbridge import AsyncMindBridge, MindBridgeClientError
from mindbridge.contracts import (
    DeletionListRequest,
    DeletionPage,
    DeletionTombstoneView,
    MediaObjectInput,
    MemoryResult,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
)
from mindbridge.core import (
    DeletionPropagationState,
    ForgetTargetType,
    JobState,
    MediaKind,
    MemoryState,
    MemoryType,
    ObjectStorageError,
    SensorKind,
    VerificationStatus,
    derive_observation_id,
)
from mindbridge.edge import (
    EdgeMediaFile,
    EdgeObservationSynchronizer,
    S3EdgeMediaUploader,
    SQLiteDeletionInbox,
    SQLiteObservationOutbox,
    SQLiteRecentMemory,
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
            observation_id=derive_observation_id(
                request.tenant_id,
                request.device_id,
                request.boot_id,
                request.sequence,
            ),
            processing_job_id="job_01",
            idempotency_key=request.idempotency_key or "server_derived_key",
            status=ObservationStatus.DUPLICATE,
            trace_id="trace_01",
        )

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        return DeletionPage(items=(), next_cursor=None, trace_id="trace_deletions")


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


class DeletingMemoryApi:
    def __init__(self, request: ObserveRequest) -> None:
        self.request = request
        self.observe_calls = 0

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.observe_calls += 1
        raise AssertionError("a tombstoned observation must not be uploaded")

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        observation_id = derive_observation_id(
            self.request.tenant_id,
            self.request.device_id,
            self.request.boot_id,
            self.request.sequence,
        )
        return DeletionPage(
            items=(
                DeletionTombstoneView(
                    tombstone_id="tombstone_01",
                    target_type=ForgetTargetType.OBSERVATION,
                    target_id=observation_id,
                    propagation_state=DeletionPropagationState.COMPLETE,
                    requested_at=NOW,
                    completed_at=NOW,
                    error_code=None,
                ),
            ),
            next_cursor=None,
            trace_id="trace_deletion_page",
        )


class CompletedMemoryApi:
    def __init__(
        self,
        receipt: ObservationReceipt,
        memory: MemoryResult,
    ) -> None:
        self.receipt = receipt
        self.memory = memory
        self.job_calls = 0

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        raise AssertionError("the acknowledged observation must not be uploaded again")

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        return DeletionPage(items=(), next_cursor=None, trace_id="trace_deletions")

    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        self.job_calls += 1
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id=self.receipt.observation_id,
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            memory_ids=("memory_deleted", self.memory.memory_id),
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
        if memory_id == "memory_deleted":
            raise MindBridgeClientError(
                "memory was deleted",
                code="memory_deleted",
                status_code=410,
            )
        assert memory_id == self.memory.memory_id
        return self.memory


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
        SQLiteDeletionInbox(tmp_path / "edge.db", clock=lambda: NOW),
        cast(AsyncMindBridge, api),
        upload,
        recent_memory=SQLiteRecentMemory(tmp_path / "edge.db", clock=lambda: NOW),
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


async def test_sync_applies_tombstone_before_media_upload(tmp_path: Path) -> None:
    request, media_file = _capture(tmp_path)
    database_path = tmp_path / "edge.db"
    outbox = SQLiteObservationOutbox(database_path, clock=lambda: NOW)
    outbox.enqueue(request, (media_file,))
    deletion_inbox = SQLiteDeletionInbox(database_path, clock=lambda: NOW)
    api = DeletingMemoryApi(request)
    upload_calls: list[str] = []

    async def upload(_request: ObserveRequest, file: EdgeMediaFile) -> None:
        upload_calls.append(file.media_object_id)

    receipt = await EdgeObservationSynchronizer(
        outbox,
        deletion_inbox,
        cast(AsyncMindBridge, api),
        upload,
        recent_memory=SQLiteRecentMemory(database_path, clock=lambda: NOW),
    ).sync_next()

    assert receipt is None
    assert api.observe_calls == 0
    assert upload_calls == []
    assert outbox.pending_count() == 0
    assert not media_file.local_path.exists()


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


async def test_sync_pending_caches_completed_cloud_memory(tmp_path: Path) -> None:
    request, media_file = _capture(tmp_path)
    database_path = tmp_path / "edge.db"
    outbox = SQLiteObservationOutbox(database_path, clock=lambda: NOW)
    outbox.enqueue(request, (media_file,))
    item = outbox.next_pending()
    assert item is not None
    observation_id = derive_observation_id(
        request.tenant_id,
        request.device_id,
        request.boot_id,
        request.sequence,
    )
    receipt = ObservationReceipt(
        observation_id=observation_id,
        processing_job_id=f"job_process_{observation_id}",
        idempotency_key=request.idempotency_key or "unexpected",
        status=ObservationStatus.ACCEPTED,
        trace_id="trace_observe",
    )
    outbox.acknowledge(item, receipt)
    memory = MemoryResult(
        memory_id="memory_01",
        memory_type=MemoryType.EPISODIC,
        summary="A person placed a tool beside the toolbox.",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        created_at=NOW,
        verification_status=VerificationStatus.VERIFIED,
        state=MemoryState.ACTIVE,
        trace_id="trace_memory",
    )
    api = CompletedMemoryApi(receipt, memory)
    recent = SQLiteRecentMemory(database_path, clock=lambda: NOW)

    async def unexpected_upload(_request: ObserveRequest, _file: EdgeMediaFile) -> None:
        raise AssertionError("no media upload is expected")

    receipts = await EdgeObservationSynchronizer(
        outbox,
        SQLiteDeletionInbox(database_path, clock=lambda: NOW),
        cast(AsyncMindBridge, api),
        unexpected_upload,
        recent_memory=recent,
    ).sync_pending()

    assert receipts == ()
    assert api.job_calls == 1
    assert outbox.pending_processing_jobs() == ()
    cached = recent.get_memory("tenant_01", "memory_01")
    assert cached is not None
    assert cached.memory_id == memory.memory_id
    assert recent.get_memory("tenant_01", "memory_deleted") is None


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
