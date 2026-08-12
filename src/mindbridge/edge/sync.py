"""Upload queued edge media with Boto3, then submit observations through the SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from mindbridge.contracts import (
    DeletionListRequest,
    MediaObjectInput,
    MemoryResult,
    ObservationReceipt,
    ObserveRequest,
)
from mindbridge.core import JobState, MemoryIntegrityError, ObjectStorageError
from mindbridge.edge.deletion_inbox import SQLiteDeletionInbox
from mindbridge.edge.outbox import EdgeMediaFile, SQLiteObservationOutbox
from mindbridge.edge.recent_memory import SQLiteRecentMemory
from mindbridge.file_integrity import sha256_file
from mindbridge.infrastructure.s3 import tenant_s3_object_key
from mindbridge.sdk import AsyncMindBridge, MindBridgeClientError
from mindbridge.telemetry import set_current_span_attributes, trace_operation

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

UploadEdgeMedia = Callable[[ObserveRequest, EdgeMediaFile], Awaitable[None]]


class S3EdgeMediaUploader:
    """Upload immutable edge files through Boto3's standard credential chain."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        client: S3Client | None = None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        self._bucket = bucket
        self._client: S3Client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    async def upload(self, request: ObserveRequest, media_file: EdgeMediaFile) -> None:
        """Verify declared bytes locally and upload them to the tenant-scoped object key."""
        media = next(
            (
                candidate
                for candidate in request.media_objects
                if candidate.media_object_id == media_file.media_object_id
            ),
            None,
        )
        if media is None:
            raise ValueError("edge media file is not present in its observation request")
        object_key = tenant_s3_object_key(self._bucket, request.tenant_id, media.uri)
        try:
            await asyncio.to_thread(self._upload_verified, media, media_file, object_key)
        except ObjectStorageError:
            raise
        except (BotoCoreError, ClientError, S3UploadFailedError, OSError) as error:
            raise ObjectStorageError("could not upload edge evidence media") from error

    def _upload_verified(
        self,
        media: MediaObjectInput,
        media_file: EdgeMediaFile,
        object_key: str,
    ) -> None:
        if media_file.local_path.stat().st_size != media.size_bytes:
            raise ObjectStorageError("edge media size does not match observation metadata")
        if sha256_file(media_file.local_path).lower() != media.sha256.lower():
            raise ObjectStorageError("edge media checksum does not match observation metadata")
        self._client.upload_file(
            str(media_file.local_path),
            self._bucket,
            object_key,
            ExtraArgs={
                "ContentType": media_file.content_type,
                "ChecksumAlgorithm": "SHA256",
            },
        )


class EdgeObservationSynchronizer:
    """Drain the durable outbox in order with at-least-once network delivery."""

    def __init__(
        self,
        outbox: SQLiteObservationOutbox,
        deletion_inbox: SQLiteDeletionInbox,
        memory: AsyncMindBridge,
        upload_media: UploadEdgeMedia,
        *,
        recent_memory: SQLiteRecentMemory,
    ) -> None:
        self._outbox = outbox
        self._deletion_inbox = deletion_inbox
        self._memory = memory
        self._upload_media = upload_media
        self._recent_memory = recent_memory

    @trace_operation("mindbridge.edge.sync_observation")
    async def sync_next(self) -> ObservationReceipt | None:
        """Synchronize the oldest item, leaving it durable after any network failure."""
        item = self._outbox.next_pending()
        polled_tenant_ids: set[str] = set()
        while item is not None and item.request.tenant_id not in polled_tenant_ids:
            _, caught_up = await self._sync_deletion_pages(item.request.tenant_id)
            if not caught_up:
                return None
            polled_tenant_ids.add(item.request.tenant_id)
            item = self._outbox.next_pending()
        if item is None:
            return None
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": item.request.tenant_id,
                "mindbridge.device.id": item.request.device_id,
                "mindbridge.outbox.attempt": item.attempts,
                "mindbridge.media.count": len(item.media_files),
            }
        )
        if not item.media_uploaded:
            try:
                for media_file in item.media_files:
                    await self._upload_media(item.request, media_file)
            except Exception as error:
                self._outbox.record_failure(item.outbox_id, _network_error_code(error))
                raise
            self._outbox.mark_media_uploaded(item.outbox_id)
        try:
            receipt = await self._memory.observe(item.request)
        except Exception as error:
            self._outbox.record_failure(item.outbox_id, _network_error_code(error))
            raise
        self._outbox.acknowledge(item, receipt)
        return receipt

    @trace_operation("mindbridge.edge.sync_deletions")
    async def sync_deletions(
        self,
        tenant_id: str,
        *,
        page_limit: int = 100,
        max_pages: int = 10,
    ) -> int:
        """Apply a bounded ordered batch of cloud tombstones before any upload."""
        applied, _ = await self._sync_deletion_pages(
            tenant_id,
            page_limit=page_limit,
            max_pages=max_pages,
        )
        return applied

    async def _sync_deletion_pages(
        self,
        tenant_id: str,
        *,
        page_limit: int = 100,
        max_pages: int = 10,
    ) -> tuple[int, bool]:
        if not 1 <= page_limit <= 100 or max_pages <= 0:
            raise ValueError("deletion page_limit must be 1..100 and max_pages must be positive")
        cursor = self._deletion_inbox.read_cursor(tenant_id)
        applied = 0
        for _ in range(max_pages):
            page = await self._memory.list_deletions(
                DeletionListRequest(
                    tenant_id=tenant_id,
                    cursor=cursor,
                    limit=page_limit,
                )
            )
            applied += self._deletion_inbox.apply_page(tenant_id, page)
            if page.next_cursor is None:
                return applied, True
            cursor = page.next_cursor
        return applied, False

    @trace_operation("mindbridge.edge.sync_recent_memories")
    async def sync_recent_memories(self, *, limit: int = 100) -> int:
        """Cache completed cloud jobs without blocking on jobs still in progress."""
        cached = 0
        for pending in self._outbox.pending_processing_jobs(limit=limit):
            self._outbox.mark_processing_job_polled(pending)
            job = await self._memory.get_observation_job(
                pending.tenant_id,
                pending.processing_job_id,
            )
            if (
                job.job_id != pending.processing_job_id
                or job.observation_id != pending.observation_id
            ):
                raise MemoryIntegrityError("cloud processing job identity changed")
            if job.state is not JobState.SUCCEEDED:
                continue
            fetched: list[MemoryResult] = []
            fetched_ids: list[str] = []
            for memory_id in job.memory_ids:
                try:
                    memory = await self._memory.get_memory(pending.tenant_id, memory_id)
                except MindBridgeClientError as error:
                    if error.code == "memory_deleted":
                        continue
                    raise
                if memory.memory_id != memory_id:
                    raise MemoryIntegrityError("cloud memory identity changed")
                fetched.append(memory)
                fetched_ids.append(memory_id)
            self._recent_memory.cache_job_memories(
                pending.tenant_id,
                pending.observation_id,
                pending.processing_job_id,
                tuple(fetched_ids),
                tuple(fetched),
            )
            cached += len(fetched)
        return cached

    @trace_operation("mindbridge.edge.sync_pending")
    async def sync_pending(self, *, limit: int = 100) -> tuple[ObservationReceipt, ...]:
        """Synchronize at most `limit` items so the caller controls scheduling and backoff."""
        if limit <= 0:
            raise ValueError("edge sync limit must be positive")
        for tenant_id in self._deletion_inbox.tenant_ids():
            await self.sync_deletions(tenant_id)
        await self.sync_recent_memories(limit=limit)
        receipts: list[ObservationReceipt] = []
        for _ in range(limit):
            receipt = await self.sync_next()
            if receipt is None:
                break
            receipts.append(receipt)
        return tuple(receipts)


def _network_error_code(error: Exception) -> str:
    if isinstance(error, MindBridgeClientError):
        return error.code
    if isinstance(error, ObjectStorageError):
        return "object_storage_error"
    return type(error).__name__[:255]
