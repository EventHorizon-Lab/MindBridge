"""Durable capture and synchronization primitives for Jetson and robot hosts."""

from typing import TYPE_CHECKING

from mindbridge.edge.capture import enqueue_captured_video
from mindbridge.edge.deletion_inbox import SQLiteDeletionInbox
from mindbridge.edge.identity import (
    LocalIdentityMatch,
    LocalIdentitySample,
    SQLiteIdentityMemory,
)
from mindbridge.edge.outbox import (
    EdgeMediaFile,
    EdgeObservationOutboxItem,
    EdgeProcessingJob,
    EdgeSyncWatermark,
    SQLiteObservationOutbox,
)
from mindbridge.edge.recent_memory import SQLiteRecentMemory

if TYPE_CHECKING:
    from mindbridge.edge.sync import EdgeObservationSynchronizer, S3EdgeMediaUploader

__all__ = [
    "EdgeMediaFile",
    "EdgeObservationOutboxItem",
    "EdgeObservationSynchronizer",
    "EdgeProcessingJob",
    "EdgeSyncWatermark",
    "LocalIdentityMatch",
    "LocalIdentitySample",
    "S3EdgeMediaUploader",
    "SQLiteDeletionInbox",
    "SQLiteIdentityMemory",
    "SQLiteObservationOutbox",
    "SQLiteRecentMemory",
    "enqueue_captured_video",
]


def __getattr__(name: str) -> object:
    """Load network synchronization only when its public adapter is requested."""
    if name in {"EdgeObservationSynchronizer", "S3EdgeMediaUploader"}:
        from mindbridge.edge.sync import EdgeObservationSynchronizer, S3EdgeMediaUploader

        return {
            "EdgeObservationSynchronizer": EdgeObservationSynchronizer,
            "S3EdgeMediaUploader": S3EdgeMediaUploader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
