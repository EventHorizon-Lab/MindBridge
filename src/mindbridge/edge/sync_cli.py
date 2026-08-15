"""One-shot edge outbox drain suitable for systemd restart and backoff policies."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

from mindbridge.edge.deletion_inbox import SQLiteDeletionInbox
from mindbridge.edge.outbox import SQLiteObservationOutbox
from mindbridge.edge.recent_memory import SQLiteRecentMemory
from mindbridge.edge.sync import EdgeObservationSynchronizer, S3EdgeMediaUploader
from mindbridge.sdk import MindBridge
from mindbridge.telemetry import configure_telemetry


def main() -> None:
    """Upload and submit a bounded batch, leaving failures durable for the next run."""
    configure_telemetry("mindbridge-edge")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint-url")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--recent-retention-hours", type=float, default=24.0)
    parser.add_argument("--tenant-id", action="append", default=[])
    arguments = parser.parse_args()
    synchronized, pending = asyncio.run(
        _synchronize(
            database_path=arguments.database,
            api_base_url=arguments.api_base_url,
            bucket=arguments.bucket,
            endpoint_url=arguments.endpoint_url,
            region=arguments.region,
            limit=arguments.limit,
            recent_retention_hours=arguments.recent_retention_hours,
            tenant_ids=tuple(arguments.tenant_id),
        )
    )
    print(json.dumps({"synchronized": synchronized, "pending": pending}))


async def _synchronize(
    *,
    database_path: Path,
    api_base_url: str,
    bucket: str,
    endpoint_url: str | None,
    region: str,
    limit: int,
    recent_retention_hours: float,
    tenant_ids: tuple[str, ...],
) -> tuple[int, int]:
    outbox = SQLiteObservationOutbox(database_path)
    deletion_inbox = SQLiteDeletionInbox(database_path)
    recent_memory = SQLiteRecentMemory(
        database_path,
        retention=timedelta(hours=recent_retention_hours),
    )
    memory = MindBridge.connect(
        base_url=api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
    )
    uploader = S3EdgeMediaUploader(
        bucket,
        endpoint_url=endpoint_url,
        region_name=region,
    )
    try:
        synchronizer = EdgeObservationSynchronizer(
            outbox,
            deletion_inbox,
            memory,
            uploader.upload,
            recent_memory=recent_memory,
        )
        for tenant_id in tenant_ids:
            await synchronizer.sync_deletions(tenant_id)
        receipts = await synchronizer.sync_pending(limit=limit)
        return len(receipts), outbox.pending_count()
    finally:
        await memory.close()


if __name__ == "__main__":
    main()
