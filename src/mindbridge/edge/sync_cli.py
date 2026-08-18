"""One-shot edge outbox drain suitable for systemd restart and backoff policies."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from mindbridge.cli import parser as build_parser
from mindbridge.edge.deletion_inbox import SQLiteDeletionInbox
from mindbridge.edge.outbox import SQLiteObservationOutbox
from mindbridge.edge.recent_memory import SQLiteRecentMemory
from mindbridge.edge.sync import EdgeObservationSynchronizer, S3EdgeMediaUploader
from mindbridge.sdk import MindBridge
from mindbridge.telemetry import configure_telemetry

EDGE_ENVIRONMENT = """environment:
  MINDBRIDGE_API_KEY  bearer token for --api-base-url; read from the environment so the
                      credential never reaches a process list or a systemd unit's argv"""


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Upload and submit a bounded batch, leaving failures durable for the next run."""
    parser = build_parser(prog=prog, description=__doc__, epilog=EDGE_ENVIRONMENT)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="SQLite file holding this device's outbox, deletion inbox, and recent memory",
    )
    parser.add_argument(
        "--api-base-url", required=True, help="base URL of the MindBridge API to submit to"
    )
    parser.add_argument("--bucket", required=True, help="object storage bucket to upload media to")
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint; omit for AWS itself")
    parser.add_argument("--region", default="us-east-1", help="region of that bucket")
    parser.add_argument(
        "--limit", type=int, default=100, help="most observations to submit in this run"
    )
    parser.add_argument(
        "--recent-retention-hours",
        type=float,
        default=24.0,
        help="how long this device keeps its own recall cache",
    )
    parser.add_argument(
        "--tenant-id",
        action="append",
        default=[],
        help="tenant whose deletions to pull before submitting; repeatable",
    )
    arguments = parser.parse_args(argv)
    # Configured after parsing so --help and a rejected flag stay side-effect free.
    configure_telemetry("mindbridge-edge")
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
