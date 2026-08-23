"""Reconcile the observation job ledger with the broker, and report what it has cost.

PostgreSQL is the documented authority for job state and payload; the broker only carries the
ID of work already recorded there. The two can drift apart, and the drift is silent in both
directions. `task_acks_late=True` acks a message the moment the task raises, so any exception
outside the Worker's `autoretry_for` -- `torch.OutOfMemoryError` during the 2026-08-21
evaluation, 479 of them, and the documented `WorkerLostError` -- discards the message while the
row stays claimable. Republishing from the ledger is the sanctioned repair, and it repairs any
such divergence rather than one exception class at a time.

Reporting is the default because republishing spends money: each message that lands runs a
generator. `--republish` is the flag that acts, and it publishes only the rows the queue cannot
already hold, because duplicating a healthy backlog costs more than the repair saves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import asdict

import psycopg
from celery import Celery

from mindbridge.cli import parser as build_parser
from mindbridge.configuration import require_environment_value
from mindbridge.core import JobId, ObservationId, TenantId
from mindbridge.infrastructure._postgres_jobs import (
    ObservationJobAccounting,
    observation_job_accounting,
    tenant_scope_required,
    unreachable_observation_jobs,
)
from mindbridge.infrastructure.task_queue import (
    CeleryObservationJobPublisher,
    create_task_queue,
)
from mindbridge.telemetry import configure_telemetry

JOBS_ENVIRONMENT = """environment:
  MINDBRIDGE_DATABASE_URL          PostgreSQL DSN (required). Read from the environment
                                   rather than a flag so the DSN never reaches a process
                                   list or this shell's history.
  MINDBRIDGE_TASK_BROKER_URL       Celery broker URL (required). The queue read is the
                                   one the Worker consumes, `task_default_queue`."""


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Report the ledger against the broker, and republish from it when asked."""
    options = _parser(prog).parse_args(argv)
    # Configured after parsing so --help and a rejected flag stay side-effect free.
    configure_telemetry("mindbridge-jobs")
    report = asyncio.run(
        reconcile(
            require_environment_value(os.environ, "MINDBRIDGE_DATABASE_URL"),
            require_environment_value(os.environ, "MINDBRIDGE_TASK_BROKER_URL"),
            tenant_id=TenantId(options.tenant_id) if options.tenant_id else None,
            include_failed=options.include_failed,
            republish=options.republish,
        )
    )
    print(json.dumps(report, sort_keys=True))


def queue_depth(task_queue: Celery) -> int:
    """Count the messages waiting on the queue the Worker consumes.

    The queue is `task_default_queue`, not `celery`: asking about `celery` during the
    2026-08-21 evaluation produced a NOT_FOUND that read as "the queue was destroyed". A virtual
    transport also deletes an empty list rather than keeping it, so NOT_FOUND on the right name
    means empty, not missing.
    """
    queue = str(task_queue.conf.task_default_queue)
    with task_queue.connection_for_read() as connection:
        try:
            return int(connection.default_channel.queue_declare(queue=queue, passive=True)[1])
        except connection.channel_errors:
            return 0


async def reconcile(
    database_url: str,
    broker_url: str,
    *,
    tenant_id: TenantId | None,
    include_failed: bool,
    republish: bool,
) -> dict[str, object]:
    """Compare the ledger with the broker, optionally repair it, and describe both."""
    task_queue = create_task_queue(broker_url)
    async with await psycopg.AsyncConnection.connect(database_url) as connection:
        if tenant_id is None and await tenant_scope_required(connection):
            raise PermissionError(
                "row-level security confines this role to one tenant, so a ledger-wide scan "
                "would report an empty ledger; pass --tenant-id, or connect as the role that "
                "runs the migrations"
            )
        if tenant_id is not None:
            await connection.execute(
                "SELECT set_config('mindbridge.tenant_id', %s, false)",
                (tenant_id,),
            )
        unreachable = await unreachable_observation_jobs(
            connection,
            tenant_id=tenant_id,
            include_failed=include_failed,
        )
        accounting = await observation_job_accounting(connection, tenant_id=tenant_id)
    depth = queue_depth(task_queue)
    return {
        "queue": str(task_queue.conf.task_default_queue),
        # Read before publishing, because kombu publishes with LPUSH and consumes with RPOP: a
        # republished job waits behind every message already queued. Six republishes of one job
        # across 84 minutes moved its attempt count not at all for exactly this reason.
        "queue_depth": depth,
        "claimable": len(unreachable),
        "include_failed": include_failed,
        "republished": (
            await _republish(task_queue, unreachable, already_queued=depth) if republish else 0
        ),
        "tenants": [_accounting_dict(tenant) for tenant in accounting],
    }


async def _republish(
    task_queue: Celery,
    unreachable: Sequence[tuple[TenantId, ObservationId, JobId]],
    *,
    already_queued: int,
) -> int:
    """Publish one message per claimable row the queue cannot already hold, oldest first.

    The ledger says a row is owed work; it cannot say whether a message for it survives, and no
    transport answers that per message. The count does: the queue holds `already_queued`
    messages for this job type and nothing else, so at most `claimable - already_queued` rows
    have none. Publishing to that bound repairs a queue that lost messages and leaves a healthy
    backlog alone.

    Both halves matter because a duplicate is not a no-op. The delivery that loses the claim
    gets `RUNNING`, and the task re-queues itself every 30 seconds up to 40 times waiting for the
    winner, so one duplicate is up to 40 more round trips. Republishing a deep backlog wholesale
    is how a repair becomes the outage.

    Oldest first, because kombu consumes with `RPOP`: the messages still queued are the most
    recently published, so the rows whose message is gone are at the front of this list. Order
    is preserved anyway, because a question can need every observation before its cutoff.
    """
    owed = unreachable[: max(len(unreachable) - already_queued, 0)]
    publisher = CeleryObservationJobPublisher(task_queue)
    for tenant_id, observation_id, job_id in owed:
        await publisher.publish_observation_processing_job(tenant_id, observation_id, job_id)
    return len(owed)


def _accounting_dict(accounting: ObservationJobAccounting) -> dict[str, object]:
    return {
        **asdict(accounting),
        "queue_wait_seconds": round(accounting.queue_wait_seconds, 3),
        "work_seconds": round(accounting.work_seconds, 3),
    }


def _parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = build_parser(prog=prog, description=__doc__, epilog=JOBS_ENVIRONMENT)
    parser.add_argument(
        "--tenant-id",
        help=(
            "restrict the report and the repair to one tenant; required when the database "
            "role's reads are confined by row-level security"
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "also count and republish failed rows. They are re-runnable, but a deterministic "
            "failure republished on a timer pays for the same rejection every time"
        ),
    )
    parser.add_argument(
        "--republish",
        action="store_true",
        help=(
            "publish one message per claimable row the queue cannot already hold, oldest "
            "first. Without it nothing is written, because each message that lands spends "
            "generator tokens"
        ),
    )
    return parser


if __name__ == "__main__":
    main()
