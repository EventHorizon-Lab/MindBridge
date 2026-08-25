"""Reconcile the observation job ledger with the broker, and report what it has cost.

PostgreSQL is the documented authority for job state and payload; the broker only carries the
ID of work already recorded there. The two can drift apart, and the drift is silent in both
directions. `task_acks_late=True` acks a message the moment the task raises, so any exception
outside the Worker's `autoretry_for` -- `torch.OutOfMemoryError` during the 2026-08-21
evaluation, 479 of them, and the documented `WorkerLostError` -- discards the message while the
row stays claimable. Republishing from the ledger is the sanctioned repair, and it repairs any
such divergence rather than one exception class at a time.

Reporting is the default because republishing spends money: each message that lands runs a
generator. `--republish` is the flag that acts, and ledger-wide it publishes only the rows the
queue cannot already hold, because duplicating a healthy backlog costs more than the repair
saves. Under `--tenant-id` that subtraction would set one tenant's rows against every tenant's
messages, so it is not made, and the report carries the count it withheld either way.
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
    RECOVERED_WORK_PRIORITY,
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
    withheld = _withheld(depth, len(unreachable), tenant_id=tenant_id)
    return {
        "queue": str(task_queue.conf.task_default_queue),
        # Read before publishing, so the depth describes the queue this repair is deciding
        # against rather than the one it just changed. A republished job no longer waits behind
        # the backlog -- it goes to `RECOVERED_WORK_PRIORITY`, which is drained first -- but the
        # withholding decision still needs the pre-publish depth to be meaningful. Before that
        # priority split, six republishes of one job across 84 minutes moved its attempt count
        # not at all, because every one of them landed behind the same backlog.
        "queue_depth": depth,
        "claimable": len(unreachable),
        "withheld": withheld,
        "include_failed": include_failed,
        "republished": (
            await _republish(task_queue, unreachable, already_queued=withheld) if republish else 0
        ),
        "tenants": [_accounting_dict(tenant) for tenant in accounting],
    }


def _withheld(depth: int, claimable: int, *, tenant_id: TenantId | None) -> int:
    """Count the claimable rows a repair treats as already carried by the queue.

    Ledger-wide the queue depth is that count: the queue carries this job type and nothing else,
    so `claimable - depth` rows can have no message left. A scoped scan breaks the arithmetic
    rather than narrowing it -- `claimable` is one tenant's, the depth is every tenant's, and
    five messages for tenant B would cancel five stranded rows for tenant A and publish nothing.
    Nothing recovers the missing term: a queue count per tenant needs each message read, which
    for every transport kombu speaks means taking it off the queue.

    So a scoped repair withholds nothing, and the report says so. The two errors are not
    symmetric: a duplicate costs broker round trips and is self-limiting, while a strand costs
    the work outright and waits for a human to notice it twice. The tool that publishes a few
    duplicates beats the one that silently repaired nothing for 479 stranded rows.
    """
    return min(depth, claimable) if tenant_id is None else 0


async def _republish(
    task_queue: Celery,
    unreachable: Sequence[tuple[TenantId, ObservationId, JobId]],
    *,
    already_queued: int,
) -> int:
    """Publish one message per claimable row past `already_queued`, oldest first.

    The ledger says a row is owed work; it cannot say whether a message for it survives, and no
    transport answers that per message. A count of the rows the queue already covers stands in
    for it -- `_withheld` decides how many that is -- so at most `claimable - already_queued`
    rows have none. Publishing to that bound repairs a queue that lost messages and leaves a
    healthy backlog alone.

    Both halves matter because a duplicate is not a no-op. The delivery that loses the claim
    gets `RUNNING`, and the task re-queues itself every 30 seconds up to 40 times waiting for the
    winner, so one duplicate is up to 40 more round trips. Republishing a deep backlog wholesale
    is how a repair becomes the outage.

    Oldest first, because kombu consumes with `RPOP`: the messages still queued are the most
    recently published, so the rows whose message is gone are at the front of this list. Order
    is preserved anyway, because a question can need every observation before its cutoff.

    Published at `RECOVERED_WORK_PRIORITY`, which the transport drains ahead of the fresh work
    at `FRESH_WORK_PRIORITY`. Repair is work the ledger already owes and has already waited for,
    so serving it behind a backlog is what made a repair indistinguishable from doing nothing.
    """
    owed = unreachable[: max(len(unreachable) - already_queued, 0)]
    publisher = CeleryObservationJobPublisher(task_queue)
    for tenant_id, observation_id, job_id in owed:
        await publisher.publish_observation_processing_job(
            tenant_id,
            observation_id,
            job_id,
            priority=RECOVERED_WORK_PRIORITY,
        )
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
            "first -- and every claimable row under --tenant-id, whose rows the queue depth "
            "does not count. Without it nothing is written, because each message that lands "
            "spends generator tokens"
        ),
    )
    return parser


if __name__ == "__main__":
    main()
