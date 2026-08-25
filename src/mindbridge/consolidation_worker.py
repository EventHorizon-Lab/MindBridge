"""Celery beat scheduling for the consolidation sweeps nothing else was running.

`mindbridge consolidate` has always been a one-shot command, and a nine-benchmark evaluation
produced zero summary-layer records because no control plane was calling it: recall only ever
saw individual clips. Celery is already the task runtime and beat ships with it, so the schedule
lives here rather than behind a second scheduler.

Two properties keep a scheduled sweep from starving the generator the write path shares. One
tick sweeps one tenant, chosen by rotation, so the configured interval bounds consolidation's
whole share of the endpoint however many tenants are listed. And the task is routed to its own
queue, so the worker running it is a different process from the one consuming observations.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from celery import Celery

from mindbridge.configuration import optional_environment_value
from mindbridge.consolidation_cli import (
    ConsolidationSettings,
    run_sweep,
    summary_dict,
    sweep_options,
)
from mindbridge.core import TenantId, utc_now
from mindbridge.telemetry import logger

CONSOLIDATE_TENANT_TASK = "mindbridge.consolidate_tenant"

CONSOLIDATION_QUEUE = "mindbridge_consolidation"
"""Kept off `task_default_queue` so a sweep never occupies an observation worker.

A sweep is minutes of generator calls and the observation queue runs at a prefetch of one, so
sharing the queue would stall ingest for the length of every sweep. Consume it with a worker
started as `-Q mindbridge_consolidation`.
"""

DEFAULT_CONSOLIDATION_INTERVAL_SECONDS = 3_600.0
"""Hourly: the frequent end of the cadence the deployment guide already recommends."""

TENANT_IDS_VARIABLE = "MINDBRIDGE_CONSOLIDATION_TENANT_IDS"
INTERVAL_VARIABLE = "MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS"

_LOGGER = logger("mindbridge.consolidation_worker")


@dataclass(frozen=True, slots=True)
class ConsolidationSchedule:
    """Which tenants a beat tick may sweep, and how often a tick comes."""

    tenant_ids: tuple[TenantId, ...]
    interval_seconds: float

    def __post_init__(self) -> None:
        if not self.tenant_ids:
            raise ValueError(f"{TENANT_IDS_VARIABLE} must name at least one tenant")
        if len(set(self.tenant_ids)) != len(self.tenant_ids):
            raise ValueError(f"{TENANT_IDS_VARIABLE} must not repeat a tenant")
        if not self.interval_seconds > 0:
            raise ValueError(f"{INTERVAL_VARIABLE} must be a positive number of seconds")

    def due_tenant(self, now: datetime) -> TenantId:
        """Pick this tick's tenant from the clock, so no scheduler state has to be stored.

        Beat and every prefork child that might run the task agree on the answer without
        sharing anything: the tick index is the wall clock divided by the interval. A missed
        or a duplicated tick therefore skips or repeats one tenant rather than desynchronising
        a stored counter, and both are harmless -- the sweeps are idempotent, and a skipped
        tenant comes round again one rotation later.
        """
        tick = int(now.timestamp() // self.interval_seconds)
        return self.tenant_ids[tick % len(self.tenant_ids)]


def consolidation_schedule(environ: Mapping[str, str]) -> ConsolidationSchedule | None:
    """Read the schedule, treating an absent tenant list as "no scheduled consolidation".

    Naming the tenants is the on-switch as well as the bound. There is no cross-tenant scan to
    discover them with -- row-level security returns no rows to a confined role rather than
    failing -- and a sweep that ran against every tenant it did find would be the load spike
    this schedule exists to avoid.
    """
    raw_tenants = optional_environment_value(environ, TENANT_IDS_VARIABLE)
    if raw_tenants is None:
        return None
    raw_interval = optional_environment_value(environ, INTERVAL_VARIABLE)
    try:
        interval = (
            DEFAULT_CONSOLIDATION_INTERVAL_SECONDS if raw_interval is None else float(raw_interval)
        )
    except ValueError as error:
        raise ValueError(f"{INTERVAL_VARIABLE} must be a number of seconds") from error
    tenant_ids = tuple(TenantId(value.strip()) for value in raw_tenants.split(",") if value.strip())
    return ConsolidationSchedule(tenant_ids=tenant_ids, interval_seconds=interval)


def register_consolidation_schedule(
    app: Celery,
    environ: Mapping[str, str] | None = None,
) -> ConsolidationSchedule | None:
    """Attach the rotating sweep to one Celery app, or leave it untouched when unconfigured.

    Returning before registering anything is what keeps a deployment that has not opted in
    identical to the observation worker it already runs.
    """
    schedule = consolidation_schedule(os.environ if environ is None else environ)
    if schedule is None:
        return None
    # Read while the app is being built, not on the first tick, so a worker with the schedule
    # enabled and the model or storage contract missing refuses to boot rather than failing an
    # hour later inside a task whose message is then thrown away.
    settings = ConsolidationSettings.from_environment(environ)
    app.conf.task_routes = {
        **(app.conf.task_routes or {}),
        CONSOLIDATE_TENANT_TASK: {"queue": CONSOLIDATION_QUEUE},
    }

    @app.task(name=CONSOLIDATE_TENANT_TASK)  # type: ignore[untyped-decorator]
    def consolidate_tenant_task() -> str:
        return run_scheduled_sweep(settings, schedule, utc_now())

    app.conf.beat_schedule = {
        **(app.conf.beat_schedule or {}),
        "mindbridge-consolidation": {
            "task": CONSOLIDATE_TENANT_TASK,
            "schedule": schedule.interval_seconds,
            # A worker that was down while beat kept ticking would otherwise come back to a
            # queue holding every missed sweep and run them one after another, which is the
            # spike the interval exists to prevent. The next tick covers what expires here.
            "options": {"expires": schedule.interval_seconds},
        },
    }
    return schedule


def run_scheduled_sweep(
    settings: ConsolidationSettings,
    schedule: ConsolidationSchedule,
    now: datetime,
) -> str:
    """Sweep this tick's tenant and log the totals a CronJob would have printed on stdout."""
    tenant_id = schedule.due_tenant(now)
    summary = asyncio.run(run_sweep(settings, sweep_options(_scheduled_argv(tenant_id))))
    _LOGGER.info("consolidation sweep complete", extra=summary_dict(summary))
    return str(tenant_id)


def _scheduled_argv(tenant_id: TenantId) -> Sequence[str]:
    """Build the scheduled sweep from the documented flags, not a parallel set of defaults.

    Entity resolution is left out because it is the only sweep that opens media and spends a
    generator call per candidate pair, and the deployment guide already recommends running it
    on a rarer cadence. It stays a `mindbridge consolidate` invocation.
    """
    return ("--tenant-id", str(tenant_id), "--skip-entity-resolution")
