"""Checks for the beat schedule that finally runs the consolidation sweeps."""

from argparse import Namespace
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from celery import Celery
from celery.app.task import Task

from mindbridge.application.consolidation_sweep import ConsolidationSweepSummary, SweepSummary
from mindbridge.consolidation_cli import ConsolidationSettings
from mindbridge.consolidation_worker import (
    CONSOLIDATE_TENANT_TASK,
    CONSOLIDATION_QUEUE,
    DEFAULT_CONSOLIDATION_INTERVAL_SECONDS,
    ConsolidationSchedule,
    consolidation_schedule,
    register_consolidation_schedule,
)
from mindbridge.core import TenantId
from mindbridge.infrastructure.task_queue import create_task_queue

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def _environment(**overrides: str) -> Mapping[str, str]:
    return {
        "MINDBRIDGE_DATABASE_URL": "postgresql://mindbridge:database-secret@postgres/mindbridge",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://:broker-secret@redis:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
        "MINDBRIDGE_EMBEDDER_API_KEY": "text-embedding-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://text.example.test/v1",
        **overrides,
    }


def _app() -> Celery:
    return create_task_queue("redis://:broker-secret@redis:6379/0")


def test_an_unconfigured_deployment_gets_no_schedule_and_no_task() -> None:
    """Opting out has to leave the observation worker exactly as it was."""
    app = _app()

    assert register_consolidation_schedule(app, _environment()) is None
    assert not app.conf.beat_schedule
    assert CONSOLIDATE_TENANT_TASK not in app.tasks


def test_a_configured_schedule_ticks_at_the_interval_and_expires_missed_ticks() -> None:
    app = _app()

    schedule = register_consolidation_schedule(
        app,
        _environment(
            MINDBRIDGE_CONSOLIDATION_TENANT_IDS="tenant_01,tenant_02",
            MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS="900",
        ),
    )

    assert schedule == ConsolidationSchedule(
        tenant_ids=(TenantId("tenant_01"), TenantId("tenant_02")),
        interval_seconds=900.0,
    )
    entry = app.conf.beat_schedule["mindbridge-consolidation"]
    assert entry["task"] == CONSOLIDATE_TENANT_TASK
    assert entry["schedule"] == 900.0
    # Without this a worker that was down for a day comes back to a day of queued sweeps and
    # runs every one of them, which is the load spike the interval is there to prevent.
    assert entry["options"] == {"expires": 900.0}


def test_the_sweep_is_routed_off_the_queue_the_observation_worker_consumes() -> None:
    """A sweep is minutes of generator calls; sharing the queue would stall ingest for it."""
    app = _app()

    register_consolidation_schedule(
        app, _environment(MINDBRIDGE_CONSOLIDATION_TENANT_IDS="tenant_01")
    )

    # Asserted before anything routes: `task_create_missing_queues` adds a routed queue to the
    # app's queue set the first time a task is routed, so reading this after a route call would
    # measure the test rather than what a worker started with no -Q consumes.
    assert list(app.amqp.queues.consume_from) == [app.conf.task_default_queue]
    assert app.conf.task_default_queue != CONSOLIDATION_QUEUE
    routed = app.amqp.router.route({}, CONSOLIDATE_TENANT_TASK)
    assert routed["queue"].name == CONSOLIDATION_QUEUE


def test_the_interval_defaults_to_the_frequent_end_of_the_documented_cadence() -> None:
    schedule = consolidation_schedule(_environment(MINDBRIDGE_CONSOLIDATION_TENANT_IDS="tenant_01"))

    assert schedule is not None
    assert schedule.interval_seconds == DEFAULT_CONSOLIDATION_INTERVAL_SECONDS


def test_one_tick_sweeps_one_tenant_so_the_interval_bounds_the_whole_fleet() -> None:
    """Three tenants must not mean three concurrent sweeps against one saturated endpoint."""
    schedule = ConsolidationSchedule(
        tenant_ids=(TenantId("tenant_01"), TenantId("tenant_02"), TenantId("tenant_03")),
        interval_seconds=900.0,
    )
    ticks = tuple(schedule.due_tenant(NOW + timedelta(seconds=900 * index)) for index in range(7))

    assert len(set(ticks[:3])) == 3
    # The rotation repeats, so every tenant is reached once per rotation and never twice in one.
    assert ticks[3:6] == ticks[:3]
    assert ticks[6] == ticks[3]


@pytest.mark.parametrize(
    "environ",
    [
        {"MINDBRIDGE_CONSOLIDATION_TENANT_IDS": "tenant_01", "interval": "0"},
        {"MINDBRIDGE_CONSOLIDATION_TENANT_IDS": "tenant_01", "interval": "-60"},
        {"MINDBRIDGE_CONSOLIDATION_TENANT_IDS": "tenant_01", "interval": "hourly"},
        {"MINDBRIDGE_CONSOLIDATION_TENANT_IDS": ",,", "interval": "900"},
        {"MINDBRIDGE_CONSOLIDATION_TENANT_IDS": "tenant_01,tenant_01", "interval": "900"},
    ],
)
def test_an_unusable_schedule_is_refused_rather_than_silently_ignored(
    environ: Mapping[str, str],
) -> None:
    """A schedule read wrong is a sweep that never runs again -- the state being repaired."""
    with pytest.raises(ValueError):
        consolidation_schedule(
            _environment(
                MINDBRIDGE_CONSOLIDATION_TENANT_IDS=environ["MINDBRIDGE_CONSOLIDATION_TENANT_IDS"],
                MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS=environ["interval"],
            )
        )


def test_the_tick_sweeps_the_due_tenant_with_the_documented_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled run reaches the sweep through the CLI parser, so it cannot drift from it."""
    seen: list[Namespace] = []

    async def run_sweep(
        _settings: ConsolidationSettings,
        options: Namespace,
    ) -> ConsolidationSweepSummary:
        seen.append(options)
        return _summary()

    monkeypatch.setattr("mindbridge.consolidation_worker.run_sweep", run_sweep)
    monkeypatch.setattr("mindbridge.consolidation_worker.utc_now", lambda: NOW)
    app = _app()
    register_consolidation_schedule(
        app,
        _environment(
            MINDBRIDGE_CONSOLIDATION_TENANT_IDS="tenant_01,tenant_02",
            MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS="900",
        ),
    )
    task = cast(Task, app.tasks[CONSOLIDATE_TENANT_TASK])

    swept = task.run()

    assert len(seen) == 1
    options = seen[0]
    assert options.tenant_id == swept
    assert swept in {"tenant_01", "tenant_02"}
    # The only sweep that opens media and spends a generator call per candidate pair stays a
    # deliberate `mindbridge consolidate` invocation rather than something an hourly tick pays
    # for. Everything else keeps the values the CLI documents.
    assert options.skip_entity_resolution is True
    assert (options.page_size, options.summary_page_size) == (16, 16)
    assert options.minimum_similarity == 0.7


def _summary() -> ConsolidationSweepSummary:
    empty = SweepSummary(
        tenant_id=TenantId("tenant_01"),
        evaluated_at=NOW,
        page_count=1,
        scanned_count=0,
        candidate_count=0,
        counts={},
    )
    return ConsolidationSweepSummary(episodes=empty, claims=empty, summaries=empty, entities=None)
