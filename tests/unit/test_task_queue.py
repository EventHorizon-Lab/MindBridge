"""Tests for Celery delivery without a live Redis broker."""

import os
import subprocess
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from celery.exceptions import (
    OperationalError,
)
from kombu.transport.redis import Channel  # type: ignore[import-untyped]
from kombu.utils.scheduling import round_robin_cycle  # type: ignore[import-untyped]

from mindbridge.consolidation_worker import CONSOLIDATION_QUEUE
from mindbridge.core import JobId, ObservationId, TaskBrokerError, TenantId
from mindbridge.infrastructure.task_queue import (
    FRESH_WORK_PRIORITY,
    OBSERVATION_QUEUE,
    OBSERVATION_QUEUE_SHARDS,
    PROCESS_OBSERVATION_TASK,
    RECOVERED_WORK_PRIORITY,
    CeleryObservationJobPublisher,
    create_task_queue,
    observation_queues,
    observation_shard,
)


def test_task_queue_is_json_only_and_retry_safe() -> None:
    """Celery is configured for bounded at-least-once job delivery."""
    task_queue = create_task_queue("memory://")

    assert task_queue.conf.accept_content == ["json"]
    assert task_queue.conf.task_acks_late is True
    assert task_queue.conf.task_reject_on_worker_lost is True
    assert task_queue.conf.worker_prefetch_multiplier == 1
    assert (
        task_queue.conf.task_time_limit
        < task_queue.conf.broker_transport_options["visibility_timeout"]
    )


async def test_publisher_sends_only_stable_tenant_and_job_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media or memory content never crosses Redis in a task message."""
    task_queue = create_task_queue("memory://")
    send_task = Mock()
    monkeypatch.setattr(task_queue, "send_task", send_task)

    await CeleryObservationJobPublisher(task_queue).publish_observation_processing_job(
        TenantId("tenant_01"),
        ObservationId("observation_01"),
        JobId("job_process_observation_01"),
    )

    send_task.assert_called_once_with(
        PROCESS_OBSERVATION_TASK,
        kwargs={
            "message": {
                "tenant_id": "tenant_01",
                "observation_id": "observation_01",
                "job_id": "job_process_observation_01",
            }
        },
        task_id="job_process_observation_01",
        priority=FRESH_WORK_PRIORITY,
        queue=observation_shard("tenant_01"),
    )


async def test_publisher_sanitizes_broker_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis connection details cannot escape the infrastructure boundary."""
    task_queue = create_task_queue("memory://")
    monkeypatch.setattr(
        task_queue,
        "send_task",
        Mock(side_effect=OperationalError("redis://secret@broker")),
    )

    with pytest.raises(TaskBrokerError, match="delivery failed"):
        await CeleryObservationJobPublisher(task_queue).publish_observation_processing_job(
            TenantId("tenant_01"),
            ObservationId("observation_01"),
            JobId("job_process_observation_01"),
        )


def test_task_queue_deadlines_follow_the_processing_budget() -> None:
    """One budget moves all three deadlines, so a slow generator cannot be cut off by them.

    Fixed constants here silently overrode the model's own timeout: the soft limit killed every
    observation mid-call, the overrun was retried as if it were transient, and the same call was
    paid for again. The ordering matters as much as the values -- re-delivery has to sit beyond
    the hard kill, or a task still running is handed to a second worker.
    """
    task_queue = create_task_queue("memory://", processing_budget_seconds=2_100.0)

    soft = task_queue.conf.task_soft_time_limit
    hard = task_queue.conf.task_time_limit
    redelivery = task_queue.conf.broker_transport_options["visibility_timeout"]

    assert soft == 2_100.0
    assert soft < hard < redelivery


def test_task_queue_rejects_a_budget_that_cannot_hold_any_work() -> None:
    """A non-positive budget would kill every task instantly instead of bounding it."""
    with pytest.raises(ValueError, match="processing_budget_seconds must be positive"):
        create_task_queue("memory://", processing_budget_seconds=0)


def test_repair_outranks_fresh_work_on_a_step_the_transport_actually_walks() -> None:
    """Repair must be drained first, and only on a step kombu's Redis transport visits.

    Asserted as an ordering rather than as two literals: the whole defect was that recovered
    work was served last, so a change that swapped the two constants would restore it while
    leaving any equality check green. The step membership matters because kombu rounds a
    priority to the nearest configured step, so an unlisted number silently becomes another
    step's queue and the split quietly stops existing.
    """
    kombu_redis_priority_steps = (0, 3, 6, 9)

    assert RECOVERED_WORK_PRIORITY < FRESH_WORK_PRIORITY
    assert RECOVERED_WORK_PRIORITY in kombu_redis_priority_steps
    assert FRESH_WORK_PRIORITY in kombu_redis_priority_steps


def test_a_worker_started_with_no_queue_flag_consumes_every_shard_publishing_reaches() -> None:
    """The publish targets and the consume set are one list, so they cannot disagree.

    A publisher writing to a shard the worker does not consume stops ingest with nothing raised
    anywhere, which is why the shard count is a constant rather than a variable and why this is
    asserted against `consume_from` -- the mapping a worker started with no `-Q` reads -- rather
    than against the constant a second time.
    """
    task_queue = create_task_queue("memory://")
    tenants = ("tenant_01", "tenant_02", "tenant_03", "egolife", "m3-web", "locomo")

    reached = {observation_shard(tenant) for tenant in tenants}
    consumed = set(task_queue.amqp.queues.consume_from)

    # Without a spread, every other assertion here would hold for a function returning one name.
    assert len(reached) > 1, "the sampled tenants all landed on one shard"
    assert reached <= consumed
    assert set(observation_queues()) == consumed
    # The pre-shard name: an upgrade finds messages on it, and nothing else would drain them.
    assert OBSERVATION_QUEUE in consumed
    assert len(consumed) == OBSERVATION_QUEUE_SHARDS + 1


def test_the_shard_set_leaves_the_consolidation_sweep_to_its_own_worker() -> None:
    """Declaring `task_queues` must not hand an observation worker a sweep.

    A sweep is minutes of generator calls against a queue running at a prefetch of one, so it is
    routed to its own queue and reached with `-Q`. Read before anything routes, because
    `task_create_missing_queues` adds a routed queue to the publishing process's own set.
    """
    task_queue = create_task_queue("memory://")

    assert CONSOLIDATION_QUEUE not in set(task_queue.amqp.queues.consume_from)


def test_the_shard_a_tenant_lands_on_is_the_same_in_every_process() -> None:
    """`hash(str)` is salted per interpreter, so it would scatter one tenant over every shard.

    Run in subprocesses because that is the only place the property is visible: the API, the
    worker and `mindbridge jobs` are separate interpreters, and with an unset `PYTHONHASHSEED`
    each would place the same tenant differently -- turning the bound this buys back into the
    starvation it replaces.
    """
    tenants = ("tenant_01", "tenant_02", "tenant_03", "egolife", "m3-web", "locomo")
    here = [observation_shard(tenant) for tenant in tenants]

    assert _shards_from_a_fresh_interpreter(tenants, hash_seed="0") == here
    assert _shards_from_a_fresh_interpreter(tenants, hash_seed="1") == here


def test_repair_on_any_shard_is_polled_before_fresh_work_on_every_shard() -> None:
    """Sharding must not cost the priority split, and the key order is what decides it.

    kombu builds one BRPOP over `priority_steps` as the outer loop and queues as the inner one,
    so every priority-0 key -- repair, on any shard -- is passed before any priority-3 key, which
    is fresh work on every shard. Built by kombu's own `_brpop_start` rather than restated here,
    so a transport that reordered the loops would fail this instead of passing it.
    """
    keys = _brpop_keys(observation_queues())

    repair = [
        keys.index(_priority_key(queue, RECOVERED_WORK_PRIORITY)) for queue in observation_queues()
    ]
    fresh = [
        keys.index(_priority_key(queue, FRESH_WORK_PRIORITY)) for queue in observation_queues()
    ]

    assert len(fresh) == OBSERVATION_QUEUE_SHARDS + 1, "no fresh-work key was found to compare"
    assert max(repair) < min(fresh)


def _shards_from_a_fresh_interpreter(tenants: tuple[str, ...], *, hash_seed: str) -> list[str]:
    """Map the tenants in a subprocess whose string hashing is salted differently."""
    probe = (
        "import sys;"
        "from mindbridge.infrastructure.task_queue import observation_shard;"
        "print('\\n'.join(observation_shard(tenant) for tenant in sys.argv[1:]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, *tenants],
        capture_output=True,
        check=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    return completed.stdout.split()


def _brpop_keys(queues: tuple[str, ...]) -> list[str]:
    """The keys kombu's Redis transport polls for these queues, in the order it passes them."""
    send_command = Mock()
    channel = SimpleNamespace(
        brpop_timeout=1,
        active_queues=queues,
        client=SimpleNamespace(connection=SimpleNamespace(send_command=send_command)),
        global_keyprefix=None,
        priority_steps=Channel.priority_steps,
        sep=Channel.sep,
        _queue_cycle=round_robin_cycle(list(queues)),
    )
    for name in ("_brpop_start", "_q_for_pri", "priority"):
        setattr(channel, name, MethodType(getattr(Channel, name), channel))

    channel._brpop_start()

    command, *keys = send_command.call_args.args
    assert command == "BRPOP"
    return [key for key in keys if isinstance(key, str)]


def _priority_key(queue: str, priority: int) -> str:
    """The Redis key one priority step of one queue lives in, named by kombu itself."""
    channel = SimpleNamespace(priority_steps=Channel.priority_steps, sep=Channel.sep)
    for name in ("_q_for_pri", "priority"):
        setattr(channel, name, MethodType(getattr(Channel, name), channel))
    return str(channel._q_for_pri(queue, priority))
