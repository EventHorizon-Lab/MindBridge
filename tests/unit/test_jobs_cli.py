"""What the ledger reconciler reads from the broker, and what it publishes back."""

from __future__ import annotations

from celery import Celery

from mindbridge.core import JobId, ObservationId, TenantId
from mindbridge.infrastructure.task_queue import (
    CeleryObservationJobPublisher,
    create_task_queue,
)
from mindbridge.jobs_cli import _republish, queue_depth


async def test_queue_depth_reads_the_queue_the_worker_consumes() -> None:
    """Asking about `celery` instead of `task_default_queue` read as a destroyed queue."""
    task_queue = _memory_queue("depth-test")
    publisher = CeleryObservationJobPublisher(task_queue)

    empty = queue_depth(task_queue)
    await publisher.publish_observation_processing_job(
        TenantId("tenant_01"), ObservationId("obs_01"), JobId("job_process_obs_01")
    )
    await publisher.publish_observation_processing_job(
        TenantId("tenant_01"), ObservationId("obs_02"), JobId("job_process_obs_02")
    )
    queued = queue_depth(task_queue)
    misnamed = _memory_queue("depth-test-under-another-name")

    # A virtual transport deletes an empty queue rather than keeping it, so a name with no
    # messages and a name that never existed are indistinguishable -- and both mean zero.
    assert (empty, queued, queue_depth(misnamed)) == (0, 2, 0)


async def test_republishing_sends_one_message_per_claimable_row() -> None:
    task_queue = _memory_queue("republish-test")

    published = await _republish(task_queue, _claimable(3), already_queued=0)

    assert (published, queue_depth(task_queue)) == (3, 3)


async def test_republishing_leaves_the_backlog_the_queue_already_holds_alone() -> None:
    """Every claimable row was republished, so a healthy backlog was duplicated message for
    message -- and a duplicate is not a no-op: the losing delivery claims nothing and the task
    then re-queues itself every 30s up to 40 times."""
    task_queue = _memory_queue("republish-bound-test")
    publisher = CeleryObservationJobPublisher(task_queue)
    for _, observation_id, job_id in _claimable(3)[1:]:
        await publisher.publish_observation_processing_job(
            TenantId("tenant_01"), observation_id, job_id
        )

    published = await _republish(task_queue, _claimable(3), already_queued=2)

    assert (published, queue_depth(task_queue)) == (1, 3)
    # And it is the oldest row that was published: kombu consumes with RPOP, so the messages
    # still queued are the newest ones and the row with none is at the front of the list.
    assert _drain(task_queue)[-1] == "job_process_obs_0"


async def test_republishing_publishes_nothing_when_the_queue_holds_every_row() -> None:
    task_queue = _memory_queue("republish-full-test")

    assert await _republish(task_queue, _claimable(3), already_queued=5) == 0
    assert queue_depth(task_queue) == 0


def _claimable(count: int) -> tuple[tuple[TenantId, ObservationId, JobId], ...]:
    """Claimable rows in the order the ledger returns them, oldest first."""
    return tuple(
        (TenantId("tenant_01"), ObservationId(f"obs_{index}"), JobId(f"job_process_obs_{index}"))
        for index in range(count)
    )


def _drain(task_queue: Celery) -> list[str]:
    """Take every message off the queue and name the job each one carries."""
    job_ids = []
    with task_queue.connection_for_read() as connection:
        channel = connection.default_channel
        while message := channel.basic_get(queue=str(task_queue.conf.task_default_queue)):
            message.ack()
            job_ids.append(message.properties["correlation_id"])
    return job_ids


def _memory_queue(name: str) -> Celery:
    """One in-process broker per test: kombu's memory transport shares queues by name."""
    task_queue = create_task_queue("memory://")
    task_queue.conf.task_default_queue = name
    return task_queue
