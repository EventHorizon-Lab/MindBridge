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
    claimable = tuple(
        (TenantId("tenant_01"), ObservationId(f"obs_{index}"), JobId(f"job_process_obs_{index}"))
        for index in range(3)
    )

    published = await _republish(task_queue, claimable)

    assert (published, queue_depth(task_queue)) == (3, 3)


def _memory_queue(name: str) -> Celery:
    """One in-process broker per test: kombu's memory transport shares queues by name."""
    task_queue = create_task_queue("memory://")
    task_queue.conf.task_default_queue = name
    return task_queue
