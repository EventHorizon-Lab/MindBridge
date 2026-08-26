"""What the ledger reconciler reads from the broker, and what it publishes back."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from celery import Celery
from kombu.transport import memory  # type: ignore[import-untyped]

from mindbridge import jobs_cli
from mindbridge.core import JobId, ObservationId, TenantId
from mindbridge.infrastructure._postgres_jobs import ObservationJobAccounting
from mindbridge.infrastructure.task_queue import (
    FRESH_WORK_PRIORITY,
    RECOVERED_WORK_PRIORITY,
    CeleryObservationJobPublisher,
    create_task_queue,
    observation_queues,
    observation_shard,
)
from mindbridge.jobs_cli import _republish, queue_depth, reconcile


@pytest.fixture(autouse=True)
def _empty_broker() -> Iterator[None]:
    """Start every test on an empty broker.

    kombu's memory transport keeps its queues in one process-wide class attribute, and the queue
    names are now constants rather than something a test can vary, so messages one test published
    are messages the next one would count.
    """
    memory.Channel.queues.clear()
    yield
    memory.Channel.queues.clear()


async def test_queue_depth_sums_every_shard_the_worker_consumes() -> None:
    """A depth covering one shard would make `--republish` flood the queue with duplicates.

    `_withheld` subtracts this from the claimable rows, so it has to count the messages for this
    job type ledger-wide. Two tenants that land on different shards are the smallest case where
    reading one queue is wrong, and the assertion on the shards keeps it from silently becoming
    a one-queue test if the mapping changes.
    """
    task_queue = _memory_queue()
    publisher = CeleryObservationJobPublisher(task_queue)

    empty = queue_depth(task_queue)
    for index, tenant_id in enumerate(("tenant_01", "tenant_02")):
        await publisher.publish_observation_processing_job(
            TenantId(tenant_id), ObservationId(f"obs_{index}"), JobId(f"job_process_obs_{index}")
        )

    assert observation_shard("tenant_01") != observation_shard("tenant_02")
    # A virtual transport deletes an empty queue rather than keeping it, so a shard with no
    # messages and one that never existed are indistinguishable -- and both mean zero.
    assert (empty, queue_depth(task_queue)) == (0, 2)


async def test_republishing_sends_one_message_per_claimable_row() -> None:
    task_queue = _memory_queue()

    published = await _republish(task_queue, _claimable(3), already_queued=0)

    assert (published, queue_depth(task_queue)) == (3, 3)


async def test_republishing_leaves_the_backlog_the_queue_already_holds_alone() -> None:
    """Every claimable row was republished, so a healthy backlog was duplicated message for
    message -- and a duplicate is not a no-op: the losing delivery claims nothing and the task
    then re-queues itself every 30s up to 40 times."""
    task_queue = _memory_queue()
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


async def test_a_scoped_repair_does_not_withhold_against_another_tenants_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue is shared and `--tenant-id` is not, so subtracting the depth from one tenant's
    rows cancelled five stranded jobs against five messages for tenant_02 and published nothing.

    Both arms run here rather than against the bound alone, because the defect was the number
    handed to the repair: the ledger-wide call is the one that must still withhold.
    """
    task_queue = _memory_queue()
    publisher = CeleryObservationJobPublisher(task_queue)
    for index in range(5):
        await publisher.publish_observation_processing_job(
            TenantId("tenant_02"), ObservationId(f"obs_other_{index}"), JobId(f"job_other_{index}")
        )
    _stub_ledger(monkeypatch, task_queue, claimable=_claimable(3))

    scoped = await _reconcile(tenant_id=TenantId("tenant_01"))
    # The repair left eight messages on the queue, which is more than the ledger-wide scan finds
    # claimable -- so the bound there is the row count, not the depth.
    ledger_wide = await _reconcile(tenant_id=None)

    assert (scoped["queue_depth"], scoped["withheld"], scoped["republished"]) == (5, 0, 3)
    assert (ledger_wide["queue_depth"], ledger_wide["withheld"]) == (8, 3)
    assert (ledger_wide["claimable"], ledger_wide["republished"]) == (3, 0)


async def _reconcile(*, tenant_id: TenantId | None) -> dict[str, object]:
    """Run the reconciler with `--republish` against the stubbed ledger."""
    return await reconcile(
        "postgresql://ledger",
        "memory://",
        tenant_id=tenant_id,
        include_failed=False,
        republish=True,
    )


def _stub_ledger(
    monkeypatch: pytest.MonkeyPatch,
    task_queue: Celery,
    *,
    claimable: Sequence[tuple[TenantId, ObservationId, JobId]],
) -> None:
    """Give the reconciler the two ledger answers it reads, and the queue it reads them against.

    The broker is real -- what the repair publishes is counted off it -- and only PostgreSQL is
    replaced, because what is under test is the arithmetic between the two.
    """

    async def unreachable(
        connection: object,
        *,
        tenant_id: TenantId | None,
        include_failed: bool,
    ) -> tuple[tuple[TenantId, ObservationId, JobId], ...]:
        return tuple(claimable)

    async def accounting(
        connection: object, *, tenant_id: TenantId | None
    ) -> tuple[ObservationJobAccounting, ...]:
        return ()

    async def unconfined(connection: object) -> bool:
        return False

    monkeypatch.setattr(jobs_cli, "create_task_queue", lambda broker_url: task_queue)
    monkeypatch.setattr(jobs_cli, "psycopg", SimpleNamespace(AsyncConnection=_LedgerConnection))
    monkeypatch.setattr(jobs_cli, "unreachable_observation_jobs", unreachable)
    monkeypatch.setattr(jobs_cli, "observation_job_accounting", accounting)
    monkeypatch.setattr(jobs_cli, "tenant_scope_required", unconfined)


class _LedgerConnection:
    """As much of an async psycopg connection as the reconciler uses: open, scope, close."""

    @classmethod
    async def connect(cls, database_url: str) -> _LedgerConnection:
        return cls()

    async def execute(self, statement: str, parameters: tuple[str, ...]) -> None:
        return None

    async def __aenter__(self) -> _LedgerConnection:
        return self

    async def __aexit__(self, *exception: object) -> None:
        return None


async def test_republishing_publishes_nothing_when_the_queue_holds_every_row() -> None:
    task_queue = _memory_queue()

    assert await _republish(task_queue, _claimable(3), already_queued=5) == 0
    assert queue_depth(task_queue) == 0


def _claimable(count: int) -> tuple[tuple[TenantId, ObservationId, JobId], ...]:
    """Claimable rows in the order the ledger returns them, oldest first."""
    return tuple(
        (TenantId("tenant_01"), ObservationId(f"obs_{index}"), JobId(f"job_process_obs_{index}"))
        for index in range(count)
    )


def _drain(task_queue: Celery) -> list[str]:
    """Take every message off every shard and name the job each one carries.

    Per shard, so the order is the consumption order kombu would produce for one tenant: every
    row `_claimable` builds belongs to one tenant and so lands on one shard.
    """
    job_ids = []
    with task_queue.connection_for_read() as connection:
        channel = connection.default_channel
        for queue in observation_queues():
            while message := channel.basic_get(queue=queue):
                message.ack()
                job_ids.append(message.properties["correlation_id"])
    return job_ids


def _memory_queue() -> Celery:
    """An app on kombu's in-process broker, which `_empty_broker` resets between tests."""
    return create_task_queue("memory://")


async def test_republish_enters_the_queue_ahead_of_the_backlog_it_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired job is published at the priority the transport drains first.

    Six republishes of one job across 84 minutes once moved its attempt count not at all,
    because each landed behind the same backlog. Pinned on the publish call because that is
    the only place the choice is made, and a memory transport does not model priority steps.
    """
    task_queue = _memory_queue()
    send_task = Mock()
    monkeypatch.setattr(task_queue, "send_task", send_task)

    republished = await _republish(
        task_queue,
        [(TenantId("tenant_01"), ObservationId("obs_01"), JobId("job_process_obs_01"))],
        already_queued=0,
    )

    assert republished == 1, "nothing was published, so the priority assertion would be vacuous"
    assert send_task.call_args.kwargs["priority"] == RECOVERED_WORK_PRIORITY
    assert RECOVERED_WORK_PRIORITY < FRESH_WORK_PRIORITY
