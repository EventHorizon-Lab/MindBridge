"""Celery/Redis delivery for durable MindBridge processing jobs."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Annotated

from celery import Celery
from celery.exceptions import (
    OperationalError,
)
from pydantic import BaseModel, ConfigDict, StringConstraints

from mindbridge.core import JobId, ObservationId, TaskBrokerError, TenantId

PROCESS_OBSERVATION_TASK = "mindbridge.process_observation"
_Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]

DEFAULT_PROCESSING_BUDGET_SECONDS = 1_080.0
"""How long one observation may take before the Worker gives up on that attempt.

This is a backstop, not the real deadline: the model client's own `request_timeout_seconds`
fires first and says which call ran long. The two have to be set together, which is why the
Worker derives this from the timeout it gave its generator instead of leaving both fixed. A
budget below the model's own deadline silently overrides it, and because a soft-limit overrun
is retried, an observation that legitimately needs longer never finishes -- it repeats the same
model call until the retries run out, paying for each one.
"""

_HARD_LIMIT_MARGIN_SECONDS = 60.0
"""Grace between the soft signal a task can unwind from and the hard kill that follows."""

_REDELIVERY_MARGIN_SECONDS = 120.0
"""Kept above the hard limit so the broker never re-delivers a task that is still running."""


def observation_delivery_window_seconds(processing_budget_seconds: float) -> float:
    """The longest a single delivery of one observation may still be alive.

    The budget plus both margins: the soft limit is the budget, the hard kill follows one
    margin later, and the broker waits another before it hands the message to anybody else.
    Exposed because a second module has to compare against this -- the stale-claim window in
    `_postgres_jobs` has to sit above it, or a live attempt is declared abandoned while its
    worker is still paying for it. Derived here rather than restated there so the two cannot
    drift apart the way three independent constants already did once.
    """
    return processing_budget_seconds + _HARD_LIMIT_MARGIN_SECONDS + _REDELIVERY_MARGIN_SECONDS


FRESH_WORK_PRIORITY = 3
RECOVERED_WORK_PRIORITY = 0
"""Where a message enters the queue, so repair is not served behind the backlog it repairs.

kombu's Redis transport publishes with `LPUSH` and consumes with `RPOP`, walking
`priority_steps = [0, 3, 6, 9]` in order and draining each step before the next. Everything used
to publish at 0, so a republished job waited behind every message already queued -- while a
message a dead worker dropped is restored with `RPUSH` and served next. Recovery was the slowest
thing in the queue and fresh work the fastest, which is backwards: a row being republished is one
the ledger already owes and has already waited for.

Publishing fresh work at 3 and repair at 0 inverts that without any broker configuration or any
consumer change -- a worker polls every step of its queue whatever it was started with, so this
needs no `-Q` and no rolling-restart coordination. The gap between the two steps is unused on
purpose: 6 and 9 stay free for anything that must yield to both.
"""

OBSERVATION_QUEUE = "mindbridge"
"""The name observations were published to before the shards, and now their prefix."""

OBSERVATION_QUEUE_SHARDS = 8
"""How many queues observations spread over, so one tenant cannot hold the whole frontier.

With one queue, queue position is a global resource. During the 2026-08-24 evaluation a single
benchmark spent five hours at zero coverage behind 400 clips of another tenant's backlog on a
frontier advancing at tens of messages an hour; operators recovered it by hand, rebuilding the
queue as a per-benchmark round robin, and the first clip processed six minutes later. Across the
same run median queue wait went from 0.7 s with one producer to 12,018 s with nine.

Fairness needs nothing but more than one queue: kombu's Redis transport polls with
`queue_order_strategy = "round_robin"` and rotates the queue it just served to the end of the
cycle, so each shard's head is reached in turn no matter how deep its neighbours are.

Be clear about the bound this buys. With N shards and T tenants it is not per-tenant fairness --
two tenants that hash to one shard are still FIFO relative to each other. The true claim is that
one tenant can no longer starve every other tenant, only its shard-mates.

Eight because the count is not free: every BRPOP passes `len(queues) * len(priority_steps)` keys,
which is 36 here, and against the nine concurrent producers that caused the incident eight shards
already leave about one shard-mate per tenant. A count far above the plausible tenant count would
pay for empty keys on every poll and halve an already small collision rate.

Not an environment variable and not a flag on purpose. A publisher writing to shards its worker
does not consume stops ingest with nothing raised anywhere, so the two sides read one constant.
"""


def observation_queues() -> tuple[str, ...]:
    """Every queue an observation worker consumes: the pre-shard queue, then the shards.

    The unsuffixed name stays in the set even though nothing publishes there any more. A
    deployment upgrading into the shards has messages already sitting on it, and a worker that
    consumed only the shards would leave that backlog with no consumer at all.
    """
    return (
        OBSERVATION_QUEUE,
        *(f"{OBSERVATION_QUEUE}.{index}" for index in range(OBSERVATION_QUEUE_SHARDS)),
    )


def observation_shard(tenant_id: str) -> str:
    """Pick the shard one tenant's observations are published to, stably across processes.

    Stability is the whole point: `hash(str)` is salted per interpreter unless `PYTHONHASHSEED`
    is set, so the API and `mindbridge jobs` would place the same tenant differently and its work
    would scatter over every shard run to run -- which is the starvation this is meant to bound.
    A digest is the same number in every process.

    Drawn out of `observation_queues` rather than formatted again, so the queue published to is
    by construction one the worker consumes.
    """
    shards = observation_queues()[1:]
    digest = hashlib.blake2b(tenant_id.encode("utf-8"), digest_size=8).digest()
    return shards[int.from_bytes(digest, "big") % len(shards)]


class ObservationProcessingTaskMessage(BaseModel):
    """Strict ID-only schema accepted at the Celery trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: _Identifier
    observation_id: _Identifier
    job_id: _Identifier


def create_task_queue(
    broker_url: str,
    *,
    processing_budget_seconds: float = DEFAULT_PROCESSING_BUDGET_SECONDS,
) -> Celery:
    """Create the shared Celery app with bounded, JSON-only delivery.

    The three deadlines below are one number with two margins on purpose. They used to be three
    independent constants, and a deployment that raised its model timeout moved none of them: the
    soft limit cut every observation short, and the re-delivery window sat between the soft and
    hard limits, so a long task could also be handed to a second worker while the first still
    held it.
    """
    if not broker_url.strip():
        raise ValueError("broker_url must not be empty")
    if processing_budget_seconds <= 0:
        raise ValueError("processing_budget_seconds must be positive")
    hard_limit = processing_budget_seconds + _HARD_LIMIT_MARGIN_SECONDS
    task_queue = Celery("mindbridge", broker=broker_url)
    task_queue.conf.update(
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=5,
        broker_transport_options={
            "visibility_timeout": int(
                observation_delivery_window_seconds(processing_budget_seconds)
            )
        },
        enable_utc=True,
        task_acks_late=True,
        task_default_queue=OBSERVATION_QUEUE,
        task_ignore_result=True,
        # Declared as a set, not one name, so a worker started with no `-Q` consumes every shard:
        # with nothing selected, `app.amqp.queues.consume_from` is exactly this mapping. Empty
        # options mean what Celery would have built for `task_default_queue` on its own -- the
        # default exchange, routing key equal to the name. `mindbridge_consolidation` is
        # deliberately absent: `-Q` still reaches it through `task_create_missing_queues`, and an
        # observation worker must not pick up a sweep.
        task_queues={name: {} for name in observation_queues()},
        task_publish_retry=True,
        task_publish_retry_policy={
            "interval_start": 0,
            "interval_step": 0.5,
            "interval_max": 2,
            "max_retries": 3,
        },
        task_reject_on_worker_lost=True,
        task_serializer="json",
        task_soft_time_limit=processing_budget_seconds,
        task_time_limit=hard_limit,
        timezone="UTC",
        worker_prefetch_multiplier=1,
    )
    return task_queue


class CeleryObservationJobPublisher:
    """Publish IDs only; PostgreSQL remains the job state and payload authority."""

    def __init__(self, task_queue: Celery) -> None:
        self._task_queue = task_queue

    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        priority: int = FRESH_WORK_PRIORITY,
    ) -> None:
        """Publish without blocking the API event loop or leaking broker details.

        Onto the tenant's shard, so one tenant's backlog bounds its shard-mates rather than every
        other tenant. Priority still dominates the shard: kombu builds its BRPOP key list with
        priority as the outer loop and queue as the inner one, so repair on any shard is served
        before fresh work on every shard. A `task.retry()` re-publishes to the `routing_key` and
        priority it was delivered with, so a retried observation stays on its own shard too.
        """
        try:
            message = ObservationProcessingTaskMessage(
                tenant_id=tenant_id,
                observation_id=observation_id,
                job_id=job_id,
            )
            await asyncio.to_thread(
                self._task_queue.send_task,
                PROCESS_OBSERVATION_TASK,
                kwargs={"message": message.model_dump(mode="json")},
                task_id=job_id,
                priority=priority,
                queue=observation_shard(tenant_id),
            )
        except OperationalError as error:
            raise TaskBrokerError("observation job delivery failed") from error
