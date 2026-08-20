"""Celery/Redis delivery for durable MindBridge processing jobs."""

from __future__ import annotations

import asyncio
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
            "visibility_timeout": int(hard_limit + _REDELIVERY_MARGIN_SECONDS)
        },
        enable_utc=True,
        task_acks_late=True,
        task_default_queue="mindbridge",
        task_ignore_result=True,
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
    ) -> None:
        """Publish without blocking the API event loop or leaking broker details."""
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
            )
        except OperationalError as error:
            raise TaskBrokerError("observation job delivery failed") from error
