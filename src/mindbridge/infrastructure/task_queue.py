"""Celery/Redis delivery for durable MindBridge processing jobs."""

from __future__ import annotations

import asyncio
from typing import Annotated

from celery import Celery
from kombu.exceptions import (
    OperationalError,
)
from pydantic import BaseModel, ConfigDict, StringConstraints

from mindbridge.core import JobId, ObservationId, TaskBrokerError, TenantId

PROCESS_OBSERVATION_TASK = "mindbridge.process_observation"
_Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class ObservationProcessingTaskMessage(BaseModel):
    """Strict ID-only schema accepted at the Celery trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: _Identifier
    observation_id: _Identifier
    job_id: _Identifier


def create_task_queue(broker_url: str) -> Celery:
    """Create the shared Celery app with bounded, JSON-only delivery."""
    if not broker_url.strip():
        raise ValueError("broker_url must not be empty")
    task_queue = Celery("mindbridge", broker=broker_url)
    task_queue.conf.update(
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=5,
        broker_transport_options={"visibility_timeout": 1_200},
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
        task_soft_time_limit=840,
        task_time_limit=900,
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
