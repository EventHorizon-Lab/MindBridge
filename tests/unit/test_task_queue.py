"""Tests for Celery delivery without a live Redis broker."""

from unittest.mock import Mock

import pytest
from kombu.exceptions import (  # type: ignore[import-untyped]  # Upstream lacks PEP 561 metadata.
    OperationalError,
)

from mindbridge.core import JobId, ObservationId, TaskBrokerError, TenantId
from mindbridge.infrastructure.task_queue import (
    PROCESS_OBSERVATION_TASK,
    CeleryObservationJobPublisher,
    create_task_queue,
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
