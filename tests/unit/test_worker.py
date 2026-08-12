"""Checks for the deployable observation Worker boundary."""

from collections.abc import Mapping
from typing import cast
from unittest.mock import Mock

import pytest
from celery import Task  # type: ignore[import-untyped]  # Upstream lacks PEP 561 metadata.
from celery.exceptions import Retry  # type: ignore[import-untyped]  # Upstream lacks types.
from pydantic import ValidationError

import mindbridge.worker as worker_module
from mindbridge.core import JobState
from mindbridge.infrastructure.task_queue import (
    PROCESS_OBSERVATION_TASK,
    ObservationProcessingTaskMessage,
)
from mindbridge.worker import WorkerSettings, create_worker_app


def test_worker_settings_pin_models_and_redact_credentials() -> None:
    """Worker startup fixes model identity without exposing injected secrets."""
    settings = WorkerSettings.from_environment(_environment())

    assert settings.vlm_model_id == "qwen3.8-max"
    assert settings.vlm_model_revision == "deployment-2026-08-11"
    assert settings.jina_model_revision == "12949877f0092093f366c6450340011320152a05"
    assert "database-secret" not in repr(settings)
    assert "broker-secret" not in repr(settings)
    assert "vlm-secret" not in repr(settings)
    assert "text-embedding-secret" not in repr(settings)


def test_worker_settings_require_explicit_vlm_revision() -> None:
    """Fallback provenance cannot silently use a mutable deployment alias."""
    environment = dict(_environment())
    del environment["MINDBRIDGE_VLM_MODEL_REVISION"]

    with pytest.raises(ValueError, match="MINDBRIDGE_VLM_MODEL_REVISION"):
        WorkerSettings.from_environment(environment)


def test_worker_task_calls_shared_use_case_with_ids_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Celery boundary forwards identities and keeps GPU concurrency bounded."""
    calls: list[tuple[str, str, str]] = []

    def run(
        settings: WorkerSettings,
        message: ObservationProcessingTaskMessage,
    ) -> JobState:
        calls.append((message.tenant_id, message.observation_id, message.job_id))
        return JobState.SUCCEEDED

    monkeypatch.setattr(worker_module, "run_observation_processing", run)
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])

    result = task.run(_task_message())

    assert result == "succeeded"
    assert calls == [("tenant_01", "observation_01", "job_process_observation_01")]
    assert app.conf.worker_concurrency == 1
    assert app.conf.worker_pool == "prefork"


def test_worker_retries_an_observation_owned_by_another_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redelivery waits long enough for a stale durable claim to become reclaimable."""
    monkeypatch.setattr(
        worker_module,
        "run_observation_processing",
        lambda *_arguments: JobState.RUNNING,
    )
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    retry = Mock(side_effect=Retry("still running"))
    monkeypatch.setattr(task, "retry", retry)

    with pytest.raises(Retry, match="still running"):
        task.run(_task_message())

    retry.assert_called_once_with(countdown=30, max_retries=40)


def test_worker_rejects_invalid_task_identity() -> None:
    """Malformed or expanded broker payloads fail at the Worker trust boundary."""
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])

    with pytest.raises(ValidationError):
        task.run({**_task_message(), "tenant_id": "", "media_url": "https://untrusted"})


def _environment() -> Mapping[str, str]:
    return {
        "MINDBRIDGE_DATABASE_URL": ("postgresql://mindbridge:database-secret@postgres/mindbridge"),
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://:broker-secret@redis:6379/0",
        "MINDBRIDGE_VLM_API_KEY": "vlm-secret",
        "MINDBRIDGE_VLM_ENDPOINT": "https://vlm.example.test/api/v1/chat/completions",
        "MINDBRIDGE_VLM_MODEL_REVISION": "deployment-2026-08-11",
        "MINDBRIDGE_TEXT_EMBEDDING_API_KEY": "text-embedding-secret",
        "MINDBRIDGE_TEXT_EMBEDDING_ENDPOINT": "https://text.example.test/api/v1/embeddings",
    }


def _task_message() -> dict[str, str]:
    return {
        "tenant_id": "tenant_01",
        "observation_id": "observation_01",
        "job_id": "job_process_observation_01",
    }
