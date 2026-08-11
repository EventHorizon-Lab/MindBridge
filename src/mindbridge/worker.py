"""Celery worker composition for observation-derived memory."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NoReturn, Protocol

from billiard.exceptions import (  # type: ignore[import-untyped]  # Celery dependency lacks types.
    SoftTimeLimitExceeded,
)
from celery import Celery  # type: ignore[import-untyped]  # Upstream lacks PEP 561 metadata.

from mindbridge.application import ProcessObservation
from mindbridge.core import (
    JobId,
    JobState,
    ModelUnavailableError,
    ObjectStorageError,
    ObservationId,
    TenantId,
)
from mindbridge.infrastructure import (
    ObservationProcessingTaskMessage,
    PostgresMemoryStore,
    S3MediaAccess,
    create_task_queue,
)
from mindbridge.infrastructure.task_queue import PROCESS_OBSERVATION_TASK
from mindbridge.models import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
    DEFAULT_OMNI_MODEL_ID,
    JinaOmniEmbedder,
    OpenAIOmniEventPerceiver,
)

_MODEL_REQUEST_TIMEOUT_SECONDS = 780.0
_RUNNING_RETRY_SECONDS = 30
_RUNNING_MAX_RETRIES = 40


class _RetryingTask(Protocol):
    def retry(self, *, countdown: int, max_retries: int) -> NoReturn: ...


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Validated Worker configuration with credentials redacted from repr."""

    database_url: str = field(repr=False)
    object_storage_bucket: str
    task_broker_url: str = field(repr=False)
    vlm_api_key: str = field(repr=False)
    vlm_endpoint: str
    vlm_model_revision: str
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "us-east-1"
    vlm_model_id: str = DEFAULT_OMNI_MODEL_ID
    jina_model_id: str = DEFAULT_JINA_OMNI_MODEL_ID
    jina_model_revision: str = DEFAULT_JINA_OMNI_REVISION
    jina_device: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("object_storage_bucket", self.object_storage_bucket),
            ("object_storage_region", self.object_storage_region),
            ("task_broker_url", self.task_broker_url),
            ("vlm_api_key", self.vlm_api_key),
            ("vlm_endpoint", self.vlm_endpoint),
            ("vlm_model_id", self.vlm_model_id),
            ("vlm_model_revision", self.vlm_model_revision),
            ("jina_model_id", self.jina_model_id),
            ("jina_model_revision", self.jina_model_revision),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name, optional_value in (
            ("object_storage_endpoint_url", self.object_storage_endpoint_url),
            ("jina_device", self.jina_device),
        ):
            if optional_value is not None and not optional_value.strip():
                raise ValueError(f"{name} must not be empty when provided")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> WorkerSettings:
        """Read the explicit Worker contract and fail before consuming jobs."""
        source = os.environ if environ is None else environ
        return cls(
            database_url=_required(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage_bucket=_required(source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"),
            object_storage_endpoint_url=_optional(source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"),
            object_storage_region=source.get("MINDBRIDGE_OBJECT_STORAGE_REGION", "us-east-1"),
            task_broker_url=_required(source, "MINDBRIDGE_TASK_BROKER_URL"),
            vlm_api_key=_required(source, "MINDBRIDGE_VLM_API_KEY"),
            vlm_endpoint=_required(source, "MINDBRIDGE_VLM_ENDPOINT"),
            vlm_model_id=source.get("MINDBRIDGE_VLM_MODEL_ID", DEFAULT_OMNI_MODEL_ID),
            vlm_model_revision=_required(source, "MINDBRIDGE_VLM_MODEL_REVISION"),
            jina_model_id=source.get("MINDBRIDGE_JINA_MODEL_ID", DEFAULT_JINA_OMNI_MODEL_ID),
            jina_model_revision=source.get(
                "MINDBRIDGE_JINA_MODEL_REVISION", DEFAULT_JINA_OMNI_REVISION
            ),
            jina_device=_optional(source, "MINDBRIDGE_JINA_DEVICE"),
        )


def create_worker_app(settings: WorkerSettings) -> Celery:
    """Register the ID-only processing task on one GPU-safe Celery app."""
    task_queue = create_task_queue(settings.task_broker_url)
    task_queue.conf.update(worker_concurrency=1, worker_pool="prefork")

    @task_queue.task(  # type: ignore[untyped-decorator]
        name=PROCESS_OBSERVATION_TASK,
        bind=True,
        autoretry_for=(ModelUnavailableError, ObjectStorageError, SoftTimeLimitExceeded),
        max_retries=5,
        retry_backoff=True,
        retry_backoff_max=300,
        retry_jitter=True,
        pydantic=True,
        pydantic_strict=True,
    )
    def process_observation_task(
        task: _RetryingTask,
        message: ObservationProcessingTaskMessage,
    ) -> str:
        state = run_observation_processing(settings, message)
        if state is JobState.RUNNING:
            task.retry(
                countdown=_RUNNING_RETRY_SECONDS,
                max_retries=_RUNNING_MAX_RETRIES,
            )
        return str(state.value)

    return task_queue


def run_observation_processing(
    settings: WorkerSettings,
    message: ObservationProcessingTaskMessage,
) -> JobState:
    """Run one synchronous Celery delivery through the shared async use case."""
    embedder = _load_embedder(
        settings.jina_model_id,
        settings.jina_model_revision,
        settings.jina_device,
    )
    return asyncio.run(
        _process_observation_once(
            settings,
            embedder,
            TenantId(message.tenant_id),
            ObservationId(message.observation_id),
            JobId(message.job_id),
        )
    )


async def _process_observation_once(
    settings: WorkerSettings,
    embedder: JinaOmniEmbedder,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
) -> JobState:
    store = PostgresMemoryStore(settings.database_url)
    media_access = S3MediaAccess(
        settings.object_storage_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        region_name=settings.object_storage_region,
    )
    perceiver = OpenAIOmniEventPerceiver.connect(
        api_key=settings.vlm_api_key,
        endpoint=settings.vlm_endpoint,
        model_id=settings.vlm_model_id,
        model_revision=settings.vlm_model_revision,
        request_timeout_seconds=_MODEL_REQUEST_TIMEOUT_SECONDS,
    )
    async with AsyncExitStack() as resources:
        resources.push_async_callback(perceiver.close)
        await store.open()
        resources.push_async_callback(store.close)
        job = await ProcessObservation(
            store,
            perceiver,
            embedder,
            media_url_signer=media_access,
        ).run(tenant_id, observation_id, job_id)
    return job.state


@lru_cache(maxsize=1)
def _load_embedder(
    model_id: str,
    model_revision: str,
    device: str | None,
) -> JinaOmniEmbedder:
    # ponytail: one frozen model per worker process; split queues when multiple models are needed.
    return JinaOmniEmbedder.load(
        model_id=model_id,
        revision=model_revision,
        device=device,
    )


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value if value is not None and value.strip() else None
