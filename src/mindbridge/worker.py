"""Celery worker composition for observation-derived memory."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, NoReturn, Protocol

from billiard.exceptions import (
    SoftTimeLimitExceeded,
)
from celery import Celery
from celery.signals import (
    worker_process_init,
    worker_process_shutdown,
)
from pydantic import Field, StrictBool

from mindbridge.application.capabilities import Embedder
from mindbridge.application.evidence_clips import ClipSampling
from mindbridge.application.pipelines import PerceptionPipeline
from mindbridge.application.process_observation import ProcessObservation
from mindbridge.configuration import (
    PluginConfigModel,
    PluginInteger,
    PluginNumber,
    copy_plugin_configuration,
    optional_environment_value,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
)
from mindbridge.core import (
    DatabaseUnavailableError,
    JobId,
    JobState,
    ModelUnavailableError,
    ObjectStorageError,
    ObservationId,
    TenantId,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import (
    ObjectStorageEnvironment,
    S3MediaAccess,
    object_storage_from_environment,
)
from mindbridge.infrastructure.task_queue import (
    PROCESS_OBSERVATION_TASK,
    ObservationProcessingTaskMessage,
    create_task_queue,
)
from mindbridge.media.clipping import (
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_VIDEO_FRAMES_PER_SECOND,
    DEFAULT_VIDEO_MAX_PIXELS,
)
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    embedding_dimension_from_environment,
    jina_media_embedder_config,
    openai_embedder_config,
    openai_generator_config,
)
from mindbridge.models.plugins import close_model, load_embedder, load_generator
from mindbridge.telemetry import configure_telemetry

_MODEL_REQUEST_TIMEOUT_SECONDS = 780.0
# The generator fetches these signed URLs itself, so they have to outlive the call that hands
# them over. Object storage signs for 300s by default, which expired mid-request and came back
# as a permanent "could not download multimodal content" 400 rather than a retryable fetch.
_MEDIA_URL_LIFETIME_SECONDS = int(_MODEL_REQUEST_TIMEOUT_SECONDS) + 300
_RUNNING_RETRY_SECONDS = 30
_RUNNING_MAX_RETRIES = 40
_TRANSIENT_MAX_RETRIES = 5
_RUNNING_RETRIES_HEADER = "mindbridge_running_retries"


@worker_process_init.connect(weak=False)  # type: ignore[untyped-decorator]
def _configure_worker_telemetry(**_kwargs: object) -> None:
    """Initialize exporters after Celery forks its process-safe worker child."""
    _dispose_worker_runtime()
    configure_telemetry("mindbridge-worker")


class _TaskRequest(Protocol):
    retries: int
    headers: Mapping[str, object] | None


class _RetryingTask(Protocol):
    request: _TaskRequest
    override_max_retries: int

    def retry(
        self,
        *,
        countdown: int,
        max_retries: int,
        headers: Mapping[str, object],
    ) -> NoReturn: ...


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Validated Worker configuration with credentials redacted from repr."""

    database_url: str = field(repr=False)
    object_storage: ObjectStorageEnvironment
    task_broker_url: str = field(repr=False)
    generator_config: Mapping[str, object] = field(repr=False)
    media_embedder_config: Mapping[str, object] = field(repr=False)
    text_embedder_config: Mapping[str, object] = field(repr=False)
    generator_plugin: str = "openai"
    media_embedder_plugin: str = "jina"
    text_embedder_plugin: str = "openai"
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    clip_sampling: ClipSampling = field(default_factory=ClipSampling)

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("task_broker_url", self.task_broker_url),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name, value in (
            ("generator_plugin", self.generator_plugin),
            ("media_embedder_plugin", self.media_embedder_plugin),
            ("text_embedder_plugin", self.text_embedder_plugin),
        ):
            validate_plugin_name(value, name)
        for name, config in (
            ("generator_config", self.generator_config),
            ("media_embedder_config", self.media_embedder_config),
            ("text_embedder_config", self.text_embedder_config),
        ):
            object.__setattr__(self, name, copy_plugin_configuration(config, name))

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> WorkerSettings:
        """Read the explicit Worker contract and fail before consuming jobs."""
        source = os.environ if environ is None else environ
        generator_plugin = source.get("MINDBRIDGE_GENERATOR_PLUGIN", "openai")
        media_plugin = source.get("MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN", "jina")
        # The Worker's text encoder must land in the same space the API queries, so it reads the
        # deployment-wide MINDBRIDGE_EMBEDDER_* contract rather than a second set of names.
        text_plugin = source.get("MINDBRIDGE_EMBEDDER_PLUGIN", "openai")
        return cls(
            database_url=require_environment_value(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage=object_storage_from_environment(source),
            task_broker_url=require_environment_value(source, "MINDBRIDGE_TASK_BROKER_URL"),
            generator_plugin=generator_plugin,
            generator_config=plugin_configuration(
                source,
                "MINDBRIDGE_GENERATOR_CONFIG_JSON",
                (
                    lambda: openai_generator_config(
                        source,
                        request_timeout_seconds=_MODEL_REQUEST_TIMEOUT_SECONDS,
                    )
                )
                if generator_plugin == "openai"
                else None,
            ),
            media_embedder_plugin=media_plugin,
            media_embedder_config=plugin_configuration(
                source,
                "MINDBRIDGE_MEDIA_EMBEDDER_CONFIG_JSON",
                (lambda: jina_media_embedder_config(source)) if media_plugin == "jina" else None,
            ),
            text_embedder_plugin=text_plugin,
            text_embedder_config=plugin_configuration(
                source,
                "MINDBRIDGE_EMBEDDER_CONFIG_JSON",
                (
                    lambda: openai_embedder_config(
                        source,
                        request_timeout_seconds=_MODEL_REQUEST_TIMEOUT_SECONDS,
                    )
                )
                if text_plugin == "openai"
                else None,
            ),
            embedding_dimension=embedding_dimension_from_environment(source),
            clip_sampling=_clip_sampling_from_environment(source),
        )


class _MediaSamplingConfig(PluginConfigModel):
    """Strict schema for the optional media sampling object.

    It reads value types as well as key names, which a hand-written key check does not: a
    quoted number or a `"false"` string for the off-switch would otherwise be accepted and
    silently mean something else.
    """

    frames_per_second: Annotated[PluginNumber, Field(gt=0)] = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: Annotated[PluginInteger, Field(gt=0)] = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: Annotated[PluginInteger, Field(gt=0)] = DEFAULT_IMAGE_MAX_PIXELS
    generation_proxy: StrictBool = True


def _clip_sampling_from_environment(source: Mapping[str, str]) -> ClipSampling:
    """Read the optional media sampling knob that sets the whole write cost of video.

    Frame rate multiplies every downstream cost in the write path: one clip cut, one encoder
    call, and one stored object per sampled window. A deployment ingesting continuous video
    has to be able to choose it, so it travels as one optional JSON object rather than four
    more fallback variables.
    """
    if optional_environment_value(source, "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON") is None:
        return ClipSampling()
    config = _MediaSamplingConfig.model_validate(
        plugin_configuration(source, "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON")
    )
    return ClipSampling(**config.model_dump())


@dataclass(slots=True)
class _WorkerRuntime:
    """One event loop and media model owned by a prefork Worker child."""

    loop: asyncio.AbstractEventLoop
    media_embedder: Embedder

    def run(self, settings: WorkerSettings, message: ObservationProcessingTaskMessage) -> JobState:
        task = self.loop.create_task(
            _process_observation_once(
                settings,
                self.media_embedder,
                TenantId(message.tenant_id),
                ObservationId(message.observation_id),
                JobId(message.job_id),
            )
        )
        try:
            return self.loop.run_until_complete(task)
        except BaseException:
            if not task.done():
                task.cancel()
                self.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            raise

    def close(self) -> None:
        try:
            self.loop.run_until_complete(close_model(self.media_embedder))
        finally:
            self.loop.close()


_worker_runtime: _WorkerRuntime | None = None
_worker_runtime_key: tuple[str, str] | None = None


def create_worker_app(settings: WorkerSettings) -> Celery:
    """Register the ID-only processing task on one GPU-safe Celery app."""
    task_queue = create_task_queue(settings.task_broker_url)
    task_queue.conf.update(worker_concurrency=1, worker_pool="prefork")

    @task_queue.task(  # type: ignore[untyped-decorator]
        name=PROCESS_OBSERVATION_TASK,
        bind=True,
        autoretry_for=(
            DatabaseUnavailableError,
            ModelUnavailableError,
            ObjectStorageError,
            SoftTimeLimitExceeded,
        ),
        max_retries=_TRANSIENT_MAX_RETRIES,
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
        running_retries = _running_retries(task)
        task.override_max_retries = running_retries + _TRANSIENT_MAX_RETRIES
        state = run_observation_processing(settings, message)
        if state is JobState.RUNNING:
            headers = dict(task.request.headers or {})
            headers[_RUNNING_RETRIES_HEADER] = running_retries + 1
            task.retry(
                countdown=_RUNNING_RETRY_SECONDS,
                max_retries=(task.request.retries - running_retries + _RUNNING_MAX_RETRIES),
                headers=headers,
            )
        return str(state.value)

    return task_queue


def _running_retries(task: _RetryingTask) -> int:
    value = (task.request.headers or {}).get(_RUNNING_RETRIES_HEADER, 0)
    if (
        type(value) is not int
        or not 0 <= value <= _RUNNING_MAX_RETRIES
        or value > task.request.retries
    ):
        raise ValueError("invalid running retry metadata")
    return value


def run_observation_processing(
    settings: WorkerSettings,
    message: ObservationProcessingTaskMessage,
) -> JobState:
    """Run one synchronous Celery delivery through the shared async use case."""
    runtime = _get_worker_runtime(
        settings.media_embedder_plugin,
        json.dumps(settings.media_embedder_config, sort_keys=True, separators=(",", ":")),
    )
    return runtime.run(settings, message)


async def _process_observation_once(
    settings: WorkerSettings,
    media_embedder: Embedder,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
) -> JobState:
    store = PostgresMemoryStore(
        settings.database_url, embedding_dimension=settings.embedding_dimension
    )
    media_access = S3MediaAccess(
        settings.object_storage,
        url_lifetime_seconds=_MEDIA_URL_LIFETIME_SECONDS,
    )
    async with AsyncExitStack() as resources:
        generator = load_generator(settings.generator_plugin, settings.generator_config)
        resources.push_async_callback(close_model, generator)
        text_embedder = load_embedder(settings.text_embedder_plugin, settings.text_embedder_config)
        resources.push_async_callback(close_model, text_embedder)
        await store.open()
        resources.push_async_callback(store.close)
        job = await ProcessObservation(
            store,
            PerceptionPipeline(generator),
            media_embedder,
            text_embedder,
            media_url_signer=media_access,
            clip_sampling=settings.clip_sampling,
        ).run(tenant_id, observation_id, job_id)
    return job.state


def _get_worker_runtime(
    plugin: str,
    config_json: str,
) -> _WorkerRuntime:
    global _worker_runtime, _worker_runtime_key
    # ponytail: one frozen model per worker process; split queues when multiple models are needed.
    key = (plugin, config_json)
    if _worker_runtime is not None:
        if _worker_runtime_key != key:
            raise RuntimeError("worker media plugin configuration changed after initialization")
        return _worker_runtime
    loop = asyncio.new_event_loop()
    try:
        media_embedder = loop.run_until_complete(_create_media_embedder(plugin, config_json))
    except BaseException:
        loop.close()
        raise
    _worker_runtime = _WorkerRuntime(loop, media_embedder)
    _worker_runtime_key = key
    return _worker_runtime


async def _create_media_embedder(plugin: str, config_json: str) -> Embedder:
    """Construct the plugin while its owning event loop is running."""
    config = json.loads(config_json)
    if not isinstance(config, dict):
        raise ValueError("cached embedder config must be a JSON object")
    return load_embedder(plugin, config)


def _dispose_worker_runtime() -> None:
    global _worker_runtime, _worker_runtime_key
    runtime = _worker_runtime
    _worker_runtime = None
    _worker_runtime_key = None
    if runtime is not None:
        runtime.close()


@worker_process_shutdown.connect(weak=False)  # type: ignore[untyped-decorator]
def _close_worker_runtime(**_kwargs: object) -> None:
    """Release the process-owned plugin before the prefork child exits."""
    _dispose_worker_runtime()
