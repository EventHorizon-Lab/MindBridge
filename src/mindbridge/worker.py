"""Celery worker composition for observation-derived memory."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, NoReturn, Protocol

from celery import Celery
from celery.exceptions import (
    SoftTimeLimitExceeded,
)
from celery.signals import (
    worker_init,
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
from mindbridge.infrastructure.postgres import (
    PostgresMemoryStore,
    resolve_database_max_pool_size,
)
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
    DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS,
    embedding_dimension_from_environment,
    jina_media_embedder_config,
    openai_embedder_config,
    openai_generator_config,
)
from mindbridge.models.plugins import close_model, load_embedder, load_generator
from mindbridge.telemetry import TelemetryProviders, configure_telemetry

_MODEL_REQUEST_TIMEOUT_SECONDS = 780.0
_POST_MODEL_BUDGET_SECONDS = 300.0
"""What one observation needs after its model call: media and text encoding, and the graph write.

Encoding a 30-second clip through a local Omni model is minutes on its own, so this cannot be the
handful of seconds a fast deployment gets away with. It is added to the model deadline to size the
Celery budget, which is what keeps a slow generator from being cut off by the task limit instead of
by its own timeout.
"""
_RUNNING_RETRY_SECONDS = 30
_RUNNING_MAX_RETRIES = 40
_TRANSIENT_MAX_RETRIES = 5
_RUNNING_RETRIES_HEADER = "mindbridge_running_retries"

# ponytail: an allowlist of plugin names, because a plugin cannot be asked whether it holds a
# device. Give `Embedder` a "runs in this process" property if a third-party local plugin ever
# needs to be covered by the guard below.
_IN_PROCESS_EMBEDDER_PLUGINS = frozenset({"jina"})

_IN_PROCESS_EMBEDDER_VRAM_GIB = 3.7
_IN_PROCESS_EMBEDDER_RSS_GIB = 1.4
"""What one prefork child holds for the bundled Jina Omni encoder, measured on an RTX 5090.

3 745 MiB of VRAM (weights plus the child's own CUDA context) and 1.36 GiB of resident host
memory, per child, before any activation memory. The 2026-08-21 evaluation ran six children at
4.3-4.8 GB each and reached 30.2 of the card's 32.6 GB with the GPU at 1-5% utilisation.
"""


_worker_telemetry: TelemetryProviders | None = None


@worker_process_init.connect(weak=False)  # type: ignore[untyped-decorator]
def _configure_worker_telemetry(**_kwargs: object) -> None:
    """Initialize exporters after Celery forks its process-safe worker child."""
    global _worker_telemetry
    _dispose_worker_runtime()
    _worker_telemetry = configure_telemetry("mindbridge-worker")


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
                _media_embedder_fallback(source, media_plugin),
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


def _media_embedder_fallback(
    source: Mapping[str, str],
    plugin: str,
) -> Callable[[], dict[str, object]] | None:
    """Let the media slot reach a served encoder without a second family of variables.

    The bundled local plugin needs a device and a model id, which the Worker-specific names
    already supply. Serving the same model instead needs credentials and an endpoint -- and the
    deployment necessarily has one of those already, for the text encoder that must write into
    the same embedding space. Reusing it makes the served media slot one variable rather than
    five, which matters because it is the configuration that keeps a model out of every child.
    """
    if plugin == "jina":
        return lambda: jina_media_embedder_config(source)
    if plugin == "openai":
        return lambda: openai_embedder_config(
            source,
            request_timeout_seconds=_MODEL_REQUEST_TIMEOUT_SECONDS,
        )
    return None


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
    proxy_audio: StrictBool = True


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


def processing_budget_seconds(generator_config: Mapping[str, object]) -> float:
    """Size the task budget from the deadline this deployment gave its generator.

    Perception is one model call per observation, so the budget is that call's own deadline plus
    what the encoding and graph write after it need. Reading the configured value rather than a
    constant is the point: a deployment on a slow generator raises `request_timeout_seconds` in
    one place and the Celery limits follow, instead of the task being killed mid-call by a limit
    that was written for a faster one.
    """
    # Absent means the plugin's own default applies, which is not the value this module
    # injects on the path where no generator JSON is supplied at all -- that path puts its
    # own key in the config, so it never reaches this fallback. Reading 780 here instead
    # sized the budget below the 1800 the bundled generator would actually take.
    configured = generator_config.get(
        "request_timeout_seconds", DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS
    )
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ValueError("generator request_timeout_seconds must be a number")
    if configured <= 0:
        raise ValueError("generator request_timeout_seconds must be positive")
    return float(configured) + _POST_MODEL_BUDGET_SECONDS


def require_bounded_in_process_models(settings: WorkerSettings, concurrency: int) -> None:
    """Refuse to fork an encoder into every child, which is what exhausted the GPU.

    A prefork child owns its own plugins, so an in-process encoder is loaded once per child and
    its device memory scales with `--concurrency`. `--max-memory-per-child` bounds resident host
    memory and structurally cannot bound VRAM, so during the 2026-08-21 evaluation nothing
    reported this until the allocator failed hours in -- 479 CUDA out-of-memory errors and a
    kernel `global_oom` on the host.

    Refusing costs no capability. One worker process per assigned GPU at concurrency 1 is what
    the deployment guide already recommends for scaling, and a served encoder removes the
    resident model from every child, which is both faster and unbounded by the card.
    """
    slots = tuple(
        f"{variable}={plugin}"
        for variable, plugin in (
            ("MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN", settings.media_embedder_plugin),
            ("MINDBRIDGE_EMBEDDER_PLUGIN", settings.text_embedder_plugin),
        )
        if plugin in _IN_PROCESS_EMBEDDER_PLUGINS
    )
    if concurrency <= 1 or not slots:
        return
    raise ValueError(
        f"{' and '.join(slots)} loads an encoder into the Worker process, and every prefork "
        f"child holds its own, so --concurrency {concurrency} means {concurrency} of them: "
        f"about {concurrency * _IN_PROCESS_EMBEDDER_VRAM_GIB:.1f} GiB of VRAM and about "
        f"{concurrency * _IN_PROCESS_EMBEDDER_RSS_GIB:.1f} GiB of resident host memory, before "
        "any activation memory. --max-memory-per-child bounds resident memory and cannot bound "
        "VRAM. Serve the encoder instead -- set the plugin to openai and give it an endpoint, "
        "which leaves no model in any child -- or run one worker process per assigned GPU at "
        "--concurrency 1."
    )


def create_worker_app(settings: WorkerSettings) -> Celery:
    """Register the ID-only processing task on one GPU-safe Celery app."""
    task_queue = create_task_queue(
        settings.task_broker_url,
        processing_budget_seconds=processing_budget_seconds(settings.generator_config),
    )
    task_queue.conf.update(worker_concurrency=1, worker_pool="prefork")

    # `worker_init` is the first point at which the pool size is settled: the CLI flag, this
    # app's default, and Celery's own CPU-count fallback have all been resolved by then, so the
    # guard reads what the worker will actually fork rather than what was asked for. The
    # dispatch UID keeps a re-created app from stacking a second receiver.
    @worker_init.connect(weak=False, dispatch_uid="mindbridge-in-process-model-guard")  # type: ignore[untyped-decorator]
    def _guard_in_process_models(sender: object = None, **_kwargs: object) -> None:
        try:
            require_bounded_in_process_models(settings, getattr(sender, "concurrency", 1))
        except ValueError as error:
            # Celery's `Signal.send` catches `Exception` from every receiver, logs it, and
            # carries on -- so raising `ValueError` here would print a traceback and then start
            # the worker anyway, which is the silent failure this guard exists to replace.
            # `SystemExit` is not an `Exception`, so it survives that handler and stops boot.
            raise SystemExit(f"refusing to start the Worker: {error}") from error

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
        settings.database_url,
        embedding_dimension=settings.embedding_dimension,
        max_pool_size=resolve_database_max_pool_size(),
    )
    media_access = S3MediaAccess(settings.object_storage)
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
    """Release the plugin and flush telemetry before the prefork child exits.

    Billiard ends a child with `os._exit`, which runs no `atexit` hook, so anything holding a
    buffer has to be drained from this signal explicitly. Both exporters batch: metrics are
    pushed on `OTEL_METRIC_EXPORT_INTERVAL` (60 s by default) and spans on the batch
    processor's own delay, so without this a recycled child discarded up to a full interval of
    measurements -- and `--max-memory-per-child` recycles children often.
    """
    global _worker_telemetry
    providers, _worker_telemetry = _worker_telemetry, None
    try:
        _dispose_worker_runtime()
    finally:
        if providers is not None:
            if providers.tracer is not None:
                providers.tracer.shutdown()
            if providers.meter is not None:
                providers.meter.shutdown()
