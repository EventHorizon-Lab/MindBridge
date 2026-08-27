"""Celery worker composition for observation-derived memory."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Collection, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from time import time
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

from mindbridge.application.capabilities import Embedder, Generator
from mindbridge.application.evidence_clips import ClipSampling
from mindbridge.application.pipelines import PerceptionPipeline
from mindbridge.application.process_observation import (
    ProcessObservation,
    record_unclaimed_processing_failure,
)
from mindbridge.configuration import (
    PluginConfigModel,
    PluginInteger,
    PluginNumber,
    configuration_source,
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
from mindbridge.infrastructure._postgres_jobs import OBSERVATION_JOB_STALE_AFTER_SECONDS
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
    observation_delivery_window_seconds,
    observation_queues,
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
    openai_embedder_config,
    openai_generator_config,
    sentence_transformers_media_embedder_config,
)
from mindbridge.models.plugins import close_model, load_embedder, load_generator
from mindbridge.telemetry import (
    TelemetryProviders,
    configure_logging,
    configure_telemetry,
    flush_timing_summary,
)

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
_TRANSIENT_BACKOFF_MAX_SECONDS = 300
_TRANSIENT_MAX_RETRIES = 2 * math.ceil(
    OBSERVATION_JOB_STALE_AFTER_SECONDS / _TRANSIENT_BACKOFF_MAX_SECONDS
)
"""A backstop for `_retry_seconds_remaining`, which is what actually ends a transient chain.

Five was the whole budget before, and full jitter draws each delay uniformly below the doubling
ceiling, so those five attempts spanned at most 1+2+4+8+16 = 31 seconds and about 15 on average
-- a dependency outage of any real length exhausted them and failed the row on its first
minute. Counting attempts cannot express "keep trying for a while" because the wall clock a
count buys depends on queue depth. What is left for the count to do is stop a run of near-zero
jitter draws from spinning: at the 300 second ceiling and half of it on average, twice the
ceiling-count is what fits inside one deadline.
"""
_RUNNING_RETRIES_HEADER = "mindbridge_running_retries"
_RETRY_DEADLINE_HEADER = "mindbridge_retry_deadline"
_RETRY_DEADLINE_SECONDS = float(OBSERVATION_JOB_STALE_AFTER_SECONDS)
"""How long one delivery chain may keep retrying, in wall clock rather than attempts.

One stale window: past it the row is claimable by any delivery, so a chain that has been
bouncing that long has nothing left that a fresh claim would not do, and holding the message
only keeps the work out of the ledger's reach. Retry counts do not bound wall clock at all --
`countdown` is a minimum and the message goes to the tail of one shared FIFO, so during the
2026-08-24 evaluation a nominal 1 second backoff took 43 hours and 30 second claim polls
outlived the rows they were polling by hours. A deadline is the one bound queue depth can only
shorten.
"""
DEFAULT_WORKER_CONCURRENCY = 1
MAXIMUM_WORKER_CONCURRENCY = 32

# ponytail: an allowlist of plugin names, because a plugin cannot be asked whether it holds a
# device. Give `Embedder` a "runs in this process" property if a third-party local plugin ever
# needs to be covered by the guard below.
_IN_PROCESS_EMBEDDER_PLUGINS = frozenset({"jina", "sentence-transformers"})

_IN_PROCESS_EMBEDDER_VRAM_GIB = 3.7
_IN_PROCESS_EMBEDDER_RSS_GIB = 1.4
"""What one prefork child holds for the bundled Jina Omni encoder, measured on an RTX 5090.

3 745 MiB of VRAM (weights plus the child's own CUDA context) and 1.36 GiB of resident host
memory, per loaded copy, before any activation memory. The 2026-08-21 evaluation ran six children at
4.3-4.8 GB each and reached 30.2 of the card's 32.6 GB with the GPU at 1-5% utilisation.
"""

_MAX_SAMPLING_FRAMES_PER_SECOND = 20.0
"""A sanity bound on the sampling rate, enforced where the value enters the process.

It is not derived from anything. It used to be `MAX_PROXY_SAMPLED_FRAMES` over
`MAX_SAMPLING_FLOOR_MS`, and stood between a deployment and a rate that would silently skip
every generation proxy; that budget is gone, and the number is kept only because it is what
deployments already run under.

It still has a job that is not arbitrary, though not the one recorded here before. The bound
was documented as what rejects `Infinity` and `NaN`; measured, it is not. `plugin_configuration`
parses this variable with `parse_constant=_reject_json_constant`, so both literals are refused
before pydantic sees them, bound or no bound. What only the bound catches is a finite literal
that overflows: `1e400` is valid JSON, becomes `inf` in Python, passes `gt=0`, and without an
upper bound reaches `ClipSampling.frames_per_second` as `inf`. Absurd finite rates go the same
way -- `1e9` was accepted -- and `_stream_rate` would hand that to libx264 as an integral H.264
frame rate. Raise it if a deployment has a real reason; do not remove it.
"""

_SUPPORTED_WORKER_POOLS = ("prefork", "solo")
"""Celery pools compatible with the child-owned synchronous Worker runtime."""


_worker_telemetry: TelemetryProviders | None = None


@worker_process_init.connect(weak=False)  # type: ignore[untyped-decorator]
def _configure_worker_telemetry(**_kwargs: object) -> None:
    """Initialize logging and exporters after Celery forks its process-safe worker child."""
    global _worker_telemetry
    _dispose_worker_runtime()
    configure_logging("mindbridge-worker")
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
    media_embedder_plugin: str = "openai"
    text_embedder_plugin: str = "openai"
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    clip_sampling: ClipSampling = field(default_factory=ClipSampling)
    worker_concurrency: int = DEFAULT_WORKER_CONCURRENCY
    """How many observations this worker may have in flight at once.

    One observation is one model call and then some encoding, so a worker against remote
    model endpoints spends most of its budget waiting on the network and a concurrency of one
    leaves that capacity unused. Raising it is the single largest throughput lever a
    deployment has.

    It defaults to one because the ceiling is not the network in every deployment: Celery's
    prefork pool builds the media embedder once per child, so a worker whose embedder plugin
    loads the model in-process multiplies device memory by this number rather than
    overlapping anything. Raise it when the models are served over the network, and remember
    that each child opens its own database pool -- see MINDBRIDGE_DATABASE_MAX_POOL_SIZE.
    """
    vram_budget_gib: float = _IN_PROCESS_EMBEDDER_VRAM_GIB

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("task_broker_url", self.task_broker_url),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not 1 <= self.worker_concurrency <= MAXIMUM_WORKER_CONCURRENCY:
            raise ValueError(
                "worker_concurrency must be between 1 and "
                f"{MAXIMUM_WORKER_CONCURRENCY}, not {self.worker_concurrency}"
            )
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
        source = configuration_source(environ)
        generator_plugin = source.get("MINDBRIDGE_GENERATOR_PLUGIN", "openai")
        media_plugin = source.get("MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN", "openai")
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
            worker_concurrency=_worker_concurrency_from_environment(source),
            vram_budget_gib=_vram_budget_from_environment(source),
        )


def _vram_budget_from_environment(source: Mapping[str, str]) -> float:
    """Read how much device memory this deployment lets resident encoders occupy.

    The default is one copy, so a second in-process encoder stays a decision rather than an
    accident. A larger card says otherwise in one number: a guard with no way to say yes gets
    routed around, and the route an operator finds is a flag the guard cannot see. The estimate
    this bounds covers weights and CUDA contexts only, so leave room for activation memory --
    the evaluation's six children measured 30.2 GB against an estimated 22.2.
    """
    value = optional_environment_value(source, "MINDBRIDGE_WORKER_VRAM_BUDGET_GIB")
    if value is None:
        return _IN_PROCESS_EMBEDDER_VRAM_GIB
    budget = float(value)
    # Finite as well as positive, and this field has no upper bound to reject an infinity on its
    # behalf: `float()` accepts `inf`, `Infinity`, and any literal that overflows to one, and
    # every estimate compares below an infinite budget -- so the one variable that raises the
    # guard would switch it off instead, silently, on a typo. NaN fails `isfinite` too.
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("MINDBRIDGE_WORKER_VRAM_BUDGET_GIB must be a finite positive number")
    return budget


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
    if plugin in {"jina", "sentence-transformers"}:
        return lambda: sentence_transformers_media_embedder_config(source)
    if plugin == "openai":
        return lambda: plugin_configuration(
            source,
            "MINDBRIDGE_EMBEDDER_CONFIG_JSON",
            lambda: openai_embedder_config(
                source,
                request_timeout_seconds=_MODEL_REQUEST_TIMEOUT_SECONDS,
            ),
        )
    return None


class _MediaSamplingConfig(PluginConfigModel):
    """Strict schema for the optional media sampling object.

    It reads value types as well as key names, which a hand-written key check does not: a
    quoted number or a `"false"` string for the off-switch would otherwise be accepted and
    silently mean something else.
    """

    # The upper bound is load-bearing beyond the rate itself: `1e400` is valid JSON, overflows
    # to `inf`, and `gt=0` admits it, so without `le=` an infinite frame rate reaches the media
    # layer and is caught -- if at all -- by a downstream invariant rather than at the boundary
    # it entered through. It is *not* what rejects the `Infinity` and `NaN` literals, which this
    # comment used to claim: `plugin_configuration` refuses those at the parser. `1e400` is the
    # case that pins this bound, and it is in the parametrized refusals for that reason.
    frames_per_second: Annotated[
        PluginNumber,
        Field(gt=0, le=_MAX_SAMPLING_FRAMES_PER_SECOND),
    ] = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: Annotated[PluginInteger, Field(gt=0)] = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: Annotated[PluginInteger, Field(gt=0)] = DEFAULT_IMAGE_MAX_PIXELS
    generation_proxy: StrictBool = True
    proxy_audio: StrictBool = True


def _worker_concurrency_from_environment(source: Mapping[str, str]) -> int:
    """Read the one knob that decides whether a network-bound worker sits idle."""
    raw = optional_environment_value(source, "MINDBRIDGE_WORKER_CONCURRENCY")
    if raw is None:
        return DEFAULT_WORKER_CONCURRENCY
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError("MINDBRIDGE_WORKER_CONCURRENCY must be an integer") from error


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
    """One event loop and all reusable clients owned by a prefork Worker child."""

    loop: asyncio.AbstractEventLoop
    embedders: tuple[Embedder, Embedder] | None = None
    store: PostgresMemoryStore | None = None
    generator: Generator | None = None
    media_access: S3MediaAccess | None = None
    resources: AsyncExitStack = field(default_factory=AsyncExitStack, init=False, repr=False)

    def run(self, settings: WorkerSettings, message: ObservationProcessingTaskMessage) -> JobState:
        task = self.loop.create_task(
            _process_observation_once(
                settings,
                self,
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

    async def encoders(self, settings: WorkerSettings) -> tuple[Embedder, Embedder]:
        """Load both encoder plugins once per child, while their owning loop is running.

        The text encoder used to be loaded inside every task and released with a `close_model`
        the bundled encoder does not implement, so each observation re-allocated its weights and
        reclaimed nothing. What is left of that cost is the first delivery to a fresh child and
        every `--max-memory-per-child` recycle after it, which is exactly when a card that is
        already full says so. Loading from here rather than while the runtime is being built is
        what puts that allocation inside an open store, where the failure can reach the ledger
        instead of being acked away with its message.
        """
        if self.embedders is None:
            media = load_embedder(settings.media_embedder_plugin, settings.media_embedder_config)
            try:
                text = (
                    media
                    if settings.media_embedder_plugin == settings.text_embedder_plugin
                    and settings.media_embedder_config == settings.text_embedder_config
                    else load_embedder(settings.text_embedder_plugin, settings.text_embedder_config)
                )
            except BaseException:
                await close_model(media)
                raise
            self.embedders = (media, text)
            self.resources.push_async_callback(close_model, media)
            if text is not media:
                self.resources.push_async_callback(close_model, text)
        return self.embedders

    async def memory_store(self, settings: WorkerSettings) -> PostgresMemoryStore:
        """Open one database pool per child instead of rebuilding it for every delivery."""
        if self.store is None:
            store = PostgresMemoryStore(
                settings.database_url,
                embedding_dimension=settings.embedding_dimension,
                max_pool_size=resolve_database_max_pool_size(),
            )
            await store.open()
            self.store = store
            self.resources.push_async_callback(store.close)
        return self.store

    def generation_model(self, settings: WorkerSettings) -> Generator:
        """Keep the provider client and its learned capability state for the child lifetime."""
        if self.generator is None:
            self.generator = load_generator(settings.generator_plugin, settings.generator_config)
            self.resources.push_async_callback(close_model, self.generator)
        return self.generator

    def media_url_signer(self, settings: WorkerSettings) -> S3MediaAccess:
        """Reuse the S3 connection pools used to sign and materialize derived media."""
        if self.media_access is None:
            self.media_access = S3MediaAccess(settings.object_storage)
            self.resources.push_async_callback(self.media_access.close)
        return self.media_access

    def close(self) -> None:
        try:
            self.loop.run_until_complete(self.resources.aclose())
        finally:
            self.loop.close()


_worker_runtime: _WorkerRuntime | None = None
_worker_runtime_settings: WorkerSettings | None = None


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


def require_whole_observation_queue_set(consume_from: Collection[str]) -> None:
    """Refuse a Worker that reads part of the observation queue set but not all of it.

    Observations are published across a shard set so one tenant's backlog cannot starve every
    other tenant, and every shard is in this app's own queue set -- so a Worker started with no
    `-Q` reads all of them and needs no coordination. Narrowing it to `-Q mindbridge`, which is
    the name the queue had before it was sharded and the one an older runbook reaches for, leaves
    the shards with no consumer. The symptom is not an error: the publish succeeds, the ledger
    says `pending`, and the work simply never runs -- the same silent shape the in-process model
    guard below exists to replace.

    Reading *none* of them is a different role rather than a mistake: the consolidation sweep runs
    that way, on its own queue. Only partial coverage is refused.
    """
    selected = set(consume_from)
    observation = set(observation_queues())
    missing = tuple(sorted(observation - selected))
    if missing and observation & selected:
        raise ValueError(
            "these observation queues would have no consumer: "
            + ", ".join(missing)
            + " -- start the Worker without -Q, or name every observation queue in it"
        )


def require_stale_window_covers_delivery(settings: WorkerSettings) -> None:
    """Refuse a Worker whose longest permitted attempt outlives the stale-claim window.

    `OBSERVATION_JOB_STALE_AFTER_SECONDS` is a constant because the claim predicate is SQL, but
    the length it has to cover is not: it is the generator's configured
    `request_timeout_seconds` plus the post-model budget, the hard-limit margin, and the
    re-delivery margin. The bundled 1 800 s generator lands at 2 280 s against a 2 400 s
    window, and every 100 s added to that timeout eats 100 s of the remaining 120.

    Past it the two disagree about the same attempt: the delivery is still running and paying
    for its model call, while the ledger already considers the row reclaimable, so a concurrent
    delivery or one `mindbridge jobs --republish` buys the same observation a second time --
    the duplicate-work defect the window was widened to close. Refusing at boot is the only
    place that can say so, because nothing about the running system looks wrong afterwards.
    """
    budget = processing_budget_seconds(settings.generator_config)
    window = observation_delivery_window_seconds(budget)
    if window <= OBSERVATION_JOB_STALE_AFTER_SECONDS:
        return
    # Everything the window adds on top of the one number an operator sets, so the remedy names
    # a value they can paste rather than an arithmetic they have to redo.
    overhead = window - (budget - _POST_MODEL_BUDGET_SECONDS)
    raise ValueError(
        f"one delivery may stay alive for {window:.0f} s, past the "
        f"{OBSERVATION_JOB_STALE_AFTER_SECONDS} s stale-claim window, so a live attempt would "
        "be reclaimable and its observation processed twice -- set the generator's "
        f"request_timeout_seconds to at most "
        f"{OBSERVATION_JOB_STALE_AFTER_SECONDS - overhead:.0f} s"
    )


def require_bounded_in_process_models(
    settings: WorkerSettings,
    concurrency: int,
    *,
    pool: object = "prefork",
) -> None:
    """Refuse to fork an encoder into every child, which is what exhausted the GPU.

    A prefork child owns its own plugins, so an in-process encoder is loaded once per child and
    its device memory scales with the pool size. `--max-memory-per-child` bounds resident host
    memory and structurally cannot bound VRAM, so during the 2026-08-21 evaluation nothing
    reported this until the allocator failed hours in -- 479 CUDA out-of-memory errors and a
    kernel `global_oom` on the host.

    The child runtime drives one event loop synchronously, so pools that can invoke it from
    concurrent greenlets or threads are refused even when they would hold only one model copy.
    Of the supported pools, only a prefork configuration that really holds device memory in more
    than one process needs the VRAM check; an encoder pinned to `device=cpu` holds no VRAM at all.
    What is left is measured against a budget the deployment can raise, because a guard that
    refuses valid configurations and offers no way to say yes gets routed around -- and the route
    is a flag this cannot see.

    Refusing costs no capability. One worker process per assigned GPU at concurrency 1 is what
    the deployment guide already recommends for scaling, and a served encoder removes the
    resident model from every child, which is both faster and unbounded by the card.
    """
    pool_name = _require_supported_worker_pool(pool)
    slots = tuple(
        f"{variable}={plugin}"
        for variable, plugin, config in (
            (
                "MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN",
                settings.media_embedder_plugin,
                settings.media_embedder_config,
            ),
            (
                "MINDBRIDGE_EMBEDDER_PLUGIN",
                settings.text_embedder_plugin,
                settings.text_embedder_config,
            ),
        )
        if plugin in _IN_PROCESS_EMBEDDER_PLUGINS and _holds_device_memory(config)
    )
    if concurrency <= 1 or not slots or "solo" in pool_name:
        return
    # Every named slot is cached for the life of the child, so a child holding two of them
    # holds two copies -- which is the evaluation's own configuration.
    copies = concurrency * len(slots)
    vram = copies * _IN_PROCESS_EMBEDDER_VRAM_GIB
    if vram <= settings.vram_budget_gib:
        return
    raise ValueError(
        f"{' and '.join(slots)} loads an encoder into the Worker process, and every prefork "
        f"child holds its own, so a pool of {concurrency} means {copies} of them: about "
        f"{vram:.1f} GiB of VRAM and about {copies * _IN_PROCESS_EMBEDDER_RSS_GIB:.1f} GiB of "
        f"resident host memory, past the {settings.vram_budget_gib:.1f} GiB this deployment "
        "allows and before any activation memory. --max-memory-per-child bounds resident memory "
        "and cannot bound VRAM. Serve the encoder instead -- set the plugin to openai and give "
        "it an endpoint, which leaves no model in any child -- or run one worker process per "
        "assigned GPU at --concurrency 1. A card that can hold them all says so in "
        "MINDBRIDGE_WORKER_VRAM_BUDGET_GIB."
    )


def _holds_device_memory(config: Mapping[str, object]) -> bool:
    """Read the device the plugin will actually select: only `cpu` costs no VRAM.

    An absent key and `auto` both resolve to CUDA whenever a card is present, so only the
    explicit host device is exempt. Matching `select_torch_device` means normalising the same
    way it does, or a config it accepts would be read here as something else.
    """
    device = config.get("device")
    return not isinstance(device, str) or device.strip().lower() != "cpu"


def _require_supported_worker_pool(pool: object) -> str:
    """Return a supported pool name or refuse a runtime shape that can enter concurrently.

    Celery has not always resolved `pool_cls` to a class when `worker_init` fires, so both the CLI
    alias and a resolved class module are accepted. An allowlist keeps a future pool implementation
    from sharing the runtime until its concurrency semantics have been reviewed explicitly.
    """
    name = (pool if isinstance(pool, str) else getattr(pool, "__module__", "")).lower()
    supported_pool = next(
        (
            supported
            for supported in _SUPPORTED_WORKER_POOLS
            if name == supported or name.startswith(f"celery.concurrency.{supported}")
        ),
        None,
    )
    if supported_pool is None:
        shown = name or type(pool).__name__
        raise ValueError(
            f"worker pool {shown!r} is unsupported; use prefork or solo because the Worker "
            "runtime owns one synchronous event loop per process"
        )
    return supported_pool


def _resolved_pool_size(sender: object) -> int:
    """Read how many children the worker will actually fork, `--autoscale` included.

    `worker_init` fires before the Pool bootstep that parses that flag: celery 5.6.3 sends the
    signal at `worker/worker.py:127` and sets `autoscale` and `max_concurrency` at
    `worker/components.py:117-127`, inside the later `blueprint.apply`. At the signal the flag
    is still an unparsed entry in `sender.options` and `concurrency` is this app's default of 1,
    so reading `concurrency` alone waved `--autoscale=6,1` through to six children -- the
    workaround an operator refused at `--concurrency` reaches for first.
    """
    autoscale = getattr(sender, "autoscale", None)
    if autoscale is None:
        options = getattr(sender, "options", None)
        autoscale = options.get("autoscale") if isinstance(options, Mapping) else None
    if isinstance(autoscale, str):
        autoscale = autoscale.split(",")
    if isinstance(autoscale, (list, tuple)):
        autoscale = autoscale[0] if autoscale else None
    return max(_child_count(getattr(sender, "concurrency", 1)), _child_count(autoscale))


def _child_count(value: object) -> int:
    """Read one pool-size value; anything unparseable makes no claim on the card."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def create_worker_app(settings: WorkerSettings) -> Celery:
    """Register the ID-only processing task on one GPU-safe Celery app."""
    task_queue = create_task_queue(
        settings.task_broker_url,
        processing_budget_seconds=processing_budget_seconds(settings.generator_config),
    )
    task_queue.conf.update(
        worker_concurrency=settings.worker_concurrency,
        worker_pool="prefork",
    )

    # `worker_init` is the earliest point at which every input to the pool size is readable:
    # the CLI flags, this app's default, and Celery's own CPU-count fallback. `--autoscale` is
    # readable but not yet parsed there, which `_resolved_pool_size` handles. The dispatch UID
    # keeps a re-created app from stacking a second receiver.
    worker_init.disconnect(dispatch_uid="mindbridge-worker-startup-guard")

    @worker_init.connect(weak=False, dispatch_uid="mindbridge-worker-startup-guard")  # type: ignore[untyped-decorator]
    def _guard_worker_startup(sender: object = None, **_kwargs: object) -> None:
        try:
            require_bounded_in_process_models(
                settings,
                _resolved_pool_size(sender),
                pool=getattr(sender, "pool_cls", "prefork"),
            )
            # `-Q` is resolved before this signal: `WorkController.setup_instance` calls
            # `setup_queues` first and sends `worker_init` twenty lines later, so unlike
            # `--autoscale` the selection is readable here rather than merely present.
            require_whole_observation_queue_set(task_queue.amqp.queues.consume_from)
            require_stale_window_covers_delivery(settings)
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
        retry_backoff_max=_TRANSIENT_BACKOFF_MAX_SECONDS,
        retry_jitter=True,
        pydantic=True,
        pydantic_strict=True,
    )
    def process_observation_task(
        task: _RetryingTask,
        message: ObservationProcessingTaskMessage,
    ) -> str:
        running_retries = _running_retries(task)
        may_retry = _retry_seconds_remaining(task) > 0
        # Past the deadline the count is set to the attempts already made, which is how a
        # `retry()` Celery raises on its own behalf re-raises the original error instead: the
        # row lands `failed`, where the claim predicate and `mindbridge jobs --republish` can
        # both still reach it, rather than staying attached to a message that will not finish.
        task.override_max_retries = (
            running_retries + _TRANSIENT_MAX_RETRIES if may_retry else task.request.retries
        )
        state = run_observation_processing(settings, message)
        if state is JobState.RUNNING and may_retry:
            headers = dict(task.request.headers or {})
            headers[_RUNNING_RETRIES_HEADER] = running_retries + 1
            task.retry(
                countdown=_RUNNING_RETRY_SECONDS,
                max_retries=(task.request.retries - running_retries + _RUNNING_MAX_RETRIES),
                headers=headers,
            )
        return str(state.value)

    return task_queue


def _retry_seconds_remaining(task: _RetryingTask) -> float:
    """Report what is left of this delivery chain's wall-clock retry budget.

    The deadline is stamped on the first delivery and then carried by the message: Celery
    re-publishes `request.headers` on every retry, including the ones `autoretry_for` raises
    without consulting this task, which is why it is written back onto the request rather than
    only passed to the `retry()` call below. A header is caller-supplied data, so a deadline
    further out than one full budget is treated as absent rather than trusted.
    """
    headers = dict(task.request.headers or {})
    deadline = headers.get(_RETRY_DEADLINE_HEADER)
    ceiling = time() + _RETRY_DEADLINE_SECONDS
    if type(deadline) is not float and type(deadline) is not int:
        deadline = ceiling
        headers[_RETRY_DEADLINE_HEADER] = deadline
        task.request.headers = headers
    return min(float(deadline), ceiling) - time()


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
    return _get_worker_runtime(settings).run(settings, message)


async def _process_observation_once(
    settings: WorkerSettings,
    runtime: _WorkerRuntime,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
) -> JobState:
    """Prepare this delivery inside the open store, so a setup failure reaches the ledger.

    The store is opened first and everything fallible after it is guarded, because `task_acks_late`
    discards the message of a task that raised: work done before `ProcessObservation` claims the
    row fails with nothing written and nothing left to redeliver it. Only the pool itself is
    outside, and a pool that cannot open raises `DatabaseUnavailableError`, which is retried, or
    the schema refusal -- a deployment-wide mismatch that would otherwise stamp `failed` on every
    row it touched, when `pending` is what `mindbridge jobs --republish` picks up once the schema
    is fixed.
    """
    store = await runtime.memory_store(settings)
    try:
        media_embedder, text_embedder = await runtime.encoders(settings)
        generator = runtime.generation_model(settings)
        processor = ProcessObservation(
            store,
            PerceptionPipeline(generator),
            media_embedder,
            text_embedder,
            media_url_signer=runtime.media_url_signer(settings),
            clip_sampling=settings.clip_sampling,
        )
    except Exception as error:
        await record_unclaimed_processing_failure(
            store,
            tenant_id,
            observation_id,
            job_id,
            error,
        )
        raise
    job = await processor.run(tenant_id, observation_id, job_id)
    return job.state


def _get_worker_runtime(settings: WorkerSettings) -> _WorkerRuntime:
    global _worker_runtime, _worker_runtime_settings
    # ponytail: one frozen resource set per worker process; split queues when more are needed.
    if _worker_runtime is not None:
        if _worker_runtime_settings != settings:
            raise RuntimeError("worker configuration changed after runtime initialization")
        return _worker_runtime
    runtime = _WorkerRuntime(asyncio.new_event_loop())
    _worker_runtime = runtime
    _worker_runtime_settings = settings
    return runtime


def _dispose_worker_runtime() -> None:
    global _worker_runtime, _worker_runtime_settings
    runtime = _worker_runtime
    _worker_runtime = None
    _worker_runtime_settings = None
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
        try:
            flush_timing_summary()
        finally:
            if providers is not None:
                if providers.tracer is not None:
                    providers.tracer.shutdown()
                if providers.meter is not None:
                    providers.meter.shutdown()
