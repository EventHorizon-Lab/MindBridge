"""Checks for the deployable observation Worker boundary."""

import asyncio
from collections.abc import Awaitable, Mapping
from datetime import datetime, timezone
from time import time
from typing import Any, cast
from unittest.mock import Mock

import pytest
from celery import Task
from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.signals import worker_init, worker_process_init, worker_process_shutdown
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from pydantic import ValidationError

import mindbridge.worker as worker_module
from mindbridge.application.capabilities import Embedder, EmbedRequest, EmbedResult
from mindbridge.application.evidence_clips import (
    MAX_PROXY_SAMPLED_FRAMES,
    MAX_SAMPLING_FLOOR_MS,
    ClipSampling,
)
from mindbridge.consolidation_worker import CONSOLIDATION_QUEUE
from mindbridge.core import (
    DatabaseUnavailableError,
    JobId,
    JobState,
    ModelUnavailableError,
    ObjectStorageError,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
)
from mindbridge.infrastructure._postgres_jobs import OBSERVATION_JOB_STALE_AFTER_SECONDS
from mindbridge.infrastructure.task_queue import (
    PROCESS_OBSERVATION_TASK,
    ObservationProcessingTaskMessage,
    create_task_queue,
    observation_queues,
)
from mindbridge.models.openai import _GeneratorConfig
from mindbridge.telemetry import TelemetryProviders
from mindbridge.worker import (
    WorkerSettings,
    create_worker_app,
    processing_budget_seconds,
    require_bounded_in_process_models,
    require_whole_observation_queue_set,
)


def test_worker_settings_pin_models_and_redact_credentials() -> None:
    """Worker startup fixes model identity without exposing injected secrets."""
    settings = WorkerSettings.from_environment(_environment())

    assert settings.generator_config["model_id"] == "qwen3.8-max"
    assert settings.media_embedder_plugin == settings.text_embedder_plugin == "openai"
    assert settings.media_embedder_config["model_id"] == (
        "jinaai/jina-embeddings-v5-omni-small-retrieval"
    )
    assert settings.media_embedder_config == settings.text_embedder_config
    assert "database-secret" not in repr(settings)
    assert "broker-secret" not in repr(settings)
    assert "generator-secret" not in repr(settings)
    assert "text-embedding-secret" not in repr(settings)


def test_worker_text_encoder_shares_the_deployment_wide_embedder_contract() -> None:
    """A separate Worker-only encoder contract could silently strand vectors in another space."""
    settings = WorkerSettings.from_environment(_environment())

    assert settings.text_embedder_config["endpoint"] == "https://text.example.test/v1"
    assert settings.text_embedder_config["space_id"] == settings.media_embedder_config["space_id"]
    assert settings.text_embedder_config["dimension"] == settings.media_embedder_config["dimension"]


def test_worker_reuses_and_closes_the_shared_embedder_on_its_own_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both encoder slots are loaded once per child, not once per delivery.

    The text slot was loaded inside the task and released with a `close_model` the bundled
    encoder does not implement, so every observation re-allocated its weights -- and that
    allocation happened before `ProcessObservation` claims the ledger row. With
    `task_acks_late` an out-of-memory error there acks and drops the message while the row
    stays `pending`, which is how the 2026-08-21 evaluation turned 479 CUDA out-of-memory
    errors into ~17 `failed` rows and ~318 stranded `pending` ones.
    """
    creations: list[tuple[str, asyncio.AbstractEventLoop]] = []
    processing_loops: list[asyncio.AbstractEventLoop] = []
    closed: list[str] = []

    class LoopBoundEmbedder:
        def __init__(self, plugin: str) -> None:
            self.plugin = plugin

        async def embed(self, _request: EmbedRequest) -> EmbedResult:
            raise AssertionError("the lifecycle check does not invoke embedding")

        async def close(self) -> None:
            assert asyncio.get_running_loop() is creations[0][1]
            closed.append(self.plugin)

    def load(plugin: str, _config: Mapping[str, object]) -> LoopBoundEmbedder:
        creations.append((plugin, asyncio.get_running_loop()))
        return LoopBoundEmbedder(plugin)

    async def process(
        settings: WorkerSettings,
        runtime: worker_module._WorkerRuntime,
        *_identifiers: object,
    ) -> JobState:
        # The stand-in asks for the encoders where the real delivery does, inside the store it
        # has open: that is what makes an out-of-memory error there a recorded failure rather
        # than a dropped message.
        media_embedder, text_embedder = await runtime.encoders(settings)
        assert isinstance(media_embedder, LoopBoundEmbedder)
        assert isinstance(text_embedder, LoopBoundEmbedder)
        assert media_embedder is text_embedder
        assert (media_embedder.plugin, text_embedder.plugin) == ("openai", "openai")
        processing_loops.append(asyncio.get_running_loop())
        return JobState.SUCCEEDED

    worker_module._dispose_worker_runtime()
    monkeypatch.setattr(worker_module, "load_embedder", load)
    monkeypatch.setattr(worker_module, "_process_observation_once", process)
    settings = WorkerSettings.from_environment(_environment())
    message = ObservationProcessingTaskMessage(**_task_message())
    try:
        assert worker_module.run_observation_processing(settings, message) is JobState.SUCCEEDED
        assert worker_module.run_observation_processing(settings, message) is JobState.SUCCEEDED
    finally:
        worker_module._dispose_worker_runtime()

    # Two deliveries, one shared load: nothing is re-allocated per observation.
    assert [plugin for plugin, _loop in creations] == ["openai"]
    assert processing_loops == [creations[0][1]] * 2
    assert closed == ["openai"]


class _RecordingJobStore:
    """The ledger half of the store the Worker opens, recording only its job writes."""

    def __init__(self) -> None:
        self.states: list[str] = []
        self.attempts: list[int] = []
        self.error_codes: list[str] = []
        self.closed = False

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def claim_observation_processing_job(
        self,
        tenant_id: str,
        observation_id: str,
        job_id: str,
    ) -> ObservationJobClaim:
        self.states.append("running")
        return ObservationJobClaim(
            job=_pending_job(JobState.RUNNING, attempt=len(self.states)),
            acquired=True,
        )

    async def mark_observation_processing_failed(
        self,
        tenant_id: str,
        observation_id: str,
        job_id: str,
        *,
        attempt: int,
        error_code: str,
    ) -> ObservationProcessingJob:
        self.states.append("failed")
        self.attempts.append(attempt)
        self.error_codes.append(error_code)
        return _pending_job(JobState.FAILED, attempt=attempt, error_code=error_code)


def _pending_job(
    state: JobState,
    *,
    attempt: int,
    error_code: str | None = None,
) -> ObservationProcessingJob:
    moment = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    return ObservationProcessingJob(
        job_id=JobId("job_process_observation_01"),
        tenant_id=TenantId("tenant_01"),
        observation_id=ObservationId("observation_01"),
        state=state,
        attempt=attempt,
        error_code=error_code,
        created_at=moment,
        updated_at=moment,
    )


def test_a_model_load_failure_is_recorded_against_the_row_it_would_have_stranded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The encoder load is the first thing a fresh child does, and it can exhaust the card.

    `task_acks_late` drops the message of a task that raised, so a load that fails before the
    ledger row is claimed leaves the row `pending` with nothing left to deliver it: 479 CUDA
    out-of-memory errors against ~17 `failed` rows and ~318 stranded `pending` ones. The setup
    now runs inside the open store, so the same failure lands on the row as a recorded state.
    """
    store = _RecordingJobStore()

    def load(_plugin: str, _config: Mapping[str, object]) -> Embedder:
        # torch.OutOfMemoryError is a RuntimeError subclass; the Worker cannot import torch to
        # name it, which is also why the class cannot simply be added to `autoretry_for`.
        raise RuntimeError("CUDA out of memory")

    worker_module._dispose_worker_runtime()
    monkeypatch.setattr(worker_module, "load_embedder", load)
    monkeypatch.setattr(worker_module, "PostgresMemoryStore", lambda *_a, **_k: store)
    settings = WorkerSettings.from_environment(_local_media_environment())
    message = ObservationProcessingTaskMessage(**_task_message())
    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            worker_module.run_observation_processing(settings, message)
    finally:
        worker_module._dispose_worker_runtime()

    # Claimed only in order to record: the row is `running` for the length of one write, so a
    # child killed outright still leaves `pending`, which `mindbridge jobs` republishes at once.
    assert store.states == ["running", "failed"]
    assert store.attempts == [1]
    assert store.error_codes == ["worker_setup_failed"]
    assert store.closed is True


def test_worker_cancels_a_delivery_interrupted_outside_its_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[bool] = []

    async def process(*_arguments: object) -> JobState:
        try:
            await asyncio.Future()
            return JobState.SUCCEEDED
        finally:
            cancelled.append(True)

    loop = asyncio.new_event_loop()
    run_until_complete = loop.run_until_complete
    calls = 0

    def interrupt_once(awaitable: Awaitable[object]) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            run_until_complete(asyncio.sleep(0))
            raise SoftTimeLimitExceeded
        return run_until_complete(awaitable)

    monkeypatch.setattr(worker_module, "_process_observation_once", process)
    monkeypatch.setattr(loop, "run_until_complete", interrupt_once)
    runtime = worker_module._WorkerRuntime(loop)
    try:
        with pytest.raises(SoftTimeLimitExceeded):
            runtime.run(
                WorkerSettings.from_environment(_environment()),
                ObservationProcessingTaskMessage(**_task_message()),
            )
        assert cancelled == [True]
        assert not asyncio.all_tasks(loop)
    finally:
        loop.close()


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
    # Pinned as a set, not a membership check: ModelRequestError documents itself as
    # "retrying an unchanged model request cannot succeed", so widening this tuple has to be
    # a deliberate edit rather than something a later change can slip in.
    assert task.autoretry_for == (
        DatabaseUnavailableError,
        ModelUnavailableError,
        ObjectStorageError,
        SoftTimeLimitExceeded,
    )
    # Attempts, not the budget: `_retry_seconds_remaining` is what ends a transient chain, and
    # this count only stops near-zero jitter draws from spinning inside that deadline.
    assert task.max_retries == 16


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

    called = dict(retry.call_args.kwargs)
    headers = dict(called.pop("headers"))
    assert called == {"countdown": 30, "max_retries": 40}
    assert headers["mindbridge_running_retries"] == 1
    # Carried on the message, because that is the only bound a deep queue cannot stretch.
    assert 0 < headers["mindbridge_retry_deadline"] - time() <= 2_400


def test_worker_preserves_transient_retry_after_running_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim polling must not exhaust the task's later transient-failure budget."""

    def run(*_arguments: object) -> JobState:
        raise DatabaseUnavailableError("database unavailable")

    monkeypatch.setattr(worker_module, "run_observation_processing", run)
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    message = _task_message()

    task.push_request(
        id="delivery_01",
        retries=40,
        headers={"mindbridge_running_retries": 40},
        called_directly=False,
        is_eager=True,
        args=(message,),
        kwargs={},
    )
    try:
        with pytest.raises(Retry):
            task.run(message)
    finally:
        task.pop_request()


def test_worker_reads_media_sampling_from_the_environment() -> None:
    """Frame rate is the whole write cost of a video deployment and had no configuration."""
    settings = WorkerSettings.from_environment(
        {
            **_environment(),
            "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON": (
                '{"frames_per_second": 0.5, "max_pixels": 50176, "generation_proxy": false,'
                ' "proxy_audio": false}'
            ),
        }
    )

    assert settings.clip_sampling.frames_per_second == 0.5
    assert settings.clip_sampling.max_pixels == 50_176
    assert settings.clip_sampling.generation_proxy is False
    # A generator that cannot hear should not be sent an audio track it never reads.
    assert settings.clip_sampling.proxy_audio is False


def test_worker_accepts_the_highest_rate_a_generation_proxy_can_still_carry() -> None:
    """The bound is the media layer's own ceiling, so the rate that reaches it must pass.

    `MAX_PROXY_SAMPLED_FRAMES` over `MAX_SAMPLING_FLOOR_MS`: 40 frames across the 2 s window a
    short span is widened to. Reading both from `evidence_clips` rather than restating 20.0 is
    what keeps this from drifting when either constant is measured again.
    """
    settings = WorkerSettings.from_environment(
        {
            **_environment(),
            "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON": (
                '{"frames_per_second": %s}'
                % (MAX_PROXY_SAMPLED_FRAMES / (MAX_SAMPLING_FLOOR_MS / 1_000))
            ),
        }
    )

    assert settings.clip_sampling.frames_per_second == 20.0


def test_worker_media_sampling_defaults_keep_the_documented_encoder_budget() -> None:
    """An unset deployment must behave exactly as it did before the knob existed."""
    settings = WorkerSettings.from_environment(_environment())

    assert settings.clip_sampling == ClipSampling()


@pytest.mark.parametrize(
    "config",
    [
        '{"fps": 0.5}',
        '{"generation_proxy": "false"}',
        '{"max_pixels": true}',
        '{"frames_per_second": "0.5"}',
        '{"max_pixels": 1.5}',
        '{"frames_per_second": 0}',
        # `json.loads` parses these out of the environment variable, and `gt=0` admitted the
        # first: it reached the media layer as a frame rate and was caught by a downstream
        # invariant rather than by the boundary it entered through.
        '{"frames_per_second": Infinity}',
        '{"frames_per_second": NaN}',
        # Past 20 fps the generation proxy is skipped for every event long enough to be widened
        # to the sampling floor, so the knob would switch off the feature it is tuning.
        '{"frames_per_second": 20.5}',
    ],
)
def test_worker_rejects_a_malformed_media_sampling_knob(config: str) -> None:
    """A typo in an optional tuning knob must fail startup, not silently mean something else.
    `"false"` is a truthy string, and a quoted or floating pixel count would reach a budget
    comparison as the wrong type. A rate can also be well-formed and still mean nothing the
    media layer can serve: `Infinity`, and anything past the proxy frame ceiling."""
    with pytest.raises(ValueError):
        WorkerSettings.from_environment(
            {**_environment(), "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON": config}
        )


def test_worker_refuses_to_fork_an_in_process_encoder_into_every_child() -> None:
    """The configuration that exhausted the GPU has to fail at startup, not hours in.

    Six children each held their own 4.3-4.8 GB copy of the media encoder and reached 30.2 of
    the card's 32.6 GB while the GPU sat at 1-5% utilisation. `--max-memory-per-child` was set
    and could not help: it bounds resident host memory and cannot bound VRAM.
    """
    settings = WorkerSettings.from_environment(_local_media_environment())

    with pytest.raises(ValueError) as failure:
        require_bounded_in_process_models(settings, 6)

    message = str(failure.value)
    assert "MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=jina" in message
    # The arithmetic, not just the objection: 6 x 3.7 GiB of VRAM and 6 x 1.4 GiB resident,
    # past the default budget of one copy that keeps this configuration refused by default.
    assert "22.2 GiB of VRAM" in message
    assert "8.4 GiB of resident host memory" in message


def test_worker_counts_every_in_process_slot_it_names() -> None:
    """A child holding two cached encoders holds two copies, which the estimate has to say.

    The 2026-08-21 evaluation ran `jina` in both Worker slots. The message named both
    variables and then quoted the arithmetic for one of them, understating the card by half.
    """
    settings = WorkerSettings.from_environment(_both_slots_in_process())

    with pytest.raises(ValueError) as failure:
        require_bounded_in_process_models(settings, 6)

    message = str(failure.value)
    assert "MINDBRIDGE_EMBEDDER_PLUGIN=jina" in message
    # Two cached models per child, six children: 12 x 3.7 GiB and 12 x 1.4 GiB.
    assert "44.4 GiB of VRAM" in message
    assert "16.8 GiB of resident host memory" in message


def test_worker_allows_one_child_to_hold_an_in_process_encoder() -> None:
    """One resident copy is the shape the deployment guide recommends, not a defect."""
    require_bounded_in_process_models(
        WorkerSettings.from_environment(_local_media_environment()), 1
    )


def test_worker_allows_many_children_when_the_encoder_holds_no_device() -> None:
    """`device=cpu` costs no VRAM, so refusing it cites a budget the run cannot exhaust."""
    settings = WorkerSettings.from_environment(
        {**_local_media_environment(), "MINDBRIDGE_MEDIA_EMBEDDER_DEVICE": "cpu"}
    )

    require_bounded_in_process_models(settings, 4)


def test_worker_allows_many_workers_in_a_pool_that_shares_one_process() -> None:
    """`--pool=threads` runs one process, so its concurrency is not a device budget."""
    settings = WorkerSettings.from_environment(_local_media_environment())

    require_bounded_in_process_models(settings, 6, pool="threads")
    require_bounded_in_process_models(settings, 6, pool="solo")


def test_worker_lets_a_larger_card_declare_the_budget_it_actually_has() -> None:
    """A guard with no way to say yes gets routed around, which is what --autoscale did.

    3.7 GiB per copy was measured on one RTX 5090. A card that can hold six copies has to be
    able to say so in a number rather than by reaching for a flag the guard cannot see.
    """
    settings = WorkerSettings.from_environment(
        {**_local_media_environment(), "MINDBRIDGE_WORKER_VRAM_BUDGET_GIB": "48"}
    )

    require_bounded_in_process_models(settings, 6)

    with pytest.raises(ValueError, match=r"48\.0 GiB"):
        require_bounded_in_process_models(settings, 14)


@pytest.mark.parametrize("budget", ["0", "-8", "nan", "inf", "Infinity", "1e400"])
def test_worker_rejects_a_vram_budget_that_bounds_nothing(budget: str) -> None:
    """An override the guard cannot compare against would disable it rather than raise it.

    `nan` falls out of a positivity test on its own, and an infinity does not: `float()` accepts
    `inf`, `Infinity`, and any literal that overflows to one, and every finite estimate compares
    below it -- so a typo in the one variable that raises the budget turns the guard off instead,
    silently, which is the failure this exists to prevent.
    """
    with pytest.raises(ValueError):
        WorkerSettings.from_environment(
            {**_environment(), "MINDBRIDGE_WORKER_VRAM_BUDGET_GIB": budget}
        )


def test_worker_allows_many_children_when_every_encoder_is_served() -> None:
    """A served encoder leaves no model in any child, so concurrency stops being a GPU budget."""
    settings = WorkerSettings.from_environment(_environment())

    require_bounded_in_process_models(settings, 12)


class _StartingWorker:
    """A Celery worker at `worker_init`, holding what Celery has settled on by then."""

    def __init__(self, concurrency: int, pool_cls: str = "prefork") -> None:
        self.concurrency = concurrency
        self.pool_cls = pool_cls


def test_worker_startup_refuses_rather_than_logging_and_carrying_on() -> None:
    """The guard has to stop boot, off the pool size Celery settled on.

    `worker_init` fires after the CLI flag, this app's own default, and Celery's CPU-count
    fallback have all been resolved, which is why the check hangs off that signal rather than
    off app creation. It must raise `SystemExit`: Celery's `Signal.send` catches `Exception`
    from every receiver, logs it, and starts the worker anyway.
    """
    create_worker_app(WorkerSettings.from_environment(_local_media_environment()))

    with pytest.raises(SystemExit, match="a pool of 4"):
        worker_init.send(sender=_StartingWorker(4))


def test_worker_startup_reads_the_pool_class_celery_has_not_resolved_yet() -> None:
    """One process holds one copy however wide it runs, so boot must not refuse it.

    `pool_cls` is still the flag's own string at `worker_init`; Celery resolves it to a class
    on the line after the signal.
    """
    create_worker_app(WorkerSettings.from_environment(_local_media_environment()))

    worker_init.send(sender=_StartingWorker(6, pool_cls="threads"))


def test_worker_startup_sees_the_autoscale_ceiling_the_pool_has_not_parsed_yet() -> None:
    """`--autoscale` is what an operator refused at `--concurrency` reaches for next.

    Celery sets `autoscale` and `max_concurrency` in the Pool bootstep, which runs inside
    `blueprint.apply` -- after `worker_init` (celery 5.6.3: `worker/worker.py:127` sends the
    signal, `worker/components.py:117-127` parses the flag). At the signal the flag is still
    an unparsed entry in `sender.options` and `concurrency` reads this app's default of 1, so
    reading `concurrency` alone waved through exactly the six children this refuses.

    This drives a real `WorkController` rather than a stand-in, because the ordering it is
    asserting is Celery's, not ours. Building one establishes no broker connection.
    """
    app = create_worker_app(WorkerSettings.from_environment(_local_media_environment()))

    with pytest.raises(SystemExit, match="a pool of 6"):
        app.Worker(autoscale=(6, 1))


def test_worker_media_slot_defaults_to_the_served_encoder_without_new_variables() -> None:
    """The media slot reaches the served path without a second family of names.

    The deployment already has one embedding endpoint, for the text encoder that has to write
    into the same space, so the served media slot reuses it. That keeps the recommended
    configuration one variable away rather than five.
    """
    settings = WorkerSettings.from_environment(_environment())

    assert settings.media_embedder_config["endpoint"] == "https://text.example.test/v1"
    assert settings.media_embedder_config["api_key"] == "text-embedding-secret"
    assert settings.media_embedder_config["space_id"] == settings.text_embedder_config["space_id"]
    assert "device" not in settings.media_embedder_config


def test_worker_can_explicitly_opt_into_the_local_jina_encoder() -> None:
    settings = WorkerSettings.from_environment(
        {
            **_environment(),
            "MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN": "jina",
            "MINDBRIDGE_MEDIA_EMBEDDER_DEVICE": "cuda",
        }
    )

    assert settings.media_embedder_plugin == "jina"
    assert settings.media_embedder_config["device"] == "cuda"
    assert "endpoint" not in settings.media_embedder_config


class _RecordingProvider:
    """A telemetry provider that only records whether it was drained."""

    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1


def test_worker_child_flushes_telemetry_before_it_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Billiard ends a child with `os._exit`, so nothing buffered survives an atexit hook.

    Both exporters batch -- metrics on `OTEL_METRIC_EXPORT_INTERVAL`, 60 s by default, and
    spans on the batch processor's own delay -- so a recycled child silently discarded up to a
    full interval of measurements. `--max-memory-per-child` recycles children often.
    """
    tracer = _RecordingProvider()
    meter = _RecordingProvider()
    monkeypatch.setattr(
        worker_module,
        "configure_telemetry",
        lambda _name: TelemetryProviders(
            tracer=cast(TracerProvider, tracer),
            meter=cast(MeterProvider, meter),
        ),
    )

    worker_process_init.send(sender=None)
    worker_process_shutdown.send(sender=None)

    assert (tracer.shutdowns, meter.shutdowns) == (1, 1)
    # A second shutdown must not double-drain a provider the child no longer owns.
    worker_process_shutdown.send(sender=None)
    assert (tracer.shutdowns, meter.shutdowns) == (1, 1)


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
        "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
        "MINDBRIDGE_EMBEDDER_API_KEY": "text-embedding-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://text.example.test/v1",
    }


def _local_media_environment() -> Mapping[str, str]:
    return {**_environment(), "MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN": "jina"}


def _both_slots_in_process() -> Mapping[str, str]:
    """The 2026-08-21 evaluation's own shape: the bundled local encoder in both slots."""
    return {
        **_environment(),
        "MINDBRIDGE_EMBEDDER_PLUGIN": "jina",
        "MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN": "jina",
        "MINDBRIDGE_EMBEDDER_CONFIG_JSON": '{"model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval"}',
    }


def _task_message() -> dict[str, str]:
    return {
        "tenant_id": "tenant_01",
        "observation_id": "observation_01",
        "job_id": "job_process_observation_01",
    }


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        # An absent key means the generator applies its own default, not the 780 this module
        # injects on the path where no generator JSON is supplied at all -- that path writes its
        # key into the config, so it never reaches the fallback.
        ({}, 2_100.0),
        ({"request_timeout_seconds": 1_800.0}, 2_100.0),
        ({"request_timeout_seconds": 780.0}, 1_080.0),
    ],
)
def test_worker_sizes_its_task_budget_from_the_generator_deadline(
    configured: dict[str, object], expected: float
) -> None:
    """The Celery budget follows the model deadline the deployment actually configured.

    These two numbers used to be independent, and a deployment that gave its generator 1800s kept
    a 840s task limit: every observation was killed mid-perception and retried, so the write path
    could never finish however long it was left running.
    """
    assert processing_budget_seconds(configured) == expected


def test_a_generator_config_that_names_no_deadline_still_outlives_the_one_it_gets() -> None:
    """Reads the plugin's own default rather than restating it, so the two cannot drift apart.

    A config supplying an endpoint but no `request_timeout_seconds` is the ordinary shape, and the
    generator then applies its own default. Sizing the budget from a different constant put a
    1800s model call inside a 1080s task -- and `SoftTimeLimitExceeded` is in `autoretry_for`, so
    that deterministic overrun was retried rather than reported.
    """
    config: dict[str, object] = {"api_key": "k", "endpoint": "https://example.test/v1"}
    client_deadline = _GeneratorConfig.model_validate(config).request_timeout_seconds

    assert processing_budget_seconds(config) > client_deadline


@pytest.mark.parametrize(
    "configured", [{"request_timeout_seconds": 0}, {"request_timeout_seconds": "soon"}]
)
def test_worker_rejects_an_unusable_generator_deadline(configured: dict[str, object]) -> None:
    """A budget derived from nonsense would be worse than the constant it replaced."""
    with pytest.raises(ValueError, match="request_timeout_seconds must be"):
        processing_budget_seconds(configured)


def test_worker_concurrency_defaults_to_one_so_a_local_model_is_never_multiplied() -> None:
    """Celery's prefork pool builds the media embedder once per child.

    An embedder plugin that loads its model in-process therefore multiplies device memory by
    the concurrency rather than overlapping anything, so the safe value has to be the default
    and raising it has to be a deployment's explicit decision.
    """
    assert WorkerSettings.from_environment(_environment()).worker_concurrency == 1


def test_a_network_bound_worker_can_raise_its_concurrency() -> None:
    """One observation is mostly waiting on a model endpoint, so serializing them idles."""
    environment = dict(_environment())
    environment["MINDBRIDGE_WORKER_CONCURRENCY"] = "6"

    settings = WorkerSettings.from_environment(environment)

    assert settings.worker_concurrency == 6
    assert create_worker_app(settings).conf.worker_concurrency == 6


@pytest.mark.parametrize("configured", ["0", "-1", "33", "many", "2.5"])
def test_worker_refuses_an_unusable_concurrency(configured: str) -> None:
    """Silently clamping would leave a deployment believing it raised its throughput."""
    environment = dict(_environment())
    environment["MINDBRIDGE_WORKER_CONCURRENCY"] = configured

    with pytest.raises(ValueError, match=r"MINDBRIDGE_WORKER_CONCURRENCY|worker_concurrency"):
        WorkerSettings.from_environment(environment)


@pytest.mark.parametrize("configured", ["", "   "])
def test_a_blank_concurrency_reads_as_unset_like_every_other_optional_value(
    configured: str,
) -> None:
    """One convention across the contract: a variable exported empty is a variable unset."""
    environment = dict(_environment())
    environment["MINDBRIDGE_WORKER_CONCURRENCY"] = configured

    assert WorkerSettings.from_environment(environment).worker_concurrency == 1


def test_the_prefork_child_reports_its_timings_before_it_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """billiard ends the child with `os._exit`, so no `atexit` handler ever runs there.

    The worker owns the write path, so its summary is the one worth having, and the last
    signal Celery delivers is the only place left to emit it from.
    """
    flushed: list[int] = []
    monkeypatch.setattr(worker_module, "flush_timing_summary", lambda: flushed.append(1))

    worker_module._close_worker_runtime()

    assert flushed == [1]


def test_a_running_row_is_not_reclaimable_until_its_own_delivery_could_have_ended() -> None:
    """A healthy slow attempt must not read as abandoned; reclaiming one pays for it twice.

    Nothing refreshes `updated_at` while an attempt runs, so the ledger's stale window is the
    only thing separating "the worker died" from "the worker is still going". It therefore has
    to outlast the longest delivery the deployment permits, which is the broker's own
    re-delivery window, itself derived from the same task budget.
    """
    budget = processing_budget_seconds({})
    queue = create_task_queue("memory://", processing_budget_seconds=budget)
    redelivery = queue.conf.broker_transport_options["visibility_timeout"]

    assert redelivery > budget > 0
    assert redelivery < OBSERVATION_JOB_STALE_AFTER_SECONDS


def test_worker_stops_retrying_a_claim_it_has_chased_past_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bouncing off another delivery's claim has to end on the clock, not on a retry count.

    Each poll re-enters the same shared FIFO, so 40 nominal 30-second waits spanned 43 hours
    during the 2026-08-24 evaluation. Once the deadline passes the row is reclaimable by any
    delivery, and this one gives it back instead of holding a message that will not finish.
    """
    monkeypatch.setattr(
        worker_module,
        "run_observation_processing",
        lambda *_arguments: JobState.RUNNING,
    )
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    retry = Mock(side_effect=Retry("still running"))
    monkeypatch.setattr(task, "retry", retry)
    message = _task_message()
    task.push_request(
        id="delivery_01",
        retries=3,
        headers={
            "mindbridge_running_retries": 3,
            "mindbridge_retry_deadline": time() - 1,
        },
        called_directly=False,
        is_eager=True,
        args=(message,),
        kwargs={},
    )
    try:
        assert task.run(message) == "running"
    finally:
        task.pop_request()

    retry.assert_not_called()
    # And the next transient failure is not retried either: the same deadline governs both.
    assert task.override_max_retries == 3


def test_worker_keeps_retrying_a_transient_failure_while_the_deadline_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dependency outage outlives five attempts; the budget it has to fit in is wall clock.

    Full jitter draws each backoff below the doubling ceiling, so the five attempts this task
    used to allow spanned at most 31 seconds -- an outage of any real length failed the row in
    its first minute.
    """

    def run(*_arguments: object) -> JobState:
        raise DatabaseUnavailableError("database unavailable")

    monkeypatch.setattr(worker_module, "run_observation_processing", run)
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    message = _task_message()
    task.push_request(
        id="delivery_01",
        retries=6,
        headers={"mindbridge_retry_deadline": time() + 60},
        called_directly=False,
        is_eager=True,
        args=(message,),
        kwargs={},
    )
    try:
        with pytest.raises(Retry):
            task.run(message)
    finally:
        task.pop_request()


def test_worker_gives_a_transient_failure_back_to_the_ledger_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the deadline the original error is raised, which is what marks the row `failed`.

    A failed row is claimable and `mindbridge jobs --republish --include-failed` reaches it; a
    message still bouncing is reachable by nothing.
    """

    def run(*_arguments: object) -> JobState:
        raise DatabaseUnavailableError("database unavailable")

    monkeypatch.setattr(worker_module, "run_observation_processing", run)
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    message = _task_message()
    task.push_request(
        id="delivery_01",
        retries=1,
        headers={"mindbridge_retry_deadline": time() - 1},
        called_directly=False,
        is_eager=True,
        args=(message,),
        kwargs={},
    )
    try:
        with pytest.raises(DatabaseUnavailableError):
            task.run(message)
    finally:
        task.pop_request()


def test_worker_will_not_take_a_retry_deadline_a_message_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headers are caller-supplied, so a deadline beyond one budget is treated as absent."""
    monkeypatch.setattr(
        worker_module,
        "run_observation_processing",
        lambda *_arguments: JobState.SUCCEEDED,
    )
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    task = cast(Task, app.tasks[PROCESS_OBSERVATION_TASK])
    message = _task_message()
    task.push_request(
        id="delivery_01",
        retries=0,
        headers={"mindbridge_retry_deadline": time() + 86_400},
        called_directly=False,
        is_eager=True,
        args=(message,),
        kwargs={},
    )
    try:
        assert worker_module._retry_seconds_remaining(cast(Any, task)) <= (
            OBSERVATION_JOB_STALE_AFTER_SECONDS
        )
    finally:
        task.pop_request()


def test_a_worker_narrowed_to_the_pre_shard_queue_alone_refuses_to_start() -> None:
    """`-Q mindbridge` leaves every shard unread, and the symptom is silence, not an error.

    The publish succeeds, the ledger says `pending`, and nothing consumes it -- so boot is the
    only place this can be said out loud. `-Q` is readable here because
    `WorkController.setup_instance` calls `setup_queues` before it sends `worker_init`.
    """
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    app.amqp.queues.select(["mindbridge"])

    with pytest.raises(SystemExit, match="would have no consumer"):
        worker_init.send(sender=_StartingWorker(1))


def test_a_worker_that_reads_every_observation_queue_starts() -> None:
    """The default: no -Q at all, so the app's own queue set is the consume set."""
    app = create_worker_app(WorkerSettings.from_environment(_environment()))

    assert set(observation_queues()) <= set(app.amqp.queues.consume_from), (
        "a bare worker does not already cover the shards, so the guard below proves nothing"
    )
    worker_init.send(sender=_StartingWorker(1))


def test_a_worker_that_reads_none_of_them_is_a_different_role_and_starts() -> None:
    """The consolidation sweep runs on its own queue; partial coverage is the only mistake."""
    app = create_worker_app(WorkerSettings.from_environment(_environment()))
    app.amqp.queues.select([CONSOLIDATION_QUEUE])

    assert not set(observation_queues()) & set(app.amqp.queues.consume_from)
    worker_init.send(sender=_StartingWorker(1))


def test_the_guard_names_every_queue_that_would_go_unread() -> None:
    """An operator needs the list, not the count: the fix is naming them or dropping -Q."""
    whole = observation_queues()

    require_whole_observation_queue_set(whole)
    require_whole_observation_queue_set(["mindbridge_consolidation"])
    with pytest.raises(ValueError, match="would have no consumer") as raised:
        require_whole_observation_queue_set(whole[:-2])

    assert all(name in str(raised.value) for name in whole[-2:])
