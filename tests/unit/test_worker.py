"""Checks for the deployable observation Worker boundary."""

import asyncio
from collections.abc import Awaitable, Mapping
from typing import cast
from unittest.mock import Mock

import pytest
from celery import Task
from celery.exceptions import Retry, SoftTimeLimitExceeded
from pydantic import ValidationError

import mindbridge.worker as worker_module
from mindbridge.application.capabilities import Embedder, EmbedRequest, EmbedResult
from mindbridge.application.evidence_clips import ClipSampling
from mindbridge.core import (
    DatabaseUnavailableError,
    JobState,
    ModelUnavailableError,
    ObjectStorageError,
)
from mindbridge.infrastructure.task_queue import (
    PROCESS_OBSERVATION_TASK,
    ObservationProcessingTaskMessage,
)
from mindbridge.models.openai import _GeneratorConfig
from mindbridge.worker import WorkerSettings, create_worker_app, processing_budget_seconds


def test_worker_settings_pin_models_and_redact_credentials() -> None:
    """Worker startup fixes model identity without exposing injected secrets."""
    settings = WorkerSettings.from_environment(_environment())

    assert settings.generator_config["model_id"] == "qwen3.8-max"
    assert settings.media_embedder_config["model_id"] == (
        "jinaai/jina-embeddings-v5-omni-small-retrieval"
    )
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


def test_worker_settings_require_explicit_generator_credentials() -> None:
    """A Worker must fail startup rather than call a provider with no key."""
    environment = dict(_environment())
    del environment["MINDBRIDGE_GENERATOR_API_KEY"]

    with pytest.raises(ValueError, match="MINDBRIDGE_GENERATOR_API_KEY"):
        WorkerSettings.from_environment(environment)


def test_worker_reuses_and_closes_media_plugin_on_its_own_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_loops: list[asyncio.AbstractEventLoop] = []
    processing_loops: list[asyncio.AbstractEventLoop] = []
    closing_loops: list[asyncio.AbstractEventLoop] = []

    class LoopBoundEmbedder:
        async def embed(self, _request: EmbedRequest) -> EmbedResult:
            raise AssertionError("the lifecycle check does not invoke embedding")

        async def close(self) -> None:
            closing_loops.append(asyncio.get_running_loop())

    embedder = LoopBoundEmbedder()

    def load(_plugin: str, _config: Mapping[str, object]) -> LoopBoundEmbedder:
        creation_loops.append(asyncio.get_running_loop())
        return embedder

    async def process(
        _settings: object,
        media_embedder: object,
        *_identifiers: object,
    ) -> JobState:
        assert media_embedder is embedder
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

    assert len(creation_loops) == 1
    assert processing_loops == creation_loops * 2
    assert closing_loops == creation_loops


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
    runtime = worker_module._WorkerRuntime(loop, cast(Embedder, object()))
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
    assert task.max_retries == 5


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

    retry.assert_called_once_with(
        countdown=30,
        max_retries=40,
        headers={"mindbridge_running_retries": 1},
    )


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
                '{"frames_per_second": 0.5, "max_pixels": 50176, "generation_proxy": false}'
            ),
        }
    )

    assert settings.clip_sampling.frames_per_second == 0.5
    assert settings.clip_sampling.max_pixels == 50_176
    assert settings.clip_sampling.generation_proxy is False


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
    ],
)
def test_worker_rejects_a_malformed_media_sampling_knob(config: str) -> None:
    """A typo in an optional tuning knob must fail startup, not silently mean something else.
    `"false"` is a truthy string, and a quoted or floating pixel count would reach a budget
    comparison as the wrong type."""
    with pytest.raises(ValueError):
        WorkerSettings.from_environment(
            {**_environment(), "MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON": config}
        )


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
    config: dict[str, object] = {
        "api_key": "k",
        "endpoint": "https://example.test/v1",
    }
    client_deadline = _GeneratorConfig.model_validate(config).request_timeout_seconds

    assert processing_budget_seconds(config) > client_deadline


@pytest.mark.parametrize(
    "configured", [{"request_timeout_seconds": 0}, {"request_timeout_seconds": "soon"}]
)
def test_worker_rejects_an_unusable_generator_deadline(configured: dict[str, object]) -> None:
    """A budget derived from nonsense would be worse than the constant it replaced."""
    with pytest.raises(ValueError, match="request_timeout_seconds must be"):
        processing_budget_seconds(configured)
