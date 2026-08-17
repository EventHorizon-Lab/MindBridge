"""Checks for the resumable observation job watch use case."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.ports import MemoryStore
from mindbridge.contracts import ObservationProcessingJobView
from mindbridge.core import (
    JobId,
    JobState,
    MemoryId,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class _ScriptedStore:
    """Returns one prepared job per read so state sequences stay deterministic."""

    def __init__(self, *jobs: ObservationProcessingJob) -> None:
        self.jobs = list(jobs)
        self.reads = 0

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        if self.reads >= len(self.jobs):
            raise AssertionError("watch read the store more times than the script allows")
        job = self.jobs[self.reads]
        self.reads += 1
        return job


class _AdvancingClock:
    """Moves forward one step per call so a duration budget expires without real waiting."""

    def __init__(self, step_seconds: float = 1.0) -> None:
        self.calls = 0
        self._step_seconds = step_seconds

    def __call__(self) -> datetime:
        moment = NOW + timedelta(seconds=self._step_seconds * self.calls)
        self.calls += 1
        return moment


async def test_watch_emits_each_change_and_stops_once_the_attempt_settles() -> None:
    store = _ScriptedStore(
        _job(JobState.PENDING, attempt=0, updated_at=NOW),
        _job(JobState.PENDING, attempt=0, updated_at=NOW),
        _job(JobState.RUNNING, attempt=1, updated_at=NOW + timedelta(seconds=2)),
        _job(JobState.RUNNING, attempt=1, updated_at=NOW + timedelta(seconds=2)),
        _job(
            JobState.SUCCEEDED,
            attempt=1,
            updated_at=NOW + timedelta(seconds=4),
            memory_ids=("memory_01",),
        ),
    )

    views = await _watch(store)

    assert [view.state for view in views] == [
        JobState.PENDING,
        JobState.RUNNING,
        JobState.SUCCEEDED,
    ]
    assert views[-1].memory_ids == ("memory_01",)
    assert store.reads == 5


async def test_watch_suppresses_a_state_the_caller_already_received() -> None:
    store = _ScriptedStore(
        _job(JobState.RUNNING, attempt=1, updated_at=NOW),
        _job(
            JobState.SUCCEEDED,
            attempt=1,
            updated_at=NOW + timedelta(seconds=1),
            memory_ids=("memory_01",),
        ),
    )

    views = await _watch(store, after_updated_at=NOW)

    assert [view.state for view in views] == [JobState.SUCCEEDED]


async def test_watch_stops_at_the_duration_budget_when_a_job_never_settles() -> None:
    store = _ScriptedStore(*(_job(JobState.RUNNING, attempt=1, updated_at=NOW) for _ in range(3)))
    clock = _AdvancingClock()

    views = await _watch(store, clock=clock, maximum_duration_seconds=3.0)

    assert [view.state for view in views] == [JobState.RUNNING]
    assert store.reads == 3


async def test_watch_closes_the_stream_on_a_failed_attempt() -> None:
    store = _ScriptedStore(
        _job(JobState.FAILED, attempt=1, updated_at=NOW, error_code="model_unavailable")
    )

    views = await _watch(store)

    assert [view.state for view in views] == [JobState.FAILED]
    assert views[0].error_code == "model_unavailable"


@pytest.mark.parametrize(
    ("poll_interval_seconds", "maximum_duration_seconds", "message"),
    [
        (-1.0, 300.0, "poll_interval_seconds"),
        (0.0, 0.0, "maximum_duration_seconds"),
    ],
)
async def test_watch_rejects_an_unusable_schedule(
    poll_interval_seconds: float,
    maximum_duration_seconds: float,
    message: str,
) -> None:
    store = _ScriptedStore(_job(JobState.SUCCEEDED, attempt=1, updated_at=NOW))

    with pytest.raises(ValueError, match=message):
        await _watch(
            store,
            poll_interval_seconds=poll_interval_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
        )

    assert store.reads == 0


async def _watch(
    store: _ScriptedStore,
    *,
    after_updated_at: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
    poll_interval_seconds: float = 0.0,
    maximum_duration_seconds: float = 300.0,
) -> list[ObservationProcessingJobView]:
    placeholder = cast(Any, object())
    kernel = MemoryKernel(
        cast(MemoryStore, store),
        placeholder,
        placeholder,
        embedding_index=placeholder,
        media_deleter=placeholder,
        media_url_signer=placeholder,
        observation_job_publisher=placeholder,
        embedder=placeholder,
        clock=clock or (lambda: NOW),
    )
    return [
        view
        async for view in kernel.watch_observation_job(
            "tenant_01",
            "job_01",
            after_updated_at=after_updated_at,
            poll_interval_seconds=poll_interval_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
        )
    ]


def _job(
    state: JobState,
    *,
    attempt: int,
    updated_at: datetime,
    error_code: str | None = None,
    memory_ids: tuple[str, ...] = (),
) -> ObservationProcessingJob:
    return ObservationProcessingJob(
        job_id=JobId("job_01"),
        tenant_id=TenantId("tenant_01"),
        observation_id=ObservationId("observation_01"),
        state=state,
        attempt=attempt,
        error_code=error_code,
        created_at=NOW,
        updated_at=updated_at,
        memory_ids=tuple(MemoryId(value) for value in memory_ids),
    )
