"""Admission limits must apply across callers, before resource allocation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import BoundedSemaphore, Condition, Event

import pytest
from _feature_support import TinyEmbedder

from mindbridge import AnswerChunk, AnswerResult, AsyncMemory, Memory, StorageError


def test_operations_reject_overload_and_recover_after_release(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        with ExitStack() as operations:
            for _ in range(32):
                operations.enter_context(memory._lifecycle.operation())
            with pytest.raises(StorageError) as failure:
                memory.list()
            assert failure.value.reason == "capacity_exceeded"
            assert failure.value.retryable
            with pytest.raises(StorageError) as ownership:
                Memory(tmp_path, embedder=TinyEmbedder())
            assert ownership.value.reason == "data_dir_in_use"
            with Memory(tmp_path / "independent", embedder=TinyEmbedder()) as independent:
                assert independent.list().items == ()
        assert memory.list().items == ()


def test_sqlite_capacity_uses_the_public_retry_contract(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        with ExitStack() as borrowers:
            for _ in range(32):
                borrowers.enter_context(memory._store._connections.connection())
            with pytest.raises(StorageError) as failure:
                memory.list()
            assert failure.value.reason == "capacity_exceeded"
            assert failure.value.stage == "admission"
            assert failure.value.retryable
        assert memory.list().items == ()


@pytest.mark.parametrize(("plane", "limit"), (("index", 4), ("speech", 2)))
def test_child_workers_are_shared_across_requests(tmp_path: Path, plane: str, limit: int) -> None:
    release = Event()
    changed = Condition()
    active = peak = 0

    def route() -> int:
        nonlocal active, peak
        with changed:
            active += 1
            peak = max(peak, active)
            changed.notify_all()
        try:
            assert release.wait(5)
            return 7
        finally:
            with changed:
                active -= 1

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:

        def dispatch() -> tuple[int, ...]:
            if plane == "speech":
                with memory._execution.overlap_speech(route) as future:
                    value = future.result()
                    return value, value
            return memory._execution.index_routes((route, route))

        with ThreadPoolExecutor(max_workers=8) as callers:
            futures = [callers.submit(dispatch) for _ in range(8)]
            try:
                with changed:
                    assert changed.wait_for(lambda: active >= limit, timeout=2)
                    assert not changed.wait_for(lambda: peak > limit, timeout=0.1)
            finally:
                release.set()
            assert [future.result(timeout=5) for future in futures] == [(7, 7)] * 8
        assert peak == limit


@pytest.mark.parametrize(("plane", "limit"), (("index", 4), ("speech", 2)))
def test_waiting_child_routes_do_not_occupy_another_directorys_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plane: str, limit: int
) -> None:
    release, saturated = Event(), Event()
    changed = Condition()
    attempts = 0

    class ObservedSlots(BoundedSemaphore):
        def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
            nonlocal attempts
            with changed:
                attempts += 1
                if attempts == limit + 2:
                    saturated.set()
            return super().acquire(blocking, timeout)

        def __enter__(self, blocking: bool = True, timeout: float | None = None, /) -> bool:
            return self.acquire(blocking, timeout)

    def held() -> int:
        assert release.wait(5)
        return 7

    with (
        Memory(tmp_path / "busy", embedder=TinyEmbedder()) as busy,
        Memory(tmp_path / "independent", embedder=TinyEmbedder()) as independent,
        ThreadPoolExecutor(max_workers=limit + 2) as children,
        ThreadPoolExecutor(max_workers=9) as callers,
    ):
        monkeypatch.setattr(busy._execution, f"_{plane}", children)
        monkeypatch.setattr(independent._execution, f"_{plane}", children)
        monkeypatch.setattr(busy._execution, f"_{plane}_slots", ObservedSlots(limit))

        def dispatch(memory: Memory, *, block: bool) -> tuple[int, ...]:
            call = held if block else lambda: 11
            if plane == "index":
                return memory._execution.index_routes((call,))
            with memory._execution.overlap_speech(call) as future:
                return (future.result(),)

        pending = [callers.submit(dispatch, busy, block=True) for _ in range(8)]
        try:
            assert saturated.wait(2)
            fresh = callers.submit(dispatch, independent, block=False)
            assert fresh.result(timeout=1) == (11,)
        finally:
            release.set()
        assert [future.result(timeout=3) for future in pending] == [(7,)] * 8


def test_partial_route_submission_failure_waits_for_started_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, failed, release = Event(), Event(), Event()
    submitted = 0

    def route() -> int:
        entered.set()
        assert release.wait(5)
        return 7

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory, ThreadPoolExecutor() as caller:
        original = memory._execution._index.submit

        def submit(fn: Callable[[Callable[[], int]], int], call: Callable[[], int]) -> Future[int]:
            nonlocal submitted
            submitted += 1
            if submitted == 2:
                failed.set()
                raise RuntimeError("injected submission failure")
            return original(fn, call)

        with monkeypatch.context() as patch:
            patch.setattr(memory._execution._index, "submit", submit)
            pending = caller.submit(memory._execution.index_routes, (route, route))
            try:
                assert entered.wait(2) and failed.wait(2)
                assert not pending.done()
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="injected submission failure"):
                pending.result(timeout=3)
        assert memory._execution.index_routes((lambda: 11,) * 4) == (11,) * 4


@pytest.mark.asyncio
async def test_stream_capacity_is_shared_and_reusable(tmp_path: Path) -> None:
    def chunks() -> Generator[AnswerChunk, None, AnswerResult]:
        yield AnswerChunk(text="answer")
        return AnswerResult(answer="answer")

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        streams = [memory._execution.pump(chunks()) for _ in range(8)]
        try:
            for stream in streams:
                assert (await anext(stream)).text == "answer"
            excess = memory._execution.pump(chunks())
            try:
                with pytest.raises(StorageError) as failure:
                    await anext(excess)
                assert failure.value.reason == "capacity_exceeded"
            finally:
                await excess.aclose()
        finally:
            for stream in streams:
                await stream.aclose()
        recovered = memory._execution.pump(chunks())
        assert (await anext(recovered)).text == "answer"
        await recovered.aclose()


@pytest.mark.asyncio
async def test_cancelled_async_work_keeps_admission_until_it_stops(tmp_path: Path) -> None:
    release = Event()
    changed = Condition()
    entered = 0

    def blocked() -> int:
        nonlocal entered
        with changed:
            entered += 1
            changed.notify_all()
        assert release.wait(5)
        return 9

    def all_entered() -> bool:
        with changed:
            return changed.wait_for(lambda: entered == 32, timeout=3)

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        tasks = [asyncio.create_task(memory._execution.run(blocked)) for _ in range(32)]
        try:
            assert await asyncio.to_thread(all_entered)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            with pytest.raises(StorageError) as failure:
                await memory._execution.run(lambda: 10)
            assert failure.value.reason == "capacity_exceeded"
            with Memory(tmp_path / "independent", embedder=TinyEmbedder()) as independent:
                assert await AsyncMemory(independent).list() == independent.list()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert await memory._execution.run(lambda: 10) == 10


@pytest.mark.asyncio
async def test_async_closed_memory_keeps_public_error_contract(tmp_path: Path) -> None:
    memory = AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder()))
    await memory.close()
    with pytest.raises(StorageError) as failure:
        await memory.list()
    assert failure.value.reason == "instance_unusable"


@pytest.mark.asyncio
async def test_stream_cleanup_failure_does_not_leak_capacity(tmp_path: Path) -> None:
    def broken_cleanup() -> Generator[AnswerChunk, None, AnswerResult]:
        try:
            yield AnswerChunk(text="answer")
            return AnswerResult(answer="answer")
        finally:
            raise ValueError("cleanup failed")

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        for _ in range(9):
            stream = memory._execution.pump(broken_cleanup())
            assert (await anext(stream)).text == "answer"
            with pytest.raises(ValueError, match="cleanup failed"):
                await stream.aclose()


@pytest.mark.asyncio
async def test_async_checks_process_ownership_before_touching_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:

        def forbidden_submit(*_args: object, **_kwargs: object) -> None:
            pytest.fail("a forked instance touched its inherited executor")

        with monkeypatch.context() as patch:
            patch.setattr(memory._lifecycle, "_owner_pid", os.getpid() + 1)
            patch.setattr(memory._execution._async, "submit", forbidden_submit)
            with pytest.raises(StorageError) as failure:
                await AsyncMemory(memory).list()
            assert failure.value.reason == "instance_unusable"


@pytest.mark.asyncio
async def test_terminal_chunk_releases_worker_before_the_caller_stops(tmp_path: Path) -> None:
    closed: list[bool] = []

    def chunks() -> Generator[AnswerChunk, None, AnswerResult]:
        result = AnswerResult(answer="answer")
        try:
            yield AnswerChunk(result=result)
            return result
        finally:
            closed.append(True)

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        streams = [memory._execution.pump(chunks()) for _ in range(9)]
        try:
            for index, stream in enumerate(streams):
                assert (await anext(stream)).result is not None
                assert len(closed) == index + 1
        finally:
            for stream in streams:
                await stream.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_terminal_cleanup_remains_cancellation(tmp_path: Path) -> None:
    closing, release = Event(), Event()

    def chunks() -> Generator[AnswerChunk, None, AnswerResult]:
        result = AnswerResult(answer="answer")
        try:
            yield AnswerChunk(result=result)
            return result
        finally:
            closing.set()
            assert release.wait(5)

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        stream = memory._execution.pump(chunks())
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(closing.wait, 2)
            pending.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
