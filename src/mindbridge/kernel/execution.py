"""Instance-owned workers and admission, independent of any transport or event loop."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Callable, Generator, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from threading import BoundedSemaphore, Condition, Lock
from time import perf_counter
from typing import ParamSpec, TypeVar, cast

from mindbridge.exceptions import StorageError
from mindbridge.infrastructure.local._capacity import Capacity
from mindbridge.kernel.lifecycle import Lifecycle
from mindbridge.types import AnswerChunk, AnswerResult

_P = ParamSpec("_P")
_T = TypeVar("_T")
_PROCESS_SUBMISSIONS = Capacity(128)
_PROCESS_REQUESTS = Capacity(128)
_STREAM_EXHAUSTED = object()
ASYNC_QUEUE_TIME_MS: ContextVar[float | None] = ContextVar(
    "mindbridge_async_queue_time_ms", default=None
)


class _Workers:
    def __init__(self) -> None:
        self.parent = ThreadPoolExecutor(max_workers=64, thread_name_prefix="mindbridge-async")
        self.index = ThreadPoolExecutor(max_workers=16, thread_name_prefix="mindbridge-index")
        self.speech = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mindbridge-speech")
        self.lock = Lock()
        self.streams: list[ThreadPoolExecutor] = []
        self.idle: list[ThreadPoolExecutor] = []

    def borrow_stream(self) -> ThreadPoolExecutor:
        with self.lock:
            if self.idle:
                return self.idle.pop()
            if len(self.streams) >= 32:
                raise _overloaded()
            worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindbridge-stream")
            self.streams.append(worker)
            return worker

    def release_stream(self, worker: ThreadPoolExecutor) -> None:
        with self.lock:
            self.idle.append(worker)


_WORKER_PID = os.getpid()
_WORKER_LOCK = Lock()
_WORKERS: _Workers | None = None


def _process_workers() -> _Workers:
    global _WORKERS, _WORKER_PID, _WORKER_LOCK
    if os.getpid() != _WORKER_PID:
        _WORKERS = None
        _WORKER_LOCK = Lock()
        _WORKER_PID = os.getpid()
    with _WORKER_LOCK:
        if _WORKERS is None:
            _WORKERS = _Workers()
        return _WORKERS


def _overloaded() -> StorageError:
    return StorageError(
        "local execution capacity exceeded", reason="capacity_exceeded", stage="admission"
    )


class StreamWorker(Executor):
    """One exclusive lease on a reusable single-thread executor."""

    def __init__(self, pool: StreamWorkers, worker: ThreadPoolExecutor) -> None:
        self._pool = pool
        self._worker = worker
        self._release_future: Future[None] | None = None

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        if self._release_future is not None:
            raise RuntimeError("stream worker is released")
        return self._worker.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del cancel_futures  # Cleanup queued behind an uncancellable next() must always run.
        released = self.release()
        if wait:
            released.result()

    def release(self) -> Future[None]:
        # The marker follows next() and close(). A cancelled asyncio waiter cannot return the
        # worker early or let the next borrower enter the stream's Context concurrently.
        if self._release_future is None:
            self._release_future = self._worker.submit(self._pool.release, self._worker)
        return self._release_future


class StreamWorkers:
    """At most eight pinned workers, shared by all wrappers around one Memory."""

    def __init__(self, *, require_open: Callable[[], None] | None = None) -> None:
        self._require_open = require_open
        self._lock = Lock()
        self._borrowed = 0
        self._closed = False

    def borrow(self) -> StreamWorker:
        if self._require_open is not None:
            self._require_open()
        with self._lock:
            if self._closed:
                raise StorageError("Memory is closed", reason="instance_unusable")
            if self._borrowed >= 8:
                raise _overloaded()
            worker = _process_workers().borrow_stream()
            self._borrowed += 1
            return StreamWorker(self, worker)

    def release(self, worker: ThreadPoolExecutor) -> None:
        with self._lock:
            self._borrowed -= 1
        _process_workers().release_stream(worker)

    def close(self) -> None:
        with self._lock:
            self._closed = True


class Execution:
    """Separate parent, index, speech, and streaming pools to avoid nested-pool deadlocks."""

    def __init__(self, *, lifecycle: Lifecycle) -> None:
        self._lifecycle = lifecycle
        self.streams = StreamWorkers(require_open=lifecycle.require_open)
        workers = _process_workers()
        self._index, self._speech, self._async = workers.index, workers.speech, workers.parent
        self._index_slots, self._speech_slots = BoundedSemaphore(4), BoundedSemaphore(2)
        self._async_slots = Capacity(32, parent=_PROCESS_SUBMISSIONS)
        self.requests = Capacity(32, parent=_PROCESS_REQUESTS)
        self._pending = 0
        self._lock = Condition()
        self._closed = False

    async def run(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        queued_at = perf_counter()

        def invoke() -> _T:
            token = ASYNC_QUEUE_TIME_MS.set((perf_counter() - queued_at) * 1_000.0)
            try:
                return fn(*args, **kwargs)
            finally:
                ASYNC_QUEUE_TIME_MS.reset(token)

        # In a forked child, an inherited executor may have idle tokens but no live workers.
        # Check ownership before even acquiring one of its inherited locks.
        self._lifecycle.require_open()
        with self._lock:
            if self._closed:
                raise StorageError("Memory is closed", reason="instance_unusable")
            if not self._async_slots.acquire():
                raise _overloaded()
            try:
                future = self._async.submit(copy_context().run, invoke)
                self._pending += 1
            except BaseException:
                self._async_slots.release()
                raise
        future.add_done_callback(self._completed)
        return await asyncio.wrap_future(future)

    async def pump(
        self,
        stream: Generator[AnswerChunk, None, AnswerResult],
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Advance an answer on one pinned worker and release it after terminal cleanup."""
        loop = asyncio.get_running_loop()
        context = copy_context()
        # One worker, not `asyncio.to_thread`: every step has to run on the same thread so the
        # span and operation contextvars the generator attaches on its first step are the ones
        # it detaches on its last. `to_thread` runs each call in a fresh context copy, which
        # would strand both. `next` is given a sentinel because a `StopIteration` crossing a
        # future becomes `RuntimeError` rather than ending the loop.
        #
        # Admission happens before a worker is created, and every wrapper and transport borrows
        # from this instance's same bounded pool.
        worker = self.streams.borrow()
        first_advance = True
        closed = False

        def advance(queued_at: float, record_queue: bool) -> object:
            token = None
            if record_queue:
                token = ASYNC_QUEUE_TIME_MS.set((perf_counter() - queued_at) * 1_000.0)
            try:
                return next(stream, _STREAM_EXHAUSTED)
            finally:
                if token is not None:
                    ASYNC_QUEUE_TIME_MS.reset(token)

        async def close_stream() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                closing = loop.run_in_executor(worker, context.run, stream.close)
                closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                await asyncio.shield(closing)
            finally:
                released = asyncio.wrap_future(worker.release())
                await asyncio.shield(released)

        try:
            while True:
                queued_at = perf_counter()
                record_queue = first_advance
                first_advance = False
                chunk = await loop.run_in_executor(
                    worker,
                    context.run,
                    advance,
                    queued_at,
                    record_queue,
                )
                if chunk is _STREAM_EXHAUSTED:
                    return
                answer_chunk = cast(AnswerChunk, chunk)
                if answer_chunk.result is not None:
                    # Reading the terminal result is a complete consumption: callers need not
                    # advance once more or explicitly close it to return worker capacity.
                    await close_stream()
                    yield answer_chunk
                    return
                yield answer_chunk
        finally:
            # A running executor call cannot be cancelled, so the close always completes even
            # when the awaiting task is gone; `wait=False` keeps that off the event loop.
            await close_stream()

    def _completed(self, future: Future[_T]) -> None:
        with self._lock:
            self._pending -= 1
            self._lock.notify_all()
        self._async_slots.release()

    @staticmethod
    def _submit(executor: Executor, slots: BoundedSemaphore, call: Callable[[], _T]) -> Future[_T]:
        # Wait in the parent: a directory waiting for its quota must not occupy another's worker.
        slots.acquire()
        try:
            future = executor.submit(copy_context().run, call)
        except BaseException:
            slots.release()
            raise
        future.add_done_callback(lambda _done: slots.release())
        return future

    def index_routes(self, calls: Sequence[Callable[[], _T]]) -> tuple[_T, ...]:
        results: list[_T] = []
        # At most four pending routes per admitted operation, including temporal supplements.
        for offset in range(0, len(calls), 4):
            futures: list[Future[_T]] = []
            try:
                for call in calls[offset : offset + 4]:
                    futures.append(self._submit(self._index, self._index_slots, call))
            finally:
                # Even a later submission failure cannot release the caller's assets early.
                wait(futures)
            results.extend(future.result() for future in futures)
        return tuple(results)

    @contextmanager
    def overlap_speech(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Iterator[Future[_T]]:
        future = self._submit(self._speech, self._speech_slots, partial(fn, *args, **kwargs))
        try:
            yield future
        finally:
            # Search failures must still wait for the recognizer before releasing its assets.
            wait((future,))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            while self._pending:
                self._lock.wait()
        self.streams.close()
