"""Instance-owned workers and admission, independent of any transport or event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
from functools import partial
from threading import BoundedSemaphore, Lock
from typing import ParamSpec, TypeVar

from mindbridge.exceptions import StorageError
from mindbridge.kernel.lifecycle import Lifecycle

_P = ParamSpec("_P")
_T = TypeVar("_T")


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
        self._workers: set[ThreadPoolExecutor] = set()
        self._idle: list[ThreadPoolExecutor] = []
        self._closed = False

    def borrow(self) -> StreamWorker:
        if self._require_open is not None:
            self._require_open()
        with self._lock:
            if self._closed:
                raise StorageError("Memory is closed", reason="instance_unusable")
            if self._idle:
                worker = self._idle.pop()
            elif len(self._workers) < 8:
                worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindbridge-stream")
                self._workers.add(worker)
            else:
                raise _overloaded()
            return StreamWorker(self, worker)

    def release(self, worker: ThreadPoolExecutor) -> None:
        with self._lock:
            if not self._closed:
                self._idle.append(worker)
                return
        worker.shutdown(wait=False)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle = tuple(self._idle)
            self._idle.clear()
        # Borrowed streams retain their worker until their queued close finishes. In particular,
        # a caller may still own the terminal chunk after the Memory operation has ended.
        for worker in idle:
            worker.shutdown(wait=True)


class Execution:
    """Separate parent, index, speech, and streaming pools to avoid nested-pool deadlocks."""

    def __init__(self, *, lifecycle: Lifecycle) -> None:
        self._lifecycle = lifecycle
        self.streams = StreamWorkers(require_open=lifecycle.require_open)
        self._index = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mindbridge-index")
        self._speech = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mindbridge-speech")
        self._async = ThreadPoolExecutor(max_workers=32, thread_name_prefix="mindbridge-async")
        self._async_slots = BoundedSemaphore(32)
        self._lock = Lock()
        self._closed = False

    async def run(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # In a forked child, an inherited executor may have idle tokens but no live workers.
        # Check ownership before even acquiring one of its inherited locks.
        self._lifecycle.require_open()
        with self._lock:
            if self._closed:
                raise StorageError("Memory is closed", reason="instance_unusable")
            if not self._async_slots.acquire(blocking=False):
                raise _overloaded()
            try:
                future = self._async.submit(copy_context().run, partial(fn, *args, **kwargs))
            except BaseException:
                self._async_slots.release()
                raise
        future.add_done_callback(lambda _done: self._async_slots.release())
        return await asyncio.wrap_future(future)

    def index_routes(self, calls: Sequence[Callable[[], _T]]) -> tuple[_T, ...]:
        results: list[_T] = []
        # At most four pending routes per admitted operation, including temporal supplements.
        for offset in range(0, len(calls), 4):
            futures = tuple(
                self._index.submit(copy_context().run, call) for call in calls[offset : offset + 4]
            )
            wait(futures)
            results.extend(future.result() for future in futures)
        return tuple(results)

    @contextmanager
    def overlap_speech(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Iterator[Future[_T]]:
        future = self._speech.submit(copy_context().run, partial(fn, *args, **kwargs))
        try:
            yield future
        finally:
            # Search failures must still wait for the recognizer before releasing its assets.
            wait((future,))

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._async.shutdown(wait=True)
        self._index.shutdown(wait=True)
        self._speech.shutdown(wait=True)
        self.streams.close()
