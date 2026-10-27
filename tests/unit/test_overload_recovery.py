"""Overload must fail before allocation, preserve ownership, and recover after release."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread

import anyio.to_thread
import pytest
from _feature_support import TinyEmbedder
from mcp import Client
from starlette.types import Message, Scope

import mindbridge.api.app as app_module
import mindbridge.infrastructure.local.store._connections as connections_module
import mindbridge.infrastructure.local.store._outbox as outbox_module
import mindbridge.kernel.lifecycle as lifecycle_module
from mindbridge import AsyncMemory, Memory, StorageError, ValidationError
from mindbridge.api.app import create_app
from mindbridge.api.mcp import build_mcp_server
from mindbridge.infrastructure.local._capacity import Capacity
from mindbridge.models.base import EmbedTask, ModelInput


def test_model_budget_keeps_lightweight_reads_available_across_directories(tmp_path: Path) -> None:
    with (
        Memory(tmp_path / "one", embedder=TinyEmbedder()) as first,
        Memory(tmp_path / "two", embedder=TinyEmbedder()) as second,
    ):
        with ExitStack() as leases:
            for _ in range(28):
                leases.enter_context(first._lifecycle.operation(model_work=True))
            with pytest.raises(StorageError) as failure:
                second.search("overloaded query")
            assert failure.value.reason == "capacity_exceeded"
            assert first.list().items == second.list().items == ()
        assert second.search("recovered") == ()


def test_instance_budget_releases_failed_and_closed_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connections_module, "_PROCESS_INSTANCES", Capacity(2))
    with Memory(tmp_path / "one", embedder=TinyEmbedder()):
        second = Memory(tmp_path / "two", embedder=TinyEmbedder())
        try:
            with pytest.raises(StorageError) as failure:
                Memory(tmp_path / "three", embedder=TinyEmbedder())
            assert failure.value.reason == "capacity_exceeded"
            assert failure.value.stage == "open"
        finally:
            second.close()
        with Memory(tmp_path / "three", embedder=TinyEmbedder()) as recovered:
            assert recovered.list().items == ()


def test_pending_capture_limit_is_atomic_and_duplicates_remain_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(outbox_module, "MAX_PENDING_CAPTURES", 1)
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        first = memory.capture("first")
        with pytest.raises(StorageError) as failure:
            memory.capture("second")
        assert failure.value.reason == "capacity_exceeded"
        assert [record.id for record in memory.list().items] == [first.id]
        assert memory.capture("first").id == first.id
        assert memory.settle() == 1
        assert memory.capture("second").id != first.id


def test_payload_budget_is_shared_and_released_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lifecycle_module, "_PROCESS_PAYLOAD", Capacity(16))
    with (
        Memory(tmp_path / "one", embedder=TinyEmbedder()) as first,
        Memory(tmp_path / "two", embedder=TinyEmbedder()) as second,
    ):
        with first._lifecycle.operation() as operation:
            first._lifecycle.reserve_payload(operation, 8)
            with pytest.raises(ValidationError, match="16 bytes"):
                first._lifecycle.reserve_payload(operation, 9)
            assert operation.payload_bytes == 8
            with pytest.raises(StorageError) as failure:
                second.add("ninebytes")
            assert failure.value.reason == "capacity_exceeded"
            assert second.list().items == ()
        with pytest.raises(ValidationError, match="16 bytes"):
            second.add("an input exceeding sixteen bytes")
        assert second.add("ninebytes").content == "ninebytes"


def test_oversized_batch_fails_before_embedding(tmp_path: Path) -> None:
    class UnusedEmbedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            pytest.fail("oversized batch called a model")

    with Memory(tmp_path, embedder=UnusedEmbedder()) as memory:
        with pytest.raises(ValidationError, match="1024"):
            memory.add_many([str(index) for index in range(1025)])
        assert memory.list().items == ()


@pytest.mark.parametrize("stage", ("operation", "backend"))
def test_close_timeout_keeps_directory_owned_until_cleanup_finishes(
    tmp_path: Path, stage: str
) -> None:
    entered, release = Event(), Event()

    class ClosingEmbedder(TinyEmbedder):
        def close(self) -> None:
            if stage == "backend":
                entered.set()
                assert release.wait(5)

    memory = Memory(tmp_path, embedder=ClosingEmbedder())
    leases = ExitStack()
    if stage == "operation":
        leases.enter_context(memory._lifecycle.operation())
    try:
        with pytest.raises(StorageError) as failure:
            memory.close(timeout_seconds=0.02)
        assert failure.value.reason == "timeout"
        assert failure.value.stage == "close"
        with pytest.raises(StorageError) as ownership:
            Memory(tmp_path, embedder=TinyEmbedder())
        assert ownership.value.reason == "data_dir_in_use"
    finally:
        release.set()
        leases.close()
        memory.close(timeout_seconds=3)
    with Memory(tmp_path, embedder=TinyEmbedder()) as recovered:
        assert recovered.list().items == ()


@pytest.mark.asyncio
async def test_closing_one_instance_does_not_stop_another_instances_workers(tmp_path: Path) -> None:
    first = Memory(tmp_path / "one", embedder=TinyEmbedder())
    with Memory(tmp_path / "two", embedder=TinyEmbedder()) as second:
        assert (await AsyncMemory(first).list()).items == ()
        first.close()
        assert (await AsyncMemory(second).add("still running")).content == "still running"


@pytest.mark.asyncio
async def test_mcp_rejects_overload_before_framework_worker_admission(tmp_path: Path) -> None:
    entered, release = Event(), Event()
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()

    def occupy() -> None:
        entered.set()
        loop.call_soon_threadsafe(ready.set)
        assert release.wait(5)

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        async with Client(build_mcp_server(memory)) as client:
            limiter = anyio.to_thread.current_default_thread_limiter()
            original = limiter.total_tokens
            limiter.total_tokens = 1
            blocker = asyncio.create_task(anyio.to_thread.run_sync(occupy))
            acquired = 0
            try:
                await asyncio.wait_for(ready.wait(), timeout=1)
                for _ in range(32):
                    assert memory._execution.requests.acquire()
                    acquired += 1
                result = await asyncio.wait_for(client.call_tool("list_memories", {}), timeout=0.5)
                assert result.is_error
                content = result.content[0]
                assert content.type == "text"
                envelope = json.loads(content.text)
                assert envelope["reason"] == "capacity_exceeded"
                assert envelope["stage"] == "admission"
                assert envelope["retryable"]
            finally:
                for _ in range(acquired):
                    memory._execution.requests.release()
                release.set()
                await blocker
                limiter.total_tokens = original


@pytest.mark.asyncio
async def test_slow_request_body_times_out_and_returns_request_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_REQUEST_BODY_TIMEOUT_SECONDS", 0.01)
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "GET",
        "path": "/v1/memories",
        "query_string": b"",
        "headers": [],
    }

    async def stalled() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        app = create_app(memory=memory)
        await asyncio.wait_for(app(scope, stalled, send), timeout=0.5)
        assert messages[0]["status"] == 503
        assert json.loads(messages[1]["body"])["reason"] == "timeout"
        assert memory._execution.requests.acquire(32)
        memory._execution.requests.release(32)


@pytest.mark.asyncio
async def test_stalled_response_send_releases_request_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_RESPONSE_SEND_TIMEOUT_SECONDS", 0.01)
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "GET",
        "path": "/v1/memories",
        "query_string": b"",
        "headers": [],
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    starts: list[int] = []

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            starts.append(message["status"])
        await asyncio.Event().wait()

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        await asyncio.wait_for(create_app(memory=memory)(scope, receive, send), timeout=0.5)
        assert starts == [200]
        assert memory._execution.requests.acquire(32)
        memory._execution.requests.release(32)


def test_close_reports_cleanup_failure_after_an_earlier_timeout(tmp_path: Path) -> None:
    release = Event()

    class FailingClose(TinyEmbedder):
        def close(self) -> None:
            assert release.wait(5)
            raise StorageError("injected cleanup failure", reason="io_failed")

    memory = Memory(tmp_path, embedder=FailingClose())
    try:
        with pytest.raises(StorageError) as timeout:
            memory.close(timeout_seconds=0.01)
        assert timeout.value.reason == "timeout"
    finally:
        release.set()
    with pytest.raises(StorageError) as failure:
        memory.close(timeout_seconds=2)
    assert failure.value.reason == "io_failed"
    memory.close()


def test_close_worker_start_failure_leaves_the_instance_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:

        def fail_start(_thread: object) -> None:
            raise RuntimeError("cannot start new thread")

        with monkeypatch.context() as patch:
            patch.setattr(Thread, "start", fail_start)
            with pytest.raises(StorageError) as failure:
                memory.close()
        assert failure.value.reason == "capacity_exceeded"
        assert failure.value.stage == "close"
        assert memory.list().items == ()


@pytest.mark.asyncio
async def test_async_close_does_not_queue_in_the_default_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder()))

    async def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("async close submitted to the default executor")

    monkeypatch.setattr(asyncio, "to_thread", forbidden)
    await memory.close()


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX descriptor limits and procfs",
)
def test_sqlite_refuses_before_real_process_descriptor_exhaustion() -> None:
    probe = """
import os, resource, tempfile
from contextlib import ExitStack
from pathlib import Path
from mindbridge import Memory, Modality, StorageError
from mindbridge.infrastructure.local.store.errors import LocalStoreCapacityError
class Embedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "descriptor-fixture-v1"
    embedding_space = "descriptor-fixture-v1:2"
    embedding_dimension = 2
    def embed(self, inputs, task=None): return tuple((1.0, 0.0) for _ in inputs)
    def close(self): pass
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(512, soft), hard))
with tempfile.TemporaryDirectory() as directory, ExitStack() as owners, ExitStack() as borrowers:
    memories = []
    for index in range(16):
        try:
            memories.append(owners.enter_context(Memory(Path(directory)/str(index), embedder=Embedder())))
        except StorageError as error:
            assert error.reason == "capacity_exceeded", error
            break
    assert memories
    admitted = 0
    for index in range(128):
        try:
            borrowers.enter_context(memories[index % len(memories)]._store._connections.connection())
            admitted += 1
        except LocalStoreCapacityError:
            break
    assert 0 < admitted < 128, admitted
    descriptor = os.open(os.devnull, os.O_RDONLY)
    os.close(descriptor)
    assert len(os.listdir("/proc/self/fd")) < min(512, soft)
    borrowers.close()
    assert all(memory.list().items == () for memory in memories)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
