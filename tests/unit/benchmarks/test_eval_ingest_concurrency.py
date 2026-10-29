"""Bounded SDK ingestion, durable prefix receipts, and shared-caption singleflight."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import cast

import pytest

from mindbridge import (
    AssetRef,
    AsyncMemory,
    EmbedTask,
    IndexQuantization,
    Memory,
    Modality,
    ModelInput,
)
from mindbridge.benchmarks import eval as runner
from mindbridge.benchmarks import eval_config
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_cache import DescriptionCache
from mindbridge.benchmarks.eval_results import FailureDetail
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import HarnessOverrides, ModelConfig
from mindbridge.benchmarks.task_catalog import TaskSpec

DAY = datetime(2024, 1, 1, tzinfo=timezone.utc)


class _BlockingEmbedder:
    embedding_model = "ingest-concurrency-fixture"
    embedding_space = "ingest-concurrency-fixture:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.lock = Lock()
        self.entered = Event()
        self.release = Event()
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.closed_while_active = False

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        del task
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            if self.active == 2:
                self.entered.set()
        try:
            assert self.release.wait(5), "test did not release the SDK embedding calls"
            return tuple((1.0, 0.0) for _value in inputs)
        finally:
            with self.lock:
                self.active -= 1

    def close(self) -> None:
        self.closed_while_active = self.active != 0


def _task(tmp_path: Path, *, cutoff: float | None = None) -> LoadedTask:
    spec = TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40)
    return LoadedTask(
        spec,
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                tuple(
                    MemoryItem(
                        str(index), (f"item {index}",), occurred_at=DAY + timedelta(seconds=index)
                    )
                    for index in range(8)
                ),
                (EvalQuestion("q", ("question",), references=("answer",), cutoff_seconds=cutoff),),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_real_sdk_chunks_share_the_model_budget_and_checkpoint_only_completed_waves(
    tmp_path: Path,
) -> None:
    backend = _BlockingEmbedder()
    run = BenchmarkRun(tmp_path / "runs", "fixture", "concurrent")
    checkpoint = runner._IngestCheckpoint(run, "unit", "fixture")
    receipts = []

    def observe(count: int) -> None:
        checkpoint.record(count, ())
        receipts.append(count)

    async with AsyncMemory(
        Memory(run.unit_dir("unit"), embedder=backend, index_quantization=IndexQuantization.NONE)
    ) as memory:
        pending = asyncio.create_task(
            runner._ingest(
                memory,
                _task(tmp_path).units[0].memories,
                batch_size=1,
                ingest_concurrency=4,
                model_semaphore=asyncio.Semaphore(2),
                on_chunk=observe,
            )
        )
        try:
            assert await asyncio.to_thread(backend.entered.wait, 3)
            assert receipts == []
            assert not run.checkpoint_path("unit").exists()
        finally:
            backend.release.set()
        assert await pending == 0
        assert len((await memory.list(limit=100)).items) == 8
    assert backend.peak == 2
    assert not backend.closed_while_active
    assert receipts == list(range(1, 9))
    assert json.loads(run.checkpoint_path("unit").read_text())["ingested"] == 8


@pytest.mark.asyncio
async def test_cancel_waits_for_real_sdk_writes_then_records_the_completed_wave(
    tmp_path: Path,
) -> None:
    backend = _BlockingEmbedder()
    run = BenchmarkRun(tmp_path / "runs", "fixture", "cancelled")
    checkpoint = runner._IngestCheckpoint(run, "unit", "fixture")

    async def ingest() -> None:
        async with AsyncMemory(
            Memory(
                run.unit_dir("unit"), embedder=backend, index_quantization=IndexQuantization.NONE
            )
        ) as memory:
            await runner._ingest(
                memory,
                _task(tmp_path).units[0].memories,
                batch_size=1,
                ingest_concurrency=2,
                model_semaphore=asyncio.Semaphore(2),
                on_chunk=lambda count: checkpoint.record(count, ()),
            )

    pending = asyncio.create_task(ingest())
    try:
        assert await asyncio.to_thread(backend.entered.wait, 3)
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
    finally:
        backend.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert backend.calls == 2
    assert not backend.closed_while_active
    assert json.loads(run.checkpoint_path("unit").read_text())["ingested"] == 2
    fresh = _BlockingEmbedder()
    fresh.release.set()
    async with AsyncMemory(
        Memory(run.unit_path("unit"), embedder=fresh, index_quantization=IndexQuantization.NONE)
    ) as memory:
        assert len((await memory.list(limit=100)).items) == 2


@pytest.mark.asyncio
async def test_later_chunk_failures_are_released_only_with_their_own_prefix() -> None:
    first_release = asyncio.Event()
    bad_finished = asyncio.Event()
    failures: list[FailureDetail] = []
    receipts: list[tuple[int, list[str | None]]] = []

    class Store:
        async def add_many(self, contents: Sequence[object], **_kwargs: object) -> None:
            if "good" in str(contents[0]):
                await first_release.wait()
            else:
                raise ValueError("fixture bad item")

        async def add(self, _content: object, **_kwargs: object) -> None:
            bad_finished.set()
            raise ValueError("fixture bad item")

    pending = asyncio.create_task(
        runner._ingest(
            cast(AsyncMemory, Store()),
            (
                MemoryItem("good", ("good",), occurred_at=DAY),
                MemoryItem("bad", ("bad",), occurred_at=DAY + timedelta(seconds=1)),
            ),
            batch_size=1,
            ingest_concurrency=2,
            on_failure=failures.append,
            on_chunk=lambda count: receipts.append(
                (count, [detail.source_id for detail in failures])
            ),
        )
    )
    await asyncio.wait_for(bad_finished.wait(), 2)
    assert failures == []
    assert receipts == []
    first_release.set()
    assert await pending == 1
    assert receipts == [(1, []), (2, ["bad"])]


@pytest.mark.asyncio
async def test_fatal_chunk_waits_for_siblings_and_stops_receipts_at_the_failed_prefix() -> None:
    release = asyncio.Event()
    failed = asyncio.Event()
    receipts: list[int] = []

    async def chunk(items: Sequence[MemoryItem]) -> int:
        if items[0].source_id == "0":
            return 0
        if items[0].source_id == "1":
            failed.set()
            raise RuntimeError("fatal fixture")
        await release.wait()
        return 0

    pending = asyncio.create_task(
        runner._ingest_chunks(
            chunk,
            tuple(
                MemoryItem(str(index), (str(index),), occurred_at=DAY + timedelta(seconds=index))
                for index in range(3)
            ),
            batch_size=1,
            on_chunk=receipts.append,
            ingest_concurrency=3,
        )
    )
    await failed.wait()
    assert not pending.done()
    assert receipts == []
    release.set()
    with pytest.raises(RuntimeError, match="fatal fixture"):
        await pending
    assert receipts == [1]


def test_ingest_concurrency_defaults_yaml_override_and_cache_identity(tmp_path: Path) -> None:
    parser = eval_config._build_parser("eval")
    base = eval_config._arguments(parser, parser.parse_args(["--tasks", "mem-gallery"]))
    assert base.ingest_concurrency == 1
    overrides = HarnessOverrides.model_validate({"run": {"ingest_concurrency": 4}})
    configured = eval_config._arguments(
        parser, parser.parse_args(["--tasks", "mem-gallery"]), overrides=overrides
    )
    assert configured.ingest_concurrency == 4
    explicit = eval_config._arguments(
        parser,
        parser.parse_args(["--tasks", "mem-gallery", "--ingest-concurrency", "2"]),
        overrides=overrides,
    )
    assert explicit.ingest_concurrency == 2
    altered = replace(base, ingest_concurrency=4)
    assert runner._cache_namespace(base, ModelConfig(), {}) != runner._cache_namespace(
        altered, ModelConfig(), {}
    )
    assert runner._ingest_digest(_task(tmp_path), base, None) != runner._ingest_digest(
        _task(tmp_path), altered, None
    )


@pytest.mark.parametrize("extra", [["--ingest", "capture"], ["--deliberate"]])
def test_parallel_ingest_rejects_stateful_modes(extra: list[str]) -> None:
    parser = eval_config._build_parser("eval")
    with pytest.raises(SystemExit) as caught:
        eval_config._arguments(
            parser,
            parser.parse_args(["--tasks", "mem-gallery", "--ingest-concurrency", "2", *extra]),
        )
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "section",
    [
        None,
        "formation: {provider: openai}",
        "speech: {provider: funasr}",
        "face: {provider: opencv, detector_model: detector.onnx, recognizer_model: recognizer.onnx}",
    ],
)
def test_cli_rejects_implicit_or_history_dependent_backends(
    tmp_path: Path, section: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--tasks", "mem-gallery", "--ingest-concurrency", "2"]
    if section is not None:
        config = tmp_path / "config.yaml"
        config.write_text(section + "\n")
        arguments += ["--config", str(config)]
    with pytest.raises(SystemExit) as caught:
        runner.main(arguments)
    assert caught.value.code == 2
    assert "without formation, speech, or face backends" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_parallel_ingest_rejects_cutoffs_before_opening_any_store(tmp_path: Path) -> None:
    def factory(_path: Path) -> AsyncMemory:
        raise AssertionError("cutoff guard must run before opening a store")

    with pytest.raises(ValueError, match="without question cutoffs"):
        await runner.run_loaded_task(
            _task(tmp_path, cutoff=10),
            run=BenchmarkRun(tmp_path / "runs", "fixture", "cutoff"),
            memory_factory=factory,
            batch_size=1,
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=10,
            ingest_concurrency=2,
        )


def _image(name: str) -> ModelInput:
    digest = name * 64
    return ModelInput(
        assets=(AssetRef(name, Modality.IMAGE, "image/png", 1, digest, path=Path("/fixture.png")),)
    )


def test_shared_asset_caption_is_described_once_across_concurrent_batches(tmp_path: Path) -> None:
    entered, release, second_started = Event(), Event(), Event()

    class Backend:
        calls = 0

        def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
            self.calls += 1
            entered.set()
            assert release.wait(5)
            return tuple(f"caption {self.calls}" for _value in inputs)

    backend = Backend()
    cache = DescriptionCache(tmp_path / "captions.db", "fixture")
    described = runner._CachedVisionDescriber(backend, cache)  # type: ignore[arg-type]

    def second() -> tuple[str, ...]:
        second_started.set()
        return described.describe((_image("a"),))

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(described.describe, (_image("a"),))
            try:
                assert entered.wait(3)
                other = workers.submit(second)
                assert second_started.wait(3)
                assert not other.done()
            finally:
                release.set()
            assert first.result() == other.result() == ("caption 1",)
        assert backend.calls == 1
    finally:
        cache.close()


def test_independent_asset_captions_can_run_concurrently(tmp_path: Path) -> None:
    together = Barrier(2, timeout=5)

    class Backend:
        def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
            together.wait()
            return tuple(value.assets[0].id for value in inputs)

    cache = DescriptionCache(tmp_path / "captions.db", "fixture")
    described = runner._CachedVisionDescriber(Backend(), cache)  # type: ignore[arg-type]
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(described.describe, (_image("a"),))
            second = workers.submit(described.describe, (_image("b"),))
            assert first.result() == ("a",)
            assert second.result() == ("b",)
    finally:
        cache.close()


class _ReverseDelayEmbedder(_BlockingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.slow_entered = Event()
        self.fast_entered = Event()

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        del task
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            if "slow" in inputs[0].text:
                self.slow_entered.set()
                assert self.release.wait(5)
            else:
                # Executor start order is unspecified: hold the fast call until the slow one
                # enters, so the independent-time overlap assertion is deterministic.
                assert self.slow_entered.wait(5)
                self.fast_entered.set()
            return tuple((1.0, 0.0) for _value in inputs)
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("clock", ["same", "independent", "unknown", "shared_start", "shared_end"])
async def test_real_sdk_reverse_delay_preserves_sqlite_order_for_overlapping_clocks(
    tmp_path: Path, clock: str
) -> None:
    backend = _ReverseDelayEmbedder()
    items = []
    for index in range(8):
        occurred = (
            None
            if clock == "unknown"
            else DAY
            if clock in {"same", "shared_start"}
            else DAY + timedelta(seconds=index)
        )
        end = (
            DAY + timedelta(seconds=10)
            if clock == "shared_end"
            else DAY + timedelta(seconds=index + 1)
            if clock == "shared_start"
            else None
        )
        items.append(
            MemoryItem(
                str(index),
                (f"{'slow' if index < 4 else 'fast'} item {index}",),
                occurred_at=occurred,
                occurred_end=end,
            )
        )
    directory = tmp_path / "store"
    async with AsyncMemory(
        Memory(directory, embedder=backend, index_quantization=IndexQuantization.NONE)
    ) as memory:
        pending = asyncio.create_task(
            runner._ingest(
                memory,
                items,
                batch_size=4,
                ingest_concurrency=2,
                model_semaphore=asyncio.Semaphore(2),
            )
        )
        try:
            assert await asyncio.to_thread(backend.slow_entered.wait, 3)
            if clock == "independent":
                assert await asyncio.to_thread(backend.fast_entered.wait, 3)

                # Delay the first chunk until the faster second one is durably visible.
                async def fast_commit_visible() -> None:
                    while len((await memory.list(limit=100)).items) < 4:  # noqa: ASYNC110 - public SDK has no commit event
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(fast_commit_visible(), 3)
            else:
                await asyncio.sleep(0.03)
                assert not backend.fast_entered.is_set()
        finally:
            backend.release.set()
            assert await pending == 0
    with closing(sqlite3.connect(directory / "state.sqlite3")) as connection:
        rows = connection.execute(
            "SELECT metadata_json FROM memory_records ORDER BY rowid"
        ).fetchall()
    order = [json.loads(row[0])["source_id"] for row in rows]
    expected = list(range(4, 8)) + list(range(4)) if clock == "independent" else list(range(8))
    assert order == [str(index) for index in expected]
    assert backend.peak == (2 if clock == "independent" else 1)
    assert not backend.closed_while_active


@pytest.mark.asyncio
async def test_fatal_predecessor_prevents_same_time_successor_but_drains_independent_work() -> None:
    called = []
    receipts: list[int] = []

    async def ingest(chunk: Sequence[MemoryItem]) -> int:
        called.append(chunk[0].source_id)
        if chunk[0].source_id == "first":
            raise RuntimeError("fatal predecessor")
        return 0

    items = (
        MemoryItem("first", ("first",), occurred_at=DAY),
        MemoryItem("dependent", ("dependent",), occurred_at=DAY),
        MemoryItem("independent", ("independent",), occurred_at=DAY + timedelta(seconds=1)),
    )
    with pytest.raises(RuntimeError, match="fatal predecessor"):
        await runner._ingest_chunks(
            ingest, items, batch_size=1, on_chunk=receipts.append, ingest_concurrency=3
        )
    assert set(called) == {"first", "independent"}
    assert receipts == []
