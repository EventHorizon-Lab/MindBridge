"""Quality evaluations must queue work within each physical SDK instance's capacity."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from threading import Event, Lock

import pytest
from _feature_support import TinyEmbedder

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    EmbedTask,
    Memory,
    Modality,
    ModelError,
    ModelInput,
    SearchHit,
)
from mindbridge.benchmarks import eval as eval_module
from mindbridge.benchmarks.eval import (
    _answer_many,
    _measure_standalone_searches,
    _model_operation,
    run_loaded_task,
)
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_arms import PRODUCT_ARM, _Arm
from mindbridge.benchmarks.eval_results import SampleResult, _AnswerOutcome
from mindbridge.benchmarks.eval_telemetry import EvaluationTelemetry
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.task_catalog import TaskSpec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_concurrency", "unit_count"), ((12, 1), (40, 1), (12, 2), (128, 4), (128, 16))
)
async def test_streaming_evaluation_queues_excess_requests_per_store(
    tmp_path: Path, request_concurrency: int, unit_count: int
) -> None:
    release = Event()
    lock = Lock()
    active = peak = 0
    target = min(request_concurrency, 8 * unit_count, 28)
    answerers: list[Answerer] = []

    class Answerer:
        generation_capabilities = frozenset({Modality.TEXT})

        def __init__(self) -> None:
            self.active = self.peak = 0

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                self.active += 1
                self.peak = max(self.peak, self.active)
                if active == target:
                    release.set()
            try:
                assert release.wait(5), "benchmark did not fill available SDK capacity"
                return AnswerResult("A", tuple(hits))
            finally:
                with lock:
                    self.active -= 1
                    active -= 1

        def close(self) -> None:
            pass

    def memory_factory(data_dir: Path) -> AsyncMemory:
        answerer = Answerer()
        answerers.append(answerer)
        return AsyncMemory(
            Memory(data_dir, embedder=TinyEmbedder(), answerer=answerer, minimum_relevance=0)
        )

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        tuple(
            EvalUnit(
                f"unit-{index}",
                (MemoryItem("evidence", ("The answer is A.",)),),
                tuple(
                    EvalQuestion(
                        f"q{question}",
                        ("What is the answer?",),
                        expected_choice="A",
                        score_kind="choice",
                    )
                    for question in range(24)
                ),
            )
            for index in range(unit_count)
        ),
    )
    try:
        samples = await run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
            memory_factory=memory_factory,
            batch_size=1,
            unit_concurrency=unit_count,
            request_concurrency=request_concurrency,
            recall_limit=1,
        )
    finally:
        release.set()

    assert len(samples) == 24 * unit_count
    assert all(sample.error_code is None for sample in samples)
    assert all(sample.prediction == "A" for sample in samples)
    assert peak == target
    assert all(answerer.peak <= 8 for answerer in answerers)


@pytest.mark.asyncio
async def test_ingest_and_answers_share_the_process_model_budget(tmp_path: Path) -> None:
    release = Event()
    lock = Lock()
    active = peak = 0

    def enter(*, wait: bool) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 28:
                release.set()
        if wait:
            assert release.wait(10), "ingest and answers did not fill the shared model budget"

    def leave() -> None:
        nonlocal active
        with lock:
            active -= 1

    class Embedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            enter(wait=task is EmbedTask.DOCUMENT and "slow" in inputs[0].text)
            try:
                return super().embed(inputs, task)
            finally:
                leave()

    class Answerer:
        generation_capabilities = frozenset({Modality.TEXT})

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            enter(wait=True)
            try:
                return AnswerResult("A", tuple(hits))
            finally:
                leave()

        def close(self) -> None:
            pass

    def factory(path: Path) -> AsyncMemory:
        return AsyncMemory(
            Memory(path, embedder=Embedder(), answerer=Answerer(), minimum_relevance=0)
        )

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        tuple(
            EvalUnit(
                str(index),
                (MemoryItem("evidence", ("fast answer A" if index < 4 else "slow answer A",)),),
                tuple(
                    EvalQuestion(str(q), ("answer?",), expected_choice="A", score_kind="choice")
                    for q in range(8)
                ),
            )
            for index in range(16)
        ),
    )
    try:
        samples = await run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "mixed"),
            memory_factory=factory,
            batch_size=1,
            unit_concurrency=16,
            request_concurrency=128,
            recall_limit=1,
        )
    finally:
        release.set()
    assert peak == 28
    assert len(samples) == 128
    assert all(sample.error_code is None and sample.prediction == "A" for sample in samples)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True))
async def test_cancelled_model_work_keeps_its_slot_until_the_sdk_stops(
    tmp_path: Path, fail: bool
) -> None:
    entered, release = Event(), Event()
    budget = asyncio.Semaphore(1)

    class Embedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            entered.set()
            assert release.wait(5)
            if fail:
                raise ValueError("injected embedding failure")
            return super().embed(inputs, task)

    async with AsyncMemory(Memory(tmp_path, embedder=Embedder())) as memory:
        pending = asyncio.create_task(_model_operation(budget, lambda: memory.add("evidence")))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            assert budget.locked()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert not budget.locked()
        assert len((await memory.list()).items) == (0 if fail else 1)


@pytest.mark.asyncio
async def test_ingest_backoff_releases_model_capacity_and_can_be_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backoff, resume = asyncio.Event(), asyncio.Event()
    budget = asyncio.Semaphore(1)
    calls = 0

    class Embedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ModelError("fixture timeout", reason="timeout", stage="embed")
            return super().embed(inputs, task)

    async def retry(operation: Callable[[], Awaitable[None]]) -> None:
        try:
            await operation()
        except ModelError:
            backoff.set()
            await resume.wait()
            await operation()

    monkeypatch.setattr(eval_module, "_retry_transient", retry)
    async with AsyncMemory(Memory(tmp_path, embedder=Embedder())) as memory:
        pending = asyncio.create_task(
            eval_module._ingest(
                memory,
                (MemoryItem("evidence", ("evidence",)),),
                batch_size=1,
                model_semaphore=budget,
            )
        )
        try:
            await asyncio.wait_for(backoff.wait(), timeout=2)
            assert not budget.locked()
        finally:
            pending.cancel()
            resume.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert calls == 1
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
async def test_cancelled_evaluation_waits_for_every_store_to_close(tmp_path: Path) -> None:
    entered = [Event(), Event()]
    release = [Event(), Event()]
    closed = [asyncio.Event(), asyncio.Event()]
    paths: list[Path] = []

    class Embedder(TinyEmbedder):
        def __init__(self, index: int) -> None:
            self.index = index

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            entered[self.index].set()
            assert release[self.index].wait(5)
            return super().embed(inputs, task)

    class TrackedMemory(AsyncMemory):
        def __init__(self, path: Path) -> None:
            self.index = len(paths)
            paths.append(path)
            super().__init__(Memory(path, embedder=Embedder(self.index)))

        async def __aexit__(self, *_error: object) -> None:
            await super().__aexit__(*_error)
            closed[self.index].set()

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        tuple(
            EvalUnit(
                str(index),
                (MemoryItem("evidence", ("evidence",)),),
                (EvalQuestion("q", ("question",), references=("answer",)),),
            )
            for index in range(2)
        ),
    )
    pending = asyncio.create_task(
        run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "cancel"),
            memory_factory=TrackedMemory,
            batch_size=1,
            unit_concurrency=2,
            request_concurrency=28,
            recall_limit=1,
        )
    )
    try:
        assert await asyncio.to_thread(lambda: all(event.wait(2) for event in entered))
        pending.cancel()
        release[0].set()
        await asyncio.wait_for(closed[0].wait(), timeout=2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=0.05)
    finally:
        for event in release:
            event.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert all(event.is_set() for event in closed)
    for path in paths:
        with Memory(path, embedder=TinyEmbedder()) as memory:
            assert len(memory.list().items) == 1


class _ConcurrentQueries(TinyEmbedder):
    """Hold the first wave until all 28 model slots fill, making overload deterministic."""

    def __init__(self) -> None:
        self.release = Event()
        self._lock = Lock()
        self._active = self.peak = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is not EmbedTask.QUERY:
            return super().embed(inputs, task)
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            if self._active == 28:
                self.release.set()
        try:
            assert self.release.wait(5), "benchmark did not fill available SDK capacity"
            return super().embed(inputs, task)
        finally:
            with self._lock:
                self._active -= 1


@pytest.mark.asyncio
async def test_non_streaming_evaluation_queues_excess_requests_per_store(tmp_path: Path) -> None:
    embedder = _ConcurrentQueries()
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        memory.add("The answer is A.")
        try:
            outcomes = await _answer_many(
                AsyncMemory(memory),
                tuple(
                    EvalQuestion(f"q{index}", ("What is the answer?",), references=("A",))
                    for index in range(128)
                ),
                request_concurrency=128,
                recall_limit=1,
                arm=_Arm("random"),
            )
        finally:
            embedder.release.set()

    assert len(outcomes) == 128
    assert all(
        isinstance(outcome, _AnswerOutcome) and outcome.error is None for outcome in outcomes
    )
    assert embedder.peak == 28


@pytest.mark.asyncio
async def test_search_replay_queues_excess_requests_per_store(tmp_path: Path) -> None:
    embedder = _ConcurrentQueries()
    questions = tuple(
        EvalQuestion(f"q{index}", ("What is the answer?",), references=("A",))
        for index in range(128)
    )
    samples = tuple(
        SampleResult(
            task="fixture",
            benchmark="Fixture",
            dataset_sha256="1" * 64,
            evaluation_sha256="2" * 64,
            unit_id="unit",
            question_id=question.question_id,
            prediction="A",
            parsed_choice="A",
            score=1.0,
            exact_match=1.0,
            latency_ms=1.0,
            confidence=1.0,
            memory_ids=(),
            ingest_failure_count=0,
            error_code=None,
            metadata={},
        )
        for question in questions
    )
    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (EvalUnit("unit", (), questions),),
    )
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0) as memory:
        memory.add("The answer is A.")
        telemetry = EvaluationTelemetry()
        try:
            await _measure_standalone_searches(
                task,
                slots=(samples,),
                unit_paths=(tmp_path,),
                stores_ready=(True,),
                memory_factory=lambda _path: AsyncMemory(memory),
                unit_concurrency=1,
                request_semaphore=asyncio.Semaphore(128),
                recall_limit=1,
                arms=(PRODUCT_ARM,),
                tracer=telemetry.tracer,
            )
            performance = telemetry.result("fixture", question_count=128)
        finally:
            embedder.release.set()
            telemetry.close()

    search_e2e = performance["search_e2e"]
    assert isinstance(search_e2e, dict)
    assert search_e2e["count"] == 128
    assert embedder.peak == 28
