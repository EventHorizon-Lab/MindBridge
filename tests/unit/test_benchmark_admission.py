"""Quality evaluations must queue work within each physical SDK instance's capacity."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
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
    ModelInput,
    SearchHit,
)
from mindbridge.benchmarks.eval import _answer_many, _measure_standalone_searches, run_loaded_task
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_arms import PRODUCT_ARM, _Arm
from mindbridge.benchmarks.eval_results import SampleResult, _AnswerOutcome
from mindbridge.benchmarks.eval_telemetry import EvaluationTelemetry
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.task_catalog import TaskSpec


@pytest.mark.asyncio
@pytest.mark.parametrize(("request_concurrency", "unit_count"), ((12, 1), (40, 1), (12, 2)))
async def test_streaming_evaluation_queues_excess_requests_per_store(
    tmp_path: Path, request_concurrency: int, unit_count: int
) -> None:
    release = Event()
    lock = Lock()
    active = peak = 0
    target = min(request_concurrency, 8 * unit_count)
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


class _ConcurrentQueries(TinyEmbedder):
    """Hold the first wave until all 32 SDK callers arrive, making overload deterministic."""

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
            if self._active == 32:
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
                    for index in range(48)
                ),
                request_concurrency=48,
                recall_limit=1,
                arm=_Arm("random"),
            )
        finally:
            embedder.release.set()

    assert len(outcomes) == 48
    assert all(
        isinstance(outcome, _AnswerOutcome) and outcome.error is None for outcome in outcomes
    )
    assert embedder.peak == 32


@pytest.mark.asyncio
async def test_search_replay_queues_excess_requests_per_store(tmp_path: Path) -> None:
    embedder = _ConcurrentQueries()
    questions = tuple(
        EvalQuestion(f"q{index}", ("What is the answer?",), references=("A",))
        for index in range(48)
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
                request_semaphore=asyncio.Semaphore(48),
                recall_limit=1,
                arms=(PRODUCT_ARM,),
                tracer=telemetry.tracer,
            )
            performance = telemetry.result("fixture", question_count=48)
        finally:
            embedder.release.set()
            telemetry.close()

    search_e2e = performance["search_e2e"]
    assert isinstance(search_e2e, dict)
    assert search_e2e["count"] == 48
    assert embedder.peak == 32
