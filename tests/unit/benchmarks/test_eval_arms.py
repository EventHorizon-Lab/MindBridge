"""Baseline arms, retrieval-metric scope, and honest official-metric stamping."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import AsyncIterator, Generator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.util.types import AttributeValue

import mindbridge.benchmarks.eval as eval_module
from mindbridge import (
    AnswerChunk,
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    ContextBudget,
    EmbedTask,
    Memory,
    MemoryConfig,
    MindBridgeConfig,
    Modality,
    ModelError,
    ModelInput,
    SearchHit,
)
from mindbridge._telemetry import (
    RECALL_COMPLETE,
    RECALL_EXHAUSTIVE_ROWS,
    RECALL_FALLBACK,
    RECALL_NON_SELECTIVE_STEPS,
    RECALL_REPLAN,
    RECALL_SHAPE,
    SPAN_KIND,
    _record_retrieval_results,
)
from mindbridge.benchmarks.eval import (
    DEFAULT_ARM,
    PRODUCT_ARM,
    RETRIEVAL_CANDIDATE_LIMIT,
    MemoryFactory,
    SampleResult,
    _answer_many,
    _Arm,
    _full_context,
    _metrics,
    _sample,
    _with_grounding_loss,
    run_loaded_task,
)
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_cache import CachedAnswer, ResponseCache
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_ANSWER_SPAN,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    RECALL_SPAN,
    EvaluationTelemetry,
    SampleGrounding,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import DEFAULT_TIMEOUT_SECONDS, ModelConfig
from mindbridge.benchmarks.official_scorers import metric_is_official, retrieval_gold_ids
from mindbridge.benchmarks.task_catalog import TaskSpec
from mindbridge.configuration import OpenAIEmbeddingConfig
from mindbridge.models.base import (
    GenerationBackend,
    RecallPlanningBackend,
    StreamingGenerationBackend,
)
from mindbridge.models.openai_sdk import OpenAIModels

_ATOMIC_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})


class _TinyEmbedder:
    """Small deterministic embedder so the fast-plane/compiler tests use a real, isolated
    `AsyncMemory` (the public SDK) instead of a memory double, with no model service."""

    embedding_capabilities = _ATOMIC_MODALITIES
    embedding_model = "tiny-eval-arm-test"
    embedding_space = "tiny-eval-arm-test:4:l2-v1"
    embedding_dimension = 4

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        del task
        vectors = []
        for value in inputs:
            digest = hashlib.sha256(value.text.encode()).digest()
            vector = tuple(1.0 + digest[index] / 255.0 for index in range(4))
            norm = math.sqrt(sum(component * component for component in vector))
            vectors.append(tuple(component / norm for component in vector))
        return tuple(vectors)

    def close(self) -> None:
        return None


class _ExcerptEmbedder:
    """Force a long aggregate to fall back to its matching bounded text part."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "excerpt-eval-arm-test"
    embedding_space = "excerpt-eval-arm-test:2"
    embedding_dimension = 2

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        if task is EmbedTask.DOCUMENT and any(len(value.text) > 3_000 for value in batch):
            raise ModelError("fixture payload is too large", reason="payload_too_large")
        return tuple(
            (1.0, 0.0)
            if task is EmbedTask.QUERY or "Ada signed the contract" in value.text
            else (0.0, 1.0)
            for value in batch
        )

    def close(self) -> None:
        return None


_NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _hit(source_id: str, score: float) -> SearchHit:
    return SearchHit(
        id=f"memory-{source_id}",
        content=source_id,
        score=score,
        created_at=_NOW,
        metadata={"source_id": source_id},
    )


def _question(question_id: str = "q1", **metadata: object) -> EvalQuestion:
    return EvalQuestion(
        question_id,
        ("who signed it?",),
        references=("Ada",),
        metadata={"qtype": "open_end", **metadata},
        source_question="who signed it?",
    )


def _task(
    name: str = "atm-bench", *, labelled: bool = True
) -> tuple[LoadedTask, EvalUnit, EvalQuestion]:
    question = _question(evidence_ids=["gold-1"]) if labelled else _question()
    unit = EvalUnit(
        "unit",
        (
            MemoryItem("s1", ("a lunch invitation",)),
            MemoryItem("gold-1", ("Ada signed the contract",)),
        ),
        (question,),
    )
    task = LoadedTask(
        TaskSpec(name, "ATM-Bench", "fixture.json", "v1", "owner/repo", "0" * 40),
        Path("fixture.json"),
        "1" * 64,
        (unit,),
    )
    return task, unit, question


class _RankedMemory:
    """A store whose ranked list and whose answer hits deliberately disagree."""

    def __init__(self, ranked: Sequence[SearchHit], answer_hits: Sequence[SearchHit]) -> None:
        self._ranked = tuple(ranked)
        self._answer_hits = tuple(answer_hits)
        self.search_limits: list[int] = []
        self.asked = 0

    async def ask(self, _question: object, **_kwargs: object) -> AnswerResult:
        self.asked += 1
        _record_retrieval_results(self._ranked)
        return AnswerResult("Ada.", self._answer_hits)

    async def search(
        self, _query: object, *, limit: int, **_kwargs: object
    ) -> tuple[SearchHit, ...]:
        self.search_limits.append(limit)
        return self._ranked


class _ForbiddenMemory:
    """Any read here means a baseline arm fell back into the product path."""

    async def ask(self, _question: object, **_kwargs: object) -> AnswerResult:
        raise AssertionError("a no-retrieval arm called Memory.ask")

    async def search(self, _query: object, **_kwargs: object) -> tuple[SearchHit, ...]:
        raise AssertionError("a no-retrieval arm called Memory.search")

    async def add_many(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise AssertionError("a no-retrieval arm ingested a corpus")


class _RecordingGenerator:
    def __init__(self, reply: str = "Ada") -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.media_calls: list[dict[str, object]] = []
        self._reply = reply

    async def answer(
        self,
        question: str,
        context: str | None,
        **_media: object,
    ) -> str:
        self.calls.append((question, context))
        self.media_calls.append(_media)
        return self._reply


def _outcome(
    memory: object,
    question: EvalQuestion,
    *,
    arm: _Arm,
    task_name: str = "atm-bench",
    context: str = "",
) -> eval_module._AnswerOutcome:
    answered = asyncio.run(
        _answer_many(
            cast(AsyncMemory, memory),
            (question,),
            request_concurrency=1,
            recall_limit=20,
            arm=arm,
            task_name=task_name,
            unit_id="unit",
            context=context,
        )
    )[0]
    assert not isinstance(answered, BaseException)
    return answered


def test_retrieval_recall_scores_the_full_ranked_list_not_the_answer_hits() -> None:
    ranked = tuple(_hit(f"noise-{index}", 0.9 - index / 100) for index in range(7))
    ranked = (*ranked, _hit("gold-1", 0.2), *(_hit(f"tail-{i}", 0.1) for i in range(4)))
    # `ask` returns the modality round robin's survivors: the gold first, everything else gone.
    memory = _RankedMemory(ranked, (_hit("gold-1", 0.2), _hit("noise-0", 0.9)))
    task, unit, question = _task()

    outcome = _outcome(memory, question, arm=PRODUCT_ARM)
    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=False,
        log_samples=False,
        arm=PRODUCT_ARM,
    )

    assert memory.search_limits == []
    assert len(ranked) == 12
    assert sample.retrieval_candidates == 12
    # Rank 8 of 12: inside recall@10, outside recall@5, and not the top-1.
    assert sample.metrics["retrieval_recall@10"] == 1.0
    assert sample.metrics["retrieval_recall@5"] == 0.0
    assert sample.metrics["retrieval_hit@1"] == 0.0
    # The answer's own hits are still reported, so budget loss stays visible.
    assert tuple(item.source_id for item in sample.evidence) == ("gold-1", "noise-0")


def test_successful_empty_retrieval_diagnostic_scores_zero_recall() -> None:
    memory = _RankedMemory((), ())
    task, unit, question = _task()

    outcome = _outcome(memory, question, arm=PRODUCT_ARM)
    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=False,
        log_samples=False,
        arm=PRODUCT_ARM,
    )
    retrieval = eval_module._retrieval_quality(
        (sample,), seed=7, bootstrap_samples=32, recall_limit=20
    )

    assert memory.search_limits == []
    assert outcome.ranked_source_ids == ()
    assert outcome.ranked_source_ids_complete is True
    assert sample.metrics["retrieval_recall@1"] == 0.0
    assert retrieval["labelled_question_count"] == 1
    assert retrieval["unranked_labelled_question_count"] == 0
    recall_at_k = cast(dict[str, dict[str, float]], retrieval["recall_at_k"])
    assert recall_at_k["20"]["mean"] == pytest.approx(0.0)


def test_retrieval_candidates_are_not_fetched_without_gold_to_score() -> None:
    memory = _RankedMemory((), (_hit("gold-1", 0.2),))
    question = _question()

    outcome = _outcome(memory, question, arm=PRODUCT_ARM)

    assert memory.search_limits == []
    assert memory.asked == 1
    assert outcome.ranked_source_ids == ()
    assert retrieval_gold_ids("atm-bench", question.metadata) == ()


def test_blind_arm_never_reads_memory_and_keeps_query_media(tmp_path: Path) -> None:
    generator = _RecordingGenerator()
    arm = _Arm("blind", generator=cast(eval_module._BaselineGenerator, generator))
    _, _, question = _task()
    query_image = tmp_path / "query.png"
    query_image.write_bytes(b"query")
    question = replace(question, content=(*question.content, query_image))

    outcome = _outcome(_ForbiddenMemory(), question, arm=arm, context="a stuffed corpus")

    assert generator.calls == [("who signed it?", None)]
    assert generator.media_calls == [{"question_assets": (query_image,)}]
    assert outcome.prediction == "Ada"
    assert outcome.memory_ids == ()
    assert outcome.ranked_source_ids == ()


def test_full_context_arm_stuffs_the_corpus_instead_of_retrieving() -> None:
    generator = _RecordingGenerator()
    arm = _Arm("full-context", generator=cast(eval_module._BaselineGenerator, generator))
    _, _, question = _task()

    outcome = _outcome(_ForbiddenMemory(), question, arm=arm, context="Ada signed the contract")

    assert generator.calls == [("who signed it?", "Ada signed the contract")]
    assert outcome.prediction == "Ada"


def test_full_context_respects_its_stated_budget_and_skips_media() -> None:
    items = (
        MemoryItem("s1", ("a" * 10,)),
        MemoryItem("s2", (Path("clip.mp4"),)),
        MemoryItem("s3", ("b" * 10,)),
        MemoryItem("s4", ("c" * 10,)),
    )

    assert _full_context(items, 25) == f"{'a' * 10}\n\n{'b' * 10}"
    assert _full_context(items, 100) == f"{'a' * 10}\n\n{'b' * 10}\n\n{'c' * 10}"


def test_random_arm_shuffles_the_same_pool_and_scores_retrieval_only() -> None:
    ranked = tuple(_hit(f"s{index}", 0.9 - index / 100) for index in range(9))
    ranked = (*ranked, _hit("gold-1", 0.1))
    memory = _RankedMemory(ranked, ())
    task, unit, question = _task()
    arm = _Arm("random", seed=7)

    outcome = _outcome(memory, question, arm=arm)
    repeat = _outcome(_RankedMemory(ranked, ()), question, arm=arm)
    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=False,
        log_samples=False,
        arm=arm,
    )

    assert outcome.ranked_source_ids == repeat.ranked_source_ids
    assert sorted(outcome.ranked_source_ids) == sorted(
        str(hit.metadata["source_id"]) for hit in ranked
    )
    assert outcome.ranked_source_ids != tuple(str(hit.metadata["source_id"]) for hit in ranked), (
        "a random ranker that preserves score order is not a baseline"
    )
    assert outcome.prediction == ""
    assert sample.score is None
    assert set(sample.metrics) == {
        "retrieval_recall@1",
        "retrieval_recall@5",
        "retrieval_recall@10",
        "retrieval_recall@gt",
        "retrieval_hit@1",
    }
    assert sample.sample_id.startswith("random:")


def test_random_arm_does_not_report_an_answer_latency() -> None:
    ranked = (_hit("gold-1", 0.9),)
    memory = _RankedMemory(ranked, ())
    _, _, question = _task()
    telemetry = EvaluationTelemetry()
    try:
        outcome = asyncio.run(
            _answer_many(
                cast(AsyncMemory, memory),
                (question,),
                request_concurrency=1,
                recall_limit=20,
                arm=_Arm("random", seed=7),
                task_name="atm-bench",
                unit_id="unit",
                tracer=telemetry.tracer,
            )
        )[0]
        performance = telemetry.result("atm-bench", arm="random", question_count=1)
    finally:
        telemetry.close()

    assert not isinstance(outcome, BaseException)
    assert outcome.ranked_source_ids == ("gold-1",)
    answer = cast(dict[str, object], performance["answer"])
    assert answer["span"] == "mindbridge.ask"
    assert answer["count"] == 0
    assert BENCHMARK_ANSWER_SPAN not in cast(dict[str, object], performance["nodes"])


@pytest.mark.asyncio
async def test_caller_answer_clock_includes_request_admission() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class Memory:
        async def ask_stream(self, question: str, **_kwargs: object) -> AsyncIterator[AnswerChunk]:
            if question == "first":
                first_started.set()
                await release_first.wait()
            yield AnswerChunk(text="A")
            yield AnswerChunk(result=AnswerResult("A"))

    telemetry = EvaluationTelemetry()
    try:
        pending = asyncio.create_task(
            _answer_many(
                cast(AsyncMemory, Memory()),
                (
                    EvalQuestion("first", ("first",), references=("A",)),
                    EvalQuestion("queued", ("queued",), references=("A",)),
                ),
                request_concurrency=1,
                recall_limit=1,
                task_name="fixture",
                unit_id="unit",
                tracer=telemetry.tracer,
            )
        )
        await first_started.wait()
        await asyncio.sleep(0.05)
        release_first.set()
        first, queued = await pending
        performance = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    assert not isinstance(first, BaseException)
    assert not isinstance(queued, BaseException)
    assert queued.latency_ms < first.latency_ms / 4
    answer = cast(dict[str, object], performance["answer"])
    caller_latency = cast(dict[str, float], answer["latency_ms"])
    caller_ttft = cast(dict[str, float], answer["end_to_end_time_to_first_token_ms"])
    assert caller_latency["p50"] >= 40
    assert caller_ttft["p50"] >= 40


@pytest.mark.asyncio
async def test_product_ranking_is_captured_without_a_scoring_search() -> None:
    answered = 0

    class Memory:
        async def ask_stream(
            self, _question: object, **_kwargs: object
        ) -> AsyncIterator[AnswerChunk]:
            nonlocal answered
            _record_retrieval_results(())
            yield AnswerChunk(text="Ada")
            yield AnswerChunk(result=AnswerResult("Ada"))
            answered += 1

        async def search(self, _question: object, **_kwargs: object) -> tuple[SearchHit, ...]:
            raise AssertionError("product retrieval issued a second scoring search")

    questions = tuple(_question(f"q{index}", evidence_ids=["gold-1"]) for index in range(3))
    telemetry = EvaluationTelemetry()
    try:
        outcomes = await _answer_many(
            cast(AsyncMemory, Memory()),
            questions,
            request_concurrency=1,
            recall_limit=20,
            arm=PRODUCT_ARM,
            task_name="atm-bench",
            unit_id="unit",
            tracer=telemetry.tracer,
        )
        performance = telemetry.result("atm-bench", question_count=3)
    finally:
        telemetry.close()

    assert all(not isinstance(outcome, BaseException) for outcome in outcomes)
    assert answered == 3
    assert all(
        isinstance(outcome, eval_module._AnswerOutcome) and outcome.ranked_source_ids_complete
        for outcome in outcomes
    )
    diagnostic = cast(dict[str, object], performance["diagnostic"])
    caller = cast(dict[str, object], diagnostic["search_e2e"])
    assert caller["attempt_count"] == caller["count"] == 0


def test_blind_only_run_answers_every_question_without_ingesting(tmp_path: Path) -> None:
    generator = _RecordingGenerator()
    task, _, _ = _task()

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, _ForbiddenMemory())

        async def __aexit__(self, *_error: object) -> None:
            return None

    telemetry = EvaluationTelemetry()
    try:
        samples = asyncio.run(
            run_loaded_task(
                task,
                run=BenchmarkRun(tmp_path / "stores", "atm-bench", "run"),
                memory_factory=cast(MemoryFactory, lambda _path: Context()),
                batch_size=4,
                unit_concurrency=1,
                request_concurrency=1,
                recall_limit=5,
                arms=(_Arm("blind", generator=cast(eval_module._BaselineGenerator, generator)),),
                tracer=telemetry.tracer,
            )
        )
    finally:
        telemetry.close()

    assert len(samples) == 1
    assert samples[0].arm == "blind"
    assert samples[0].error_code is None
    assert samples[0].prediction == "Ada"
    assert "retrieval_recall@10" not in samples[0].metrics


def test_cache_only_product_run_does_not_fabricate_search_replay(tmp_path: Path) -> None:
    task, _, _ = _task()
    factory_calls = 0

    class Cache:
        def get(self, _name: str, _unit: str, _question: str) -> CachedAnswer:
            return CachedAnswer("Ada", 0.0, (), ranked_source_ids=("gold-1",))

        def put(self, *_args: object) -> None:
            raise AssertionError("a cache-only run attempted to write its response cache")

    def factory(_path: Path) -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("a cache-only run opened a memory store")

    telemetry = EvaluationTelemetry()
    try:
        samples = asyncio.run(
            run_loaded_task(
                task,
                run=BenchmarkRun(tmp_path / "stores", "atm-bench", "run"),
                memory_factory=cast(MemoryFactory, factory),
                batch_size=1,
                unit_concurrency=1,
                request_concurrency=1,
                recall_limit=5,
                predict_only=True,
                response_cache=cast(ResponseCache, Cache()),
                tracer=telemetry.tracer,
            )
        )
        performance = telemetry.result("atm-bench", question_count=0)
    finally:
        telemetry.close()

    assert len(samples) == 1 and samples[0].cached
    assert factory_calls == 0
    search_e2e = cast(dict[str, object], performance["search_e2e"])
    assert search_e2e["attempt_count"] == search_e2e["count"] == 0


@pytest.mark.asyncio
async def test_cached_product_arm_does_not_force_ingest_for_a_pending_blind_arm(
    tmp_path: Path,
) -> None:
    generator = _RecordingGenerator()
    task, unit, _ = _task()

    class Cache:
        def get(self, name: str, _unit: str, _question: str) -> CachedAnswer | None:
            return (
                None
                if name.startswith("blind:")
                else CachedAnswer("Ada", 0.0, (), ranked_source_ids=("gold-1",))
            )

        def put(self, *_args: object) -> None:
            return None

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, _ForbiddenMemory())

        async def __aexit__(self, *_error: object) -> None:
            return None

    samples = await eval_module._run_unit(
        task,
        unit,
        tmp_path / "store",
        memory_factory=cast(MemoryFactory, lambda _path: Context()),
        batch_size=1,
        request_concurrency=1,
        request_semaphore=asyncio.Semaphore(1),
        recall_limit=5,
        predict_only=True,
        log_samples=False,
        response_cache=cast(ResponseCache, Cache()),
        arms=(
            PRODUCT_ARM,
            _Arm("blind", generator=cast(eval_module._BaselineGenerator, generator)),
        ),
    )

    assert [sample.cached for sample in samples] == [True, False]
    assert [sample.ingest_failure_count for sample in samples] == [0, 0]


def test_every_arm_answers_the_same_questions_against_one_ingest(tmp_path: Path) -> None:
    generator = _RecordingGenerator()
    task, _, _ = _task()
    ingested: list[int] = []

    class Memory:
        async def add_many(self, contents: Sequence[object], **_kwargs: object) -> tuple[()]:
            ingested.append(len(contents))
            return ()

        async def ask(self, _question: object, **_kwargs: object) -> AnswerResult:
            _record_retrieval_results((_hit("s1", 0.9), _hit("gold-1", 0.5)))
            return AnswerResult("Ada.", (_hit("gold-1", 0.5),))

        async def search(self, _query: object, **_kwargs: object) -> tuple[SearchHit, ...]:
            return (_hit("s1", 0.9), _hit("gold-1", 0.5))

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory())

        async def __aexit__(self, *_error: object) -> None:
            return None

    samples = asyncio.run(
        run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "atm-bench", "run"),
            memory_factory=cast(MemoryFactory, lambda _path: Context()),
            batch_size=4,
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=5,
            arms=(
                PRODUCT_ARM,
                _Arm("blind", generator=cast(eval_module._BaselineGenerator, generator)),
                _Arm("random", seed=3),
            ),
        )
    )

    assert sum(ingested) == 2
    assert tuple(sample.arm for sample in samples) == (DEFAULT_ARM, "blind", "random")
    assert {sample.sample_id for sample in samples} == {
        "atm-bench/unit/q1",
        "blind:atm-bench/unit/q1",
        "random:atm-bench/unit/q1",
    }
    assert samples[0].metrics["retrieval_recall@10"] == 1.0
    assert "retrieval_recall@10" not in samples[1].metrics
    assert samples[2].metrics["retrieval_recall@10"] == 1.0


def test_compile_arm_answers_from_a_rendered_bundle_via_the_public_sdk(tmp_path: Path) -> None:
    """The compile arm against a real, isolated `AsyncMemory` -- the public SDK, no doubles."""
    generator = _RecordingGenerator("Ada")
    arm = _Arm("compile", generator=cast(eval_module._BaselineGenerator, generator))
    _, _, question = _task()

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(
            Memory(tmp_path, embedder=_TinyEmbedder(), minimum_relevance=0)
        ) as memory:
            await memory.add("Ada signed the contract")
            await memory.add("a lunch invitation")
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=20,
                arm=arm,
                task_name="atm-bench",
                unit_id="unit",
                compile_budget=ContextBudget(max_items=5, max_chars=2_000),
            )
            return answered[0]

    outcome = asyncio.run(run())
    assert not isinstance(outcome, BaseException)

    assert outcome.prediction == "Ada"
    assert outcome.compiled_items is not None and outcome.compiled_items >= 1
    assert outcome.compiled_chars is not None and outcome.compiled_chars > 0
    assert len(outcome.memory_ids) == outcome.compiled_items
    assert len(generator.calls) == 1
    question_text, rendered_context = generator.calls[0]
    assert question_text == "who signed it?"
    # Fed the compiled bundle, not the stuffed corpus a full-context arm would build.
    assert rendered_context is not None
    assert rendered_context.startswith("# Context: who signed it?")
    assert "Ada signed the contract" in rendered_context


def test_the_product_arm_answers_a_compile_surface_task_from_the_compiled_bundle(
    tmp_path: Path,
) -> None:
    """On a task `task_answer_surface` maps to `compile`, the product arm is `Memory.compile`
    plus the generator standing in for the host assistant -- under the assistant prompt, not the
    baseline's "answer only, guess" one -- and never reaches `Memory.ask`."""
    generator = _RecordingGenerator("a post in your voice")
    arm = _Arm(DEFAULT_ARM, generator=cast(eval_module._BaselineGenerator, generator))
    _, _, question = _task(labelled=False)

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(
            Memory(tmp_path, embedder=_TinyEmbedder(), minimum_relevance=0)
        ) as memory:
            await memory.add("Ada signed the contract")
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=20,
                arm=arm,
                task_name="personamem-v3",
                unit_id="unit",
                compile_budget=ContextBudget(max_items=5, max_chars=2_000),
            )
            return answered[0]

    outcome = asyncio.run(run())
    assert not isinstance(outcome, BaseException)
    assert outcome.prediction == "a post in your voice"
    assert outcome.compiled_items is not None and outcome.compiled_items >= 1
    assert len(generator.calls) == 1
    _, rendered = generator.calls[0]
    assert rendered is not None and rendered.startswith("# Context: who signed it?")
    assert generator.media_calls[0]["system_prompt"] == eval_module._COMPILE_SURFACE_SYSTEM_PROMPT
    # No `ask` request was made, so the row records no policy; and with no ranked list to
    # observe the row is not flagged as a missing retrieval diagnostic.
    assert outcome.retrieval_diagnostic_error is None
    task, unit, _ = _task("personamem-v3", labelled=False)
    row = _sample(
        task, unit, question, outcome, ingest_failures=0, predict_only=True, log_samples=False
    )
    assert row.answer_policy is None
    assert not row.abstained


def test_the_product_arm_keeps_asking_on_an_ask_surface_task_even_with_a_generator() -> None:
    """A multi-task run lends the product arm a generator for its compile-surface task; that
    must not turn its other tasks into a full-context baseline."""
    generator = _RecordingGenerator()
    arm = _Arm(DEFAULT_ARM, generator=cast(eval_module._BaselineGenerator, generator))
    memory = _RankedMemory((), ())
    _, _, question = _task(labelled=False)

    outcome = _outcome(memory, question, arm=arm, task_name="atm-bench")

    assert outcome.prediction == "Ada."
    assert memory.asked == 1
    assert generator.calls == []


def test_the_generator_is_built_for_the_product_arm_only_when_a_task_needs_the_compile_surface() -> (
    None
):
    assert eval_module._generator_arms(("mindbridge",), ("locomo-refined",)) == frozenset()
    assert eval_module._generator_arms(("mindbridge",), ("personamem-v3",)) == {"mindbridge"}
    assert eval_module._generator_arms(("mindbridge", "blind"), ("locomo-refined",)) == {"blind"}
    assert eval_module._generator_arms(("compile", "random"), ("personamem-v3",)) == {"compile"}


def test_compile_arm_records_partial_sources_separately_from_full_evidence(
    tmp_path: Path,
) -> None:
    generator = _RecordingGenerator("Ada")
    arm = _Arm(
        "compile",
        generator=cast(eval_module._BaselineGenerator, generator),
        allow_partial_sources=True,
    )
    task, unit, question = _task()
    source_text = (
        "Conversation header\n"
        + "a" * 2_300
        + " Ada signed the contract. "
        + "b" * 2_300
        + " Later text may qualify that statement."
    )

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(
            Memory(
                tmp_path,
                embedder=_ExcerptEmbedder(),
                minimum_relevance=0,
            )
        ) as memory:
            source = await memory.add(source_text)
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=20,
                arm=arm,
                task_name="atm-bench",
                unit_id="unit",
                compile_budget=ContextBudget(max_items=1, max_chars=3_000),
            )
            outcome = answered[0]
            if isinstance(outcome, eval_module._AnswerOutcome):
                assert outcome.excerpt_source_ids == (source.id,)
            return outcome

    outcome = asyncio.run(run())
    assert not isinstance(outcome, BaseException)
    assert outcome.memory_ids == ()
    assert outcome.evidence == ()
    assert len(outcome.excerpt_evidence) == 1
    assert outcome.compiled_items == 1
    assert outcome.compiled_chars is not None and outcome.compiled_chars <= 3_000
    assert generator.calls[0][1] is not None
    assert "partial source" in generator.calls[0][1]

    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=True,
        log_samples=False,
        arm=arm,
    )
    document = sample.json()
    assert document["excerpt_source_ids"] == outcome.excerpt_source_ids
    assert document["excerpt_evidence"]
    assert sample.metrics["compile_bundle_excerpts"] == 1.0


def test_compile_arm_sends_query_and_evidence_images_on_the_final_provider_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native media follows the selected public bundle without consulting gold metadata."""
    import openai

    requests: list[dict[str, object]] = []

    class Completions:
        async def create(self, **request: object) -> object:
            requests.append(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="robot"))],
                usage=None,
            )

    class Client:
        def __init__(self, **options: object) -> None:
            requests.append({"client": options})
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    query_image = tmp_path / "query.png"
    query_image.write_bytes(b"query-image")
    evidence_image = tmp_path / "evidence.png"
    evidence_image.write_bytes(b"evidence-image")
    question = EvalQuestion(
        "q-image",
        ("Which robot is shown?", query_image),
        references=("robot",),
        metadata={},
        source_question="Which robot is shown?",
    )
    generator = eval_module._BaselineGenerator(
        ModelConfig(
            generation_model="gpt-5-mini",
            generation_base_url="http://generate/v1",
            generation_api_key="secret",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        ),
        seed=42,
        gen_kwargs="",
    )

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(Memory(tmp_path / "store", embedder=_TinyEmbedder())) as memory:
            await memory.add(
                ("The evidence shows a robot.", evidence_image),
                occurred_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                metadata={"source_id": "turn-1"},
            )
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=20,
                arm=_Arm("compile", generator=generator),
                task_name="mem-gallery",
                unit_id="gallery",
                compile_budget=ContextBudget(max_items=5, max_chars=16_000),
            )
            return answered[0]

    try:
        outcome = asyncio.run(run())
    finally:
        asyncio.run(generator.close())
    assert not isinstance(outcome, BaseException)
    assert outcome.error is None
    assert len(requests) == 2
    messages = cast(list[dict[str, object]], requests[1]["messages"])
    content = cast(list[dict[str, object]], messages[1]["content"])
    image_urls = [
        cast(dict[str, str], part["image_url"])["url"]
        for part in content
        if part["type"] == "image_url"
    ]
    assert image_urls == [
        "data:image/png;base64,cXVlcnktaW1hZ2U=",
        "data:image/png;base64,ZXZpZGVuY2UtaW1hZ2U=",
    ]
    assert "The evidence shows a robot." in cast(str, content[0]["text"])
    # Each JSON binding is followed by the product's attachment marker and then the media it
    # announces, so the wire reads: context, query binding, marker, image, memory binding,
    # marker, image.
    assert [part["type"] for part in content] == [
        "text",
        "text",
        "text",
        "image_url",
        "text",
        "text",
        "image_url",
    ]
    assert [cast(str, content[index]["text"]) for index in (2, 5)] == [
        "attachment 1",
        "attachment 2",
    ]
    question_binding = json.loads(cast(str, content[1]["text"]))
    evidence_binding = json.loads(cast(str, content[4]["text"]))["memory_assets"]
    assert question_binding == {
        "query_assets": [hashlib.sha256(b"query-image").hexdigest()],
    }
    assert evidence_binding["asset_ids"] == [hashlib.sha256(b"evidence-image").hexdigest()]
    assert evidence_binding["memory_id"] in outcome.memory_ids
    assert evidence_binding["memory_type"] == "semantic"
    assert evidence_binding["source_id"] == "turn-1"
    assert evidence_binding["event_time"] == "2026-01-02T00:00:00+00:00"
    assert "content" not in evidence_binding
    wire_text = "\n".join(cast(str, part["text"]) for part in content if part["type"] == "text")
    assert wire_text.count("The evidence shows a robot.") == 1


def test_ingest_capture_produces_real_capture_settle_spans_for_the_compile_arm(
    tmp_path: Path,
) -> None:
    """`--ingest capture` drives `capture()`/`settle()` through the public SDK, so
    `eval_telemetry` reports real capture-acknowledgement and time-to-searchable numbers
    instead of an empty block, and the compile arm still answers off the settled store.
    """
    generator = _RecordingGenerator("Ada")
    task, _, _ = _task()
    telemetry = EvaluationTelemetry()

    def memory_factory(path: Path) -> AsyncMemory:
        return AsyncMemory(
            Memory(path, embedder=_TinyEmbedder(), minimum_relevance=0, tracer=telemetry.tracer)
        )

    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: task.spec.name, SPAN_KIND: "benchmark"},
        ):
            samples = asyncio.run(
                run_loaded_task(
                    task,
                    run=BenchmarkRun(tmp_path / "stores", task.spec.name, "run"),
                    memory_factory=cast(MemoryFactory, memory_factory),
                    batch_size=4,
                    unit_concurrency=1,
                    request_concurrency=1,
                    recall_limit=5,
                    arms=(
                        _Arm("compile", generator=cast(eval_module._BaselineGenerator, generator)),
                    ),
                    compile_budget=ContextBudget(max_items=5, max_chars=2_000),
                    ingest_mode="capture",
                    tracer=telemetry.tracer,
                )
            )
        performance = telemetry.result(task.spec.name, arm="compile", question_count=1)
    finally:
        telemetry.close()

    assert len(samples) == 1
    sample = samples[0]
    assert sample.arm == "compile"
    assert sample.error_code is None
    assert sample.ingest_failure_count == 0
    assert sample.prediction == "Ada"
    assert sample.metrics["compile_bundle_items"] >= 1.0

    capture = cast(Mapping[str, object], performance["capture"])
    assert cast(int, capture["count"]) >= 1
    formation = cast(Mapping[str, object], performance["formation"])
    assert cast(int, formation["count"]) >= 1
    time_to_searchable = cast(Mapping[str, object], performance["time_to_searchable_ms"])
    assert cast(int, time_to_searchable["count"]) >= 1


def _sample_with(
    metrics: dict[str, float], *, arm: str = DEFAULT_ARM, task: str = "atm-bench"
) -> SampleResult:
    return SampleResult(
        task=task,
        benchmark="ATM-Bench",
        dataset_sha256="1" * 64,
        evaluation_sha256="2" * 64,
        unit_id="unit",
        question_id="q1",
        prediction="Ada",
        parsed_choice=None,
        score=metrics.get("accuracy"),
        exact_match=None,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        arm=arm,
        metrics=metrics,
        scorer_protocol="atm_bench_scorer_ef4e5dff1a47_minimal_v1",
        judge_model="gpt-5-mini",
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("accuracy", True),
        ("retrieval_recall@10", False),
        ("retrieval_hit@1", False),
        ("joint_strict@5", False),
        ("joint_partial@5", False),
    ],
)
def test_invented_metrics_are_never_stamped_official(metric: str, expected: bool) -> None:
    task, _, _ = _task()
    # `_metrics` reads `recall_limit` for the retrieval cutoffs and `blind` for the
    # controls block; both arrived with the arms merge. `_Arguments` bounds
    # `recall_limit` to 1..100, so 10 is a value the real parser would accept.
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(seed=7, bootstrap_samples=20, recall_limit=10, blind=False),
    )
    sample = _sample_with({metric: 1.0})

    result = _metrics(task, (sample,), arguments)
    rows = cast(dict[str, dict[str, object]], result["metrics"])

    assert metric_is_official("atm-bench", metric, "gpt-5-mini", uses_judge=False) is expected
    assert rows[metric]["official_metric"] is expected


def test_a_baseline_arm_never_reports_an_official_metric() -> None:
    task, _, _ = _task()
    # `_metrics` reads `recall_limit` for the retrieval cutoffs and `blind` for the
    # controls block; both arrived with the arms merge. `_Arguments` bounds
    # `recall_limit` to 1..100, so 10 is a value the real parser would accept.
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(seed=7, bootstrap_samples=20, recall_limit=10, blind=False),
    )
    sample = _sample_with({"accuracy": 1.0}, arm="full-context")

    result = _metrics(task, (sample,), arguments, arm="full-context")
    rows = cast(dict[str, dict[str, object]], result["metrics"])

    assert result["arm"] == "full-context"
    assert result["official_metric"] is False
    assert rows["accuracy"]["official_metric"] is False


def test_budget_loss_is_joined_onto_the_sample_that_lost_it() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with (
            eval_module._answer_span(telemetry.tracer, "atm-bench", "atm-bench/unit/q1"),
            telemetry.tracer.start_as_current_span(
                "model",
                attributes={
                    "mindbridge.span.kind": "model",
                    "mindbridge.grounding.dropped_hits": 4,
                    "mindbridge.grounding.media_elided_hits": 1,
                },
            ),
        ):
            pass
        grounding = telemetry.sample_grounding("atm-bench/unit/q1")
        assert grounding is not None
        assert (grounding.dropped_hits, grounding.media_elided_hits) == (4, 1)
        assert telemetry.sample_grounding("atm-bench/unit/q2") is None

        joined = _with_grounding_loss(
            (_sample_with({"accuracy": 1.0}), _sample_with({"accuracy": 1.0}, arm="blind")),
            telemetry,
        )

        assert joined[0].dropped_hits == 4
        assert joined[1].dropped_hits is None
    finally:
        telemetry.close()


def test_recall_planning_activation_is_aggregated_per_task_and_stamped_per_sample() -> None:
    """A run has to say whether the planner ran, in what shape, and how often it fell back.

    Every planning failure resolves to the fallback plan on purpose, so a result document that
    reports only scores cannot distinguish "planning is off" from "planning ran and decided
    nothing": the r0910 R1 arm looked like an active planner and had made no plan call at all.
    """
    telemetry = EvaluationTelemetry()
    plans: tuple[tuple[str, dict[str, AttributeValue]], ...] = (
        (
            "atm-bench/unit/q1",
            {
                RECALL_SHAPE: "set",
                RECALL_EXHAUSTIVE_ROWS: 12,
                RECALL_COMPLETE: False,
                RECALL_NON_SELECTIVE_STEPS: 1,
            },
        ),
        (
            "atm-bench/unit/q2",
            {
                RECALL_SHAPE: "point",
                RECALL_EXHAUSTIVE_ROWS: 0,
                RECALL_COMPLETE: True,
                RECALL_FALLBACK: True,
            },
        ),
        (
            "atm-bench/unit/q2",
            {
                RECALL_SHAPE: "set",
                RECALL_EXHAUSTIVE_ROWS: 6,
                RECALL_COMPLETE: False,
                RECALL_REPLAN: True,
            },
        ),
    )
    try:
        for sample_id, attributes in plans:
            with (
                eval_module._answer_span(telemetry.tracer, "atm-bench", sample_id),
                telemetry.tracer.start_as_current_span(
                    RECALL_SPAN,
                    attributes={SPAN_KIND: "stage", **attributes},
                ),
            ):
                pass

        recall = cast(
            dict[str, object],
            cast(dict[str, object], telemetry.result("atm-bench", question_count=2))["recall"],
        )
        assert recall["plan_count"] == 3
        assert recall["shapes"] == {"point": 1, "set": 2}
        assert recall["fallback_count"] == 1
        assert recall["replan_count"] == 1
        assert recall["incomplete_count"] == 2
        # A predicate that matched most of the corpus contributed no rows, which is a
        # different failure from a read that filled its bound and has to be visible as one.
        assert recall["non_selective_steps"] == 1
        rows = cast(dict[str, object], recall["exhaustive_rows"])
        assert rows["average"] == pytest.approx(6.0)

        # The last plan a sample ran is the one its answer was grounded on.
        assert telemetry.sample_grounding("atm-bench/unit/q1") == SampleGrounding(0, 0, "set")
        joined = _with_grounding_loss(
            (
                _sample_with({"accuracy": 1.0}),
                _sample_with({"accuracy": 1.0}, arm="blind"),
            ),
            telemetry,
        )
        assert joined[0].recall_shape == "set"
        assert joined[1].recall_shape is None
    finally:
        telemetry.close()


def test_answer_span_attributes_name_the_task_and_the_sample() -> None:
    telemetry = EvaluationTelemetry()
    seen: list[dict[str, object]] = []
    try:
        with eval_module._answer_span(
            telemetry.tracer, "atm-bench", "blind:atm-bench/unit/q1"
        ) as _:
            # `get_current_span` is typed as the write-only `Span`; the SDK object it returns
            # at runtime is a `ReadableSpan`, which is where `attributes` lives.
            span = cast(ReadableSpan, trace.get_current_span())
            seen.append(dict(span.attributes or {}))
    finally:
        telemetry.close()

    assert seen[0][BENCHMARK_TASK] == "atm-bench"
    assert seen[0][BENCHMARK_SAMPLE] == "blind:atm-bench/unit/q1"


def test_results_report_each_arm_beside_the_product_arm() -> None:
    task, _, _ = _task("atm-bench-main", labelled=False)
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            arms=(DEFAULT_ARM, "blind", "random"),
            answer_policy=None,
            full_context_chars=24_000,
            ingest="add",
            deliberate=False,
            compile_max_items=24,
            compile_max_chars=16_000,
            seed=7,
            seeds=(7, 7, 7, 7),
            bootstrap_samples=20,
            # `_results` records the baseline document it was compared against; None is the
            # in-run case, where the blind arm itself satisfies the control. `blind` reaches
            # `_controls` as `is_blind_run`: this fake selects arms explicitly rather than via
            # `--blind`, so it is False.
            blind_baseline=None,
            blind=False,
            tasks=(task.spec.name,),
            benchmarks_root=Path("root"),
            data_root=Path("data"),
            media_overrides={},
            media_manifest=None,
            limit=None,
            offset=0,
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=20,
            predict_only=False,
            num_fewshot=0,
            log_samples=False,
            use_cache=None,
            resume=False,
            allow_unverified_data=False,
            run_id="run",
            model=DEFAULT_ARM,
            model_args="",
            gen_kwargs="",
            device=None,
        ),
    )
    samples = (
        _sample_with({"accuracy": 1.0, "retrieval_recall@10": 1.0}, task="atm-bench-main"),
        _sample_with({"accuracy": 0.0}, arm="blind", task="atm-bench-main"),
        _sample_with({}, arm="random", task="atm-bench-main"),
    )

    results = eval_module._results(
        arguments,
        ModelConfig(),
        eval_module._JudgeConfig(model="gpt-5-mini", base_url="http://judge/v1"),
        (task,),
        samples,
        1.0,
        {task.spec.name: 1},
        {
            task.spec.name: {
                DEFAULT_ARM: {
                    "duration_seconds": {"total": 1.0, "average": 1.0},
                    "search_e2e": {"complete": False},
                    "token_usage": {"total_tokens": 10, "average_tokens": 10.0},
                },
                "blind": {
                    "duration_seconds": {"total": 0.5, "average": 0.5},
                    "token_usage": {"total_tokens": 4, "average_tokens": 4.0},
                },
                "random": {
                    "duration_seconds": {"total": 0.25, "average": 0.25},
                    "token_usage": {"total_tokens": 0, "average_tokens": 0.0},
                },
            }
        },
    )
    rows = cast(list[dict[str, object]], results["tasks"])
    arms = cast(dict[str, object], results["arms"])
    table = eval_module._table(results)

    assert [row["arm"] for row in rows] == [DEFAULT_ARM, "blind", "random"]
    assert [row["score"]["mean"] for row in rows] == [1.0, 0.0, None]  # type: ignore[index]
    assert rows[0]["official_metric"] is True
    assert rows[1]["official_metric"] is False
    assert arms["selected"] == [DEFAULT_ARM, "blind", "random"]
    assert arms["retrieval_candidate_limit"] == 60
    assert arms["retrieval_candidate_limit_arm"] == DEFAULT_ARM
    assert arms["search_e2e_limit"] == arguments.recall_limit
    definitions = cast(dict[str, dict[str, object]], arms["definitions"])
    assert set(definitions) == {DEFAULT_ARM, "blind", "random"}
    assert definitions[DEFAULT_ARM]["retrieval_candidate_limit"] == 60
    assert definitions[DEFAULT_ARM]["answer_retrieval_candidate_limit"] == 60
    assert definitions[DEFAULT_ARM]["answer_surface"] == {"atm-bench-main": "ask"}
    assert (
        definitions[DEFAULT_ARM]["compile_surface_prompt"]
        == eval_module.COMPILE_SURFACE_PROMPT_VERSION
    )
    assert cast(str, definitions[DEFAULT_ARM]["retrieval"]).startswith(
        "Memory.ask in-answer ranked list; no second scoring search"
    )
    assert definitions["blind"]["prompt"] == eval_module.BLIND_PROMPT_VERSION
    assert definitions["blind"]["official_metrics"] is False
    retrieval = {row["arm"]: cast(dict[str, object], row["retrieval"]) for row in rows}
    assert retrieval[DEFAULT_ARM]["retrieval_candidate_limit"] == 60
    assert retrieval["random"]["retrieval_candidate_limit"] == arguments.recall_limit
    assert "retrieval_candidate_limit" not in retrieval["blind"]
    assert results["status"] == "completed_with_errors"
    assert eval_module._execution_has_errors(samples, results)
    assert "atm-bench-main [blind]" in table


def test_answer_retrieval_candidate_limit_matches_budget_policy() -> None:
    budgeted = MindBridgeConfig(
        embedding=OpenAIEmbeddingConfig(provider="openai"),
        settings=MemoryConfig(evidence_budget_chars=4_242),
    )

    assert eval_module._answer_retrieval_candidate_limit(1, None) == 3
    assert eval_module._answer_retrieval_candidate_limit(20, None) == 60
    assert eval_module._answer_retrieval_candidate_limit(50, None) == RETRIEVAL_CANDIDATE_LIMIT
    assert eval_module._answer_retrieval_candidate_limit(1, budgeted) == RETRIEVAL_CANDIDATE_LIMIT


def test_baseline_generator_uses_the_configured_generation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    requests: list[dict[str, object]] = []

    class Completions:
        async def create(self, **request: object) -> object:
            requests.append(request)
            message = SimpleNamespace(content=" Ada ")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    class Client:
        def __init__(self, **options: object) -> None:
            requests.append({"client": options})
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    generator = eval_module._BaselineGenerator(
        ModelConfig(
            generation_model="gpt-5-mini",
            generation_base_url="http://generate/v1",
            generation_api_key="secret",
        ),
        seed=11,
        gen_kwargs="max_tokens=64,enable_thinking=false",
    )

    blind = asyncio.run(generator.answer("who signed it?", None))
    stuffed = asyncio.run(generator.answer("who signed it?", "Ada signed the contract"))
    stuffed_with_text_hit = asyncio.run(
        generator.answer(
            "who signed it?",
            "Ada signed the contract",
            evidence_hits=(
                SearchHit(
                    id="text-only",
                    content="Ada signed the contract",
                    score=1.0,
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ),
        )
    )
    asyncio.run(generator.close())

    assert requests[0]["client"] == {
        "api_key": "secret",
        "base_url": "http://generate/v1",
        "timeout": DEFAULT_TIMEOUT_SECONDS,
    }
    assert blind == stuffed == stuffed_with_text_hit == "Ada"
    for request in requests[1:]:
        assert request["model"] == "gpt-5-mini"
        assert request["temperature"] == 0.0
        assert request["seed"] == 11
        assert request["max_tokens"] == 64
        assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    messages = cast(list[dict[str, str]], requests[1]["messages"])
    assert messages[0]["content"] == eval_module._BLIND_SYSTEM_PROMPT
    assert messages[1]["content"] == "who signed it?"
    stuffed_messages = cast(list[dict[str, str]], requests[2]["messages"])
    assert stuffed_messages[0]["content"] == eval_module._FULL_CONTEXT_SYSTEM_PROMPT
    assert "Ada signed the contract" in stuffed_messages[1]["content"]
    assert requests[3] == requests[2]


def test_a_failed_answer_keeps_retrieval_and_scores_the_joint_metrics_zero() -> None:
    """A question the run never answered is a question it got wrong.

    `joint_*` is accuracy times recall, so it follows the answer to zero while the separately
    measured recall stays; the failure is still counted in `error_count`.
    """
    ranked = (_hit("gold-1", 0.9),)
    memory = _RankedMemory(ranked, ranked)
    task, unit, question = _task()
    # `joint_*` exists only where the family scores an answer deterministically beside recall.
    question = replace(question, metadata={**question.metadata, "qtype": "list_recall"})
    outcome = replace(_outcome(memory, question, arm=PRODUCT_ARM), error=ModelError("boom"))

    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=False,
        log_samples=False,
        arm=PRODUCT_ARM,
    )

    assert sample.error_code is not None
    assert sample.score == 0.0
    assert sample.metrics["retrieval_recall@5"] == 1.0
    joint = {name: value for name, value in sample.metrics.items() if name.startswith("joint_")}
    assert joint and set(joint.values()) == {0.0}


def test_a_sample_with_no_ranked_list_carries_no_retrieval_metrics() -> None:
    """The per-sample metrics and the task's `retrieval` block must share one denominator.

    `_retrieval_quality` excludes a labelled question that recorded no ranked list; a sample
    that still carried `retrieval_recall@10 = 0` made the same task publish two recalls.
    """
    task, unit, question = _task()
    outcome = replace(
        _outcome(_RankedMemory((), (_hit("gold-1", 0.9),)), question, arm=PRODUCT_ARM),
        ranked_source_ids=(),
        ranked_source_ids_complete=False,
    )

    sample = _sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=False,
        log_samples=False,
        arm=PRODUCT_ARM,
    )

    assert sample.ranked_source_ids == ()
    assert not [name for name in sample.metrics if name.startswith(("retrieval_", "joint_"))]
    assert (
        eval_module._retrieval_quality((sample,), seed=7, bootstrap_samples=8, recall_limit=100)[
            "unranked_labelled_question_count"
        ]
        == 1
    )


def test_a_replayed_answer_still_carries_the_ranked_list_it_was_scored_from(
    tmp_path: Path,
) -> None:
    """`--use-cache` must not silently delete a run's retrieval numbers.

    Recall is measured over the ranked list, so a cache that stored only the answer replayed as
    a labelled question with no ranked list and the whole run reported `recall_at_k = {}`.
    """
    ranked = (_hit("noise", 0.9), _hit("gold-1", 0.5))
    memory = _RankedMemory(ranked, (_hit("gold-1", 0.5),))
    task, unit, question = _task()
    outcome = _outcome(memory, question, arm=PRODUCT_ARM)

    cache = ResponseCache(tmp_path / "responses", "run-a", "namespace")
    try:
        eval_module._cache_outcome(cache, task, unit, question, outcome, 0)
        replayed = eval_module._cached_results(
            cache, task, unit, predict_only=False, log_samples=False
        )
    finally:
        cache.close()

    sample = replayed[(PRODUCT_ARM.name, question.question_id)]
    assert sample.cached is True
    assert sample.ranked_source_ids == ("noise", "gold-1")
    assert sample.metrics["retrieval_recall@5"] == 1.0
    retrieval = eval_module._retrieval_quality(
        (sample,), seed=7, bootstrap_samples=8, recall_limit=100
    )
    recall = cast(dict[str, dict[str, float]], retrieval["recall_at_k"])
    assert recall["5"]["mean"] == pytest.approx(1.0)
    assert retrieval["unranked_labelled_question_count"] == 0
    # The bound recall was measured under, not the `--recall-limit` that bounds `ask`.
    assert retrieval["retrieval_candidate_limit"] == RETRIEVAL_CANDIDATE_LIMIT


@pytest.mark.parametrize(
    ("task_name", "override", "expected"),
    [
        ("m3-bench-robot", None, "best_effort"),
        ("mm-lifelong-day-test", None, "best_effort"),
        ("atm-bench", None, "strict"),
        ("atm-bench", "best_effort", "best_effort"),
        ("m3-bench-robot", "strict", "strict"),
    ],
)
def test_the_task_policy_reaches_the_lent_answerer_through_the_real_harness_path(
    tmp_path: Path, task_name: str, override: str | None, expected: str
) -> None:
    """The whole chain: `_arm_answer` -> `AsyncMemory.ask_stream` -> `_BorrowedGenerationBackend`.

    The harness lends one answerer to every per-unit memory, so the borrowed wrapper is the last
    hop before the provider. It once declared the keyword and dropped it, which made the whole
    arm a silent no-op that no fake-`ask` test could see. The same wrapper forwards `exhaustive`,
    which is why this answerer accepts it: a wrapper that dropped an argument the protocol
    declares would fail the call instead of the arm.
    """
    recorded: list[str] = []

    class Answerer:
        generation_capabilities = _ATOMIC_MODALITIES
        generation_model = "fake-generator"

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: str = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            del question
            recorded.append(answer_policy)
            return AnswerResult("Ada.", tuple(hits))

        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: str = "strict",
            exhaustive: bool = False,
        ) -> Generator[str, None, tuple[SearchHit, ...]]:
            del question
            recorded.append(answer_policy)
            yield "Ada."
            return tuple(hits)

        def close(self) -> None:
            return None

    borrowed = eval_module._BorrowedGenerationBackend(Answerer())
    _, _, question = _task()

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(
            Memory(
                tmp_path,
                embedder=_TinyEmbedder(),
                answerer=cast(GenerationBackend, borrowed),
                minimum_relevance=0,
            )
        ) as memory:
            await memory.add("Ada signed the contract")
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=5,
                arm=PRODUCT_ARM,
                task_name=task_name,
                unit_id="unit",
                answer_policy=cast(AnswerPolicy | None, override),
            )
            return answered[0]

    outcome = asyncio.run(run())
    assert not isinstance(outcome, BaseException)
    assert outcome.prediction == "Ada."
    assert recorded == [expected]

    # The buffered half of the same wrapper, which a non-streaming provider takes.
    recorded.clear()
    borrowed.answer(ModelInput("who signed it?"), (), answer_policy="best_effort")
    assert recorded == ["best_effort"]


def test_the_lent_answerer_declares_the_planning_capability_its_pool_actually_has() -> None:
    """`isinstance` against a `runtime_checkable` protocol reads attributes statically.

    So a `plan_recall` reachable only through the proxy's `__getattr__` was invisible to
    `Memory._recall_plan`, which fell back to the point plan for every question under the
    harness: measured, one generation call per question and the baseline's own hits. The
    declaration has to follow the pooled backend both ways -- absent when the pool cannot plan,
    or route-by-capability would claim a capability no one has.
    """
    pooled = OpenAIModels(OpenAI(api_key="test-key", base_url="http://127.0.0.1:1/v1"))
    lent = eval_module._BorrowedGenerationBackend(pooled)

    assert isinstance(lent, RecallPlanningBackend)
    assert isinstance(lent, StreamingGenerationBackend)

    class Unplanned:
        generation_capabilities = _ATOMIC_MODALITIES

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: str = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            del question, answer_policy, exhaustive
            return AnswerResult("Ada.", tuple(hits))

        def close(self) -> None:
            return None

    bare = eval_module._BorrowedGenerationBackend(Unplanned())
    assert isinstance(bare, GenerationBackend)
    assert not isinstance(bare, RecallPlanningBackend)
    assert not isinstance(bare, StreamingGenerationBackend)


def test_the_planner_runs_once_per_question_through_the_real_harness_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole chain: `_arm_answer` -> `AsyncMemory.ask_stream` -> the borrowed proxy.

    `recall_planning` is only observable through the plan call, and the fallback is silent by
    design, so a hidden capability reads exactly like a planner that chose a point plan. This
    asserts the call happened, and the trace attribute says it was the planner's plan rather
    than the fallback.
    """
    planned: list[str] = []

    class Planner:
        generation_capabilities = _ATOMIC_MODALITIES
        generation_model = "fake-planner"

        def plan_recall(
            self,
            question: str,
            *,
            reference_at: datetime,
            corpus_digest: str,
            attempted: str = "",
        ) -> str | None:
            del reference_at, corpus_digest, attempted
            planned.append(question)
            return json.dumps({"shape": "set", "steps": [{"op": "match", "terms": ["signed"]}]})

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: str = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            del question, answer_policy, exhaustive
            return AnswerResult("Ada.", tuple(hits))

        def close(self) -> None:
            return None

    borrowed = eval_module._BorrowedGenerationBackend(Planner())
    _, _, question = _task()
    telemetry = EvaluationTelemetry()

    async def run() -> eval_module._AnswerOutcome | BaseException:
        async with AsyncMemory(
            Memory(
                tmp_path,
                embedder=_TinyEmbedder(),
                answerer=cast(GenerationBackend, borrowed),
                minimum_relevance=0,
                recall_planning=True,
                tracer=telemetry.tracer,
            )
        ) as memory:
            await memory.add("Ada signed the contract")
            answered = await _answer_many(
                memory,
                (question,),
                request_concurrency=1,
                recall_limit=5,
                arm=PRODUCT_ARM,
                task_name="atm-bench",
                unit_id="unit",
                tracer=telemetry.tracer,
            )
            return answered[0]

    try:
        with caplog.at_level(logging.WARNING, logger="mindbridge.memory"):
            outcome = asyncio.run(run())
        # The store saw a planner at wiring time, not a proxy it had to fall back around.
        assert not [message for message in caplog.messages if "cannot plan" in message]
        assert not isinstance(outcome, BaseException)
        assert outcome.prediction == "Ada."
        assert planned == [str(question.content[0])]
        recall = cast(
            dict[str, object],
            cast(dict[str, object], telemetry.result("atm-bench", question_count=1))["recall"],
        )
        assert recall["plan_count"] == 1
        assert recall["shapes"] == {"set": 1}
        assert recall["fallback_count"] == 0
        # The recall stage runs inside the answer span, so its shape reaches the sample row.
        grounding = telemetry.sample_grounding("atm-bench/unit/q1")
        assert grounding is not None
        assert grounding.recall_shape == "set"
    finally:
        telemetry.close()
