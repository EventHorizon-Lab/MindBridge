"""Buffered SDK answers honor sixteen callers without changing stream admission limits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Barrier, Lock

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
from mindbridge.benchmarks import eval_config
from mindbridge.benchmarks.eval import run_loaded_task
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_config import _arguments, _build_parser
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import HarnessOverrides, RunOverrides
from mindbridge.benchmarks.task_catalog import TASKS


def test_buffered_answer_cli_overrides_the_default() -> None:
    parser = _build_parser("eval")
    assert _arguments(parser, parser.parse_args(["--tasks", "atm-bench-main"])).stream_answers
    assert not _arguments(
        parser, parser.parse_args(["--tasks", "atm-bench-main", "--no-stream-answers"])
    ).stream_answers


def test_buffered_answer_yaml_and_cli_precedence() -> None:
    parser = _build_parser("eval")
    overrides = HarnessOverrides(run=RunOverrides(stream_answers=False))
    default = parser.parse_args(["--tasks", "atm-bench-main"])
    streamed = parser.parse_args(["--tasks", "atm-bench-main", "--stream-answers"])
    assert not eval_config._arguments(parser, default, overrides=overrides).stream_answers
    assert eval_config._arguments(parser, streamed, overrides=overrides).stream_answers


async def test_buffered_runner_reaches_sixteen_real_sdk_answers(tmp_path: Path) -> None:
    class Models:
        embedding_capabilities = frozenset({Modality.TEXT})
        generation_capabilities = frozenset({Modality.TEXT})
        embedding_dimension = 4
        embedding_model = "buffered-fixture-v1"
        embedding_space = "buffered-fixture-v1"

        def __init__(self) -> None:
            self.barrier = Barrier(16)
            self.lock = Lock()
            self.active = self.peak = 0

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0, 0.0, 0.0, 0.0) for _ in inputs)

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.barrier.wait(timeout=10)
                return AnswerResult("blue", hits=tuple(hits))
            finally:
                with self.lock:
                    self.active -= 1

        def close(self) -> None:
            pass

    models = Models()

    def factory(directory: Path) -> AsyncMemory:
        return AsyncMemory(
            Memory(directory, embedder=models, answerer=models, reinforce_on_answer=False)
        )

    task = LoadedTask(
        TASKS["atm-bench-main"],
        tmp_path / "fixture.json",
        "0" * 64,
        (
            EvalUnit(
                "main",
                (MemoryItem("source", ("The toolbox is blue.",)),),
                tuple(
                    EvalQuestion(
                        f"q{index}",
                        ("What color is the toolbox?",),
                        ("blue",),
                        metadata={"qtype": "open_end", "evidence_ids": ["source"]},
                    )
                    for index in range(16)
                ),
            ),
        ),
    )
    samples = await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "atm-bench-main", "buffered"),
        memory_factory=factory,
        batch_size=1,
        unit_concurrency=1,
        request_concurrency=16,
        recall_limit=1,
        stream_answers=False,
    )
    assert models.peak == 16
    assert len(samples) == 16
    assert all(sample.error_code is None and sample.prediction == "blue" for sample in samples)
