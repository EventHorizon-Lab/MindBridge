"""`mindbridge-bench eval --deliberate`: the harness actually running the loop it configured.

A configuration file's `consolidation` section already built a consolidator, registered it for
close, and never called it. These pin the flag that calls it, where in the unit lifecycle it
runs, and what the run report then says about it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from mindbridge import AnswerPolicy, AnswerResult, AsyncMemory, DeliberationReport
from mindbridge.benchmarks import eval as eval_module
from mindbridge.benchmarks.eval import (
    MemoryFactory,
    run_loaded_task,
)
from mindbridge.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
    MemoryItem,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.task_catalog import TaskSpec

_CONFIG = """
generation:
  provider: openai
  api_key: unused
  base_url: http://127.0.0.1:9/v1
"""


class _Memory:
    """Records the order the harness drives one unit in."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def add_many(
        self,
        contents: Sequence[object],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        self.events.append(f"add:{len(contents)}")
        return ()

    async def add(self, content: object, **_kwargs: object) -> object:
        self.events.append(f"add:{content}")
        return object()

    async def deliberate(self, **_kwargs: object) -> DeliberationReport:
        self.events.append("deliberate")
        return DeliberationReport(rounds=1, weighed=1, applied=3)

    async def ask(
        self,
        question: object,
        *,
        limit: int,
        reference_at: datetime | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult:
        del reference_at, limit, answer_policy
        self.events.append(f"ask:{question}")
        return AnswerResult("A")


def _task(tmp_path: Path) -> LoadedTask:
    spec = TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40)
    return LoadedTask(
        spec,
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                tuple(MemoryItem(str(number), (str(number),)) for number in range(2)),
                (EvalQuestion("q1", ("question",), references=("answer",)),),
            ),
        ),
    )


async def _run(tmp_path: Path, *, deliberate: bool) -> list[str]:
    events: list[str] = []

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return _memory

        async def __aexit__(self, *_error: object) -> None:
            return None

    _memory = cast(AsyncMemory, _Memory(events))
    await run_loaded_task(
        _task(tmp_path),
        run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
        memory_factory=cast(MemoryFactory, lambda _path: Context()),
        batch_size=4,
        unit_concurrency=1,
        request_concurrency=1,
        recall_limit=1,
        deliberate=deliberate,
    )
    return events


@pytest.mark.asyncio
async def test_the_loop_runs_after_ingest_and_before_the_questions(tmp_path: Path) -> None:
    eval_module._deliberation_applied = 0

    events = await _run(tmp_path, deliberate=True)

    assert events == ["add:2", "deliberate", "ask:question"]
    assert eval_module._deliberation_applied == 3


@pytest.mark.asyncio
async def test_the_loop_is_off_by_default(tmp_path: Path) -> None:
    eval_module._deliberation_applied = 0

    events = await _run(tmp_path, deliberate=False)

    assert "deliberate" not in events
    assert eval_module._deliberation_applied == 0


def test_deliberate_is_refused_without_a_configured_consolidation_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG, encoding="utf-8")

    with pytest.raises(SystemExit) as refusal:
        eval_module.main(
            [
                "--tasks",
                "locomo-refined",
                "--config",
                str(config),
                "--deliberate",
                "--output-path",
                str(tmp_path / "out"),
            ]
        )

    assert refusal.value.code == 2
    assert "--deliberate requires" in capsys.readouterr().err
