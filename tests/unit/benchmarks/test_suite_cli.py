"""Sweep behaviour for the multi-benchmark `suite` runner.

Each benchmark's own arguments are covered where that benchmark's CLI is. What is checked here
is what the sweep adds: the invocation it builds per task, the isolation it derives, and what it
records when one task fails, is interrupted, or exits 0 without leaving a result.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from mindbridge import telemetry
from mindbridge.benchmarks.cli import INTERRUPT_EXIT_CODE, RUNNERS, USAGE_EXIT_CODE, Runner
from mindbridge.benchmarks.suite import (
    SUMMARY_FILENAME,
    BenchmarkSuite,
    SuiteRunSummary,
    SuiteTask,
    main,
)
from mindbridge.benchmarks.task_catalog import ROOT, TASKS, CatalogTask, listing, task_payloads


def test_the_dispatcher_can_reach_the_suite_runner() -> None:
    assert RUNNERS["suite"].module == "mindbridge.benchmarks.suite"
    assert RUNNERS["suite"].extra is None


def test_each_task_gets_its_own_run_id_and_output_so_one_benchmark_can_run_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parameterisations of one benchmark share a tenant prefix and unit IDs.

    Only the run ID separates their tenants, so a sweep that forwarded its own would have the
    second task answering from the first task's memories.
    """
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(
        tmp_path,
        _task("first", arguments=["--split", "main"]),
        _task("second", arguments=["--split", "hard"]),
    )

    assert _sweep(suite, tmp_path) == 0

    run_ids = [_flag_value(argv, "--run-id") for argv in invocations]
    outputs = [_flag_value(argv, "--output") for argv in invocations]
    assert run_ids == ["sweep-01-first", "sweep-01-second"]
    assert outputs == [str(tmp_path / "first.jsonl"), str(tmp_path / "second.jsonl")]
    assert [_flag_value(argv, "--split") for argv in invocations] == ["main", "hard"]


def test_a_task_may_override_a_shared_tunable_but_not_the_flags_the_sweep_owns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared flags come first, so a benchmark needing its own recall budget can say so."""
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("wide", arguments=["--recall-limit", "50"]))

    assert _sweep(suite, tmp_path, "--recall-limit", "20") == 0

    argv = invocations[0]
    positions = [index for index, item in enumerate(argv) if item == "--recall-limit"]
    assert positions[0] < argv.index("--dataset"), "the sweep's copy must come first"
    assert argv[positions[-1] + 1] == "50", "argparse keeps the last value, so the task wins"


def test_only_the_shared_flags_that_were_given_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset tunable stays out of the argv, so each runner keeps its own default.

    A sweep that defaulted these itself would pin a copy of every runner's default and go stale
    the moment one of them changed.
    """
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("plain"))

    assert _sweep(suite, tmp_path) == 0

    assert "--recall-limit" not in invocations[0]
    assert "--request-concurrency" not in invocations[0]
    assert "--request-timeout-seconds" not in invocations[0]


def test_a_sweep_records_every_task_and_keeps_going_past_one_that_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benchmark dying hours in must cost its own result, not the whole sweep."""

    def handler(argv: Sequence[str], *, prog: str) -> int:
        if "--split" in argv and argv[argv.index("--split") + 1] == "hard":
            raise RuntimeError("upstream refused the request")
        _write_predictions(argv)
        return 0

    _stub_benchmark(monkeypatch, handler=handler)
    suite = _suite_file(
        tmp_path,
        _task("main", arguments=["--split", "main"]),
        _task("hard", arguments=["--split", "hard"]),
        _task("late", arguments=["--split", "main"]),
    )

    assert _sweep(suite, tmp_path) == 1

    summary = _summary(tmp_path)
    assert summary.completed_task_count == 2
    assert summary.failed_task_count == 1
    assert [(task.name, task.status) for task in summary.tasks] == [
        ("main", "completed"),
        ("hard", "failed"),
        ("late", "completed"),
    ]
    assert summary.task_count == 3


def test_a_task_whose_flags_argparse_rejects_does_not_end_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """argparse exits by raising `SystemExit`, which the shared failure contract does not catch.

    Unguarded it leaves through the sweep, so one mistyped task would end the run and take the
    summary of every task before it along with it.
    """

    def handler(argv: Sequence[str], *, prog: str) -> int:
        if "--split" in argv:
            raise SystemExit(2)
        _write_predictions(argv)
        return 0

    _stub_benchmark(monkeypatch, handler=handler)
    suite = _suite_file(tmp_path, _task("mistyped", arguments=["--split", "x"]), _task("later"))

    assert _sweep(suite, tmp_path) == 1

    summary = _summary(tmp_path)
    assert [(task.name, task.status, task.exit_code) for task in summary.tasks] == [
        ("mistyped", "failed", 2),
        ("later", "completed", 0),
    ]


def test_an_interrupt_stops_the_sweep_and_still_writes_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C must not fall through into the next benchmark's several hours of work."""

    def handler(argv: Sequence[str], *, prog: str) -> int:
        raise KeyboardInterrupt

    _stub_benchmark(monkeypatch, handler=handler)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))

    assert _sweep(suite, tmp_path) == INTERRUPT_EXIT_CODE

    summary = _summary(tmp_path)
    assert summary.task_count == 2
    assert [task.name for task in summary.tasks] == ["first"]


def test_a_task_that_exits_zero_without_predictions_is_not_reported_as_a_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary is an audit artifact; a result nobody can open may not be called one."""
    _stub_benchmark(monkeypatch, writes=False)
    suite = _suite_file(tmp_path, _task("silent"))

    assert _sweep(suite, tmp_path) == 1

    summary = _summary(tmp_path)
    assert summary.tasks[0].status == "failed"
    assert summary.tasks[0].exit_code == 0
    assert summary.failed_task_count == 1


def test_each_task_reports_its_own_cost_rather_than_the_sweep_so_far(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher logs a ranked per-operation breakdown after every task it runs.

    That accumulator is process-wide, so unless the sweep clears it between tasks each task's
    breakdown carries every earlier task's work under its own label -- and which stage owned a
    task's wall clock is the one thing about it nobody can reconstruct from the artifacts.
    """
    monkeypatch.setattr(telemetry, "_timings", {})

    def handler(argv: Sequence[str], *, prog: str) -> int:
        @telemetry.operation_span(f"mindbridge.test.{Path(_flag_value(argv, '--output')).stem}")
        async def instrumented() -> None:
            return None

        asyncio.run(instrumented())
        _write_predictions(argv)
        return 0

    _stub_benchmark(monkeypatch, handler=handler)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))

    assert _sweep(suite, tmp_path) == 0

    assert [row.operation for row in telemetry.timing_summary()] == ["mindbridge.test.second"]


def test_a_sweep_refuses_to_start_when_any_of_its_outputs_already_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each runner makes this check for itself, but only once it is already running."""
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))
    (tmp_path / "second.jsonl").write_text("earlier run", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"second\.jsonl"):
        _sweep(suite, tmp_path)

    assert invocations == []


def test_a_dry_run_prints_each_invocation_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The printed run IDs are how a sweep's tenants get authorized before it starts."""
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))

    assert _sweep(suite, tmp_path, "--dry-run") == 0

    printed = capsys.readouterr().out
    assert "sweep-01-first" in printed
    assert "sweep-01-second" in printed
    assert invocations == []
    assert not (tmp_path / SUMMARY_FILENAME).exists()


def test_a_named_task_narrows_the_sweep_without_reordering_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"), _task("second"), _task("third"))

    assert _sweep(suite, tmp_path, "--tasks", "third,first") == 0

    assert [_flag_value(argv, "--run-id") for argv in invocations] == [
        "sweep-01-first",
        "sweep-01-third",
    ]


def test_an_unknown_task_name_is_refused_before_anything_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"))

    with pytest.raises(ValueError, match="unknown suite task names: absent"):
        _sweep(suite, tmp_path, "--tasks", "absent")

    assert invocations == []


def test_a_suite_naming_a_benchmark_the_dispatcher_has_no_row_for_is_refused() -> None:
    with pytest.raises(ValidationError, match="unknown benchmark: not-a-benchmark"):
        BenchmarkSuite.model_validate(
            {"tasks": [{"name": "x", "benchmark": "not-a-benchmark", "arguments": []}]}
        )


@pytest.mark.parametrize("flag", ["--output", "-o", "--run-id", "--output=elsewhere.jsonl"])
def test_a_suite_may_not_set_the_flags_the_sweep_derives(flag: str) -> None:
    """They are appended last, so argparse would let the sweep's copy win silently."""
    with pytest.raises(ValidationError, match="the sweep derives"):
        SuiteTask.model_validate({"name": "x", "benchmark": "locomo-refined", "arguments": [flag]})


def test_two_tasks_may_not_share_a_name_or_a_predictions_file() -> None:
    """Sharing either is silent: one tenant for two tasks, or one file for two results."""
    with pytest.raises(ValidationError, match="suite task names must be unique: twice"):
        BenchmarkSuite.model_validate(
            {
                "tasks": [
                    {"name": "twice", "benchmark": "locomo-refined"},
                    {"name": "twice", "benchmark": "locomo-refined"},
                ]
            }
        )
    with pytest.raises(ValidationError, match=r"must not share a predictions file: shared\.json"):
        BenchmarkSuite.model_validate(
            {
                "tasks": [
                    {"name": "a", "benchmark": "locomo-refined", "output_name": "shared.json"},
                    {"name": "b", "benchmark": "locomo-refined", "output_name": "shared.json"},
                ]
            }
        )


@pytest.mark.parametrize("name", ["../escape", "sub/dir", ".hidden", ""])
def test_a_task_name_must_stay_inside_the_output_directory(name: str) -> None:
    """The name becomes a file name, so a suite file must not be able to write outside it."""
    with pytest.raises(ValidationError):
        SuiteTask.model_validate({"name": name, "benchmark": "locomo-refined"})


def test_every_catalog_task_is_one_the_sweep_can_actually_dispatch() -> None:
    """The catalog is the surface `--tasks` names, so a typo in it must fail here, not mid-sweep."""
    suite = BenchmarkSuite.model_validate({"tasks": task_payloads(["all"], root=Path("/corpus"))})

    assert len(suite.tasks) == len(TASKS)
    assert {task.benchmark for task in suite.tasks} <= set(RUNNERS)
    # Every dataset path has to come from --benchmarks-root, or the entry has a path baked in
    # that no flag can move.
    for task in suite.tasks:
        paths = [argument for argument in task.arguments if argument.startswith(("/", "."))]
        assert paths, f"{task.name} names no release"
        assert all(path.startswith("/corpus/") for path in paths), task.name


def test_the_released_text_group_needs_no_operator_produced_media() -> None:
    """It is the group whose numbers are a memory-layer claim, which is what makes it citable."""
    suite = BenchmarkSuite.model_validate(
        {"tasks": task_payloads(["released-text"], root=Path("/corpus"))}
    )

    assert [task.name for task in suite.tasks] == [
        "locomo-refined",
        "memlens-32k",
        "atm-main-sgm",
        "atm-hard-sgm",
    ]
    assert all("--prepared-media" not in task.arguments for task in suite.tasks)
    assert all("--prepared-images" not in task.arguments for task in suite.tasks)


def test_a_group_and_a_task_naming_the_same_benchmark_run_it_once_in_catalog_order() -> None:
    """Two names for one task would otherwise become two runs writing the same file."""
    payloads = task_payloads(["released-text", "locomo-refined", "m3-robot"], root=Path("/corpus"))

    names = [payload["name"] for payload in payloads]
    assert names == [
        "locomo-refined",
        "m3-robot",
        "memlens-32k",
        "atm-main-sgm",
        "atm-hard-sgm",
    ]


def test_an_unknown_catalog_name_says_how_to_find_the_real_ones() -> None:
    with pytest.raises(ValueError, match=r"unknown task: nope; `--list-tasks`"):
        task_payloads(["nope"], root=Path("/corpus"))


def test_the_listing_separates_a_task_whose_inputs_are_present_from_one_missing_them(
    tmp_path: Path,
) -> None:
    """`--list-tasks` is the answer to "which of these can I actually run right now"."""
    dataset = tmp_path / "locomo-refined" / "data" / "raw"
    dataset.mkdir(parents=True)
    (dataset / "locomo_refined.json").write_text("[]", encoding="utf-8")

    lines = {line.split()[0]: line for line in listing(root=tmp_path).splitlines() if line.strip()}

    assert lines["locomo-refined"].endswith("ready")
    # Both of this one's inputs are absent, and it names both rather than only the first.
    assert f"{tmp_path}/m3-agent/data/annotations/robot.json" in lines["m3-robot"]
    assert f"{tmp_path}/m3-prepared-robot.json" in lines["m3-robot"]


def test_the_catalog_path_needs_no_file_and_derives_where_everything_goes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one-line invocation: task names, a run ID, and a corpus root."""
    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    _stub_benchmark(monkeypatch, writes=True)

    assert (
        main(
            [
                "--tasks",
                "stub-task",
                "--run-id",
                "smoke-01",
                "--benchmarks-root",
                str(tmp_path),
                "--limit",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert f"--dataset {tmp_path}/stub/release.json" in printed
    assert "--limit 2" in printed
    assert f"--output {tmp_path}/results/smoke-01/stub-task.jsonl" in printed
    assert "--run-id smoke-01-stub-task" in printed
    assert f"--deployment-config {tmp_path}/deployment.json" in printed
    assert "--api-base-url http://localhost:8000" in printed


def test_naming_no_task_at_all_points_at_the_listing_rather_than_running_everything() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--run-id", "smoke-01"])

    assert exit_info.value.code == USAGE_EXIT_CODE


def test_a_sweep_without_a_run_id_is_refused_because_it_is_what_isolates_the_tenants() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--tasks", "locomo-refined"])

    assert exit_info.value.code == USAGE_EXIT_CODE


def _stub_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writes: bool = True,
    handler: Callable[..., object] | None = None,
) -> list[tuple[str, ...]]:
    """Register one dispatchable stand-in benchmark and record the argv it is called with."""
    invocations: list[tuple[str, ...]] = []

    def record(argv: Sequence[str], *, prog: str) -> object:
        invocations.append(tuple(argv))
        if handler is not None:
            return handler(argv, prog=prog)
        if writes:
            _write_predictions(argv)
        return None

    module = ModuleType("mindbridge_stub_benchmark")
    module.main = record  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mindbridge_stub_benchmark", module)
    monkeypatch.setitem(
        RUNNERS,
        "stub",
        Runner("mindbridge_stub_benchmark", "Stub benchmark", extra=None),
    )
    return invocations


def _write_predictions(argv: Sequence[str]) -> None:
    """Write what a real runner writes, so the sweep can see the task produced a result."""
    Path(_flag_value(argv, "--output")).write_text("{}\n", encoding="utf-8")


def _task(name: str, *, arguments: Sequence[str] = ()) -> dict[str, object]:
    return {
        "name": name,
        "benchmark": "stub",
        "arguments": ["--dataset", "release.json", *arguments],
    }


def _suite_file(directory: Path, *tasks: dict[str, object]) -> Path:
    path = directory / "suite.json"
    path.write_text(json.dumps({"tasks": list(tasks)}), encoding="utf-8")
    return path


def _sweep(suite: Path, output_dir: Path, *extra: str) -> int:
    """Run a sweep over a hand-written suite file, which is the escape hatch from the catalog."""
    return main(
        [
            "--suite",
            str(suite),
            "--output-dir",
            str(output_dir),
            "--api-base-url",
            "http://localhost:8000",
            "--deployment-config",
            str(suite.parent / "deployment.json"),
            "--run-id",
            "sweep-01",
            *extra,
        ]
    )


def _summary(output_dir: Path) -> SuiteRunSummary:
    return SuiteRunSummary.model_validate_json(
        (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    )


def _flag_value(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]
