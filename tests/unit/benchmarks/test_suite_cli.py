"""Sweep behaviour for the multi-benchmark `eval` runner.

Each benchmark's own arguments are covered where that benchmark's CLI is. What is checked here
is what the sweep adds: the invocation it builds per task, the isolation it derives, and what it
records when one task fails, is interrupted, or exits 0 without leaving a result.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest
from pydantic import ValidationError

from mindbridge import telemetry
from mindbridge.benchmarks.artifacts import sidecar_manifest_path
from mindbridge.benchmarks.atm_bench_runner import AtmMediaSource
from mindbridge.benchmarks.atm_cli import AtmSplit
from mindbridge.benchmarks.cli import INTERRUPT_EXIT_CODE, RUNNERS, USAGE_EXIT_CODE, Runner
from mindbridge.benchmarks.memlens_cli import MemLensContextWindow
from mindbridge.benchmarks.mm_lifelong import MMLifelongSplit
from mindbridge.benchmarks.official_score import score_sidecar_path
from mindbridge.benchmarks.prepare import PREPARERS, Producer
from mindbridge.benchmarks.suite import (
    SUMMARY_FILENAME,
    BenchmarkSuite,
    SuiteRunSummary,
    SuiteTask,
    main,
)
from mindbridge.benchmarks.task_catalog import ROOT, TASKS, CatalogTask, listing, task_payloads


def test_the_dispatcher_can_reach_the_suite_runner() -> None:
    assert RUNNERS["eval"].module == "mindbridge.benchmarks.suite"
    assert RUNNERS["eval"].extra is None
    assert "suite" not in RUNNERS, "the sweep is spelled `eval`; `suite` is the task-list file"


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
    assert outputs == [
        str(tmp_path / "first" / "predictions.jsonl"),
        str(tmp_path / "second" / "predictions.jsonl"),
    ]
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


def test_a_producer_backed_task_gets_a_manifest_path_of_its_own_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest names objects under one run's tenant, so a shared path is unreadable later.

    Shared, the first run's file was found on disk, preparation was skipped as already-done, and
    the run failed on a manifest describing another run's tenant.
    """
    prepared: list[tuple[str, ...]] = []
    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    monkeypatch.setitem(
        PREPARERS,
        "stub",
        Producer("--prepared-media", lambda request: prepared.append(request.argv)),
    )
    invocations = _stub_benchmark(monkeypatch, writes=True)

    for run in ("run-a", "run-b"):
        assert _catalog_sweep(tmp_path, run) == 0

    manifests = [_flag_value(argv, "--prepared-media") for argv in invocations]
    assert manifests == [
        str(tmp_path / "results" / "run-a" / "stub-task" / "run-a-prepared.json"),
        str(tmp_path / "results" / "run-b" / "stub-task" / "run-b-prepared.json"),
    ]
    # Preparation ran for both, because neither found the other's file.
    assert len(prepared) == 2


def test_two_runs_sharing_an_explicit_output_dir_still_prepare_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default output directory carries the run ID; `--output-dir` given explicitly does not.

    So the run ID has to be in the file name too. `--overwrite` clears the predictions and the
    summary but not the prepared manifest, and `_prepare_task` skips a manifest that exists --
    so a second sweep into the same directory ingested objects staged for the first run's tenant,
    which its own tenant cannot address. The default path hides this, which is why the test
    above cannot see it.
    """
    shared = tmp_path / "shared"
    prepared: list[tuple[str, ...]] = []
    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    monkeypatch.setitem(
        PREPARERS,
        "stub",
        Producer("--prepared-media", lambda request: prepared.append(request.argv)),
    )
    invocations = _stub_benchmark(monkeypatch, writes=True)

    for run in ("run-a", "run-b"):
        argv = [
            "--tasks",
            "stub-task",
            "--run-id",
            run,
            "--benchmarks-root",
            str(tmp_path),
            "--output-dir",
            str(shared),
            "--no-download",
        ]
        if run != "run-a":
            argv.append("--overwrite")
        assert main(argv) == 0

    manifests = [_flag_value(argv, "--prepared-media") for argv in invocations]
    assert manifests == [
        str(shared / "stub-task" / "run-a-prepared.json"),
        str(shared / "stub-task" / "run-b-prepared.json"),
    ]
    assert len(prepared) == 2


def test_a_suite_supplying_its_own_manifest_keeps_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overriding a hand-supplied manifest would silently prepare over the one asked for.

    That path is how a benchmark with no producer gets staged at all, and how a producer-backed
    one reuses media prepared elsewhere.
    """
    prepared: list[tuple[str, ...]] = []
    monkeypatch.setitem(
        PREPARERS,
        "stub",
        Producer("--prepared-media", lambda request: prepared.append(request.argv)),
    )
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(
        tmp_path,
        _task("mine", arguments=["--prepared-media", str(tmp_path / "mine.json")]),
    )
    (tmp_path / "mine.json").write_text("{}", encoding="utf-8")

    assert _sweep(suite, tmp_path) == 0

    # Once, not twice: appended as well, argparse would keep the sweep's copy and the task's
    # own would read as ignored rather than refused.
    assert invocations[0].count("--prepared-media") == 1
    assert _flag_value(invocations[0], "--prepared-media") == str(tmp_path / "mine.json")
    assert prepared == []


def test_a_preparation_that_fails_is_that_task_and_not_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging can fail on credentials or a missing source; the other tasks still deserve a run."""

    def refuse(request: object) -> None:
        raise RuntimeError("no credentials for the bucket")

    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    monkeypatch.setitem(PREPARERS, "stub", Producer("--prepared-media", refuse))
    invocations = _stub_benchmark(monkeypatch, writes=True)

    assert _catalog_sweep(tmp_path, "run-a") == 1

    assert invocations == []
    summary = _summary(tmp_path / "results" / "run-a")
    assert summary.tasks[0].status == "failed"


def test_an_interrupt_while_staging_still_writes_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staging is the longest phase outside a runner, so it is where Ctrl-C tends to land.

    `except Exception` does not catch it, so it unwound `main` and the summary was never
    written -- losing every outcome the sweep had already collected before it.
    """

    def interrupt(request: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    monkeypatch.setitem(PREPARERS, "stub", Producer("--prepared-media", interrupt))
    invocations = _stub_benchmark(monkeypatch, writes=True)

    assert _catalog_sweep(tmp_path, "run-i") == INTERRUPT_EXIT_CODE

    assert invocations == []
    summary = _summary(tmp_path / "results" / "run-i")
    assert summary.tasks[0].exit_code == INTERRUPT_EXIT_CODE


def test_a_task_carrying_the_manifest_flag_as_one_word_keeps_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--flag=value` is a spelling argparse accepts, so the sweep has to read it as one.

    Treating it as absent appended the sweep's own derived path, argparse took the later
    occurrence, and the operator's hand-supplied manifest was silently discarded -- for a
    manifest naming objects the sweep cannot produce, which is the one case the escape exists
    for.
    """
    supplied = tmp_path / "operator.json"
    supplied.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        TASKS,
        "stub-task",
        CatalogTask(
            "stub",
            ("--dataset", f"{ROOT}/stub/release.json", f"--prepared-media={supplied}"),
        ),
    )
    monkeypatch.setitem(PREPARERS, "stub", Producer("--prepared-media", _never_prepares))
    invocations = _stub_benchmark(monkeypatch, writes=True)

    assert _catalog_sweep(tmp_path, "run-w") == 0

    argv = invocations[0]
    assert argv.count("--prepared-media") == 0, argv
    assert f"--prepared-media={supplied}" in argv


def test_no_download_refuses_an_absent_release_rather_than_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag says "fail on an absent official release"; it used to do neither.

    Returning before the check meant nothing was fetched and nothing failed, so the sweep ran
    and each absent corpus surfaced as its own task's failure, hours apart.
    """
    monkeypatch.setitem(
        TASKS,
        "stub-task",
        CatalogTask("stub", ("--dataset", f"{ROOT}/locomo-refined/data/raw/locomo_refined.json")),
    )
    invocations = _stub_benchmark(monkeypatch, writes=True)

    with pytest.raises(ValueError, match="--no-download"):
        _catalog_sweep(tmp_path, "run-n")

    assert invocations == []


def _never_prepares(request: object) -> None:
    raise AssertionError("the task supplied its own manifest, so nothing should be prepared")


def _catalog_sweep(root: Path, run_id: str) -> int:
    return main(
        [
            "--tasks",
            "stub-task",
            "--run-id",
            run_id,
            "--benchmarks-root",
            str(root),
            "--no-download",
        ]
    )


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
        label = Path(_flag_value(argv, "--output")).parent.name

        @telemetry.operation_span(f"mindbridge.test.{label}")
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
    (tmp_path / "second").mkdir()
    (tmp_path / "second" / "predictions.jsonl").write_text("earlier run", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"second/predictions\.jsonl"):
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


def test_two_tasks_may_not_share_a_name() -> None:
    """Sharing one is silent: two tasks land in one tenant and answer from each other."""
    with pytest.raises(ValidationError, match="suite task names must be unique: twice"):
        BenchmarkSuite.model_validate(
            {
                "tasks": [
                    {"name": "twice", "benchmark": "locomo-refined"},
                    {"name": "twice", "benchmark": "locomo-refined"},
                ]
            }
        )


def test_one_output_name_used_twice_is_two_files_because_each_task_has_its_own_directory() -> None:
    """What used to need a second uniqueness check the task directory now makes impossible."""
    suite = BenchmarkSuite.model_validate(
        {
            "tasks": [
                {"name": "a", "benchmark": "locomo-refined", "output_name": "shared.json"},
                {"name": "b", "benchmark": "locomo-refined", "output_name": "shared.json"},
            ]
        }
    )

    assert [str(task.predictions_path()) for task in suite.tasks] == [
        "a/shared.json",
        "b/shared.json",
    ]


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


def test_the_catalog_names_every_value_of_the_choices_that_partition_a_corpus() -> None:
    """A split with no task is a leaderboard cell `--tasks` silently cannot name.

    The expected values come from each adapter's own `Literal`, not from a list restated here,
    so a release that grows a split fails this until the catalog carries it.
    """
    named = {
        benchmark: {
            argument
            for name, task in TASKS.items()
            if task.benchmark == benchmark
            for argument in task.arguments
        }
        for benchmark in ("mm-lifelong", "memlens", "atm", "m3")
    }

    assert set(get_args(MMLifelongSplit)) <= named["mm-lifelong"]
    assert set(get_args(MemLensContextWindow)) <= named["memlens"]
    assert set(get_args(AtmSplit)) <= named["atm"]
    assert set(get_args(AtmMediaSource)) <= named["atm"]
    assert {"robot", "web"} <= named["m3"]


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


def test_the_listing_says_which_of_four_things_stands_between_a_task_and_a_run(
    tmp_path: Path,
) -> None:
    """Ready, a download away, a preparation away, or blocked on a manifest with no producer."""
    dataset = tmp_path / "locomo-refined" / "data" / "raw"
    dataset.mkdir(parents=True)
    (dataset / "locomo_refined.json").write_text("[]", encoding="utf-8")

    lines = {line.split()[0]: line for line in listing(root=tmp_path).splitlines() if line.strip()}

    assert lines["locomo-refined"].endswith("ready")
    # Every MEMLENS input comes from a release, so nothing is asked of the operator.
    assert lines["memlens-32k"].endswith("download")
    # M3-Bench's prepared media has a producer, so the sweep will stage it rather than ask.
    assert lines["m3-robot"].endswith("prepare")
    # EgoLifeQA's has none yet, so its manifest is named rather than silently promised.
    assert f"{tmp_path}/egolife-prepared-a1.json" in lines["egolife"]


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
    assert f"--output {tmp_path}/results/smoke-01/stub-task/predictions.jsonl" in printed
    assert "--run-id smoke-01-stub-task" in printed
    assert f"--deployment-config {tmp_path}/deployment.json" in printed
    assert "--api-base-url http://localhost:8000" in printed


def test_naming_no_task_at_all_points_at_the_listing_rather_than_running_everything() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--run-id", "smoke-01"])

    assert exit_info.value.code == USAGE_EXIT_CODE


def test_a_sweep_without_a_run_id_gets_a_timestamped_one_that_still_isolates_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--tasks` alone is the whole invocation; what the run ID has to be is unique, not named."""
    monkeypatch.setitem(
        TASKS, "stub-task", CatalogTask("stub", ("--dataset", f"{ROOT}/stub/release.json"))
    )
    _stub_benchmark(monkeypatch, writes=True)

    assert main(["--tasks", "stub-task", "--benchmarks-root", str(tmp_path), "--dry-run"]) == 0

    printed = capsys.readouterr().out
    derived = re.search(r"--run-id (sweep-\d{8}-\d{6})-stub-task", printed)
    assert derived, printed
    # The output directory is derived from the same ID, so a rerun never meets its own files.
    assert f"--output {tmp_path}/results/{derived.group(1)}/stub-task/predictions.jsonl" in printed


def _stub_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writes: bool = True,
    handler: Callable[..., object] | None = None,
    media: bool = False,
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
        Runner("mindbridge_stub_benchmark", "Stub benchmark", extra=None, media=media),
    )
    return invocations


def _write_predictions(argv: Sequence[str]) -> None:
    """Write what a real runner writes, so the sweep can see the task produced a result.

    Creating the directory too, because `write_text_atomically` does and the sweep's default
    output directory is one it has never had to create itself before.
    """
    path = Path(_flag_value(argv, "--output"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


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


def test_the_media_knobs_reach_the_runners_that_take_them_and_no_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handing `--device-id` to a text-only runner is an argparse error that fails its task.

    Which is why the sweep would not forward them at all, and a media sweep could not raise its
    processing timeout without a hand-written suite file.
    """
    ingesting = _stub_benchmark(monkeypatch, writes=True, media=True)
    monkeypatch.setitem(
        RUNNERS, "text", Runner("mindbridge_stub_benchmark", "Stub benchmark", media=False)
    )
    suite = _suite_file(
        tmp_path,
        _task("ingesting"),
        {"name": "textual", "benchmark": "text", "arguments": ["--dataset", "release.json"]},
    )

    assert _sweep(suite, tmp_path, "--processing-timeout-seconds", "3600") == 0

    media, textual = ingesting
    assert _flag_value(media, "--processing-timeout-seconds") == "3600.0"
    assert "--processing-timeout-seconds" not in textual
    # The universal flags still reach both, which is what makes this a distinction and not a split.
    assert "--api-base-url" in media
    assert "--api-base-url" in textual


def test_an_unset_media_knob_is_not_forwarded_either(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep that defaulted these would pin a stale copy of each runner's own default."""
    invocations = _stub_benchmark(monkeypatch, writes=True, media=True)

    assert _sweep(_suite_file(tmp_path, _task("plain")), tmp_path) == 0

    assert "--device-id" not in invocations[0]
    assert "--poll-interval-seconds" not in invocations[0]
    assert "--processing-timeout-seconds" not in invocations[0]


@pytest.mark.parametrize("name", sorted({task.benchmark for task in TASKS.values()}))
def test_each_runner_takes_the_media_knobs_exactly_when_the_dispatcher_says_it_does(
    name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`Runner.media` is a declaration the sweep trusts before it has imported anything.

    Wrong in one direction the knob is silently dropped from a media task; wrong in the other a
    text-only task dies on an unrecognised argument mid-sweep. A parser's own `--help` is the
    complete list of what it accepts, so asking each runner for it is what keeps the declaration
    honest. Parametrised over every benchmark the catalog can name, which includes the one
    text-only runner, so the assertion cannot pass against a table that is all True.
    """
    with pytest.raises(SystemExit):
        RUNNERS[name].handler()(["--help"], prog="probe")

    accepted = "--device-id" in capsys.readouterr().out
    assert accepted is RUNNERS[name].media


def test_a_finished_sweep_prints_a_table_naming_every_task_and_where_its_numbers_came_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sweep's own stdout, so a run's result does not have to be dug out of two JSON files."""

    def handler(argv: Sequence[str], *, prog: str) -> int:
        _write_predictions(argv)
        output = Path(_flag_value(argv, "--output"))
        if output.parent.name == "scored":
            sidecar_manifest_path(output).write_text(
                json.dumps({"metrics": {"accuracy": 0.5}}), encoding="utf-8"
            )
        return 0

    _stub_benchmark(monkeypatch, handler=handler)
    suite = _suite_file(tmp_path, _task("scored"), _task("bare"))

    assert _sweep(suite, tmp_path) == 0

    printed = capsys.readouterr().out
    assert "accuracy" in printed
    assert "0.5000" in printed
    assert "runner" in printed
    assert "not scored" in printed, "a task with no numbers has to say so, not print nothing"
    assert "mindbridge-bench score" in printed, "and name what would give it one"


def test_the_table_can_be_reprinted_later_because_most_benchmarks_are_scored_afterwards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An official scorer runs after the sweep, so its numbers can never be in the sweep's own table."""
    _stub_benchmark(monkeypatch, writes=True)
    assert _sweep(_suite_file(tmp_path, _task("later")), tmp_path) == 0
    assert "not scored" in capsys.readouterr().out

    score_sidecar_path(tmp_path / "later" / "predictions.jsonl").write_text(
        json.dumps({"metrics": {"llm_judge": 0.61}}), encoding="utf-8"
    )

    assert main(["--report", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "llm_judge" in printed
    assert "0.6100" in printed
    assert "official" in printed
    assert "not scored" not in printed


def test_each_task_is_announced_with_a_rule_so_its_output_can_be_found_in_the_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runners write their own `[i/n]` progress, so without a boundary a sweep is one long stream."""
    _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))

    assert _sweep(suite, tmp_path) == 0

    banners = [line for line in capsys.readouterr().err.splitlines() if "─" in line]
    assert len(banners) == 2
    assert banners[0].startswith("[1/2] first · stub · sweep-01-first ")
    assert banners[1].startswith("[2/2] second · stub · sweep-01-second ")


def test_predict_only_reaches_every_task_so_a_sweep_needs_no_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seven runners now contact a judge to score themselves; this is the flag that says not to."""
    invocations = _stub_benchmark(monkeypatch, writes=True)
    suite = _suite_file(tmp_path, _task("first"), _task("second"))

    assert _sweep(suite, tmp_path, "--predict-only") == 0

    assert all("--predict-only" in argv for argv in invocations)


def test_a_sweep_that_did_not_ask_for_it_does_not_forward_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations = _stub_benchmark(monkeypatch, writes=True)

    assert _sweep(_suite_file(tmp_path, _task("only")), tmp_path) == 0

    assert "--predict-only" not in invocations[0]
