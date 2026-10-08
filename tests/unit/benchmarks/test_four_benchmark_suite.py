from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mindbridge.benchmarks.isolation import BenchmarkRun


@pytest.mark.parametrize("status", ["completed", "completed_with_errors", "unknown"])
def test_suite_preserves_report_status_and_resumes_early_interruptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    path = Path(__file__).parents[3] / "benchmarks/four_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("suite_driver", path)
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    recipe_path = tmp_path / "suite.json"
    recipe_path.write_text(
        json.dumps(
            {
                "suite_id": "test-suite",
                "tasks": ["completed", "interrupted", "ingesting", "output-only", "empty", "fresh"],
                "phase": "evaluate",
                "evidence_budget_chars": 1000,
                "limit": 1,
                "unit_concurrency": 1,
                "request_concurrency": 1,
                "recall_limit": 5,
            }
        )
    )
    monkeypatch.setattr(driver, "__file__", str(recipe_path.with_suffix(".py")))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    support = ModuleType("support_coverage")
    support._env = lambda: {  # type: ignore[attr-defined]
        "MINDBRIDGE_BENCH_ROOT": str(tmp_path / "original"),
        "MINDBRIDGE_GENERATION_MODEL": "mock",
        "MINDBRIDGE_GENERATION_BASE_URL": "http://mock/v1",
        "MINDBRIDGE_EMBEDDING_MODEL": "mock",
        "MINDBRIDGE_EMBEDDING_BASE_URL": "http://mock/v1",
    }
    monkeypatch.setitem(sys.modules, "support_coverage", support)
    monkeypatch.setattr(driver, "acquire_inputs", lambda *args: None)
    monkeypatch.setattr(driver, "acquire_media", lambda *args, **kwargs: None)
    results = tmp_path / ".local/share/openresearch/benchmark-cache/mindbridge/test-suite/results"
    completed = results / "completed"
    completed.mkdir(parents=True)
    report = completed / "results.jsonl"
    report_text = json.dumps({"status": status}) + "\n"
    report.write_text(report_text)
    (completed / "samples.jsonl").write_text("finished samples")
    interrupted = results / "interrupted"
    interrupted.mkdir()
    (interrupted / "samples.partial.jsonl").write_text("partial samples")
    workspace = results.parent
    benchmark_run = BenchmarkRun(
        workspace / "stores/ingesting", "ingesting", "test-suite-ingesting"
    )
    checkpoint = benchmark_run.checkpoint_path("unit")
    checkpoint.parent.mkdir()
    checkpoint.write_text("durable ingest checkpoint")
    output_only = results / "output-only"
    output_only.mkdir()
    (output_only / "config.json").write_text("crash config")
    (results / "empty").mkdir()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main() == int(status != "completed")
    assert [command[command.index("--tasks") + 1] for command in calls] == [
        "interrupted",
        "ingesting",
        "output-only",
        "empty",
        "fresh",
    ]
    assert ["--resume" in command for command in calls] == [True, True, True, False, False]
    assert report.read_text() == report_text
    assert checkpoint.read_text() == "durable ingest checkpoint"


@pytest.mark.parametrize("suite", [None, "complementary", "four-benchmark", "support-coverage"])
def test_launcher_selects_each_suite_explicitly(
    monkeypatch: pytest.MonkeyPatch, suite: str | None
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "benchmarks"))
    support_coverage = importlib.import_module("support_coverage")

    calls: list[str] = []

    def record(name: str) -> int:
        calls.append(name)
        return 7

    for name in ("complementary_trial", "four_benchmark_suite"):
        module = ModuleType(name)
        module.main = lambda name=name: record(name)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    forwarded: list[list[str]] = []

    def original_runner(argv: list[str]) -> int:
        forwarded.append(argv)
        return record("support_coverage")

    monkeypatch.setattr(support_coverage, "main", original_runner)
    arguments = [] if suite is None else ["--suite", suite]
    if suite == "support-coverage":
        arguments.extend(["--limit", "1"])
    assert support_coverage.launch(arguments) == 7
    expected = {
        "complementary": "complementary_trial",
        "four-benchmark": "four_benchmark_suite",
        "support-coverage": "support_coverage",
    }
    assert calls == [expected[suite or "complementary"]]
    assert forwarded == ([["--limit", "1"]] if suite == "support-coverage" else [])
