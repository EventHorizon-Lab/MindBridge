from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_suite_skips_completed_tasks_and_resumes_only_partial_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "tasks": ["completed", "interrupted", "empty", "fresh"],
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
    report.write_text('{"accuracy": 1}\n')
    (completed / "samples.jsonl").write_text("finished samples")
    interrupted = results / "interrupted"
    interrupted.mkdir()
    (interrupted / "samples.partial.jsonl").write_text("partial samples")
    (results / "empty").mkdir()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main() == 0
    assert [command[command.index("--tasks") + 1] for command in calls] == [
        "interrupted",
        "empty",
        "fresh",
    ]
    assert ["--resume" in command for command in calls] == [True, False, False]
    assert report.read_text() == '{"accuracy": 1}\n'
