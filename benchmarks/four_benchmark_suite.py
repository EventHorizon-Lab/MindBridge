"""Run the committed four-task recipe through the existing public evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mindbridge.benchmarks.download import acquire_inputs, acquire_media
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.task_catalog import TASKS


def main() -> int:
    from support_coverage import _env

    recipe = json.loads(Path(__file__).with_suffix(".json").read_text())
    env = dict(_env())
    workspace = (
        Path.home() / ".local/share/openresearch/benchmark-cache/mindbridge" / recipe["suite_id"]
    )
    root = workspace / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    original = Path(env["MINDBRIDGE_BENCH_ROOT"])
    for name in ("longmemeval", "atm-bench", "mem-gallery"):
        target = root / name
        if not target.exists():
            target.symlink_to(original / name, target_is_directory=True)
    print("SUITE_RECIPE " + json.dumps(recipe), flush=True)
    print(
        "ICM input: video frames plus official speakerless ASR; no native speaker recognition.",
        flush=True,
    )
    icm = TASKS["icm-bench"]
    acquire_inputs(icm, root)
    acquire_media(icm, root, announce=lambda line: print(line, flush=True))
    generation = {
        "provider": "openai",
        "model": env["MINDBRIDGE_GENERATION_MODEL"],
        "base_url": env["MINDBRIDGE_GENERATION_BASE_URL"],
        "api_key": env.get("MINDBRIDGE_GENERATION_API_KEY", "EMPTY"),
        "modalities": ["text", "image", "video"],
        "timeout": 180.0,
        "max_retries": 2,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    config = {
        "embedding": {
            "provider": "openai",
            "model": env["MINDBRIDGE_EMBEDDING_MODEL"],
            "base_url": env["MINDBRIDGE_EMBEDDING_BASE_URL"],
            "api_key": env.get("MINDBRIDGE_EMBEDDING_API_KEY", "EMPTY"),
            "dimension": 2048,
            "modalities": ["text", "image", "video"],
            "request_format": "messages",
            "timeout": 180.0,
            "max_retries": 2,
        },
        "generation": {**generation, "video_limit": 8},
        "vision": {**generation, "modalities": ["image", "video"], "temperature": 0.0, "seed": 0},
        "settings": {
            "retrieval_mode": "hybrid",
            "evidence_budget_chars": recipe["evidence_budget_chars"],
            "reinforce_on_answer": False,
            "recall_planning": True,
            "recall_set_budget_chars": recipe["evidence_budget_chars"],
            "recall_set_max_rows": 60,
            "recall_rounds": 2,
        },
        "benchmark": {
            "judge": {
                "model": generation["model"],
                "base_url": generation["base_url"],
                "api_key": generation["api_key"],
                "timeout_seconds": 180.0,
            },
            "run": {
                "arms": "mindbridge",
                "limit": recipe["limit"],
                "batch_size": "auto",
                "max_batch_size": 64,
                "unit_concurrency": recipe["unit_concurrency"],
                "request_concurrency": recipe["request_concurrency"],
                "judge_concurrency": 8,
                "recall_limit": recipe["recall_limit"],
                "seed": [0, 1234, 1234, 1234],
                "log_samples": True,
                "download": False,
                "stream_results": True,
            },
        },
    }
    results = []
    with tempfile.TemporaryDirectory(prefix="mindbridge-eval-config-") as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        for task in recipe["tasks"]:
            output = workspace / "results" / task
            report = output / "results.jsonl"
            if report.is_file():
                status = json.loads(report.read_text()).get("status")
                entry = {
                    "task": task,
                    "exit_code": int(status != "completed"),
                    "output": str(output),
                }
                results.append(entry)
                print("TASK_STATUS " + json.dumps(entry), flush=True)
                print("TASK_RESULT " + report.read_text(), flush=True)
                continue
            run_id = recipe["suite_id"] + "-" + task
            data_root = workspace / "stores" / task
            run_path = BenchmarkRun.path_for(data_root, task, run_id)
            command = [
                sys.executable,
                "-m",
                "mindbridge.benchmarks.eval",
                "--config",
                str(path),
                "--tasks",
                task,
                "--fallback-reference-at",
                "2026-09-17T00:00:00Z",
                "--benchmarks-root",
                str(root),
                "--data-root",
                str(data_root),
                "--output-path",
                str(output),
                "--run-id",
                run_id,
                "--use-cache",
                str(workspace / "responses" / task),
            ]
            if (output.is_dir() and any(output.iterdir())) or (
                run_path.is_dir() and any(run_path.iterdir())
            ):
                command.append("--resume")
            if recipe["phase"] == "prepare":
                command.extend(["--check-integrity", "--no-download"])
            print("TASK_START " + task, flush=True)
            finished = subprocess.run(
                command, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            entry = {"task": task, "exit_code": finished.returncode, "output": str(output)}
            results.append(entry)
            print("TASK_STATUS " + json.dumps(entry), flush=True)
            if report.is_file():
                print("TASK_RESULT " + report.read_text(), flush=True)
    print("SUITE_STATUS " + json.dumps({"phase": recipe["phase"], "tasks": results}), flush=True)
    return int(any(row["exit_code"] != 0 for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
