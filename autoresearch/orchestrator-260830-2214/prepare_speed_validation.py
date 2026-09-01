"""Validate three pre-registered AB/BA performance repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from prepare_final_manifests import (
    BASELINE_NAME,
    CURRENT_NAME,
    EXPECTED_MODEL,
    PAIR_DOCUMENT_FIELDS,
    PAIR_SAMPLE_FIELDS,
    PAIR_TASK_FIELDS,
    REQUIRED_SAMPLE_FIELDS,
    locked_identities,
    media_manifest_identity_sha256,
    validate_media_manifest,
    validate_provenance,
    validate_sample_judge_models,
)

ROOT = Path(__file__).resolve().parent
PERFORMANCE = ROOT / "performance"
PERFORMANCE_DATA = Path("/dev/shm/mindbridge-ar-perf-e806cf3b1dd4")
REPETITIONS = (1, 2, 3)
TASKS: dict[str, dict[str, object]] = {
    "locomo": {
        "allow_unverified_data": True,
        "limit": 1,
        "offset": 0,
        "question_count": 20,
        "task": "locomo-refined",
        "batch_size": 32,
    },
    "atm-hard": {
        "allow_unverified_data": False,
        "limit": None,
        "offset": 0,
        "question_count": 31,
        "task": "atm-bench-hard",
        "batch_size": 8,
    },
    "gallery": {
        "allow_unverified_data": True,
        "limit": 1,
        "offset": 0,
        "question_count": 20,
        "task": "mem-gallery",
        "batch_size": 8,
    },
    "m3": {
        "allow_unverified_data": True,
        "limit": None,
        "offset": 0,
        "question_count": 14,
        "task": "m3-bench-robot",
        "batch_size": 8,
    },
    "egolife": {
        "allow_unverified_data": False,
        "limit": 10,
        "offset": 50,
        "question_count": 10,
        "task": "egolifeqa",
        "batch_size": 8,
    },
}
EXPECTED_IDENTITIES = locked_identities("speed")
SCHEDULE = {
    ("baseline", 1): 0,
    ("current", 1): 1,
    ("current", 2): 2,
    ("baseline", 2): 3,
    ("baseline", 3): 4,
    ("current", 3): 5,
}
SCHEDULE_BY_INDEX = {index: pair for pair, index in SCHEDULE.items()}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a finite number, found {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"expected a positive finite number, found {value!r}")
    return number


def _percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _artifact(side: str, repetition: int, slug: str) -> dict[str, Any]:
    frozen_name = BASELINE_NAME if side == "baseline" else CURRENT_NAME
    result_dir = (PERFORMANCE / frozen_name / f"r{repetition}" / slug).resolve()
    result_path = result_dir / "results.json"
    samples_path = result_dir / "samples.jsonl"
    _require(result_path.is_file(), f"{side}/r{repetition}/{slug}: missing results")
    _require(samples_path.is_file(), f"{side}/r{repetition}/{slug}: missing samples")
    document = _load(result_path)
    _require(not result_dir.is_symlink(), f"{side}/r{repetition}/{slug}: result symlink")
    _require(not result_path.is_symlink(), f"{side}/r{repetition}/{slug}: results symlink")
    _require(not samples_path.is_symlink(), f"{side}/r{repetition}/{slug}: samples symlink")
    _require(
        all(field in document for field in PAIR_DOCUMENT_FIELDS),
        f"{side}/r{repetition}/{slug}: missing result fields",
    )
    expected_run_id = f"{frozen_name}-perf-r{repetition}-{slug}"
    expected_data_root = (PERFORMANCE_DATA / frozen_name / f"r{repetition}" / slug).resolve()
    for field, expected in {
        "allow_unverified_data": TASKS[slug]["allow_unverified_data"],
        "bootstrap_samples": 2000,
        "cached_judge_count": 0,
        "cached_response_count": 0,
        "data_root": str(expected_data_root),
        "limit": TASKS[slug]["limit"],
        "log_samples": True,
        "offset": TASKS[slug]["offset"],
        "predict_only": True,
        "recall_limit": 20,
        "response_cache": None,
        "run_id": expected_run_id,
        "runner_version": "mindbridge_eval_official_v9",
        "schema_version": 7,
        "seed": 20260830,
        "seeds": [20260830, 20260830, 20260830, 20260830],
        "status": "completed",
        "unit_concurrency": 1,
        "request_concurrency": 1,
    }.items():
        _require(document.get(field) == expected, f"{side}/r{repetition}/{slug}: wrong {field}")
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{side}/r{repetition}/{slug}: missing model")
    for field, expected in EXPECTED_MODEL.items():
        _require(model.get(field) == expected, f"{side}/r{repetition}/{slug}: wrong model {field}")
    judge = document.get("judge")
    _require(
        isinstance(judge, dict)
        and judge.get("model") == "qwen3.8-flash"
        and judge.get("base_url") == "https://inner-prism.cece.com/api/v1"
        and judge.get("concurrency") == 1,
        f"{side}/r{repetition}/{slug}: wrong judge config",
    )

    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError(f"{side}/r{repetition}/{slug}: expected one task")
    task = tasks[0]
    _require(task.get("task") == TASKS[slug]["task"], f"{side}/r{repetition}/{slug}: wrong task")
    _require(
        all(field in task for field in PAIR_TASK_FIELDS),
        f"{side}/r{repetition}/{slug}: missing task fields",
    )
    _require(
        task.get("batch_size") == TASKS[slug]["batch_size"],
        f"{side}/r{repetition}/{slug}: wrong batch size",
    )
    question_count_value = TASKS[slug]["question_count"]
    if isinstance(question_count_value, bool) or not isinstance(question_count_value, int):
        raise AssertionError(f"{slug}: invalid expected question count")
    question_count = question_count_value
    _require(
        task.get("question_count") == question_count,
        f"{side}/r{repetition}/{slug}: wrong question count",
    )
    _require(task.get("error_count") == 0, f"{side}/r{repetition}/{slug}: task error")
    _require(task.get("ingest_failure_count") == 0, f"{side}/r{repetition}/{slug}: ingest error")

    performance = task.get("performance")
    nodes = performance.get("nodes") if isinstance(performance, dict) else None
    ask = nodes.get("mindbridge.ask") if isinstance(nodes, dict) else None
    generation = nodes.get("mindbridge.model.generation") if isinstance(nodes, dict) else None
    ttft = generation.get("ttft_ms") if isinstance(generation, dict) else None
    latency = task.get("latency_ms")
    if (
        not isinstance(ask, dict)
        or not isinstance(generation, dict)
        or not isinstance(ttft, dict)
        or not isinstance(latency, dict)
    ):
        raise ValueError(f"{side}/r{repetition}/{slug}: missing performance metric")
    _require(ask.get("count") == question_count, f"{side}/r{repetition}/{slug}: ask count")
    _require(generation.get("count") == question_count, f"{side}/r{repetition}/{slug}: gen count")
    _require(ttft.get("count") == question_count, f"{side}/r{repetition}/{slug}: TTFT count")
    _require(
        math.isclose(
            _number(ask.get("average_ms")),
            _number(ask.get("total_seconds")) * 1_000 / question_count,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ),
        f"{side}/r{repetition}/{slug}: ask average mismatch",
    )

    samples_bytes = samples_path.read_bytes()
    samples_hash = hashlib.sha256(samples_bytes).hexdigest()
    _require(document.get("samples_sha256") == samples_hash, f"{side}/r{repetition}/{slug}: hash")
    rows = [json.loads(line) for line in samples_bytes.decode().splitlines()]
    _require(len(rows) == question_count, f"{side}/r{repetition}/{slug}: sample count")
    _require(all(isinstance(row, dict) for row in rows), f"{side}/r{repetition}/{slug}: sample row")
    _require(
        all(all(field in row for field in REQUIRED_SAMPLE_FIELDS) for row in rows),
        f"{side}/r{repetition}/{slug}: missing sample fields",
    )
    validate_sample_judge_models(document, rows, f"{side}/r{repetition}/{slug}")
    _require(all(row.get("cached") is False for row in rows), f"{side}/r{repetition}/{slug}: cache")
    _require(
        all(row.get("error_code") is None for row in rows), f"{side}/r{repetition}/{slug}: error"
    )
    raw_sample_ids = tuple(row.get("sample_id") for row in rows)
    _require(
        all(isinstance(value, str) and value for value in raw_sample_ids)
        and len(set(raw_sample_ids)) == question_count,
        f"{side}/r{repetition}/{slug}: ID",
    )
    sample_ids = tuple(value for value in raw_sample_ids if isinstance(value, str))
    identity = EXPECTED_IDENTITIES.get(slug)
    if identity is None:
        raise ValueError(f"{slug}: speed identity is not pinned")
    _require(
        identity.get("question_count") == question_count,
        f"{side}/r{repetition}/{slug}: wrong pinned question count",
    )
    for field in ("dataset_sha256", "evaluation_sha256", "input_sha256"):
        _require(task.get(field) == identity[field], f"{side}/r{repetition}/{slug}: wrong {field}")
    validate_media_manifest(
        result_dir,
        document,
        str(TASKS[slug]["task"]),
        identity,
        f"{side}/r{repetition}/{slug}",
    )
    ordered_ids_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    _require(
        ordered_ids_hash == identity["ordered_sample_ids_sha256"],
        f"{side}/r{repetition}/{slug}: wrong ordered IDs",
    )
    latencies = sorted(_number(row.get("latency_ms")) for row in rows)
    latency_p95 = _percentile(latencies, 0.95)
    _require(
        math.isclose(_number(latency.get("p95")), latency_p95, rel_tol=1e-12, abs_tol=1e-9),
        f"{side}/r{repetition}/{slug}: p95 mismatch",
    )
    sequence_index = SCHEDULE[(side, repetition)]
    previous = None
    if sequence_index:
        previous_side, previous_repetition = SCHEDULE_BY_INDEX[sequence_index - 1]
        previous_name = BASELINE_NAME if previous_side == "baseline" else CURRENT_NAME
        previous = (
            PERFORMANCE / previous_name / f"r{previous_repetition}" / slug / "run-provenance.json"
        ).resolve()
    validate_provenance(
        result_dir,
        side,
        result_path,
        samples_hash,
        expected_sequence={
            "id": f"speed-{slug}",
            "index": sequence_index,
            "label": f"{side}-r{repetition}",
            "previous_provenance": None if previous is None else str(previous),
            "previous_provenance_sha256": (
                None if previous is None else hashlib.sha256(previous.read_bytes()).hexdigest()
            ),
            "total": 6,
        },
    )
    return {
        "ask": _number(ask["average_ms"]),
        "document": document,
        "latency_p95": latency_p95,
        "rows": rows,
        "sample_ids": sample_ids,
        "task": task,
        "ttft": _number(ttft["average"]),
    }


def _geometric(values: list[float]) -> float:
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def validate_speed() -> dict[str, Any]:  # noqa: C901 - fail-closed repeated-pair audit
    artifacts: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        "baseline": {},
        "current": {},
    }
    errors: list[str] = []
    for side in artifacts:
        for repetition in REPETITIONS:
            artifacts[side][repetition] = {}
            for slug in TASKS:
                try:
                    artifacts[side][repetition][slug] = _artifact(side, repetition, slug)
                except (
                    KeyError,
                    OSError,
                    TypeError,
                    UnicodeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    errors.append(str(error))

    regressions: list[str] = []
    ratios: dict[str, dict[str, float]] = {}
    if not errors:
        for slug in TASKS:
            try:
                reference = artifacts["baseline"][1][slug]
                metric_ratios: dict[str, list[float]] = {
                    "ask": [],
                    "latency_p95": [],
                    "ttft": [],
                }
                for repetition in REPETITIONS:
                    before = artifacts["baseline"][repetition][slug]
                    now = artifacts["current"][repetition][slug]
                    _require(
                        before["sample_ids"] == reference["sample_ids"],
                        f"{slug}: baseline IDs differ",
                    )
                    _require(
                        now["sample_ids"] == reference["sample_ids"],
                        f"{slug}: candidate IDs differ",
                    )
                    for artifact in (before, now):
                        for field in PAIR_TASK_FIELDS:
                            _require(
                                artifact["task"].get(field) == reference["task"].get(field),
                                f"{slug}: task {field}",
                            )
                        for field in PAIR_DOCUMENT_FIELDS:
                            _require(
                                artifact["document"].get(field) == reference["document"].get(field),
                                f"{slug}: document {field}",
                            )
                        for reference_row, row in zip(
                            reference["rows"], artifact["rows"], strict=True
                        ):
                            for field in PAIR_SAMPLE_FIELDS:
                                _require(
                                    reference_row[field] == row[field],
                                    f"{slug}: sample {field}",
                                )
                    for metric in metric_ratios:
                        metric_ratios[metric].append(float(now[metric]) / float(before[metric]))
                ratios[slug] = {
                    metric: statistics.median(values) for metric, values in metric_ratios.items()
                }
                for metric, ratio in ratios[slug].items():
                    if ratio > 1.05:
                        regressions.append(f"{slug}: median {metric} ratio {ratio:.6f} > 1.05")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))

    complete = len(ratios) == len(TASKS)
    ask_ratio = _geometric([ratios[slug]["ask"] for slug in TASKS]) if complete else None
    ttft_ratio = _geometric([ratios[slug]["ttft"] for slug in TASKS]) if complete else None
    ready = (
        not errors
        and not regressions
        and ask_ratio is not None
        and ttft_ratio is not None
        and min(ask_ratio, ttft_ratio) <= 0.95
    )
    return {
        "ask_geometric_ratio": ask_ratio,
        "errors": errors,
        "ready": ready,
        "regressions": regressions,
        "repetitions": list(REPETITIONS),
        "task_median_ratios": ratios,
        "ttft_geometric_ratio": ttft_ratio,
    }


def _self_check() -> None:
    assert statistics.median((1.0, 9.0, 2.0)) == 2.0
    assert math.isclose(_geometric([0.5, 2.0]), 1.0)
    assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    legacy_manifest = {"unit": []}
    expected_manifest_hash = hashlib.sha256(b'{"unit":[]}').hexdigest()
    assert media_manifest_identity_sha256(legacy_manifest) == expected_manifest_hash
    assert media_manifest_identity_sha256({"units": legacy_manifest}) == expected_manifest_hash
    for invalid_manifest in ([], {"units": []}):
        try:
            media_manifest_identity_sha256(invalid_manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-object task media manifest was accepted")
    validate_sample_judge_models(
        {"judge": {"model": "fixture-judge"}},
        [{"judge_model": None}, {"judge_model": "fixture-judge"}],
        "fixture",
    )
    try:
        validate_sample_judge_models(
            {"judge": {"model": "fixture-judge"}},
            [{"judge_model": "different-judge"}],
            "fixture",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unconfigured sample judge model was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
        print(0)
        return 0
    report = validate_speed()
    (ROOT / "speed-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
