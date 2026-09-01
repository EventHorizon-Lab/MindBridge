"""Report lexicographic quality, speed, and token units for autoresearch."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prepare_final_manifests import validate as validate_final_pair
from prepare_speed_validation import validate_speed

ROOT = Path(__file__).resolve().parent
REQUIRED_TASKS = (
    "locomo-refined",
    "atm-bench-hard",
    "mem-gallery",
    "m3-bench-robot",
    "egolifeqa",
)
QUALITY_GAIN = 0.01
QUALITY_TOLERANCE = 0.005
TOKEN_NOISE = 0.02
TOKEN_GAIN = 0.05


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _fixed_final_tasks() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]
]:
    summary, artifacts = validate_final_pair(require_manifests=True)
    tasks: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for side in ("current", "baseline"):
        for slug, artifact in artifacts[side].items():
            task = artifact["task"]
            document = dict(artifact["document"])
            document["_path"] = str(artifact["dir"] / "results.json")
            tasks[f"{side}:{task['task']}"] = task
            documents[f"{side}:{slug}"] = document
    return (
        {
            name.removeprefix("current:"): value
            for name, value in tasks.items()
            if name.startswith("current:")
        },
        {name: value for name, value in documents.items() if name.startswith("current:")},
        {
            name.removeprefix("baseline:"): value
            for name, value in tasks.items()
            if name.startswith("baseline:")
        },
        {name: value for name, value in documents.items() if name.startswith("baseline:")},
        [str(error) for error in summary["errors"]],
    )


def _primary(task: Mapping[str, Any]) -> float | None:
    score = task.get("score")
    return _number(score.get("mean")) if isinstance(score, Mapping) else None


def _performance(task: Mapping[str, Any]) -> Mapping[str, Any]:
    value = task.get("performance")
    return value if isinstance(value, Mapping) else {}


def _tokens(task: Mapping[str, Any]) -> float | None:
    usage = _performance(task).get("token_usage")
    if not isinstance(usage, Mapping):
        return None
    modules = usage.get("by_module")
    generation = modules.get("generation") if isinstance(modules, Mapping) else None
    if not isinstance(generation, Mapping) or generation.get("complete") is not True:
        return None
    return _number(generation.get("average_tokens"))


def _ratio(current: Sequence[float], baseline: Sequence[float]) -> float | None:
    if len(current) != len(baseline) or not current or any(value <= 0 for value in baseline):
        return None
    return math.exp(
        sum(math.log(now / before) for now, before in zip(current, baseline, strict=True))
        / len(current)
    )


def _identity_errors(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    errors = []
    for name in REQUIRED_TASKS:
        now, before = current[name], baseline[name]
        for key in ("dataset_sha256", "evaluation_sha256", "scorer_protocol", "question_count"):
            if now.get(key) != before.get(key):
                errors.append(f"{name}: {key} differs from baseline")
    return errors


def _validate(tasks: Mapping[str, Any], documents: Mapping[str, Any]) -> list[str]:
    errors = [f"missing task: {name}" for name in REQUIRED_TASKS if name not in tasks]
    for document in documents.values():
        if document.get("status") != "completed":
            errors.append(f"{document.get('_path')}: run is not completed")
    for name in REQUIRED_TASKS:
        task = tasks.get(name)
        if not isinstance(task, Mapping):
            continue
        if task.get("score_valid") is not True:
            errors.append(f"{name}: score is invalid")
        for key in ("error_count", "ingest_failure_count"):
            value = task.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value:
                errors.append(f"{name}: {key}={value!r}")
        if _primary(task) is None:
            errors.append(f"{name}: primary score is missing")
    return errors


def _stage_units(
    *,
    errors: int,
    regressions: int,
    quality_ready: bool,
    speed_ready: bool,
    tokens_ready: bool,
) -> int:
    if errors:
        return 100 + errors + regressions
    if regressions:
        return 50 + regressions
    if not quality_ready:
        return 3
    if not speed_ready:
        return 2
    if not tokens_ready:
        return 1
    return 0


def _report(suite: str) -> dict[str, Any]:  # noqa: C901 - ordered acceptance gates
    if suite != "dev":
        raise ValueError("only the frozen dev manifest alias is supported")
    current, current_documents, baseline, baseline_documents, pair_errors = _fixed_final_tasks()
    errors = [*pair_errors, *_validate(current, current_documents)]
    regressions: list[str] = []
    quality_delta: float | None = None
    ask_service_average_ratio: float | None = None
    ttft_ratio: float | None = None
    token_ratio: float | None = None
    speed_report: dict[str, Any] = {
        "errors": ["performance repetitions have not been validated"],
        "ready": False,
        "regressions": [],
        "repetitions": [],
        "task_median_ratios": {},
    }
    quality_ready = speed_ready = tokens_ready = False

    if not errors and baseline:
        baseline_errors = _validate(baseline, baseline_documents)
        errors.extend(f"baseline: {error}" for error in baseline_errors)
        if not errors:
            errors.extend(_identity_errors(current, baseline))

    if not errors and baseline:
        current_quality = [_primary(current[name]) for name in REQUIRED_TASKS]
        baseline_quality = [_primary(baseline[name]) for name in REQUIRED_TASKS]
        if all(value is not None for value in (*current_quality, *baseline_quality)):
            now = [float(value) for value in current_quality if value is not None]
            before = [float(value) for value in baseline_quality if value is not None]
            for name, value, prior in zip(REQUIRED_TASKS, now, before, strict=True):
                if value + QUALITY_TOLERANCE < prior:
                    regressions.append(f"{name}: primary score {value:.6f} < {prior:.6f}")
            quality_delta = sum(now) / len(now) - sum(before) / len(before)
            quality_ready = not regressions and quality_delta >= QUALITY_GAIN

        if quality_ready:
            speed_report = validate_speed()
            speed_errors = speed_report.get("errors")
            speed_regressions = speed_report.get("regressions")
            if isinstance(speed_errors, list):
                errors.extend(f"speed: {error}" for error in speed_errors)
            if isinstance(speed_regressions, list):
                regressions.extend(f"speed: {error}" for error in speed_regressions)
            ask_service_average_ratio = _number(speed_report.get("ask_geometric_ratio"))
            ttft_ratio = _number(speed_report.get("ttft_geometric_ratio"))
            speed_ready = speed_report.get("ready") is True

        current_tokens = [_tokens(current[name]) for name in REQUIRED_TASKS]
        baseline_tokens = [_tokens(baseline[name]) for name in REQUIRED_TASKS]
        comparable_tokens = all(value is not None for value in (*current_tokens, *baseline_tokens))
        if comparable_tokens:
            now_tokens = [float(value) for value in current_tokens if value is not None]
            before_tokens = [float(value) for value in baseline_tokens if value is not None]
            token_ratio = _ratio(now_tokens, before_tokens)
            for name, value, prior in zip(REQUIRED_TASKS, now_tokens, before_tokens, strict=True):
                if value > prior * (1 + TOKEN_NOISE):
                    regressions.append(f"{name}: generation tokens regressed")
        tokens_ready = (
            speed_ready
            and comparable_tokens
            and not any("tokens regressed" in item for item in regressions)
            and token_ratio is not None
            and token_ratio <= 1 - TOKEN_GAIN
        )

    units = _stage_units(
        errors=len(errors),
        regressions=len(regressions),
        quality_ready=quality_ready,
        speed_ready=speed_ready,
        tokens_ready=tokens_ready,
    )
    return {
        "suite": suite,
        "units": units,
        "failing_tests": len(errors),
        "open_hard_regressions": len(regressions),
        "metric_delta": units - len(errors) - len(regressions),
        "metric_target": 1,
        "errors": errors,
        "regressions": regressions,
        "quality": {"macro_delta": quality_delta, "ready": quality_ready},
        "speed": {
            "ask_service_average_ratio": ask_service_average_ratio,
            "generation_ttft_ratio": ttft_ratio,
            "ready": speed_ready,
            "repetitions": speed_report.get("repetitions"),
            "task_median_ratios": speed_report.get("task_median_ratios"),
        },
        "tokens": {"generation_ratio": token_ratio, "ready": tokens_ready},
    }


def _self_check() -> None:
    assert (
        _stage_units(
            errors=5, regressions=0, quality_ready=False, speed_ready=False, tokens_ready=False
        )
        == 105
    )
    assert (
        _stage_units(
            errors=0, regressions=0, quality_ready=False, speed_ready=False, tokens_ready=False
        )
        == 3
    )
    assert (
        _stage_units(
            errors=0, regressions=0, quality_ready=True, speed_ready=True, tokens_ready=True
        )
        == 0
    )
    assert _ratio((5.0, 20.0), (10.0, 10.0)) == 1.0
    assert _ratio((1.0,), ()) is None
    assert _ratio((1.0,), (0.0,)) is None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="dev")
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
        print(0)
        return 0
    report = _report(arguments.suite)
    (ROOT / f"verification-{arguments.suite}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report["units"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
