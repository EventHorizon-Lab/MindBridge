"""Comparable performance-budget checks for benchmark result artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PERFORMANCE_BUDGET_NAMES = (
    "answer_e2e_ttft_p95",
    "answer_e2e_latency_p95",
    "retrieval_e2e_latency_p95",
    "tokens_per_call",
    "answer_throughput",
)


@dataclass(frozen=True, slots=True)
class _Metric:
    path: tuple[str, ...]
    unit: str
    direction: Literal["lower", "higher"]
    requires_complete_distribution: bool = False
    complete_block: tuple[str, ...] | None = None


_METRICS: Mapping[str, _Metric] = {
    "answer_e2e_ttft_p95": _Metric(
        ("answer", "end_to_end_time_to_first_token_ms", "p95"),
        "ms",
        "lower",
        requires_complete_distribution=True,
    ),
    "answer_e2e_latency_p95": _Metric(("answer", "latency_ms", "p95"), "ms", "lower"),
    "retrieval_e2e_latency_p95": _Metric(
        ("search_e2e", "latency_ms", "p95"),
        "ms",
        "lower",
        complete_block=("search_e2e",),
    ),
    "tokens_per_call": _Metric(
        ("token_usage", "product", "average_tokens_per_request"), "tokens", "lower"
    ),
    "answer_throughput": _Metric(
        ("answer", "throughput_per_active_second"), "answers/second", "higher"
    ),
}

_GLOBAL_PATHS = (
    ("schema_version",),
    ("runner_version",),
    ("limit",),
    ("offset",),
    ("unit_concurrency",),
    ("request_concurrency",),
    ("recall_limit",),
    ("seeds",),
    ("predict_only",),
    ("arms", "selected"),
    ("arms", "ingest"),
    ("environment", "python_version"),
    ("environment", "platform"),
    ("environment", "runtime_versions"),
)
_MODEL_PATHS = tuple(
    ("model", name)
    for name in (
        "adapter",
        "embedding_model",
        "embedding_revision",
        "embedding_dimension",
        "embedding_warmup",
        "device",
        "generation_model",
        "generation_base_url",
        "generation_modalities",
        "generation_seed",
        "generation_temperature",
        "generation_kwargs",
        "generation_min_video_seconds",
        "transcription_model",
        "timeout_seconds",
        "memory_config",
    )
)
_MISSING = object()


def load_result(path: Path) -> dict[str, object]:
    """Load the result beside a directory, results file, or samples file."""
    resolved = path.expanduser().resolve()
    candidates: tuple[Path, ...]
    if resolved.is_dir():
        candidates = (resolved / "results.jsonl", resolved / "results.json")
    elif resolved.name == "samples.jsonl":
        candidates = (
            resolved.with_name("results.jsonl"),
            resolved.with_name("results.json"),
        )
    else:
        candidates = (resolved,)
    result_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if result_path is None:
        raise FileNotFoundError(f"baseline results do not exist beside: {resolved}")
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"baseline result is invalid JSON: {result_path}") from error
    if not isinstance(document, dict):
        raise ValueError("baseline result must be one JSON object")
    schema = document.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 10:
        raise ValueError("baseline result schema version is unsupported")
    return document


def performance_comparisons(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    budgets: Mapping[str, float],
) -> list[dict[str, object]]:
    """Compare task performance after rejecting non-equivalent run conditions."""
    if not budgets:
        return []
    unknown = sorted(set(budgets) - set(_METRICS))
    if unknown:
        raise ValueError(f"unknown performance budget(s): {', '.join(unknown)}")
    _require_comparable(candidate, baseline)
    previous = _product_tasks(baseline)
    rows: list[dict[str, object]] = []
    for current in _product_tasks(candidate).values():
        task_name = str(current["task"])
        prior = previous.get(task_name)
        if prior is None:
            raise ValueError(f"performance baseline has no product row for {task_name}")
        _require_task_comparable(current, prior)
        current_performance = _mapping(current.get("performance"), "candidate performance")
        prior_performance = _mapping(prior.get("performance"), "baseline performance")
        for name, budget in budgets.items():
            if not math.isfinite(budget) or budget < 0:
                raise ValueError(f"performance budget {name} must be non-negative and finite")
            metric = _METRICS[name]
            current_value = _metric_value(current_performance, metric, name, "candidate")
            baseline_value = _metric_value(prior_performance, metric, name, "baseline")
            regression = _regression_fraction(
                current_value, baseline_value, direction=metric.direction, metric=name
            )
            rows.append(
                {
                    "kind": "performance",
                    "task": task_name,
                    "metric": name,
                    "unit": metric.unit,
                    "direction": f"{metric.direction}_is_better",
                    "candidate": current_value,
                    "baseline": baseline_value,
                    "regression_fraction": regression,
                    "max_regression_fraction": budget,
                    "regressed": regression > budget,
                }
            )
    return rows


def _require_comparable(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> None:
    for side, document in (("candidate", candidate), ("baseline", baseline)):
        if document.get("status") != "completed":
            raise ValueError(f"{side} performance result is not completed without product errors")
    for path in (*_GLOBAL_PATHS, *_MODEL_PATHS):
        current = _at(candidate, path)
        previous = _at(baseline, path)
        if current != previous:
            _not_comparable(".".join(path), current, previous)
    if bool(candidate.get("response_cache")) or bool(baseline.get("response_cache")):
        raise ValueError("performance results using a response cache are not comparable")

    current_protocol = _at(candidate, ("measurement_protocol", "state"), "legacy_unspecified")
    previous_protocol = _at(baseline, ("measurement_protocol", "state"), "legacy_unspecified")
    if current_protocol != previous_protocol:
        _not_comparable("measurement_protocol.state", current_protocol, previous_protocol)

    _require_hardware_comparable(candidate, baseline)
    _require_acceleration_comparable(candidate, baseline)
    current_provenance = _response_provenance(candidate)
    previous_provenance = _response_provenance(baseline)
    if current_provenance and previous_provenance and current_provenance != previous_provenance:
        _not_comparable("model response provenance", current_provenance, previous_provenance)


def _require_hardware_comparable(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> None:

    current_hardware = _mapping(_at(candidate, ("environment", "hardware")), "candidate hardware")
    previous_hardware = _mapping(_at(baseline, ("environment", "hardware")), "baseline hardware")
    for name in ("machine", "processor", "cpu_model", "logical_cores", "ram_total_bytes"):
        if current_hardware.get(name) != previous_hardware.get(name):
            _not_comparable(
                f"environment.hardware.{name}",
                current_hardware.get(name),
                previous_hardware.get(name),
            )
    current_gpus, previous_gpus = current_hardware.get("gpus"), previous_hardware.get("gpus")
    if current_gpus is not None and previous_gpus is not None:
        if current_gpus != previous_gpus:
            _not_comparable("environment.hardware.gpus", current_gpus, previous_gpus)
    elif current_hardware.get("cuda_device_uuids") != previous_hardware.get("cuda_device_uuids"):
        _not_comparable(
            "environment.hardware.cuda_device_uuids",
            current_hardware.get("cuda_device_uuids"),
            previous_hardware.get("cuda_device_uuids"),
        )


def _require_acceleration_comparable(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> None:
    previous_acceleration = _at(baseline, ("environment", "acceleration_runtime"), _MISSING)
    if previous_acceleration is not _MISSING:
        current_acceleration = _at(candidate, ("environment", "acceleration_runtime"), _MISSING)
        if current_acceleration != previous_acceleration:
            _not_comparable(
                "environment.acceleration_runtime", current_acceleration, previous_acceleration
            )


def _require_task_comparable(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> None:
    for side, task in (("candidate", candidate), ("baseline", baseline)):
        if task.get("error_count") not in (None, 0) or task.get("ingest_failure_count") not in (
            None,
            0,
        ):
            raise ValueError(f"{side} performance task contains product errors")
    for name in (
        "dataset_sha256",
        "input_sha256",
        "evaluation_sha256",
        "batch_size",
        "input_modalities",
        "unit_count",
        "question_count",
    ):
        if candidate.get(name) != baseline.get(name):
            _not_comparable(
                f"tasks[{candidate.get('task')}].{name}", candidate.get(name), baseline.get(name)
            )


def _product_tasks(document: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = document.get("tasks")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("benchmark result tasks must be an array")
    rows: dict[str, Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or item.get("arm", "mindbridge") != "mindbridge":
            continue
        name = item.get("task")
        if not isinstance(name, str) or not name:
            raise ValueError("benchmark result task names must be non-empty strings")
        if name in rows:
            raise ValueError(f"benchmark result has duplicate product rows for {name}")
        rows[name] = item
    if not rows:
        raise ValueError("benchmark result has no product task rows")
    return rows


def _response_provenance(document: Mapping[str, object]) -> tuple[tuple[object, ...], ...]:
    identities: set[tuple[object, ...]] = set()
    for task in _product_tasks(document).values():
        performance = task.get("performance")
        nodes = performance.get("nodes") if isinstance(performance, Mapping) else None
        if not isinstance(nodes, Mapping):
            continue
        for span_name, node in nodes.items():
            breakdown = node.get("breakdown") if isinstance(node, Mapping) else None
            if not isinstance(breakdown, Sequence) or isinstance(breakdown, str | bytes):
                continue
            for row in breakdown:
                if not isinstance(row, Mapping):
                    continue
                response_model = row.get("response_model")
                fingerprint = row.get("system_fingerprint")
                response_models = _string_values(row.get("response_models")) or (
                    (response_model,) if isinstance(response_model, str) else ()
                )
                fingerprints = _string_values(row.get("response_system_fingerprints")) or (
                    (fingerprint,) if isinstance(fingerprint, str) else ()
                )
                if response_models or fingerprints:
                    identities.add(
                        (
                            span_name,
                            row.get("requested_model"),
                            response_models,
                            fingerprints,
                        )
                    )
    return tuple(sorted(identities, key=repr))


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _metric_value(
    performance: Mapping[str, object], definition: _Metric, metric: str, side: str
) -> float:
    if definition.complete_block is not None:
        block = _at(performance, definition.complete_block)
        if not isinstance(block, Mapping) or block.get("complete") is not True:
            raise ValueError(f"{side} result has no complete {metric} metric")
    if definition.requires_complete_distribution:
        distribution = _at(performance, definition.path[:-1])
        if not isinstance(distribution, Mapping) or distribution.get("complete") is not True:
            raise ValueError(f"{side} result has no complete {metric} metric")
    value = _at(performance, definition.path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{side} result has no complete {metric} metric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{side} result has invalid {metric} metric")
    return number


def _regression_fraction(
    current: float,
    baseline: float,
    *,
    direction: Literal["lower", "higher"],
    metric: str,
) -> float:
    if baseline == 0:
        if current == 0:
            return 0.0
        raise ValueError(f"baseline {metric} is zero and cannot define a relative budget")
    return (
        (current - baseline) / baseline if direction == "lower" else (baseline - current) / baseline
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _at(document: Mapping[str, object], path: Sequence[str], default: object = _MISSING) -> object:
    current: object = document
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            if default is not _MISSING:
                return default
            raise ValueError(f"benchmark result is missing {'.'.join(path)}")
        current = current[name]
    return current


def _not_comparable(name: str, candidate: object, baseline: object) -> None:
    raise ValueError(
        f"performance baseline is not comparable: {name} differs "
        f"(candidate={candidate!r}, baseline={baseline!r})"
    )
