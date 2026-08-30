"""Audit the preregistered Jina video recipe v4/v5 benchmark pair.

Usage:
    uv run --frozen python autoresearch/orchestrator-260830-2214/\
analyze_jina_video_ablation.py
    uv run --frozen python autoresearch/orchestrator-260830-2214/\
analyze_jina_video_ablation.py --write
    uv run --frozen python autoresearch/orchestrator-260830-2214/\
analyze_jina_video_ablation.py --capture-store-audit RESULT_DIR

The command is fail-closed: all four result directories must be complete and
comparable before it emits a passing verdict. ``--write`` creates JSON and
Markdown companions next to this script; without it, JSON is printed only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import statistics
from contextlib import ExitStack
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "ablation-jina-video-20260831"
OUTPUT_JSON = HERE / "jina-video-ablation-pair.json"
OUTPUT_MARKDOWN = HERE / "jina-video-ablation-pair.md"

VERSIONS = ("v4", "v5")
TASKS: dict[str, dict[str, Any]] = {
    "egolife-n10": {
        "task": "egolifeqa",
        "limit": 10,
        "dataset_sha256": "688ae079f458132f13150711b7e099fbe4fdedd97f19f5a33bdebb7bfde74a52",
        "manifest_sha256": "74562712d9f47defde018ed09fd6d998c188bf9d5127a34f052a50476f016011",
    },
    "m3-l1": {
        "task": "m3-bench-robot",
        "limit": 1,
        "dataset_sha256": "565d872817e1ab92266883fd4381ebc5c1432555d0b85b958d8178bd086db1c3",
        "manifest_sha256": "09982dd6367761f3895988f855b3969a4ae86abe9475fb551b8e475ae9b9c450",
    },
}
DOCUMENT_CONFIG_FIELDS = (
    "allow_unverified_data",
    "bootstrap_samples",
    "environment",
    "judge",
    "limit",
    "log_samples",
    "media_manifest_path",
    "media_roots",
    "model",
    "num_fewshot",
    "offset",
    "predict_only",
    "recall_limit",
    "request_concurrency",
    "response_cache",
    "runner_version",
    "schema_version",
    "seed",
    "seeds",
    "unit_concurrency",
)
TASK_CONFIG_FIELDS = (
    "adapter_version",
    "batch_size",
    "benchmark",
    "dataset_path",
    "dataset_sha256",
    "evaluation_sha256",
    "input_sha256",
    "judge_model",
    "judge_model_official",
    "media_source",
    "official_judge_model",
    "official_metric",
    "primary_metric",
    "question_count",
    "scorer_protocol",
    "source_repository",
    "source_revision",
    "task",
    "variant",
)
SAMPLE_INPUT_FIELDS = (
    "benchmark",
    "dataset_sha256",
    "evaluation_sha256",
    "judge_model",
    "metadata",
    "prompt",
    "question_id",
    "ref_at_300",
    "references",
    "sample_id",
    "schema_version",
    "scorer_protocol",
    "task",
    "unit_id",
)
ALLOWED_STORE_METADATA_DIFFERENCES = frozenset({"embedding.space_id"})
STORE_COUNT_FIELDS = (
    "memory_records",
    "media_assets",
    "speech_analyses",
    "search_index_queue",
)
LIVE_CAPTURE = "analyzer_live_sqlite"
MANUAL_CAPTURE = "manual_from_live_sqlite_before_cleanup"


class AuditError(ValueError):
    """A benchmark artifact violated the preregistered comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label}: expected an object")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AuditError(f"{label}: expected a finite number")
    result = float(value)
    _require(math.isfinite(result), f"{label}: expected a finite number")
    _require(not positive or result > 0, f"{label}: expected a positive number")
    return result


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{label}: expected an integer")
    _require(not positive or value > 0, f"{label}: expected a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{path}: cannot read JSON: {error}") from error


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
        rows = [
            _object(json.loads(line), f"{path}:{line_number}")
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1)
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{path}: cannot read JSONL: {error}") from error
    return rows, hashlib.sha256(payload).hexdigest()


def _read_stores(data_root: Path, label: str) -> list[dict[str, Any]]:
    databases = sorted(data_root.rglob("state.sqlite3"))
    _require(bool(databases), f"{label}: no state.sqlite3 below data_root {data_root}")
    result = []
    units: set[str] = set()
    with ExitStack() as stack:
        for database in databases:
            lock_path = database.parent / ".mindbridge.lock"
            _require(lock_path.is_file(), f"{label}: missing store lock {lock_path}")
            lock = stack.enter_context(lock_path.open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise AuditError(f"{label}: store is still owned: {database.parent}") from error
        for database in databases:
            unit = database.parent.name
            _require(unit.startswith("unit-"), f"{label}: unexpected store path {database}")
            _require(unit not in units, f"{label}: duplicate physical store for {unit}")
            units.add(unit)
            try:
                with sqlite3.connect(
                    f"{database.resolve().as_uri()}?mode=ro", uri=True
                ) as connection:
                    check = connection.execute("PRAGMA integrity_check").fetchall()
                    counts = {
                        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        for table in STORE_COUNT_FIELDS
                    }
                    rows = connection.execute(
                        "SELECT key, value FROM store_metadata ORDER BY key"
                    ).fetchall()
            except sqlite3.Error as error:
                raise AuditError(f"{label}/{unit}: cannot audit SQLite: {error}") from error
            _require(check == [("ok",)], f"{label}/{unit}: SQLite integrity_check failed")
            _require(
                bool(rows)
                and all(isinstance(key, str) and isinstance(value, str) for key, value in rows),
                f"{label}/{unit}: invalid store metadata",
            )
            metadata = dict(rows)
            _require(len(metadata) == len(rows), f"{label}/{unit}: duplicate store metadata key")
            _require(
                "embedding.space_id" in metadata,
                f"{label}/{unit}: missing embedding.space_id",
            )
            result.append(
                {
                    "unit": unit,
                    "relative_database_path": database.relative_to(data_root).as_posix(),
                    "sqlite_integrity_check": "ok",
                    "row_counts": counts,
                    "store_metadata": metadata,
                }
            )
    return result


def _capture_store_audit(result_dir: Path) -> dict[str, Any]:
    result_path = result_dir / "results.json"
    samples_path = result_dir / "samples.jsonl"
    document = _load_object(result_path)
    _require(document.get("status") == "completed", f"{result_dir}: result is not completed")
    samples_sha256 = _sha256(samples_path)
    _require(
        document.get("samples_sha256") == samples_sha256,
        f"{result_dir}: samples SHA-256 mismatch",
    )
    data_root = Path(str(document.get("data_root"))).resolve()
    _require(data_root.is_dir(), f"{result_dir}: data_root does not exist: {data_root}")
    stat = data_root.stat()
    return {
        "schema_version": 1,
        "capture_method": LIVE_CAPTURE,
        "run_id": document.get("run_id"),
        "results_sha256": _sha256(result_path),
        "samples_sha256": samples_sha256,
        "data_root": str(data_root),
        "data_root_stat": {"device": stat.st_dev, "inode": stat.st_ino},
        "exclusive_lock_available": True,
        "stores": _read_stores(data_root, str(result_dir)),
    }


def _store_audit(
    result_dir: Path,
    document: dict[str, Any],
    results_sha256: str,
    samples_sha256: str,
    label: str,
) -> dict[str, Any]:
    sidecar = _load_object(result_dir / "store-audit.json")
    _require(sidecar.get("schema_version") == 1, f"{label}: wrong store-audit schema")
    method = sidecar.get("capture_method")
    _require(method in (LIVE_CAPTURE, MANUAL_CAPTURE), f"{label}: invalid capture method")
    _require(sidecar.get("run_id") == document.get("run_id"), f"{label}: sidecar run mismatch")
    _require(
        sidecar.get("results_sha256") == results_sha256,
        f"{label}: sidecar results hash mismatch",
    )
    _require(
        sidecar.get("samples_sha256") == samples_sha256,
        f"{label}: sidecar samples hash mismatch",
    )
    data_root = Path(str(document.get("data_root"))).resolve()
    _require(
        isinstance(sidecar.get("data_root"), str)
        and Path(sidecar["data_root"]).resolve() == data_root,
        f"{label}: sidecar data_root mismatch",
    )
    root_stat = sidecar.get("data_root_stat")
    if method == LIVE_CAPTURE:
        stat = _object(root_stat, f"{label}.data_root_stat")
        _integer(stat.get("device"), f"{label}.data_root_stat.device")
        _integer(stat.get("inode"), f"{label}.data_root_stat.inode", positive=True)
    else:
        _require(root_stat is None, f"{label}: manual sidecar data_root_stat must be null")
    _require(
        sidecar.get("exclusive_lock_available") is True,
        f"{label}: exclusive store lock was not proven",
    )
    raw_stores = sidecar.get("stores")
    _require(isinstance(raw_stores, list) and bool(raw_stores), f"{label}: no store audits")
    stores: dict[str, dict[str, Any]] = {}
    assert isinstance(raw_stores, list)
    for index, raw_store in enumerate(raw_stores):
        store = _object(raw_store, f"{label}.stores[{index}]")
        unit = store.get("unit")
        _require(
            isinstance(unit, str) and unit.startswith("unit-"),
            f"{label}.stores[{index}]: invalid unit",
        )
        assert isinstance(unit, str)
        _require(unit not in stores, f"{label}: duplicate store unit {unit}")
        _require(
            store.get("sqlite_integrity_check") == "ok",
            f"{label}/{unit}: SQLite integrity check failed",
        )
        counts = _object(store.get("row_counts"), f"{label}/{unit}.row_counts")
        for field in STORE_COUNT_FIELDS:
            count = _integer(counts.get(field), f"{label}/{unit}.row_counts.{field}")
            _require(
                field == "search_index_queue" or count > 0,
                f"{label}/{unit}: empty {field}",
            )
        _require(counts["search_index_queue"] == 0, f"{label}/{unit}: outbox is not empty")
        metadata = _object(store.get("store_metadata"), f"{label}/{unit}.store_metadata")
        _require(
            bool(metadata)
            and all(
                isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
            ),
            f"{label}/{unit}: invalid store metadata",
        )
        _require("embedding.space_id" in metadata, f"{label}/{unit}: missing embedding space")
        stores[unit] = {"row_counts": counts, "store_metadata": metadata}
    return {
        "capture_method": method,
        "data_root_stat": root_stat,
        "stores": stores,
    }


def _validate_usage(usage: object, label: str) -> dict[str, Any]:
    aggregate = _object(usage, label)
    modules = _object(aggregate.get("by_module"), f"{label}.by_module")
    _require(bool(modules), f"{label}.by_module: expected at least one module")
    for module_name, raw_module in modules.items():
        module = _object(raw_module, f"{label}.by_module.{module_name}")
        _require(module.get("complete") is True, f"{label}.{module_name}: incomplete usage")
        _require(
            module.get("modality_breakdown_complete") is True,
            f"{label}.{module_name}: incomplete modality usage",
        )
        _require(
            module.get("unreported_request_count") == 0,
            f"{label}.{module_name}: unreported requests",
        )
        _integer(module.get("request_count"), f"{label}.{module_name}.request_count")
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            _number(module.get(field), f"{label}.{module_name}.{field}")
    _require(aggregate.get("complete") is True, f"{label}: incomplete aggregate usage")
    _require(
        aggregate.get("modality_breakdown_complete") is True,
        f"{label}: incomplete aggregate modality usage",
    )
    _require(
        aggregate.get("unreported_request_count") == 0,
        f"{label}: aggregate has unreported requests",
    )
    return aggregate


def _artifact(root: Path, version: str, slug: str) -> dict[str, Any]:
    label = f"{version}-{slug}"
    result_dir = root / label
    document = _load_object(result_dir / "results.json")
    rows, samples_sha256 = _load_rows(result_dir / "samples.jsonl")
    _require(document.get("status") == "completed", f"{label}: status is not completed")
    _require(document.get("response_cache") is None, f"{label}: response cache configured")
    _require(document.get("cached_response_count") == 0, f"{label}: cached responses found")
    _require(document.get("cached_judge_count") == 0, f"{label}: cached judges found")
    _require(document.get("log_samples") is True, f"{label}: sample logging disabled")
    _require(
        document.get("samples_sha256") == samples_sha256,
        f"{label}: samples SHA-256 mismatch",
    )
    for field in DOCUMENT_CONFIG_FIELDS:
        _require(field in document, f"{label}: missing document config {field}")

    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise AuditError(f"{label}: expected one task")
    task = _object(tasks[0], f"{label}.task")
    for field in TASK_CONFIG_FIELDS:
        _require(field in task, f"{label}: missing task config {field}")
    expected = TASKS[slug]
    _require(task.get("task") == expected["task"], f"{label}: wrong task")
    _require(document.get("limit") == expected["limit"], f"{label}: wrong fixed limit")
    _require(document.get("offset") == 0, f"{label}: wrong fixed offset")
    _require(task.get("dataset_sha256") == expected["dataset_sha256"], f"{label}: wrong dataset")
    _require(task.get("primary_metric") == "accuracy", f"{label}: primary metric is not accuracy")
    _require(task.get("score_valid") is True, f"{label}: score is invalid")
    _require(task.get("error_count") == 0, f"{label}: task errors found")
    _require(task.get("ingest_failure_count") == 0, f"{label}: ingest failures found")

    dataset_path = Path(str(task["dataset_path"])).resolve()
    _require(dataset_path.is_file(), f"{label}: dataset path does not exist: {dataset_path}")
    _require(
        _sha256(dataset_path) == task["dataset_sha256"], f"{label}: dataset file hash mismatch"
    )
    manifest_path = Path(str(document["media_manifest_path"])).resolve()
    _require(manifest_path.is_file(), f"{label}: media manifest does not exist: {manifest_path}")
    _require(
        _sha256(manifest_path) == expected["manifest_sha256"],
        f"{label}: preregistered media-manifest file hash mismatch",
    )

    question_count = _integer(task.get("question_count"), f"{label}.question_count", positive=True)
    score = _object(task.get("score"), f"{label}.score")
    score_mean = _number(score.get("mean"), f"{label}.score.mean")
    _require(score.get("sample_count") == question_count, f"{label}: score sample count mismatch")
    _require(len(rows) == question_count, f"{label}: sample count mismatch")
    sample_ids = [row.get("sample_id") for row in rows]
    _require(all(isinstance(value, str) for value in sample_ids), f"{label}: bad sample ID")
    _require(len(set(sample_ids)) == question_count, f"{label}: duplicate sample ID")

    sample_scores = []
    for index, row in enumerate(rows):
        row_label = f"{label}.sample[{index}]"
        for field in SAMPLE_INPUT_FIELDS:
            _require(field in row, f"{row_label}: missing input field {field}")
        _require(row.get("cached") is False, f"{row_label}: cached response")
        _require(row.get("judge_cached") is False, f"{row_label}: cached judge")
        _require(row.get("error_code") is None, f"{row_label}: sample error")
        _require(row.get("scorer_error") is None, f"{row_label}: scorer error")
        _require(row.get("ingest_failure_count") == 0, f"{row_label}: ingest failure")
        sample_score = _number(row.get("score"), f"{row_label}.score")
        _require(sample_score in (0.0, 1.0), f"{row_label}: expected binary accuracy score")
        sample_scores.append(sample_score)
        _number(row.get("latency_ms"), f"{row_label}.latency_ms", positive=True)
        memory_ids = row.get("memory_ids")
        if not isinstance(memory_ids, list) or not all(
            isinstance(memory_id, str) for memory_id in memory_ids
        ):
            raise AuditError(f"{row_label}: invalid memory_ids")
        _require(len(set(memory_ids)) == len(memory_ids), f"{row_label}: duplicate memory ID")
    correct_count = int(sum(sample_scores))
    _require(
        math.isclose(score_mean, correct_count / question_count, rel_tol=0, abs_tol=1e-12),
        f"{label}: task score does not match samples",
    )

    performance = _object(task.get("performance"), f"{label}.performance")
    nodes = _object(performance.get("nodes"), f"{label}.performance.nodes")
    generation = _object(nodes.get("mindbridge.model.generation"), f"{label}.generation")
    ask = _object(nodes.get("mindbridge.ask"), f"{label}.ask")
    _require(generation.get("count") == question_count, f"{label}: generation count mismatch")
    _require(ask.get("count") == question_count, f"{label}: ask count mismatch")
    _number(generation.get("average_ms"), f"{label}.generation.average_ms", positive=True)
    _number(ask.get("average_ms"), f"{label}.ask.average_ms", positive=True)
    ttft = _object(generation.get("ttft_ms"), f"{label}.generation.ttft_ms")
    _require(ttft.get("count") == question_count, f"{label}: TTFT count mismatch")
    _number(ttft.get("average"), f"{label}.generation.ttft_ms.average", positive=True)
    usage = _validate_usage(performance.get("token_usage"), f"{label}.token_usage")
    generation_usage = _object(
        _object(usage["by_module"], f"{label}.token_usage.by_module").get("generation"),
        f"{label}.token_usage.generation",
    )
    _require(
        generation_usage.get("request_count") == question_count,
        f"{label}: generation usage count mismatch",
    )

    data_root = Path(str(document.get("data_root"))).resolve()
    results_sha256 = _sha256(result_dir / "results.json")
    store_audit = _store_audit(
        result_dir,
        document,
        results_sha256,
        samples_sha256,
        label,
    )
    return {
        "label": label,
        "result_dir": result_dir.resolve(),
        "document": document,
        "task": task,
        "rows": rows,
        "nodes": nodes,
        "usage": usage,
        "data_root": data_root,
        "store_audit": store_audit,
        "samples_sha256": samples_sha256,
        "results_sha256": results_sha256,
        "correct_count": correct_count,
        "score_mean": score_mean,
    }


def _same_fields(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> None:
    for field in fields:
        _require(left[field] == right[field], f"{label}: config differs at {field}")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _comparison(v4: float | int, v5: float | int) -> dict[str, Any]:
    before = float(v4)
    after = float(v5)
    return {
        "v4": v4,
        "v5": v5,
        "delta": after - before,
        "delta_percent": None if before == 0 else (after - before) / before * 100,
    }


def _node_comparisons(
    v4: dict[str, Any], v5: dict[str, Any], label: str
) -> dict[str, dict[str, Any]]:
    _require(v4.keys() == v5.keys(), f"{label}: performance node sets differ")
    result = {}
    for name in sorted(v4):
        before = _object(v4[name], f"{label}.v4.nodes.{name}")
        after = _object(v5[name], f"{label}.v5.nodes.{name}")
        result[name] = {
            "count": _comparison(
                _integer(before.get("count"), f"{label}.v4.{name}.count"),
                _integer(after.get("count"), f"{label}.v5.{name}.count"),
            ),
            "average_ms": _comparison(
                _number(before.get("average_ms"), f"{label}.v4.{name}.average_ms"),
                _number(after.get("average_ms"), f"{label}.v5.{name}.average_ms"),
            ),
        }
    return result


def _token_comparisons(v4: dict[str, Any], v5: dict[str, Any], label: str) -> dict[str, Any]:
    left_modules = _object(v4["by_module"], f"{label}.v4.usage.by_module")
    right_modules = _object(v5["by_module"], f"{label}.v5.usage.by_module")
    _require(left_modules.keys() == right_modules.keys(), f"{label}: usage module sets differ")

    def summarize(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {
            field: _comparison(
                _number(left.get(field), f"{label}.v4.usage.{field}"),
                _number(right.get(field), f"{label}.v5.usage.{field}"),
            )
            for field in ("input_tokens", "output_tokens", "total_tokens")
        } | {
            "request_count": _comparison(
                _integer(left.get("request_count"), f"{label}.v4.usage.request_count"),
                _integer(right.get("request_count"), f"{label}.v5.usage.request_count"),
            )
        }

    return {
        "aggregate": summarize(v4, v5),
        "by_module": {
            name: summarize(
                _object(left_modules[name], f"{label}.v4.usage.{name}"),
                _object(right_modules[name], f"{label}.v5.usage.{name}"),
            )
            for name in sorted(left_modules)
        },
    }


def _retrieval(rows_v4: list[dict[str, Any]], rows_v5: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_equal = 0
    set_equal = 0
    top1_equal = 0
    overlaps = []
    jaccards = []
    for before, after in zip(rows_v4, rows_v5, strict=True):
        left = before["memory_ids"]
        right = after["memory_ids"]
        left_set = set(left)
        right_set = set(right)
        intersection = left_set & right_set
        union = left_set | right_set
        ordered_equal += left == right
        set_equal += left_set == right_set
        top1_equal += bool(left and right and left[0] == right[0])
        overlaps.append(len(intersection))
        jaccards.append(len(intersection) / len(union) if union else 1.0)
    count = len(rows_v4)
    return {
        "sample_count": count,
        "ordered_equal_count": ordered_equal,
        "set_equal_count": set_equal,
        "top1_equal_count": top1_equal,
        "mean_intersection_count": statistics.fmean(overlaps),
        "minimum_intersection_count": min(overlaps),
        "mean_set_jaccard": statistics.fmean(jaccards),
    }


def _pair(slug: str, v4: dict[str, Any], v5: dict[str, Any]) -> dict[str, Any]:
    _same_fields(v4["document"], v5["document"], DOCUMENT_CONFIG_FIELDS, slug)
    _same_fields(v4["task"], v5["task"], TASK_CONFIG_FIELDS, slug)
    _require(v4["data_root"] != v5["data_root"], f"{slug}: data_root is shared")
    stores_v4 = v4["store_audit"]["stores"]
    stores_v5 = v5["store_audit"]["stores"]
    _require(
        stores_v4.keys() == stores_v5.keys(),
        f"{slug}: store unit sets differ",
    )
    embedding_spaces: dict[str, dict[str, str]] = {}
    for unit in stores_v4:
        before_store = stores_v4[unit]
        after_store = stores_v5[unit]
        _require(
            before_store["row_counts"] == after_store["row_counts"],
            f"{slug}/{unit}: durable row counts differ",
        )
        before = before_store["store_metadata"]
        after = after_store["store_metadata"]
        changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        _require(
            changed == ALLOWED_STORE_METADATA_DIFFERENCES,
            f"{slug}/{unit}: store metadata differences {sorted(changed)}; "
            "only embedding.space_id is allowed",
        )
        embedding_spaces[unit] = {
            "v4": before["embedding.space_id"],
            "v5": after["embedding.space_id"],
        }

    rows_v4 = v4["rows"]
    rows_v5 = v5["rows"]
    _require(
        [row["sample_id"] for row in rows_v4] == [row["sample_id"] for row in rows_v5],
        f"{slug}: sample identity/order differs",
    )
    for index, (before, after) in enumerate(zip(rows_v4, rows_v5, strict=True)):
        for field in SAMPLE_INPUT_FIELDS:
            _require(
                before[field] == after[field],
                f"{slug}: sample {index} input differs at {field}",
            )

    quality = _comparison(v4["score_mean"], v5["score_mean"])
    quality["metric"] = v4["task"]["primary_metric"]
    quality["v4_correct_count"] = v4["correct_count"]
    quality["v5_correct_count"] = v5["correct_count"]
    quality["delta_percentage_points"] = (v5["score_mean"] - v4["score_mean"]) * 100
    quality["gains"] = sum(
        before["score"] < after["score"] for before, after in zip(rows_v4, rows_v5, strict=True)
    )
    quality["losses"] = sum(
        before["score"] > after["score"] for before, after in zip(rows_v4, rows_v5, strict=True)
    )

    sample_latency_v4 = [float(row["latency_ms"]) for row in rows_v4]
    sample_latency_v5 = [float(row["latency_ms"]) for row in rows_v5]
    generation_v4 = v4["nodes"]["mindbridge.model.generation"]
    generation_v5 = v5["nodes"]["mindbridge.model.generation"]
    result = {
        "integrity": {
            "status_completed": True,
            "score_valid": True,
            "sample_identity_order_and_inputs_equal": True,
            "configuration_equal": True,
            "physical_data_roots_distinct": True,
            "store_metadata_difference_allowlist": sorted(ALLOWED_STORE_METADATA_DIFFERENCES),
            "embedding_spaces": embedding_spaces,
            "zero_errors_ingest_failures_and_cache": True,
            "usage_complete": True,
        },
        "artifacts": {
            version: {
                "result_dir": str(artifact["result_dir"]),
                "data_root": str(artifact["data_root"]),
                "run_id": artifact["document"].get("run_id"),
                "results_sha256": artifact["results_sha256"],
                "samples_sha256": artifact["samples_sha256"],
                "store_audit_capture_method": artifact["store_audit"]["capture_method"],
                "data_root_stat": artifact["store_audit"]["data_root_stat"],
            }
            for version, artifact in (("v4", v4), ("v5", v5))
        },
        "quality": quality,
        "latency_ms": {
            "nodes": _node_comparisons(v4["nodes"], v5["nodes"], slug),
            "ttft": _comparison(
                _number(generation_v4["ttft_ms"]["average"], f"{slug}.v4.ttft"),
                _number(generation_v5["ttft_ms"]["average"], f"{slug}.v5.ttft"),
            ),
            "sample_p50": _comparison(
                _percentile(sample_latency_v4, 0.50),
                _percentile(sample_latency_v5, 0.50),
            ),
            "sample_p95": _comparison(
                _percentile(sample_latency_v4, 0.95),
                _percentile(sample_latency_v5, 0.95),
            ),
        },
        "tokens": _token_comparisons(v4["usage"], v5["usage"], slug),
        "retrieval_id_overlap": _retrieval(rows_v4, rows_v5),
    }
    result["verdict"] = "accept" if quality["delta"] >= 0 else "reject"
    return result


def audit(root: Path) -> dict[str, Any]:
    missing = [
        root / f"{version}-{slug}" / filename
        for slug in TASKS
        for version in VERSIONS
        for filename in ("results.json", "samples.jsonl", "store-audit.json")
        if not (root / f"{version}-{slug}" / filename).is_file()
    ]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise AuditError(f"missing benchmark artifacts:\n{rendered}")

    artifacts = {
        slug: {version: _artifact(root, version, slug) for version in VERSIONS} for slug in TASKS
    }
    roots = [artifact["data_root"] for task in artifacts.values() for artifact in task.values()]
    _require(len(set(roots)) == len(roots), "experiment: all four data_root paths must be distinct")
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            _require(
                not left.is_relative_to(right) and not right.is_relative_to(left),
                "experiment: data_root directories must not be nested",
            )

    tasks = {slug: _pair(slug, pair["v4"], pair["v5"]) for slug, pair in artifacts.items()}
    regressions = [slug for slug, result in tasks.items() if result["verdict"] == "reject"]
    evidence_limitations = [
        f"{version}-{slug}: data_root device/inode unavailable after cleanup; "
        "sidecar manually transcribed from the live SQLite audit"
        for slug, pair in artifacts.items()
        for version, artifact in pair.items()
        if artifact["store_audit"]["capture_method"] == MANUAL_CAPTURE
    ]
    return {
        "schema_version": 1,
        "experiment": "Jina video metadata recipe v4 versus v5",
        "root": str(root.resolve()),
        "integrity": {
            "ready": True,
            "all_four_data_root_paths_distinct_and_not_nested": True,
            "paired_configuration_equal_except_embedding_space_recipe": True,
            "evidence_limitations": evidence_limitations,
        },
        "tasks": tasks,
        "verdict": {
            "decision": "reject" if regressions else "accept",
            "quality_rule": "reject v5 if primary accuracy regresses on either task",
            "quality_regressions": regressions,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Jina video metadata paired ablation",
        "",
        f"Verdict: **{report['verdict']['decision'].upper()}**. "
        "Quality is the first-priority gate; any task regression rejects v5.",
        "",
        "| Task | v4 accuracy | v5 accuracy | Delta | TTFT delta | Generation tokens delta | Retrieval set Jaccard | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for slug, task in report["tasks"].items():
        quality = task["quality"]
        ttft = task["latency_ms"]["ttft"]
        tokens = task["tokens"]["by_module"]["generation"]["total_tokens"]
        retrieval = task["retrieval_id_overlap"]
        lines.append(
            f"| {slug} | {quality['v4']:.4%} | {quality['v5']:.4%} | "
            f"{quality['delta_percentage_points']:+.4f} pp | {ttft['delta_percent']:+.4f}% | "
            f"{tokens['delta_percent']:+.4f}% | {retrieval['mean_set_jaccard']:.4f} | "
            f"{task['verdict']} |"
        )
    lines.extend(
        (
            "",
            "All pairs passed identity/order/input-hash/configuration checks, use distinct "
            "physical stores, have zero errors/ingest/cache, and report complete usage. "
            "Store metadata differs only at `embedding.space_id`.",
            "",
            "Node-level latency, TTFT, tokens by module, artifact hashes, embedding-space "
            "identities, and retrieval-ID overlap are preserved in the JSON companion.",
            "",
        )
    )
    return "\n".join(lines)


def _replace(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed paired audit for the preregistered Jina v4/v5 video runs.",
        epilog=(
            "Run --capture-store-audit RESULT_DIR before deleting each completed data_root. "
            "Run without --write for the final JSON on stdout; add --write to persist it."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--capture-store-audit", type=Path, metavar="RESULT_DIR")
    arguments = parser.parse_args()
    try:
        if arguments.capture_store_audit is not None:
            if arguments.write:
                parser.error("--write cannot be combined with --capture-store-audit")
            result_dir = arguments.capture_store_audit.resolve()
            captured = json.dumps(_capture_store_audit(result_dir), indent=2, sort_keys=True) + "\n"
            _replace(result_dir / "store-audit.json", captured)
            print(captured, end="")
            return 0
        report = audit(arguments.root.resolve())
    except AuditError as error:
        parser.exit(1, f"error: {error}\n")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.write:
        _replace(OUTPUT_JSON, rendered)
        _replace(OUTPUT_MARKDOWN, _markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
