"""Fail-closed validation and manifest generation for the frozen final pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final"
LOCKED_DATA = Path("/dev/shm/mindbridge-ar-locked-b2d7f2918105")
BASELINE_NAME = "baseline-ba4bcced90b9-locked-v2"
CURRENT_NAME = "current-b2d7f2918105-locked-v2"
REVISIONS = {
    "baseline": "ba4bcced90b916bf28265576320639b8c1a0218a",
    "current": "b2d7f291810530b4788d7890deb6302685acb1ab",
}
EXPECTED_REPOSITORY = {
    "baseline": {
        "product_tree": "56d4bd6782bd395cb45e8b6e1b53b0f691d8b784",
        "pyproject_blob": "774126eaa95deec1c650156a8b8f9c7201b49176",
        "uv_lock_blob": "706d6bb754793976ba4821f5574c110032eea356",
    },
    "current": {
        "product_tree": "2360c99440ac2d4db6568f723330af52be4e8b0e",
        "pyproject_blob": "774126eaa95deec1c650156a8b8f9c7201b49176",
        "uv_lock_blob": "706d6bb754793976ba4821f5574c110032eea356",
    },
}
EXPECTED_GPU = {
    "cuda_visible_devices": "0",
    "driver_version": "580.178.04",
    "index": 0,
    "name": "NVIDIA GeForce RTX 5090",
    "uuid": "GPU-6fa4d834-e033-e492-66f5-9d2f3792c4dd",
}
EXPECTED_WRAPPER_SHA256 = "370179d9e12a21f1ed37e7bc8a7bb60e1c6137d02bde86f09ff243a08e19be32"
LOCKED_IDENTITIES = ROOT / "locked-identities.json"
LOCKED_IDENTITIES_SHA256 = "481d3c40d3ca3a060fc7cfbab72ab4abaa1f5819d05132a5c540ebee3ad54ccc"
EXPECTED_MODEL = {
    "adapter": "mindbridge",
    "device": "cuda",
    "embedding_dimension": 1024,
    "embedding_model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
    "embedding_revision": "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
    "embedding_warmup": {"count": 1, "task": "retrieval.query"},
    "generation_base_url": "https://inner-prism.cece.com/api/v1",
    "generation_kwargs": "temperature=0,do_sample=false,seed=20260830,enable_thinking=false",
    "generation_min_video_seconds": 2.0,
    "generation_modalities": ["image", "text", "video"],
    "generation_model": "qwen3.8-flash",
    "generation_seed": 20260830,
    "generation_temperature": 0.0,
    "timeout_seconds": 600.0,
    "transcription_model": "FunAudioLLM/Fun-ASR-Nano-2512",
}
TASKS = {
    "locomo-locked": "locomo-refined",
    "atm-hard": "atm-bench-hard",
    "gallery-locked": "mem-gallery",
    "m3-locked": "m3-bench-robot",
    "egolife-q50-99": "egolifeqa",
}
EXPECTED_RUNS: dict[str, dict[str, object]] = {
    "locomo-locked": {
        "allow_unverified_data": False,
        "limit": 3,
        "offset": 3,
        "question_count": 455,
        "request_concurrency": 16,
        "unit_concurrency": 4,
    },
    "atm-hard": {
        "allow_unverified_data": False,
        "limit": None,
        "offset": 0,
        "question_count": 31,
        "request_concurrency": 8,
        "unit_concurrency": 1,
    },
    "gallery-locked": {
        "allow_unverified_data": False,
        "limit": 5,
        "offset": 5,
        "question_count": 407,
        "request_concurrency": 8,
        "unit_concurrency": 2,
    },
    "m3-locked": {
        "allow_unverified_data": True,
        "limit": None,
        "offset": 0,
        "question_count": 87,
        "request_concurrency": 8,
        "unit_concurrency": 1,
    },
    "egolife-q50-99": {
        "allow_unverified_data": False,
        "limit": 50,
        "offset": 50,
        "question_count": 50,
        "request_concurrency": 8,
        "unit_concurrency": 1,
    },
}
PAIR_DOCUMENT_FIELDS = (
    "allow_unverified_data",
    "bootstrap_samples",
    "environment",
    "judge",
    "limit",
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
PAIR_TASK_FIELDS = (
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
    "primary_metric",
    "question_count",
    "scorer_protocol",
    "source_repository",
    "source_revision",
    "task",
    "variant",
)
PAIR_SAMPLE_FIELDS = (
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


def _finite(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    number = float(value)
    return math.isfinite(number) and (not positive or number > 0)


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def locked_identities(namespace: str) -> dict[str, dict[str, object]]:
    payload = LOCKED_IDENTITIES.read_bytes()
    _require(
        hashlib.sha256(payload).hexdigest() == LOCKED_IDENTITIES_SHA256,
        "locked identity registry changed",
    )
    document = json.loads(payload)
    values = document.get(f"{namespace}_identities") if isinstance(document, dict) else None
    if not isinstance(values, dict) or not all(
        isinstance(name, str) and isinstance(identity, dict) for name, identity in values.items()
    ):
        raise ValueError(f"invalid {namespace} identity registry")
    return values


def validate_media_manifest(
    result_dir: Path,
    document: dict[str, Any],
    task_name: str,
    identity: dict[str, object],
    label: str,
) -> None:
    _require("media_manifest_path" in document, f"{label}: missing manifest field")
    input_sha256 = identity.get("input_sha256")
    if not isinstance(input_sha256, dict):
        raise ValueError(f"{label}: invalid pinned inputs")
    expected_hash = input_sha256.get("media_manifest")
    if expected_hash is None:
        _require(document.get("media_manifest_path") is None, f"{label}: unexpected manifest")
        return
    path = result_dir / "media-manifest.json"
    _require(
        document.get("media_manifest_path") == str(path)
        and path.is_file()
        and not path.is_symlink(),
        f"{label}: wrong media manifest path",
    )
    manifest = _load(path)
    tasks = manifest.get("tasks")
    task_manifest = tasks.get(task_name) if isinstance(tasks, dict) else None
    if not isinstance(task_manifest, dict):
        raise ValueError(f"{label}: missing task media manifest")
    payload = json.dumps(
        task_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _require(hashlib.sha256(payload).hexdigest() == expected_hash, f"{label}: wrong manifest hash")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_provenance(
    result_dir: Path,
    side: str,
    result_path: Path,
    samples_hash: str,
    *,
    expected_sequence: dict[str, object],
) -> None:
    provenance_path = result_dir / "run-provenance.json"
    _require(
        provenance_path.is_file() and not provenance_path.is_symlink(),
        f"{side}/{result_dir.name}: missing run provenance",
    )
    provenance = _load(provenance_path)
    _require(
        provenance.get("schema_version") == 2,
        f"{side}/{result_dir.name}: bad provenance schema",
    )
    _require(
        provenance.get("expected_revision") == REVISIONS[side]
        and provenance.get("worktree_revision") == REVISIONS[side],
        f"{side}/{result_dir.name}: wrong product revision",
    )
    for field, expected in EXPECTED_REPOSITORY[side].items():
        _require(
            provenance.get(field) == expected,
            f"{side}/{result_dir.name}: wrong provenance {field}",
        )
    _require(
        provenance.get("results_sha256") == hashlib.sha256(result_path.read_bytes()).hexdigest(),
        f"{side}/{result_dir.name}: result provenance mismatch",
    )
    _require(
        provenance.get("samples_sha256") == samples_hash,
        f"{side}/{result_dir.name}: sample provenance mismatch",
    )
    _require(
        provenance.get("gpu") == EXPECTED_GPU,
        f"{side}/{result_dir.name}: wrong GPU provenance",
    )
    _require(
        provenance.get("wrapper_sha256") == EXPECTED_WRAPPER_SHA256,
        f"{side}/{result_dir.name}: wrong provenance wrapper",
    )
    result = _load(result_path)
    _require(provenance.get("output_dir") == str(result_dir), f"{side}: wrong output binding")
    _require(provenance.get("data_root") == result.get("data_root"), f"{side}: wrong data binding")
    _require(provenance.get("run_id") == result.get("run_id"), f"{side}: wrong run binding")
    argv = provenance.get("argv")
    _require(
        isinstance(argv, list) and all(isinstance(value, str) for value in argv),
        f"{side}/{result_dir.name}: bad provenance argv",
    )
    argv_hash = hashlib.sha256(
        json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    _require(
        provenance.get("argv_sha256") == argv_hash,
        f"{side}/{result_dir.name}: argv provenance mismatch",
    )
    sequence = provenance.get("sequence")
    _require(sequence == expected_sequence, f"{side}/{result_dir.name}: wrong run sequence")
    started = provenance.get("started_at")
    ended = provenance.get("ended_at")
    try:
        started_at = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        ended_at = datetime.fromisoformat(str(ended).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{side}/{result_dir.name}: invalid provenance time") from error
    _require(ended_at >= started_at, f"{side}/{result_dir.name}: reversed provenance time")
    _require(
        _finite(provenance.get("duration_seconds"), positive=True),
        f"{side}/{result_dir.name}: invalid provenance duration",
    )
    attempt_path = result_dir.parent / f"{result_dir.name}.attempt.json"
    _require(
        attempt_path.is_file() and not attempt_path.is_symlink(),
        f"{side}/{result_dir.name}: missing attempt ledger",
    )
    attempt_hash = hashlib.sha256(attempt_path.read_bytes()).hexdigest()
    _require(
        provenance.get("attempt_ledger") == str(attempt_path)
        and provenance.get("attempt_ledger_sha256") == attempt_hash,
        f"{side}/{result_dir.name}: attempt ledger mismatch",
    )
    attempt = _load(attempt_path)
    _require(
        attempt.get("schema_version") == 1 and attempt.get("state") == "started",
        f"{side}/{result_dir.name}: invalid attempt ledger",
    )
    for field in (
        "argv",
        "argv_sha256",
        "data_root",
        "expected_revision",
        "gpu",
        "output_dir",
        "product_tree",
        "pyproject_blob",
        "run_id",
        "sequence",
        "started_at",
        "uv_lock_blob",
        "worktree",
        "worktree_revision",
        "wrapper_sha256",
    ):
        _require(
            attempt.get(field) == provenance.get(field),
            f"{side}/{result_dir.name}: attempt {field} mismatch",
        )


def _artifact(side: str, slug: str) -> dict[str, Any]:  # noqa: C901 - fail-closed artifact audit
    frozen_name = BASELINE_NAME if side == "baseline" else CURRENT_NAME
    result_dir = (FINAL / frozen_name / slug).resolve()
    expected_data_root = (LOCKED_DATA / frozen_name / slug).resolve()
    result_path = result_dir / "results.json"
    samples_path = result_dir / "samples.jsonl"
    _require(not result_dir.is_symlink(), f"{side}/{slug}: result directory is a symlink")
    _require(result_path.is_file(), f"{side}/{slug}: missing {result_path}")
    _require(samples_path.is_file(), f"{side}/{slug}: missing {samples_path}")
    _require(not result_path.is_symlink(), f"{side}/{slug}: results are a symlink")
    _require(not samples_path.is_symlink(), f"{side}/{slug}: samples are a symlink")
    document = _load(result_path)
    _require(
        all(field in document for field in PAIR_DOCUMENT_FIELDS),
        f"{side}/{slug}: result document is missing paired fields",
    )
    expected_run_id = f"{frozen_name}-{slug}"
    _require(document.get("run_id") == expected_run_id, f"{side}/{slug}: wrong run_id")
    _require(
        isinstance(document.get("data_root"), str)
        and Path(document["data_root"]).resolve() == expected_data_root,
        f"{side}/{slug}: wrong data_root",
    )
    _require(document.get("status") == "completed", f"{side}/{slug}: run not completed")
    _require(
        document.get("runner_version") == "mindbridge_eval_official_v9",
        f"{side}/{slug}: wrong runner",
    )
    _require(document.get("response_cache") is None, f"{side}/{slug}: response cache configured")
    _require(document.get("cached_response_count") == 0, f"{side}/{slug}: cached response")
    _require(document.get("cached_judge_count") == 0, f"{side}/{slug}: cached judge")
    _require(document.get("log_samples") is True, f"{side}/{slug}: samples not logged")
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{side}/{slug}: missing model config")
    for field, expected in EXPECTED_MODEL.items():
        _require(model.get(field) == expected, f"{side}/{slug}: wrong model {field}")
    for field, expected in {
        "bootstrap_samples": 2000,
        "num_fewshot": 0,
        "predict_only": False,
        "recall_limit": 20,
        "schema_version": 7,
        "seed": 20260830,
        "seeds": [20260830, 20260830, 20260830, 20260830],
    }.items():
        _require(document.get(field) == expected, f"{side}/{slug}: wrong {field}")
    for field in (
        "allow_unverified_data",
        "limit",
        "offset",
        "request_concurrency",
        "unit_concurrency",
    ):
        _require(document.get(field) == EXPECTED_RUNS[slug][field], f"{side}/{slug}: wrong {field}")

    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError(f"{side}/{slug}: expected one task")
    task = tasks[0]
    if not isinstance(task, dict) or task.get("task") != TASKS[slug]:
        raise ValueError(f"{side}/{slug}: wrong task")
    _require(
        all(field in task for field in PAIR_TASK_FIELDS),
        f"{side}/{slug}: task is missing paired fields",
    )
    question_count_value = EXPECTED_RUNS[slug]["question_count"]
    if isinstance(question_count_value, bool) or not isinstance(question_count_value, int):
        raise AssertionError(f"{slug}: invalid expected question count")
    question_count = question_count_value
    _require(task.get("question_count") == question_count, f"{side}/{slug}: bad question_count")
    identity = locked_identities("quality").get(slug)
    if identity is None:
        raise ValueError(f"{slug}: locked identity is not pinned")
    _require(
        identity.get("question_count") == question_count,
        f"{side}/{slug}: wrong pinned question count",
    )
    for field in ("dataset_sha256", "evaluation_sha256", "input_sha256"):
        _require(task.get(field) == identity[field], f"{side}/{slug}: wrong {field}")
    validate_media_manifest(result_dir, document, TASKS[slug], identity, f"{side}/{slug}")
    _require(task.get("score_valid") is True, f"{side}/{slug}: invalid score")
    _require(task.get("error_count") == 0, f"{side}/{slug}: task errors")
    _require(task.get("ingest_failure_count") == 0, f"{side}/{slug}: ingest failures")
    score = task.get("score")
    if not isinstance(score, dict) or not _finite(score.get("mean")):
        raise ValueError(f"{side}/{slug}: bad score")
    primary_metric = task.get("primary_metric")
    metrics = task.get("metrics")
    primary = metrics.get(primary_metric) if isinstance(metrics, dict) else None
    if not isinstance(primary_metric, str) or not isinstance(primary, dict):
        raise ValueError(f"{side}/{slug}: missing primary metric")

    performance = task.get("performance")
    nodes = performance.get("nodes") if isinstance(performance, dict) else None
    usage = performance.get("token_usage") if isinstance(performance, dict) else None
    generation_usage = (
        usage.get("by_module", {}).get("generation") if isinstance(usage, dict) else None
    )
    ask = nodes.get("mindbridge.ask") if isinstance(nodes, dict) else None
    generation = nodes.get("mindbridge.model.generation") if isinstance(nodes, dict) else None
    ttft = generation.get("ttft_ms") if isinstance(generation, dict) else None
    for label, value in (("ask", ask), ("generation", generation), ("ttft", ttft)):
        if not isinstance(value, dict):
            raise ValueError(f"{side}/{slug}: missing {label} span")
        _require(value.get("count") == question_count, f"{side}/{slug}: {label} count mismatch")
        _require(
            _finite(value.get("average" if label == "ttft" else "average_ms"), positive=True),
            f"{side}/{slug}: bad {label} average",
        )
    if not isinstance(generation_usage, dict):
        raise ValueError(f"{side}/{slug}: missing generation usage")
    _require(
        generation_usage.get("complete") is True, f"{side}/{slug}: incomplete generation usage"
    )
    _require(
        generation_usage.get("modality_breakdown_complete") is True,
        f"{side}/{slug}: incomplete modality usage",
    )
    _require(
        generation_usage.get("request_count") == question_count,
        f"{side}/{slug}: generation request mismatch",
    )
    _require(
        generation_usage.get("unreported_request_count") == 0,
        f"{side}/{slug}: unreported generation request",
    )
    _require(
        _finite(generation_usage.get("average_tokens"), positive=True),
        f"{side}/{slug}: bad generation tokens",
    )
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        _require(
            _nonnegative_integer(generation_usage.get(field)),
            f"{side}/{slug}: bad generation {field}",
        )
    total_tokens = generation_usage["total_tokens"]
    _require(
        total_tokens == generation_usage["input_tokens"] + generation_usage["output_tokens"],
        f"{side}/{slug}: generation token total mismatch",
    )
    _require(
        math.isclose(
            float(generation_usage["average_tokens"]),
            total_tokens / question_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{side}/{slug}: generation token average mismatch",
    )
    modality_total = 0
    for field in ("input_by_modality", "output_by_modality"):
        values = generation_usage.get(field)
        if not isinstance(values, dict):
            raise ValueError(f"{side}/{slug}: missing generation {field}")
        _require(
            all(
                isinstance(key, str) and _nonnegative_integer(value)
                for key, value in values.items()
            ),
            f"{side}/{slug}: bad generation {field}",
        )
        modality_total += sum(values.values())
    _require(modality_total == total_tokens, f"{side}/{slug}: modality token total mismatch")

    samples_bytes = samples_path.read_bytes()
    samples_hash = hashlib.sha256(samples_bytes).hexdigest()
    _require(
        document.get("samples_sha256") == samples_hash, f"{side}/{slug}: samples hash mismatch"
    )
    rows = [json.loads(line) for line in samples_bytes.decode().splitlines()]
    _require(len(rows) == question_count, f"{side}/{slug}: sample count mismatch")
    _require(all(isinstance(row, dict) for row in rows), f"{side}/{slug}: non-object sample")
    _require(
        all(all(field in row for field in PAIR_SAMPLE_FIELDS) for row in rows),
        f"{side}/{slug}: sample is missing paired fields",
    )
    sample_ids = [row.get("sample_id") for row in rows]
    _require(all(isinstance(value, str) for value in sample_ids), f"{side}/{slug}: bad sample_id")
    _require(len(set(sample_ids)) == question_count, f"{side}/{slug}: duplicate sample_id")
    _require(all(row.get("cached") is False for row in rows), f"{side}/{slug}: cached sample")
    _require(
        all(row.get("judge_cached") is False for row in rows), f"{side}/{slug}: cached sample judge"
    )
    _require(all(row.get("error_code") is None for row in rows), f"{side}/{slug}: sample error")
    _require(all(row.get("scorer_error") is None for row in rows), f"{side}/{slug}: scorer error")
    _require(
        all(row.get("ingest_failure_count") == 0 for row in rows),
        f"{side}/{slug}: sample ingest failure",
    )
    _require(
        all(_finite(row.get("score")) and 0 <= float(row["score"]) <= 1 for row in rows),
        f"{side}/{slug}: bad sample score",
    )
    _require(
        all(
            isinstance(row.get("metrics"), dict)
            and _finite(row["metrics"].get(primary_metric))
            and math.isclose(
                float(row["metrics"][primary_metric]),
                float(row["score"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for row in rows
        ),
        f"{side}/{slug}: sample primary metric mismatch",
    )
    sample_mean = math.fsum(float(row["score"]) for row in rows) / question_count
    _require(
        math.isclose(float(score["mean"]), sample_mean, rel_tol=1e-12, abs_tol=1e-12),
        f"{side}/{slug}: task score does not match samples",
    )
    _require(score.get("sample_count") == question_count, f"{side}/{slug}: score count mismatch")
    _require(
        primary.get("sample_count") == question_count
        and _finite(primary.get("mean"))
        and math.isclose(float(primary["mean"]), sample_mean, rel_tol=1e-12, abs_tol=1e-12),
        f"{side}/{slug}: primary metric summary mismatch",
    )
    ordered_ids_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    _require(
        ordered_ids_hash == identity["ordered_sample_ids_sha256"],
        f"{side}/{slug}: wrong ordered sample IDs",
    )

    previous = (
        None if side == "baseline" else (FINAL / BASELINE_NAME / slug / "run-provenance.json")
    )
    validate_provenance(
        result_dir,
        side,
        result_path,
        samples_hash,
        expected_sequence={
            "id": f"quality-{slug}",
            "index": 0 if side == "baseline" else 1,
            "label": side,
            "previous_provenance": None if previous is None else str(previous.resolve()),
            "previous_provenance_sha256": (
                None if previous is None else hashlib.sha256(previous.read_bytes()).hexdigest()
            ),
            "total": 2,
        },
    )
    return {
        "dir": result_dir,
        "document": document,
        "task": task,
        "rows": rows,
        "samples_sha256": samples_hash,
        "ordered_sample_ids_sha256": ordered_ids_hash,
    }


def _pair(slug: str, baseline: dict[str, Any], current: dict[str, Any]) -> None:
    for field in PAIR_DOCUMENT_FIELDS:
        _require(
            baseline["document"].get(field) == current["document"].get(field),
            f"{slug}: document {field} differs",
        )
    for field in PAIR_TASK_FIELDS:
        _require(
            baseline["task"].get(field) == current["task"].get(field),
            f"{slug}: task {field} differs",
        )
    left = baseline["rows"]
    right = current["rows"]
    _require(
        [row["sample_id"] for row in left] == [row["sample_id"] for row in right],
        f"{slug}: sample identity/order differs",
    )
    for index, (before, now) in enumerate(zip(left, right, strict=True)):
        for field in PAIR_SAMPLE_FIELDS:
            _require(
                (field in before) == (field in now) and before.get(field) == now.get(field),
                f"{slug}: sample {index} {field} differs",
            )


def _write_manifest(path: Path, result_dirs: list[Path]) -> None:
    value = {"results": [str(item) for item in result_dirs]}
    if path.exists():
        _require(_load(path) == value, f"refusing to replace different manifest: {path}")
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate(
    slugs: tuple[str, ...] = tuple(TASKS),
    *,
    require_manifests: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    artifacts: dict[str, dict[str, dict[str, Any]]] = {"baseline": {}, "current": {}}
    errors = []
    for slug in slugs:
        for side in artifacts:
            try:
                artifacts[side][slug] = _artifact(side, slug)
            except (
                KeyError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                errors.append(str(error))
        if all(slug in artifacts[side] for side in artifacts):
            try:
                _pair(slug, artifacts["baseline"][slug], artifacts["current"][slug])
            except ValueError as error:
                errors.append(str(error))

    if require_manifests and slugs == tuple(TASKS):
        expected = {
            ROOT / "dev-baseline-current.json": [
                str((FINAL / BASELINE_NAME / slug).resolve()) for slug in TASKS
            ],
            ROOT / "dev-current.json": [
                str((FINAL / CURRENT_NAME / slug).resolve()) for slug in TASKS
            ],
        }
        for path, result_dirs in expected.items():
            try:
                _require(
                    path.is_file() and not path.is_symlink(), f"missing fixed manifest: {path}"
                )
                _require(_load(path) == {"results": result_dirs}, f"wrong fixed manifest: {path}")
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                errors.append(str(error))

    summary = {
        "checked": list(slugs),
        "errors": errors,
        "ready": not errors,
        "samples_sha256": {
            side: {slug: artifact["samples_sha256"] for slug, artifact in values.items()}
            for side, values in artifacts.items()
        },
    }
    return summary, artifacts


def write_manifests(artifacts: dict[str, dict[str, dict[str, Any]]]) -> None:
    _write_manifest(
        ROOT / "dev-baseline-current.json",
        [artifacts["baseline"][slug]["dir"] for slug in TASKS],
    )
    _write_manifest(
        ROOT / "dev-current.json", [artifacts["current"][slug]["dir"] for slug in TASKS]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=tuple(TASKS))
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    slugs = tuple(arguments.only or TASKS)
    if arguments.write and slugs != tuple(TASKS):
        parser.error("--write requires all five tasks")

    summary, artifacts = validate(slugs)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errors"]:
        return 1
    if arguments.write:
        write_manifests(artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
