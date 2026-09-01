#!/usr/bin/env python3
"""Measure whether a larger retrieval candidate pool improves Top-20 evidence coverage.

The sweep replays fixed benchmark prompts through public ``Memory.search``. Gold IDs are
joined only after search returns and are never available to query preparation or ranking.
Baseline stores are copied to temporary per-unit directories before MindBridge opens them.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

from mindbridge import ContentInput, IndexUnavailableError

ROOT = Path(__file__).resolve().parent
INCUMBENT = 50
CANDIDATES = (50, 100, 200)
TOP_K = 20
TASKS = {
    "locomo-refined": ("locomo", "locomo"),
    "mem-gallery": ("gallery-dev", "gallery-dev"),
    "atm-bench-main-sgm": ("atm-sgm-dev", "atm-sgm-dev"),
}


class _Borrowed:
    """Let each temporary Memory close its adapters without closing shared weights."""

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def close(self) -> None:
        return None


class _Hit(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...


class _SearchMemory(Protocol):
    def search(self, query: ContentInput, *, limit: int) -> Sequence[_Hit]: ...


def _candidate_pool(module: ModuleType) -> int:
    return cast(int, module.__dict__["_RERANK_CANDIDATES"])


def _set_candidate_pool(module: ModuleType, value: int) -> None:
    module.__dict__["_RERANK_CANDIDATES"] = value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_id(database: Path) -> str:
    encoded = next(part[5:] for part in database.parts if part.startswith("unit-"))
    padded = encoded.upper() + "=" * (-len(encoded) % 8)
    return base64.b32decode(padded).decode()


def _locomo_gold(dataset: Path) -> dict[str, tuple[str, ...]]:
    result = {}
    for record in json.loads(dataset.read_text(encoding="utf-8")):
        for index, question in enumerate(record["qa"]):
            result[f"{record['sample_id']}#q{index:04d}"] = tuple(
                dict.fromkeys(
                    part.strip()
                    for value in question.get("evidence", ())
                    for part in value.split(";")
                    if part.strip()
                )
            )
    return result


def _load_samples(task: str, limit: int | None) -> tuple[list[dict[str, Any]], Path]:
    artifact, data_name = TASKS[task]
    artifact_dir = ROOT / "baseline" / artifact
    sample_path = artifact_dir / "samples.jsonl"
    result = json.loads((artifact_dir / "results.json").read_text(encoding="utf-8"))
    if _sha256(sample_path) != result["samples_sha256"]:
        raise ValueError(f"{task}: baseline samples digest does not match results.json")
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    if task == "locomo-refined":
        dataset = Path(result["tasks"][0]["dataset_path"])
        if _sha256(dataset) != result["tasks"][0]["dataset_sha256"]:
            raise ValueError("locomo-refined: dataset digest does not match results.json")
        gold = _locomo_gold(dataset)
        for sample in samples:
            sample["_gold"] = gold[sample["question_id"]]
    else:
        key = "clue_ids" if task == "mem-gallery" else "evidence_ids"
        for sample in samples:
            sample["_gold"] = tuple(sample["metadata"].get(key, ()))
    if limit is not None and len(samples) > limit:
        samples = sorted(
            samples,
            key=lambda sample: hashlib.sha256(f"{task}/{sample['sample_id']}".encode()).digest(),
        )[:limit]
    return samples, ROOT / "baseline-data" / data_name


def _stores(data_root: Path) -> dict[str, Path]:
    result = {}
    for database in data_root.rglob("state.sqlite3"):
        unit_id = _unit_id(database)
        if unit_id in result:
            raise ValueError(f"duplicate store for unit {unit_id!r}")
        result[unit_id] = database.parent
    return result


def _content(prompt: Sequence[str]) -> str | Path | tuple[str | Path, ...]:
    parts: list[str | Path] = []
    for value in prompt:
        if value.startswith("/"):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"query media does not exist: {path}")
            parts.append(path)
        else:
            parts.append(value)
    return parts[0] if len(parts) == 1 else tuple(parts)


def _candidate_order(sample_id: str, candidates: Sequence[int]) -> tuple[int, ...]:
    offset = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:2], "big") % len(
        candidates
    )
    return (*candidates[offset:], *candidates[:offset])


def _search_rows(
    memory: _SearchMemory,
    sample: dict[str, Any],
    candidates: Sequence[int],
    top_k: int,
) -> list[dict[str, Any]]:
    """Run search without accepting labels or references as inputs."""
    memory_module = importlib.import_module("mindbridge.memory")
    query = _content(sample["prompt"])
    rows = []
    for position, candidate in enumerate(_candidate_order(sample["sample_id"], candidates)):
        _set_candidate_pool(memory_module, candidate)
        started = time.perf_counter_ns()
        errors = []
        for attempt in range(2):
            attempt_started = time.perf_counter_ns()
            try:
                hits = memory.search(query, limit=top_k)
            except IndexUnavailableError as error:
                errors.append(
                    {
                        "type": type(error.__cause__).__name__,
                        "elapsed_ms": (time.perf_counter_ns() - attempt_started) / 1_000_000,
                    }
                )
                if attempt:
                    raise
            else:
                successful_attempt_ms = (time.perf_counter_ns() - attempt_started) / 1_000_000
                break
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        sources = tuple(hit.metadata.get("source_id") for hit in hits)
        if any(not isinstance(source, str) or not source for source in sources):
            raise ValueError(f"{sample['sample_id']}: search hit lacks source_id")
        rows.append(
            {
                "task": sample["task"],
                "unit_id": sample["unit_id"],
                "sample_id": sample["sample_id"],
                "question_id": sample["question_id"],
                "candidate_pool": candidate,
                "effective_initial_pool": max(candidate, top_k * 3),
                "measurement_position": position,
                "latency_ms": latency_ms,
                "successful_attempt_ms": successful_attempt_ms,
                "retry_errors": errors,
                "returned": len(hits),
                "sources": sources,
            }
        )
    return rows


def _score(rows: Iterable[dict[str, Any]], sample: dict[str, Any]) -> None:
    """Join labels after retrieval; this function cannot affect query or rank order."""
    gold = frozenset(cast(Sequence[str], sample["_gold"]))
    baseline = tuple(item["source_id"] for item in sample["evidence"])
    for row in rows:
        found = len(gold.intersection(row["sources"]))
        row["annotated"] = bool(gold)
        row["gold_count"] = len(gold)
        row["matched_gold"] = found
        row["hit"] = int(found > 0) if gold else None
        row["recall"] = found / len(gold) if gold else None
        row["full_coverage"] = int(found == len(gold)) if gold else None
        row["incumbent_replay_exact"] = (
            tuple(row["sources"]) == baseline if row["candidate_pool"] == INCUMBENT else None
        )
        row["incumbent_replay_set_exact"] = (
            set(row["sources"]) == set(baseline) if row["candidate_pool"] == INCUMBENT else None
        )
        row["incumbent_replay_overlap"] = (
            len(set(row["sources"]).intersection(baseline)) / len(baseline)
            if row["candidate_pool"] == INCUMBENT and baseline
            else None
        )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"], row["candidate_pool"]].append(row)
    tasks: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        tasks[task] = {}
        for candidate in CANDIDATES:
            batch = grouped[task, candidate]
            annotated = [row for row in batch if row["annotated"]]
            latencies = [row["latency_ms"] for row in batch]
            replay = [row["incumbent_replay_exact"] for row in batch if candidate == INCUMBENT]
            replay_set = [
                row["incumbent_replay_set_exact"] for row in batch if candidate == INCUMBENT
            ]
            replay_overlap = [
                row["incumbent_replay_overlap"] for row in batch if candidate == INCUMBENT
            ]
            tasks[task][str(candidate)] = {
                "samples": len(batch),
                "annotated_samples": len(annotated),
                "hit_rate": sum(row["hit"] for row in annotated) / len(annotated),
                "recall": sum(row["recall"] for row in annotated) / len(annotated),
                "full_coverage_rate": sum(row["full_coverage"] for row in annotated)
                / len(annotated),
                "returned_mean": sum(row["returned"] for row in batch) / len(batch),
                "retry_count": sum(len(row["retry_errors"]) for row in batch),
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies),
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "p99": _percentile(latencies, 0.99),
                    "max": max(latencies),
                },
                "incumbent_replay_exact_rate": (
                    sum(bool(value) for value in replay) / len(replay) if replay else None
                ),
                "incumbent_replay_set_exact_rate": (
                    sum(bool(value) for value in replay_set) / len(replay_set)
                    if replay_set
                    else None
                ),
                "incumbent_replay_overlap_mean": (
                    sum(replay_overlap) / len(replay_overlap) if replay_overlap else None
                ),
            }
    return tasks


def _verdict(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    comparisons = {}
    for candidate in CANDIDATES[1:]:
        improved = []
        material = []
        nondecreasing = True
        deltas = {}
        for task, values in tasks.items():
            baseline, current = values[str(INCUMBENT)], values[str(candidate)]
            hit_delta = current["hit_rate"] - baseline["hit_rate"]
            recall_delta = current["recall"] - baseline["recall"]
            deltas[task] = {"hit_rate": hit_delta, "recall": recall_delta}
            nondecreasing &= hit_delta >= -1e-12 and recall_delta >= -1e-12
            if hit_delta > 1e-12 or recall_delta > 1e-12:
                improved.append(task)
            if hit_delta >= 0.01 or recall_delta >= 0.01:
                material.append(task)
        qualifies = nondecreasing and len(material) >= 2
        comparisons[str(candidate)] = {
            "deltas": deltas,
            "nondecreasing_on_every_task": nondecreasing,
            "improved_tasks": improved,
            "materially_improved_tasks": material,
            "qualifies_for_answer_quality_validation": qualifies,
        }
        if qualifies:
            eligible.append(candidate)
    return {
        "decision": (
            "validate_answer_quality" if eligible else "stop_no_cross_task_retrieval_gain"
        ),
        "eligible_candidates": eligible,
        "recommended_candidate": min(eligible) if eligible else None,
        "comparisons": comparisons,
        "promotion_rule": (
            "Hit and Recall must not decrease on any task, and at least two tasks must improve "
            "Hit or Recall by at least 1 percentage point. Select the smallest qualifying pool "
            "for held-out answer-quality validation; no retrieval-only result changes a default."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval candidate-pool ablation",
        "",
        "Public `Memory.search(..., limit=20)` was measured after one warm-up per isolated "
        "unit. Wall time includes query preparation, Jina query embedding, index search, "
        "hydration, and reranking; it excludes store cloning/opening and model loading.",
        "",
        "| Task | Pool constant / effective | Hit (delta) | Recall (delta) | p50 / p95 ms "
        "(delta) | N50 replay order / set / overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, values in report["tasks"].items():
        baseline = values[str(INCUMBENT)]
        for candidate in CANDIDATES:
            row = values[str(candidate)]
            latency = row["latency_ms"]
            hit_delta = 100 * (row["hit_rate"] - baseline["hit_rate"])
            recall_delta = 100 * (row["recall"] - baseline["recall"])
            p50_delta = latency["p50"] - baseline["latency_ms"]["p50"]
            p95_delta = latency["p95"] - baseline["latency_ms"]["p95"]
            replay = (
                "—"
                if candidate != INCUMBENT
                else f"{row['incumbent_replay_exact_rate']:.4f} / "
                f"{row['incumbent_replay_set_exact_rate']:.4f} / "
                f"{row['incumbent_replay_overlap_mean']:.4f}"
            )
            lines.append(
                f"| {task} | {candidate} / {max(candidate, TOP_K * 3)} | "
                f"{row['hit_rate']:.4f} ({hit_delta:+.2f}pp) | "
                f"{row['recall']:.4f} ({recall_delta:+.2f}pp) | "
                f"{latency['p50']:.2f} / {latency['p95']:.2f} "
                f"({p50_delta:+.2f} / {p95_delta:+.2f}) | {replay} |"
            )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            f"- Stop decision: `{report['verdict']['decision']}`.",
            f"- Smallest qualifying candidate: "
            f"`{report['verdict']['recommended_candidate']}` for held-out answer-quality "
            "validation, not a product-default change.",
            "",
            report["verdict"]["promotion_rule"],
            "",
            "Gold IDs were used only by the post-search scorer. This retrieval diagnostic does "
            "not establish answer quality and cannot by itself change a product default.",
            "",
        )
    )
    return "\n".join(lines)


def _self_check() -> None:
    memory_module = importlib.import_module("mindbridge.memory")

    class FakeMemory:
        def __init__(self) -> None:
            self.queries: list[tuple[object, int]] = []

        def search(self, query: ContentInput, *, limit: int) -> Sequence[_Hit]:
            self.queries.append((query, limit))
            sources = {
                50: ("a", "b"),
                100: ("a", "gold"),
                200: ("a", "gold", "other"),
            }[_candidate_pool(memory_module)]
            return tuple(
                cast(_Hit, SimpleNamespace(metadata={"source_id": value})) for value in sources
            )

    sample = {
        "task": "locomo-refined",
        "unit_id": "unit",
        "sample_id": "sample",
        "question_id": "question",
        "prompt": ["query only"],
        "evidence": [{"source_id": "a"}, {"source_id": "b"}],
        "_gold": ("gold",),
    }
    original = _candidate_pool(memory_module)
    try:
        fake = FakeMemory()
        rows = _search_rows(fake, sample, CANDIDATES, TOP_K)
        _score(rows, sample)
        by_candidate = {row["candidate_pool"]: row for row in rows}
        assert all(query == "query only" and limit == TOP_K for query, limit in fake.queries)
        assert by_candidate[50]["hit"] == 0 and by_candidate[50]["incumbent_replay_exact"]
        assert by_candidate[100]["hit"] == 1 and by_candidate[100]["recall"] == 1.0
        assert sorted(_candidate_order("sample", CANDIDATES)) == list(CANDIDATES)
    finally:
        _set_candidate_pool(memory_module, original)


def _resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[row["task"], row["sample_id"]].append(row)
    rows = [
        row
        for group in grouped.values()
        if len(group) == len(CANDIDATES)
        and {row["candidate_pool"] for row in group} == set(CANDIDATES)
        for row in group
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901 - one bounded sweep lifecycle
    from mindbridge import JinaOmniEmbedder, Memory
    from mindbridge.models.base import EmbeddingBackend

    memory_module = importlib.import_module("mindbridge.memory")
    original = _candidate_pool(memory_module)
    if original != INCUMBENT:
        raise RuntimeError(f"expected incumbent candidate pool {INCUMBENT}, found {original}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not (args.overwrite or args.resume):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    rows = _resume_rows(rows_path) if args.resume else []
    completed = {
        (task, sample_id)
        for task, sample_id in {(row["task"], row["sample_id"]) for row in rows}
        if {
            row["candidate_pool"]
            for row in rows
            if row["task"] == task and row["sample_id"] == sample_id
        }
        == set(CANDIDATES)
    }
    load_started = time.perf_counter()
    embedder = JinaOmniEmbedder.load(device=args.device, batch_size=args.batch_size)
    model_load_seconds = time.perf_counter() - load_started
    borrowed = cast(EmbeddingBackend, _Borrowed(embedder))
    clone_seconds = 0.0
    warmup_ms = []
    try:
        with rows_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
            for task in TASKS:
                samples, data_root = _load_samples(task, args.sample_limit)
                stores = _stores(data_root)
                selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for sample in samples:
                    selected[sample["unit_id"]].append(sample)
                for unit_id, unit_samples in selected.items():
                    unit_samples = [
                        sample
                        for sample in unit_samples
                        if (task, sample["sample_id"]) not in completed
                    ]
                    if not unit_samples:
                        continue
                    source = stores.get(unit_id)
                    if source is None:
                        raise FileNotFoundError(f"{task}: no baseline store for {unit_id}")
                    with tempfile.TemporaryDirectory(
                        prefix="mindbridge-candidate-pool-", dir=args.scratch_root
                    ) as temporary:
                        clone_started = time.perf_counter()
                        cloned = shutil.copytree(source, Path(temporary) / "store")
                        clone_seconds += time.perf_counter() - clone_started
                        with Memory(cloned, embedder=borrowed) as memory:
                            _set_candidate_pool(memory_module, INCUMBENT)
                            warm_started = time.perf_counter_ns()
                            memory.search(_content(unit_samples[0]["prompt"]), limit=args.top_k)
                            warmup_ms.append((time.perf_counter_ns() - warm_started) / 1_000_000)
                            for index, sample in enumerate(unit_samples, 1):
                                measured = _search_rows(
                                    cast(_SearchMemory, memory), sample, CANDIDATES, args.top_k
                                )
                                _score(measured, sample)
                                for row in measured:
                                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                                stream.flush()
                                rows.extend(measured)
                                if index % 10 == 0 or index == len(unit_samples):
                                    print(
                                        f"{task}/{unit_id}: {index}/{len(unit_samples)}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
    finally:
        _set_candidate_pool(memory_module, original)
        embedder.close()
    tasks = _summary(rows)
    report = {
        "schema_version": 1,
        "candidate_pool_constants": CANDIDATES,
        "top_k": args.top_k,
        "selection": (
            "all baseline samples"
            if args.sample_limit is None
            else f"lowest SHA-256(task/sample_id), at most {args.sample_limit} per task"
        ),
        "gold_boundary": "post-search scoring only",
        "memory_source_sha256": _sha256(Path(cast(str, memory_module.__file__))),
        "model": {
            "embedding_model": embedder.embedding_model,
            "embedding_space": embedder.embedding_space,
            "device": args.device,
            "batch_size": args.batch_size,
            "load_seconds": model_load_seconds,
        },
        "setup": {
            "store_clone_seconds": clone_seconds,
            "unit_warmup_ms": warmup_ms,
        },
        "tasks": tasks,
        "verdict": _verdict(tasks),
    }
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "candidate-pool-ablation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        print("candidate-pool self-check: ok")
        return 0
    if args.overwrite and args.resume:
        parser.error("overwrite and resume are mutually exclusive")
    if args.sample_limit < 0 or args.top_k != TOP_K or args.batch_size <= 0:
        parser.error("sample-limit must be non-negative, top-k must be 20, and batch-size positive")
    args.sample_limit = args.sample_limit or None
    report = _run(args)
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
