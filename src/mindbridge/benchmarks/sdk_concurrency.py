"""Closed-loop public SDK load test with deterministic, credential-free embeddings.

Each workload/concurrency/repetition gets a fresh physical data directory. Successful
latencies exclude rejected calls; throughput counts only successful operations. These
measurements cover the embedded runtime, not model-service capacity or model quality.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import platform
import random
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from time import perf_counter, process_time, sleep

from mindbridge import (
    EmbedTask,
    IndexQuantization,
    Memory,
    MemoryRecord,
    Modality,
    ModelInput,
    RetrievalMode,
    StorageError,
)
from mindbridge.benchmarks.eval_statistics import percentile


class SyntheticEmbedder:
    """Precomputed independent unit vectors, with optional simulated model latency."""

    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "sdk-load-fixture-v1"
    embedding_space = "sdk-load-fixture-v1"

    def __init__(self, *, rows: int, dimension: int, latency_ms: float) -> None:
        self.embedding_dimension = dimension
        self.latency_ms = latency_ms
        rng = random.Random(42)
        self.vectors = []
        for _ in range(rows):
            vector = tuple(rng.uniform(-1, 1) for _ in range(dimension))
            norm = math.hypot(*vector)
            self.vectors.append(tuple(value / norm for value in vector))

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if self.latency_ms:
            sleep(self.latency_ms / 1_000)
        return tuple(self.vectors[int(value.text.rsplit(" ", 1)[1])] for value in inputs)

    def close(self) -> None:
        pass


def _measure_worker(
    memory: Memory,
    records: Sequence[MemoryRecord],
    *,
    number: int,
    workload: str,
    concurrency: int,
    requests: int,
) -> tuple[list[float], Counter[str], list[float], list[str]]:
    latencies: list[float] = []
    rejected: Counter[str] = Counter()
    rejection_latencies: list[float] = []
    written: list[str] = []
    rows = len(records)
    for request in range(number, requests, concurrency):
        row = (request * 37) % rows
        kind = workload
        if workload == "mixed":
            kind = "add" if request % 10 == 0 else "get" if request % 10 < 3 else "search"
        started = perf_counter()
        try:
            if kind == "get":
                record = memory.get(records[row].id)
                assert record.content == records[row].content
            elif kind == "search":
                hits = memory.search(records[row].content, limit=10)
                assert hits and hits[0].id == records[row].id
            else:
                text = f"Synthetic memory row {rows + request}"
                added = memory.add(text)
                assert added.content == text
                written.append(added.id)
        except StorageError as error:
            if error.reason != "capacity_exceeded":
                raise
            rejected[f"{error.stage}:{error.reason}"] += 1
            rejection_latencies.append(perf_counter() - started)
        else:
            latencies.append(perf_counter() - started)
    return latencies, rejected, rejection_latencies, written


def run_case(
    *,
    workload: str,
    concurrency: int,
    requests: int,
    rows: int,
    dimension: int,
    latency_ms: float = 0,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    quantization: IndexQuantization = IndexQuantization.NONE,
    profile_dir: Path | None = None,
) -> dict[str, object]:
    """Measure one isolated case; unexpected errors and incorrect results fail the run."""
    if workload not in {"get", "search", "add", "mixed"}:
        raise ValueError("unknown workload")
    if min(concurrency, requests, rows, dimension) <= 0 or requests < concurrency:
        raise ValueError("positive shapes and at least one request per worker are required")
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be non-negative and finite")
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=False)
    embedder = SyntheticEmbedder(rows=rows + requests, dimension=dimension, latency_ms=0)
    with (
        TemporaryDirectory(prefix="mindbridge-sdk-load-") as directory,
        Memory(
            directory,
            embedder=embedder,
            index_quantization=quantization,
            retrieval_mode=retrieval_mode,
        ) as memory,
    ):
        records = tuple(
            record
            for offset in range(0, rows, 1_024)
            for record in memory.add_many(
                tuple(
                    f"Synthetic memory row {row}"
                    for row in range(offset, min(rows, offset + 1_024))
                )
            )
        )
        memory.optimize()
        for row in range(min(rows, 8)):
            assert memory.get(records[row].id).content == records[row].content
            assert memory.search(records[row].content, limit=10)[0].id == records[row].id
        embedder.latency_ms = latency_ms
        barrier = Barrier(concurrency + 1)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _start_worker,
                    barrier,
                    profile_dir,
                    number,
                    partial(
                        _measure_worker,
                        memory,
                        records,
                        number=number,
                        workload=workload,
                        concurrency=concurrency,
                        requests=requests,
                    ),
                )
                for number in range(concurrency)
            ]
            cpu_started, started = process_time(), perf_counter()
            barrier.wait(timeout=30)
            results = [future.result() for future in futures]
            seconds, cpu_seconds = perf_counter() - started, process_time() - cpu_started
        latencies = [latency for successful, _, _, _ in results for latency in successful]
        rejected: Counter[str] = Counter()
        for _, failures, _, _ in results:
            rejected.update(failures)
        rejected_latencies = [latency for _, _, failed, _ in results for latency in failed]
        written_ids = [memory_id for _, _, _, written in results for memory_id in written]
        # Verify every acknowledged write is still authoritative after contention ends.
        for memory_id in written_ids:
            assert memory.get(memory_id).id == memory_id
        stored = 0
        cursor = None
        while True:
            page = memory.list(limit=100, cursor=cursor)
            stored += len(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert stored == rows + len(written_ids)
    return {
        "workload": workload,
        "concurrency": concurrency,
        "requests": requests,
        "rows": rows,
        "dimension": dimension,
        "retrieval_mode": retrieval_mode.value,
        "quantization": quantization.value,
        "model_latency_ms": latency_ms,
        "successful": len(latencies),
        "rejected": sum(rejected.values()),
        "rejection_reasons": dict(rejected),
        "seconds": seconds,
        "cpu_seconds": cpu_seconds,
        "successful_qps": len(latencies) / seconds,
        "success_latency_ms": _latencies(latencies),
        "rejection_latency_ms": _latencies(rejected_latencies),
        "stored_rows": stored,
    }


def _start_worker(
    barrier: Barrier,
    profile_dir: Path | None,
    number: int,
    call: Callable[[], tuple[list[float], Counter[str], list[float], list[str]]],
) -> tuple[list[float], Counter[str], list[float], list[str]]:
    barrier.wait(timeout=30)
    # CPython's profiling backend permits one active cProfile collector.
    if profile_dir is None or number != 0:
        return call()
    with cProfile.Profile() as profile:
        measured = call()
    profile.dump_stats(str(profile_dir / f"worker-{number}.prof"))
    return measured


def _latencies(values: Sequence[float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        value = percentile(values, fraction)
        result[label] = None if value is None else value * 1_000
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 28, 32, 64, 128]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("get", "search", "add", "mixed"),
        default=["get", "search", "mixed"],
    )
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-latency-ms", type=float, default=0)
    parser.add_argument(
        "--retrieval-mode", choices=tuple(mode.value for mode in RetrievalMode), default="hybrid"
    )
    parser.add_argument(
        "--quantization", choices=tuple(mode.value for mode in IndexQuantization), default="none"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        environment = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "scope": "public_sdk_synthetic_embeddings",
            }
        }
        output.write(json.dumps(environment) + "\n")
        for repeat in range(args.repeats):
            for workload in args.workloads:
                for concurrency in args.concurrency:
                    profile_dir = (
                        None
                        if args.profile_dir is None
                        else args.profile_dir / f"{repeat}-{workload}-{concurrency}"
                    )
                    result = run_case(
                        workload=workload,
                        concurrency=concurrency,
                        requests=args.requests,
                        rows=args.rows,
                        dimension=args.dimension,
                        latency_ms=args.model_latency_ms,
                        retrieval_mode=RetrievalMode(args.retrieval_mode),
                        quantization=IndexQuantization(args.quantization),
                        profile_dir=profile_dir,
                    )
                    result["repeat"] = repeat
                    line = json.dumps(result, sort_keys=True)
                    output.write(line + "\n")
                    output.flush()
                    print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
