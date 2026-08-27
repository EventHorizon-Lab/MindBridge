"""Standalone SQLite + Zvec ingestion and recall benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mindbridge.infrastructure.local import (
    IndexDocument,
    LocalStore,
    StoredEmbedding,
    StoredMemory,
)
from mindbridge.infrastructure.local.zvec_index import ZvecIndex

_BATCH_SIZE = 512
_MODEL_ID = "synthetic-fp32"
_SPACE_ID = "synthetic-cosine-v1"
_TASK = "retrieval.document"
_NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def run_benchmark(
    data_dir: str | Path,
    *,
    rows: int = 1_000,
    dimension: int = 128,
    queries: int = 20,
    k: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """Run one clean, deterministic local-index benchmark and return JSON-ready metrics."""
    _validate_shape(rows=rows, dimension=dimension, queries=queries, k=k)
    root = Path(data_dir).resolve()
    _require_empty_directory(root)
    rng = random.Random(seed)
    fp32 = struct.Struct(f"<{dimension}f")

    with LocalStore(root) as store, ZvecIndex(root / "index", dimension) as index:
        ingest_started = perf_counter()
        for offset in range(0, rows, _BATCH_SIZE):
            batch_rows = min(_BATCH_SIZE, rows - offset)
            memories, embeddings, documents = _synthetic_batch(
                rng,
                fp32,
                offset=offset,
                rows=batch_rows,
            )
            store.write_memories(memories, embeddings)
            _drain_batch(store, index, documents)
        ingest_seconds = perf_counter() - ingest_started

        optimize_started = perf_counter()
        index.optimize()
        index.flush()
        optimize_seconds = perf_counter() - optimize_started

        query_vectors = tuple(_normalized_fp32(rng, fp32) for _query in range(queries))
        recall, latencies = _measure_queries(index, query_vectors, k=k)

    query_seconds = sum(latencies)
    sqlite_bytes = sum(
        path.stat().st_size for path in root.glob("state.sqlite3*") if path.is_file()
    )
    zvec_bytes = _tree_bytes(root / "index")
    return {
        "rows": rows,
        "dimension": dimension,
        "queries": queries,
        "k": k,
        "seed": seed,
        "ingest_seconds": ingest_seconds,
        "optimize_seconds": optimize_seconds,
        "recall_at_k": recall,
        "query_latency_ms": {
            "p50": _percentile(latencies, 0.50) * 1_000.0,
            "p95": _percentile(latencies, 0.95) * 1_000.0,
            "p99": _percentile(latencies, 0.99) * 1_000.0,
        },
        "query_qps": queries / query_seconds,
        "disk_bytes": {
            "sqlite": sqlite_bytes,
            "zvec": zvec_bytes,
            "total": sqlite_bytes + zvec_bytes,
        },
    }


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Parse CLI arguments, run the benchmark, and print one JSON document."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--rows", type=_positive_int, default=1_000)
    parser.add_argument("--dimension", type=_positive_int, default=128)
    parser.add_argument("--queries", type=_positive_int, default=20)
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_benchmark(
        arguments.data_dir,
        rows=arguments.rows,
        dimension=arguments.dimension,
        queries=arguments.queries,
        k=arguments.k,
        seed=arguments.seed,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _synthetic_batch(
    rng: random.Random,
    fp32: struct.Struct,
    *,
    offset: int,
    rows: int,
) -> tuple[list[StoredMemory], list[StoredEmbedding], list[IndexDocument]]:
    memories = []
    embeddings = []
    documents = []
    for row in range(offset, offset + rows):
        memory = StoredMemory(
            memory_id=f"memory-{row:09d}",
            content=f"Synthetic memory row {row}",
            metadata_json=f'{{"row":{row}}}',
            created_at=_NOW,
            updated_at=_NOW,
        )
        embedding = StoredEmbedding(
            embedding_id=f"embedding-{row:09d}",
            memory_id=memory.memory_id,
            values=_normalized_fp32(rng, fp32),
            model_id=_MODEL_ID,
            space_id=_SPACE_ID,
            task=_TASK,
            created_at=_NOW,
            normalized=True,
        )
        memories.append(memory)
        embeddings.append(embedding)
        documents.append(
            IndexDocument(
                embedding=embedding,
                content=memory.content,
                metadata_json=memory.metadata_json,
            )
        )
    return memories, embeddings, documents


def _drain_batch(
    store: LocalStore,
    index: ZvecIndex,
    documents: Sequence[IndexDocument],
) -> None:
    operations = store.pending_index_operations(limit=len(documents))
    expected_ids = tuple(document.embedding.embedding_id for document in documents)
    if tuple(operation.embedding_id for operation in operations) != expected_ids or any(
        operation.action != "upsert" for operation in operations
    ):
        raise RuntimeError("local index outbox does not match the synthetic write batch")
    index.upsert(documents)
    index.flush()
    if store.acknowledge_index_operations(operations) != len(operations):
        raise RuntimeError("local index outbox acknowledgement was incomplete")


def _measure_queries(
    index: ZvecIndex,
    query_vectors: Sequence[tuple[float, ...]],
    *,
    k: int,
) -> tuple[float, tuple[float, ...]]:
    index.search(
        query_vectors[0],
        limit=k,
        space_id=_SPACE_ID,
        task=_TASK,
    )
    approximate_ids = []
    latencies = []
    for vector in query_vectors:
        started = perf_counter()
        approximate = index.search(
            vector,
            limit=k,
            space_id=_SPACE_ID,
            task=_TASK,
        )
        latencies.append(perf_counter() - started)
        approximate_ids.append({hit.id for hit in approximate})

    recalls = []
    for vector, found_ids in zip(query_vectors, approximate_ids, strict=True):
        truth = index.search(
            vector,
            limit=k,
            space_id=_SPACE_ID,
            task=_TASK,
            exact=True,
        )
        truth_ids = {hit.id for hit in truth}
        recalls.append(len(truth_ids & found_ids) / len(truth_ids))
    return sum(recalls) / len(recalls), tuple(latencies)


def _normalized_fp32(rng: random.Random, fp32: struct.Struct) -> tuple[float, ...]:
    dimension = fp32.size // 4
    while True:
        values = [rng.random() * 2.0 - 1.0 for _value in range(dimension)]
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            return fp32.unpack(fp32.pack(*(value / norm for value in values)))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _require_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"benchmark data path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"benchmark data directory is not empty: {path}")
        return
    path.mkdir(mode=0o700, parents=True)


def _validate_shape(*, rows: int, dimension: int, queries: int, k: int) -> None:
    for name, value in (("rows", rows), ("dimension", dimension), ("queries", queries), ("k", k)):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if k > rows:
        raise ValueError("k must not exceed rows")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
