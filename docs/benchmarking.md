# Benchmarking

Benchmark correctness starts with storage isolation. MindBridge has no logical benchmark scope, so
every concurrently executing unit must own a different physical data directory.

## Isolation contract

Use one hierarchy under a disposable root:

```text
.benchmarks/
└── benchmark-label/
    └── run-label/
        ├── case-a/    # its own SQLite, Zvec, and lock
        └── case-b/    # its own SQLite, Zvec, and lock
```

The labels belong to the harness and filesystem only. They must not be passed into `Memory.add`,
stored as hidden product fields, or used to filter a shared database.

`BenchmarkRun` creates collision-safe path components and atomically allocates unit directories:

```python
from mindbridge import Memory
from mindbridge.benchmarks.isolation import BenchmarkRun

run = BenchmarkRun(".benchmarks", "retrieval", "trial-001")

for case_id in ("case-a", "case-b"):
    data_dir = run.unit_dir(case_id)
    with Memory(data_dir) as memory:
        memory.add(f"Evidence for {case_id}")
```

Actual path components are encoded rather than using raw labels, preventing traversal and naming
collisions. Without `resume=True`, a non-empty run directory fails immediately. Unit creation also
fails on reuse, so two workers cannot silently select the same store.

For parallel execution, allocate every `unit_dir` before or within its worker and keep one
`Memory` owner alive per leaf. Distinct leaves can run at the same time.

## Clean-run rules

A publishable result should record:

- MindBridge and Zvec versions.
- Python version and platform.
- CPU, memory, and storage medium.
- Embedding and generation model identity.
- Dataset revision and case selection.
- Random seed and retrieval limit.
- Whether the index was cold, warm, or optimized.
- The isolated data root layout.

Do not reuse a populated directory for a clean-run score. Resume mode is for deliberate recovery
or continuation and must be reported as such.

## Local-index microbenchmark

The built-in synthetic benchmark measures the SQLite-to-Zvec adapter path without model-network
variance:

```bash
python -m mindbridge.benchmarks.local_index_benchmark \
  --data-dir .benchmarks/local-index/trial-001 \
  --rows 1000 \
  --dimension 128 \
  --queries 20 \
  --k 10 \
  --seed 42
```

`--data-dir` must be empty. The command emits one JSON object with:

- Row, dimension, query, `k`, and seed parameters.
- Ingest and optimize seconds.
- Recall at `k` against Zvec exact search.
- Query latency p50, p95, and p99 in milliseconds.
- Query throughput in QPS.
- SQLite, Zvec, and total disk bytes.

Synthetic vectors isolate storage and index behavior. They do not measure embedding quality,
grounded-answer quality, or remote model latency.

## End-to-end retrieval benchmarks

Behavior benchmarks should exercise only the public SDK:

1. Allocate a new directory for the case.
2. Construct `Memory` with the model configuration under test.
3. Add the case corpus through `add` or `add_many`.
4. Call `optimize()` if the benchmark protocol specifies an optimized index.
5. Query through `search` or `ask`.
6. Score returned public values.
7. Close the instance before archiving artifacts.

Do not import `LocalStore` or `ZvecIndex` in an end-to-end benchmark. The synthetic local-index
microbenchmark is the deliberate exception because those adapters are exactly what it measures.

## Comparing performance

Report quality and speed together. A faster approximate search is not an improvement if recall
falls outside the declared tolerance. Keep ingestion, optimization, and query phases separate;
combining them hides write amplification and one-time build costs.

Latency percentiles require enough queries to be meaningful. Warm-up queries should not be mixed
with measured queries, and concurrent throughput should state concurrency and directory count.

## Artifact safety

Benchmark directories contain source text and embeddings. Treat them as dataset artifacts, apply
appropriate access controls, and delete them according to dataset terms. Never point a benchmark
at an application's live `data_dir`.
