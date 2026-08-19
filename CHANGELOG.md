# Changelog

All notable changes to MindBridge are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 onward.

Until then: the REST contract, the Python SDK, and the MCP tool surface are stable enough to
build against, but the storage schema still changes through numbered migrations and carries no
compatibility promise across them.

## Unreleased

Nothing has been released yet. `0.1.0` in `pyproject.toml` is the development version.

The capabilities present on `master` today:

### Memory

- Evidence-grounded write path: observations derive Event, Entity, and Claim records from original
  audio and video, with millisecond-accurate `EvidenceSpan` pointers back into the recording.
- Six memory types (`episodic`, `semantic`, `procedural`, `prospective`, `working`, `perceptual`)
  with explainable lifecycle: strength, salience, decay, and hot/cold/compressed transitions.
- Consolidation sweeps for Episode, Claim, Summary, and cross-clip entity resolution.
- Feedback that versions rather than overwrites: `useful`, `wrong`, `missing`, and `correction`.
- Transitive, durable `forget()` with content-free tombstones that offline devices reconcile
  against.

### Retrieval

- Two-stage reciprocal rank fusion across evidence, memory, graph, and hierarchy rankings, then
  against PostgreSQL full-text search.
- Three recall modes: `answer`, `search`, and `enumerate`, the last failing explicitly rather than
  truncating an oversized scope.
- Multimodal queries — text, stored media, or both.
- Grounded follow-up through a strict `memory_ids` scope.

### Interfaces

- REST API with a generated OpenAPI document and a closed error-code contract.
- Official MCP server over stdio with seven tools, sharing the REST schemas.
- Typed asynchronous Python SDK in the base package, with resumable job-progress streaming.
- `mindbridge` and `mindbridge-bench` command trees with a documented exit-status contract.

### Platform

- Multi-tenant isolation through forced PostgreSQL row-level security plus API-key allowlists.
- Model plugin architecture over `importlib.metadata` entry points; models are frozen.
- Versioned embedding spaces with a startup probe that refuses to serve a stranded tenant.
- Platform-neutral edge path with on-device anonymous identity, an encrypted local store, a
  durable outbox, and offline recall.
- OpenTelemetry across REST, model calls, PostgreSQL, S3, and queued jobs, capturing no user
  content.
- Benchmark harness driving the production API across nine official datasets plus the Agent
  Memory Leaderboard offline replay.

### Known gaps

- No complete public-benchmark baseline currently stands. See
  [benchmarking](docs/benchmarking.md#current-baseline-status).
- No per-tenant quotas or rate limiting.
- No automatic re-embedding when embedding space or dimension changes.
- Retrieval does not traverse `same_as` entity edges yet.
- Storage schema compatibility is not guaranteed across migrations.

---

## Conventions

Each release records changes under **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**,
and **Security** as they apply.

Two kinds of entry always get called out explicitly, because they cost operators real work:

- **Migrations.** Named by number, with any manual step spelled out.
- **Configuration changes.** New, renamed, or removed environment variables.
