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
- Embedding spaces with a startup probe that refuses to serve a stranded tenant.
- Platform-neutral edge path with on-device anonymous identity, an encrypted local store, a
  durable outbox, and offline recall.
- OpenTelemetry across REST, model calls, PostgreSQL, S3, and queued jobs, capturing no user
  content.
- Benchmark harness driving the production API across nine official datasets plus the Agent
  Memory Leaderboard offline replay.
- Python 3.10 through 3.14, with the whole quality gate — format, lint, types, tests — run on
  every one of them.

### Upgrading an existing deployment

Called out on their own because these are the changes that cost an operator real work.

- **Migration `0021`** drops `model_revision` from `events`, `claims`, `memory_records`, and
  `embeddings`, and `space_revision` from `embeddings`. No manual step: it recreates the unique key
  and both embedding-space indexes without those columns, deletes the later of any two vectors that
  differed only by revision, and strips the retired key from stored identity spans.
- **Removed environment variables:** `MINDBRIDGE_GENERATOR_MODEL_REVISION`, which was *required*,
  plus the optional `MINDBRIDGE_EMBEDDER_MODEL_REVISION` and
  `MINDBRIDGE_EMBEDDING_SPACE_REVISION`.
- **`MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION` is back**, optional, and it is the one of these
  names that was not merely a record: it pins the commit the local Jina encoder downloads and
  therefore which remote code executes under `trust_remote_code=True`. Unset, the pin is resolved
  from the model id — the bundled commit for the bundled repository, and nothing for a repository
  you named yourself, which that commit could not resolve against. Change it and change
  `MINDBRIDGE_EMBEDDING_SPACE_ID` with it; nothing else can see that the encoder moved.
- **A `*_CONFIG_JSON` object naming a retired name no longer fails startup.** `model_revision`,
  `space_revision`, and `association_model_revision` are ignored where the plugin does not declare
  them; every other unrecognized key still fails the factory. Where a plugin does declare one — the
  local Jina encoder's `model_revision` — the operator's value is kept rather than defaulted.
- **`POST /v1/observations` accepts and ignores `model_revision`** inside `identity_observations`,
  and so does the `observe` MCP tool, so a rolling upgrade that moves the server first does not
  422 the fleet behind it. Removing a field is not the same as forbidding it. Any *other* unknown
  field is still `request_validation_failed`.
- **Migration `0025`** widens the `embeddings` unique key with `space_id`, clears `observe`
  idempotency claims and identity-bearing observation digests recorded before `0021` (both were
  digested by a recipe that no longer exists, so a byte-identical resend could never match again;
  the reprocess is idempotent), makes `observations.content_digest` nullable, and grants
  `mindbridge_runtime` SELECT on `schema_migrations`. No manual step. Re-embedding into a second
  space is restored for memory records; the other object types still derive a space-blind
  `embedding_id` and collide on the primary key, which the migration comment records.

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
