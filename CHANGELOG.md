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
- Typed asynchronous Python SDK in the base package, with resumable job-progress streaming, and
  `observe_file()` for a local path: it hashes the file, takes a presigned upload signed only into
  the caller's own tenant prefix, sends the bytes to object storage directly rather than through
  the API, and observes the resulting URI.
- `mindbridge observe <path> --tenant-id <id>` for the same thing from a shell, so handing
  MindBridge a file already on disk needs neither a Python script nor three `curl` calls in the
  right order. It prints the receipt as JSON, and its `processing_job_id` is the reminder that the
  observation is stored while the memory derived from it does not exist yet.
- `mindbridge` and `mindbridge-bench` command trees with a documented exit-status contract.

### Platform

- Multi-tenant isolation through forced PostgreSQL row-level security plus API-key allowlists.
- Model plugin architecture over `importlib.metadata` entry points; models are frozen.
- Embedding spaces with a startup probe that refuses to serve a stranded tenant.
- Platform-neutral edge path with on-device anonymous identity, an encrypted local store, a
  durable outbox, and offline recall.
- OpenTelemetry across REST, model calls, PostgreSQL, S3, and queued jobs, capturing no user
  content.
- Benchmark harness driving the production API across twelve official datasets plus the Agent
  Memory Leaderboard offline replay. `mindbridge-bench eval --tasks a,b,c` runs several of them
  from one invocation against one deployment, from a shipped task catalog rather than a file you
  write, downloading each official release it needs at a pinned revision and verifying it against
  a committed digest. Task names are the only argument with no default. Each task writes into a
  directory of its own, and the sweep ends by printing a results table naming, per task, the
  numbers its manifest and score sidecar carry and which of the two each came from; `--report DIR`
  prints it again for an earlier run, which is how a benchmark scored afterwards reports without
  running twice. `--limit N` scopes any runner to its first N units for a smoke run, the media
  ingest deadlines reach every task whose runner accepts them, and naming a task obtains what it
  reads: the annotations, the media behind them, and the prepared-media manifest, staged into the
  deployment's own bucket per run — **every task in the catalog, with no exception left for the
  operator to fill in.** Ego4D and M3-Bench's web videos are the two media sets no pinned snapshot
  supplies, so they are acquired rather than downloaded: the sweep drives the `ego4d` CLI for the
  first and `yt-dlp` for the second, narrowed to the units the run selected, and `--list-tasks`
  marks those two `acquire` so the prerequisite is visible before the run rather than minutes into
  it. When a prerequisite is genuinely absent — no Ego4D signature, no `yt-dlp`, no acquirer
  installed — the operator's own instructions are still what gets printed, now alongside what
  actually failed. `--no-download` refuses the annotation fetch, the media fetch, and the
  acquisition instead of performing any of them. Scoring follows
  lmms-eval's contract: each benchmark declares who scores it, the four whose protocol is exact
  match score themselves, the seven whose answers are free text are judged by a model called from
  inside the run, and `--predict-only` reports the `999` bypass sentinel instead. A judge that
  cannot be read scores the answer `0.0` — upstream's behaviour — with the count of floored
  answers recorded beside the number and printed under the table. A judged benchmark with no
  `MINDBRIDGE_BENCH_JUDGE_ENDPOINT` is refused before it ingests anything, not after, because a
  judged run that finishes and then cannot score writes no predictions at all.
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
  `mindbridge_runtime` SELECT on `schema_migrations`. No manual step. Necessary but not
  sufficient for re-embedding into a second space — see `0026`, which supplies the rest.
- **Migration `0027`** adds `embeddings.object_part` and widens the vectors unique key with it,
  so an object embedded in pieces gets one row per piece. An evidence span longer than the
  encoder's audio window is cut into several clips of different sound; all of them share one
  `object_id`, because `recall` reads that column back as an `EvidenceId`, so the second clip
  used to conflict with the first, be read as content drift, and raise **inside the single
  transaction that commits an observation's derived records** — one long audio span cost the
  whole observation, events and claims and memories included, not just its own vector. No manual
  step and no re-encode: `embedding_id` already hashed the clip ordinal, so no ID changes.
- **Migration `0026`** adds `embeddings.embedding_id_recipe`, recording which recipe derived each
  stored `embedding_id`. `embedding_id` now hashes `space_id` in every recipe, and that is what
  makes re-embedding into a second space actually work for claims, events, entities, evidence
  spans and consolidated summaries rather than only for memory records: the table is also keyed
  by `embedding_id`, so widening the unique key alone left the second vector colliding on the
  primary key. No manual step — the first write to touch a row an older recipe named re-keys it
  in place, so the first pass after this migration pays one re-encode per object it has not yet
  seen under its new name. Memory-record vectors keep the IDs they already have.

### Two harness costs that were being paid silently

Called out because both were invisible until a real run was watched rather than a test read.

- **A `--limit 1` M3-Bench run fetched the whole subset.** `prepare_m3` was the one producer that
  asked `ensure_media` for its media set without narrowing it, so cutting one 2 GB robot video
  downloaded all 100 of them — about 200 GB — and the same call would have asked an acquirer for
  all 920 web videos. It now names the single file it is missing, which is what the other producers
  already did. The absent-media message is split-aware too: the robot half is 100 files of about
  2 GB, the web half about 20 MB each, and one figure for both overstated the web split by roughly
  a hundredfold — enough to read as "you have no disk for this" and stop.
- **Those fetches were also silent.** `prepare_m3` and the Video-MME/EgoTempo `_source` helper
  passed no `announce`, so a 712 MB acquisition and a multi-gigabyte Hub download ran to completion
  with nothing on stderr. Both now report, and both honour `--quiet`.

### Known gaps

- No complete public-benchmark baseline currently stands. See
  [benchmarking](docs/benchmarking.md#current-baseline-status).
- No per-tenant quotas or rate limiting.
- No automatic re-embedding when embedding space or dimension changes.
- Retrieval follows direct `same_as` entity aliases but deliberately does not compute transitive
  closure.
- Storage schema compatibility is not guaranteed across migrations.

---

## Conventions

Each release records changes under **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**,
and **Security** as they apply.

Two kinds of entry always get called out explicitly, because they cost operators real work:

- **Migrations.** Named by number, with any manual step spelled out.
- **Configuration changes.** New, renamed, or removed environment variables.
