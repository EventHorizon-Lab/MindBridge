# Command line

Two binaries. `mindbridge` runs the deployable processes; `mindbridge-bench` runs the
reproducible benchmark harness.

They are separate because `benchmarks/` is a leaf that no product module may import. A single
command tree would have to address benchmark modules by string to route to them, which is
exactly the loophole the import guard was tightened to catch.

Either command with no arguments prints its table. Any subcommand's `--help` documents its own
flags, their defaults, the environment variables it reads, and its exit status.

```bash
uv run mindbridge
uv run --extra server mindbridge lifecycle --help
```

| Command | Subcommands |
| --- | --- |
| `mindbridge` | `config check`, `consolidate`, `jobs`, `lifecycle`, `mcp`, `jina serve`, `edge sync` |
| `mindbridge-bench` | `locomo-refined`, `m3`, `egolife`, `egomem`, `egotempo`, `memlens`, `mm-lifelong`, `atm`, `mem-gallery`, `supermemory`, `video-mme`, `video-mme-v2`, `aml` |
| `mindbridge-bench` support | `eval`, `score`, `datasets`, `jina`, `bakeoff` |

`mindbridge-consolidate`, `mindbridge-lifecycle`, and `mindbridge-mcp` remain as aliases for the
subcommands of the same name. They route through the same module and report the same codes.

The API and memory worker are long-running processes started by `uvicorn` and `celery`. The Jina
embedding service is started by `mindbridge jina serve`. See [deployment](../deployment.md).

## The shared contract

Both trees behave identically on these points.

**stdout is output, stderr is progress.** `consolidate` and `lifecycle` print one JSON object on
stdout. The benchmark runners write predictions and a manifest to `--output` and print nothing
on stdout. `mcp` speaks JSON-RPC on stdout, so never redirect it. `-q`/`--quiet` silences
progress on the commands that report it.

**Exit status is documented and stable:**

| Code | Meaning |
| --- | --- |
| `0` | The command completed. |
| `1` | The command failed; the message on stderr names what. |
| `2` | The invocation was unusable. |
| `130` | The command was interrupted. |

Every failure is one line by default, infrastructure failures included. Set
`MINDBRIDGE_TRACEBACK=1` to get the frames back.

**Credentials come from the environment, never a flag**, so a recorded invocation, a process
list, and a systemd unit never carry them. Each command's `--help` names the variables it reads.

**A missing extra says so.** The import happens inside the same guard as the run, so a command
whose extra is absent fails the way an incomplete environment does — including for `--help` —
and names the extra to install rather than printing frames from a missing third-party package.

**`-V`/`--version`** reports the installed distribution on either tree and every subcommand.

**Irreversible work has a preview.** `mindbridge lifecycle --reclaim-orphan-clips --dry-run`
writes nothing at all.

---

## `mindbridge config check`

Reports whether one role's configuration is complete, before that role is started.

```bash
mindbridge config check --role api
```

`--role` is required and takes one of `api`, `mcp`, `worker`, `consolidate`, `lifecycle`, or
`edge-sync` — the six columns of the matrix in [configuration](../configuration.md).

Every missing setting is reported in one pass. Starting a process fails on the first one, so an
operator missing nine discovered them one restart at a time; this asks the role's own settings
class what it requires and walks the whole list. Each resolved setting is reported with the
source that won, `environment` or `mindbridge.toml`.

**No value is ever printed.** A credential is reported as present or missing and nothing more,
because the same code path handles credentials and structure. If settings are missing and a
`.env` file is present in the working directory, the report says so — nothing in MindBridge loads
`.env`, and `uv run --env-file .env` is the usual omission.

Exit status is 0 when the role is ready and 1 when it is not, so a deployment script can gate on
it. This command is in the core install: it has to run before any extra is present, which is the
state an operator reaches for it from. A role whose settings class needs an extra says which one.

## `mindbridge consolidate`

Runs four evidence-verified sweeps — Episode, Claim, Summary, and entity resolution — as one
tenant-scoped scheduled job. Requires `--extra server`.

```bash
uv run --extra server mindbridge consolidate --tenant-id tenant_01
```

Each sweep fixes one `evaluated_at`, scans bounded candidate pages, and lets the generator
inspect exact source AV. Concurrent runs are idempotent, so schedule it with whatever CronJob,
systemd timer, or Celery beat the deployment already has.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--tenant-id` | required | Tenant whose memories are swept. |
| `--evaluated-at` | now | The one aware instant the whole sweep evaluates at. |
| `--page-size` | 16 | Episode candidates per bounded page. |
| `--maximum-gap-seconds` | 900 | Longest silence two events may span and still join one episode. |
| `--minimum-similarity` | 0.7 | Lowest similarity two events may have and still join one episode. |
| `--claim-page-size` | 16 | Claim candidates per page. |
| `--claim-maximum-gap-seconds` | 2592000 | 30 days. |
| `--claim-minimum-similarity` | 0.8 | |
| `--summary-page-size` | 16 | |
| `--summary-maximum-gap-seconds` | 2592000 | |
| `--summary-minimum-similarity` | 0.8 | |
| `--entity-page-size` | 16 | Entity seeds per page. |
| `--entity-maximum-gap-seconds` | 2592000 | |
| `--entity-candidate-limit` | 8 | Peers per seed, by vector affinity. |
| `--entity-minimum-confidence` | 0.75 | Below this, no verdict is recorded in either direction. |
| `--entity-evidence-per-side` | 3 | Evidence spans reopened per entity when judging a pair. |
| `--entity-maximum-pairs` | 64 | Pairs judged per page; the rest are reported as dropped, not hidden. |
| `--entity-type` | `person` | Repeatable. One of `person`, `object`, `place`, `device`, `organization`, `topic`. |
| `--entity-readjudicate` | off | Re-judge pairs that already carry a verdict, replacing it. |
| `--skip-entity-resolution` | off | Skip the entity sweep; the other three still run. |

Entity resolution is the only sweep that opens media and spends a generator call **per candidate
pair**, which makes it by far the most expensive. `--skip-entity-resolution` declines it. When
skipped, the summary reports `entities` as `null` rather than zeros — a run that never called
the generator is not a run that called it and paired nothing.

Verdicts are pairwise and never composed: `same_as` between A and B and between B and C implies
nothing about A and C. One pair owns one verdict row; `--entity-readjudicate` replaces it rather
than adding a contradiction.

`--entity-type` defaults to `person` only because that is where name fragmentation actually
happens. Widening it carries identical risk for much less value.

**Environment:** `MINDBRIDGE_DATABASE_URL` (required), the object-storage variables (source AV
the generator inspects), and the generator and embedder variables.

---

## `mindbridge jobs`

Reports the observation job ledger against the broker, and repairs the difference. Requires
`--extra server`.

```bash
uv run --extra server mindbridge jobs --tenant-id tenant_01
```

PostgreSQL is the authority for job state; the broker only carries the ID of work already
recorded there. The two drift apart silently, in both directions. `task_acks_late=True` acks a
message the moment its task raises, so any exception outside the worker's `autoretry_for` throws
the message away while the row stays claimable — 479 `torch.OutOfMemoryError` did exactly that
during one evaluation, leaving rows no worker would ever be told about again. Republishing from
the ledger repairs any such divergence rather than one exception class at a time.

**Reporting is the default, because republishing spends money:** every message that lands runs a
generator. `--republish` is the flag that acts.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--tenant-id` | all tenants | Restrict the report and the repair to one tenant. Required when row-level security confines the role's reads — see below. |
| `--include-failed` | off | Also count and republish `failed` rows. |
| `--republish` | off | Publish one message per claimable row the queue cannot already hold, oldest observation first — `--tenant-id` changes that bound, below. |

`--include-failed` is off because a deterministic failure republished on a timer pays for the
same rejection every time. A row is *claimable* when a worker would still accept a claim for it:
`pending`, stale `running` (what a lost worker leaves behind), and `failed` only when asked — and
its observation still exists, because a job whose observation was deleted would only fail again.
That last condition is why a tenant can report `pending: 5` with `claimable: 0`.

**Across the whole ledger, `--republish` publishes at most `claimable - queue_depth` rows.** The
ledger knows a row is owed work but cannot know whether a message for it survives, and no broker
answers that per message; the count does, since the queue carries this job type and nothing else.
Without the bound, a `--republish` against a healthy deep backlog duplicates every queued
message — and a duplicate is not a no-op: the delivery that loses the claim gets `RUNNING` and then
re-queues itself every 30 seconds up to 40 times waiting for the winner. Oldest first, because the
messages still queued are the most recently published, so the rows whose message is gone are the
oldest.

**With `--tenant-id`, nothing is subtracted and every claimable row is published.** That
subtraction sets one tenant's rows against a queue carrying every tenant's messages: five messages
for another tenant would cancel five stranded rows for this one and publish nothing at all. The
missing term cannot be recovered, because no transport counts a queue per tenant without reading
each message off it. The two mistakes also cost differently — a duplicate costs broker round trips
and stops on its own, while a strand costs the work until someone notices it a second time. So on a
deep queue, read `queue_depth` before repairing one tenant: it belongs to the whole queue, and the
messages it counts for this tenant will be duplicated.

**Scope is not optional under row-level security.** `jobs` is `FORCE ROW LEVEL SECURITY`, so a
cross-tenant scan by a confined role returns no rows rather than failing — and a repair tool that
reads that as "nothing to repair" is worse than one that refuses. Without `--tenant-id` the
command requires a role that can see every tenant.

### What the report means

```json
{
  "queue": "mindbridge",
  "queue_depth": 0,
  "claimable": 0,
  "withheld": 0,
  "include_failed": false,
  "republished": 0,
  "tenants": [
    {
      "tenant_id": "tenant_01",
      "jobs": 1,
      "pending": 0,
      "running": 0,
      "failed": 0,
      "succeeded": 1,
      "queue_wait_seconds": 0.005,
      "work_seconds": 0.404,
      "input_tokens": 0,
      "output_tokens": 0
    }
  ]
}
```

`queue` names the queue family the worker consumes, `task_default_queue` — **`mindbridge`, not
`celery`**. Looking at the wrong queue is how one investigation concluded the queue had been
destroyed. Observations are published across a shard set (`mindbridge` plus `mindbridge.0` …
`mindbridge.7`), all of which a worker started **without `-Q`** consumes; `queue_depth` is the
sum across the whole set, so it will not match `LLEN mindbridge` on its own.

`queue_depth` is read *before* any repair, so it describes what was found, and `withheld` below
is decided against that pre-repair number. A republished job no longer waits behind the backlog:
repair publishes at the priority the transport drains first, so it is served ahead of fresh work
on every shard. Before that split, six republishes of one job across 84 minutes moved its attempt
count not at all, because every one of them landed behind the same backlog.

`withheld` is how many claimable rows the repair treated as already carried by the queue:
`min(claimable, queue_depth)` across the ledger, and `0` under `--tenant-id`. `claimable - withheld`
is what `--republish` publishes, so a repair that published less than it found says so in the
report rather than leaving the difference to be inferred.

`tenants` is ordered by `work_seconds`, descending — the summary exists to answer *who is
consuming the worker*.

| Field | Meaning |
| --- | --- |
| `jobs`, `pending`, `running`, `failed`, `succeeded` | Row counts by state. |
| `queue_wait_seconds` | Total time this tenant's jobs spent waiting to be claimed, plus the wait a still-pending job has accrued so far. |
| `work_seconds` | Total time workers spent on them, **across every attempt**, plus the attempt in flight. |
| `input_tokens`, `output_tokens` | What the models charged, across every attempt — the generator, plus the embedder when it is a served endpoint rather than a local GPU. |

Four things worth knowing before acting on those numbers:

- **Both durations count every attempt, not the last one.** A job that failed twice and succeeded
  on the third reports all three, which is what makes them comparable to the token columns beside
  them — a failed attempt held a worker and was billed as surely as a successful one.
- **Both include the interval still open.** A pending job's wait so far, and a running job's work
  so far. Without that, the tenant holding a worker *right now* would contribute nothing to the
  column the report is sorted by.
- **An abandoned attempt stops accruing at the stale window** (2 400 s), whether it is still open
  or has been reclaimed. Past it the claim treats the row as reclaimable, so whatever held it is
  gone; charging it forever would sort every live tenant below a worker that died. The reclaim
  charges it to `work_seconds` alone — an abandoned attempt was running, not queued.
- **`work_seconds` is worker time, not wall time.** A job created an hour ago and processed in two
  seconds reports two seconds of work and an hour of wait.

The duration and token columns are `null` on a row that last moved before they existed
(migrations 0022 and 0023), where `0` means measured and that fast. The report does not preserve
the difference: it sums a tenant's rows, and a `null` contributes nothing, so a tenant whose jobs
all predate the migrations reports `0` rather than `null`. Read a suspiciously cheap tenant
against `jobs` — an old ledger, not a fast one.

**Environment:** `MINDBRIDGE_DATABASE_URL` and `MINDBRIDGE_TASK_BROKER_URL`, both required. Read
from the environment rather than flags so neither reaches a process list or a shell history.

---

## `mindbridge lifecycle`

Runs explainable memory decay and state transitions as a tenant-scoped scheduled job. A complete
run uses stable bounded pages and one fixed evaluation instant; concurrent feedback or deletion
wins through optimistic guards. Requires `--extra server`.

```bash
uv run --extra server mindbridge lifecycle --tenant-id tenant_01
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--tenant-id` | required | |
| `--evaluated-at` | now | The one aware instant the sweep evaluates at. |
| `--page-size` | 100 | Memories per bounded page. |
| `--reclaim-orphan-clips` | off | Also delete derived clip objects no committed record references. |
| `-n`, `--dry-run` | off | Write nothing. |
| `--access-weight` | 1.0 | Weight recall frequency contributes to strength. |
| `--positive-feedback-weight` | 1.0 | |
| `--negative-feedback-weight` | 1.0 | Subtracted. |
| `--age-decay-weight` | 0.005 | Strength lost per idle day. |
| `--strengthen-at` | 1.25 | Strength at or above which a memory becomes hot. |
| `--cold-below` | 0.0 | Strength below which a memory becomes cold. |
| `--compress-below` | unset | Strength at or under which a cold memory also drops rebuildable clips. Must not exceed `--cold-below`. |

Idle days to cold are salience divided by `--age-decay-weight`, so the default cools a memory of
median salience (0.5) after 100 unused days.

These are CLI flags rather than model weights on purpose: retention policy and hardware cadence
have to be calibratable without changing code or re-running a model.

`--reclaim-orphan-clips` exists because derived clips are uploaded before the transaction that
registers them, so an interrupted attempt can leave an object no record references. Clip keys
are content-addressed, which stops a retry from multiplying that object; this flag deletes what
is already there. It reads the object-storage variables, so enable it only where those are
configured.

`--dry-run` is a genuine no-op: it counts the orphan clips `--reclaim-orphan-clips` would delete,
deletes none, and skips both the strength sweep and the `--compress-below` purge, whose counters
then stay empty.

**Environment:** `MINDBRIDGE_DATABASE_URL` (required); object-storage variables, read only when
`--reclaim-orphan-clips` is given.

---

## `mindbridge mcp`

Serves the MCP server over stdio. Requires `--extra server`.

```bash
uv run --extra server mindbridge mcp
```

JSON-RPC goes on stdout. Never redirect it. See [MCP tools](mcp.md).

---

## `mindbridge jina serve`

Serves Jina v5 Omni through SentenceTransformers and an authenticated OpenAI-compatible embedding
API. Requires `server` and `cloud-models`.

```bash
export MINDBRIDGE_EMBEDDER_API_KEY=replace-with-at-least-32-random-characters
uv run --extra server --extra cloud-models mindbridge jina serve --host 0.0.0.0 \
  --media-origin https://media.example.com
```

The default port is `8002`; `--device`, `--model-id`, and `--max-concurrency` control the one model
process. Repeat `--media-origin` for every exact HTTP(S) origin that may serve presigned media;
remote URLs from any other origin and all redirects are rejected. `/health` is public.
`/v1/models` and `/v1/embeddings` require the bearer token.

---

## `mindbridge edge sync`

Drains one edge device's observation outbox once. Suitable for systemd restart and backoff
policies. Requires `--extra edge`.

```bash
export MINDBRIDGE_API_KEY=...
uv run --extra edge mindbridge edge sync \
  --database /var/lib/mindbridge/edge.db \
  --api-base-url https://memory.example.com \
  --bucket mindbridge-media \
  --region us-east-1 \
  --recent-retention-hours 24 \
  --limit 100
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--database` | required | SQLite file holding this device's outbox, deletion inbox, and recent memory. |
| `--api-base-url` | required | Base URL of the MindBridge API. |
| `--bucket` | required | Object storage bucket to upload media to. |
| `--endpoint-url` | unset | S3-compatible endpoint; omit for AWS itself. |
| `--region` | `us-east-1` | Region of that bucket. |
| `--limit` | 100 | Most observations to submit in this run. |
| `--recent-retention-hours` | 24.0 | How long this device keeps its own recall cache. |
| `--tenant-id` | unset | Tenant whose deletions to pull before submitting; repeatable. |

One shot by design. Use the robot service manager or a systemd timer for retry scheduling and
backoff — a daemon that owns its own retry loop is a second scheduler to reason about.

A failed run keeps the row, its sanitized error code, and its attempt count. Once media has
uploaded, later retries send only the idempotent observation metadata.

**Environment:** `MINDBRIDGE_API_KEY` (bearer token). AWS credentials come from Boto3's standard
chain. Neither is ever written to SQLite.

---

## `mindbridge-bench`

```bash
uv run mindbridge-bench
```

| Runner | Extra | Benchmark |
| --- | --- | --- |
| `locomo-refined` | — | LoCoMo-Refined |
| `m3` | — | M3-Bench |
| `egolife` | — | EgoLifeQA |
| `egomem` | — | EgoMemReason |
| `egotempo` | — | EgoTempo |
| `memlens` | — | MemLens |
| `mm-lifelong` | — | MM-Lifelong |
| `atm` | — | ATM-Bench |
| `mem-gallery` | — | Mem-Gallery |
| `supermemory` | — | SuperMemory VQA |
| `video-mme` | `benchmarks` | Video-MME |
| `video-mme-v2` | `benchmarks` | Video-MME-v2 |
| `aml` | — | Agent Memory Leaderboard offline replay |

| Support command | Extra | Purpose |
| --- | --- | --- |
| `eval` | `benchmarks` | Run one or more of the runners above by name, downloading what they read. |
| `score` | — | Record an official scorer's verdict beside a run. |
| `datasets` | `benchmarks` | Check every official release parses and pins its digest. |
| `jina` | `cloud-models` | Check the local Jina Omni embedder answers. |
| `bakeoff` | `cloud-models` | Compare candidate adapters on one prepared corpus. |

Runners drive the production REST API. There is no evaluation-only path, which is the point: a
benchmark that bypasses the product measures something the product does not do.

Every benchmark runner above accepts `--limit N` to run only the first N of its own units, and
`--predict-only` to write predictions without scoring them — no judge is contacted and every
declared metric reports `999`, lmms-eval's bypass sentinel. `aml` is a replay rather than a
benchmark and takes neither.

`--limit` bounds the units answered, and for the two benchmarks whose corpus is one shared
archive per tenant rather than one per unit it narrows the archive with them: ATM-Bench ingests
only the items its selected questions cite, including the distractors the release declares for
each, and MM-Lifelong only the timeline segments those questions localize into. Without the flag
both ingest everything, unchanged. A limited run's manifest records the smaller `email_count`,
`media_item_count` or `segment_count`, which is what says its numbers came from a scoped corpus
and are not a benchmark result.

Seven benchmarks score their own free-text answers by calling a judge model from inside the run,
which needs `MINDBRIDGE_BENCH_JUDGE_ENDPOINT` set; a judge that cannot be read scores the answer
`0.0`. See [benchmarking](../benchmarking.md#scoring-and-what-copying-lmms-eval-costs).

`eval` runs one or more runners in one invocation. `--tasks` names entries in a shipped catalog
and each official release is downloaded at a pinned revision if absent, so the common case needs
neither a file to write nor a corpus to populate first:

```bash
uv run --extra benchmarks mindbridge-bench eval --tasks released-text --limit 2
```

```bash
uv run --extra benchmarks mindbridge-bench eval --list-tasks
```

The task names are the only thing with no default; `--run-id` falls back to `sweep-<UTC
timestamp>`. Pass it explicitly for any task reading a prepared-media manifest you staged
yourself: the manifest's object URIs are only readable under the tenant its run ID derives, so
that ID has to be the one you staged for. Downloads are pinned to a commit and, where the smoke
manifest names a digest for them, verified against
`benchmarks/manifests/dataset-adapters-smoke.json`; prepared-media manifests are the one input no
release supplies.

`--suite FILE` is the escape hatch for a task the catalog does not name — a JSON file of
`{"tasks": [{"name": ..., "benchmark": ..., "arguments": [...]}]}`, validated exactly as strictly
as a catalog entry, with `--tasks` narrowing it. Nothing is downloaded for a suite file, because
its paths are literal and guessing which of a task's arguments are files is how a tool starts
fetching a `--split` value. The sweep gives each task a directory of its own under `--output-dir` and
derives its `--run-id`, so two parameterisations of one benchmark cannot share a tenant. It
continues past a task that fails, records every outcome in `suite-summary.json`, and prints a
results table on stdout naming, per task, the numbers its manifest and score sidecar carry and
which of the two each came from. `--report DIR` prints that table again for a directory an
earlier run wrote, which is how a benchmark scored afterwards gets its numbers on screen. See
[benchmarking](../benchmarking.md#running-several-benchmarks-in-one-command).

See [benchmarking](../benchmarking.md) for datasets, protocol, and what the numbers license you
to claim.
