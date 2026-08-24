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
| `mindbridge` | `config check`, `consolidate`, `lifecycle`, `mcp`, `jina serve`, `edge sync` |
| `mindbridge-bench` | `locomo-refined`, `m3`, `egolife`, `egomem`, `egotempo`, `memlens`, `mm-lifelong`, `supermemory`, `video-mme`, `aml` |
| `mindbridge-bench` support | `score`, `datasets`, `jina`, `bakeoff` |

`mindbridge-consolidate`, `mindbridge-lifecycle`, and `mindbridge-mcp` remain as aliases for the
subcommands of the same name. They route through the same module and report the same codes.

The API and the memory worker are long-running processes started by `uvicorn` and `celery`, not
subcommands. See [deployment](../deployment.md).

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

Loads the pinned Jina v5 Omni model with SentenceTransformers and serves all embedding
modalities. Requires `--extra server --extra cloud-models`.

```bash
MINDBRIDGE_EMBEDDER_API_KEY=... \
uv run --extra server --extra cloud-models mindbridge jina serve --device cuda
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Listen address. |
| `--port` | `8002` | Listen port. |
| `--device` | `cuda` | Torch device passed to SentenceTransformers. |
| `--model-id` | pinned Jina Small | Hugging Face repository ID. |
| `--model-revision` | pinned commit | Model and remote-code revision. |
| `--max-concurrency` | `1` | Concurrent model calls. |

**Environment:** `MINDBRIDGE_EMBEDDER_API_KEY` (required bearer token for `/v1/*`). `/health`
remains public for liveness.

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
| `supermemory` | — | SuperMemory VQA |
| `video-mme` | `benchmarks` | Video-MME |
| `aml` | — | Agent Memory Leaderboard offline replay |

| Support command | Extra | Purpose |
| --- | --- | --- |
| `score` | — | Record an official scorer's verdict beside a run. |
| `datasets` | `benchmarks` | Check every official release parses and pins its digest. |
| `jina` | `cloud-models` | Check the local Jina Omni embedder answers. |
| `bakeoff` | — | Compare candidate adapters on one prepared corpus. |

Runners drive the production REST API. There is no evaluation-only path, which is the point: a
benchmark that bypasses the product measures something the product does not do.

See [benchmarking](../benchmarking.md) for datasets, protocol, and what the numbers license you
to claim.
