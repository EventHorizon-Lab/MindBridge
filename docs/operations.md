# Operations

Running a MindBridge deployment day to day: the scheduled sweeps, what they cost, telemetry, and
the drills worth rehearsing.

## Consolidation

Four evidence-verified sweeps in one command, per tenant:

```bash
uv run --extra server mindbridge consolidate --tenant-id tenant_01
```

Each sweep fixes one `evaluated_at`, scans bounded candidate pages, and lets the generator inspect
exact source AV rather than re-reading its own summaries. Concurrent runs are idempotent.

| Sweep | What it produces | Cost |
| --- | --- | --- |
| Episode | Parent events that atomically claim child events | Moderate |
| Claim | Evidence-unioned semantic claims, plus `contradicts`/`supersedes` edges | Moderate |
| Summary | A single-parent memory tree, recursively expandable to its sources | Moderate |
| Entity resolution | `same_as`/`not_same_as` verdicts between separately-named entities | **High** |

### Episode, Claim, Summary

Episode writes atomically claim child events, so a child cannot end up under two parents. Claim
writes create evidence-unioned semantic claims or durable `contradicts`/`supersedes` edges;
supersession also versions the represented memory record. Summary writes form a single-parent
tree, inspect original AV before grouping, and stay recursively expandable to their sources.

`--page-size`, `--maximum-gap-seconds`, and `--minimum-similarity` calibrate episodes; the
`--claim-*` and `--summary-*` options calibrate the other two. Defaults are in
[the CLI reference](api/cli.md#mindbridge-consolidate).

Raise `--minimum-similarity` if grouping is too aggressive; lower `--maximum-gap-seconds` if
unrelated events an hour apart are joining one episode.

### Entity resolution

Perception names what it sees once per clip, so the same person can accumulate a fresh name in
every clip. Identical names already collapse and the edge identity signal keeps anonymous people
stable, which leaves one real problem: the same entity described two different ways.

The sweep pairs same-type entities whose mentioning events fall inside
`--entity-maximum-gap-seconds`, shortlists `--entity-candidate-limit` peers per entity by vector
affinity, reopens up to `--entity-evidence-per-side` spans of each record's original recording,
and asks the generator to judge that pair alone.

Four properties worth knowing before you tune it:

- **A verdict is written only when the judge reached one** — a confident `same_as`, or a confident
  `not_same_as` so the pair is not paid for again. Unreadable media, unparseable output, or
  confidence below `--entity-minimum-confidence` in either direction leaves the pair unjudged for
  a later sweep.
- **Verdicts are pairwise and never composed.** `same_as` between A and B and between B and C
  implies nothing about A and C.
- **The judge must name the cue its verdict rested on**, and that cue is kept in
  `entity_resolution_verdicts` beside the confidence and the instant it was reached, keyed to the
  same pair as the edge. A merge that turns out wrong can be read back rather than guessed at.
  Re-judgement supersedes the cue even when the answer holds, because the stored reason has to be
  the one the standing verdict was actually reached on.
- **`--entity-maximum-pairs` bounds a page and reports what it left behind** rather than hiding
  it.

`--entity-type` defaults to `person` only. That is where the fragmentation is; widening it carries
identical risk for much less value.

Retrieval does not traverse `same_as` yet. The edge is written for the graph and for agents
reading it.

**This is the only sweep that opens media and spends a generator call per candidate pair.** If
consolidation cost is a problem, this is why. Split the cadence:

```bash
mindbridge consolidate --tenant-id tenant_01 --skip-entity-resolution   # frequent
mindbridge consolidate --tenant-id tenant_01                            # occasional
```

When skipped, the summary reports `entities` as `null` rather than zeros — a run that never
called the generator is not a run that called it and paired nothing. Do not chart those as the
same value.

Improving a bad merge is `--entity-readjudicate`, which replaces the existing verdict rather than
adding a contradicting one.

## Lifecycle

```bash
uv run --extra server mindbridge lifecycle --tenant-id tenant_01
```

Explainable decay: strength rises with useful access and positive feedback, falls with negative
feedback and idle time. Every coefficient is a flag, not a model weight.

| Concern | Flag |
| --- | --- |
| Memories cool too fast | Lower `--age-decay-weight` |
| Nothing ever reaches hot | Lower `--strengthen-at` |
| Cold set is not shrinking storage | Set `--compress-below` |
| Feedback under-counted | Raise `--positive-feedback-weight` |

Idle days to cold are salience divided by `--age-decay-weight`, so the default of 0.005 cools a
memory of median salience after 100 unused days. Set the decay weight from your actual retention
policy: a robot ingesting continuously and a personal assistant have no reason to share it.

`--compress-below` must not exceed `--cold-below`. Compression drops rebuildable derived clips
while keeping the record and its pointers into the original recording — a storage reduction, not
a deletion.

A complete run uses stable bounded pages and one fixed evaluation instant. Concurrent feedback or
deletion wins through optimistic guards, so a sweep never overwrites a user's correction.

## Orphan clip reclamation

Derived clips are uploaded before the transaction that registers them, so an interrupted attempt
can leave an object no record references. Clip keys are content-addressed, which stops a retry
from multiplying that object; reclamation deletes what is already there.

```bash
mindbridge lifecycle --tenant-id tenant_01 --reclaim-orphan-clips --dry-run
mindbridge lifecycle --tenant-id tenant_01 --reclaim-orphan-clips
```

`--dry-run` writes nothing at all: it counts what would be deleted, deletes none of it, and skips
both the strength sweep and the `--compress-below` purge, whose counters then stay empty. Run it
first and read the count — a number far larger than expected means something else is wrong.

The flag reads the object-storage variables, so enable it only where those are configured.

## Job ledger reconciliation

PostgreSQL is the authority for job state; the broker only carries the ID of work already recorded
there. The two drift apart silently. `task_acks_late=True` acks a message the moment its task
raises, so any exception outside the worker's `autoretry_for` throws the message away while the row
stays claimable — the row is still `pending`, and no worker will ever be told about it again.

The symptom is observations that never become searchable while the queue sits empty and the workers
look idle. Nothing errors, because from each side's own point of view nothing is wrong.

```bash
mindbridge jobs --tenant-id tenant_01
mindbridge jobs --tenant-id tenant_01 --republish
```

Reporting is the default because republishing spends money: every message that lands runs a
generator. Read `claimable` against `queue_depth`. A non-zero count beside an empty `queue_depth`
is exactly this divergence; a non-zero count beside a deep queue usually just means the workers
are behind.

`queue_depth` is the sum across every queue the worker consumes — `mindbridge` and its eight
per-tenant shards — not one of them. That is what the withholding arithmetic below subtracts from,
so a depth covering one shard would withhold an eighth of the backlog and republish the rest as
duplicates. It also means the number will not match `LLEN mindbridge` on the broker.

What `--republish` then does depends on the scope, and the report's `withheld` field says which
happened. Across the whole ledger it publishes only the difference — the queue already holds a
message for the rest, and duplicating those is not free: the delivery that loses the claim
re-queues itself every 30 seconds up to 40 times waiting for the winner. Under `--tenant-id` it
publishes every claimable row and withholds nothing, because the depth counts every tenant's
messages and cancelling one tenant's stranded rows against another's backlog would repair nothing.
On a deep queue, then, judge a scoped repair by `claimable` before running it: the messages already
queued for that tenant will be duplicated.

The same report answers "who is consuming the worker", ordered by `work_seconds` — total worker
time across every attempt, including the attempt in flight. Both durations and both token counts
cover every attempt, because a failed attempt held a worker and was billed as surely as a
successful one. Field-by-field meanings are in the [CLI reference](api/cli.md#mindbridge-jobs).

Two cautions. `--include-failed` is off by default because a deterministic failure republished on a
timer pays for the same rejection every time — republish `failed` rows once you know why they
failed, not on a schedule. And a repair is published at a priority the broker drains ahead of fresh
work, on the shard it belongs to, so it does not wait behind the backlog it repairs; before that was
true, six republishes of one job across 84 minutes moved its attempt count not at all.

Under row-level security a cross-tenant scan returns no rows rather than failing, so a confined
role reading "nothing to repair" is the one answer you cannot trust. Pass `--tenant-id`, or use a
role that can see every tenant; the command refuses rather than guessing.

## Scheduling

Both commands are one-shot, idempotent, and tenant-scoped. Either drive them from the
deployment's existing control plane, or — for consolidation only — turn on the built-in Celery
beat schedule below.

### Built-in consolidation schedule

Consolidation is the sweep that has to run for recall to see anything above a single clip: with
no Episode, Claim, or Summary records, a query can only ever come back with the individual
moments it matched. A nine-benchmark evaluation produced **zero summary-tier records**, because
nothing was calling the command.

Two variables turn it on, read by the same Celery app the worker already uses:

```bash
export MINDBRIDGE_CONSOLIDATION_TENANT_IDS=tenant_01,tenant_02
export MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS=3600   # optional, this is the default
```

Then run beat, and a worker for the queue the sweep is routed to:

```bash
celery -A mindbridge.celery_app:app beat --loglevel=INFO
celery -A mindbridge.celery_app:app worker -Q mindbridge_consolidation --concurrency=1
```

Both processes need the same two variables: beat decides *when*, the worker decides *which
tenant*, and they have to agree on the list.

**One tick sweeps one tenant, by rotation.** The interval is therefore a ceiling on
consolidation's whole share of the generator, not a per-tenant cadence: two tenants at an hour
each get swept once every two hours. Raising the tenant count never raises the load; it lengthens
the rotation. The tick's tenant is derived from the clock rather than a stored counter, so beat
restarts and worker recycles need nothing to stay in step, and a missed tick skips one tenant
until the rotation comes round again.

The scheduled sweep runs with `--skip-entity-resolution`. Entity resolution is the sweep that
opens media and spends a generator call per candidate pair, and it stays a deliberate
`mindbridge consolidate` invocation on its own rarer cadence.

### What a running sweep does to ingest

| Resource | Shared with ingest? |
| --- | --- |
| Celery queue | **No.** The task is routed to `mindbridge_consolidation`; the observation queues are `mindbridge` and its shards, and a worker started with no `-Q` consumes those and not this one. A sweep never takes a worker slot or a queue position from an observation. |
| Generator endpoint | **Yes.** Episode, Claim, and Summary pages run strictly one after another, so a sweep holds **at most one concurrent generator call** for as long as it runs. |
| PostgreSQL | Yes. The consolidation worker opens its own pool — count it in `MINDBRIDGE_DATABASE_MAX_POOL_SIZE`, which is per process. |
| Object storage | Only through entity resolution, which the schedule skips. |

For scale: the 2026-08-24 evaluation measured its generator endpoint at 816 / 1010 / 1680 video
calls per hour at 4 / 12 / 24 concurrent calls. One concurrent call is the smallest share of that
a background job can take.

Two independent ways to turn it off, and neither needs the API or the observation workers
restarted:

- Stop the `-Q mindbridge_consolidation` worker. Queued ticks expire after one interval, so
  nothing accumulates and nothing runs when it comes back.
- Unset `MINDBRIDGE_CONSOLIDATION_TENANT_IDS` and restart beat. With no tenant list there is no
  schedule and no task at all.

A sweep still runs under the app-wide Celery task budget, which is sized for one observation. A
tenant with a large unconsolidated backlog — a first run, or a bulk import — will exceed it and
be cut off, and because the sweep's cursor lives for one run, the next tick restarts from the
beginning. Prime such a tenant once from the command line, which has no deadline, and let the
schedule carry the increment from then on.

### External schedulers

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mindbridge-consolidate-tenant-01
spec:
  schedule: "17 * * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: consolidate
              image: your-registry/mindbridge:0.1.0
              args: ["mindbridge", "consolidate", "--tenant-id", "tenant_01",
                     "--skip-entity-resolution"]
              envFrom: [{secretRef: {name: mindbridge-env}}]
```

`concurrencyPolicy: Forbid` is belt-and-braces — the sweeps are idempotent — but it keeps two runs
from competing for the same generator quota.

Offset the schedules of different tenants. Every tenant sweeping on the hour turns a smooth model
bill into a spike.

Both commands print one JSON object on stdout. Capture it: it is the record of what the sweep did,
and the only place the dropped-pair counts appear. The beat-scheduled sweep has no stdout to
capture, so it logs the same object as the fields of a `consolidation sweep complete` record.

## Logs

Every process writes structured records to stderr with no collector and no configuration. Each
instrumented operation logs its own duration and outcome as it completes, so the write and read
paths are attributable from `docker logs` alone.

```bash
export MINDBRIDGE_LOG_FORMAT=json     # or text for a terminal; unset picks by TTY
export MINDBRIDGE_LOG_LEVEL=INFO      # WARNING keeps failures and drops the operation stream
```

```json
{"level":"INFO","message":"operation success","operation":"mindbridge.recall",
 "duration_ms":812.4,"self_ms":31.2,"mindbridge.recall.answered":true,
 "trace_id":"4f1c...","service":"mindbridge-api"}
```

`duration_ms` is inclusive and `self_ms` excludes nested instrumented operations. Records carry
IDs and counts, never memory text, prompts, media, or credentials — the same rule the spans
follow.

Four warnings exist because the condition is otherwise invisible in a working deployment:

| Warning | What it means |
| --- | --- |
| `structured output rejected, retrying once` | The model left its output contract and a second generation was paid for. Read `constrained`: a retry that is still constrained repeats the first attempt's arguments, so the bound that failed is what to look at. |
| `generation proxy skipped, perceiving the untouched source` | Media was silently downgraded; perception still succeeded on the source. |
| `database failure classified as transient` | Carries the SQLSTATE the retry translation otherwise discards. |
| `provider request failed` | Carries the status code and whether it was treated as retryable. |

### Finding the bottleneck

```bash
MINDBRIDGE_TIMING_SUMMARY=1 mindbridge consolidate --tenant-id tenant_01
```

One row per operation at exit, ranked by self time, with `calls`, `self_seconds`,
`total_seconds`, `mean_ms`, `max_ms`, and `self_share`. Rank by self time rather than total:
by total, the outermost operation always wins because it contains everything.

`mindbridge-bench` emits it at the end of every run without the variable, including a run that
died halfway — which is when knowing what owned the clock is worth most.

## Telemetry

OpenTelemetry activates per signal. The default OTLP exporter is a no-op without an endpoint;
`console` emits locally without a collector, and `none` disables the signal. The logs above do
not depend on it.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1
```

The official FastAPI, HTTPX, psycopg, Celery, and Botocore instrumentations propagate W3C context
across REST, model calls, PostgreSQL, S3, and queued jobs. Each process has a distinct default
`service.name`; override per process with `OTEL_SERVICE_NAME` when a deployment needs a namespace.

**Nothing sensitive is captured.** No authorization headers, request bodies, prompts, memory text,
or media.

Response `trace_id` values take the form `trace_<32-hex W3C trace ID>`, so the suffix maps
directly onto the configured backend — a user reporting a bad answer with a trace ID gives you the
whole request.

### What to watch

`mindbridge.stage.duration` reports bounded `stage` values covering edge capture-to-upload and
acknowledgement, cloud job claim and searchable readiness, and recall first-answer and completion
latency.

Generator spans additionally report media count, JSON retry, whether decoding was
schema-constrained, time to first token, token usage, and the bounded recall phase and round.
None of these attributes carry user content or IDs.

| Signal | Why it matters |
| --- | --- |
| Recall first-answer latency | Perceived latency. Measure time to first token, not wall clock — they diverge substantially. |
| `mindbridge.model.schema_constrained` | False across the fleet means the endpoint refused schema decoding and every structured call is back on prompt-only output. |
| Job claim → searchable readiness | How stale memory is relative to capture. |
| Generator JSON retry rate | A rising rate means the model is drifting off its output contract. |
| `task_broker_unavailable` rate | Observations are being rejected outright. |
| Job `failed` count | Remember it settles the attempt, not the job. |
| Claimable rows against queue depth | `mindbridge jobs`. Claimable work beside an empty queue is ledger/broker divergence, not backlog. |
| Consolidation dropped pairs | From the stdout JSON, not telemetry. Persistent drops mean a page bound is too tight. |

## Suggested alerts

| Alert | Condition |
| --- | --- |
| Ingest rejected | `task_broker_unavailable` over a few minutes. |
| Memory going stale | Job claim-to-readiness above your ingest interval. |
| Model degrading | `model_output_invalid` or JSON retry rate rising. |
| Storage inconsistent | Any `memory_integrity_failed`. This should be zero. |
| Deletion stalled | Tombstones in `failed`, or in `propagating` beyond your SLA. |
| Work recorded but not queued | `claimable` non-zero while `queue_depth` is zero. The ledger owes work no worker will be told about. |

`memory_integrity_failed` deserves a page rather than a dashboard. It means stored state is
inconsistent, which is not a load condition and will not clear on its own.

## Drills worth rehearsing

**Deletion propagation.** Issue a `forget()`, confirm `propagation_state` reaches central
`complete`, then separately confirm that an offline device reconciles on reconnect. The central
state is not a device acknowledgement.

**Restore-then-reconcile.** A backup that outlives a deletion reintroduces deleted content on
restore. Tombstones are content-free and survive the content precisely so a restore can be
reconciled against them — but only if you have practised it. Decide the backup retention window
and the deletion propagation window together.

**Key rotation.** Add a second key to a tenant's list, deploy, move clients, remove the first.
The list exists for exactly this; verify it end to end before you need it under pressure.

**Embedding space migration.** The startup probe stops you serving a half-migrated deployment. It
does not perform the migration. Know the rebuild path before you change embedder or dimension.

## Related

- [CLI reference](api/cli.md) — every flag and default.
- [Deployment](deployment.md) — process topology and scaling.
- [Troubleshooting](troubleshooting.md) — when a sweep or a recall misbehaves.
