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
generator. Read `claimable` first. A non-zero count beside an empty `queue_depth` is exactly this
divergence; a non-zero count beside a deep queue usually just means the workers are behind.

The same report answers "who is consuming the worker", ordered by `work_seconds` — total worker
time across every attempt, including the attempt in flight. Both durations and both token counts
cover every attempt, because a failed attempt held a worker and was billed as surely as a
successful one. Field-by-field meanings are in the [CLI reference](api/cli.md#mindbridge-jobs).

Two cautions. `--include-failed` is off by default because a deterministic failure republished on a
timer pays for the same rejection every time — republish `failed` rows once you know why they
failed, not on a schedule. And kombu publishes with `LPUSH` while the worker consumes with `RPOP`,
so a republished job waits behind everything already queued: on a deep queue, expect the repair to
take effect slowly rather than assuming it did not work.

Under row-level security a cross-tenant scan returns no rows rather than failing, so a confined
role reading "nothing to repair" is the one answer you cannot trust. Pass `--tenant-id`, or use a
role that can see every tenant; the command refuses rather than guessing.

## Scheduling

Both commands are one-shot, idempotent, and tenant-scoped. Use the deployment's existing control
plane.

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
and the only place the dropped-pair counts appear.

## Telemetry

OpenTelemetry activates only when a standard OTLP endpoint is configured; otherwise it is a no-op.

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

Generator spans additionally report media count, JSON retry, time to first token, token usage, and
the bounded recall phase and round. None of these attributes carry user content or IDs.

| Signal | Why it matters |
| --- | --- |
| Recall first-answer latency | Perceived latency. Measure time to first token, not wall clock — they diverge substantially. |
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

**Deletion propagation.** Issue a `forget()`, then confirm `propagation_state` reaches `complete`
and that an offline device reconciles on reconnect. Only `complete` means every copy is gone.

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
