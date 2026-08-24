# Troubleshooting

Symptoms, what they actually indicate, and what to do. Most MindBridge failures are loud on
purpose — a refusal at startup is preferred over a degraded service that looks healthy.

## The server will not start

Startup validates configuration eagerly. Each refusal below is a deliberate gate.

### `MINDBRIDGE_TENANT_API_KEYS_JSON must be configured for the REST API`

There is no anonymous mode. Set the variable, mapping each tenant to a list of keys:

```bash
export MINDBRIDGE_TENANT_API_KEYS_JSON='{"tenant_01":["at-least-32-random-characters"]}'
```

### `API keys must be strings of at least 32 characters`

Keys shorter than 32 characters are rejected at startup rather than at first request.
`openssl rand -hex 24` produces 48.

### `<VARIABLE> must be configured`

A required variable is missing or blank. Blank counts as missing — an empty string is not a
value. See [configuration](configuration.md#which-process-reads-what) for what each process
requires.

### `<VARIABLE> must contain valid JSON` / `must contain a JSON object with non-empty keys`

A `*_CONFIG_JSON` value is malformed. These variables replace a whole `mindbridge.toml` section
and are rarely the easier way to write one — a section in the file needs no shell quoting at all.
If you do set one, check it parses before starting anything:

```bash
python -c "import json,os;json.loads(os.environ['MINDBRIDGE_GENERATOR_CONFIG_JSON'])"
```

A malformed `mindbridge.toml` reports its own parse position instead, and
`mindbridge config check --role <role>` surfaces either failure before a process starts.

### `Extra inputs are not permitted` on a plugin config

Plugin configs use `extra="forbid"`. An unrecognized key fails startup rather than being ignored,
because "that setting had no effect" is a much worse outcome to debug than a failed boot. Check
the key against the plugin's documented fields.

After upgrading past migration `0021` the usual culprits are `model_revision` and
`space_revision`, which no longer exist on any plugin. Delete them from your `*_CONFIG_JSON`
objects; `model_id` and `space_id` still carry the identity they were paired with.

### `embedding dimension must be one of 32, 64, 128, 256, 512, 768, 1024`

`MINDBRIDGE_EMBEDDING_DIMENSION` accepts only widths Jina v5 was trained to truncate to. Any other
value is an untrained truncation that silently degrades recall quality.

### Startup refuses, naming stranded object types

The embedding-space probe found a tenant holding vectors the configured space cannot reach. This
is the guard that stops a changed embedder from turning every recall into an empty result.

Either restore the previous `MINDBRIDGE_EMBEDDING_SPACE_ID` / `_DIMENSION`, or
complete the re-embedding. Vectors in several spaces are accepted **while** a migration is in
progress, so a partial rebuild is not itself the problem.

The probe reports each object type separately — memory records the API wrote and evidence, events,
and claims the worker wrote can be stranded independently, which usually points at the worker's
embedder configuration having drifted from the API's.

## Recall returns nothing

### Nothing has been derived yet

`observe()` returns before memory exists. Check the job:

```bash
curl -s "localhost:8000/v1/jobs/$JOB_ID?tenant_id=tenant_01" -H "Authorization: Bearer $KEY"
```

`pending` means the worker has not claimed it — is the worker running, and pointed at the same
broker? `running` means wait. `succeeded` carries `memory_ids`; read those directly instead of
searching for what was just written.

### The time filter excluded it

`occurred_before` is **exclusive** while `occurred_after` is inclusive. A memory occurring exactly
at `occurred_before` is excluded. This is the single most common off-by-one here.

Also check that `occurred_at` means what you think: it is when the content is *about*, not when it
was written.

### `memory_ids` scoped it away

`memory_ids` is a strict scope, not a ranking hint. If it is set, nothing outside that set is
searched.

### The similarity floor is too high

`MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY` defaults to 0.0 for good reason: a floor discards
candidates that a graph hop or a lexical match would have rescued. If you raised it, lower it back
and let fusion do the filtering.

### The memory was compressed or cooled

Check `state` on records you expect. A `compressed` memory has dropped its rebuildable clips.
Review your `--compress-below` and `--age-decay-weight` settings — see
[operations](operations.md#lifecycle).

### Vectors are in a different space

If the API starts but recall is empty across the board, and the startup probe passed, confirm the
worker writes into the space the API queries. The worker compares its own two slots and fails a
job on mismatch, but it cannot detect that both of its slots disagree with the API's.

## Answers are wrong or unsupported

### `answer` is null

Abstention is a result, not a failure. MindBridge returns null when nothing supports an answer.
Check `memories` — if it is empty, the problem is retrieval; if it is populated but `answer` is
null, the evidence did not support a conclusion.

Do not tune abstention thresholds by reflex. When this was last investigated, abstention tracked
evidence availability closely — far higher when no evidence had been retrieved than when it had —
and confidence was monotonically calibrated. The bottleneck was retrieval coverage, not the
threshold. Check which of the two you actually have before changing anything: count how often
`memories` comes back empty.

### Confidence is high but the answer is wrong

Pull the evidence. `RecallResult.evidence` carries a signed URL to media covering the cited
`start_ms`–`end_ms` span: normally a clip derived for that span, whose own timeline starts at
zero and which may run wider than the cited range, and otherwise the whole source object. Play
it from the start rather than seeking to `start_ms`, and it tells you immediately whether
retrieval or perception failed — which is the entire reason evidence is attached rather than
referenced.

Then record it, so the lifecycle layer learns:

```python
await memory.record_feedback(
    FeedbackRequest(
        tenant_id="tenant_01",
        feedback_type=FeedbackType.CORRECTION,
        memory_id=wrong_memory_id,
        correction_summary="What it should have said.",
    )
)
```

### Duplicate people in results

Perception names what it sees once per clip, so the same person accumulates a fresh name per clip.
Run entity resolution — see [operations](operations.md#entity-resolution). Note that verdicts are
pairwise and never composed, so A=B and B=C does not merge A and C.

## Jobs fail or stall

### `state: failed`

`failed` settles the **attempt**, not the job. The stale-job sweep can retry it. Read `error_code`
and `attempt`.

| `error_code` | Meaning |
| --- | --- |
| `model_unavailable` | The generator or embedder endpoint is down. |
| `model_output_invalid` | The model returned something unusable. Persistent means it drifted off contract. |
| `object_storage_unavailable` | Media could not be read. Check the URI, credentials, and endpoint. |
| `memory_integrity_failed` | Stored state is inconsistent. Investigate; this should be zero. |

### `request_validation_failed` naming `model_revision` on observe

An edge device older than migration `0021` is still sending `model_revision` inside
`identity_observations`. The field used to be required and is now rejected, because request
contracts refuse unknown fields rather than dropping them silently — the alternative is a device
believing it recorded provenance the server threw away.

Upgrade the device. There is no server-side setting to accept the old shape, and the field has no
replacement: `model_id` alone now records which edge model produced a span.

### `task_broker_unavailable` on observe

Redis is unreachable from the API. Nothing is half-written — the transaction and the enqueue are
ordered so a broker failure rejects the request outright.

### The same observation is retried until it gives up

The task budget expired mid-model-call. A soft-limit overrun is retried as though it were
transient, so an observation that legitimately needs longer never finishes — it repeats the same
generator call, paying for each one, until the retries run out. Nothing is written either way.

Raise `request_timeout_seconds` in `MINDBRIDGE_GENERATOR_CONFIG_JSON`; the worker's Celery limits
are derived from it and follow automatically.

### Jobs stay `pending`

Two different causes, and they are told apart by the queue rather than by the rows.

**The worker is not consuming.** Check it is running, points at the same
`MINDBRIDGE_TASK_BROKER_URL`, and is installed with the extras it actually needs:
`--extra server --extra media` at minimum. `media` carries the PyAV, Pillow, and SoundFile
decoders that cut evidence clips, which the worker does whatever its embedder slots say — without
it the process starts fine and fails the first observation that carries media. Add
`--extra cloud-models` only for an in-process encoder, which brings `media` with it.

**Or the message is gone and the row is not.** `task_acks_late=True` acks a message the moment its
task raises, so any exception outside the worker's `autoretry_for` discards the message while the
row stays claimable. The row will sit `pending` forever because nothing will tell a worker about it
again. The tell is a non-zero `claimable` beside an empty `queue_depth`:

```bash
mindbridge jobs --tenant-id tenant_01
mindbridge jobs --tenant-id tenant_01 --republish
```

Reporting is the default because each republished message runs a generator. See
[operations](operations.md#job-ledger-reconciliation).

### `TypeError: 'NoneType' object is not callable` on the first frame

Torchvision is missing. Jina Omni's image and video processor is a Qwen3-VL processor that refuses
to construct without it, and the upstream loader swallows that `ImportError` and leaves the
processor unset — so the model loads, embeds text happily, and dies on the first frame it is given.

MindBridge now detects the empty processor slot and says so instead. Install the extra, which
pins `torchvision` from the same CUDA index as torch:

```bash
uv sync --extra server --extra cloud-models
```

A `torchvision` from PyPI links against a different CUDA runtime, so installing it by hand
generally does not work.

### The worker fails every media job

The two embedder slots resolved to different embedding spaces. The worker compares them before
processing and fails rather than writing media and text vectors that cannot be compared. Align
`MINDBRIDGE_MEDIA_EMBEDDER_*` and `MINDBRIDGE_EMBEDDER_*` on one space.

## Performance

### Recalls are slow

Use `mode="search"` where you do not need a generated answer. It skips the generator entirely and
is substantially cheaper.

Measure time to first token rather than wall clock. They diverge a lot here, and wall clock will
mislead you about what a user experiences.

### Connection pool exhaustion

One recall peaks near ten PostgreSQL connections. `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` defaults to
32 and is read by **every** process that opens a pool — four processes at 32 ask for 128 against a
default `max_connections` of 100. Size it across your whole deployment, not per process.

### Ingest is too expensive

Frame rate sets the entire write cost: one clip cut, one encoder call, one stored object per
sampled window. Lower `frames_per_second` in `MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON` first.

Above roughly 1.3 fps at 30-second segments, the generation proxy also stops working — past about
forty sampled frames its encode fails on the flush that drains the encoder. Raising frame rate
therefore trades the proxy away as well. This was documented as the MP4 muxer refusing to
interleave a sparse video track with continuous audio; it is not, and so turning `proxy_audio`
off buys no frames. Lower the frame rate or segment shorter.

If your generator ignores audio, `proxy_audio: false` is still worth setting — it is a smaller
file to encode and transfer, just not a longer one.

### Consolidation is expensive

Entity resolution is why: it opens media and spends a generator call per candidate pair. Split the
cadence with `--skip-entity-resolution` on frequent runs. See
[operations](operations.md#entity-resolution).

## Deletion

### `propagation_state` is not `complete`

Only `complete` means every copy is gone. `propagating` is normal briefly. `failed` carries an
`error_code`, usually object storage being unreachable — the tombstone is marked failed and the
error re-raised rather than swallowed.

### An offline device still holds deleted content

That device has not synced. Deletion reconciles when it runs `mindbridge edge sync --tenant-id`,
which pulls tombstones before submitting. Cache rows are removed **before** the cursor advances,
so an interruption re-processes rather than skips.

### A restore reintroduced deleted content

Expected, and the reason tombstones are content-free and outlive the content. Reconcile the
restored data against the tombstone list. Rehearse this before you need it — see
[operations](operations.md#drills-worth-rehearsing).

## Development

### Tests pass but never touched PostgreSQL

Without `MINDBRIDGE_TEST_DATABASE_URL` the whole integration suite — Golden Recall included —
skips silently. A green run may have exercised nothing.

```bash
export MINDBRIDGE_TEST_DATABASE_URL=postgresql://mindbridge:mindbridge@localhost:5432/mindbridge_test
MINDBRIDGE_REQUIRE_INTEGRATION=1 uv run pytest -W error
```

With `MINDBRIDGE_REQUIRE_INTEGRATION=1`, a missing database fails the run instead of skipping it.

### The test fixture refuses to rebuild the database

Its name must end in `_test`. This guard exists to stop a mistyped DSN from dropping a real
database.

### A CLI failure prints one line and no traceback

That is the contract. Set `MINDBRIDGE_TRACEBACK=1` to get the frames back.

### A command says an extra is missing

```bash
uv sync --extra server   # or edge, media, cloud-models
```

The import happens inside the same guard as the run, so a missing extra fails the way an
incomplete environment does — including for `--help` — instead of printing frames from a
third-party package.

## Reading the logs

Every process writes structured records to stderr with no collector needed, so the first place to
look is the process output rather than a dashboard:

```bash
MINDBRIDGE_LOG_FORMAT=text MINDBRIDGE_LOG_LEVEL=DEBUG mindbridge mcp
```

Each instrumented operation logs its own `duration_ms` and `outcome` on completion, and every
record carries `trace_id` when a span is active — so the ID from a failing response greps
straight to the operations behind it. Four warnings name conditions that are otherwise invisible
in a deployment that looks healthy: a structured-output retry, a silently downgraded generation
proxy, a transient database failure with its SQLSTATE, and a provider error with its status code.
[Operations](operations.md) lists them.

To find where a slow run spends its time, set `MINDBRIDGE_TIMING_SUMMARY=1` and read the ranked
per-operation summary at exit. `mindbridge-bench` prints it for every run without the variable.

## Getting help

Include the `trace_id` from the failing response. It maps directly onto the OTLP backend and
identifies the whole request, and it appears in the logs above, so timings and span attributes
are both reachable from it. None of it contains user content.
