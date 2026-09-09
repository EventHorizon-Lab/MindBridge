# Command-line API

## Surface

`mindbridge` provides 35 SDK operation commands plus `doctor`; each invocation runs one command in
one process and emits one JSON result. Local commands dispatch to the corresponding `Memory`
method, except for the diagnostic `doctor`; remote commands forward to a running owner's `/v1`
route. The CLI owns argument decoding, composition selection, JSON projection, and exit-code
mapping, not storage or retrieval policy.

Benchmark commands belong to the separate `mindbridge-bench` surface documented in
[benchmarking](../benchmarking.md).

## Invoke the CLI

```text
mindbridge [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

The console entry point calls
`mindbridge.cli.main(argv: Sequence[str] | None = None) -> int`.

Choose exactly one composition on every invocation:

```bash
mindbridge --app my_application:build_memory search "where is the spare key"
mindbridge --embedder jina-omni add "The spare key is in the blue toolbox."
mindbridge --url http://127.0.0.1:8000 list
```

Global options precede the command.
Use `mindbridge COMMAND --help` for the command-specific flags summarized below.

| Option | Meaning |
| --- | --- |
| `--app MODULE:ATTR` | `Memory` instance or zero-argument callable returning one |
| `--embedder NAME` | construct a bundled embedding recipe |
| `--url URL` | address a running owner over `/v1` |
| `--data-dir PATH` | local directory for `--embedder`; default `.mindbridge` |
| `--timeout SECONDS` | positive finite remote timeout for `--url`; default `30` |
| `--answerer NAME` | generation recipe for `--embedder` |
| `--former NAME` | formation recipe for `--embedder`; omit it to disable formation |
| `--consolidator NAME` | consolidation recipe for `--embedder`; omit it to disable `consolidate` |
| `--transcriber NAME` | speech recipe for `--embedder` |
| `--vision NAME` | visual description recipe for `--embedder`; omit it to disable captioning |
| `--index-speech`, `--no-index-speech` | enable or disable transcript indexing; default enabled |
| `--minimum-relevance FLOAT` | relevance floor; default `0.10` |
| `--ambiguity-margin FLOAT` | top-two gate when `limit=1`; default `0.01` |
| `--decay-half-life-days FLOAT` | optional positive recency half-life; default none |
| `--explain` | print resolved composition and execute no operation |
| `-q`, `--quiet` | suppress the composition banner on stderr |
| `-V`, `--version` | print the installed version |

`--data-dir`, backend options, and tuning options apply only to `--embedder`; `--timeout` applies
only to `--url`. Inapplicable options fail instead of being ignored. `--app` follows the
`MODULE:ATTR` convention and adds the working directory to `sys.path`.

The listed flags are the complete direct-composition surface. Use `--app MODULE:ATTR` for index
quantization, face analysis, speaker/face thresholds, a custom backend, or any other `Memory`
setting. In particular the `faces` command needs a face analyzer, which is built from local model
files rather than a named recipe, so it has no flag: compose it in an application and select that
application with `--app`.

Composition, model identity, credentials, and settings are documented in
[configuration](../configuration.md).

## Lifecycle and ownership

Local `--app` and `--embedder` commands open one `Memory`, perform one operation, and close it before
exit. One physical `data_dir` may have only one live owner; when another process owns it, use
`--url` or a different directory. Remote mode does not open local storage and never closes the
server's owner.

## Contract

### Bundled recipes

| Recipe | Allowed slot | Default identity |
| --- | --- | --- |
| `jina-omni` | `--embedder` | pinned Jina Omni model, revision, and 1024 dimensions |
| `funasr` | `--transcriber` | pinned FunASR model and component revisions |
| `openai` | `--embedder`, `--answerer`, `--former`, `--consolidator`, `--transcriber`, `--vision` | `text-embedding-3-small`, `gpt-5-mini`, `gpt-5-mini`, `gpt-5-mini`, `whisper-1`, `gpt-5-mini` |

Only `openai` accepts a model suffix, for example `--former openai:gpt-5-mini`. Selecting a former
opts into one formation model call after each source observation commits; omitting `--former`
keeps formation off. `--consolidator` is what the `consolidate` command needs, and it costs
nothing until that command runs: unlike a former it is not on the write path. `--vision` selects
the describer that captions visual memories so the lexical half of retrieval can reach an image
with no words of its own; it declares the `image` modality, costs one chat completion per write
batch that carries an image, and omitting it keeps the write path as cheap as it is without one.
Recipe names form a closed table; use `--app` for other backends. Provider trust, license, model
identity, and credential behavior live in [configuration](../configuration.md).

```bash
mindbridge --embedder openai --former openai --vision openai add @panel.png
```

### Commands

| Command | Operands and options | JSON result | `--url` |
| --- | --- | --- | --- |
| `add` | content; `--occurred-at`; `--occurred-end`; `--metadata`; `--memory-type`; `--context` | memory object | yes |
| `add-many` | optional JSONL source; `--memory-type` | `{"memories":[...]}` | yes |
| `add-stream` | optional JSONL source; `--memory-type`; `--capture` | `{"memories":[...]}` | no |
| `capture` | same operands and options as `add` | memory object | yes |
| `settle` | optional `MEMORY_ID...`; `--limit`; `--max-attempts` | `{"settled":int}` | yes |
| `pending-captures` | optional `MEMORY_ID...`; `--limit` | `{"pending":[...]}` | yes |
| `search` | content; `--limit`; `--memory-type`; `--reference-at`; `--scope`; `--occurred-from`; `--occurred-until` | `{"hits":[...]}` | yes |
| `search-with-trace` | search options | `{"hits":[...],"trace":{...}}` | yes |
| `ask` | content; `--limit`; `--memory-type`; `--reference-at`; `--scope`; `--link-identities`/`--no-link-identities` | answer object | yes (`--link-identities` is local-only) |
| `compile` | content; `--max-chars`; `--max-items`; `--max-media-items`; repeatable `--memory-type`; `--min-confidence`; `--freshness-seconds`; `--max-latency-ms`; `--reference-at`; `--scope`; `--allow-partial-sources` | context bundle plus `rendered` | yes |
| `get` | `MEMORY_ID` | memory object | yes |
| `speech` | `MEMORY_ID` | `{"segments":[...]}` | yes when the owner enables `embodied_operations` |
| `faces` | `MEMORY_ID` | `{"observations":[...]}` | yes when the owner enables `embodied_operations` |
| `register-speaker` | `SPEAKER_ID NAME`; `--relationship` | `{}` | no |
| `register-identity` | `IDENTITY_ID NAME`; `--relationship` | `{}` | yes when the owner enables `identity_operations` |
| `identity` | `IDENTITY_ID` | `{"identity":{...}}` or `{"identity":null}` | yes when the owner enables `identity_operations` |
| `record-consent` | `IDENTITY_ID STATE`; `--note` | `{"operation":{...}}` or `{"operation":null}` | yes when the owner enables `identity_operations` |
| `consent` | `IDENTITY_ID` | `{"consent":"granted"}` or `{"consent":null}` | yes when the owner enables `identity_operations` |
| `forget-identity` | `IDENTITY_ID` | erasure counts | yes when the owner enables `identity_operations` |
| `unlink-identity` | `ALIAS_ID` | `{"restored_identity_id":...}` | yes when the owner enables `identity_operations` |
| `reinforce` | one or more `MEMORY_ID` values | `{"reinforced":int}` | yes |
| `consolidation-candidates` | `--limit`; `--idle` | `{"candidates":[{"trigger":...,"memory_ids":[...],"evidence_count":int}]}` | no |
| `consolidate` | optional goal content; `--evidence-id`; `--limit`; `--trigger` | `{"operations":[...],"rejected":[...],"weighed":int}` | no |
| `deliberate` | `--limit`; `--max-rounds`; `--idle` | `{"rounds":int,"weighed":int,"skipped":int,"applied":int,"rejected":int,"model_calls":int}` | no |
| `apply` | `--operation` | `{"operation":{...}}` | no |
| `record-outcome` | `OPERATION_ID OUTCOME`; `--note` | `{"recorded":bool}` | no |
| `forget` | one or more `MEMORY_ID` values | `{"operation":{...}}` or `{"operation":null}` | no |
| `rollback` | `OPERATION_ID` | `{"rolled_back":bool}` | no |
| `operations` | `--limit` | `{"operations":[...]}`, each row carrying the `claim` an `identify` applied, the `identity` a `merge` moved, the `consent` a subject stated, and the `proposal` a `consolidate` wrote, or `null` | no |
| `export` | exactly one of `--identity-id` or repeatable `--memory-id` | export bundle | yes when the owner enables `identity_operations` |
| `apply-retention` | `--dry-run` | retention report | yes when the owner enables `identity_operations` |
| `list` | `--limit`; `--cursor` | `{"items":[...],"next_cursor":...}` | yes |
| `delete` | `MEMORY_ID` | `{"deleted":bool}` | yes |
| `reindex` | none | `{"memories":int}` | no |
| `optimize` | none | `{}` | no |
| `doctor` | none | composition and loader report | yes |

Defaults match the SDK: `add`, `add-many`, `add-stream`, and `capture` use
`memory_type=semantic`; search uses `limit=10`; ask uses `limit=5`; list, `settle`,
`pending-captures`, and `operations` use `limit=100`; `settle` also uses `max-attempts=3`;
`consolidate`, `consolidation-candidates`, and `deliberate` use `limit=32`, `consolidate` defaults
to `trigger=manual`, and `deliberate` also defaults to `max-rounds=4`; optional
retrieval roles and timestamps are unset. Timestamps must be timezone-aware ISO 8601 values.
Cursors are opaque and passed through unchanged. `ask` requires the selected composition to supply an answerer. `speech` and `faces`
return an empty result without a model call when the record has no corresponding media; otherwise
they require the matching capability. `compile` mirrors the
[`ContextBudget` defaults](../context-compilation.md#budget) and repeats `--memory-type` to keep
more than one type; `--max-latency-ms` is a deadline the compiler checks between stages, and the
printed bundle carries `elapsed_ms`, `deadline_exceeded`, and `unknowns` alongside its sections.
`--allow-partial-sources` explicitly enables digest-bound exact spans of eligible raw text parents
when a full parent does not fit. It is off by default, which preserves full-record-only compilation
and leaves the additive `excerpts` array empty. Each rendered excerpt line is labelled partial and
warns that omitted source text may qualify it.
Each row `operations` prints carries `operation_id`, `intent`, `trigger`, `evidence_ids`,
`target_ids`, `claim`, `consent`, `identity`, `rationale`, `model_id`, `recipe`, `created_ids`,
`changed_ids`, `forgotten_ids`, `superseded`, `applied_at`, `rolled_back_at`, `outcome`, and
`outcome_note` -- the same fields `apply` and `record-outcome` print back for the one operation
each names. `identity` is
set on the three rows that name a person instead of records -- the cross-modal `merge` the kernel
commits, the `correct` that `unlink-identity` logs, and the irreversible `forget` that
`forget-identity` logs -- and is `null` on every other row. `claim` carries the name an `identify`
row asserted and `consent` the state a `consent` row recorded, so a row about a person always says
what it said about them. `outcome` and `outcome_note` are `null` until `record-outcome` names the
operation, are post-hoc and never fed back into a decision, and a later `record-outcome` call
replaces an earlier judgement the same way `rollback` replaces a standing operation.

`consolidation-candidates`'s `--idle` declares an approved idle window, admitting lineages
nothing has ever weighed; the CLI never infers idleness from a clock. `deliberate` runs
`consolidation-candidates` and `consolidate` in a loop, each round consolidating every row
`consolidation-candidates` returns with that row's own trigger, until a round yields no
candidates or `--max-rounds` is reached; the counters it prints are summed across every round,
and its own `--idle` passes the same declaration through to `consolidation-candidates` on every
round. `apply` applies one host-supplied operation -- read from `--operation` in the same JSON
shape `operations` logs a row in -- through the same kernel validation a proposal gets, which is
the public replay path: reproducing a logged sequence against a fresh store configured with the
recipe that produced it reproduces the same derived IDs. A refused operation exits
`validation_error` naming the kernel's rejection reason.

`record-consent` takes `granted`, `withheld`, or `withdrawn`, and prints the logged operation, or
`null` when that statement already stands. `consent` reads back the standing state, where `null`
means nobody has recorded one. What the two restrained states change is listed under
[consent](python-sdk.md#consent).

`export` prints `exported_at`, `identity_id`, `identities`, `records`, and `operations` -- every
version of every record the subject appears in or is named by, and every log row that moved any of
it. Media is named by asset identity and digest; no bytes are printed, so a document stays safe to
pipe. Under `--url` the operation rows in an `export` bundle, and the row `record-consent` prints,
come from the [REST operation object](rest.md#response-objects), which carries the same fields --
`proposal` included -- so a row is the same document on either transport.
`apply-retention` prints `dry_run`, `media_memory_ids`, `forgotten_memory_ids`,
`cascade_memory_ids`, `asset_ids`, `capture_memory_ids`, and `deleted`;
`cascade_memory_ids` names additional derived records whose support the direct selections remove.
It deletes through the same path as `delete`, so
[what `delete` removes](python-sdk.md#what-delete-removes) applies unchanged. Start with
`--dry-run`.

`apply-retention` acts on the policy the composition declares. An `--embedder` composition
declares none, so it deletes nothing and reports empty lists; use `--app` with a memory built from
a configuration that has a [`retention` section](../configuration.md#retention-policy). The CLI
deliberately has no flag for the ages: a policy that deletes should be reviewable in a file, not
retyped on each invocation.

`--freshness-seconds` must be positive, finite, and representable as a Python `timedelta`; invalid
values use the CLI's `validation_error` exit status without sending a remote request.

`forget` is cognitive forgetting, reversible with `rollback`; `delete` is erasure. With the
default `reinforce_on_answer=True`, `ask` also reinforces the hits the answerer cites; use
`--app` to construct a memory with that policy disabled.

`ask` may run face recognition on a retrieved photo or video before answering. With the default
`--link-identities` (true), a voice-and-face pair corroborated across enough assets is fused into
one identity and logged as a `merge` row, the same as `analyze_faces`; `--no-link-identities`
still runs recognition to answer the question but never commits that bind. `--url` compositions
send no such request to REST: `POST /v1/answers` decides the same question from `create_app`'s own
`embodied_operations` switch instead, so `--link-identities` has no effect there and the remote
answerer links identities only when the host has opted in to embodied operations.

### Content and JSONL input

Content commands (`add`, `capture`, `search`, `search-with-trace`, `ask`, `compile`, and
`consolidate`) accept exactly one of these forms:

- Positional atoms: bare values are text, `@PATH` is a local file, `@@TEXT` is text beginning with
  a literal `@`, and `-` reads stdin. Order is preserved.
- `--content-json VALUE`: a JSON string or a REST-shaped content-parts array. `VALUE` is literal
  JSON, `@PATH` to a UTF-8 JSON file, or `-` for stdin. Local mode applies the Python size bounds.

Passing positional atoms and `--content-json` together returns `validation_error` before any
backend is composed or request is sent. Missing positional content reads stdin as one text atom;
stdin may be claimed only once per invocation.

Local `--content-json` adds one part source to the REST union:

```json
{"type":"input_file","path":"/srv/media/panel.png"}
```

`input_file.path` and positional `@PATH` are rejected with `--url`; send base64 media accepted by
the [REST content contract](rest.md#content-input).

`--metadata` and `--context` on `add`, and `--scope` on the retrieval commands, each accept a
literal JSON object, `@PATH`, or `-`. `add-many` and `add-stream` read non-empty JSONL from a
literal value, `@PATH`, or stdin. Each non-blank line is an object with required `content` and
optional `occurred_at`, `occurred_end`, `metadata`, and `context`; unknown fields are rejected.
`--memory-type` applies to every line.

The CLI context decoder accepts `basis`, `source_id`, `confidence`, `valid_from`, `valid_until`,
and `spatial`; its scope decoder accepts `valid_at`, `known_at`, `near`, `radius_m`, `place_id`,
and `identity_id`, which is the whole of [`RetrievalScope`](python-sdk.md#public-values).

```bash
mindbridge --embedder jina-omni add "The mug is on the table" \
  --context '{"basis":"observation","source_id":"camera-1:42","confidence":0.94}'

mindbridge --embedder jina-omni search "Where is the mug?" \
  --scope '{"near":{"frame_id":"home/map","anchor":"subject","x":2,"y":1},"radius_m":0.75}'
```

`add-many` collects all lines for one model batch and one SQLite transaction. `add-stream` commits
and makes each line searchable before reading the next, but collects returned records for the
CLI's single-document stdout result. It is therefore for finite JSONL; use
[`Memory.add_stream`](python-sdk.md#memory-operations) for an unbounded source.

### Operations without a command

One Python operation has no command. `ask-stream` does not exist because the value of
[`Memory.ask_stream`](python-sdk.md#memory-operations) is delivery of the answer while the model
is still producing it, and this CLI emits one stable JSON document per invocation. Collecting the
deltas to fit that contract would produce exactly what `ask` already prints, so the command would
be a second spelling of `ask` rather than a streaming one. Use `ask` here, and the SDK when time
to first token matters.

### Output

On success, stdout contains one JSON document and a trailing newline. Local results use the same
memory, hit, answer, page, and deletion field vocabulary as [REST response objects](rest.md#response-objects).
`search-with-trace` serializes the [Python retrieval trace](python-sdk.md#public-values).
`SpeakerSegment` and `FaceObservation` use their public Python fields. Memory and hit documents
carry `context` when typed semantics exist, identically in local and `--url` mode, with enum
values as JSON strings and datetimes as ISO 8601. Every timestamp the CLI prints -- including
`enqueued_at`, `applied_at`, and `rolled_back_at` -- renders a UTC instant with a trailing `Z`,
never `+00:00`, so one instant has one spelling across commands.

`forget-identity` returns `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`,
`face_observations`, and `speech_segments`, matching the fields of `IdentityErasure`.

`pending-captures` returns one object per record whose deferred work is not finished — `memory_id`,
ISO 8601 `enqueued_at`, `attempts`, `last_error`, and `awaiting` — oldest first, matching the
fields of `PendingCapture`. `awaiting` is `"enrichment"` for a record that has no vectors yet and
`"formation"` for one that is already searchable and owes only formation. Naming memory IDs
restricts the report to them; an ID that is absent from the result is not pending, so it is either
settled or unknown, and `get` tells the two apart.

`settle` accepts the same memory IDs. Naming records settles only those and ignores
`--max-attempts` for them, which is how a capture parked at the retry ceiling is retried by hand.
`add-stream --capture` commits each item through `capture` rather than `add`, so the items are
durable but unsearchable until `settle` runs.

Unless `--quiet` is set, commands write the resolved composition as one JSON document on stderr
before executing. `--url` forwards successful owner response objects unchanged, except for the
four whose REST shape is not the command's own document: `pending-captures` prints `pending` where
the route returns `items`, `forget-identity` prints the erasure counts the route nests under
`erasure`, `register-identity` prints `{}` rather than the route's `registered`, and `identity`
prints the three fields the local command prints, dropping the route's `confirmed` and
`evidence_ids`. One command prints one shape on both transports. Runtime
failures write one [shared error envelope](rest.md#error-envelope) to stderr and nothing to stdout;
remote envelopes, including `trace_id`, are forwarded unchanged. Local CLI and MCP errors retain
`subject`, including local paths; unauthenticated REST redacts storage subjects.

### Doctor and explain

`--explain` resolves and prints the selected composition without running the command. A command is
still syntactically required.

`doctor` returns the installed MindBridge and Python versions plus composition-specific checks:

- `--embedder` constructs each configured recipe, exercises its published loader probe, and
  reports the data-directory state without writing memory data. The probed backends also fill
  `capabilities`, which is `MemoryCapabilities.document()` -- the same document `GET /healthz`
  serves and the MCP server greets an agent with, narrowed on each network surface to the
  operations it serves, while `doctor` prints the whole derived `operations` set. It is
  declared by the backends, so producing it opens no store and creates no data directory. Every
  probed backend stays loaded until the capability summary is built and is then closed, so a
  doctor run holds the weights of all configured slots at once.
- `--app` imports and resolves the target but does not call a factory, because that could open the
  store, so `capabilities` is `null`: the application owns its own backends.
- `--url` calls the owner's `GET /healthz` with the configured timeout.

Loader failures are reported inside the successful doctor document so all configured slots can be
inspected together. Operational diagnosis is covered in [operations](../operations.md).

## Errors and limits

### Exit codes

| Exit | `code` | Meaning |
| --- | --- | --- |
| 0 | — | success |
| 1 | `internal_error` or an unmapped code | unexpected failure |
| 2 | — | argparse syntax or type error |
| 3 | `validation_error` | input rejected |
| 4 | `memory_not_found` | memory does not exist |
| 5 | `speaker_not_found` | speaker does not exist |
| 6 | `model_error` | inspect `reason` and `retryable` |
| 7 | `storage_error` | storage or remote request failure |
| 8 | `index_unavailable` | vector index unavailable |
| 9 | `storage_error/data_dir_in_use` | another live process owns the directory; use `--url` |
| 10 | `configuration_error` | composition or remote-capability failure |
| 11 | `model_output_truncated` | generation hit its output limit |
| 12 | `identity_not_found` | face/voice identity does not exist |
| 130 | — | interrupted |

`configuration_error` reasons are `composition_missing`, `option_not_applicable`, `app_invalid`,
`missing_dependency`, `composition_failed`, and `unsupported_in_remote_mode`. Exit 9 is selected by
`reason` rather than outer code. Runtime errors use the JSON envelope; argparse exit 2 and interrupt
exit 130 use their conventional plain stderr diagnostics.

### Operations without a remote route

Every command whose operation has a `/v1` route runs under `--url`; the `--url` column above is
the complete list. Eleven commands exit 10 with `unsupported_in_remote_mode` because the
operation has no route at all, not because the CLI declines to send one:

- The control plane is SDK-only by [documented design](../context-os.md): `consolidation-candidates`,
  `consolidate`, `deliberate`, `apply`, `record-outcome`, `forget`, `rollback`, and `operations`.
- `add-stream`, `reindex`, and `optimize` are local ingestion and index maintenance on the owner's
  own directory, and `register-speaker` has no route of its own.

A gated command sent to an owner that did not enable the matching switch also exits 10, naming
that switch: `create_app` never registers those routes, so the owner answers that the route does
not exist rather than saying anything about the person or record the command named. The complete
route boundary is listed in
[REST operations without a route](rest.md#operations-without-a-route).

### Input limits

Local `--app` and `--embedder` mode applies the Python bounds: at most 128 content parts, 65,536
combined normalized text characters, 262,144 UTF-8 metadata bytes, 512 MiB per local asset, and
search, answer, or page limits from 1 through 100. `--url` applies the REST bounds, including 16
content parts and the 8 MiB request body. Local JSONL has no separate item-count cap; remote
`add-many` is capped at 100 items by REST.

There is no configuration file, `MINDBRIDGE_*` composition variable, plugin registry, interactive
prompt, streaming stdout, output-format switch, `serve` command, capture-event reducer, or
coordinate-frame transform. Use the Python configuration boundary and an ASGI server instead of a
second CLI configuration or server implementation. `add-stream` is for finite finalized JSONL;
use `AsyncAudioStream`, `AsyncVisionStream`, or `AsyncCaptureStream` in Python for live sensor
packets and associated update/final/cancel capture events.
