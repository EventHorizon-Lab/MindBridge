# Command-line API

## Surface

`mindbridge` runs one product operation in one process and emits one JSON result. Local commands
dispatch to the corresponding `Memory` method, except for the diagnostic `doctor`; remote commands
forward to a running owner's `/v1` route. The CLI owns argument decoding, composition selection,
JSON projection, and exit-code mapping, not storage or retrieval policy.

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
| `--former NAME` | formation recipe for `--embedder` |
| `--transcriber NAME` | speech recipe for `--embedder` |
| `--index-speech`, `--no-index-speech` | index transcripts on add for `--embedder`; default on |
| `--minimum-relevance FLOAT` | relevance floor; default `0.10` |
| `--ambiguity-margin FLOAT` | top-two gate when `limit=1`; default `0.01` |
| `--decay-half-life-days FLOAT` | optional positive recency half-life; default none |
| `--explain` | print resolved composition and execute no operation |
| `-q`, `--quiet` | suppress the composition banner on stderr |
| `-V`, `--version` | print the installed version |

`--data-dir`, backend options, and tuning options apply only to `--embedder`; `--timeout` applies
only to `--url`. Inapplicable options fail instead of being ignored. `--app` follows the
`MODULE:ATTR` convention and adds the working directory to `sys.path`.

The listed flags are the complete direct-composition surface. Use `--app` for index quantization,
face analysis, speaker/face thresholds, a custom backend, or any other `Memory` setting.

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
| `openai` | `--embedder`, `--answerer`, `--former`, `--transcriber` | `text-embedding-3-small`, `gpt-5-mini` (generation and formation), `whisper-1` |

Only `openai` accepts a model suffix, for example `--answerer openai:gpt-5-mini`. Recipe names form
a closed table; use `--app` for other backends. Provider trust, license, model identity, and
credential behavior live in [configuration](../configuration.md).

### Commands

| Command | Operands and options | JSON result | `--url` |
| --- | --- | --- | --- |
| `add` | content; `--occurred-at`; `--occurred-end`; `--metadata`; `--memory-type`; `--context` | memory object | yes |
| `add-many` | optional JSONL source; `--memory-type` | `{"memories":[...]}` | yes |
| `add-stream` | optional JSONL source; `--memory-type` | `{"memories":[...]}` | no |
| `search` | content; `--limit`; `--memory-type`; `--reference-at`; `--scope`; `--occurred-from`; `--occurred-until` | `{"hits":[...]}` | yes |
| `search-with-trace` | search options | `{"hits":[...],"trace":{...}}` | no |
| `ask` | content; `--limit`; `--memory-type`; `--reference-at`; `--scope` | answer object | yes |
| `get` | `MEMORY_ID` | memory object | yes |
| `speech` | `MEMORY_ID` | `{"segments":[...]}` | no |
| `faces` | `MEMORY_ID` | `{"observations":[...]}` | no |
| `register-speaker` | `SPEAKER_ID NAME`; `--relationship` | `{}` | no |
| `register-identity` | `IDENTITY_ID NAME`; `--relationship` | `{}` | no |
| `identity` | `IDENTITY_ID` | `{"identity":{...}}` or `{"identity":null}` | no |
| `unlink-identity` | `ALIAS_ID` | `{"restored_identity_id":...}` | no |
| `reinforce` | one or more `MEMORY_ID` values | `{"reinforced":int}` | no |
| `list` | `--limit`; `--cursor` | `{"items":[...],"next_cursor":...}` | yes |
| `delete` | `MEMORY_ID` | `{"deleted":bool}` | yes |
| `reindex` | none | `{"memories":int}` | no |
| `optimize` | none | `{}` | no |
| `doctor` | none | composition and loader report | yes |

Defaults match the SDK: `add`, `add-many`, and `add-stream` use `memory_type=semantic`; search uses
`limit=10`; ask uses `limit=5`; list uses `limit=100`; optional retrieval roles and timestamps are
unset. Timestamps must be timezone-aware ISO 8601 values. Cursors are opaque and passed through
unchanged. `ask` requires the selected composition to supply an answerer. `speech` and `faces`
return an empty result without a model call when the record has no corresponding media; otherwise
they require the matching capability.

### Content and JSONL input

Content commands (`add`, `search`, `search-with-trace`, and `ask`) accept exactly one of these
forms:

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

### Output

On success, stdout contains one JSON document and a trailing newline. Local results use the same
memory, hit, answer, page, and deletion field vocabulary as [REST response objects](rest.md#response-objects).
`search-with-trace` serializes the [Python retrieval trace](python-sdk.md#public-values).
`SpeakerSegment` and `FaceObservation` use their public Python fields. Memory and hit documents
carry `context` when typed semantics exist, identically in local and `--url` mode, with enum
values as JSON strings and datetimes as ISO 8601.

Unless `--quiet` is set, commands write the resolved composition as one JSON document on stderr
before executing. `--url` forwards successful owner response objects unchanged. Runtime
failures write one [shared error envelope](rest.md#error-envelope) to stderr and nothing to stdout;
remote envelopes, including `trace_id`, are forwarded unchanged. Local CLI and MCP errors retain
`subject`, including local paths; unauthenticated REST redacts storage subjects.

### Doctor and explain

`--explain` resolves and prints the selected composition without running the command. A command is
still syntactically required.

`doctor` returns the installed MindBridge and Python versions plus composition-specific checks:

- `--embedder` constructs each configured recipe, exercises its published loader probe, closes it,
  and reports the data-directory state without writing memory data.
- `--app` imports and resolves the target but does not call a factory, because that could open the
  store.
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

With `--url`, only `add`, `add-many`, `search`, `ask`, `get`, `list`, `delete`, and `doctor` are
available. Other commands exit 10 with `unsupported_in_remote_mode`; their SDK operations have no
REST route. The complete route boundary is listed in
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
