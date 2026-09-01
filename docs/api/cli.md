# Command-line reference

MindBridge ships two console scripts. Together they are one documented CLI surface.

| Script | Entry point | Purpose |
| --- | --- | --- |
| `mindbridge` | `mindbridge.cli:main` | Product memory operations over one composed `Memory` |
| `mindbridge-bench` | `mindbridge.benchmarks.cli:main` | Benchmark harnesses over the public SDK |

They are two scripts rather than one command tree for a reason worth knowing before anyone tries to
merge them. `tests/test_package.py` walks the AST of every non-benchmark module under
`src/mindbridge` and fails on any reference to the benchmark package — including a bare string
constant. A single dispatcher could not reach the benchmark family even through a lazy
`import_module("mindbridge.benchmarks.cli")`, because the string itself trips the guard. The guard is
deliberate: product modules must not import benchmark modules. So `mindbridge` never names the
benchmark package in any form, and the two families stay in two entry points.

`mindbridge` is a per-invocation process: it composes one `Memory`, runs one operation, and closes
it. Opening a `Memory` acquires the directory lock, opens SQLite and the index, drains the outbox,
and — for a local model recipe — loads weights. **That makes it the wrong tool for a loop.** Use the
[Python SDK](python-sdk.md) in-process, or run one owner and address it with `--url`.

## Composition

A CLI cannot accept Python objects, and `Memory` requires `embedder` as a keyword with no default.
Exactly one of three flags supplies it. There is **no default composition and no environment
variable that selects a backend**; a `MINDBRIDGE_*` variable that quietly picked a model would be
exactly the hidden provider construction
[the design principles forbid](../design-principles.md#prefer-explicit-configuration-and-observable-automation).

```bash
mindbridge add "The spare key is in the blue toolbox."
```

fails with exit `10` and names all three paths.

### `--app MODULE:ATTR`

The primary path, and the only extensible one.

```bash
mindbridge --app my_application:build_memory search "where is the spare key"
```

`ATTR` is a `Memory` instance or a zero-argument callable returning one. This is the same
`module:attr` convention as `uvicorn my_application:app`, and the same application composition the
SDK, REST, and MCP already use. The working directory is placed on `sys.path`, as `uvicorn` does.

**Any backend MindBridge does not bundle is reached this way.** There is deliberately no plugin
registry, no entry-point discovery, and no way to register a backend by name.

The application owns the data directory and the backends it built, so `--data-dir`, `--answerer`,
`--transcriber`, and the tuning options are refused with `--app` rather than silently ignored.

### `--embedder NAME`

A closed table over the bundled backends, published as `mindbridge.recipes` so the CLI has no
private path into the package:

```python
from mindbridge import recipes

recipes.names()  # ("funasr", "jina-omni", "openai")
recipes.describe("jina-omni")  # static identity; constructs nothing
embedder = recipes.embedder("jina-omni")  # the object, which the caller now owns
```

| Recipe | Fills | Model identity | Extra |
| --- | --- | --- | --- |
| `jina-omni` | `--embedder` | `DEFAULT_JINA_MODEL_ID` at `DEFAULT_JINA_REVISION`, 1024 dimensions | `local` |
| `funasr` | `--transcriber` | `DEFAULT_FUNASR_RECIPE`, pinned model and component revisions | `local` |
| `openai` | `--embedder`, `--answerer`, `--transcriber` | `text-embedding-3-small`, `gpt-5-mini`, `whisper-1` | `openai` |

`openai:<model>` selects a different model for the slot it fills, for example
`--answerer openai:gpt-5-mini`.

```bash
mindbridge --embedder jina-omni add "The spare key is in the blue toolbox."
mindbridge --embedder jina-omni --answerer openai:gpt-5-mini ask "where is the spare key"
```

**The pinned `jina-omni` weights are licensed CC BY-NC 4.0 — non-commercial use only.** That licence
covers the model, not MindBridge. `mindbridge --embedder jina-omni --explain <command>` prints it,
and so does `doctor`.

This is not hidden construction, on each of the four counts that would make it so:

| Property of hidden construction | Why it does not hold |
| --- | --- |
| The caller did not name what was built | The recipe name is typed on every invocation. There is no default and no environment fallback. |
| The caller cannot obtain the object | `recipes.embedder(name)` returns it. Call it, print it, wrap it, or ignore it and build your own. |
| The choice can drift | The table is closed, versioned with the package, and pins model ID and revision to constants in the source. No discovery, no network lookup, no third-party registration. |
| It happens as a side effect | It happens only when the flag is passed, and the resolved identity is echoed to stderr on every run. |

MindBridge never reads a credential. `--answerer openai:gpt-5-mini` constructs `OpenAI()`, and the
official SDK performs its own documented `OPENAI_API_KEY` lookup. The CLI reports the *source* of the
key and never its value.

Changing `--embedder` against an existing `--data-dir` cannot silently migrate an embedding space:
the recorded `embedding_model`, `embedding_space`, and `embedding_dimension` are re-checked on open
and force a rebuild.

### `--url URL`

Address a running owner over `/v1`. One physical directory has one live owner, so a CLI that finds
the directory already taken must reach that owner through a supported transport rather than opening
a second `Memory`.

```bash
mindbridge --url http://127.0.0.1:8000 search "where is the spare key"
```

In this mode the CLI constructs no backends at all: it sends JSON to `/v1` and echoes the response
body unchanged. HTTP uses `urllib.request` from the standard library. Each blocking remote operation
has a 30-second timeout by default; set another positive finite value with `--timeout SECONDS`.
This bounds the CLI connection/read wait and does not change the owner's model-provider timeouts.
A timeout exits `7` with `storage_error`, `reason="timeout"`, `stage="request"`, and
`retryable=true`. The resolved timeout appears in the normal composition banner and `--explain`
output as `timeout_seconds`.

Exit `9` from a local composition means another process owns the directory — retry with `--url`.

## Global options

```text
--data-dir PATH          local memory directory (default: .mindbridge)
--app MODULE:ATTR      }
--embedder NAME        }  exactly one is required
--url URL              }
--timeout SECONDS        remote request timeout with --url (default: 30)
--answerer NAME          generation recipe, with --embedder
--transcriber NAME       speech recipe, with --embedder
--index-speech           index transcripts and speaker identities on add
--minimum-relevance F    weak-evidence floor (default: 0.55)
--ambiguity-margin F     top-two gate when limit=1 (default: 0.01)
--decay-half-life-days F opt-in recency decay (default: none)
--explain                print the resolved composition to stdout and execute nothing
-q, --quiet              suppress the stderr composition banner
-V, --version            print the installed version
```

Every default above is read from the `Memory` signature it is passed to, so `--help` shows the real
value and the CLI cannot drift from it. Global options precede the command:
`mindbridge --embedder jina-omni -q add "..."`.

There is **no `--format` flag and no configuration file**. Output is always JSON, and a second
configuration system is out of scope.

## Commands

Derived mechanically from the SDK: each public `Memory` operation is the command of the same name,
kebab-cased. Nothing is renamed, grouped, or invented.

| SDK operation | Command | Writes? |
| --- | --- | --- |
| `add` | `mindbridge add [TEXT ...]` | idempotent write |
| `add_many` | `mindbridge add-many [JSONL]` | idempotent write |
| `add_stream` | `mindbridge add-stream [JSONL]` | incremental idempotent writes |
| `search` | `mindbridge search [QUERY ...]` | read, plus a lazy transcript cache |
| `search_with_trace` | `mindbridge search-with-trace [QUERY ...]` | read, plus a lazy transcript cache |
| `ask` | `mindbridge ask [QUESTION ...]` | read, plus a lazy transcript cache |
| `get` | `mindbridge get MEMORY_ID` | read |
| `speech` | `mindbridge speech MEMORY_ID` | read, plus a transcript cache |
| `faces` | `mindbridge faces MEMORY_ID` | read, plus a face-analysis cache |
| `register_speaker` | `mindbridge register-speaker SPEAKER_ID NAME` | write |
| `register_identity` | `mindbridge register-identity IDENTITY_ID NAME` | write |
| `reinforce` | `mindbridge reinforce MEMORY_ID ...` | write |
| `list` | `mindbridge list` | read |
| `delete` | `mindbridge delete MEMORY_ID` | destructive, idempotent |
| `reindex` | `mindbridge reindex` | maintenance |
| `optimize` | `mindbridge optimize` | maintenance |
| `close` | — | lifecycle: one invocation opens and closes one `Memory` |

Plus exactly one command with no SDK counterpart:

| Command | Purpose |
| --- | --- |
| `mindbridge doctor` | Resolve the composition, exercise each configured backend's loader, and report — writing nothing |

Per-command options mirror the SDK keywords: `--occurred-at`, `--occurred-end`, `--metadata`, and
`--memory-type` on `add`; `--memory-type` on `add-many` and `add-stream`; `--limit`,
`--memory-type`, and `--reference-at` on `search`, `search-with-trace`, and `ask`; the two search
commands also accept `--occurred-from` and `--occurred-until` for strict event-overlap filtering;
`--limit` and `--cursor` apply to `list`.

`--cursor` is passed through exactly as it was returned and is never parsed.

## Input

Three forms, chosen so generated input never needs shell quoting.

**Positional atoms.** A bare argument is text, `@PATH` is a local file, and `@@TEXT` is text with a
literal leading `@`. Order is preserved and becomes the ordered `ContentInput` sequence:

```bash
mindbridge --embedder jina-omni add "Inspection evidence" @panel.png @note.wav
```

**Standard input.** A missing positional, or `-` in the atom list, reads all of stdin as one text
atom. Stdin can be read once per invocation.

```bash
printf 'The spare key is in the blue toolbox.' | mindbridge --embedder jina-omni add
```

**`--content-json`.** Reads the same parts array REST and MCP accept — the identical
`input_text` / `input_image` / `input_file` union, documented in
[Content input](rest.md#content-input). The value is `-` for stdin, `@PATH` for a file, or the JSON
itself. `--metadata` accepts the same three forms.

```bash
mindbridge --embedder jina-omni add --content-json - <<'JSON'
[{"type": "input_text", "text": "Inspection evidence"},
 {"type": "input_file", "path": "/srv/media/panel.png"}]
JSON
```

`--content-json` and the positional atoms are two ways to supply the **same** operand, not two
halves of one, so passing both is refused with exit `3` during argument validation — before any
backend is composed and before any request is sent. Preferring either would store a memory missing
what the caller typed, or run a different query than the one that was asked.

The CLI adds exactly one part type to that union, `{"type": "input_file", "path": "..."}`. It is
valid in local mode only, because the CLI runs on the same machine as the data directory and the SDK
already accepts a `pathlib.Path` atom. REST and MCP refuse local paths on purpose, so in `--url`
mode both this part type and `@PATH` are rejected with `unsupported_in_remote_mode`; send base64
media in a data URL instead.

`add-many` and `add-stream` read JSONL, one object per line, each with `content` plus optional
`occurred_at`, `occurred_end`, and `metadata`. `--memory-type` applies to every item. `add-many`
collects the finite input into one embedding batch and one SQLite transaction. `add-stream` parses
and commits one line before reading the next; a later failure leaves the committed prefix in the
store. Each line's `content` is the same union, checked by the same rule, so a local path is refused
in `--url` mode exactly as a single `add` refuses one.

```bash
mindbridge --embedder jina-omni add-many @import.jsonl
mindbridge --embedder jina-omni add-stream @completed-observations.jsonl
```

`add-stream` keeps input lazy but collects returned records for the CLI's one-document stdout
contract, so it is for finite JSONL. Use Python `Memory.add_stream` or `AsyncMemory.add_stream` for
an unbounded source.

## Output

**One JSON document per invocation on stdout, and nothing else.** Diagnostics — the composition
banner, warnings — are JSON on stderr. On failure, stdout is empty.

Shapes are the REST response shapes, projected from the same values REST projects them from, so one
field vocabulary covers three surfaces:

| Command | stdout |
| --- | --- |
| `add`, `get` | `MemoryResponse` |
| `add-many`, `add-stream` | `{"memories": [MemoryResponse, ...]}` |
| `search` | `{"hits": [SearchHitResponse, ...]}` |
| `search-with-trace` | `{"hits": [SearchHitResponse, ...], "trace": RetrievalTrace}` |
| `ask` | `{"answer": str, "hits": [SearchHitResponse, ...], "abstained": bool, "abstention_reason": str \| null}` |
| `list` | `{"items": [MemoryResponse, ...], "next_cursor": str \| null}` |
| `delete` | `{"deleted": bool}` |
| `speech` | `{"segments": [SpeakerSegment, ...]}` |
| `faces` | `{"observations": [FaceObservation, ...]}` |
| `reinforce` | `{"reinforced": int}` |
| `reindex` | `{"memories": int}` |
| `register-speaker`, `register-identity`, `optimize` | `{}` |
| `doctor`, `--explain` | the composition document below |

`SpeakerSegment` has no REST or MCP precedent, because `speech` has no route and no tool. Its fields
are the public [`SpeakerSegment`](python-sdk.md) dataclass: `asset_id`, `start_ms`, `end_ms`, `text`,
`speaker_id`, `speaker_name`, `identity_score`. When `speech` gains a route, the route must adopt
this shape rather than inventing a second one.

`FaceObservation` likewise follows the public dataclass fields: `asset_id`, `observed_at_ms`,
normalized `bounding_box`, `identity_id`, `identity_name`, and `identity_score`.

Every run echoes the resolved composition to stderr before executing, so a log records which model
identity produced which write. `-q` suppresses it.

```json
{"source": "--embedder jina-omni", "data_dir": "/srv/assistant/.mindbridge",
 "embedder": {"recipe": "jina-omni", "class": "mindbridge.models.jina.JinaOmniEmbedder",
              "slots": ["embedder"], "models": {"embedder": "jinaai/jina-embeddings-v5-omni-small-retrieval"},
              "revision": "e3ae4b6e…", "embedding_dimension": 1024,
              "license": "CC BY-NC 4.0", "extra": "local"},
 "answerer": null, "transcriber": null}
```

## Exit codes

Stable, one per error code, so an agent can branch on `$?` without parsing anything.

| Exit | `code` | Meaning | Retry? |
| --- | --- | --- | --- |
| 0 | — | Success | — |
| 1 | `internal_error` | Unexpected failure; a bug | no |
| 2 | — | Argument parsing; argparse's own status | no |
| 3 | `validation_error` | Input rejected | no |
| 4 | `memory_not_found` | | no |
| 5 | `speaker_not_found` | | no |
| 6 | `model_error` | See `reason` and `retryable` | conditional |
| 7 | `storage_error` | See `reason` | conditional |
| 8 | `index_unavailable` | | usually |
| 9 | `storage_error` + `data_dir_in_use` | Another process owns `--data-dir`; **retry with `--url`** | yes |
| 10 | `configuration_error` | No composition, unknown recipe, missing extra, an option the chosen composition does not own, or an operation with no `/v1` route in `--url` mode | no |
| 11 | `model_output_truncated` | Generation stopped at an output token limit | no |
| 12 | `identity_not_found` | Face/voice identity does not exist | no |
| 130 | — | Interrupted | — |

Exit `9` is the one status selected by `reason` rather than `code`, because it is the CLI's single
transport decision. Exit `10` is the CLI's own: `configuration_error` is not an SDK exception, and
none was added for it — the composition layer is the one policy the CLI owns.

On failure the CLI writes the [shared error envelope](rest.md#error-envelope) to **stderr** as JSON
and nothing to stdout. In `--url` mode the owner's envelope is forwarded verbatim, `trace_id` and
all, and its `code` selects the exit status.

Unlike REST, the CLI fills `subject` for every code. It runs as the invoking user on the machine
that owns `data_dir`, so a local path or a failing batch position is information the caller already
holds; over an unauthenticated HTTP API it would be server state.

## `mindbridge doctor`

Resolves the composition, exercises each configured backend's loader, and reports — without writing
anything.

```bash
mindbridge --embedder jina-omni --transcriber funasr --data-dir /srv/assistant/.mindbridge doctor
```

```json
{
  "version": "0.2.0",
  "python": "3.12.7",
  "data_dir": "/srv/assistant/.mindbridge",
  "data_dir_state": "free",
  "composition": {
    "source": "--embedder jina-omni",
    "embedder": {
      "recipe": "jina-omni",
      "class": "mindbridge.models.jina.JinaOmniEmbedder",
      "models": {"embedder": "jinaai/jina-embeddings-v5-omni-small-retrieval"},
      "revision": "e3ae4b6e…",
      "license": "CC BY-NC 4.0",
      "extra": "local",
      "probe": "weights",
      "loader": "ok",
      "embedding_model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "embedding_space": "…",
      "embedding_dimension": 1024,
      "embedding_modalities": ["audio", "image", "text", "video"]
    },
    "answerer": null,
    "transcriber": {
      "recipe": "funasr",
      "class": "mindbridge.models.funasr.FunASRTranscriber",
      "probe": "import",
      "loader": "failed",
      "reason": "missing_dependency",
      "detail": "torchaudio"
    }
  }
}
```

`loader` is the point of the command: an under-declared dependency becomes one line before the first
write instead of a run of silent ingestion failures. `probe` says how deep the check reached, so the
report never overstates it:

| `probe` | What `doctor` ran |
| --- | --- |
| `weights` | Constructed the backend and loaded its pinned weights |
| `client` | Constructed the official SDK client, which resolves its own credentials |
| `import` | Imported the deferred runtime. `FunASRTranscriber` publishes no loader, so this is the deepest check available without reaching into it |

`data_dir_state` is `absent`, `free`, or `in use by another process`, determined without creating
anything: with no lock file there is no live owner. With `--app` it is `owned by the application`,
and the composition is resolved but **not called**, because calling the factory would open the
store.

`doctor` does not report the store's recorded embedding space. Reading it means either opening a
`Memory` — which may re-embed, and so is a write — or duplicating the on-disk schema inside the CLI.
The existing store-metadata guard already fails the first real operation with a clear error.

## Current limits

### Operations without a remote route

`--url` covers `add`, `add-many`, `search`, `ask`, `get`, `list`, and `delete`. The other nine CLI
commands exit `10` with `unsupported_in_remote_mode` and name the surfaces that do support them.
That mirrors the [REST gap](rest.md#operations-without-a-route) honestly rather than hiding it:
`speech`, `faces`, identity registration, and `reinforce` are implementation gaps, and `reindex`
and `optimize` are owner-process maintenance that must not be reachable by an unauthenticated
client. `add-stream` remains local because REST has no client-streaming route.
Remote callers submit completed observations with ordinary `add` requests.

### Backends without a recipe

`SentenceTransformersEmbedder` and `OpenCVFaceAnalyzer` have no named recipes. Their contracts
require explicit model identities and, for OpenCV, explicit YuNet and SFace weight paths.
`SentenceTransformersEmbedder` requires an explicit 40-character
commit hash, and the choice of model is the application's; a recipe would have to pin one on the
caller's behalf. Use `--app`, or the two-line composition in
[Choosing an embedding backend](../quickstart.md#choose-an-embedding-backend).

### Input limits

Local mode applies the Python limits: 128 content parts per operation, 65,536 characters per text
part, `limit` between 1 and 100, and media bounded only by disk. `--url` mode applies the REST
limits, including the 8 MiB request body. See
[REST input limits](rest.md#input-limits).

### Absent features

No `--format` flag, configuration file, `MINDBRIDGE_*` composition variable, plugin registry,
backend registration by name, streaming output, interactive prompt, `serve` command, or metadata
filter. `uvicorn my_application:app` already starts a server. `add-stream` still emits one document
at EOF and is therefore for finite JSONL; use the Python iterator for an unbounded source.

## Benchmarks

The benchmark family is unchanged:

```bash
mindbridge-bench --help
mindbridge-bench eval --tasks list
mindbridge-bench locomo-refined --help
mindbridge-bench local-index --help
```

Benchmark commands exercise the public SDK and never define an alternate product path. Every
benchmark unit receives a separate physical directory. See
[the benchmark guide](../benchmarking.md).
