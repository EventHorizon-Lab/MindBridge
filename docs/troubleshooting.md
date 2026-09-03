# Troubleshooting

Diagnose the stable error fields before reading the message:

```python
from mindbridge import MindBridgeError

try:
    result = memory.ask("What happened yesterday?")
except MindBridgeError as error:
    print(error.code, error.reason, error.stage, error.subject, error.retryable)
```

`code` is the stable family, `reason` narrows the cause, `stage` locates the failed pipeline step,
`subject` identifies the affected item when available, and `retryable` states whether the identical
request can succeed later. CLI, REST, and MCP envelopes add a `trace_id`; preserve it with request
metadata. See the [REST error contract](api/rest.md#codes-and-reasons).

For composition failures, run the same selection with `mindbridge ... doctor`. Bundled recipes
exercise their loaders, `--app` verifies the import without calling a factory, and `--url` checks
`/healthz`.

## Error triage

| Signal | First check | Do not assume |
| --- | --- | --- |
| `validation_error`, REST 422 | Input shape, media type, event bounds, IDs, and declared modality support | Retrying unchanged input will help. |
| `model_error` | Backend slot, credentials, endpoint, timeout, provider response, and capability declaration | SQLite or Zvec is damaged. |
| `model_output_truncated` | Generation output limit and ask evidence limit | An identical retry will finish. |
| `index_unavailable` | Zvec open, mutation, flush, file descriptors, and collection files | The preceding SQLite write rolled back. |
| `storage_error` | Directory ownership, filesystem, assets, SQLite, schema, and compatibility metadata | Deleting `zvec/` repairs authoritative data. |
| `memory_not_found`, REST 404 | The ID and authoritative SQLite record | A stale Zvec candidate is a record. |

Retry only when `retryable` is true. The closed retryable reasons are `connection_failed`,
`data_dir_in_use`, `flush_failed`, `index_missing`, `rate_limited`, and `timeout`. REST returns
`Retry-After: 1` only for retryable 503 responses.

## Memory does not open

### The directory is in use

`reason="data_dir_in_use"` means a live `Memory` owns the physical directory. Close that owner,
call its REST interface, or choose a different directory. Do not delete `.mindbridge.lock`: the
file is only the operating-system lock target and deleting it does not release an owner.

Python, CLI, and MCP retain the local path in `subject`. REST redacts subjects for storage, index,
and internal failures. Protect MCP envelopes when using a network transport.

### The schema is unsupported

`reason="schema_unsupported"` means this MindBridge version cannot read the SQLite schema or
required tables. Reopen the backup with a compatible version. Do not edit `state.sqlite3` or
`PRAGMA user_version`.

### Store metadata mismatch

The store records embedding model, vector space and dimension, transcription space, configured
face spaces, and the index recipe. Reopen with the original configuration, apply only a supported
upgrade, or ingest source content into a new directory.

`reindex()` cannot convert incompatible embeddings, transcripts, or face analyses. A known
index-only recipe change, including quantization, rebuilds Zvec from SQLite. A recognized embedding
recipe upgrade may re-embed before publishing its new compatibility marker.

### Zvec is missing or damaged

A missing `zvec/` directory is rebuilt on open without re-embedding stored content. If Zvec is
damaged, follow the [index repair runbook](operations.md#index-maintenance-and-repair): stop the
owner, retain a backup, move only `zvec/` aside, and reopen. Never move `state.sqlite3` or `assets/`.

## A write returned an index error

MindBridge commits SQLite before applying and flushing Zvec. An `index_unavailable` response can
therefore mean the record or deletion is durable even though the call failed. The corresponding
outbox rows remain pending until startup or another draining operation succeeds.

1. Preserve the error envelope and `trace_id`.
2. Check authoritative state with `get()` or `list()` if the ID is known.
3. Correct the Zvec, disk-space, or file-descriptor failure.
4. Reopen the owner or run index maintenance from the existing owner.
5. Retry an add only with the exact same canonical input; its content-derived ID makes that retry
   idempotent.

Do not delete SQLite rows or clear `search_index_queue` to make the error disappear.

## Search returns no hits

An empty tuple is a valid search result. Candidates whose gate confidence falls below
`minimum_relevance` are rejected; an unresolved top-two tie may also be withheld when `limit=1`.

`minimum_relevance` is not compared against the `score` on a returned `SearchHit`. That score is
the final ranking score, while the gate compares a separate gate confidence, so tuning the floor
against observed scores gives the wrong value. `search_with_trace()` reports both per candidate;
use it rather than inferring the threshold from hits that were returned.

Before changing thresholds, rule out a record that was captured but never settled:
`pending_captures()` lists records that are durable and returned by `get()` and `list()` but hold
no vectors, and nothing settles them on its own; pass `memory_ids=` to ask about one record, and
read its `awaiting`, `attempts`, and `last_error` if it keeps failing. Call `settle()` —
`settle(memory_ids=...)` for that record alone, past its retry ceiling — or add the same content,
which settles it, and search again. A record whose `forgotten_at` is set is excluded
from recall by policy; `rollback()` on the logged operation restores it.

Then:

1. Confirm the record exists with `get()` or `list()`.
2. Confirm query and stored modalities are supported by the embedder.
3. Remove unintended memory-type, event-time, bitemporal, or spatial filters.
4. Run `search_with_trace()` and inspect terminal rejection reasons such as `stale_index`,
   `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`, `ambiguity`, and
   `limit`.
5. If authoritative records exist but the projection is suspect, run the
   [index repair procedure](operations.md#index-maintenance-and-repair).

SQLite hydration deliberately drops stale Zvec IDs. The defaults and ranges are in
[local memory settings](configuration.md#local-memory-settings).

## A stored media record fails to load

SQLite cannot recreate original media bytes. If `get()`, search hydration, or model work reports a
storage failure for a known media record, verify that the digest-named regular file exists under
`assets/` and restore the whole asset from a tested backup. Do not create an empty placeholder or
edit the SQLite descriptor.

## A model operation fails

### Backend is not configured

Every `Memory` needs an embedder. `ask()` also needs an answerer. `speech()` and `faces()` return
`()` when a record has no corresponding media; otherwise they need their matching capability. Add
the missing backend slot or use a configured operation. See [configuration](configuration.md).

### Optional dependency is missing

Install the narrow extra for the selected adapter: `local`, `openai`, or `face`. REST and MCP use
the `server` and `mcp` extras. See the
[extras table](configuration.md#install-only-the-surfaces-you-use).

### Modality is unsupported

Select a backend that declares the required modality. Audio may use transcript fallback only when
a configured transcription backend supports it. MindBridge does not silently discard unsupported
image or video evidence.

### Provider request or output fails

Inspect `reason` and `stage`, then check credentials, base URL, proxy, timeout, rate limit, provider
status, and response shape. Provider exceptions remain available as the Python `__cause__`, but
REST and MCP never serialize their bodies.

For `code="model_output_truncated"`, raise the adapter's generation limit or reduce the `ask()`
evidence limit. Provider-specific controls are in the
[Python adapter reference](api/python-sdk.md#bundled-adapters).

## Content is rejected or not fetched

MindBridge never fetches HTTP(S) content. In Python, a URL-shaped string is text. Download media
with the application's HTTP client, then pass a `Blob` or regular local `Path`. REST and MCP reject
remote URLs and server filesystem paths; use inline bytes or an existing asset ID.

REST rejects invalid content with 422 and request bodies over its limit with 413. Use the ordered
part shapes and limits in the [REST reference](api/rest.md#input-limits).

## REST or MCP cannot share the store

Separate Python, REST, and MCP processes cannot open the same `data_dir`. Put the required
adapters around one constructed `Memory`, call the running REST owner, or allocate deliberately
separate memory domains. REST has twelve product routes under `/v1`, or eighteen with both
opt-in switches enabled; MCP has fifteen tools.

Neither network adapter adds authentication. Apply the controls in
[deployment](deployment.md#choose-a-topology).

## Latency or telemetry looks wrong

`/healthz` is liveness, not a model or retrieval probe. Compare operation, stage, and model spans
to separate model latency from SQLite, index synchronization, search, and ranking. If token totals
look low, check `mindbridge.token_usage.complete`, expected requests, and reported requests;
MindBridge does not estimate missing usage.

For face or speaker recognition, compare `mindbridge.identity.observations`, `.matched_existing`,
and `.created`. Zero observations means the analyzer ran and detected none. Many creations with
few existing matches indicate identity fragmentation; changing a similarity threshold cannot fix
an embedding model that does not separate the deployed material.

`AsyncMemory` runs the synchronous core in worker threads. Backends must be thread-safe and own
their network concurrency and timeouts. Tune or replace the backend; do not open a second owner on
the same directory. See [telemetry](operations.md#telemetry) for the attribute contract.
