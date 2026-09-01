# Troubleshooting

Start with the structured error instead of matching its message:

```python
from mindbridge import MindBridgeError

try:
    result = memory.ask("What happened yesterday?")
except MindBridgeError as error:
    print(error.code, error.reason, error.stage, error.subject, error.retryable)
```

`code` is the stable error family. `reason` narrows the cause, `stage` identifies the failed
pipeline step, `subject` identifies the affected item when available, and `retryable` says whether
the same request can succeed later. CLI, REST, and MCP envelopes return the same classification
fields and add a `trace_id`; see the [REST error contract](api/rest.md#codes-and-reasons).

For composition failures, run the same selection through `mindbridge ... doctor`: bundled recipes
exercise their loader, `--url` checks `/healthz`, and `--app` verifies the import without calling a
factory that could open the store.

## Memory does not open

### Data directory is already in use

`reason="data_dir_in_use"` means another live `Memory` or process owns the directory. Close that
owner, address it through its REST interface, or choose a different directory. Do not add a tenant
field to share one directory: physical directories are the isolation boundary.

The error is retryable because the current owner may exit. Python, CLI, and MCP errors retain the
local path in `subject`; unauthenticated REST redacts storage, index, and internal subjects. Treat
MCP envelopes as sensitive when the host uses a network transport.

### Store metadata mismatch

The directory was created with a different embedding model, vector space, dimension,
transcription space, face-analysis recipe, or index recipe. Reopen it with the original
configuration, or create a new directory and ingest the source content with the new model.

MindBridge performs only recognized bundled recipe upgrades automatically. `reindex()` cannot
convert embeddings or transcripts; it rebuilds the Zvec projection from the durable values already
in SQLite.

### Schema is unsupported

`reason="schema_unsupported"` means this MindBridge version cannot read the SQLite schema. Use a
version that supports the store. Do not edit `state.sqlite3` by hand.

### Zvec is missing or damaged

A missing `zvec/` directory is rebuilt on the next open without calling the embedding model for
stored records. If the directory is damaged, close its owner, move only `zvec/` aside, and reopen
MindBridge. Keep the moved directory until a known search succeeds. Never move `state.sqlite3` or
`assets/`; they are authoritative.

For backup and recovery procedures, see [operations](operations.md).

## Search returns no hits

An empty tuple is a valid search result. Candidates below `minimum_relevance` are rejected; an
unresolved top-two tie may also be withheld when `limit=1`.

Before lowering thresholds:

1. Confirm the query and stored content use modalities supported by the embedder.
2. Confirm the expected record exists with `get()` or `list()`.
3. Run `search_with_trace()` and inspect each candidate's terminal rejection reason.
4. Check event-time and memory-type filters.

The defaults and valid ranges are in [local memory settings](configuration.md#local-memory-settings).

## A model operation fails

### Backend is not configured

Every memory requires an embedder. `ask()` additionally requires an answerer; `speech()` and
`faces()` require their matching capabilities. Add the missing declarative slot or inject the
corresponding backend. See [configuration](configuration.md).

### Optional dependency is missing

Install the narrow extra for the selected adapter: `local`, `openai`, or `face`. REST and MCP need
their own `server` and `mcp` extras. The [extras table](configuration.md#install-only-the-surfaces-you-use)
lists each surface.

### Modality is unsupported

Adapters declare the media they accept. Select a model with that capability or configure a
transcriber for supported audio fallback. MindBridge does not silently drop unsupported image or
video evidence.

### Provider request fails

MindBridge preserves the provider exception as `__cause__` and classifies known SDK failures in
`reason`. Retry only when `error.retryable` is true. Configure credentials, base URL, timeout,
proxy, and provider retry policy on the provider SDK or declarative OpenAI slot.

If `code="model_output_truncated"`, generation reached its output-token limit. Increase the
adapter's generation limit or reduce the `ask()` evidence limit; retrying the identical request is
not a fix.

Provider-specific constructor controls and media limits are documented in the
[Python adapter reference](api/python-sdk.md#bundled-adapters) and
[REST input limits](api/rest.md#input-limits).

## Content is rejected or not fetched

MindBridge never fetches HTTP(S) content. In Python, a URL-shaped string is application text;
download media with the application's HTTP client, then pass `Blob` or a local `Path`. REST and MCP
reject network URLs and server filesystem paths.

REST media validation failures return `422`. Use the ordered content-part formats in the
[REST reference](api/rest.md); do not send a local path from the client machine.

## REST or MCP cannot share the store

The Python application, a REST process, and an MCP process cannot open the same `data_dir`
concurrently. Pass one constructed `Memory` into the transport running in that process, address the
existing REST owner, or allocate a separate directory.

The REST app and network-hosted MCP server have no MindBridge authentication. Put them behind the
access controls described in [deployment](deployment.md); stdio MCP inherits the host process
principal.

## Async operations are unexpectedly slow

`AsyncMemory` runs the synchronous embedded core in worker threads. Provider adapters must still
be thread-safe and control their own network timeouts and concurrency. Tune or replace the adapter;
do not open a second owner on the same directory.

For spans, model latency, and token accounting, see [telemetry](operations.md#telemetry).
