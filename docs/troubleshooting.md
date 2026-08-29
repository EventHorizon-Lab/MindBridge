# Troubleshooting

## `embedder` is required

`Memory` does not choose a model. Construct an `EmbeddingBackend` explicitly:

```python
memory = Memory("./data", embedder=JinaOmniEmbedder())
```

Install `mindbridge[local]` for the built-in local adapter, or supply another implementation.

## Provider client is not configured

`OpenAIModels` accepts caller-owned clients per operation. Supply the client used by the failing
operation:

```python
models = OpenAIModels(generation_client=client)
```

Configure keys, base URL, proxy, timeout, and retries on `client`, not on MindBridge.

## Provider request failed

MindBridge intentionally returns a sanitized `ModelError`. Inspect provider SDK telemetry and
logs, while avoiding memory content and credentials. The SDK owns retry and network behavior.

## Remote URL is rejected

HTTP(S) URLs are not content atoms and REST/MCP do not download them. Fetch the resource with the
application's HTTP stack, then pass `Blob(data, media_type, name)` or a local `Path`.

## Data directory is already in use

Exactly one live `Memory` owns a directory. Stop the other process or select another physical
directory. Do not add a logical tenant or request field to work around the lock.

## Store metadata mismatch

The directory was opened with a different embedding model, embedding space, dimension,
transcription space, or incompatible index recipe. Use the original adapter recipe or ingest the
source content into a new directory. `reindex()` cannot change stored vectors or transcripts.

## Missing or damaged Zvec index

Close the owner and remove only the `zvec/` subdirectory if it is damaged. Reopening the same
directory rebuilds the projection from SQLite without re-embedding content. Do not remove
`state.sqlite3` or `assets/`.

## Missing local model dependency

Install the matching optional extra:

```bash
uv add "mindbridge[local]"
```

Jina also needs its pinned remote code and media processors. FunASR is loaded only when speech
analysis is first requested.

## Unsupported modality

Adapter capabilities are explicit. Either select a model that supports the modality or configure
a transcriber for audio fallback. MindBridge does not silently remove image or video evidence.

## Media exceeds 64 MiB for a model call

The OpenAI adapter bounds aggregate inline media. Use a provider-specific adapter that uploads or
streams large assets through that provider's SDK. MindBridge does not expose local `file://` paths
as a compatibility transport.

## REST returns 422 for media

REST accepts base64 data URLs, `file_data`, or a stored `file_id`. It rejects remote URLs and local
paths. Decode failures, MIME mismatches, ambiguous sources, and unknown fields are validation
errors.

## REST has no authentication

This is the intended boundary. Add authentication middleware or deploy behind an authenticated
gateway. Do not place the bare app on an untrusted network.

## MCP cannot open the directory

The Python application, REST process, and MCP process cannot share a directory concurrently.
Either expose MCP from the existing owner or assign a separate physical memory domain.

## Async calls block unexpectedly

`AsyncMemory` moves the synchronous embedded core to worker threads. Provider adapters still need
to be thread-safe. Tune or replace the provider adapter rather than adding a second retry or async
compatibility layer inside MindBridge.
