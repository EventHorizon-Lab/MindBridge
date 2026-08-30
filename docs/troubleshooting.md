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

## Reading a MindBridge error

Every exception deriving from `MindBridgeError` carries four attributes beside its stable `code`:

```python
try:
    memory.ask("what happened yesterday?")
except MindBridgeError as error:
    error.code  # stable outer taxonomy, for example "model_error"
    error.reason  # closed sub-vocabulary, for example "rate_limited"; may be None
    error.stage  # which pipeline stage failed, for example "generate"; may be None
    error.subject  # asset ID, memory ID, or batch position; may be None
    error.retryable  # bool derived from reason, never a guess
```

All four are optional and default to an unclassified, non-retryable failure, so existing handlers
keep working. REST and MCP serialize the same five values; see
[the REST error contract](api/rest.md#codes-and-reasons) for the vocabularies.

## Provider request failed

MindBridge returns a `ModelError` whose message is an author-written literal, never the provider's
response. The provider's own exception survives as `__cause__`, so a local traceback still shows the
root cause, and `reason` classifies it from the official SDK's exception classes:

| `reason` | Provider condition | Retry |
| --- | --- | --- |
| `auth_failed` | `openai.AuthenticationError` | Never; fix the credential |
| `rate_limited` | `openai.RateLimitError` | Yes, after a delay |
| `timeout` | `openai.APITimeoutError` | Yes |
| `connection_failed` | `openai.APIConnectionError` | Yes |
| `request_rejected` | `openai.BadRequestError` | Never; the request itself is wrong |
| unset | Anything else | Treated as permanent |

Check `error.retryable` instead of inspecting the message. It is `False` unless `reason` is one of
the transient values above, because failing to retry a transient error costs one call while retrying
a permanent one never terminates. The provider SDK still owns its own retry and network behavior.

## The operation says which stage failed

Every public error carries `stage`: `open`, `content.prepare`, `embed`, `transcribe`, `generate`,
`storage.write`, `storage.hydrate`, `storage.lookup`, `index.search`, `index.sync`, `retrieval.rank`,
or `close`. These are the same names the OpenTelemetry spans use, so a failure and its trace agree.
`stage` is `null` when MindBridge cannot attribute the failure to one stage.

## A batch failed and I do not know which item

`add_many` writes in one transaction, so one bad item fails the whole batch. The raised error's
`subject` names the position, for example `contents[7]`. Fix that item and resubmit the batch.

## Answering returns 501 rather than 502

`ask` without an `answerer` is a permanent misconfiguration, not an upstream failure, so REST reports
`501` with `reason: "backend_not_configured"`. Construct `Memory` with a generation backend. A `502`
means a real upstream failure that will not succeed on retry; only `503` invites one.

## Remote URL is rejected

HTTP(S) URLs are not content atoms and REST/MCP do not download them. Fetch the resource with the
application's HTTP stack, then pass `Blob(data, media_type, name)` or a local `Path`.

## Data directory is already in use

Exactly one live `Memory` owns a directory. Stop the other process or select another physical
directory. Do not add a logical tenant or request field to work around the lock. This failure is
`storage_error` with `reason: "data_dir_in_use"` and is the one storage failure that is retryable;
the directory travels in `subject` rather than in the message, so an unauthenticated REST client
never sees the local path. `reason: "schema_unsupported"` is its opposite: it is permanent, reports
HTTP 500, and needs a different MindBridge version or a new directory.

## Store metadata mismatch

The directory was opened with a different embedding model, embedding space, dimension,
transcription space, or unknown index recipe. Known older retrieval-key recipes migrate by
re-embedding automatically; other mismatches require the original adapter or a new directory.
`reindex()` itself cannot change stored vectors or transcripts.

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

The OpenAI adapter bounds the media the request actually carries. Media travels base64-encoded, so
the limit is 64 MiB on the wire and roughly 48 MiB of files on disk. Answer generation keeps ranked
evidence assets that fit and falls back to their text; an oversized question asset or embedding
input still fails. `mindbridge.grounding.media_elided_hits` and
`mindbridge.grounding.dropped_hits` on the generation span count the retrieved evidence the budget
removed, so a shrunken payload is never silent.
Use a provider-specific adapter that uploads or streams large assets through that provider's SDK.
MindBridge does not expose local `file://` paths as a compatibility transport.

## Answers fail with `model_output_truncated`

Generation reached an output token limit before it finished, so MindBridge refused the partial
answer instead of returning it. This is deterministic; retrying changes nothing. Raise
`generation_max_tokens` on the adapter, or lower the `ask` limit so less evidence competes for the
model's output budget. `gen_ai.response.finish_reasons` on `mindbridge.model.generation` records
the provider's stop reason for every call, truncated or not.

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
