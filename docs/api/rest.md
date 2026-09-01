# REST API

## Purpose

The optional FastAPI adapter exposes seven `Memory` operations under `/v1`. It validates transport
input, calls the injected synchronous memory, and serializes the public SDK values; it is not a
separate storage or retrieval implementation.

The generated FastAPI schema is the machine-readable contract. A running application serves it at
`/openapi.json`, with Swagger UI at `/docs` and ReDoc at `/redoc`.

## Invocation

Install the server extra together with the extras required by the chosen backends, construct one
`Memory`, and inject it:

```bash
uv add "mindbridge[local,server]"
```

```python
from mindbridge import Memory
from mindbridge.api import create_app

memory = Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
)
app = create_app(memory=memory)
```

The example uses the pinned Jina recipe; review its upstream code and license boundary in
[configuration](../configuration.md#embedding-choices).

```text
create_app(*, memory: Memory) -> fastapi.FastAPI
```

`create_app` does not own or close `memory`. The host application must close the memory and its
caller-owned provider clients at shutdown. Run exactly one owner for each physical `data_dir`.
MindBridge adds no authentication; put the app behind the deployment's gateway, service mesh, or
FastAPI/Starlette authentication middleware. See [deployment](../deployment.md).

With the host running, the smallest write is:

```bash
curl --fail-with-body \
  --header 'content-type: application/json' \
  --data '{"content":"The spare key is in the blue toolbox."}' \
  http://127.0.0.1:8000/v1/memories
```

## Contract

### Content input

The `content`, `query`, and `question` fields accept either a trimmed, non-blank string or an
ordered array of 1 through 16 strict content parts. Unknown fields are rejected.

```json
{"type":"input_text","text":"The prototype after the review"}
```

An image part supplies exactly one of `image_url` or `file_id`:

```json
{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}
```

```json
{"type":"input_image","file_id":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

A file part supplies exactly one of `file_url`, `file_data`, or `file_id`:

```json
{
  "type": "input_file",
  "file_data": "UklGRg==",
  "media_type": "audio/wav",
  "filename": "note.wav"
}
```

```json
{
  "type": "input_file",
  "file_url": "data:video/mp4;base64,AAAA",
  "media_type": "video/mp4",
  "filename": "clip.mp4"
}
```

```json
{"type":"input_file","file_id":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","media_type":"video/*"}
```

`image_url` and `file_url` accept only base64 `data:` URLs. `file_data` is raw base64 and requires
a concrete image, video, or audio MIME type. An optional MIME type must agree with the data URL.
`filename` is a safe basename of at most 255 characters. Remote URLs, local paths, `file:` URLs,
and `input_image.detail` are not accepted. Fetch remote media in the host application or use the
[Python content contract](python-sdk.md#content-contract).

`file_id` is an existing asset's 64-character lowercase SHA-256 identifier in the same
`data_dir`. Its transport field accepts at most 255 characters so malformed IDs reach the shared
SDK validator and error contract.

### Endpoints

| Method and path | Input | Success |
| --- | --- | --- |
| `GET /healthz` | none | `200 {"status":"ok"}` |
| `POST /v1/memories` | `MemoryCreate` | `201 MemoryResponse` |
| `POST /v1/memories/batch` | `MemoryBatchCreate` | `201 {"memories":[...]}` |
| `GET /v1/memories` | `limit`, `cursor` query parameters | `200 PageResponse` |
| `POST /v1/memories/search` | `QueryRequest` | `200 {"hits":[...]}` |
| `GET /v1/memories/{memory_id}` | non-empty path value | `200 MemoryResponse` |
| `DELETE /v1/memories/{memory_id}` | non-empty path value | `200 {"deleted":bool}` |
| `POST /v1/answers` | `AnswerRequest` | `200 AnswerResponse` |

Request fields and defaults are:

| Request | Fields |
| --- | --- |
| `MemoryCreate` | required `content`; optional `occurred_at`, `occurred_end`, `metadata`; `memory_type="semantic"` |
| `MemoryBatchCreate` | `contents` with 1–100 items; optional per-item arrays `occurred_at`, `occurred_end`, `metadata`; `memory_type="semantic"` for the complete batch |
| `QueryRequest` | required `query`; `limit=10`; optional `memory_type`, `reference_at`, `occurred_from`, `occurred_until` |
| `AnswerRequest` | required `question`; `limit=5`; optional `memory_type`, `reference_at` |
| List query | `limit=100`; optional opaque `cursor` |

All timestamps must include a timezone. An event end requires a start and must be later than it.
If a batch supplies a per-item array, it must contain exactly one value per content. Search event
bounds are a half-open overlap filter; two bounds require `occurred_until > occurred_from`, and
records without `occurred_at` do not match. Pass `next_cursor` back unchanged to continue listing.
Time and role behavior is defined in
[memory types, time, and decay](../memory-types-time-and-decay.md).

Creation is content-addressed and idempotent. Batch results preserve input order. Deletion is also
idempotent: `deleted` reports whether a record existed. `ask` requires an answerer configured in
the injected memory.

### Response objects

| Object | Fields |
| --- | --- |
| `AssetResponse` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResponse` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata` |
| `SearchHitResponse` | all memory fields plus `score` from 0 through 1 |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResponse` | `items`, `next_cursor` |

`modality` is `text`, `image`, `video`, `audio`, or `omni`. `memory_type` is `semantic`,
`episodic`, or `procedural`. `abstention_reason` is `no_evidence`, `insufficient_evidence`, or
`null`. `abstained` reports that the answerer returned the exact sentence reserved for having no
usable evidence, not that the model declined to answer in its own words. Asset filesystem paths are
never serialized.

## Errors and limits

### Error envelope

Every REST failure uses one flat JSON shape:

```json
{
  "code": "validation_error",
  "reason": "input_invalid",
  "retryable": false,
  "stage": null,
  "subject": null,
  "message": "request validation failed",
  "trace_id": "trace_0123456789abcdef0123456789abcdef",
  "issues": [
    {
      "location": ["body", "content"],
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

`code` is the stable outer category. `reason` narrows it, `stage` identifies the failed pipeline
stage, and `subject` identifies an input, asset, memory, or batch position. Any may be `null` when
unclassified. `issues` is populated for request-schema failures. `trace_id` correlates the response
with owner logs.

For unauthenticated REST, `subject` is withheld for `storage_error`, `index_unavailable`, and
`internal_error` because it may name local server state. Provider exception details and credentials
are never serialized.

### Codes and reasons

| `code` | `reason` values used by the current implementation |
| --- | --- |
| `validation_error` | `input_invalid` |
| `request_too_large` | `payload_too_large` |
| `memory_not_found` | `memory_not_found` |
| `speaker_not_found` | `speaker_not_found` |
| `identity_not_found` | `identity_not_found` |
| `model_error` | unset, `backend_not_configured`, `unsupported_modality`, `auth_failed`, `rate_limited`, `quota_exhausted`, `timeout`, `connection_failed`, `request_rejected`, `response_invalid`, `payload_too_large`, `asset_unavailable`, `asset_changed` |
| `model_output_truncated` | `output_truncated` |
| `storage_error` | unset, `data_dir_in_use`, `schema_unsupported`, `io_failed` |
| `index_unavailable` | unset |
| `mindbridge_error` | unset |
| `internal_error` | `unexpected` |
| `not_found`, `method_not_allowed`, `http_error` | unset |

`retryable` is true only for `connection_failed`, `data_dir_in_use`, `flush_failed`,
`index_missing`, `rate_limited`, and `timeout`. `flush_failed` and `index_missing` are reserved
retry reasons with no current raise site. A retryable 503 response includes `Retry-After: 1`.

HTTP status mapping is:

| Status | Failure |
| --- | --- |
| 404 | unknown route, memory, speaker, or identity |
| 405 | method not allowed |
| 413 | `/v1` request body exceeds 8 MiB |
| 422 | request or SDK validation; `model_error/unsupported_modality` |
| 500 | unexpected failure, generic `MindBridgeError`, or `storage_error/schema_unsupported` |
| 501 | `model_error/backend_not_configured` |
| 502 | permanent `model_error` or `model_output_truncated` |
| 503 | other storage/index failures or retryable model failures |

### Operations without a route

REST has no route for these Python operations:

| Operation | Boundary |
| --- | --- |
| `add_stream` | Send each completed observation to `POST /v1/memories` |
| `search_with_trace` | Owner-process retrieval diagnostics |
| `speech`, `faces` | Owner-process media analysis |
| `register_speaker`, `register_identity` | Owner-process identity naming |
| `reinforce` | Owner-process feedback |
| `reindex`, `optimize` | Owner-process index maintenance |

Use the [Python SDK](python-sdk.md) in the owning process for those operations.

### Input limits

| Bound | REST value |
| --- | --- |
| Complete `/v1` request body | 8 MiB before JSON parsing |
| Content parts | 1 through 16 |
| One URL source string | 8,192 characters |
| Normalized text, including combined text parts | 65,536 characters |
| Batch contents | 1 through 100 |
| Search, answer, or page `limit` | 1 through 100 |
| Serialized metadata for one memory | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |

`file_data` is bounded by the complete HTTP body. A data URL is also bounded by the 8,192-character
source field. The transport has no local-path input, remote fetch, upload endpoint, logical scope,
or authentication policy. The owner-side Python input ceiling is 512 MiB per asset, but configured
backends may be lower; the [OpenAI adapter](python-sdk.md#bundled-adapters) has smaller inline
request budgets.
