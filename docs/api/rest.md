# REST API

## Surface

The optional FastAPI adapter exposes nine `Memory` operations under `/v1`. It validates transport
input, calls the injected synchronous memory, and serializes the public SDK values; it is not a
separate storage or retrieval implementation.

REST accepts finalized media. Live audio packets, vision frames, partials, and scene boundaries
have no client-streaming route; run `AsyncAudioStream`, `AsyncVisionStream`, or
`AsyncCaptureStream` in the application that owns the connection.

The generated FastAPI schema is the machine-readable contract. A running application serves it at
`/openapi.json`, with Swagger UI at `/docs` and ReDoc at `/redoc`.

## Start the adapter

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

```text
create_app(*, memory: Memory) -> fastapi.FastAPI
```

With the host running, the smallest write is:

```bash
curl --fail-with-body \
  --header 'content-type: application/json' \
  --data '{"content":"The spare key is in the blue toolbox."}' \
  http://127.0.0.1:8000/v1/memories
```

## Lifecycle and ownership

`create_app` borrows `memory`; it neither opens nor closes it. The host must keep that owner alive
for the app lifetime and close it during shutdown. Run one process and one `Memory` for each
physical `data_dir`; use a different directory for another owner.

MindBridge adds no REST authentication. The host owns authentication, authorization, TLS, and
request-rate policy. See [deployment](../deployment.md) for supported process shapes and
[operations](../operations.md) for shutdown and recovery.

## Contract

### Content input

The `content`, `query`, `question`, and `goal` fields accept either a trimmed, non-blank string
or an ordered array of 1 through 16 strict content parts. Unknown fields are rejected.

| Part | Required fields | Source rule | Optional fields |
| --- | --- | --- | --- |
| `input_text` | `type`, `text` | `text` is trimmed and non-blank | none |
| `input_image` | `type` | exactly one of `image_url`, `file_id` | none |
| `input_file` | `type` | exactly one of `file_url`, `file_data`, `file_id` | `media_type`, `filename` |

```json
[
  {"type": "input_text", "text": "At the station"},
  {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
  {
    "type": "input_file",
    "file_data": "UklGRg==",
    "media_type": "audio/wav",
    "filename": "note.wav"
  }
]
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

| Method and path | Operation ID | Input | Success |
| --- | --- | --- | --- |
| `GET /healthz` | `health` | none | `200 {"status":"ok"}` |
| `POST /v1/memories` | `createMemory` | `MemoryCreate` | `201 MemoryResponse` |
| `POST /v1/memories/batch` | `createMemories` | `MemoryBatchCreate` | `201 {"memories":[...]}` |
| `GET /v1/memories` | `listMemories` | `limit`, `cursor` query parameters | `200 PageResponse` |
| `POST /v1/memories/search` | `searchMemories` | `QueryRequest` | `200 {"hits":[...]}` |
| `GET /v1/memories/{memory_id}` | `getMemory` | non-empty path value | `200 MemoryResponse` |
| `DELETE /v1/memories/{memory_id}` | `deleteMemory` | non-empty path value | `200 {"deleted":bool}` |
| `POST /v1/answers` | `answer` | `AnswerRequest` | `200 AnswerResponse` |
| `POST /v1/context` | `compileContext` | `ContextRequest` | `200 ContextBundleResponse` |
| `GET /v1/capabilities` | `capabilities` | none | `200 CapabilitiesResponse` |

Request fields and defaults are:

| Request | Fields |
| --- | --- |
| `MemoryCreate` | required `content`; optional `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` |
| `MemoryBatchCreate` | `contents` with 1–100 items; optional per-item arrays `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` for the complete batch |
| `QueryRequest` | required `query`; `limit=10`; optional `memory_type`, `reference_at`, `occurred_from`, `occurred_until`, `scope` |
| `AnswerRequest` | required `question`; `limit=5`; optional `memory_type`, `reference_at`, `scope` |
| `ContextRequest` | required `goal`; optional `budget`, `reference_at`, `scope` |
| `ContextBudgetRequest` | `max_chars=6000`; `max_items=24`; `min_confidence=0.0`; optional `memory_types` with at least one value; optional `freshness_seconds` |
| List query | `limit=100`; optional opaque `cursor` |

All timestamps must include a timezone. An event end requires a start and must be later than it.
If a batch supplies a per-item array, it must contain exactly one value per content. Search event
bounds are a half-open overlap filter; two bounds require `occurred_until > occurred_from`, and
records without `occurred_at` do not match. Pass `next_cursor` back unchanged to continue listing.
Time and role behavior is defined in
[memory types, time, and decay](../memory-types-time-and-decay.md).

An input `context` is an optional typed observation. `scope` is an optional retrieval filter:
`valid_at` and `known_at` are timezone-aware world-time and transaction-time instants, while
`near` and a non-negative `radius_m` must appear together and restrict results to the same
coordinate frame and observer/subject anchor. SQLite authoritatively reapplies every scope filter
after candidate retrieval.

Create-request context:

```json
{
  "content": "The mug is on the kitchen table.",
  "context": {
    "basis": "observation",
    "source_id": "camera-1:frame-42",
    "confidence": 0.94,
    "valid_from": "2026-08-27T09:00:00Z",
    "spatial": {
      "frame_id": "home/map",
      "anchor": "subject",
      "x": 2.0,
      "y": 1.0,
      "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
      "position_uncertainty_m": 0.08
    }
  }
}
```

Search-request scope:

```json
{
  "query": "Where was the mug?",
  "scope": {
    "valid_at": "2026-08-27T10:00:00Z",
    "known_at": "2026-08-27T12:00:00Z",
    "near": {"frame_id": "home/map", "anchor": "subject", "x": 2.0, "y": 1.0},
    "radius_m": 0.75
  }
}
```

Context-request budget:

```json
{
  "goal": "What should I bring to the workshop?",
  "budget": {
    "max_chars": 2000,
    "max_items": 8,
    "memory_types": ["semantic", "episodic"],
    "min_confidence": 0.5,
    "freshness_seconds": 2592000
  }
}
```

Creation is content-addressed and idempotent. Batch results preserve input order. Deletion is also
idempotent: `deleted` reports whether a record existed. `ask` requires an answerer configured in
the injected memory.

`compileContext` is a read-only view: it selects and structures existing evidence, reports
conflicts without resolving them, calls no generation model, and writes nothing. Its request
`budget` is the transport form of `ContextBudget`, with the `freshness` timedelta expressed as
`freshness_seconds`. `max_chars` accepts 1 through 65,536 and `max_items` 1 through 100; the
[compiler reference](../context-compilation.md) owns section, selection, and conflict semantics.
`capabilities` reports the injected memory's configured composition and calls no model, so an
agent can read what an instance supports instead of discovering a missing backend by failing.

### Response objects

| Object | Fields |
| --- | --- |
| `AssetResponse` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResponse` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context`, `forgotten_at` |
| `SearchHitResponse` | all memory fields plus `score` from 0 through 1 |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `ContextBudgetResponse` | `max_chars`, `max_items`, `memory_types` sorted or `null`, `min_confidence`, `freshness_seconds` |
| `ContextConflictResponse` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextBundleResponse` | `goal`, `reference_at`, `budget`, the hit arrays `actors`, `episodes`, `facts`, `procedures`, `affect`, `traits`, plus `conflicts`, `occurred_from`, `occurred_until`, `frames`, `omitted`, `chars`, `rendered` |
| `CapabilitiesResponse` | `modalities` sorted, `answer`, `transcribe`, `faces`, `describe_vision`, `form`, `consolidate`, `decay` |
| `PageResponse` | `items`, `next_cursor` |

`modality` is `text`, `image`, `video`, `audio`, or `omni`. `memory_type` is `semantic`,
`episodic`, or `procedural`. `abstention_reason` is `no_evidence`, `insufficient_evidence`, or
`null`. `abstained` reports that the answerer returned the exact sentence reserved for having no
usable evidence, not that the model declined to answer in its own words. A response `context` is
the authoritative `MemoryContext`: typed kind and basis, confidence, valid and transaction time,
visibility, lineage/source/evidence/supersession IDs, model recipe, optional
subject/predicate/value, spatial pose, and affect cue fields. It is `null` on a raw record formed
without typed context. Asset filesystem paths are never serialized, in bundle sections as
elsewhere. Every bundle section is an array of `SearchHitResponse` values, and `rendered` is the
deterministic text of `ContextBundle.render()`.

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
| `model_error` | unset, `backend_not_configured`, `unsupported_modality`, `model_failed`, `auth_failed`, `rate_limited`, `quota_exhausted`, `timeout`, `connection_failed`, `request_rejected`, `response_invalid`, `payload_too_large`, `asset_unavailable`, `asset_changed` |
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
| `identity`, `unlink_identity` | Owner-process identity inspection and merge reversal |
| `reinforce` | Owner-process feedback |
| `reindex`, `optimize` | Owner-process index maintenance |

Use the [Python SDK](python-sdk.md) in the owning process for those operations.

### Input limits

| Bound | REST value |
| --- | --- |
| Complete `/v1` request body | 8 MiB before JSON parsing |
| Context `budget.max_chars` | 1 through 65,536 |
| Content parts | 1 through 16 |
| One URL source string | 8,192 characters |
| Normalized text, including combined text parts | 65,536 characters |
| Batch contents | 1 through 100 |
| Search, answer, or page `limit`, and `budget.max_items` | 1 through 100 |
| Serialized metadata for one memory | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |

`file_data` is bounded by the complete HTTP body. A data URL is also bounded by the 8,192-character
source field. The transport has no local-path input, remote fetch, upload endpoint, client-streaming capture
route, coordinate-frame transform, logical scope, or authentication policy. The owner-side Python input ceiling is 512 MiB per asset, but configured
backends may be lower; the [OpenAI adapter](python-sdk.md#bundled-adapters) has smaller inline
request budgets.
