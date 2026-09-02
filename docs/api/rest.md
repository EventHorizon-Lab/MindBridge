# REST API

## Purpose

The optional FastAPI adapter exposes eight `Memory` operations under `/v1`. It validates transport
input, calls the injected synchronous memory, and serializes the public SDK values; it is not a
separate storage or retrieval implementation.

REST accepts finalized media. Live audio packets, vision frames, partials, and scene boundaries
have no client-streaming route; run `AsyncAudioStream`, `AsyncVisionStream`, or
`AsyncCaptureStream` in the application that owns the connection.

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
| `GET /healthz` | none | `200 HealthResponse` |
| `POST /v1/memories` | `MemoryCreate` | `201 MemoryResponse` |
| `POST /v1/memories/batch` | `MemoryBatchCreate` | `201 {"memories":[...]}` |
| `GET /v1/memories` | `limit`, `cursor` query parameters | `200 PageResponse` |
| `POST /v1/memories/search` | `QueryRequest` | `200 {"hits":[...],"trace":null}` |
| `POST /v1/memories/reinforce` | `ReinforceRequest` | `200 {"reinforced":int}` |
| `GET /v1/memories/{memory_id}` | non-empty path value | `200 MemoryResponse` |
| `DELETE /v1/memories/{memory_id}` | non-empty path value | `200 {"deleted":bool}` |
| `POST /v1/answers` | `AnswerRequest` | `200 AnswerResponse` |

Request fields and defaults are:

| Request | Fields |
| --- | --- |
| `MemoryCreate` | required `content`; optional `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` |
| `MemoryBatchCreate` | `contents` with 1–100 items; optional per-item arrays `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` for the complete batch |
| `QueryRequest` | required `query`; `limit=10`; `explain=false`; optional `memory_type`, `reference_at`, `occurred_from`, `occurred_until`, `scope` |
| `ReinforceRequest` | required `memory_ids` with 1–100 IDs |
| `AnswerRequest` | required `question`; `limit=5`; optional `memory_type`, `reference_at`, `scope` |
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

```json
{
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
  },
  "scope": {
    "valid_at": "2026-08-26T12:00:00Z",
    "known_at": "2026-08-27T12:00:00Z",
    "near": {"frame_id": "home/map", "anchor": "subject", "x": 2.0, "y": 1.0},
    "radius_m": 0.75
  }
}
```

Creation is content-addressed and idempotent. Batch results preserve input order. Deletion is also
idempotent: `deleted` reports whether a record existed. `ask` requires an answerer configured in
the injected memory. Reinforcement is not idempotent: every call raises `access_count` for the named
memories and moves the ranker's reinforcement factor, so a lost response must not be retried
blindly. Unknown IDs are skipped, and `reinforced` counts the ones that existed.

### Retrieval trace

`POST /v1/memories/search` with `"explain": true` routes to `Memory.search_with_trace` and returns a
`trace` object beside the unchanged `hits`. `trace.candidates` lists every candidate considered with
its effective score components (`dense_relevance`, `dense_confidence`, `lexical_relevance`,
`lexical_rerank_bonus`, `lexical_match`, `gate_confidence`, `base_relevance`,
`reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `rank`) and, when it
did not become a hit, a `rejected_by` value of `stale_index`, `occurrence_range`, `missing_memory`,
`memory_type`, `minimum_relevance`, `ambiguity`, or `limit`. `trace.candidate_limit` is how many
candidates were fetched, `trace.exhaustive` says whether that bound was reached, and
`trace.ambiguous` says whether the result was suppressed for being too close to call. Without
`explain`, `trace` is `null` and no extra work is done. A trace names candidates and scores only; it
never carries evidence content.

`SearchHitResponse.score` is the final ranking score, while the relevance gate compares
`gate_confidence`, a different quantity, so a floor tuned against the returned `score` compares the
wrong two numbers. `minimum_relevance` and `ambiguity_margin` are fixed when the owner constructs
`Memory` and no request field can widen them for one call; an empty result is answered by reading
`trace` and changing the query, the filters, or the owner's configuration.

### Response objects

| Object | Fields |
| --- | --- |
| `AssetResponse` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResponse` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context` |
| `SearchHitResponse` | all memory fields plus `score` from 0 through 1 |
| `SearchResponse` | `hits`, and `trace` when the request set `explain` |
| `ReinforceResponse` | `reinforced` |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResponse` | `items`, `next_cursor` |
| `CapabilitiesResponse` | `embedding`, `embedding_model`, `embedding_space`, `embedding_dimension`, `generation`, `transcription`, `vision`, `face`, `formation`, `generation_model`, `transcription_space`, `vision_model`, `face_model`, `formation_model`, `speaker_recognition`, `streaming_generation` |
| `HealthResponse` | `status`, `capabilities` |

`modality` is `text`, `image`, `video`, `audio`, or `omni`. `memory_type` is `semantic`,
`episodic`, or `procedural`. `abstention_reason` is `no_evidence`, `insufficient_evidence`, or
`null`. `abstained` reports that the answerer returned the exact sentence reserved for having no
usable evidence, not that the model declined to answer in its own words. A response `context` is
the authoritative `MemoryContext`: typed kind and basis, confidence, valid and transaction time,
visibility, lineage/source/evidence/supersession IDs, model recipe, optional
subject/predicate/value, spatial pose, and affect cue fields. It is `null` on a raw record formed
without typed context. Asset filesystem paths are never serialized.

`/healthz` reports liveness and the composition behind the process, so an operator does not have
to send a probe write to learn what the deployment can do:

```json
{
  "status": "ok",
  "capabilities": {
    "embedding": ["audio", "image", "text", "video"],
    "embedding_model": "jina-v5-omni",
    "embedding_space": "jina-v5-omni:1024",
    "embedding_dimension": 1024,
    "generation": ["text"],
    "transcription": ["audio"],
    "vision": [],
    "face": [],
    "formation": [],
    "generation_model": "qwen3-omni",
    "transcription_space": "funasr-nano:cam++",
    "vision_model": null,
    "face_model": null,
    "formation_model": null,
    "speaker_recognition": true,
    "streaming_generation": false
  }
}
```

The six modality lists are the declarations routing reads, sorted for a stable document; an empty
list means the backend is absent, not that it supports nothing. A `null` model ID means the same.
`embedding_space` is the value that decides whether stored vectors and a new backend belong to the
same space. `speaker_recognition` is not derivable from `transcription`: a transcription backend
and a speech backend occupy one slot and declare the same modalities, but only the second resolves
speakers, so this is the field that says whether `speech` will work. Values are captured when
`Memory` is constructed, so the route performs no I/O and no model call.

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
| `model_error` | unset, `backend_not_configured`, `unsupported_modality`, `auth_failed`, `rate_limited`, `quota_exhausted`, `timeout`, `connection_failed`, `request_rejected`, `response_invalid`, `payload_too_large`, `asset_unavailable`, `asset_changed`, `model_failed` |
| `model_output_truncated` | `output_truncated` |
| `storage_error` | unset, `data_dir_in_use`, `schema_unsupported`, `io_failed`, `flush_failed`, `instance_unusable` |
| `index_unavailable` | unset, `index_missing` |
| `mindbridge_error` | unset |
| `internal_error` | `unexpected` |
| `not_found`, `method_not_allowed`, `http_error` | unset |

`retryable` is true only for `connection_failed`, `data_dir_in_use`, `flush_failed`,
`index_missing`, `rate_limited`, and `timeout`. A retryable 503 response includes `Retry-After: 1`.

The status is a function of `reason` alone, read from one table in `mindbridge.api.errors`. Which
exception class carried the failure, and which raise site produced it, do not change the answer:
one condition has one status everywhere, and a reason with no row falls back to a coarse status
for its `code`. Every 503 is a reason in `RETRYABLE_REASONS` and every retryable reason is a 503,
in both directions, so a client can act on the status without also reading the reason.

| Status | `reason` |
| --- | --- |
| 404 | unknown route; `memory_not_found`, `speaker_not_found`, `identity_not_found` |
| 405 | method not allowed |
| 413 | `payload_too_large`, whether the `/v1` request body exceeded 8 MiB or a configured backend rejected one asset as too large |
| 422 | `input_invalid`, `unsupported_modality` |
| 500 | `unexpected`, `schema_unsupported`, `io_failed`, `instance_unusable`, or a generic `MindBridgeError` with no reason |
| 501 | `backend_not_configured` |
| 502 | `auth_failed`, `quota_exhausted`, `request_rejected`, `response_invalid`, `output_truncated`, `asset_unavailable`, `asset_changed`, `model_failed`, or a `model_error` with no reason |
| 503 | `connection_failed`, `timeout`, `rate_limited`, `data_dir_in_use`, `flush_failed`, `index_missing`, or a `storage_error` with no reason |

Two rows are worth stating explicitly, because both used to answer twice. `payload_too_large` is
one condition seen from two sides and both are fixed by sending less, so the provider path no
longer reports 502. `io_failed` is the coarse label the storage wrapper puts on a failure it
cannot classify, programming errors included, and it is deliberately not retryable, so it reports
500 rather than telling a client the condition is transient.

### Operations without a route

REST has no route for these Python operations:

| Operation | Boundary |
| --- | --- |
| `add_stream` | Send each completed observation to `POST /v1/memories` |
| `search_with_trace` | Send `POST /v1/memories/search` with `"explain": true` |
| `speech`, `faces` | No route; both have an [MCP tool](mcp.md#tools) |
| `register_speaker`, `register_identity` | No route; both have an MCP tool |
| `identity`, `unlink_identity`, `forget_identity` | No route; all three have an MCP tool |
| `reindex`, `optimize` | Index maintenance an operator schedules |

Use the [Python SDK](python-sdk.md) in the owning process, or the MCP adapter where the table
names a tool. None of these is a REST limitation: the adapter runs in the process that owns
`Memory`, so a route is unwritten work rather than an impossibility.

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
source field. The transport has no local-path input, remote fetch, upload endpoint, client-streaming capture
route, coordinate-frame transform, logical scope, or authentication policy. The owner-side Python input ceiling is 512 MiB per asset, but configured
backends may be lower; the [OpenAI adapter](python-sdk.md#bundled-adapters) has smaller inline
request budgets.
