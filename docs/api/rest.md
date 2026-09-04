# REST API

## Surface

The optional FastAPI adapter exposes twelve `Memory` operations under `/v1`, or eighteen when the
host builds it with `identity_operations=True` and `embodied_operations=True`. It validates
transport input, calls the injected synchronous memory, and serializes the public SDK values; it
is not a separate storage or retrieval implementation.

The fast-capture plane -- `capture`, `settle`, and `pending_captures` -- is always on: it is an
ordinary application operation on the caller's own records, not administrative authority over a
person, so it needs no opt-in switch. Naming, reading, splitting, or erasing an identity, and
running face or speech analysis, are different: both switches default to off, mirroring
`build_mcp_server`'s own `identity_operations`, because REST is a network surface and the
[interfaces model](../context-os.md#interfaces) keeps that authority host-side unless the host
opts in. A switched-off route is never registered, so a caller gets 404 rather than 403 and cannot
discover through the error what an enabled deployment would offer:

```python
from mindbridge.api import create_app

app = create_app(memory=memory, identity_operations=True, embodied_operations=True)
```

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
for the app lifetime and close it, together with its caller-owned provider clients, during
shutdown. Run one process and one `Memory` for each physical `data_dir`; use a different directory
for another owner.

MindBridge adds no REST authentication. The host owns authentication, authorization, TLS, and
request-rate policy, through its gateway, service mesh, or FastAPI/Starlette middleware. See
[deployment](../deployment.md) for supported process shapes and [operations](../operations.md) for
shutdown and recovery.

## Contract

### Content input

The `content`, `query`, and `question` fields accept either a trimmed, non-blank string or an
ordered array of 1 through 16 strict content parts. Unknown fields are rejected.

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
| `GET /healthz` | `health` | none | `200 HealthResponse` |
| `POST /v1/memories` | `createMemory` | `MemoryCreate` | `201 MemoryResponse` |
| `POST /v1/memories/batch` | `createMemories` | `MemoryBatchCreate` | `201 {"memories":[...]}` |
| `GET /v1/memories` | `listMemories` | `limit`, `cursor` query parameters | `200 PageResponse` |
| `POST /v1/memories/search` | `searchMemories` | `QueryRequest` | `200 SearchResponse` |
| `POST /v1/memories/reinforce` | `reinforceMemories` | `ReinforceRequest` | `200 ReinforceResponse` |
| `GET /v1/memories/{memory_id}` | `getMemory` | non-empty path value | `200 MemoryResponse` |
| `DELETE /v1/memories/{memory_id}` | `deleteMemory` | non-empty path value | `200 {"deleted":bool}` |
| `POST /v1/answers` | `answer` | `AnswerRequest` | `200 AnswerResponse` |
| `POST /v1/context` | `compileContext` | `ContextRequest` | `200 ContextBundleResponse` |
| `POST /v1/capture` | `captureMemory` | `MemoryCreate` | `201 MemoryResponse` |
| `POST /v1/settle` | `settleCaptures` | `SettleRequest` | `200 {"settled":int}` |
| `GET /v1/pending_captures` | `pendingCaptures` | `limit`, `memory_ids` query parameters | `200 {"items":[PendingCaptureResponse,...]}` |

The next ten exist only when the host enables the matching switch:

| Method and path | Operation ID | Input | Success | Switch |
| --- | --- | --- | --- | --- |
| `POST /v1/speech` | `analyzeSpeech` | `AnalyzeRequest` | `200 {"segments":[...]}` | `embodied_operations` |
| `POST /v1/faces` | `analyzeFaces` | `AnalyzeRequest` | `200 {"observations":[...]}` | `embodied_operations` |
| `POST /v1/identities` | `registerIdentity` | `IdentityRegisterRequest` | `200 {"registered":true}` | `identity_operations` |
| `GET /v1/identities/{identity_id}` | `getIdentity` | non-empty path value | `200 {"identity":IdentityProfile\|null}` | `identity_operations` |
| `POST /v1/identities/{alias_id}/unlink` | `unlinkIdentity` | non-empty path value | `200 {"restored_identity_id":str\|null}` | `identity_operations` |
| `DELETE /v1/identities/{identity_id}` | `forgetIdentity` | non-empty path value | `200 {"erasure":IdentityErasure}` | `identity_operations` |
| `POST /v1/identities/{identity_id}/consent` | `recordConsent` | `ConsentRequest` | `200 {"operation":MemoryOperationResponse\|null}` | `identity_operations` |
| `GET /v1/identities/{identity_id}/consent` | `getConsent` | non-empty path value | `200 {"consent":ConsentState\|null}` | `identity_operations` |
| `GET /v1/export` | `exportSubject` | `identity_id` or `memory_ids` query parameters | `200 ExportResponse` | `identity_operations` |
| `POST /v1/retention` | `applyRetention` | `RetentionRequest` | `200 RetentionResponse` | `identity_operations` |

Request fields and defaults are:

| Request | Fields |
| --- | --- |
| `MemoryCreate` | required `content`; optional `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` |
| `MemoryBatchCreate` | `contents` with 1–100 items; optional per-item arrays `occurred_at`, `occurred_end`, `metadata`, `context`; `memory_type="semantic"` for the complete batch |
| `QueryRequest` | required `query`; `limit=10`; `explain=false`; optional `memory_type`, `reference_at`, `occurred_from`, `occurred_until`, `scope` |
| `ReinforceRequest` | required `memory_ids` with 1–100 IDs |
| `AnswerRequest` | required `question`; `limit=5`; optional `memory_type`, `reference_at`, `scope` |
| `ConsentRequest` | required `state` (`granted`, `withheld`, or `withdrawn`); optional `note` |
| `RetentionRequest` | `dry_run=false` |
| `ContextRequest` | required `goal`; optional `budget`, `reference_at`, `scope` |
| `ContextBudgetRequest` | `max_chars=16000`; `max_items=24`; `min_confidence=0.0`; optional `max_media_items` (`0` for a text-only bundle); optional `memory_types` with at least one value; optional `freshness_seconds`; optional `max_latency_ms` |
| `SettleRequest` | `limit=100`; `max_attempts=3`; optional `memory_ids` with 1–100 IDs |
| `AnalyzeRequest` | required `memory_id` |
| `IdentityRegisterRequest` | required `identity_id`, `name`; optional `relationship` |
| List query | `limit=100`; optional opaque `cursor` |

All timestamps must include a timezone. An event end requires a start and must be later than it.
If a batch supplies a per-item array, it must contain exactly one value per content. Search event
bounds are a half-open overlap filter; two bounds require `occurred_until > occurred_from`, and
records without `occurred_at` do not match. Pass `next_cursor` back unchanged to continue listing.
Time and role behavior is defined in
[memory types, time, and decay](../memory-types-time-and-decay.md).

An input `context` is an optional typed observation. Its `place_id` is a trimmed symbolic place
label independent of its metric `spatial` pose. `scope` is an optional retrieval filter:
`valid_at` and `known_at` are timezone-aware world-time and transaction-time instants; `place_id`
matches the stored label exactly; and `near` with a non-negative `radius_m` restricts results to
the same coordinate frame and observer/subject anchor. SQLite authoritatively reapplies every
scope filter after candidate retrieval.

Create-request context:

```json
{
  "content": "The mug is on the kitchen table.",
  "context": {
    "basis": "observation",
    "source_id": "camera-1:frame-42",
    "confidence": 0.94,
    "valid_from": "2026-08-27T09:00:00Z",
    "place_id": "kitchen",
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
    "place_id": "kitchen",
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
    "freshness_seconds": 2592000,
    "max_latency_ms": 250
  }
}
```

Creation is content-addressed and idempotent. Batch results preserve input order. Deletion is also
idempotent: `deleted` reports whether a record existed. `ask` requires an answerer configured in
the injected memory and, with the default `reinforce_on_answer=True`, reinforces the hits it cites.
Explicit reinforcement is not idempotent: every call raises `access_count` for the named memories
and moves the ranker's reinforcement factor, so a lost response must not be retried blindly.
Unknown IDs are skipped, and `reinforced` counts the ones that existed.

`answer` may run face recognition on a retrieved photo or video to identify who appears in it, the
same as `analyzeFaces`, and that recognition can corroborate and commit a cross-modal identity
merge. `POST /v1/answers` passes `Memory.ask(..., link_identities=embodied_operations)`, so a
caller with recall access alone cannot acquire that merge authority through an answer unless the
host has also opted `create_app` into `embodied_operations`: with the switch off, an answer may
still identify who appears in the evidence, but the bind is never committed.

`compileContext` selects and structures existing evidence, reports conflicts without resolving
them, calls no generation model, and stores no memory. It is not a pure read: it runs the same
retrieval path `searchMemories` runs and can make the same one write, caching a transcript for
spoken query media, which is a cache of the caller's own input rather than a new memory. A
compilation that finds nothing does record that, as the bounded signal the control plane's
`QUERY_FAILURE` trigger reads, which changes no evidence and no memory. Its request `budget` is
the transport form of `ContextBudget`, with the `freshness` timedelta expressed as
`freshness_seconds` and `max_latency_ms` a deadline the compiler checks between stages rather than
a timeout that aborts. The bundle reports `elapsed_ms`, `deadline_exceeded`, and an `unknowns`
array naming what the request implied and the bundle does not carry. The
[compiler reference](../context-compilation.md) owns section, selection, unknown, and conflict
semantics.

`captureMemory` commits without calling any model; the returned `MemoryResponse` is the same
content-addressed record `createMemory` would return for identical input. It is durable and
readable through `getMemory` immediately and invisible to `searchMemories` until settled.
`settleCaptures` enriches, embeds, indexes, and forms up to `limit` queued records in enqueue
order and returns how many settled; a record that has already failed `max_attempts` times is
skipped rather than retried, so one poisoned capture cannot block the queue, and naming it in
`memory_ids` bypasses that ceiling. `pendingCaptures` answers per-record readiness: an ID absent
from `items` is not pending, and `getMemory` tells whether it is settled or was never stored.

The identity and embodied routes follow the SDK operation they dispatch to, exactly as the
[MCP tools](mcp.md#tools) do, because both adapters call the same injected `Memory`.
`analyzeSpeech` and `analyzeFaces` read one stored memory's assets and return an empty result
rather than failing when no matching asset exists; both fail with `model_error` when the
required backend is unavailable. `registerIdentity` asserts a name on the host's behalf as a
versioned, reversible memory record and fails with `identity_not_found` for an unregistered ID;
omitting `relationship` leaves any recorded relationship intact, and there is deliberately no way
to clear one. `getIdentity` follows merge aliases and never fails for an unknown ID, returning
`identity: null` instead. `unlinkIdentity` reverses one face-and-voice merge and returns
`restored_identity_id: null`, changing nothing, when the merge cannot be reversed. `forgetIdentity`
erases a person's biometric templates, aliases, and indexed name -- memories and media survive --
and cannot be undone; a second call reports `identity_not_found` because the person is already
gone.

### Retrieval trace

`POST /v1/memories/search` with `"explain": true` routes to `Memory.search_with_trace` and returns a
`trace` object beside the unchanged `hits`. `trace.candidates` lists every candidate considered with
its effective score components (`dense_relevance`, `dense_confidence`, `lexical_relevance`,
`lexical_rerank_bonus`, `lexical_match`, `gate_relevance`, `base_relevance`,
`reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `rank`) and, when it
did not become a hit, a `rejected_by` value of `stale_index`, `occurrence_range`, `missing_memory`,
`memory_type`, `minimum_relevance`, `ambiguity`, or `limit`. `trace.candidate_limit` is how many
candidates were fetched, `trace.exhaustive` says whether that bound was reached, and
`trace.ambiguous` says whether the result was suppressed for being too close to call. Without
`explain`, `trace` is `null` and no extra work is done. A trace names candidates and scores only; it
never carries evidence content.

`SearchHitResponse.score` is the final ranking score, while the relevance gate compares
`gate_relevance`, a different quantity, so a floor tuned against the returned `score` compares the
wrong two numbers. `minimum_relevance` and `ambiguity_margin` are fixed when the owner constructs
`Memory` and no request field can widen them for one call; an empty result is answered by reading
`trace` and changing the query, the filters, or the owner's configuration.

### Response objects

| Object | Fields |
| --- | --- |
| `AssetResponse` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResponse` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context`, `place_id`, `forgotten_at` |
| `SearchHitResponse` | all memory fields plus `score` from 0 through 1 |
| `SearchResponse` | `hits`, `trace`; `trace` is `null` unless the request set `explain=true` |
| `ReinforceResponse` | `reinforced` |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResponse` | `items`, `next_cursor` |
| `SettleResponse` | `settled` |
| `PendingCaptureResponse` | `memory_id`, `enqueued_at`, `attempts`, `last_error`, `awaiting` (`"enrichment"` or `"formation"`) |
| `PendingCapturesResponse` | `items` |
| `SpeechResponse` | `segments`: `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `FacesResponse` | `observations`: `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `IdentityResponse` | `identity`: `identity_id`, `name`, `relationship`, `confirmed`, `evidence_ids`, or `null` |
| `RegisterResponse` | `registered` |
| `UnlinkResponse` | `restored_identity_id` |
| `ForgetResponse` | `erasure`: `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`, `face_observations`, `speech_segments` |
| `ConsentResponse` | `consent`: `granted`, `withheld`, `withdrawn`, or `null` when nobody has recorded a statement |
| `MemoryOperationResponse` | `operation_id`, `intent`, `trigger`, `evidence_ids`, `target_ids`, `claim`, `consent`, `identity`, `rationale`, `model_id`, `recipe`, `created_ids`, `changed_ids`, `forgotten_ids`, `superseded`, `applied_at`, `rolled_back_at`, `outcome`, `outcome_note` |
| `RecordConsentResponse` | `operation`, or `null` when the same statement already stands |
| `ExportResponse` | `exported_at`, `identity_id`, `identities`, `records`, `operations` |
| `RetentionResponse` | `dry_run`, `media_memory_ids`, `forgotten_memory_ids`, `asset_ids`, `capture_memory_ids`, `deleted` |
| `ContextBudgetResponse` | `max_chars`, `max_items`, `max_media_items` or `null`, `memory_types` or `null`, `min_confidence`, `freshness_seconds`, `max_latency_ms` |
| `ContextConflictResponse` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextUnknownResponse` | `kind`, `detail` |
| `ContextBundleResponse` | `goal`, `reference_at`, `budget`, the hit arrays `relationships`, `scene`, `episodes`, `facts`, `procedures`, `affect`, `traits`, the mixed `actors` array of hits, `NamedActorResponse`, and `ProvisionalActorResponse` objects, plus `conflicts`, `unknowns`, `occurred_from`, `occurred_until`, `frames`, `places`, `omitted`, `chars`, `elapsed_ms`, `deadline_exceeded`, `rendered` |
| `ProvisionalActorResponse` | `identity_id`, `memory_ids`: one recognized person in the included evidence whom no visible naming assertion names |
| `NamedActorResponse` | `identity_id`, `name`, `memory_ids`, `naming_assertion_id`: one person a naming assertion names, reached through other evidence |
| `CapabilitiesResponse` | `embedding`, `embedding_model`, `embedding_space`, `embedding_dimension`, `generation`, `transcription`, `vision`, `face`, `formation`, `generation_model`, `transcription_space`, `vision_model`, `face_model`, `formation_model`, `consolidation_model`, `speaker_recognition`, `streaming_generation`, `operations` |
| `HealthResponse` | `status`, `capabilities` |

`modality` is `text`, `image`, `video`, `audio`, or `omni`. `memory_type` is `semantic`,
`episodic`, or `procedural`. `abstention_reason` is `no_evidence`, `insufficient_evidence`, or
`null`. `abstained` reports that the answerer emitted the reserved `[insufficient_evidence]`
token, or that grounding found no usable evidence at all; a model that declines in its own words
some other way is an ordinary answer. See
[the SDK contract](python-sdk.md#public-values) for the exact rule. A response `context` is
the authoritative `MemoryContext`: typed kind and basis, confidence, valid and transaction time,
visibility, lineage/source/evidence/supersession IDs, model recipe, optional
subject/predicate/value, spatial pose, and affect cue fields. It is `null` on a raw record formed
without typed context. `forgotten_at` is set on a cognitively forgotten record, which `get`
still returns and retrieval skips. Asset filesystem paths are never serialized, in bundle sections
as in hits, and `rendered` is the deterministic text of `ContextBundle.render()`.

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
    "consolidation_model": null,
    "speaker_recognition": true,
    "streaming_generation": false,
    "operations": ["ask", "speech", "transcribe"]
  }
}
```

The six modality lists are the declarations routing reads, sorted for a stable document; an empty
list means the backend is absent, not that it supports nothing. A `null` model ID means the same.
`embedding_space` is the value that decides whether stored vectors and a new backend belong to the
same space. `speaker_recognition` is not derivable from `transcription`: a transcription backend
and a speech backend occupy one slot and declare the same modalities, but only the second resolves
speakers, so this is the field that says whether `speech` will work. `operations` is derived from
those backends rather than declared, and names which optional operations this composition can
serve, so a caller does not have to know that `ask` needs generation or `consolidate` a
consolidation backend. Values are captured when `Memory` is constructed, so the route performs no
I/O and no model call.

This is `MemoryCapabilities.document()` verbatim. The MCP server embeds the same document in its
instructions and `mindbridge doctor` prints it under `capabilities`, so no surface can describe
one composition differently.

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
| `ask_stream` | Send `POST /v1/answers`, which returns the same grounded answer once it is complete |
| `search_with_trace` | Send `POST /v1/memories/search` with `"explain": true` |
| `register_speaker` | No route; has an [MCP tool](mcp.md#tools). REST names an identity, not a voice-only speaker |
| `reindex`, `optimize` | Index maintenance an operator schedules |
| `consolidation_candidates`, `consolidate`, `deliberate`, `apply`, `record_outcome`, `forget`, `rollback`, `operations` | No route and no MCP tool; the memory control plane stays under host authority |

`exportSubject` is the one place a control-plane value reaches a transport: an export carries the
operation-log rows that moved the subject's records, because the right of access covers how the
data was processed and not only what is held. It is read-only and gated on `identity_operations`,
so nothing about the control plane's authority changes -- no route proposes, applies, or reverses
an operation.

A record containing two recognized people is returned for both of their exports, with the other
person's observations embedded in it, for the reasons the
[SDK contract](python-sdk.md#data-subject-rights) states. Over a network that matters more than it
does in-process: a host serving this route decides who receives the response, and the response
may contain a second person's data.

`applyRetention` deletes irrecoverably. It is behind `identity_operations` rather than always on
like `deleteMemory`, because it acts on a declared policy over the whole store rather than on one
record the caller named. Send `{"dry_run": true}` first: it reports the same identifiers and
deletes nothing.

`speech`, `faces`, `register_identity`, `identity`, `unlink_identity`, `forget_identity`,
`record_consent`, `consent`, `export`, and `apply_retention` do
have routes, listed in [Endpoints](#endpoints) -- but only when the host enables
`embodied_operations` or `identity_operations`. A host that leaves a switch off gets no route for
the operations it gates, exactly as the table above describes for the operations no switch can
turn on.

Use the [Python SDK](python-sdk.md) in the owning process, or the MCP adapter where the table
names a tool. Only `ask_stream` needs more than a route: incremental delivery needs a streaming
response, so exposing it means choosing a wire format rather than binding an existing one. The
rest are not REST limitations: the adapter runs in the process that owns `Memory`, so a route is
unwritten work rather than an impossibility.

### Input limits

| Bound | REST value |
| --- | --- |
| Complete `/v1` request body | 8 MiB before JSON parsing |
| Content parts | 1 through 16 |
| One URL source string | 8,192 characters |
| Normalized text, including combined text parts | 65,536 characters |
| Batch contents | 1 through 100 |
| Search, answer, or page `limit` | 1 through 100 |
| Context `budget.max_chars` | 1 through 65,536 |
| Context `budget.max_items` | 1 through 100 |
| Context `budget.max_media_items` | 0 or more, or `null` |
| Context `budget.max_latency_ms` | 1 or greater |
| Serialized metadata for one memory | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |
| `settleCaptures` or `pendingCaptures` `memory_ids` | 1 through 100 IDs |
| `settleCaptures` `max_attempts` | 1 or greater |

`file_data` is bounded by the complete HTTP body. A data URL is also bounded by the
8,192-character source field. The transport has no local-path input, remote fetch, upload endpoint,
client-streaming capture route, coordinate-frame transform, logical scope, or authentication
policy. The owner-side Python input ceiling is 512 MiB per asset, but configured backends may be
lower; the [OpenAI adapter](python-sdk.md#bundled-adapters) has smaller inline request budgets.
