# REST API

Base path `/v1`. Every schema on this page is generated from the Pydantic contracts in
`mindbridge.contracts`, which are the same models the [Python SDK](python-sdk.md) and the
[MCP tools](mcp.md) use. A live OpenAPI document is served at `/openapi.json`, with Swagger UI
at `/docs` and ReDoc at `/redoc`.

## Authentication

Bearer token, bound to an explicit tenant allowlist:

```http
Authorization: Bearer <api-key>
```

Keys are configured through `MINDBRIDGE_TENANT_API_KEYS_JSON`, which maps each tenant ID to one
or more keys:

```json
{"tenant_01": ["at-least-32-characters-long", "second-key-during-rotation"]}
```

Two or more keys per tenant is how you rotate without downtime. Keys shorter than 32 characters
fail at startup rather than at first request, and MindBridge retains no plaintext copy — only a
digest.

Every `/v1` operation rejects a body or query `tenant_id` outside the authenticated allowlist
with `tenant_access_denied`, whether or not the row exists. Only `/healthz` is public.

## Conventions

**Time** is RFC 3339 with an explicit offset. Naive timestamps are rejected at the boundary, not
coerced. Every value is normalized to UTC on the way in.

**Idempotency.** Every write accepts `idempotency_key`. Omit it and one is derived from the
request content, so an identical resend returns the original record with a `duplicate` status
rather than storing a second copy. Reusing a key with a *different* body fails with
`idempotency_conflict` — a retry is safe, but it is never silent.

**Tracing.** Every response carries `trace_id` in the form `trace_<32-hex W3C trace ID>`. The
suffix maps directly onto your OTLP backend, and `trace_id` from a recall is what `missing`
feedback refers back to.

**Errors** always use one envelope:

```json
{
  "code": "memory_not_found",
  "message": "memory does not exist",
  "trace_id": "trace_4bf92f3577b34da6a3ce929d0e0e4736",
  "issues": []
}
```

Branch on `code`, never on `message`. `issues` is populated only for validation failures.

---

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness. Public. |
| `POST` | `/v1/observations` | Submit one sensor observation. |
| `POST` | `/v1/memories` | Explicitly retain content. |
| `POST` | `/v1/memories/batch` | Retain up to 100 memories in one encoder round trip. |
| `GET` | `/v1/memories/{memory_id}` | Read one memory with signed evidence. |
| `POST` | `/v1/recall` | Recall and answer. |
| `POST` | `/v1/feedback` | Record a useful/wrong/missing/correction signal. |
| `POST` | `/v1/forget` | Erase one memory or one observation. |
| `GET` | `/v1/deletions` | Page deletion tombstones from a cursor. |
| `GET` | `/v1/deletions/{tombstone_id}` | Read one deletion's propagation state. |
| `GET` | `/v1/jobs/{job_id}` | Read one observation-processing job. |
| `GET` | `/v1/jobs/{job_id}/events` | Follow that job as an SSE stream. |

---

### `GET /healthz`

Liveness only. It makes no claim about PostgreSQL, Redis, or the model endpoints — a health
check that asserts dependency readiness it has not tested is worse than one that stays silent.
Use it for load-balancer liveness, not for readiness gating.

```json
{"status": "ok", "trace_id": "trace_..."}
```

---

### `POST /v1/observations`

Submits one timestamped observation. Returns `202 Accepted`.

The bytes must already be in object storage. MindBridge reads the URI you give it; it does not
accept an upload.

**Request — `ObserveRequest`**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `tenant_id` | string | yes | Must be authorized by the key. |
| `device_id` | string | yes | Capturing device. |
| `boot_id` | string | yes | Changes on every device restart, so `sequence` need not survive one. |
| `sequence` | int ≥ 0 | yes | Monotonic within one `boot_id`; orders and deduplicates. |
| `sensor` | `camera` \| `microphone` | yes | The only two sensors that can carry required evidence. |
| `media_objects` | `MediaObjectInput[]` | yes | 1–8 items, no repeated `media_object_id`. |
| `occurred_at` | datetime | yes | When the observed events began. |
| `ended_at` | datetime | yes | Must not precede `occurred_at`. |
| `observed_at` | datetime | yes | When the device recorded them; may trail `ended_at`. |
| `clock_offset_ms` | int | no | Known device clock skew. Default `0`. |
| `identity_observations` | `IdentityObservationInput[]` | no | ≤ 512 spans. |
| `idempotency_key` | string | no | Derived from content when omitted. |

**`MediaObjectInput`**

| Field | Type | Notes |
| --- | --- | --- |
| `media_object_id` | string | Caller-assigned, unique within the observation. |
| `kind` | `image` \| `video` \| `audio` | Cross-checked against the URI extension when recognized. |
| `uri` | string | `s3://<bucket>/tenants/<tenant_id>/<key>`. |
| `sha256` | 64 hex chars | Digest of the exact bytes at `uri`. |
| `size_bytes` | int ≥ 0 | |
| `created_at` | datetime | When captured, not when uploaded. |
| `duration_ms` | int \| null | Must not exceed the observation span. Omit for a still image. |

**`IdentityObservationInput`**

Anonymous by construction — an `identity_id`, never a face or voice template.

| Field | Type | Notes |
| --- | --- | --- |
| `identity_id` | string | Device-local identity this span belongs to. |
| `kind` | `face` \| `voice` | Gates `transcript` and `visual_bbox_xyxy`. |
| `start_ms`, `end_ms` | int ≥ 0 | Milliseconds from the observation's start; must fall inside its span. |
| `confidence` | 0.0–1.0 | The edge detector's own confidence. |
| `model_id` | string | Provenance: the edge model that produced the span. |
| `scope` | `device` \| `observation` | Default `device`. |
| `transcript` | string \| null | `voice` only. All transcripts in one observation ≤ 65,536 characters. |
| `visual_bbox_xyxy` | 4 floats \| null | `face` only. **0..1 normalized** `(left, top, right, bottom)`, not pixels. Must have positive width and height. |

Pixels here are a real and previously shipped bug: a detector that leaks pixel coordinates makes
the annotator draw name boxes off-screen, and the failure surfaces much later as a model that
cannot identify people.

**Response — `ObservationReceipt`**

```json
{
  "observation_id": "obs_...",
  "processing_job_id": "job_...",
  "evidence_ids": ["evd_..."],
  "idempotency_key": "idem_...",
  "status": "accepted",
  "trace_id": "trace_..."
}
```

`status` is `accepted` or `duplicate`. `evidence_ids` covers spans registered synchronously,
before any derivation ran.

**Memory does not exist when this receipt returns.** Poll `GET /v1/jobs/{job_id}` until
`succeeded`, or follow the event stream, before issuing a recall that depends on it.

**Errors:** tenant errors, `idempotency_conflict`, `domain_invariant_failed`,
`task_broker_unavailable`.

---

### `POST /v1/memories`

Explicitly retains content that did not come from a sensor. Returns `201 Created`.

**Request — `RememberRequest`**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `tenant_id` | string | yes | |
| `summary` | string | yes | Written so it stays useful out of context. |
| `memory_type` | enum | yes | `episodic`, `semantic`, `procedural`, `prospective`, `working`, `perceptual`. |
| `occurred_at` | datetime | yes | What the content is *about*, not necessarily now. |
| `ended_at` | datetime | no | Defaults to `occurred_at`; must not precede it. |
| `evidence_ids` | string[] | no | ≤ 100 existing spans, no repeats. |
| `idempotency_key` | string | no | |

**Response — `RememberResult`**: a `MemoryView` plus `evidence`, `trace_id`, and `status`
(`created` or `duplicate`).

A memory written with no `evidence_ids` is `attested`, not `verified`. The domain layer refuses
to construct a `verified` record without evidence, so there is no way to write an unsupported
claim that later reads as confirmed.

**Errors:** tenant errors, `idempotency_conflict`, `domain_invariant_failed`, embedding errors,
evidence errors.

---

### `POST /v1/memories/batch`

Retains several memories in one call, and so in one embedder round trip. Returns `201 Created`.

**Request — `RememberBatchRequest`**

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `memories` | `RememberRequest[]` | yes | 1–100 entries, each exactly as the single write accepts it. |

**Response — `RememberBatchResult`**: `memories`, one `RememberResult` per request **in the order
sent**, so a caller can pair results with inputs positionally rather than by matching summaries.

Prefer this over a loop whenever more than one memory is on hand. The single-write path spends one
embedder call per memory, which on a served encoder is a round trip each; here the batch is encoded
together. Measured on one RTX 5090 against the same model, encoding 128 real memory summaries: 600
per second in batches of 32 against 183 per second one at a time.

Each entry is validated and applied on its own terms — the same tenant checks, the same
`idempotency_key` handling, the same `attested`-without-evidence rule as the single write. An entry
that duplicates an existing key comes back as `duplicate` in its own slot.

**Errors:** as `POST /v1/memories`, plus `domain_invariant_failed` for an empty or over-long
`memories` array.

---

### `GET /v1/memories/{memory_id}`

Query parameter: `tenant_id` (required).

Returns `MemoryResult` — the full `MemoryView`, its signed `EvidenceView[]`, and `trace_id`.
Evidence arrives attached rather than referenced, so verifying an answer needs no second call to
private storage.

**`MemoryView`**

| Field | Type | Notes |
| --- | --- | --- |
| `memory_id` | string | |
| `memory_type` | enum | |
| `summary` | string | |
| `evidence_ids` | string[] | |
| `occurred_at`, `ended_at`, `created_at` | datetime | |
| `verification_status` | `verified` \| `attested` \| `unverified` | Whether original media was inspected. |
| `state` | `active` \| `strengthened` \| `cold` \| `compressed` | |
| `salience` | 0.0–1.0 | |
| `strength` | float | Raised by useful access, lowered by decay. |
| `useful_access_count` | int | |
| `positive_feedback_count`, `negative_feedback_count` | int | |
| `last_accessed_at` | datetime \| null | |
| `supersedes_memory_id` | string \| null | The earlier version this replaced. |
| `superseded_at` | datetime \| null | Null while current. |

**`EvidenceView`**

| Field | Type | Notes |
| --- | --- | --- |
| `evidence_id` | string | |
| `media_object_id` | string | |
| `start_ms`, `end_ms` | int | Span within that media. |
| `media_url` | string | Short-lived signed URL. |
| `media_url_expires_at` | datetime | Re-read the memory for a fresh one. |

**Errors:** tenant errors, `memory_not_found`, `memory_deleted` (410), evidence errors.

`410 Gone` is not `404`. A memory that was explicitly deleted is a different fact from one that
never existed, and a client resolving a stale ID should be able to tell them apart.

---

### `POST /v1/recall`

**Request — `RecallRequest`**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `tenant_id` | string | — | |
| `query` | `RecallQuery` | — | |
| `memory_ids` | string[] | `[]` | ≤ 100, unique. A **strict scope**, not a ranking hint. |
| `filters` | `RecallFilters` | `{}` | Applied before ranking. |
| `mode` | `answer` \| `search` \| `enumerate` | `answer` | |
| `limit` | 1–100 | `20` | |
| `include_evidence` | bool | `true` | |

**`RecallQuery`** — `text` (string) and/or `media_object_ids` (≤ 8, unique). At least one is
required; neither is privileged. Stored media stands in for the query itself.

**`RecallFilters`**

| Field | Notes |
| --- | --- |
| `person_ids` | ≤ 100. |
| `device_ids` | ≤ 100. |
| `memory_types` | Restrict to these types. |
| `occurred_after` | **Inclusive.** |
| `occurred_before` | **Exclusive** — a memory exactly at this instant is excluded. Must not precede `occurred_after`. |

The asymmetry is intentional so adjacent windows tile without overlap, but it is the single most
common source of an off-by-one in a timeline query.

**Response — `RecallResult`**

```json
{
  "answer": "It was last seen beside the blue toolbox on the right of the workbench.",
  "confidence": 0.91,
  "memories": [ { "memory_id": "mem_...", "...": "MemoryView" } ],
  "evidence": [
    {
      "evidence_id": "evd_...",
      "media_object_id": "media_...",
      "start_ms": 184200,
      "end_ms": 188900,
      "media_url": "https://objects.example.com/signed",
      "media_url_expires_at": "2026-08-19T02:05:00Z"
    }
  ],
  "trace_id": "trace_..."
}
```

`answer` is null in `search` mode and when nothing supports one. Abstention is a result, not a
failure — an unsupported query that returns null is behaving correctly.

**Grounded follow-up:** pass selected `memory_id`s from a previous result as `memory_ids`. They
become the strict candidate scope; tenant, lifecycle, deletion, and evidence checks still apply,
but no unrelated memory is searched.

**Errors:** tenant errors, `enumeration_limit_exceeded`, embedding errors, evidence errors.

---

### `POST /v1/feedback`

Returns `201 Created`. Which fields are required depends on `feedback_type`:

| `feedback_type` | Requires | Must omit |
| --- | --- | --- |
| `useful` | `memory_id` | `correction_summary` |
| `wrong` | `memory_id` | `correction_summary` |
| `correction` | `memory_id`, `correction_summary` | — |
| `missing` | `recall_trace_id` | `memory_id` |

`missing` reports that a recall found nothing usable, so it names the *recall* rather than a
memory — which is why it takes `recall_trace_id`, the `trace_id` that recall returned.

`correction` does not overwrite. It writes a new version that supersedes the original, returned
as `corrected_memory_id`.

**Response — `FeedbackReceipt`**: `feedback_id`, `feedback_type`, `memory_id`,
`corrected_memory_id`, `resulting_state`, `resulting_strength`, `created_at`, `trace_id`.
`resulting_state` and `resulting_strength` are always both present or both null.

**Errors:** tenant errors, `memory_not_found`, `memory_deleted`, `idempotency_conflict`,
`domain_invariant_failed`, embedding errors, evidence errors. A correction writes a new memory
version, so it encodes one before replying — which is why the embedding errors apply here.

---

### `POST /v1/forget`

| Field | Type | Notes |
| --- | --- | --- |
| `tenant_id` | string | |
| `target_type` | `memory_record` \| `observation` | |
| `target_id` | string | |
| `idempotency_key` | string \| null | Forget is idempotent either way. |

`memory_record` erases that one memory. `observation` erases the source observation and
everything derived from it, including identity samples the edge learned from that source.

**Response — `ForgetReceipt`**

| Field | Notes |
| --- | --- |
| `tombstone_id` | Stable ID of this deletion barrier; usable as a cursor. |
| `target_type`, `target_id` | What was erased. |
| `propagation_state` | `pending` \| `propagating` \| `complete` \| `failed`. |
| `requested_at` | |
| `completed_at` | Null while propagation is incomplete. |
| `error_code` | Why propagation stalled, or null. |
| `trace_id` | |

**Only `complete` means every copy is gone.** A `200` here means the deletion was recorded and
started, not that it finished.

**Errors:** tenant errors, `forget_target_not_found`, `idempotency_conflict`,
`object_storage_unavailable`. Erasing the bytes is part of the command; a failure to reach them
marks the tombstone `failed` and re-raises rather than being swallowed.

---

### `GET /v1/deletions`

Query parameters: `tenant_id` (required), `cursor` (optional), `limit` (1–100, default 100).

Returns `DeletionPage` — `items` in a stable order, plus `next_cursor` (null on the last page).

This is how an offline device reconciles. A stale or foreign cursor is rejected with
`domain_invariant_failed` rather than answered with an empty page, because an edge device must
never read truncation as completion.

### `GET /v1/deletions/{tombstone_id}`

Query parameter: `tenant_id`. Returns `ForgetReceipt`. Tombstones are content-free by
construction, so they remain readable after the content is physically erased.

---

### `GET /v1/jobs/{job_id}`

Query parameter: `tenant_id`. Returns `ObservationProcessingJobView`:

| Field | Notes |
| --- | --- |
| `job_id`, `observation_id` | |
| `state` | `pending` \| `running` \| `succeeded` \| `failed`. |
| `attempt` | How many times the job has been claimed. |
| `error_code` | Why the last attempt failed, or null. |
| `memory_ids` | Memories this job derived. Present only when `succeeded`. |
| `created_at`, `updated_at` | `updated_at` is the only value on the row that always rises. |
| `trace_id` | |

`failed` settles the **attempt**, not the job: the stale-job sweep can still retry it.

Read `memory_ids` directly instead of searching for what was just written.

**Errors:** tenant errors, `job_not_found`.

---

### `GET /v1/jobs/{job_id}/events`

Server-sent events. Query parameter: `tenant_id`. Optional `Last-Event-ID` header resumes after
a dropped connection.

```text
event: job
id: 7
data: {"job_id":"job_...","state":"running","attempt":1,...}
```

Every event carries the **complete job view**, not a delta, so resuming needs only the last ID
received. State changes between two server reads are coalesced — you always observe the newer
state, but not necessarily every intermediate `attempt`. Treat `event_id` as opaque.

The stream ends when the attempt settles or the server closes its window. Reconnecting is the
caller's decision.

Once the stream is open, a failure arrives as an `error` event carrying the same `ErrorResponse`
envelope rather than as a status code — the status line has already been sent.

---

## Error codes

One table in `api/errors.py` generates both the raise sites and the OpenAPI document, so a code
cannot reach a caller without also reaching the published contract.

| Code | HTTP | Meaning |
| --- | --- | --- |
| `authentication_required` | 401 | A valid bearer API key is required. |
| `authentication_failed` | 401 | The bearer API key is invalid. |
| `tenant_access_denied` | 403 | The authenticated tenant cannot access this resource. |
| `forget_target_not_found` | 404 | Forget target or deletion tombstone does not exist. |
| `memory_not_found` | 404 | Memory does not exist. |
| `job_not_found` | 404 | Observation processing job does not exist. |
| `idempotency_conflict` | 409 | The idempotency key already stores different content. |
| `memory_deleted` | 410 | Memory content was explicitly deleted. |
| `request_validation_failed` | 422 | Request validation failed; see `issues`. |
| `domain_invariant_failed` | 422 | Well-formed, but violates a memory invariant. |
| `enumeration_limit_exceeded` | 422 | The `enumerate` scope exceeds the bound; narrow the filters. |
| `memory_integrity_failed` | 500 | Stored memory is inconsistent. |
| `internal_error` | 500 | The request failed for a reason the server did not anticipate. |
| `model_output_invalid` | 502 | Memory model returned invalid output. |
| `model_request_failed` | 502 | Memory model rejected its configured request. |
| `database_unavailable` | 503 | Memory storage is temporarily unavailable. |
| `model_unavailable` | 503 | Memory model is unavailable. |
| `object_storage_unavailable` | 503 | Evidence media is unavailable. |
| `task_broker_unavailable` | 503 | Observation processing is temporarily unavailable. |

Every authenticated `/v1` operation can return `authentication_required`,
`authentication_failed`, `tenant_access_denied`, `request_validation_failed`, and
`database_unavailable` whatever else it does. Per-endpoint sections above name only the codes
that endpoint adds.

### Retry guidance

| Class | Codes | Action |
| --- | --- | --- |
| Retry with backoff | 503 codes | Transient dependency. The same request will work. |
| Retry cautiously | 502 codes | A model failed. Retrying is reasonable; retrying forever is not. |
| Do not retry unchanged | 4xx codes | Fix the request. `idempotency_conflict` in particular means your key is reused with a different body. |
| Investigate | 500 codes | `memory_integrity_failed` indicates stored state is inconsistent; capture `trace_id`. |

## AML routes

`POST /aml/add` and `POST /aml/search` are registered only when `MINDBRIDGE_AML_API_KEY` is
configured. They exist to let the Agent Memory Leaderboard harness drive the production kernel
through its own expected shape. They are not part of the stable public contract and should stay
off in production.
