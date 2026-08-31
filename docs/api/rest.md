# REST API

The optional FastAPI adapter exposes the same local `Memory` core under `/v1`.

## Start the API

```bash
uv add "mindbridge[local,openai,server]"
```

Construct `Memory` and provider clients in the host application, then pass the instance to
`create_app(memory=...)`:

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels
from mindbridge.api import create_app

client = OpenAI(timeout=30.0, max_retries=3)
memory = Memory(
    ".mindbridge",
    embedder=JinaOmniEmbedder(),
    answerer=OpenAIModels(generation_client=client),
)
app = create_app(memory=memory)
```

Host it with the application's ASGI stack, register shutdown hooks for `memory` and `client`, and
use exactly one worker for a directory. OpenAPI JSON is served at `/openapi.json`, Swagger UI at
`/docs`, and ReDoc at `/redoc`. See [deployment](../deployment.md) before exposing the service.

## Deployment authentication and request limits

MindBridge does not implement an identity or token system. `create_app()` returns an
unauthenticated ASGI application so a deployment can use its existing API gateway, service mesh,
or Starlette/FastAPI authentication middleware. Each `/v1` request body is limited to 8 MiB before
parsing, including JSON and base64 overhead.

## Content input

The `content`, `query`, and `question` fields accept either a non-blank string or an ordered array
of 1 through 16 OpenAI-compatible content parts. Extra fields are rejected.

### Text part

```json
{"type":"input_text","text":"The prototype after the review"}
```

### Image part

Supply exactly one of `image_url` or `file_id`:

```json
{
  "type": "input_image",
  "image_url": "data:image/png;base64,iVBORw0KGgo="
}
```

```json
{
  "type": "input_image",
  "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

`image_url` accepts only a base64 `data:image/...` URL and is limited to 8,192 characters. Remote
URLs are rejected; fetch them with the host application's HTTP client. The OpenAI `detail` field
is not accepted because MindBridge does not currently carry a sampling-detail contract.

### File part

Use `input_file` for image, video, or audio. Supply exactly one of `file_url`, `file_data`, or
`file_id`:

```json
{
  "type": "input_file",
  "file_url": "data:video/mp4;base64,AAAA",
  "media_type": "video/mp4",
  "filename": "demo.mp4"
}
```

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
  "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

`file_data` is raw base64 without a `data:` prefix and requires a concrete MIME type. `file_url`
accepts only a base64 data URL. Remote URLs, local paths, and `file:` URLs are never accepted over
REST.

## Response objects

A memory response keeps textual content and adds explicit modality and safe asset metadata:

```json
{
  "id": "sha256-memory-id",
  "content": "The prototype after the review",
  "modality": "image",
  "memory_type": "episodic",
  "assets": [
    {
      "id": "sha256-asset-id",
      "modality": "image",
      "media_type": "image/png",
      "size_bytes": 42001,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "name": "prototype.png"
    }
  ],
  "created_at": "2026-08-27T09:30:00Z",
  "occurred_at": null,
  "occurred_end": null,
  "metadata": {"source": "design-review"}
}
```

`modality` is persisted by the core and is one of `text`, `image`, `video`, `audio`, or `omni`.
`memory_type` is `semantic`, `episodic`, or `procedural`. Asset filesystem paths are never
serialized.

## Endpoints

### Health

```http
GET /healthz
```

Response `200`:

```json
{"status":"ok"}
```

### Create a memory

```http
POST /v1/memories
Content-Type: application/json
```

```json
{
  "content": [
    {"type": "input_text", "text": "The prototype after the review"},
    {
      "type": "input_image",
      "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "memory_type": "episodic",
  "occurred_at": "2026-08-27T09:00:00Z",
  "occurred_end": "2026-08-27T09:05:00Z",
  "metadata": {"source": "design-review"}
}
```

`occurred_at`, `occurred_end`, `metadata`, and `memory_type` are optional; memory type defaults to
`semantic`. An event end requires a timezone-aware start and must be later than it.
Response `201` is one memory object. Repeating the same canonical input returns the existing record
without another model call.

### Create a batch

```http
POST /v1/memories/batch
Content-Type: application/json
```

```json
{
  "contents": [
    "First memory",
    [
      {"type": "input_text", "text": "Second memory"},
      {
        "type": "input_file",
        "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      }
    ]
  ],
  "occurred_at": ["2026-08-27T09:00:00Z", null],
  "occurred_end": [null, null],
  "metadata": [{"source": "design-review"}, null],
  "memory_type": "procedural"
}
```

`contents` has 1 through 100 items. One optional `memory_type` applies to the complete batch and
defaults to `semantic`. Response `201` is `{"memories": [...]}` in input order.

`occurred_at`, `occurred_end`, and `metadata` are optional per-item arrays. Supplying one requires
exactly one entry per item; omitting it applies `null` to every item. These values are part of a
memory's content-addressed identity, so a batch import that omits them produces different IDs than
the same records added with them — over any surface.

### Search memories

```http
POST /v1/memories/search
Content-Type: application/json
```

```json
{
  "query": [
    {"type": "input_text", "text": "Find a prototype like this"},
    {
      "type": "input_image",
      "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "limit": 10,
  "memory_type": "episodic",
  "reference_at": "2026-08-27T12:00:00Z"
}
```

`limit` defaults to 10 and ranges from 1 through 100. `memory_type` optionally filters one role.
`reference_at` resolves relative date expressions and must include a timezone; current UTC is the
default unless the query declares a valid English reference date such as
`Today is May 2, 2024`; an explicit value always wins. Response `200` is `{"hits": [...]}`; each
hit has the memory fields plus `score`. The complete ordered query and bounded focused keys from
its first text atom and media supply dense candidates; the focused text also supplies lexical
candidates. All routes collapse aggregate or atomic document keys to parent memories.

### Answer from memories

```http
POST /v1/answers
Content-Type: application/json
```

```json
{
  "question": "What changed last week?",
  "limit": 5,
  "memory_type": "episodic",
  "reference_at": "2026-08-27T12:00:00Z"
}
```

`limit` defaults to 5. Response `200`:

```json
{
  "answer": "I don't know based on the available memories.",
  "hits": [],
  "abstained": true,
  "abstention_reason": "no_evidence"
}
```

`memory_type` and `reference_at` have the same semantics as search. `hits` are the exact search
results used to ground the answer. `abstention_reason` is `no_evidence`,
`insufficient_evidence`, or `null` when `abstained` is false. The outbound generation request
includes their content, `memory_type`, `occurred_at`, `occurred_end`, `created_at`, metadata, and
media. In the built-in model request, a distinct question/evidence asset is serialized once even
when multiple hits refer to it.

### Get a memory

```http
GET /v1/memories/{memory_id}
```

Response `200` is one memory object. A missing ID returns `404`.

### List memories

```http
GET /v1/memories?limit=100&cursor=opaque-value
```

`limit` defaults to 100 and ranges from 1 through 100, matching `Memory.list`. Omit `cursor` for the
first page. Response `200`:

```json
{"items":[],"next_cursor":null}
```

Pass `next_cursor` back unchanged until it is `null`.

### Delete a memory

```http
DELETE /v1/memories/{memory_id}
```

Response `200` is `{"deleted": true}` when the ID existed and `{"deleted": false}` when it did not.
Deletion is idempotent either way, and the flag matches the Python `bool` and the MCP
`delete_memory` result so a reconciliation loop can tell the two apart.

## Error envelope

Every error response has one flat shape:

```json
{
  "code": "validation_error",
  "reason": "input_invalid",
  "retryable": false,
  "stage": null,
  "subject": null,
  "message": "request validation failed",
  "trace_id": "trace_0123456789abcdef",
  "issues": [
    {
      "location": ["body", "content"],
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

`code` is the stable outer taxonomy. `reason` narrows it to a closed sub-vocabulary, `stage` names
the pipeline stage that failed, and `subject` names the asset, memory, or batch position the failure
is about. `issues` is empty for non-validation failures. Any of `reason`, `stage`, and `subject` may
be `null` when MindBridge cannot classify a failure further; an unclassified failure is never
reported as retryable.

**`retryable` is a lookup on `reason`, never a guess.** It is `true` only for `rate_limited`,
`timeout`, `connection_failed`, `data_dir_in_use`, `flush_failed`, and `index_missing`; the last two
have no raise site today and are reserved. A retryable
failure that maps to `503` also carries a `Retry-After` header.

### Codes and reasons

| `code` | `reason` values |
| --- | --- |
| `validation_error` | `input_invalid`, `unknown_field`, `payload_too_large` |
| `memory_not_found` | `memory_not_found` |
| `speaker_not_found` | `speaker_not_found` |
| `identity_not_found` | `identity_not_found` |
| `model_error` | `backend_not_configured`, `unsupported_modality`, `auth_failed`, `rate_limited`, `quota_exhausted`, `timeout`, `connection_failed`, `request_rejected`, `response_invalid`, `payload_too_large`, `asset_unavailable`, `asset_changed` |
| `model_output_truncated` | `output_truncated` |
| `storage_error` | `data_dir_in_use`, `schema_unsupported`, `io_failed` |
| `index_unavailable` | *(unset; an index failure is never assumed retryable)* |
| `internal_error` | `unexpected` |

`model_error` reasons that name a provider condition are classified from the official OpenAI SDK's
own exception classes when the bundled adapter is in use; MindBridge does not invent a parallel
taxonomy. The original provider exception stays as the raised error's `__cause__` in the owner
process and is never serialized.

### Status codes

| Status | Code and reason | Meaning |
| --- | --- | --- |
| 404 | `not_found` | Route does not exist |
| 404 | `memory_not_found` | Memory ID does not exist |
| 404 | `speaker_not_found` | Local speaker identity does not exist |
| 404 | `identity_not_found` | Face/voice identity does not exist |
| 405 | `method_not_allowed` | Method is not allowed for this route |
| 413 | `request_too_large` | Request body exceeds 8 MiB |
| 422 | `validation_error` | Request, media source, or public input is invalid |
| 422 | `model_error` + `unsupported_modality` | No configured backend accepts this modality |
| 500 | `storage_error` + `schema_unsupported` | On-disk schema needs a different MindBridge version |
| 500 | `internal_error` | Unexpected failure |
| 501 | `model_error` + `backend_not_configured` | The operation needs a backend this deployment never supplied |
| 502 | `model_error` | A permanent upstream or response failure; retrying the same call will not help |
| 502 | `model_output_truncated` | Generation stopped at an output token limit; retrying is pointless |
| 503 | `model_error` with `retryable: true` | Transient provider failure; honour `Retry-After` |
| 503 | `index_unavailable` or `storage_error` | Embedded index or durable state failed |
| — | `http_error` | Any other framework-level HTTP failure |

`501` and `502` mean the same call can never succeed; only `503` invites a retry. Use `trace_id` to
correlate a response with server logs. Messages are author-written literals: they never carry
provider responses, credentials, local paths, or native-index details. `subject` is withheld for
`storage_error`, `index_unavailable`, and `internal_error`, because it names server state rather
than caller input.

## Current limits

### Operations without a route

REST covers `add`, `add_many`, `search`, `ask`, `get`, `list`, and `delete` with the same defaults,
field meanings, and error semantics as the Python SDK. Seven documented SDK operations have no
route:

| Operation | Why there is no route |
| --- | --- |
| `speech` | Not implemented on any transport yet |
| `faces` | Python-only visual identity analysis |
| `register_speaker` | Not implemented on any transport yet |
| `register_identity` | Python-only face/voice identity naming |
| `reinforce` | Not implemented on any transport yet |
| `reindex` | Owner-process maintenance: it rebuilds the whole index and must not be reachable by an unauthenticated client |
| `optimize` | Owner-process maintenance, for the same reason |

`speech`, `faces`, identity registration, and `reinforce` are implementation gaps, not a different
execution model. Use the Python API in the owner process until they ship. See
[Python SDK](python-sdk.md) for the full inventory.

### Input limits

| Limit | REST | MCP | Python |
| --- | --- | --- | --- |
| Content parts per operation | 16 | 16 | 128 |
| Characters per text part | 65,536 | 65,536 | 65,536 |
| Inline media per part | Bounded by the request body | 8 MiB | Unbounded on disk |
| Total request size | 8 MiB | Unbounded framing | Unbounded |

The transports are deliberately narrower than the Python API: an HTTP body is bounded before parsing
so an oversized request cannot reach the memory core. Fetch large media in the host application and
use the Python `Path`/`Blob` contract or a provider-specific adapter.

### Absent features

The REST API has no local-path input, large-file upload endpoint, update route, metadata filter,
logical scope parameter, chunking contract, per-asset vector control, or learned reranker. The OpenAI
adapter inlines at most 20 MiB per base64-encoded media item and 64 MiB per embedding or generation
call, roughly 15 MiB per file and 48 MiB in aggregate on disk; generation admits ranked evidence
within those budgets, and answer text evidence is limited to 4 MiB.
