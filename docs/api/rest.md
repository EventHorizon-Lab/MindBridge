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
  "memory_type": "procedural"
}
```

`contents` has 1 through 100 items. One optional `memory_type` applies to the complete batch and
defaults to `semantic`. Response `201` is `{"memories": [...]}` in input order. The batch contract
has no per-item event time or metadata.

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
default. Response `200` is `{"hits": [...]}`; each hit has the memory fields plus `score`. A routed
query containing text uses hybrid dense/full-text retrieval; pure media uses dense retrieval.

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
  "hits": []
}
```

`memory_type` and `reference_at` have the same semantics as search. `hits` are the exact search
results used to ground the answer. The outbound generation request includes their content,
`memory_type`, `occurred_at`, `occurred_end`, `created_at`, metadata, and media. In the built-in model request, a
distinct question/evidence asset is serialized once even when multiple hits refer to it.

### Get a memory

```http
GET /v1/memories/{memory_id}
```

Response `200` is one memory object. A missing ID returns `404`.

### List memories

```http
GET /v1/memories?limit=50&cursor=opaque-value
```

`limit` defaults to 50. Omit `cursor` for the first page. Response `200`:

```json
{"items":[],"next_cursor":null}
```

Pass `next_cursor` back unchanged until it is `null`.

### Delete a memory

```http
DELETE /v1/memories/{memory_id}
```

Response is `204` with no body whether or not the ID existed.

## Error envelope

Every error response has one flat shape:

```json
{
  "code": "validation_error",
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

`issues` is empty for non-validation failures.

| Status | Typical code | Meaning |
| --- | --- | --- |
| 413 | `request_too_large` | Request body exceeds 8 MiB |
| 404 | `memory_not_found` | `GET` memory ID does not exist |
| 422 | `validation_error` | Request, media source, or public input is invalid |
| 502 | `model_error` | Routing, embedding, transcription, or generation failed |
| 502 | `model_output_truncated` | Generation stopped at an output token limit; retrying is pointless |
| 503 | `index_unavailable` or `storage_error` | Embedded index or durable state failed |
| 500 | `internal_error` | Unexpected failure |

Use `trace_id` to correlate a response with server logs. Model, index, storage, and unexpected
messages intentionally avoid provider, local path, or native-index details.

## Current limits

The REST API has no local-path input, large-file upload endpoint, update route, metadata filter,
logical scope parameter, chunking contract, per-asset vector control, or learned reranker. Fetch large
media in the host application and use the Python `Path`/`Blob` contract or a provider-specific
adapter. The OpenAI adapter inlines at most 64 MiB of base64-encoded media per embedding or
generation call, which is roughly 48 MiB of files on disk; generation admits ranked evidence within
that budget, and answer text evidence is limited to 4 MiB.
