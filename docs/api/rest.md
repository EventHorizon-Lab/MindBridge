# REST API

The optional FastAPI adapter exposes the same local `Memory` core under `/v1`.

## Start the API

```bash
uv add "mindbridge[local,server]"
mindbridge serve --data-dir .mindbridge
```

This development command binds to loopback with one worker and no inbound authentication. OpenAPI
JSON is served at `/openapi.json`, Swagger UI at `/docs`, and ReDoc at `/redoc`. See
[deployment](../deployment.md) before exposing the service.

## Authentication and request limits

When `create_app(api_key="...")` is configured, every `/v1` request requires:

```text
Authorization: Bearer your-service-key
```

`/healthz` is unauthenticated. The inbound service key is unrelated to outbound model keys.
Authentication is checked before body parsing. Each `/v1` request body is limited to 8 MiB,
including JSON and base64 overhead.

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
  "image_url": "https://media.example/prototype.png"
}
```

```json
{
  "type": "input_image",
  "file_id": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

`image_url` may instead be a base64 `data:image/...` URL. URL/data source strings are limited to
8,192 characters. HTTPS hosts must be explicitly enabled with `MINDBRIDGE_ALLOWED_URL_HOSTS`;
each redirect resolves and pins its connection to a verified public IP. A concrete MIME hint must
match response `Content-Type` exactly; a family hint must match its media family. The OpenAI
`detail` field is not accepted because MindBridge does not currently carry a sampling-detail
contract.

### File part

Use `input_file` for image, video, or audio. Supply exactly one of `file_url`, `file_data`, or
`file_id`:

```json
{
  "type": "input_file",
  "file_url": "https://media.example/demo.mp4",
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
accepts HTTPS or a base64 data URL. For HTTPS, `media_type` may be concrete, a family hint such as
`video/*`, or omitted when a common suffix determines it. Local paths and `file:` URLs are never
accepted over REST.

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
      "image_url": "https://media.example/prototype.png"
    }
  ],
  "memory_type": "episodic",
  "occurred_at": "2026-08-27T09:00:00Z",
  "metadata": {"source": "design-review"}
}
```

`occurred_at`, `metadata`, and `memory_type` are optional; memory type defaults to `semantic`.
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
`memory_type`, `occurred_at`, `created_at`, metadata, and media. In the built-in model request, a
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
| 401 | `authentication_error` | Missing or invalid bearer service key |
| 413 | `request_too_large` | Request body exceeds 8 MiB |
| 404 | `memory_not_found` | `GET` memory ID does not exist |
| 422 | `validation_error` | Request, media source, or public input is invalid |
| 502 | `model_error` | Routing, embedding, transcription, or generation failed |
| 503 | `index_unavailable` or `storage_error` | Embedded index or durable state failed |
| 500 | `internal_error` | Unexpected failure |

Use `trace_id` to correlate a response with server logs. Model, index, storage, and unexpected
messages intentionally avoid provider, local path, or native-index details.

## Current limits

The REST API has no local-path input, large-file upload endpoint, update route, metadata filter,
logical scope parameter, chunking contract, per-asset vectors, or learned reranker. Use Python
`Path` or an allowed HTTPS `URL` for media too large to place in one JSON request. The built-in `data` model
transport separately rejects embedding or generation calls above 64 MiB of aggregate raw media;
answer text evidence is limited to 4 MiB. Large video requires trusted co-located `file` transport
or a custom streaming/upload backend.
