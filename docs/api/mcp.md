# MCP API

## Surface

The optional MCP adapter exposes exactly six typed tools over one injected synchronous `Memory`.
It validates tool input, calls the matching SDK operation, and returns structured public values.
It does not own storage, provider selection, or the injected memory. Finalized media arrives
through ordinary content parts; live audio and vision packet ingestion, and `StreamEvent`
reduction, stay Python-only because a tool call is a finite request.

## Start the adapter

Install the MCP extra together with the extras required by the chosen backends, then host the
server in the application that owns `Memory`:

```bash
uv add "mindbridge[local,mcp]"
```

```python
from mindbridge import Memory
from mindbridge.api.mcp import build_mcp_server

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    build_mcp_server(memory).run("stdio")
```

```text
build_mcp_server(memory: Memory) -> MCPServer[None]
```

## Lifecycle and ownership

`build_mcp_server` borrows `memory`; it neither opens nor closes it. The host must keep the owner
alive for the server lifetime and close it during shutdown. Do not run another `Memory`, REST, or
MCP owner against the same physical `data_dir`.

MindBridge adds no authentication to any MCP transport. Stdio inherits local process permissions;
an SSE or streamable-HTTP host must add authentication, authorization, TLS, request limits, and
rate limits. MCP error `subject` is unredacted and may contain an owner-local path. See
[deployment](../deployment.md) for process and transport boundaries.

## Contract

### Content input

`content`, `query`, and `question` use the same strict `input_text`, `input_image`, and `input_file`
union as [REST content input](rest.md#content-input): a non-blank string or 1 through 16 ordered
parts. Source fields are mutually exclusive. Data URLs must contain base64 media; `file_data`
requires a concrete image, video, or audio MIME type.

Remote URLs, local paths, `file:` URLs, unknown nested fields, and `input_image.detail` are
rejected. The MCP-specific media bounds are listed below.

### Tools

| Tool | Arguments and defaults | Structured result | Annotation |
| --- | --- | --- | --- |
| `add_memory` | required `content`; `occurred_at=None`; `occurred_end=None`; `metadata=None`; `memory_type="semantic"`; `context=None` | `MemoryResult` | idempotent write |
| `search_memories` | required `query`; `limit=10`; `memory_type=None`; `reference_at=None`; `occurred_from=None`; `occurred_until=None`; `scope=None` | `{"hits":[SearchHitResult,...]}` | retrieval |
| `ask_memory` | required `question`; `limit=5`; `memory_type=None`; `reference_at=None`; `scope=None` | `AnswerResponse` | retrieval |
| `get_memory` | required `memory_id` | `MemoryResult` | read-only |
| `list_memories` | `limit=100`; `cursor=None` | `PageResult` | read-only |
| `delete_memory` | required `memory_id` | `{"deleted":bool}` | destructive, idempotent |

All timestamps must be timezone-aware. An event end requires a start and must be later than it.
Search event bounds are a half-open overlap filter; two bounds require
`occurred_until > occurred_from`. `memory_type` is `semantic`, `episodic`, or `procedural`.
Pagination cursors are opaque and must be passed back unchanged.

`context` carries typed observation basis, source ID, confidence, validity, and optional spatial
pose. `scope.valid_at` selects world validity and `scope.known_at` selects the transaction
version known then; `scope.near` and `scope.radius_m` must appear together, and their frame ID
and observer/subject anchor must match the stored spatial context. SQLite reapplies both filters
after candidate retrieval.

`add_memory` is content-addressed. `delete_memory` reports whether a record existed. Search and
answer are not marked read-only because their SDK path may persist lazy transcript caches; they are
also not advertised as idempotent. Every tool has `open_world_hint=false`.
`ask_memory` requires an answerer in the injected memory; without one it returns
`model_error/backend_not_configured`.

### Result objects

Successful calls populate MCP `structuredContent`:

| Object | Fields |
| --- | --- |
| `AssetResult` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResult` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context`, `forgotten_at` |
| `SearchHitResult` | all memory fields plus `score` |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResult` | `items`, `next_cursor` |

These fields have the same meanings as the [REST response objects](rest.md#response-objects).
Successful result objects never serialize filesystem paths; error `subject` is the exception
described above.

## Errors and limits

### Validation and errors

Only the six documented names and their exact top-level arguments are accepted. An unknown tool or
top-level argument returns `validation_error/unknown_field`; schema and SDK input failures return
`validation_error/input_invalid`. Unknown values are not echoed.

Failed tool calls set `isError` and carry the same JSON envelope as
[REST](rest.md#error-envelope) in their text content:

```json
{
  "code": "validation_error",
  "reason": "unknown_field",
  "retryable": false,
  "stage": null,
  "subject": null,
  "message": "tool arguments contain unknown fields",
  "trace_id": "trace_0123456789abcdef0123456789abcdef",
  "issues": [
    {
      "location": ["arguments", "run_id"],
      "message": "Extra inputs are not permitted",
      "type": "extra_forbidden"
    }
  ]
}
```

Stable SDK codes are `mindbridge_error`, `validation_error`, `memory_not_found`,
`speaker_not_found`, `identity_not_found`, `model_error`, `model_output_truncated`,
`storage_error`, and `index_unavailable`; unexpected failures use `internal_error`. Reasons and
retryability use the [shared vocabulary](rest.md#codes-and-reasons).

Unlike unauthenticated REST, the MCP adapter retains `subject` for every SDK code to support owner
diagnostics. That is safe only for trusted clients; a network host must protect or redact the error
envelope at its outer boundary. Provider exceptions, credentials, and unexpected implementation
details are still never serialized.

### Operations without a tool

The following Python operations have no MCP tool:

| Operation | Boundary |
| --- | --- |
| `add_many` | Call `add_memory` per item or use the REST batch route |
| `add_stream` | Call `add_memory` for each completed observation |
| `search_with_trace` | Python and local-CLI diagnostics |
| `speech`, `faces` | Owner-process media analysis |
| `register_speaker`, `register_identity` | Owner-process identity naming |
| `identity`, `unlink_identity` | Owner-process identity inspection and merge reversal |
| `reinforce` | Owner-process feedback |
| `reindex`, `optimize` | Owner-process index maintenance |

`list_memories` is part of the six-tool surface and supports the same default page size and opaque
cursor contract as `Memory.list`.

### Input limits

| Bound | MCP value |
| --- | --- |
| Content parts | 1 through 16 |
| One data-URL source string | 8,192 characters |
| One decoded `file_data` value | 8 MiB |
| Normalized text, including combined text parts | 65,536 characters |
| Search, answer, or page `limit` | 1 through 100 |
| Serialized metadata | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |

MCP has no aggregate framing budget, but each inline media value is bounded before model or storage
work. It has no local-path input, remote fetch, large-file upload tool, capture-stream tool,
coordinate-frame transform, logical scope, or separate authentication policy; the MCP host owns
transport access control. Configured model backends may
impose a smaller aggregate budget, including the [OpenAI inline limits](python-sdk.md#bundled-adapters).
