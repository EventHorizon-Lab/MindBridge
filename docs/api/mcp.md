# MCP API

The optional MCP adapter currently exposes five typed tools over one local `Memory` through stdio.
It is a schema and dispatch surface over the SDK execution plane, not a separate memory service.

## Install and run

```bash
uv add "mindbridge[local,mcp,openai]"
```

Host the adapter in application code so provider clients and their lifecycle remain explicit:

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels
from mindbridge.api.mcp import build_mcp_server

with OpenAI() as client:
    with Memory(
        "/var/lib/mindbridge/assistant",
        embedder=JinaOmniEmbedder(),
        answerer=OpenAIModels(generation_client=client),
    ) as memory:
        build_mcp_server(memory).run("stdio")
```

The application owns the provider client and `Memory`. Do not run REST, another MCP process, or a
second Python `Memory` against the same path concurrently.

## Multimodal content

`content`, `query`, and `question` accept a non-blank string or 1 through 16 ordered parts:

- `{"type":"input_text","text":"..."}`
- `{"type":"input_image","image_url":"data:image/png;base64,..."}`
- `{"type":"input_image","file_id":"..."}`
- `{"type":"input_file","file_url":"data:video/mp4;base64,..."}`
- `{"type":"input_file","file_data":"<base64>","media_type":"audio/wav"}`
- `{"type":"input_file","file_id":"..."}`

Source fields are mutually exclusive. Inline base64 is length-checked before decoding and
byte-checked afterward; `file_data` has an 8 MiB decoded ceiling, while data URL strings have a
tighter 8,192-character schema limit. Remote URLs, local paths, `file:` URLs, unknown nested
fields, and `input_image.detail` are rejected. Fetch remote media in the host application or use
the Python API for direct `Path` input.

## Tools

### `add_memory`

Stores one stable text or multimodal record.

| Argument | Type | Required | Default |
| --- | --- | --- | --- |
| `content` | string or ordered content parts | yes | — |
| `occurred_at` | timezone-aware ISO 8601 datetime or null | no | null |
| `occurred_end` | timezone-aware ISO 8601 datetime or null | no | null |
| `metadata` | JSON object or null | no | null |
| `memory_type` | `semantic`, `episodic`, or `procedural` | no | `semantic` |

The structured result contains `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`,
`occurred_at`, `occurred_end`, and `metadata`. An event end requires a start and must be later than
it. Asset results contain safe metadata but never a local path. Repeating canonical input returns
the existing record. The tool is marked as a non-destructive, idempotent write.

Example arguments:

```json
{
  "content": [
    {"type": "input_text", "text": "Design review recording"},
    {
      "type": "input_file",
      "file_data": "AAAA",
      "media_type": "video/mp4"
    }
  ],
  "memory_type": "episodic",
  "metadata": {"source": "review"}
}
```

### `search_memories`

Searches local memories.

| Argument | Type | Required | Default |
| --- | --- | --- | --- |
| `query` | string or ordered content parts | yes | — |
| `limit` | integer from 1 through 100 | no | 10 |
| `memory_type` | one memory role or null | no | null |
| `reference_at` | timezone-aware ISO 8601 datetime or null | no | current UTC |

The result is `{"hits": [...]}`. Each hit contains memory fields plus `score`. Routed queries with
text use hybrid dense/full-text retrieval; pure media uses dense retrieval. Relative temporal
expressions resolve against `reference_at`. Search is conservatively marked non-read-only because
it may drain durable index work or populate transcript caches; it never reinforces a hit merely for
being returned.

### `ask_memory`

Answers only from retrieved local memories.

| Argument | Type | Required | Default |
| --- | --- | --- | --- |
| `question` | string or ordered content parts | yes | — |
| `limit` | integer from 1 through 100 | no | 5 |
| `memory_type` | one memory role or null | no | null |
| `reference_at` | timezone-aware ISO 8601 datetime or null | no | current UTC |

The result contains `answer` and the exact grounding `hits`. Like search, the tool is marked
non-read-only because retrieval may maintain local caches/index state. The built-in outbound answer
request serializes each distinct question/evidence asset once even if several hits refer to it. It
reserves the raw-media budget for question assets, keeps ranked evidence media that fits, and
retains overflow hits as text when possible. It also sends each hit's content, `memory_type`,
`occurred_at`, `occurred_end`, `created_at`, and metadata to the configured generation endpoint.

### `get_memory`

Reads one record by stable ID.

| Argument | Type | Required |
| --- | --- | --- |
| `memory_id` | non-blank, trimmed string | yes |

The structured result is one memory record. The tool is read-only.

### `delete_memory`

Idempotently deletes one record.

| Argument | Type | Required |
| --- | --- | --- |
| `memory_id` | non-blank, trimmed string | yes |

The result is `{"deleted":true}` or `{"deleted":false}`. The tool is marked destructive and
idempotent.

## Validation and errors

Only the five documented tool names and their exact top-level arguments are accepted. Unknown,
tenant, user, and run fields are rejected rather than silently ignored.

Tool failures expose a compact JSON object in the MCP error result:

```json
{"code":"validation_error","message":"tool arguments are invalid"}
```

Stable codes include `validation_error`, `memory_not_found`, `model_error`, `storage_error`,
`index_unavailable`, and `internal_error`. Provider responses, credentials, filesystem paths, and
native-index details are not included.

## Agent consumption contract

MCP is MindBridge's current agent-native surface. An agent can select and call every exposed
operation from the tool name, description, annotations, and JSON schema; human-oriented output is
not part of the contract.

- Tool schemas are strict. Unknown arguments fail instead of being guessed or ignored.
- Successful calls return structured records, hits, answers, or deletion state with stable IDs.
- Failures use stable codes, so recovery does not depend on matching prose.
- Read, idempotency, and destructive annotations disclose side effects before a call.
- Search and answer limits are bounded; exact grounding hits remain available for audit or a later
  agent step.
- Tool names, argument schemas, result schemas, annotations, and error codes are public contracts;
  changing one requires the same compatibility treatment as changing the Python or REST API.
- MCP capabilities are derived from the SDK. A tool validates transport input, calls the matching
  `Memory` operation, and serializes its result; it must not duplicate memory policy.

An integration may search before a turn or add a result afterward, but that lifecycle policy is not
hidden inside the MCP server. Automated writes must remain observable and preserve whether content
came from a user, an agent, or source evidence.

## Programmatic adapter

Applications embedding MCP pass an existing synchronous memory:

```python
from mindbridge import JinaOmniEmbedder, Memory
from mindbridge.api.mcp import build_mcp_server

with Memory("./data/agent", embedder=JinaOmniEmbedder()) as memory:
    server = build_mcp_server(memory)
    server.run("stdio")
```

`build_mcp_server` does not take ownership of the supplied instance.
The MCP SDK runs these synchronous tool functions in its AnyIO worker pool; MindBridge does not
maintain a second sync/async dispatch layer.

## Current limits

The five current tools do not yet cover the complete SDK capability inventory. Batch addition,
listing, speech analysis, speaker registration, reindexing, and optimization remain Python
operations. This is an implementation gap, not a separate MCP execution model. Additive tools must
map to the existing SDK operations and carry appropriate read, idempotency, and destructive
annotations.

There is no large-file upload tool, local-path input, logical scope, chunking option, per-asset
vector control, or learned reranker option. The OpenAI adapter inlines at most 64 MiB of raw media
per embedding or generation call; generation admits
ranked evidence within that budget. Answer text evidence is limited to 4 MiB. Use a
provider-specific upload adapter for larger media.
