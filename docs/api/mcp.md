# MCP tools

MindBridge speaks the official Model Context Protocol over stdio, exposing the same production
kernel that serves REST. Tool input and structured output schemas are generated from the same
Pydantic contracts, so an agent and an application see one contract rather than two that drift.

```bash
uv run --extra server mindbridge mcp
```

## Client configuration

```json
{
  "mcpServers": {
    "mindbridge": {
      "command": "uv",
      "args": ["run", "--extra", "server", "mindbridge", "mcp"],
      "env": {
        "MINDBRIDGE_DATABASE_URL": "postgresql://...",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://localhost:6379/0",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "mindbridge-media",
        "MINDBRIDGE_GENERATOR_API_KEY": "...",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.com/v1",
        "MINDBRIDGE_GENERATOR_MODEL_REVISION": "deployment-2026-08-11",
        "MINDBRIDGE_EMBEDDER_API_KEY": "...",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://embeddings.example.com/v1"
      }
    }
  }
}
```

The stdio process connects to the database directly and has no configured tenant list, so it
does **not** run the embedding-space startup probe the REST API runs. Pointing it at a
deployment mid-re-embedding will not be caught for you.

Deploy remote MCP only behind authenticated process or gateway isolation. The shipped command
exposes stdio deliberately rather than an unauthenticated HTTP listener.

## Tools

| Tool | Kind | Purpose |
| --- | --- | --- |
| `memory_observe` | idempotent write | Store one timestamped multimodal observation. |
| `memory_remember` | idempotent write | Retain one explicit memory. |
| `memory_recall` | read-only | Recall memories and answer with inspectable evidence. |
| `memory_get` | read-only | Read one memory by ID. |
| `memory_job` | read-only | Read how far one observation's processing has got. |
| `memory_feedback` | idempotent write | Record useful / wrong / missing / correction. |
| `memory_forget` | **destructive** | Erase one memory or source observation. |

Each carries MCP tool annotations, so a client can gate on them: reads are marked
`read_only_hint`, writes `idempotent_hint`, and `memory_forget` additionally
`destructive_hint`. All tools are `open_world_hint: false` — they touch one tenant's memory, not
the internet.

## Argument shape

Each tool takes the contract's **own fields** as its arguments rather than one nested `request`
object:

```json
{
  "name": "memory_recall",
  "arguments": {
    "tenant_id": "tenant_01",
    "query": {"text": "Where did I leave the red screwdriver?"},
    "mode": "answer",
    "limit": 20
  }
}
```

Every field carries a description, so the cross-field rules an agent would otherwise discover by
failing a call — which `feedback_type` needs which field, what `observe` requires of media it did
not upload — are in the schema it reads before calling.

**Unknown arguments are faulted, not dropped.** A middleware layer compares supplied argument
names against the contract's fields and rejects anything unrecognized. Silently discarding a
misspelled argument would answer a different question than the one asked, and the agent would
have no way to know.

## Tool notes

### `memory_observe`

Returns `observation_id`, `processing_job_id`, `evidence_ids`, `idempotency_key`, `status`, and
`trace_id`. Memory does not exist when it returns; exchange the `processing_job_id` through
`memory_job`.

Media must already be in object storage at `s3://<bucket>/tenants/<tenant_id>/<key>`, with a
matching SHA-256. MCP does not upload bytes.

### `memory_recall`

Modes are `answer` (default), `search`, and `enumerate`. Returns the answer, a confidence, the
memories it rests on, and signed evidence URLs pointing at exact `start_ms`–`end_ms` slices of
the original recording — so the agent can verify rather than trust.

For a grounded follow-up, pass `memory_ids` from the previous result. It is a strict scope.

`answer` is null when nothing supports one. An agent should treat that as "MindBridge does not
know", not as an error to retry.

### `memory_get`

Takes `tenant_id` and `memory_id`. Returns the memory with signed `EvidenceView`s attached, so
an agent does not need a second private-storage call to inspect the source.

### `memory_job`

Takes `tenant_id` and `job_id`. This tool exists so the `processing_job_id` that
`memory_observe` hands back is redeemable on the MCP face — without it an agent could only retry
recall blindly, which is exactly what the REST documentation tells callers not to do.

Following every intermediate state as a *stream* stays REST-only. MCP tools are
request/response, and inventing a streaming convention before a caller needs one would be
inventing a contract to maintain.

### `memory_feedback`

Which arguments are required depends on `feedback_type`:

| `feedback_type` | Requires |
| --- | --- |
| `useful`, `wrong` | `memory_id` |
| `correction` | `memory_id` and `correction_summary` |
| `missing` | `recall_trace_id` (and **not** `memory_id`) |

`missing` judges a recall rather than a memory, which is why it takes the `trace_id` the recall
returned.

### `memory_forget`

`target_type` is `memory_record` (that one memory) or `observation` (the source observation and
everything derived from it). The receipt reports that erasure was recorded and started;
`propagation_state` reaches `complete` only when every copy is gone, including on offline
devices.

This is the one tool marked destructive. Clients that gate destructive tools behind confirmation
will do so here.

## Errors

Failures report the same stable codes REST does — `code_for` resolves a raised exception to the
same code the REST handler would publish, so an agent reading `memory_not_found` from MCP and
from REST is reading one vocabulary. See [the error table](rest.md#error-codes).

## Known limitation

`mcp` 2.0 flattens the contract's fields into tool arguments, and its `pre_parse_json` step
consumes string values for `X | None` fields before MindBridge's validator sees them. Optional
string arguments therefore behave slightly differently over MCP than over REST. This is upstream
and not currently fixable from this side; if it affects you, use REST for that call.
