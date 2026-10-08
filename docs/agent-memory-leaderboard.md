# Agent Memory Leaderboard

`mindbridge-bench eval serve` exposes MindBridge in the
[Agent Memory Leaderboard (AML) Add/Search format](https://agentmemoryleaderboard.ai/api-guide).
It is a participant-hosted benchmark adapter. AML calls these endpoints and runs its own Answer
and Eval stages; this command does not generate final answers or publish local evaluation scores.

## Start the server

Install the benchmark, model, and HTTP extras:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --extra benchmarks --extra local --extra openai --extra server
```

Use a dedicated benchmark directory and the existing evaluation configuration format. Only
configured ingestion and retrieval backends run; the adapter disables `generation` and ignores
the `benchmark:` section and the configuration's `data_dir`. An omitted embedding section uses
the evaluation runner's pinned default. A minimal configuration is:

```yaml
embedding:
  provider: sentence-transformers
  model: jinaai/jina-embeddings-v5-omni-small-retrieval
  revision: e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4
  dimension: 1024
```

Set the participant's Memory System Key in the environment and start the adapter. The `server`
extra includes Uvicorn for this launch command:

```bash
export MINDBRIDGE_AML_API_KEY="..."

uv run --frozen mindbridge-bench eval serve \
  --config .benchmarks/configs/aml.yaml \
  --data-root .benchmarks/aml \
  --host 127.0.0.1 \
  --port 8000
```

Use `--api-key-env NAME` to read a different environment variable. Add and Search accept
`Authorization: Bearer <key>`, `Authorization: Token <key>`, or `X-Api-Key: <key>`.
`GET /health` needs no key. `--public-smoke` permits an absent key for public smoke only; if a key
is set, authentication still applies.

Expose the service through your HTTPS deployment and register its `/add`, `/search`, and
`/health` URLs with AML. The loopback launch above is for local validation; AML needs a publicly
reachable deployment. Run one server process per data root. All users share loaded model weights,
and each request opens and closes its own user's store. Advertise concurrency based on your
deployed models and storage capacity.

## Request and response format

An Add request writes messages in source order. Source timestamps are Unix milliseconds:

```json
{
  "request_id": "eval:run-1:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "Alice's favorite color is red."
    }
  ],
  "user_id": "eval:run-1:conversation-0",
  "session_id": "eval:run-1:session-0"
}
```

`POST /add` returns HTTP 200 only after SDK ingestion, search-index flush, and the adapter's
source and retry records succeed. The response echoes identifiers without trimming:

```json
{
  "success": true,
  "request_id": "eval:run-1:chunk-0",
  "user_id": "eval:run-1:conversation-0",
  "session_id": "eval:run-1:session-0"
}
```

Retries with the same `user_id`, `request_id`, and payload reuse the logical write, including
after restart. Reusing that request ID with a different payload returns 422. Message roles,
session IDs, and source positions are stored as application metadata.

A Search request keeps multiple-choice options at the top level:

```json
{
  "query": "What is Alice's favorite color?",
  "options": ["A. red", "B. blue"],
  "user_id": "eval:run-1:conversation-0",
  "top_k": 100
}
```

`POST /search` calls `Memory.search()` with the original query; options are accepted but do not
change retrieval. It returns `{"data": [...]}` in SDK rank order, with stable `id`, `content`,
`score`, and ISO 8601 `created_at` fields. The timestamp uses source time when available and
persistence time otherwise. A user with no memories gets `{"data": []}`. `top_k` must be an
integer from 1 through 100.

For multimodal Add content and Search queries, use ordered content arrays:

```json
[
  {"type": "text", "text": "Caption before the image"},
  {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,<encoded image bytes>"}
  },
  {"type": "text", "text": "Text after the image"}
]
```

Original messages return with their exact text and part order. Derived memories return their
SDK evidence text and any supported images. Images must be inline JPEG, PNG, or WebP, at most
10 MiB decoded each and 30 MiB total per Add request or Search response. Remote image URLs are
rejected. Oversized responses return a clear 422 instead of silently removing evidence. The
underlying SDK's input and backend capability limits still apply and can reject unsupported
inputs; there is no adapter text truncation.

Invalid payloads return 422, invalid credentials return 401, and model/storage failures use
the [shared HTTP error mapping](api/rest.md#errors-and-limits). The health endpoint reports process
readiness.

## Isolation and data lifecycle

Each exact `user_id` maps to `users/<sha256(user_id)>`, a separate physical MindBridge directory.
Requests for the same user are serialized; different users can operate concurrently. The data
root has exclusive ownership, so a second process using it fails immediately. Hashing also keeps
external identifiers out of filesystem paths. No `user_id` field or scope is added to the product
SDK or `/v1` API.

The dedicated root also contains `sources/` for ordered source payloads and `requests/` for
payload fingerprints and completion receipts. Source files retain inline image data as well as
the SDK's stored assets, so account for this additional disk use. Back up or delete the entire
root together, including these adapter files. Access logging is disabled by the launcher.

Use evaluation data only for the evaluation, keep it private, and remove the dedicated root
within AML's required retention period. For open-source submissions, configure any Add-stage
model calls according to AML's current model requirements. These operational requirements are
defined by the linked AML guide; serving the protocol alone does not establish leaderboard
eligibility or score comparability.
