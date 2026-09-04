# MCP API

## Surface

The optional MCP adapter exposes fifteen typed tools over one injected synchronous `Memory`, or
fewer when the host withholds a group of them.
It validates tool input, calls the matching SDK operation, and returns structured public values.
It does not own storage, provider selection, or the injected memory. Finalized media arrives
through ordinary content parts; live audio and vision packet ingestion, and `StreamEvent`
reduction, stay Python-only because a tool call is a finite request.

The tools cover the common memory path, one context-compilation tool, and the embodied and
identity operations: an agent driving a companion robot can compile a budgeted context bundle for
its next turn, ask who spoke and who was seen, name them, reverse a wrong face-and-voice merge,
erase a person on request, and record which memories were useful. `build_mcp_server` runs
inside the process that holds the `Memory` it is given and every tool is one call on it, so an
embodied operation is no less reachable here than `add_memory` is.

Naming a person is host authority. `register_speaker` and `register_identity` assert a name on
the host's behalf: the assertion is a versioned memory record, it is written to the operation
log, and it is reversible, and the tool descriptions say so. Splitting a wrong merge is written
to the log and reversible too, and erasing a person leaves an irreversible row so the request
stays auditable; `get_identity` reads and writes nothing.

### Withholding a group of tools

Three keyword switches decide which authority an agent holds. Each defaults to True, and each
False withholds its group by never registering it, so calling one of its tools fails as an
unknown tool rather than as a refusal. The server instructions name the missing groups, because
an agent should not discover a boundary by failing.

| Switch | Withholds | Why a host would |
| --- | --- | --- |
| `identity_operations` | `register_speaker`, `register_identity`, `get_identity`, `unlink_identity`, `forget_identity` | Naming and erasing a person is host authority |
| `embodied_operations` | `analyze_speech`, `analyze_faces` | `analyze_faces` also commits the corroborated cross-modal identity merge, so leaving it registered keeps identity *binding* on the wire even with the identity tools withheld. `ask_memory` reaches the same merge through its own face recognition, so this switch also passes `link_identities=embodied_operations` into every `ask_memory` call |
| `write_operations` | `add_memory`, `delete_memory`, `reinforce_memories` | The agent should consume context and change nothing |

```python
build_mcp_server(
    memory,
    identity_operations=False,
    embodied_operations=False,
    write_operations=False,
)
```

With all three False the surface is exactly the five read tools -- `search_memories`,
`ask_memory`, `compile_context`, `get_memory` and `list_memories` -- which is recall and compile
alone. `ask_memory` stays because it is a recall surface; the reinforcement it records for the
memories its answer cited is the owner's `reinforce_on_answer` setting rather than a capability
these switches grant, so a host that wants a strictly read-only memory turns that off on the
`Memory` it injects. `embodied_operations=False` also withholds the one write `ask_memory` could
otherwise still reach: it passes `link_identities=False` into `Memory.ask`, so answering over a
photo or video may still identify who appears in it, but a corroborated voice-and-face pair is
never fused into one identity. Without that, "recall and compile alone" would be true of the tool
list and false of what `ask_memory` could do.

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
build_mcp_server(
    memory: Memory,
    *,
    identity_operations: bool = True,
    embodied_operations: bool = True,
    write_operations: bool = True,
) -> MCPServer[None]
```

`build_mcp_server` reads `memory.capabilities` once and publishes it as the server `instructions`,
so a connecting agent learns the configured modalities, embedding identity, and backends without
spending a tool call. There is no capabilities tool. The text also states that `compile_context` is
the preferred way to obtain context, and that cognitive forgetting, consolidation, and operation
rollback have no tool here. MCP fixes `instructions` at construction, so a server built before a
composition change keeps advertising the composition it was built with;
[`GET /healthz`](rest.md#endpoints) is the live reading.

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
| `add_memory` | required `content`; `occurred_at=None`; `occurred_end=None`; `metadata=None`; `memory_type="semantic"`; `context=None` | `MemoryResult` | write, idempotent |
| `search_memories` | required `query`; `limit=10`; `memory_type=None`; `reference_at=None`; `occurred_from=None`; `occurred_until=None`; `scope=None`; `explain=false` | `{"hits":[SearchHitResult,...],"trace":null}` | write, not idempotent |
| `ask_memory` | required `question`; `limit=5`; `memory_type=None`; `reference_at=None`; `scope=None` | `AnswerResponse` | write, not idempotent |
| `compile_context` | required `goal`; `budget=None`; `reference_at=None`; `scope=None` | `ContextBundleResult` | write, not idempotent |
| `get_memory` | required `memory_id` | `MemoryResult` | read-only |
| `list_memories` | `limit=100`; `cursor=None` | `PageResult` | read-only |
| `delete_memory` | required `memory_id` | `{"deleted":bool}` | destructive, idempotent |
| `reinforce_memories` | required `memory_ids`, 1 through 100 | `{"reinforced":int}` | write, not idempotent |
| `analyze_speech` | required `memory_id` | `{"segments":[SpeakerSegment,...]}` | write, not idempotent |
| `analyze_faces` | required `memory_id` | `{"observations":[FaceObservation,...]}` | write, not idempotent |
| `register_speaker` | required `speaker_id`, `name`; `relationship=None` | `{"registered":true}` | idempotent write |
| `register_identity` | required `identity_id`, `name`; `relationship=None` | `{"registered":true}` | idempotent write |
| `get_identity` | required `identity_id` | `{"identity":IdentityProfile\|null}` | read-only |
| `unlink_identity` | required `alias_id` | `{"restored_identity_id":str\|null}` | destructive, idempotent |
| `forget_identity` | required `identity_id` | `{"erasure":IdentityErasure}` | destructive, not idempotent |

All timestamps must be timezone-aware. An event end requires a start and must be later than it.
Search event bounds are a half-open overlap filter; two bounds require
`occurred_until > occurred_from`. `memory_type` is `semantic`, `episodic`, or `procedural`.
Pagination cursors are opaque and must be passed back unchanged.

`context` carries typed observation basis, source ID, confidence, validity, optional spatial pose,
and optional symbolic `place_id`. `scope.valid_at` selects world validity and `scope.known_at`
selects the transaction version known then; `scope.place_id` matches the stored label exactly;
`scope.near` and `scope.radius_m` must appear together, and their frame ID and observer/subject
anchor must match the stored spatial context. SQLite reapplies every filter after candidate
retrieval.

`add_memory` is content-addressed. `delete_memory` reports whether a record existed. Search,
answer, compile, and the two analysis tools are not marked read-only because their SDK path
persists lazy transcript caches, identity evidence, and -- when a recall comes back with nothing,
an answer abstains, or a bundle carries no evidence -- the bounded recall-failure signal the
control plane's `QUERY_FAILURE` trigger reads; they are also not advertised as idempotent. Every tool has `open_world_hint=false`. `ask_memory` requires an answerer in the
injected memory; without one it returns `model_error/backend_not_configured`. With the default
`reinforce_on_answer=True`, it also reinforces the hits the answerer cites.

`compile_context` is the preferred way to get task-ready context: it returns a structured,
budgeted bundle with provenance instead of one sentence, calls no generation model, and stores no
memory. It shares the retrieval path with `search_memories`, including the two writes both can
make -- a cached transcript for spoken query media, and a recorded recall failure when the bundle
holds no evidence -- which is why neither is annotated read-only. It
reports lineage conflicts and never resolves them, and names in `unknowns` what the request
implied that the bundle does not carry. `ask_memory` remains a convenience for one grounded
answer. The optional `budget` object is the transport form of `ContextBudget` with the
`freshness` timedelta expressed as `freshness_seconds`; `max_chars`
accepts 1 through 65,536, `max_items` 1 through 100, `max_media_items` any non-negative count
or `null`, and `max_latency_ms` is a deadline the
compiler checks between stages rather than a timeout that aborts. The
[compiler reference](../context-compilation.md) owns section, selection, and conflict semantics.

The embodied and identity tools follow the SDK operation they dispatch to:

- `analyze_speech` and `analyze_faces` read one stored memory's assets. Results cover that
  memory's audio and video, or image and video, assets and are not paged, because the bound is
  the stored media rather than a caller limit. A memory with no matching asset returns an empty
  result instead of failing. When matching media exists, the operation needs a speech-capable
  transcription backend or a face backend and returns `model_error` when it is unavailable.
- `register_speaker` names a recognized voice and `register_identity` names an identity that may
  also have been seen. Both replace an existing name and both leave a recorded `relationship`
  intact when the argument is omitted; there is deliberately no way to clear one. An unknown ID
  returns `speaker_not_found` or `identity_not_found`.
- `get_identity` follows merge aliases and returns `identity: null` when no registered profile is
  found, including for an unknown ID.
- `unlink_identity` reverses one face-and-voice merge and returns the restored ID, or `null` when
  no record names which modality was contributed. It resets that pair's accumulated evidence and
  does not suppress the pair: a voice and face that keep co-occurring are merged again.
- `forget_identity` erases a person -- biometric templates, aliases, and the indexed name -- and
  returns counts of what was destroyed as the audit record. Memories and media survive and a
  transcript keeps its words with the speaker attribution dropped. It cannot be undone, and the
  second call reports `identity_not_found` because the person is already gone, which is why it is
  the one tool advertised as destructive and not idempotent.
- `reinforce_memories` records cumulative positive feedback: each call raises `access_count` for
  the named memories and moves the ranker's reinforcement factor, so it is published with
  `idempotent_hint=false` and a lost response must not be retried blindly. Duplicate IDs count
  once and unknown IDs are skipped, so `reinforced` below the number sent means the rest do not
  exist.

Tool descriptions are published through `inspect.cleandoc`, so the text an agent reads is
identical on every supported interpreter. CPython 3.13 strips a docstring's common indentation at
compile time and earlier versions do not, and the description is public contract.

Every tool and every tool argument carries a published description. Read those before guessing at
coordinate frames, units, or media sources; they are part of the tool schema, not a separate guide.

### Retrieval trace

`search_memories` with `explain=true` routes to `Memory.search_with_trace` and adds a `trace`
object beside the unchanged `hits`. `trace.candidates` lists every candidate that was considered
with its effective score components (`dense_relevance`, `dense_confidence`, `lexical_relevance`,
`lexical_rerank_bonus`, `lexical_match`, `gate_relevance`, `base_relevance`,
`reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `rank`) and, when it
did not become a hit, a `rejected_by` value of `stale_index`, `occurrence_range`, `missing_memory`,
`memory_type`, `minimum_relevance`, `ambiguity`, or `limit`. `trace.candidate_limit` is how many
candidates were fetched, `trace.exhaustive` says whether that bound was reached, and
`trace.ambiguous` says whether the result was suppressed for being too close to call. Without
`explain`, `trace` is `null` and no extra work is done.

A `SearchHitResult.score` is `final_score`, but the relevance gate compares `gate_relevance`, which
is a different quantity. Tuning a floor against the returned `score` therefore compares the wrong
two numbers. The floor itself, `minimum_relevance`, and `ambiguity_margin` are fixed when the owner
constructs `Memory`; no tool argument can widen them for one call, so an empty result is answered by
reading `trace` and changing the query, filters, or the owner's configuration.

### Result objects

Successful calls populate MCP `structuredContent`:

| Object | Fields |
| --- | --- |
| `AssetResult` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name` |
| `MemoryResult` | `id`, `content`, `modality`, `memory_type`, `assets`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `context`, `place_id`, `forgotten_at` |
| `SearchHitResult` | all memory fields plus `score` |
| `SearchResult` | `hits`, `trace`; `trace` is `null` unless `explain=true` |
| `ReinforceResult` | `reinforced` |
| `AnswerResponse` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `PageResult` | `items`, `next_cursor` |
| `ContextBudgetResult` | `max_chars`, `max_items`, `max_media_items` or `null`, `memory_types` or `null`, `min_confidence`, `freshness_seconds`, `max_latency_ms` |
| `ContextConflictResult` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextUnknownResult` | `kind`, `detail` |
| `ContextBundleResult` | `goal`, `reference_at`, `budget`, the `SearchHitResult` arrays `relationships`, `scene`, `episodes`, `facts`, `procedures`, `affect`, `traits`, the mixed `actors` array of `SearchHitResult` and `ProvisionalActorResult`, plus `conflicts`, `unknowns`, `occurred_from`, `occurred_until`, `frames`, `places`, `omitted`, `chars`, `elapsed_ms`, `deadline_exceeded`, `rendered` |
| `ProvisionalActorResult` | `identity_id`, `memory_ids` |
| `SpeakerSegment` | `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `FaceObservation` | `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `IdentityProfile` | `identity_id`, `name`, `relationship`, `confirmed`, `evidence_ids` |
| `IdentityErasure` | `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`, `face_observations`, `speech_segments` |

The last four are the SDK dataclasses `Memory.speech`, `Memory.faces`, `Memory.identity`, and
`Memory.forget_identity` return, published field for field rather than reshaped:
`bounding_box` is `[x, y, width, height]` normalized within the frame, `identity_score` and
`speaker_name` are `null` until an identity is resolved and named, and `observed_at_ms` is the
offset within a video. A `speaker_id` or `identity_id` from either analysis tool is accepted by
`get_identity`, `register_speaker`, `register_identity`, `unlink_identity`, and
`forget_identity`.

The other fields have the same meanings as the [REST response objects](rest.md#response-objects).
Successful result objects never serialize filesystem paths; error `subject` is the exception
described above.

## Errors and limits

### Validation and errors

Only the documented names and their exact top-level arguments are accepted. An unknown
tool or top-level argument returns `validation_error/unknown_field`; schema and SDK input failures return
`validation_error/input_invalid`. Unknown values are not echoed.

Failed tool calls set `isError` and their text content is exactly the same JSON envelope as
[REST](rest.md#error-envelope), with nothing before or after it. One `JSON.parse` of the text
succeeds for every failure, including `memory_not_found`, `model_error`, and `storage_error`:

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

Twenty Python operations have no MCP tool. One is a transport limitation and the rest are
decisions; none is withheld because it touches owner-process state, since every tool already
does.

| Operation | Why, and what to call instead |
| --- | --- |
| `add_stream` | **Transport limitation.** It consumes a lazy iterable and yields records as it goes, and one tool call is a finite request with one response, so a stream cannot be started, fed, and drained through it. Call `add_memory` for each completed observation, or use the SDK for a live source. |
| `add_many` | Every item is already reachable through `add_memory`, so this buys one model batch rather than a capability. Its parallel arrays must line up with `contents` position by position, and a misalignment stores real memories under the wrong timestamps, which is a worse failure than slower ingestion. Use `POST /v1/memories/batch` for bulk loading. |
| `search_with_trace` | No tool of its own, because the same trace is reachable from the tool that produces the hits: call `search_memories` with `explain=true`. Use the SDK or the CLI when diagnosing retrieval outside an agent loop. |
| `reindex` | Rebuilds the whole search projection from SQLite. The duration grows with the store and has no upper bound, so it does not belong behind a client that expects one timely response. It is also an operator decision, not a caller's. |
| `optimize` | Merges staged vectors into the index. An agent has no basis for deciding when that is worth doing, and the operator scheduling it has the CLI. |
| `capture`, `settle`, `pending_captures` | Deferred enrichment is the owning process's scheduling decision: how long a record may stay unsearchable is a property of that host's loop, not of a caller's request. Use `add_memory`, which returns searchable. |
| `record_consent`, `consent` | Consent is a statement its subject makes, relayed by the host. An agent that could record one could manufacture permission to process a person, which is the same reason a `CONSENT` operation proposed by a model is refused as `unauthorized`. A host application collects the statement and calls the SDK, or `POST /v1/identities/{identity_id}/consent` behind its own `identity_operations` switch. |
| `export`, `apply_retention` | The two remaining data-subject rights, and both are host authority over a person's whole record: an export is everything held about somebody, and retention physically deletes. They stay with the process that owns the `data_dir`, and reach a network only through REST's `identity_operations` switch. |
| `consolidation_candidates`, `consolidate`, `deliberate`, `apply`, `record_outcome`, `forget`, `rollback`, `operations` | The memory control plane rewrites and retires derived memory and is deliberately not reachable from a client. It stays with the process that owns the `Memory`, which is also the process that can audit and reverse it through the operation log. `deliberate` additionally spends that owner's model budget on its own schedule, and `apply` applies an operation with no proposal behind it, which is host authority itself. |

`Memory.capabilities` has no tool either: it is a property rather than an operation, published
here as the server `instructions` and by REST in [`GET /healthz`](rest.md#endpoints). The tools
that need a backend say so in their description and return `model_error` when it is missing.

`list_memories` supports the same default page size and opaque cursor contract as `Memory.list`.

### Input limits

| Bound | MCP value |
| --- | --- |
| Content parts | 1 through 16 |
| One data-URL source string | 8,192 characters |
| One decoded `file_data` value | 8 MiB |
| Normalized text, including combined text parts | 65,536 characters |
| Search, answer, or page `limit` | 1 through 100 |
| Context `budget.max_chars` | 1 through 65,536 |
| Context `budget.max_items` | 1 through 100 |
| Context `budget.max_media_items` | 0 or more, or `null` |
| Serialized metadata | 262,144 UTF-8 bytes |
| `file_id` or `filename` | 255 characters |

MCP has no aggregate framing budget, but each inline media value is bounded before model or storage
work. It has no local-path input, remote fetch, large-file upload tool, capture-stream tool,
coordinate-frame transform, logical scope, or separate authentication policy; the MCP host owns
transport access control. Configured model backends may
impose a smaller aggregate budget, including the [OpenAI inline limits](python-sdk.md#bundled-adapters).
