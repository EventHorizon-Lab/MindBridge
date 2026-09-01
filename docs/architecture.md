# Architecture

MindBridge is an embedded consistency core surrounded by caller-owned model and transport stacks.

```text
Application
├── provider SDK clients
├── authentication / gateway / HTTP client
└── Memory
    ├── memory semantics and modality routing
    ├── optional typed formation and bitemporal evidence projection
    ├── retrieval and grounded-answer orchestration
    ├── SQLite + media CAS (authoritative)
    └── Zvec (derived projection)
```

## Shared execution plane

`Memory` is the canonical execution plane and the Python SDK exposes it directly. REST, MCP, and the
product CLI are interfaces over the same application-composed instance, not separate
implementations of memory behavior.

```text
Developer / Agent
├── Python SDK ─────────────────────────┐
├── REST adapter ───────────────────────┤
├── MCP adapter ────────────────────────┤
└── product CLI ─────────────────────────┤
                                        v
                         application-composed Memory
                   validation · routing · retrieval · consistency
                                        |
                           models · SQLite/CAS · Zvec
```

Interface code may decode transport values, call a public operation, and encode its result. It must
not own a second provider configuration, modality router, retrieval pipeline, persistence path, or
error taxonomy. SDK behavior is the capability baseline; MCP and CLI schemas project the same
operations with transport-appropriate types and side-effect annotations.

One execution plane does not allow several processes to open one `data_dir`. An embedded interface
calls its process-owned `Memory`. A CLI addressing an already running owner must reach that owner
through a supported transport rather than opening the directory again. In both cases, the operation
executes exactly once in the owning `Memory`.

The current release follows this rule for Python, REST, MCP, and the CLI. The CLI composes one
`Memory` per invocation with `--app` or `--embedder`, and addresses a directory another process
already owns with `--url`. See [command-line usage](api/cli.md).

## Responsibility boundary

MindBridge owns:

- Canonical content and memory types.
- Stable IDs, event time, typed validity/provenance, spatial context, and local face-and-voice
  identity.
- Formation validation, evidence independence, state supersession, and transaction-time history.
- Capability-aware embedding, transcription fallback, retrieval, and grounding.
- SQLite/CAS durability and the SQLite-to-Zvec outbox.
- Validation and sanitization at Python, REST, and MCP boundaries.

MindBridge does not own:

- Provider credential discovery or rotation.
- HTTP connection pools, proxies, retries, timeouts, or endpoint normalization.
- Compatibility registries for model services.
- REST identity, authorization, TLS, quotas, or audit logging.
- Remote URL downloading.
- Sensor capture, stream reconnection, observation segmentation, or turn detection.
- A second model runtime when Sentence Transformers, FunASR, or a provider SDK already supplies it.

## Model composition

`Memory` depends on narrow operation contracts rather than a combined provider abstraction.

```text
Content ──> EmbeddingBackend ──> retrieval vector
Committed observation ──> FormationBackend ──> typed source-grounded proposals
Question + hits ──> GenerationBackend ──> grounded answer
Audio/video ──> TranscriptionBackend ──> fallback text
Audio/video ──> SpeechBackend ──> timed turns + speakers
Image/video ──> FaceBackend ──> bounded face observations
```

One adapter may implement several contracts. `OpenAIModels` does so with official SDK clients. Local
embedding, speech, and face analysis can be composed independently. `Memory.from_config` resolves a
closed catalog of bundled adapters; direct construction accepts custom adapter objects. Both
converge on `Memory.from_plugins` and the same execution plane. Provider names never reach the
kernel, and there is no global runtime plugin registry.

Formation does not make the model authoritative. The source observation commits first. The kernel
validates proposals and then atomically writes derived records, evidence edges, bitemporal versions,
embeddings, a durable formation-recipe marker, and index-outbox work. Model failure leaves the raw
observation durable and retryable.

## Write consistency

```text
normalize -> materialize CAS -> model work -> SQLite transaction
                                              |
                                              v
                                      durable index outbox
                                              |
                                              v
                                      Zvec mutate + flush
                                              |
                                              v
                                      acknowledge outbox
```

SQLite commits before Zvec changes. Concurrent record commits may accumulate one outbox batch;
Zvec mutation, flush, and acknowledgement remain serialized. A failed index operation stays in the
outbox and is replayed when the owner recovers. Zvec never becomes authoritative.

## Read consistency

MindBridge batches each complete ordered query with bounded focused aggregate and atomic keys from
its first text atom and media, then asks Zvec for dense and lexical candidates concurrently. Dense
search matches aggregate and atomic document embeddings and groups them by parent memory inside
Zvec, with a bounded ordinary-query fallback when best-effort grouping is incomplete. SQLite then
hydrates those IDs, drops stale IDs, and collapses the remaining derived
keys before reranking. A missing index is rebuilt from stored FP32 embeddings without re-embedding
content.

## Isolation and concurrency

One physical directory is one memory domain and one live owner. There are no account, tenant,
request, or benchmark scope identifiers in the product API. Metadata is application data.

Provider work, independent SQLite record commits, and Zvec queries may overlap across calls. Outbox
replay and final hydration/asset leasing are short serialized critical sections; collection
replacement is exclusive. `close()` waits for active operations before closing adapters and
storage.

`AsyncMemory` uses threads around this synchronous embedded core. Provider-specific async APIs are
not normalized by MindBridge; a custom adapter can use the provider's native client where its
contract permits.

`add_stream` preserves the same write lifecycle by invoking the ordinary add path once per
completed observation. `AsyncOmniPrefetch` serializes speculative searches for one turn, replaces
queued snapshots instead of cancelling synchronous work already running in a thread, and confirms
the exact final snapshot before returning.

## Protocol interfaces

`create_app(memory=...)` and `build_mcp_server(memory)` expose an existing memory. They never
construct providers or own the memory lifecycle. FastAPI and MCP own sync/async request dispatch;
deployment infrastructure supplies auth and transport policy. The product CLI follows the same
boundary: it reuses the shared execution plane and owns only command decoding, output formatting,
and process lifecycle.
