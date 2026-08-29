# Architecture

MindBridge is an embedded consistency core surrounded by caller-owned model and transport stacks.

```text
Application
├── provider SDK clients
├── authentication / gateway / HTTP client
└── Memory
    ├── memory semantics and modality routing
    ├── retrieval and grounded-answer orchestration
    ├── SQLite + media CAS (authoritative)
    └── Zvec (derived projection)
```

## Responsibility boundary

MindBridge owns:

- Canonical content and memory types.
- Stable IDs, event time, metadata, and local speaker identity.
- Capability-aware embedding, transcription fallback, retrieval, and grounding.
- SQLite/CAS durability and the SQLite-to-Zvec outbox.
- Validation and sanitization at Python, REST, and MCP boundaries.

MindBridge does not own:

- Provider credential discovery or rotation.
- HTTP connection pools, proxies, retries, timeouts, or endpoint normalization.
- Compatibility registries for model services.
- REST identity, authorization, TLS, quotas, or audit logging.
- Remote URL downloading.
- A second model runtime when Sentence Transformers, FunASR, or a provider SDK already supplies it.

## Model composition

`Memory` depends on narrow operation contracts rather than a combined provider abstraction.

```text
Content ──> EmbeddingBackend ──> retrieval vector
Question + hits ──> GenerationBackend ──> grounded answer
Audio/video ──> TranscriptionBackend ──> fallback text
Audio/video ──> SpeechBackend ──> timed turns + speakers
```

One adapter may implement several contracts. `OpenAIModels` does so with caller-owned official SDK
clients. Local embedding and speech can be composed independently.

No registry or factory is needed: ordinary Python construction is the provider selection
mechanism.

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

SQLite commits before Zvec changes. A failed index operation remains in the outbox and is replayed
when the owner recovers. Zvec never becomes authoritative.

## Read consistency

MindBridge embeds a query, asks Zvec for candidates, hydrates those IDs from one SQLite snapshot,
drops stale IDs, and reranks the surviving records. A missing index is rebuilt from stored FP32
embeddings without re-embedding content.

## Isolation and concurrency

One physical directory is one memory domain and one live owner. There are no account, tenant,
request, or benchmark scope identifiers in the product API. Metadata is application data.

Provider work may overlap across calls. SQLite write/outbox and Zvec sections are short serialized
critical sections. `close()` waits for active operations before closing adapters and storage.

`AsyncMemory` uses threads around this synchronous embedded core. Provider-specific async APIs are
not normalized by MindBridge; a custom adapter can use the provider's native client where its
contract permits.

## Optional protocol adapters

`create_app(memory=...)` and `build_mcp_server(memory)` expose an existing memory. They never
construct providers or own the memory lifecycle. FastAPI and MCP own sync/async request dispatch;
deployment infrastructure supplies auth and transport policy.
