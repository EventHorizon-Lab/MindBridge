# MindBridge

Fast, embedded multimodal memory for Python agents.

MindBridge owns three things: memory semantics, retrieval orchestration, and durable local
consistency. Applications construct model adapters explicitly. Provider SDKs and deployment
infrastructure own credentials, network transport, retries, timeouts, and authentication.

SQLite is authoritative for records, embeddings, metadata, and the search-index outbox. Media is
stored in a local content-addressed store, and Zvec is a rebuildable search projection. One
physical `data_dir` has exactly one live `Memory` owner.

## Install

Install the local Jina and FunASR adapters, plus the official OpenAI SDK when grounded answers use
OpenAI:

```bash
uv add "mindbridge[local,openai]"
```

The base package depends only on Pydantic and Zvec. Torch, provider SDKs, REST, and MCP remain in
optional extras.

## Search local memory

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    memory.add("Ada prefers concise status updates.")
    hits = memory.search("How should I write Ada's update?")
    print(hits[0].content)
```

`Memory` never selects a provider or reads provider credentials.

## Add grounded answers

Construct and own the provider client in application code:

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels

with OpenAI() as client:
    answerer = OpenAIModels(
        generation_client=client,
        generation_model="gpt-5-mini",
    )
    with Memory(
        "./data/assistant",
        embedder=JinaOmniEmbedder(),
        answerer=answerer,
    ) as memory:
        memory.add("The release review is Thursday at 10:00 UTC.")
        print(memory.ask("When is the release review?").answer)
```

`OpenAIModels` maps MindBridge model inputs to the official SDK. It does not create or close SDK
clients. Configure API keys, base URLs, proxies, retries, and timeouts with `OpenAI(...)` or the
provider's own SDK.

## Content contract

`add`, `search`, and `ask` accept one atom or an ordered sequence of atoms:

| Value | Meaning |
| --- | --- |
| `str` | Application text |
| `pathlib.Path` | Local image, video, or audio copied into the CAS |
| `Blob` | In-memory media bytes with a concrete MIME type |
| `AssetRef` | Media already stored in the same directory |

Remote URLs are not a product input. Fetch remote content with the application's HTTP stack and
pass a `Blob` or local `Path`. REST and MCP accept base64 data URLs and stored asset IDs, but never
fetch network URLs or read arbitrary server paths.

```python
from pathlib import Path

from mindbridge import Blob, JinaOmniEmbedder, Memory

with Memory("./data/media", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add(
        (
            "Design review evidence",
            Path("./whiteboard.png"),
            Blob(Path("./summary.wav").read_bytes(), "audio/wav", "summary.wav"),
        )
    )
    print(record.modality)
```

Use `FunASRTranscriber()` explicitly when audio fallback, timed speech, or speaker matching is
needed. Unsupported modalities fail before inference; model support is declared by each adapter.

## Model boundaries

MindBridge exposes narrow, operation-specific protocols:

- `EmbeddingBackend` for retrieval vectors and stable embedding-space identity.
- `GenerationBackend` for grounded answers.
- `TranscriptionBackend` for plain transcripts.
- `SpeechBackend` for timed transcripts and local speaker centroids.

A single provider adapter may implement several protocols, but `Memory` does not require one fat
backend. There is no provider registry, endpoint normalizer, credential store, retry layer, or
sync/async provider compatibility layer.

The built-in adapters are deliberately thin:

- `SentenceTransformersEmbedder` uses the installed Sentence Transformers runtime.
- `JinaOmniEmbedder` pins Jina v5 Omni and calls its official `encode_query` and
  `encode_document` methods.
- `FunASRTranscriber` delegates model loading and execution to `funasr.AutoModel`.
- `OpenAIModels` uses caller-owned official OpenAI SDK clients.

## Storage and consistency

```text
data_dir/
├── state.sqlite3       # authoritative records, embeddings, metadata, outbox
├── assets/             # immutable content-addressed media
├── .mindbridge.lock    # exclusive process ownership
└── zvec/               # disposable search projection
```

A durable write commits SQLite before updating Zvec. Outbox work is acknowledged only after the
Zvec flush succeeds. Missing or stale index data is rebuilt and hydrated from SQLite without
re-embedding stored content.

## REST and MCP adapters

Both adapters require a caller-constructed memory:

```python
from mindbridge.api import create_app

app = create_app(memory=memory)
```

```python
from mindbridge.api.mcp import build_mcp_server

build_mcp_server(memory).run("stdio")
```

They do not own or close `memory`. `create_app` is unauthenticated; put it behind the application's
gateway or ASGI middleware. There is no generic product CLI because provider construction and
lifecycle belong to the host application.

## Benchmarks

The only console command is the benchmark dispatcher:

```bash
mindbridge-bench --help
mindbridge-bench eval --tasks list
mindbridge-bench local-index --help
```

Every benchmark unit receives a separate physical directory. See
[the benchmark guide](docs/benchmarking.md).

## Documentation

- [Quick start](docs/quickstart.md)
- [Python API](docs/api/python-sdk.md)
- [Configuration and composition](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [REST](docs/api/rest.md) and [MCP](docs/api/mcp.md)
- [Deployment](docs/deployment.md) and [operations](docs/operations.md)

The pinned default Jina model weights are licensed CC BY-NC 4.0. Inject another embedding backend
when that license does not fit the application.
