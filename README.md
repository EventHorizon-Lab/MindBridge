# MindBridge

MindBridge is an Agentic Native Embodied Memory System: fast, embedded multimodal memory for
machines that see, hear, and act.

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

Add `mindbridge[face]` only when local face identity is required; model weights remain explicit
caller-provided files.

The base package depends only on Pydantic and Zvec. Torch, provider SDKs, REST, and MCP remain in
optional extras.

`JinaOmniEmbedder` pins Jina v5 Omni, whose weights are licensed **CC BY-NC 4.0 — non-commercial
use only**. That licence covers the model, not MindBridge. Inject another embedding backend when it
does not fit the application; see
[choose an embedding backend](docs/quickstart.md#choose-an-embedding-backend) for a pinned
Apache-2.0 alternative.

## Search local memory

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    memory.add("Ada prefers concise status updates.")
    hits = memory.search("How should I write Ada's update?")
    for hit in hits:
        print(hit.score, hit.content)
```

`Memory` never selects a provider or reads provider credentials.

`search` returns a possibly empty tuple. MindBridge withholds weak or ambiguous evidence rather
than always returning rows, so callers must handle the empty case; see
[search can legitimately return nothing](docs/quickstart.md#search-can-legitimately-return-nothing).

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
- `SpeechBackend` for timed transcripts and local voice exemplars.
- `FaceBackend` for local face observations and identity exemplars.

A single provider adapter may implement several protocols, but `Memory` does not require one fat
backend. There is no provider registry, endpoint normalizer, credential store, retry layer, or
sync/async provider compatibility layer.

The built-in adapters are deliberately thin:

- `SentenceTransformersEmbedder` uses the installed Sentence Transformers runtime.
- `JinaOmniEmbedder` pins Jina v5 Omni and calls its official `encode_query` and
  `encode_document` methods.
- `FunASRTranscriber` delegates model loading and execution to `funasr.AutoModel`.
- `OpenCVFaceAnalyzer` delegates explicit local YuNet and SFace models to OpenCV.
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
re-embedding stored content. Concurrent record commits may share one serialized outbox flush.

## Shared execution plane

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
gateway or ASGI middleware. The Python SDK, REST, and MCP therefore share one execution plane rather
than implementing memory behavior separately.

The `mindbridge` command is a third interface over the same plane, not a second one:

```bash
mindbridge --embedder jina-omni add "The spare key is in the blue toolbox."
mindbridge --embedder jina-omni search "where is the spare key"
mindbridge --app my_application:build_memory ask "where is the spare key"
mindbridge --url http://127.0.0.1:8000 search "where is the spare key"
```

Exactly one of `--app MODULE:ATTR`, `--embedder NAME`, and `--url URL` composes the run; there is no
default and no environment variable that selects a backend. `--app` is the same
application composition the SDK and REST use, and it is how any non-bundled backend is reached.
`--embedder` names an entry in the closed `mindbridge.recipes` table, which returns the object it
built. `--url` addresses a running owner rather than opening a directory a second time.

Data is JSON on stdout, diagnostics are JSON on stderr, and exit codes are stable — exit `9` means
another process owns the directory, so retry with `--url`. `mindbridge doctor` resolves the
composition and exercises each backend's loader before the first write. Because each invocation
opens and closes one `Memory`, the CLI is not the right tool for a loop; use the SDK or `--url`. See
[command-line usage](docs/api/cli.md).

## Benchmarks

The benchmark dispatcher is the second console command:

```bash
mindbridge-bench --help
mindbridge-bench eval --tasks list
mindbridge-bench local-index --help
```

Every benchmark unit receives a separate physical directory. See
[the benchmark guide](docs/benchmarking.md).

## Documentation

- [Product goals and design principles](docs/design-principles.md)
- [Quick start](docs/quickstart.md)
- [Python API](docs/api/python-sdk.md)
- [MCP API](docs/api/mcp.md) and [command-line usage](docs/api/cli.md)
- [Configuration and composition](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [REST API](docs/api/rest.md)
- [Deployment](docs/deployment.md) and [operations](docs/operations.md)
