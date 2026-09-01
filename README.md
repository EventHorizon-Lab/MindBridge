# MindBridge

MindBridge is a local-first, agent-native multimodal memory substrate for embodied systems.

MindBridge owns three things: memory semantics, retrieval orchestration, and durable local
consistency. Applications may select bundled adapters with validated configuration or inject
custom adapters directly. Provider SDKs and deployment infrastructure still own credentials,
network transport, retries, timeouts, and authentication.

SQLite is authoritative for records, embeddings, metadata, and the search-index outbox. Media is
stored in a local content-addressed store, and Zvec is a rebuildable search projection. One
physical `data_dir` has exactly one live `Memory` owner.

## Install

Install the local Jina and FunASR adapters, plus the official OpenAI SDK when generation or ASR uses
OpenAI:

```bash
uv add "mindbridge[local,openai]"
```

Add `mindbridge[face]` only when local face identity is required; model weights remain explicit
caller-provided files.

Install every optional adapter, protocol, and benchmark dependency only when a complete development
environment is required:

```bash
uv add "mindbridge[all]"
```

The base package depends only on Pydantic and Zvec. Torch, provider SDKs, REST, and MCP remain in
optional extras.

`JinaOmniEmbedder` pins Jina v5 Omni, whose weights are licensed **CC BY-NC 4.0 — non-commercial
use only**. That licence covers the model, not MindBridge. Inject another embedding backend when it
does not fit the application; see
[choose an embedding backend](docs/quickstart.md#choose-an-embedding-backend) for a pinned
Apache-2.0 alternative.

## Search local memory

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    memory.add("Ada prefers concise status updates.")
    hits = memory.search("How should I write Ada's update?")
    for hit in hits:
        print(hit.score, hit.content)
```

`Memory.from_config` resolves a small, typed catalog of bundled adapters. The `Memory` kernel still
routes only by capability and never reads provider credentials.

`search` returns a possibly empty tuple. MindBridge always withholds weak evidence and withholds an
unresolved top-two tie when `limit=1`; multi-result search returns the qualified candidates for the
caller to handle. `ask` applies the same rule to evidence retrieval. See
[search can legitimately return nothing](docs/quickstart.md#search-can-legitimately-return-nothing).

## Add grounded answers

The declarative path constructs and closes the official SDK client; the SDK reads its standard
credential environment variables:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
        "generation": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "temperature": 0.1,
        },
    }
) as memory:
    memory.add("The release review is Thursday at 10:00 UTC.")
    print(memory.ask("When is the release review?").answer)
```

Inject `OpenAIModels` directly when the application must supply an existing client, proxy, custom
credential source, or capabilities beyond the declarative schema. Caller-supplied clients remain
caller-owned.

## Content contract

`add`, `add_stream`, `search`, and `ask` use the same atom or ordered sequence contract:

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

For continuous input, pass completed application-owned chunks lazily. Each yielded record is
already durable and searchable:

```python
from mindbridge import MemoryType, StreamInput

for record in memory.add_stream(
    StreamInput(path, metadata={"sequence": index}, memory_type=MemoryType.EPISODIC)
    for index, path in enumerate(completed_clip_paths)
):
    handle(record)
```

MindBridge does not open cameras or microphones. Applications may feed canonical audio packets to
`AsyncAudioStream`, or immutable encoded `VisionFrame`, `VisionPartial`, and `SceneBoundary`
values to `AsyncVisionStream`. `AsyncCaptureStream` associates interleaved `StreamEvent` values by
`stream_id`, coalesces speculative retrieval independently, and returns the same ID on each exact
final commit. See [omni streaming and interaction memory](docs/omni-streaming-and-interaction-memory.md).

## Model boundaries

MindBridge exposes narrow, operation-specific protocols:

- `EmbeddingBackend` for retrieval vectors and stable embedding-space identity.
- `GenerationBackend` for grounded answers.
- `TranscriptionBackend` for plain transcripts.
- `SpeechBackend` for timed transcripts and local voice exemplars.
- `VisionDescriptionBackend` for caption/OCR/detector text used by visual fallback.
- `FaceBackend` for local face observations and identity exemplars.
- `FormationBackend` for typed, source-grounded semantic proposals after observation commit.

A single provider adapter may implement several protocols, but `Memory` does not require one fat
backend. The declarative catalog constructs only bundled adapters; there is no global plugin
registry, credential store, retry layer, or sync/async provider compatibility layer.

The built-in adapters are deliberately thin:

- `SentenceTransformersEmbedder` uses the installed Sentence Transformers runtime.
- `JinaOmniEmbedder` pins Jina v5 Omni and calls its official `encode_query` and
  `encode_document` methods.
- `FunASRTranscriber` delegates model loading and execution to `funasr.AutoModel`.
- `OpenCVFaceAnalyzer` delegates explicit local YuNet and SFace models to OpenCV.
- `OpenAIModels` uses caller-owned official OpenAI SDK clients for embeddings, generation, typed
  formation, and completed-file transcription.

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
