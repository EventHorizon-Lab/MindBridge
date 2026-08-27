# MindBridge

Fast, local multimodal memory for Python agents.

MindBridge gives an application one small `Memory` object for durable text, image, video, and
audio memory, dense/hybrid retrieval, and grounded answers. SQLite and a content-addressed store
are authoritative; Zvec is a rebuildable local search index. There is no database service, queue
worker, object store, or account hierarchy to configure.

## Quick start

Install MindBridge with its default local embedding and speech runtime:

```bash
uv add "mindbridge[local]"
```

The base `mindbridge` package stays small for applications that inject model backends. The local
extra includes Jina v5 Omni embedding and FunASR speech analysis. Set an OpenAI-compatible
credential only when using the default grounded-answer backend:

```bash
export OPENAI_API_KEY="your-api-key"
```

Add `face` when the application calls `Memory.faces`; it installs InsightFace, ONNX Runtime, and
OpenCV without adding them to the base SDK:

```bash
uv add "mindbridge[local,face]"
```

A text-only use case stays a three-line flow; it is one native route, not a product limitation:

```python
from mindbridge import Memory

with Memory() as memory:
    memory.add("Ada prefers concise status updates.")
    hits = memory.search("How should I write Ada's update?")
    answer = memory.ask("How should I write Ada's update?")

    print(hits[0].content)
    print(answer.answer)
```

The same methods accept local paths, HTTPS sources, inline bytes, stored asset references, and
ordered combinations. Jina v5 Omni is the default embedding model and accepts all four atomic
modalities. Fun-ASR-Nano is the default ASR, with FSMN-VAD and CAM++ speaker recognition. For
grounded answers, declare the modalities accepted by the generator, then allow the remote media
host:

```bash
export MINDBRIDGE_GENERATION_MODALITIES=text,image,video
export MINDBRIDGE_ALLOWED_URL_HOSTS=media.example
```

```python
from pathlib import Path

from mindbridge import Blob, Memory, URL

with Memory(data_dir="./data/assistant-memory") as memory:
    record = memory.add(
        [
            "The whiteboard after the design review",
            Path("./whiteboard.png"),
            Blob(
                Path("./summary.wav").read_bytes(),
                media_type="audio/wav",
                name="summary.wav",
            ),
            URL("https://media.example/demo.mp4", media_type="video/mp4"),
        ],
        metadata={"source": "design-review"},
    )
    print(record.modality)  # Modality.OMNI
    print(record.assets)
```

`Memory()` stores data in `.mindbridge/`. Applications, tests, and benchmarks should pass an
explicit directory. One directory belongs to exactly one running `Memory` instance; a second
owner fails immediately.

## One content contract

Python accepts these content atoms:

| Value | Use |
| --- | --- |
| `str` | Text |
| `pathlib.Path` | A local image, video, or audio file copied into local storage |
| `URL` | An explicitly typed HTTPS media source |
| `Blob` | Inline media bytes with a concrete MIME type |
| `AssetRef` | Reuse an asset already stored in the same `data_dir` |

Pass one atom or an ordered sequence of atoms to `add`, `search`, or `ask`. MindBridge reports
`text`, `image`, `video`, or `audio` when an input has no media or one media family, and `omni`
when it combines two or more media families. Text alongside one media family keeps that media
modality.

REST and MCP use the equivalent OpenAI-compatible content parts: `input_text`, `input_image`, and
`input_file`. Local filesystem paths are intentionally Python-only.

Identical media bytes share one CAS descriptor. Its first authoritative non-empty name is reused
across memories; use record text or metadata when a label must differ per memory.

## Storage and retrieval

```text
data_dir/
├── state.sqlite3       # records, asset metadata, FP32 embeddings, and index outbox
├── assets/             # immutable content-addressed media bytes
├── .mindbridge.lock    # exclusive ownership lock
└── zvec/               # disposable hybrid-search index
```

SQLite and `assets/` are the source of truth. If `zvec/` is lost, MindBridge rebuilds it from
stored embeddings without embedding historical content again. Keep the complete directory when
backing up or moving a memory. On POSIX, opening a directory enforces top-level mode `0700`.

Each memory currently has one aggregate embedding. A routed query with text uses hybrid
dense/full-text search; a pure-media query uses dense search. Text and ordered media are routed
according to model capabilities. MindBridge does not yet create chunks, per-asset vectors, or a
reranking stage.

## Python API

The synchronous surface is deliberately small:

```python
record = memory.add(content, metadata={"source": "notes"})
records = memory.add_many(["first", Path("second.png")])
hits = memory.search(query, limit=10)
answer = memory.ask(question, limit=5)
turns = memory.speech(record.id)  # timed text and stable local speaker_id values
if turns and turns[0].speaker_id:
    memory.register_speaker(turns[0].speaker_id, "Ada")
faces = memory.faces(record.id)  # normalized boxes and stable local identity_id values
if faces and turns and turns[0].identity_id:
    memory.merge_identities(faces[0].identity_id, turns[0].identity_id)
    memory.register_identity(faces[0].identity_id, "Ada")
record = memory.get(record.id)
page = memory.list(limit=100, cursor=None)
deleted = memory.delete(record.id)
memory.reindex()
memory.optimize()
```

`speech` lazily runs Fun-ASR-Nano, VAD, and CAM++ while `faces` lazily runs InsightFace SCRFD and
ArcFace. Both enroll opaque IDs in one SQLite identity registry, and biometric vectors never leave
the local store. Face and voice embeddings are not directly comparable, so
`merge_identities(canonical, duplicate)` is deliberately explicit and should follow application or
user confirmation. Existing `speaker_id` and `speaker_name` fields remain aliases for the shared
identity on `SpeakerSegment`.

`AsyncMemory` exposes the same operations with `await`. See the
[Python API reference](docs/api/python-sdk.md) for exact values, routing, and failures.
Calls on one owner may overlap remote model work; short SQLite commit/outbox and Zvec access
sections serialize to keep local state consistent.

## REST and MCP

Install and run one optional transport:

```bash
uv add "mindbridge[local,server]"
mindbridge serve --data-dir .mindbridge
```

```bash
uv add "mindbridge[local,mcp]"
mindbridge mcp --data-dir .mindbridge
```

The REST API is available under `/v1`; interactive OpenAPI documentation is served at `/docs`.
The MCP adapter exposes five typed tools. REST, MCP, and an embedded Python instance cannot own
the same directory concurrently.

Binding REST beyond loopback requires `MINDBRIDGE_API_KEY`, `--tls-certfile`, and
`--tls-keyfile`. See the [REST reference](docs/api/rest.md),
[MCP reference](docs/api/mcp.md), and [deployment guide](docs/deployment.md).

## Embedding and model backends

`Memory()` uses the pinned Jina v5 Omni Sentence Transformers adapter by default. The heavy local
runtime is isolated in the `local` extra, so importing the base SDK never imports Torch,
Transformers, or Sentence Transformers. Jina's model code is confined to `JinaOmniEmbedder`; the
generic `SentenceTransformersEmbedder` always uses standard multimodal dict/message inputs and
discovers support through `supports()`.

Qwen3-VL-Embedding therefore needs no provider branch:

```python
from mindbridge import Memory, SentenceTransformersEmbedder

embedder = SentenceTransformersEmbedder.load(
    "Qwen/Qwen3-VL-Embedding-2B",
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    device="cuda",
    batch_size=1,
)

with Memory("./data/qwen", embedder=embedder) as memory:
    memory.add("Qwen uses the standard text path.")
```

Qwen3-VL accepts text, image, video, and their combinations, but not audio. MindBridge rejects
unsupported audio before inference and routes it through ASR while retaining supported visual
parts. Start video batches at one and increase only after measuring device memory.

Generation uses the OpenAI-compatible backend by default. Passing an explicit combined
`ModelBackend` without `embedder=` or `transcriber=` keeps all of its cloud paths; the narrower
keywords compose local or custom embedding and speech backends independently.

Each directory also persists `transcription_space`, the stable ID for its ASR model and
transcript-affecting recipe, and refuses a different value at startup. The built-in `data` transport
limits each embedding or generation call to 64 MiB of aggregate raw media; trusted co-located
`file` transport or a streaming custom backend is the path for larger video. A grounded answer
sends retrieved content, timestamps, metadata, and media to the generation endpoint, limits their
serialized text evidence to 4 MiB, and sends each distinct binary asset once.

Capabilities are explicit. Configure only the modalities each operation actually accepts. When
embedding or generation does not support audio, MindBridge transcribes it and sends the transcript
together with any supported image or video input. Native audio-capable models receive audio
directly. Routing is based on declared capabilities, never guessed from a model name.

One `Memory` instance may call a backend concurrently. Custom `EmbeddingBackend`, `SpeechBackend`,
`FaceBackend`, and `ModelBackend` implementations must therefore be thread-safe until `close()`.

Model ID, immutable revision, effective dimension, normalization, query/document semantics, and
input recipe determine `space_id`. Switching any of them creates a different space. Re-encode
content into a new `data_dir`; `reindex()` only rebuilds Zvec from the existing SQLite vectors.

The [default Jina model weights](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval)
use CC BY-NC 4.0. Applications whose use is not compatible with that license should inject another
Sentence Transformers model or cloud backend.

Face recognition defaults to InsightFace `buffalo_l`, the upstream auto-downloadable pack combining
SCRFD/RetinaFace-class detection with ArcFace recognition. The adapter also accepts other upstream
model-pack names for measured CPU, GPU, or quality trade-offs. InsightFace code is MIT, but its
[pretrained model packs are restricted to non-commercial research](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#license);
production deployments must supply appropriately licensed weights.

See [configuration](docs/configuration.md) for every variable and custom backend guidance.

## Benchmark isolation

Isolation is physical, not logical. Concurrent benchmark tasks must never share a `data_dir`:

```text
.benchmarks/<benchmark>/<run>/<case>/
```

Each leaf owns its own SQLite database, asset store, Zvec collection, and lock. Benchmark labels
organize paths in the harness; they are not fields in the product API or storage schema. See
[benchmarking](docs/benchmarking.md).

## Documentation

- [Get started](docs/quickstart.md)
- [Core concepts](docs/concepts.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Python API](docs/api/python-sdk.md)
- [REST API](docs/api/rest.md)
- [MCP API](docs/api/mcp.md)
- [Command-line API](docs/api/cli.md)
- [Deployment](docs/deployment.md)
- [Operations](docs/operations.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Breaking upgrade guide](docs/quickstart.md#upgrading-from-the-service-based-release)

## Development

MindBridge supports Python 3.10 through 3.14 and uses
[uv](https://docs.astral.sh/uv/) with a checked-in lockfile.

```bash
uv sync --all-groups --all-extras
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a change.

## License

[Apache License 2.0](LICENSE)
