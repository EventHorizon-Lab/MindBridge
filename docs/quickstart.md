# Quick start

This guide takes a clean Python project from installation to durable multimodal retrieval.
MindBridge runs in your process. Embedding and speech analysis are local by default; generation
may use a configured remote endpoint.

## Requirements

- Python 3.10 through 3.14.
- A writable local directory.
- The `local` extra for default Jina v5 Omni embedding and FunASR speech analysis.
- An OpenAI-compatible endpoint only when the workload uses `ask`.

Text needs embeddings for `add` and `search`, plus generation for `ask`.

## Install

With `uv`:

```bash
uv add "mindbridge[local]"
```

Or with `pip`:

```bash
python -m pip install "mindbridge[local]"
```

Set the shared model credential when using grounded generation:

```bash
export OPENAI_API_KEY="your-api-key"
```

Install the base `mindbridge` package without `local` only when injecting cloud embedding and
transcription backends.

## Create a text memory

The shortest valid program is:

```python
from mindbridge import Memory

with Memory() as memory:
    memory.add("The design review starts at 14:00 UTC.")
    print(memory.search("When is the design review?"))
```

`Memory()` creates `.mindbridge/` in the current working directory. Production code should choose
the path explicitly:

```python
from mindbridge import Memory

with Memory(data_dir="./data/product-assistant") as memory:
    record = memory.add(
        "Nora prefers weekly summaries on Friday.",
        metadata={"source": "preferences"},
    )
    print(record.id, record.modality)
```

## Add media

The default Jina embedder accepts text, image, video, and audio natively. Before calling `ask`,
declare capabilities that match the generator. Local FunASR handles audio fallback:

```bash
export MINDBRIDGE_GENERATION_MODALITIES=text,image,video
export MINDBRIDGE_ALLOWED_URL_HOSTS=media.example
```

Python has five content atoms:

```python
from pathlib import Path

from mindbridge import AssetRef, Blob, URL

text = "The prototype shown during the review"
local_file = Path("./prototype.png")
inline_file = Blob(
    Path("./review.wav").read_bytes(),
    media_type="audio/wav",
    name="review.wav",
)
remote_file = URL("https://media.example/demo.mp4", media_type="video/mp4")
existing_file = AssetRef(id="0" * 64)  # Replace with record.assets[n].id.
```

`Path` is copied into the local content-addressed asset store; changing or deleting the source
file later does not change the memory. `Blob` requires non-empty bytes and a concrete image,
video, or audio MIME type. `URL` requires HTTPS and an expected MIME type; redirects and the
downloaded `Content-Type` are validated before storage, and each connection is pinned to a verified
public DNS result. An `AssetRef` is resolved only inside the same `data_dir` and fails if its ID or
optional modality hint does not match SQLite.

Pass one atom or an ordered sequence to `add`:

```python
record = memory.add(
    [
        "The prototype shown during the review",
        Path("./prototype.png"),
        Blob(
            Path("./review.wav").read_bytes(),
            media_type="audio/wav",
            name="review.wav",
        ),
    ]
)

print(record.content)
print(record.modality)  # Modality.OMNI: image + audio
print(record.assets[0].id)
```

Atom order contributes to stable memory identity, and `record.assets` preserves media order. Text
parts are joined in order for the aggregate model input. A record with no media is `text`; a record
with one media family is `image`, `video`, or `audio`, even when it also has text; two or more media
families produce `omni`. Identical media bytes share one CAS descriptor, including its first
authoritative non-empty name; filenames are not per-memory labels.

## Retrieve and answer across modalities

`search` and `ask` accept the same content values:

```python
hits = memory.search(
    ["Find the prototype with this layout", Path("./query-layout.png")],
    limit=5,
)
for hit in hits:
    print(hit.score, hit.modality, hit.content, hit.assets)

result = memory.ask(
    ["What changed between the stored prototype and this image?", Path("./latest.png")],
    limit=5,
)
print(result.answer)
print(result.hits)
```

A routed query containing text combines dense and full-text signals in Zvec; a pure-media query
uses dense search. MindBridge then hydrates ranked records and assets from SQLite. `ask` returns
the generated answer and source hits selected for grounding.
If there are no hits, the default backend answers that it does not know.

## Understand model routing

Routing follows declared capabilities, never a model-name heuristic:

- The default Jina embedding adapter receives text, image, video, audio, and mixed input natively.
- Any model operation that supports the input modality receives it natively.
- When embedding does not support audio, MindBridge transcribes the audio, removes the audio part,
  and embeds the transcript together with any still-supported image or video parts.
- Generation follows the same rule. A visual-language model without audio receives ASR text plus
  the retained image or video evidence.
- Audio-only input becomes transcript-only after this fallback; visual parts are never dropped to
  force an unsupported fallback.
- If no valid route exists, MindBridge raises `ModelError` instead of ignoring content.

One memory currently produces one aggregate embedding. There is no automatic chunking, separate
vector per asset, or learned reranker. Temporal and decay ranking are deterministic. With the
built-in `data` transport, one embedding or generation call may contain at most 64 MiB of aggregate
raw media before base64 expansion. Use trusted co-located `file` transport or a streaming custom
backend for larger video.

`transcription_space` is also part of the directory's durable compatibility identity. Keep it
stable for one ASR model and transcript-affecting preprocessing recipe. To change that recipe,
allocate a new directory and re-ingest rather than editing stored metadata.

Call `memory.speech(record.id)` to get timed `SpeakerSegment` values. The first recording enrolls
an opaque local `speaker_id`; later recordings with a clear CAM++ match reuse that ID. Analysis is
lazy and cached in SQLite. Assign a name after enrollment; registration also updates already
cached recordings:

```python
turns = memory.speech(record.id)
if turns and turns[0].speaker_id:
    memory.register_speaker(turns[0].speaker_id, "Ada")
    assert memory.speech(record.id)[0].speaker_name == "Ada"
```

## Choose another Sentence Transformers model

Qwen3-VL-Embedding uses the generic adapter and standard multimodal dict/message inputs:

```python
from pathlib import Path

from mindbridge import Memory, SentenceTransformersEmbedder

qwen = SentenceTransformersEmbedder.load(
    "Qwen/Qwen3-VL-Embedding-2B",
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    device="cuda",
    batch_size=1,
)

with Memory("./data/qwen", embedder=qwen) as memory:
    memory.add(["Prototype", Path("./prototype.png")])
```

The generic adapter asks the loaded model which atomic modalities it supports. Qwen3-VL supports
text, image, video, and their combinations; it does not advertise audio, so MindBridge uses the ASR
fallback described above. Its native 2B vector dimension is 2048. A smaller dimension is accepted
only when the loaded model exposes machine-readable Matryoshka metadata for that dimension.

Every adapter derives `space_id` from its type, model, immutable revision, effective dimension,
normalization, retrieval-side methods, and input recipe. Changing any component requires a new
`data_dir` and re-encoding the source content. `reindex()` is intentionally not a model migration:
it rebuilds Zvec from the vectors already stored in SQLite.

## Add a memory role, event time, and metadata

`MemoryType.SEMANTIC` is the default. Use episodic for situated experiences and procedural for
reusable instructions. `occurred_at` is optional and must include a timezone:

```python
from datetime import datetime, timezone

from mindbridge import MemoryType

record = memory.add(
    "The deployment completed.",
    memory_type=MemoryType.EPISODIC,
    occurred_at=datetime(2026, 8, 27, 9, 30, tzinfo=timezone.utc),
    metadata={"source": "release-log"},
)
```

Identity covers normalized text, ordered asset content, memory role, event time, and canonical
metadata. Adding the same logical record returns the existing record without another model call.

Resolve relative dates against an explicit clock when a reproducible answer matters:

```python
hits = memory.search(
    "What happened last week?",
    memory_type=MemoryType.EPISODIC,
    reference_at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
)
```

See [memory types, temporal reasoning, and decay](memory-types-time-and-decay.md) for recognized
expressions, optional decay, and exact limits.

## Add a batch

`add_many` accepts the same content contract, with no per-item event time or metadata:

```python
records = memory.add_many(
    [
        "The staging region is eu-west-1.",
        ["Production dashboard", Path("./production.png")],
    ],
    memory_type=MemoryType.PROCEDURAL,
)
```

New items are embedded in one model batch and committed in one SQLite transaction. Returned
records match input order; duplicate inputs can repeat the same ID.

## Read, list, and delete

```python
record = memory.get(record.id)

page = memory.list(limit=50)
while page.next_cursor is not None:
    page = memory.list(limit=50, cursor=page.next_cursor)

deleted = memory.delete(record.id)
deleted_again = memory.delete(record.id)
assert deleted is True
assert deleted_again is False
```

Deleting the last reference to an asset removes its SQLite descriptor and content-addressed file.
List cursors are opaque; store and return them unchanged.

## Use the async API

`AsyncMemory` is a thin asynchronous facade over the same local core:

```python
import asyncio
from pathlib import Path

from mindbridge import AsyncMemory


async def main() -> None:
    async with AsyncMemory(data_dir="./data/async-assistant") as memory:
        await memory.add(["Incident recording", Path("./incident.wav")])
        hits = await memory.search("What happened during the incident?")
        print(hits)


asyncio.run(main())
```

Do not create separate synchronous and asynchronous owners for the same path.
Concurrent calls on the one owner may overlap remote model work; MindBridge briefly serializes
SQLite commit/outbox and Zvec access where consistency requires it.

## Isolate applications and benchmarks

The directory is the isolation boundary:

```python
from mindbridge import Memory

with (
    Memory("./data/experiment-a") as experiment_a,
    Memory("./data/experiment-b") as experiment_b,
):
    experiment_a.add("Only A can retrieve this.")
    assert experiment_b.search("Only A") == ()
```

A second owner of the same path fails immediately. Parallel benchmark cases must each allocate a
different leaf directory; a run label belongs in the path, never in a product request field. On
POSIX, opening each directory enforces owner-only mode `0700`.

## Upgrading from the service-based release

This is a breaking upgrade, not an in-place PostgreSQL migration.

- There is no server-side account hierarchy or logical store scope.
- Former tenant, user, and run fields are rejected by Python, REST, and MCP.
- The old PostgreSQL schema is not opened or converted automatically.
- Background workers, Redis, and S3 configuration no longer apply.
- SQLite, local content-addressed assets, and Zvec now live under one `data_dir`.

Export the content, event time, metadata, and source media you intend to keep. Allocate one new
directory for each genuinely separate memory domain, then ingest through the new public API.
MindBridge generates new embeddings and a fresh Zvec index.

Do not encode an old scope identifier into metadata and treat it as isolation. Metadata is payload,
not an authorization boundary.

Next, read [core concepts](concepts.md), [configuration](configuration.md), or the
[MCP API](api/mcp.md).
