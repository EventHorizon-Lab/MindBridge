# Quick start

## Install

```bash
uv add "mindbridge[local]"
```

The `local` extra supplies the pinned Jina embedding adapter and FunASR speech adapter. Add
`face`, `openai`, `server`, or `mcp` only when the application uses those surfaces.

## Open a configured memory

The shortest supported path validates a pure-data configuration, constructs the bundled adapters,
and closes every resource with the memory:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    memory.add("The spare key is in the blue toolbox.")
    print(memory.search("Where is the spare key?"))
```

Configuration supports bundled adapters. Direct object injection remains the plugin API for custom
models and caller-owned SDK clients.

## Choose an embedding backend

Every memory requires an embedding capability. Declarative configuration selects a bundled
provider; direct construction accepts any `EmbeddingBackend`. Neither path makes the kernel read
provider credentials. Every example below uses the pinned multimodal Jina adapter.

> **The pinned Jina weights are licensed CC BY-NC 4.0 — non-commercial use only.** That licence
> covers the model, not MindBridge. Commercial deployments must inject a backend whose weights they
> are licensed to use.

`SentenceTransformersEmbedder` loads any Sentence Transformers model as that replacement. The
`local` extra already supplies it, so no additional install is needed:

```python
from mindbridge import SentenceTransformersEmbedder

embedder = SentenceTransformersEmbedder.load(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
```

That model is Apache-2.0 and 384-dimensional, and it declares `text` capability only. Adding media
to a store built on it fails before inference with
`ModelError: configured embedding model does not support: image`, rather than silently dropping the
media. `JinaOmniEmbedder` remains the default precisely because it covers image, video, and audio
as well.

`revision` is required and must be a 40-character immutable commit hash. A branch name is rejected —
passing `revision="main"` raises
`ValidationError: revision must be an immutable 40-character commit hash`. The pin is not ceremony:
the revision, model ID, dimension, and input recipe together form the durable embedding space
recorded in `data_dir`, and MindBridge re-checks it on open so one store can never silently mix two
embedding spaces.

Read the commit from the model's Hugging Face page under *Files and versions*, or resolve the
current head of `main` once and paste it into the call:

```bash
curl -s https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 \
  | python -c "import json, sys; print(json.load(sys.stdin)['sha'])"
```

Changing the pin later changes the embedding space and forces a rebuild, so treat it like any other
dependency version: choose it deliberately and record it in source.

## Add and search

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hits = memory.search("Where is the spare key?")

print(record.id)
for hit in hits:
    print(hit.score, hit.content)
if not hits:
    print("no stored memory was relevant enough to return")
```

Choose an explicit directory. One directory can have only one running owner. `Memory` also closes
the adapters it was given, so construct a fresh embedder per store rather than sharing one across
two `Memory` objects.

### Search can legitimately return nothing

`search` returns a `tuple[SearchHit, ...]`, and that tuple is empty whenever the store holds nothing
good enough. An empty result is a normal answer, not an error. Two constructor gates produce it:

- `minimum_relevance` (default `0.55`) drops every candidate below the threshold. Confidence is the
  dense cosine similarity rescaled to `[0, 1]`, so `0.55` is roughly `0.10` cosine. A full-text
  lexical match instead sets a candidate's confidence to `0.6`, which clears the default gate on
  its own — so a shared keyword can return a hit that the vectors alone would have rejected.
- `ambiguity_margin` (default `0.01`) withholds an unresolved choice from `search(limit=1)` or
  `ask(limit=1)` when the top two dense confidences are effectively tied and the winner has neither
  a lexical nor a temporal anchor. Larger limits preserve the qualified candidates for the caller
  or answerer.

Both accept `0` to disable that gate, but reaching for that first is the wrong move. Withholding
one unresolved choice instead of returning a confidently ranked wrong row is deliberate. With a
larger limit, candidate handling belongs to the caller or answerer. Write the empty branch before
tuning a threshold.

An unrelated query against a small store therefore returns `()` rather than the nearest available
row. Any caller that indexes `search(...)[0]` will eventually raise `IndexError` in production.

## Add media

```python
from pathlib import Path

from mindbridge import Blob, JinaOmniEmbedder, Memory

with Memory("./data/media", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add(
        (
            "Inspection evidence",
            Path("./panel.png"),
            Blob(Path("./note.wav").read_bytes(), "audio/wav", "note.wav"),
        )
    )
```

Python accepts `str`, `Path`, `Blob`, and `AssetRef`. Fetch remote media before calling MindBridge;
network URL fetching belongs to the application transport stack.

## Add speech fallback

```python
from pathlib import Path

from mindbridge import FunASRTranscriber, JinaOmniEmbedder, Memory

with Memory(
    "./data/speech",
    embedder=JinaOmniEmbedder(),
    transcriber=FunASRTranscriber(),
    index_speech=True,
) as memory:
    record = memory.add(Path("./meeting.wav"))
    turns = memory.speech(record.id)
```

Speech analysis is lazy by default; this example opts into add-time indexing so transcript and
identity text can be retrieved. FunASR owns model execution; MindBridge maps its timed turns and
speaker observations into bounded durable exemplars. When an answerer is configured, `ask` reuses that
identity cache as grounding evidence while returning the original source hits.

## Recognize face and voice identity

```bash
uv add "mindbridge[face,local]"
```

```python
from pathlib import Path

from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/identity",
        "embedding": {"provider": "jina-omni"},
        "speech": {"provider": "funasr", "device": "cuda"},
        "face": {
            "provider": "opencv",
            "detector_model": "./models/face_detection_yunet.onnx",
            "recognizer_model": "./models/face_recognition_sface.onnx",
        },
        "settings": {"index_speech": True},
    }
) as memory:
    record = memory.add(Path("./introduction.mp4"))
    face = memory.faces(record.id)[0]
    memory.register_identity(face.identity_id, "Ada")
```

The model paths are explicit and stay local. A single-face/single-speaker video can establish one
shared identity; ambiguous scenes are kept separate. Each modality has its own matching threshold
and bounded exemplar collection.

## Add a provider answerer

```bash
uv add "mindbridge[openai]"
export OPENAI_API_KEY="your-key"
```

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels

with OpenAI(timeout=30.0, max_retries=3) as client:
    answerer = OpenAIModels(generation_client=client, generation_model="gpt-5-mini")
    with Memory(
        "./data/answers",
        embedder=JinaOmniEmbedder(),
        answerer=answerer,
    ) as memory:
        memory.add("The deployment window starts at 22:00 UTC.")
        result = memory.ask("When does deployment start?")
        print(result.answer)
        print(result.hits)
```

The SDK client remains caller-owned. It controls credentials, HTTP behavior, retries, and
timeouts. `OpenAIModels.close()` intentionally leaves it open.

## Use async application code

```python
import asyncio

from mindbridge import AsyncMemory, JinaOmniEmbedder


async def main() -> None:
    async with AsyncMemory(
        "./data/async",
        embedder=JinaOmniEmbedder(),
    ) as memory:
        await memory.add("A durable async call.")
        print(await memory.search("durable"))


asyncio.run(main())
```

`AsyncMemory` is an async facade over the embedded synchronous consistency core. It does not wrap
or emulate provider SDK clients; adapters decide how provider calls are implemented.

Next: [configuration and composition](configuration.md), [Python API](api/python-sdk.md), and
[architecture](architecture.md).
