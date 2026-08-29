# Quick start

## Install

```bash
uv add "mindbridge[local]"
```

The `local` extra supplies the pinned Jina embedding adapter and FunASR speech adapter. Add
`openai`, `server`, or `mcp` only when the application uses those surfaces.

## Add and search

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hit = memory.search("Where is the spare key?")[0]

print(record.id)
print(hit.content)
```

Choose an explicit directory. One directory can have only one running owner.

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
speaker centroids into durable memory semantics. When an answerer is configured, `ask` reuses that
identity cache as grounding evidence while returning the original source hits.

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
