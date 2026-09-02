# Quick start

This guide teaches the smallest complete MindBridge loop: store a memory, close the process-local
runtime, reopen it, and retrieve the same memory. The main example needs no database service or API
key. Optional steps then add grounded answers, typed events, and omni-modal input one concept at a
time.

## 1. Install MindBridge

The SDK supports Python 3.10 through 3.14. Install every optional integration when you want a
feature-complete development environment:

```bash
uv add "mindbridge[all]"
```

With `pip`:

```bash
python -m pip install "mindbridge[all]"
```

For the first example alone, the smaller local installation is enough:

```bash
uv add "mindbridge[local]"
```

| Install | Choose it when |
| --- | --- |
| `mindbridge[all]` | You want every optional model runtime, transport, observability, and benchmark dependency. |
| `mindbridge[local]` | You want local embedding and speech without an API key. |
| `mindbridge[openai]` | You want OpenAI or an OpenAI-compatible model endpoint. |
| `mindbridge` | Your application supplies its own embedding backend. |

`mindbridge[all]` installs dependencies; it does not configure providers, download benchmark
datasets, or supply YuNet and SFace model files. Install the smallest surface that fits your
deployment once you know which capabilities it needs.

## 2. Store and retrieve one memory

Save this as `quickstart.py`:

```python
from mindbridge import Memory

# Everything stored by this example lives under this directory.
config = {
    "data_dir": "./data/mindbridge-quickstart",
    # The bundled Jina adapter embeds text, images, video, and audio locally.
    "embedding": {"provider": "jina-omni"},
    # These two zeroes make this one-record tutorial deterministic.
    # Remove the settings block to use the production defaults.
    "settings": {
        "minimum_relevance": 0.0,
        "ambiguity_margin": 0.0,
    },
}

# Open the store, add one durable record, then close every resource.
with Memory.from_config(config) as memory:
    stored = memory.add("The spare key is in the blue toolbox.")
    print("stored:", stored.id)

# Reopen the same directory and search the persisted record.
with Memory.from_config(config) as memory:
    hits = memory.search("Where is the spare key?", limit=1)
    if not hits:
        raise SystemExit("No memory matched the question.")
    print("found:", hits[0].content)
```

Run it:

```bash
uv run python quickstart.py
```

The first run may pause while the pinned local model downloads. Success ends with:

```text
found: The spare key is in the blue toolbox.
```

The program demonstrates four core contracts:

1. `Memory.from_config()` opens one embedded memory system. No external database is required.
2. `add()` returns a durable `MemoryRecord`; repeating the same canonical input returns the same
   record instead of creating a duplicate.
3. Leaving the first `with` block closes the model, SQLite connection, and search index. Reopening
   the same `data_dir` proves that the memory survived the process-local runtime.
4. `search()` returns ranked `SearchHit` values. An empty tuple is a valid result when no evidence
   clears the relevance or ambiguity gates.

SQLite owns the authoritative records and embeddings. Zvec is a rebuildable search projection, so
a missing index can be restored from SQLite without embedding stored content again.

## 3. Add grounded answers

Search returns evidence for your application to use. `ask()` additionally asks a configured
generation model to answer from that evidence.

If you installed only `local`, add the OpenAI integration and provide a key. Skip this installation
step if you already use `mindbridge[all]`.

```bash
uv add "mindbridge[local,openai]"
export OPENAI_API_KEY="replace-with-your-key"
```

Append this to `quickstart.py`:

```python
# Reuse the local embedding space and add only a generation backend.
answer_config = {
    **config,
    "generation": {"provider": "openai", "model": "gpt-5-mini"},
}

with Memory.from_config(answer_config) as memory:
    # ask() searches first, then gives the retrieved evidence to the answerer.
    result = memory.ask("Where is the spare key?", limit=1)

    if result.abstained:
        print("no answer:", result.abstention_reason)
    else:
        print("answer:", result.answer)
        print("evidence:", [hit.id for hit in result.hits])
```

`AnswerResult.hits` is the evidence used by the answer backend. Treat abstention as a normal result,
not an exception. In this composition, records and embeddings remain local, while the question and
retrieved evidence are sent to the configured OpenAI endpoint.

## 4. Add one capability at a time

The same `Memory` surface accepts richer observations without changing the storage model.

### Typed events

Use `MemoryType` and timezone-aware event times when *what happened when* matters:

```python
from datetime import datetime, timezone

from mindbridge import MemoryType

# Run inside an open `with Memory.from_config(...) as memory:` block.
memory.add(
    "Maya placed the spare key in the toolbox during the workshop.",
    memory_type=MemoryType.EPISODIC,
    occurred_at=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
)

hits = memory.search(
    "What happened during the workshop?",
    memory_type=MemoryType.EPISODIC,
)
```

MindBridge also supports semantic memories for facts and procedural memories for instructions.
Event intervals, bitemporal validity, spatial scope, decay, and reinforcement are covered in
[Memory types, time, and decay](memory-types-time-and-decay.md).

### Omni-modal observations

With the local Jina embedder, one record can preserve ordered text and media:

```python
from pathlib import Path

# Text and image remain one observation with one stable memory ID.
record = memory.add(
    ("Maya is showing the blue toolbox.", Path("./workshop.jpg")),
)
print(record.modality, record.assets)
```

`ContentInput` accepts text, local `Path`, inline `Blob`, existing `AssetRef`, or an ordered
combination. The selected embedding backend must support the supplied modalities; unsupported input
fails explicitly instead of silently discarding evidence.

## 5. Keep these boundaries in mind

- One physical `data_dir` has one live owner. Use separate directories for independent applications,
  processes, tenants, tests, or security domains.
- Metadata is application data. It is not a retrieval filter, authorization rule, or isolation
  boundary.
- An embedding model, revision, dimension, and request recipe define a durable vector space. Start a
  new directory when intentionally changing that space.
- The bundled Jina recipe downloads pinned model code and loads it with `trust_remote_code=True`.
  Its weights are CC BY-NC 4.0; review both boundaries before sensitive or commercial use.
- Remote URLs are not fetched. Download and validate remote media in the host application before
  passing a `Blob` to MindBridge.

## Where to go next

| Goal | Continue with |
| --- | --- |
| See the complete product surface | [Product capabilities](product-capabilities.md) |
| Configure providers and optional formation | [Configuration](configuration.md) |
| Use every SDK operation | [Python SDK](api/python-sdk.md) |
| Build live audio, vision, or interaction memory | [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) |
| Expose memory to other runtimes | [REST](api/rest.md), [MCP](api/mcp.md), or [CLI](api/cli.md) |
| Diagnose startup or retrieval | [Troubleshooting](troubleshooting.md) |
