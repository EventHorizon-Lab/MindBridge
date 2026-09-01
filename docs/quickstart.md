# Quick start

This guide opens one local memory, stores a fact, and searches it. At the end you will have one
durable `data_dir` that can be reopened with the same embedding configuration. No database service
or model API key is required.

## 1. Install MindBridge

MindBridge supports Python 3.10 through 3.14. In a project managed by
[uv](https://docs.astral.sh/uv/), install the bundled local adapters:

```bash
uv add "mindbridge[local]"
```

The first model call loads Jina, downloading it when absent, and executes that model's code with
`trust_remote_code=True`; both model and code are pinned to one immutable revision. The weights are
CC BY-NC 4.0 for non-commercial use. Review the revision and license, or select another
[embedding choice](configuration.md#embedding-choices), before using this recipe with sensitive or
commercial workloads.

## 2. Create a memory

Save this as `quickstart.py`:

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/assistant",
    "embedding": {"provider": "jina-omni"},
}

with Memory.from_config(config) as memory:
    record = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hits = memory.search("Where is the spare key?")

print(f"stored {record.id}")
for hit in hits:
    print(f"{hit.score:.3f}: {hit.content}")
if not hits:
    print("No stored memory was relevant enough to return.")
```

Run it:

```bash
uv run python quickstart.py
```

The checkpoint is a `stored` line followed by either a matching hit or the explicit no-hit line.
IDs and scores vary, so do not compare them with fixed sample output.

`search()` returns a tuple and may return `()` when no candidate clears the relevance gate. Treat
that as a normal result; do not assume `hits[0]` exists.

The context manager closes the models, SQLite connection, and search index. Only one live
`Memory` may own `./data/assistant`; use a different directory for an independent memory domain.

## 3. Add the capability you need

- Add images, video, audio, or mixed input with `Path`, `Blob`, and `AssetRef`; see the
  [Python content contract](api/python-sdk.md#content-contract).
- Add grounded answers with the `openai` extra and a generation backend; see
  [configuration](configuration.md#declarative-configuration).
- Use `AsyncMemory` in async applications; see the
  [Python API](api/python-sdk.md#asyncmemory).
- Expose the same memory through [REST](api/rest.md), [MCP](api/mcp.md), or the
  [command line](api/cli.md).

Next, read [core concepts](concepts.md) before choosing production storage and model settings.
