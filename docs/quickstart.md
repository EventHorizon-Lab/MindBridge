# Quick start

This guide creates one durable local memory, stores a fact, and retrieves it. It uses MindBridge's
bundled Jina embedding adapter, so no database service or model API key is required.

## 1. Install the local recipe

MindBridge supports Python 3.10 through 3.14. In a project managed by
[uv](https://docs.astral.sh/uv/), run:

```bash
uv add "mindbridge[local]"
```

If your application supplies its own `EmbeddingBackend`, install the base package instead. Other
bundled models and transports have separate extras; see
[install choices](configuration.md#install-only-the-surfaces-you-use).

### Review the bundled model boundary

The first embedding call downloads `jinaai/jina-embeddings-v5-omni-small-retrieval` when it is not
already cached. MindBridge pins both its model and executable remote code to one immutable
revision, but loading it still uses `trust_remote_code=True`. Its weights are licensed CC BY-NC
4.0 for non-commercial use.

Review that code and license before using this recipe with sensitive or commercial workloads. If
those terms do not fit, choose another [embedding backend](configuration.md#embedding-choices).

## 2. Store and retrieve one memory

Save this complete example as `quickstart.py`:

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/quickstart",
    "embedding": {"provider": "jina-omni"},
    "settings": {
        "minimum_relevance": 0,
        "ambiguity_margin": 0,
    },
}

with Memory.from_config(config) as memory:
    stored = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hits = memory.search("Where is the spare key?", limit=1)

    assert hits and hits[0].id == stored.id
    print(f"stored: {stored.id}")
    print(f"found: {hits[0].content}")
```

Run it:

```bash
uv run python quickstart.py
```

The first run may pause while the pinned model downloads. Success prints a stable record ID and:

```text
found: The spare key is in the blue toolbox.
```

The two retrieval settings disable relevance and ambiguity rejection so this one-record checkpoint
is deterministic. Remove them to restore the defaults before tuning retrieval for a real corpus.
`search()` normally returns a tuple that may be empty when no evidence clears those gates.

## 3. Reuse the directory safely

Run the script again to reopen `./data/quickstart`; adding the same canonical input returns the
same record instead of creating a duplicate. The context manager closes the models, SQLite
connection, and search index before the process exits.

Only one live `Memory` may own that physical directory. Give each independent application,
account, test, or process its own `data_dir`; metadata is application data, not an isolation or
authorization boundary. SQLite holds the authoritative records and embeddings, while a missing
Zvec index is rebuilt from that data without re-embedding stored content.

## Next steps

- Store image, video, audio, or mixed input with `Path`, `Blob`, and `AssetRef` in the
  [Python content contract](api/python-sdk.md#content-contract).
- Add grounded answers, opt in to automatic typed formation, or choose a different embedding
  backend in [configuration](configuration.md).
- Use `AsyncMemory` from the [Python SDK](api/python-sdk.md#asyncmemory).
- Expose the same memory through [REST](api/rest.md), [MCP](api/mcp.md), or the
  [command line](api/cli.md).
- Read [core concepts](concepts.md) before choosing production storage and model settings.
