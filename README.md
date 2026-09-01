# MindBridge

MindBridge is fast local multimodal memory for Python agents. It stores text, image, video, audio,
and mixed observations in one embedded runtime, then retrieves them or grounds model answers in
the stored evidence.

SQLite is authoritative, media lives in a local content-addressed store, and Zvec is a rebuildable
search projection. One physical `data_dir` has one live `Memory` owner.

## Get started

MindBridge supports Python 3.10 through 3.14. Install the bundled local embedding adapter:

```bash
uv add "mindbridge[local]"
```

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    memory.add("The spare key is in the blue toolbox.")
    for hit in memory.search("Where is the spare key?"):
        print(hit.score, hit.content)
```

The bundled Jina model supports text and media, but its weights are licensed CC BY-NC 4.0 for
non-commercial use. Use another embedding backend when that licence does not fit your application.

## Documentation

- [Quick start](docs/quickstart.md)
- [Core concepts](docs/concepts.md)
- [Configuration](docs/configuration.md)
- [Python, REST, MCP, and CLI reference](docs/README.md#api-reference)
- [Architecture, deployment, and operations](docs/README.md#operate-mindbridge)
- [Troubleshooting](docs/troubleshooting.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) to develop MindBridge. MindBridge-authored code is licensed
under the [Apache License 2.0](LICENSE); third-party benchmark material retains its
[upstream terms](src/mindbridge/benchmarks/_official/NOTICE.md).
