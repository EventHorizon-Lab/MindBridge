# MindBridge

MindBridge is an embedded multimodal memory library for Python agents. It stores text, image,
video, audio, and mixed observations locally, then retrieves relevant records or grounds model
answers in the stored evidence.

It runs in your process: SQLite is authoritative, media uses a local content-addressed store, and
Zvec is a rebuildable search projection. One physical `data_dir` belongs to one live `Memory`
owner, so separate memory domains use separate directories.

## Install

MindBridge supports Python 3.10 through 3.14. Choose the smallest install for your application:

- First local run, with bundled multimodal embedding: `uv add "mindbridge[local]"`
- Application-supplied embedding backend: `uv add mindbridge`
- OpenAI models, REST serving, or MCP transport: add the `openai`, `server`, or `mcp` extra

The local recipe needs no model API key, but it downloads and executes a pinned Jina model whose
weights are restricted to non-commercial use. Read the [quick start](docs/quickstart.md) before
using that model with sensitive or commercial workloads. All install combinations are listed in
[configuration](docs/configuration.md#install-only-the-surfaces-you-use).

## Start

Follow the [quick start](docs/quickstart.md) to install the local recipe and run one complete
store-and-search example. No database service, cache, queue, or object store is required.

Then choose the page for your task:

- [Core concepts](docs/concepts.md) — records, retrieval, persistence, and isolation
- [Configuration](docs/configuration.md) — bundled providers and custom backends
- [Python SDK](docs/api/python-sdk.md) — supported imports from `mindbridge`
- [REST API](docs/api/rest.md) — the `/v1` HTTP surface
- [MCP tools](docs/api/mcp.md) — the seven supported tools
- [Deployment and operations](docs/README.md#deploy-and-operate)
- [Troubleshooting](docs/troubleshooting.md)

The [documentation index](docs/README.md) owns the complete learning and reference map.

## License

MindBridge-authored code is available under the [Apache License 2.0](LICENSE). Models, datasets,
and vendored benchmark material may use different terms; the bundled Jina weights are CC BY-NC
4.0, and benchmark notices retain their
[upstream licenses](src/mindbridge/benchmarks/_official/NOTICE.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository setup and quality gates.
