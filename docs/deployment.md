# Deployment

Deploy one live `Memory` owner per physical `data_dir`. Separate applications, tenants, or workers
that require isolation must use separate directories.

This page chooses and configures a topology. See [architecture](architecture.md) for invariants,
[operations](operations.md) for runbooks, and [troubleshooting](troubleshooting.md) when startup or
traffic fails.

## Choose a topology

| Need | Topology | Constraint |
| --- | --- | --- |
| One Python application owns memory | Embed `Memory` in that process | Keep one instance open and close it during shutdown. |
| Other processes need HTTP | Expose the owner with `create_app()` | Run one ASGI worker for that directory and supply network controls outside MindBridge. |
| One agent host needs tools | Run `build_mcp_server()` in the owner process | Prefer stdio; secure SSE or streamable HTTP as a network service. |
| Independent workers need independent state | Give each worker a distinct directory | The directories are separate memory domains; they do not synchronize. |

Do not place a Python owner, REST process, and MCP process over the same directory. Put the desired
adapters around one constructed `Memory`, or have other processes call the existing REST owner.

## Deployment checklist

Before starting the owner, verify:

- Python 3.10 through 3.14 and compatible 64-bit wheels exist for Zvec and every selected backend.
- The data directory is on one local, durable filesystem with working SQLite WAL, file locking,
  atomic rename, and durable writes.
- The Python build links SQLite 3.35 or newer with the JSON1 extension, which is compiled in by
  default from 3.38. `consolidation_candidates` reads the operation log with `json_tree`, so a
  build without it fails that one call.
- The service account alone can read the directory, model credentials, and backups.
- Disk capacity covers SQLite, original media, the derived index, and temporary rebuild or
  compaction space.
- RAM, accelerator memory, and file-descriptor limits cover model inference and the memory-mapped
  index.
- The embedding, transcription, face, and index recipes match an existing store.
- A shutdown hook calls `Memory.close()` and closes caller-owned provider clients.

Network filesystems and shared-volume multi-writer deployments are unsupported. On POSIX,
MindBridge sets the top-level directory to `0700` and its SQLite, lock, staging, and asset files to
restrictive modes. The operator still owns parent permissions, service accounts, disk encryption,
backup access, and retention.

## Embedded Python

Install only the optional surfaces used by the application. This example uses the bundled local
embedder and OpenAI generation adapter:

```bash
uv add "mindbridge[local,openai]"
```

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels

client = OpenAI(timeout=30.0, max_retries=3)
try:
    with Memory(
        "/var/lib/mindbridge/assistant",
        embedder=JinaOmniEmbedder(),
        answerer=OpenAIModels(generation_client=client),
    ) as memory:
        memory.add("Process-owned memory")
finally:
    client.close()
```

Keep one `Memory` open for a long-running process. `Memory.close()` closes supplied backend objects;
`OpenAIModels.close()` deliberately leaves caller-owned OpenAI clients open. Construct `Memory`
after any process fork, never before it.

The Jina adapter executes pinned upstream model code and uses non-commercial weights. Review
[embedding choices](configuration.md#embedding-choices) before deploying it.

## REST owner

Install the server surface with the selected model extras:

```bash
uv add "mindbridge[local,openai,server]"
```

Compose one caller-owned instance in `my_application.py`:

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels
from mindbridge.api import create_app

client = OpenAI(timeout=30.0, max_retries=3)
memory = Memory(
    "/var/lib/mindbridge/assistant",
    embedder=JinaOmniEmbedder(),
    answerer=OpenAIModels(generation_client=client),
)
app = create_app(memory=memory)
```

Register framework shutdown handling for both `memory.close()` and `client.close()`, then run one
worker:

```bash
uvicorn my_application:app --host 127.0.0.1 --port 8000 --workers 1
```

`create_app()` does not own or close either resource. It also adds no authentication,
authorization, TLS, rate limiting, quota, or audit log. Bind to a trusted interface or put the app
behind the deployment's existing gateway or middleware. Keep `/healthz` protected consistently; it
reports liveness and the live composition's capability declaration, not model or retrieval
readiness. The app publishes thirteen product routes under `/v1`, or twenty-three when the host enables
`identity_operations` and `embodied_operations`.

Do not retry a timed-out request blindly. Adds are content-idempotent, but the SQLite commit can
precede a transport timeout or index failure; preserve or recover the stable returned ID.

## MCP owner

Install the MCP surface and pass it the same process-owned instance:

```bash
uv add "mindbridge[local,mcp]"
```

```python
from mindbridge import Memory
from mindbridge.api.mcp import build_mcp_server

with Memory.from_config(
    {
        "data_dir": "/var/lib/mindbridge/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    build_mcp_server(memory).run("stdio")
```

`build_mcp_server()` does not close `memory`. It reads `memory.capabilities` once while building
the server and publishes it as the server instructions, so build the server after the composition
is final. A stdio server inherits the host user's filesystem, environment, and model credentials.
If the host chooses SSE or streamable HTTP, it must add authentication, authorization, TLS, request
limits, and rate limiting. The server publishes fifteen tools; MCP error subjects may contain
owner-local paths.

## Media and model placement

Python accepts regular local `Path` values, inline `Blob` bytes, and `AssetRef` values from the
same data directory. REST and MCP accept inline data or stored asset IDs, never server paths or
HTTP(S) URLs. Fetch remote media in application code so redirects, SSRF controls, credentials,
timeouts, retries, download limits, and telemetry remain in the application's network policy.

After a successful add, MindBridge owns an immutable copy under `assets/`; the source file is no
longer required. Keep SQLite and `assets/` together through backup and restore.

Model placement is explicit. Local adapters accept or delegate device selection; custom backends
may use local or remote inference. MindBridge does not schedule GPUs, select quantization for
models, or move work between edge and cloud.
