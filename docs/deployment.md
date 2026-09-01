# Deployment

MindBridge runs inside the host process. Deploy one live `Memory` owner for each physical
`data_dir`; applications that need separate memory domains use separate directories.

Prefer embedded Python when one application owns the memory. Add REST or MCP only when an existing
host needs that protocol boundary; neither transport creates a second storage service.

## Embedded Python

Construct provider clients and backends explicitly, then close caller-owned clients separately:

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

`Memory` closes the backend objects supplied to it. `OpenAIModels` intentionally leaves supplied
SDK clients open because the application may share them. Do not construct `Memory` before forking;
a child process cannot use or close the parent's instance.

The Jina adapter in this example runs pinned upstream model code and uses non-commercial weights;
review [embedding choices](configuration.md#embedding-choices) before selecting it for deployment.

For a long-running loop, keep one instance open. A local `mindbridge` CLI invocation opens and
closes one instance per command; use the Python SDK or address a running REST owner with `--url`
when repeated operations must share one process.

## REST service

Install the optional server surface:

```bash
uv add "mindbridge[local,openai,server]"
```

Expose an application-owned instance:

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

Run exactly one worker for that directory:

```bash
uvicorn my_application:app --host 127.0.0.1 --port 8000 --workers 1
```

The application must register shutdown handling for both `memory.close()` and `client.close()`.
`create_app` does not own either resource.

The REST adapter has no authentication, authorization, TLS, rate limiting, or audit log. Bind to a
trusted interface or place it behind the application's existing gateway or middleware. That outer
layer also owns request identity and retry policy. Do not retry a timed-out non-idempotent request
blindly; adds are content-idempotent, but the caller still needs the returned stable ID.

Never start multiple ASGI workers against one directory. Giving each worker a different directory
is valid only when each is intentionally a separate memory domain.

## MCP

Install the MCP extra and supply the same caller-owned instance:

```python
from mindbridge.api.mcp import build_mcp_server

server = build_mcp_server(memory)
server.run("stdio")
```

`build_mcp_server` does not close `memory`. A stdio server inherits the host process user,
filesystem access, environment, and model credentials; sandbox it as part of the host
application.

The returned server also accepts SSE and streamable-HTTP transports. If the host selects one, it
must add authentication, authorization, TLS, request limits, and rate limiting. MindBridge adds
none of those controls, and MCP error subjects can expose owner-local paths.

## Edge and filesystem

Edge deployment uses the same embedded topology; no separate MindBridge service is required.
Before selecting a device, verify:

- Python 3.10 through 3.14 and a compatible 64-bit Zvec wheel are available.
- Wheels and runtime support exist for every selected optional model backend.
- RAM and accelerator memory cover model inference, query batches, and Zvec's memory-mapped index.
- Local storage covers SQLite, original media, authoritative FP32 embeddings, and the derived
  index.
- The filesystem provides reliable SQLite WAL, file locking, atomic rename, and durable local
  writes.

Keep the complete directory on one local durable filesystem. SQLite, `assets/`, `.mindbridge.lock`,
and `zvec/` form one operational unit even though Zvec can be rebuilt. Network filesystems and
shared-volume multi-writer topologies are not supported deployment shortcuts.

On POSIX, startup sets the top-level directory to `0700`, and MindBridge creates its SQLite, lock,
staging, and CAS files with restrictive permissions. The operator still owns parent-directory
permissions, service accounts, disk encryption, backup access, and retention policy.

Model placement is explicit. `JinaOmniEmbedder` and `SentenceTransformersEmbedder` accept a device;
`FunASRTranscriber` delegates to `funasr.AutoModel`; generation may use a local or remote custom
backend. MindBridge does not schedule GPUs, choose quantization, or move work between edge and cloud.

## Media and network boundary

Python accepts regular local `Path` values, inline `Blob` bytes, and `AssetRef` values from the same
directory. REST and MCP accept inline data or stored asset IDs, never server paths or HTTP(S) URLs.

Fetch remote media in application code using its established client. This keeps credentials,
redirects, SSRF controls, timeouts, retries, download limits, and network telemetry in one policy
boundary. After a successful add, the original source file may be removed because MindBridge has
copied its immutable bytes into `assets/`.

See [operations](operations.md) for health checks, backups, restore, repair, and telemetry.
