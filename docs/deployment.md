# Deployment

MindBridge is embedded. The host process constructs provider clients, adapters, and one `Memory`.

## Python process

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels

client = OpenAI(timeout=30.0, max_retries=3)
answerer = OpenAIModels(generation_client=client)
memory = Memory(
    "/var/lib/mindbridge/assistant",
    embedder=JinaOmniEmbedder(),
    answerer=answerer,
)

try:
    memory.add("Process-owned memory")
finally:
    memory.close()
    client.close()
```

Use one process owner per physical directory. Do not create `Memory` before forking.

## REST

Install the optional surface:

```bash
uv add "mindbridge[local,openai,server]"
```

Application module:

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

Run it with the application's ASGI stack:

```bash
uvicorn my_application:app --host 127.0.0.1 --port 8000 --workers 1
```

Register process shutdown hooks that close `memory` and `client`. `create_app` deliberately does
not own either resource.

The ASGI app has no authentication or TLS. Put it behind the same gateway, service mesh, or ASGI
middleware used by the rest of the application. The outer layer owns identity, authorization,
rate limiting, TLS, request retries, and audit logging.

Never use multiple workers against one directory. Allocate a different directory per worker only
when each worker is intentionally a different memory domain.

## MCP

```python
from mindbridge.api.mcp import build_mcp_server

with memory:
    build_mcp_server(memory).run("stdio")
```

The MCP server uses stdio and inherits the host process permissions. The caller owns model clients
and memory cleanup.

## Remote media

Fetch media in application code using its established HTTP client and security policy. Pass bytes
as `Blob` or a downloaded local file as `Path`. This preserves one place for authentication,
redirect handling, SSRF protection, retries, observability, and streaming limits.

## Filesystem

Store the complete directory on a local durable filesystem. The SQLite database, CAS, lock, and
Zvec directory move together. On POSIX, MindBridge enforces restrictive permissions on its own
files, but the operator remains responsible for parent directories, service accounts, encryption,
and backups.

## Health and observability

REST exposes `/healthz`, which reports process availability rather than provider readiness. Use
the provider SDK's own telemetry and the deployment platform for network metrics. MindBridge
errors use stable categories and intentionally omit provider response bodies.
