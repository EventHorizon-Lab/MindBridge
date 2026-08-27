# Deployment

MindBridge runs inside one application process and owns one local data directory. The primary
deployment invariant is one directory, one process, one `Memory` instance.

## Embedded Python

Install the default local embedding and speech runtime:

```bash
uv add "mindbridge[local]"
```

Use an absolute persistent path and deterministic lifecycle:

```python
from mindbridge import Memory

with Memory("/var/lib/my-agent/memory") as memory:
    memory.add("The support rotation changes on Monday.")
```

The process account needs read/write permission on the complete MindBridge directory and read
permission on any Path inputs. Source files are copied into MindBridge, so they do not need to
remain after a successful add. On POSIX, opening the store forces the top-level directory to
owner-only mode `0700`, even if an operator created it with broader group permissions.

## Configure model operations

Embedding is local Jina v5 Omni and speech is local FunASR by default. Generation uses the shared
OpenAI-compatible defaults:

```bash
export OPENAI_API_KEY="model-provider-secret"
export OPENAI_BASE_URL=https://models.example.com/v1
```

A heterogeneous deployment can point generation elsewhere and use a combined backend for remote
transcription:

```bash
export MINDBRIDGE_GENERATION_BASE_URL=https://generation.example.com/v1
export MINDBRIDGE_GENERATION_API_KEY="generation-secret"

export MINDBRIDGE_GENERATION_MODALITIES=text,image,video
```

Capabilities must match the deployed endpoints. Choose another local model with
`SentenceTransformersEmbedder`, or pass an explicit combined `OpenAIHTTP`/`ModelBackend` to move
embedding/transcription to a cloud or edge endpoint. See [configuration](configuration.md). When using a vLLM
service, verify the required route and media-part support in that deployment.

## REST server

Install the optional transport:

```bash
uv add "mindbridge[local,server]"
```

For local development:

```bash
mindbridge serve --data-dir .mindbridge
```

The command always starts one worker. It refuses a non-loopback bind unless
`MINDBRIDGE_API_KEY` and a TLS certificate/key pair are configured:

```bash
export MINDBRIDGE_DATA_DIR=/var/lib/mindbridge/assistant
export MINDBRIDGE_API_KEY="replace-with-a-secret"

mindbridge serve \
  --data-dir "$MINDBRIDGE_DATA_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --tls-certfile /etc/mindbridge/tls/fullchain.pem \
  --tls-keyfile /etc/mindbridge/tls/privkey.pem
```

For ASGI composition, pass storage and authentication explicitly:

```python
import os

from mindbridge.api import create_app

data_dir = os.environ.get("MINDBRIDGE_DATA_DIR", "/var/lib/mindbridge/default")
service_key = os.environ.get("MINDBRIDGE_API_KEY") or None

app = create_app(data_dir=data_dir, api_key=service_key)
```

Then start exactly one worker behind trusted TLS termination or with Uvicorn TLS:

```bash
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --ssl-certfile /etc/mindbridge/tls/fullchain.pem \
  --ssl-keyfile /etc/mindbridge/tls/privkey.pem
```

`MINDBRIDGE_DATA_DIR` is a shell convenience; the CLI receives it through `--data-dir`.
`create_app` does not read deployment variables automatically. `Memory` does read the model
variables documented in [configuration](configuration.md).

## Why one worker is mandatory

Uvicorn and Gunicorn workers are separate processes. Each would try to own the same SQLite, CAS,
and Zvec directory, and the second is rejected immediately. Do not use multiple workers, pre-fork
servers, process auto-scaling, or overlapping rolling deployments against one path.

Threads and asynchronous HTTP requests inside the one process are coordinated by `Memory`.
Remote model calls can overlap; short SQLite commit/outbox and Zvec access sections serialize to
preserve consistency. Independent services on one machine use different directories and ports;
they do not share state.

## Persistent volume

Persist and back up the complete directory:

- `state.sqlite3` and its WAL state.
- `assets/`, which contains authoritative media bytes.
- `zvec/`, which is derived but avoids rebuild time after restore.
- `.mindbridge.lock`, which is harmless when no process owns its operating-system lock.

Keep the directory on a durable local filesystem that supports advisory locks and SQLite WAL.
Validate locking, atomic rename, directory fsync, and crash behavior before using a network-mounted
filesystem.

Capacity must cover SQLite records and FP32 embeddings, content-addressed media, WAL growth, Zvec
normal/rebuild space, and a separate backup. Media can dominate capacity even though duplicate
bytes are stored once.

## Network and media security

When `api_key` is set, every `/v1` route requires `Authorization: Bearer <key>`. Authentication is
checked before reading the body, and request bodies over 8 MiB are rejected. `/healthz` is public.
The shared inbound key is service authentication, not user identity.

Remote media ingestion is separately denied by default. Enable exact public hostnames only:

```bash
export MINDBRIDGE_ALLOWED_URL_HOSTS=media.example,cdn.example
```

Every redirect is rechecked; private, loopback, link-local, and otherwise non-public resolutions
are refused. Each hop connects to a public IP selected from the validated DNS result while keeping
the original hostname for TLS SNI and HTTP, preventing a later resolver change from redirecting the
connection. A concrete requested MIME must match exactly; a family hint matches only its declared
media family. Restrict outbound network policy to the allowed media and model hosts as defense in
depth.

`MINDBRIDGE_MEDIA_TRANSPORT=file` exposes local asset file URLs to the model endpoint. Use it only
with a trusted, co-located backend that can read the same files. The default `data` transport sends
base64 media over the model connection and is safer for remote endpoints, at the cost of encoding
and request size. The built-in backend rejects an embedding or generation call above 64 MiB of
aggregate raw media before base64 expansion. Use co-located `file` transport or a custom streaming
or file-upload backend for larger video; upstream provider limits may be lower.

Keep inbound service keys, outbound model keys, media content, and local paths out of access logs.
Use filesystem encryption and process-account permissions appropriate for the data.

## Startup and shutdown

Startup opens SQLite, verifies the schema/vector identity, checkpoints a missing index, and drains
pending work. A startup failure should stop the deployment; do not serve a partially opened store.

Allow graceful shutdown so FastAPI lifespan closes its owned `Memory`. Before backup, relocation,
or replacement, stop the old process and confirm it released the directory lock.

See [operations](operations.md) for backup and recovery.
