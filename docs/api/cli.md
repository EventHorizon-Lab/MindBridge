# Command-line API

The `mindbridge` command exposes only local lifecycle operations. It does not create logical stores
or manage background services.

## Help and version

```bash
mindbridge --help
mindbridge --version
```

Running `mindbridge` with no arguments prints help and exits successfully.

## Serve REST

Install the server extra:

```bash
uv add "mindbridge[local,server]"
```

Start one HTTP process:

```bash
mindbridge serve \
  --data-dir /var/lib/mindbridge/assistant \
  --host 127.0.0.1 \
  --port 8000
```

Defaults are `.mindbridge`, `127.0.0.1`, and port `8000`. The command always uses one Uvicorn
worker.

Binding outside a loopback address requires an inbound service key and a TLS certificate/key pair:

```bash
export MINDBRIDGE_API_KEY="replace-with-a-secret"
mindbridge serve \
  --data-dir /var/lib/mindbridge/assistant \
  --host 0.0.0.0 \
  --tls-certfile /etc/mindbridge/tls/fullchain.pem \
  --tls-keyfile /etc/mindbridge/tls/privkey.pem
```

The key protects `/v1` with bearer authentication, while the TLS files protect credentials and
memory content in transit. The service key is separate from `OPENAI_API_KEY`, which is sent to the
model endpoint. See [deployment](../deployment.md).

## Rebuild the index

The service that normally owns the directory must be stopped first:

```bash
mindbridge reindex --data-dir /var/lib/mindbridge/assistant
```

The command opens one `Memory`, rebuilds Zvec from SQLite, and closes it. It does not re-embed
stored content.

## Optimize the index

Run with no other owner of the directory:

```bash
mindbridge optimize --data-dir /var/lib/mindbridge/assistant
```

This compacts and flushes Zvec, then releases the directory.

## Serve MCP

Install the MCP extra and run the stdio server:

```bash
uv add "mindbridge[local,mcp]"
mindbridge mcp --data-dir /var/lib/mindbridge/assistant
```

The standalone alias has the same arguments:

```bash
mindbridge-mcp --data-dir /var/lib/mindbridge/assistant
```

The MCP process becomes the only owner of that directory until it exits. See the
[MCP reference](mcp.md).

## Local-index benchmark

Choose an empty directory:

```bash
python -m mindbridge.benchmarks.local_index_benchmark \
  --data-dir .benchmarks/local-index/example \
  --rows 1000 \
  --dimension 128 \
  --queries 20 \
  --k 10 \
  --seed 42
```

The command prints one compact JSON document. It refuses a non-empty directory so an old SQLite or
Zvec collection cannot contaminate a measurement. This synthetic adapter benchmark does not call
the model endpoint or measure answer quality. See [benchmarking](../benchmarking.md).

## Exit behavior

Successful commands return 0, command failures return 1, command-line usage errors return 2, and
an interrupted operation returns 130. Failures are printed as one line on standard error.
