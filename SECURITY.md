# Security policy

## Reporting a vulnerability

Do not open a public issue for a security vulnerability. Use GitHub private vulnerability
reporting: **Security → Advisories → Report a vulnerability**.

Include the affected version or commit, component, reproduction, and impact when possible. Expect
an acknowledgement within a few days and allow a reasonable remediation window before public
disclosure.

## Supported versions

MindBridge is pre-1.0. Security fixes land on `master`; there are no maintained release branches.

## Security model

### Filesystem isolation

One `data_dir` is one memory domain. MindBridge has no tenant, user, or request scope inside that
directory. Any process that can read the directory can read its SQLite records, metadata, and FP32
embeddings.

MindBridge holds an operating-system lock so two live instances cannot own the same directory. On
POSIX, it creates new data directories with mode `0700` and sets `state.sqlite3` to `0600`; it does
set an existing top-level data directory back to `0700`. Operators remain responsible for parent
directory permissions, host account isolation, full-disk encryption, and backup permissions.

Give applications with different trust boundaries different operating-system accounts and
directories. Do not use metadata as an authorization boundary.

### REST deployment boundary

`create_app(memory=...)` is an unauthenticated ASGI application. MindBridge does not ship a product
server. Internet-facing deployments must host the app behind their existing authenticated gateway,
service mesh, or ASGI authentication middleware.

`/v1` bodies larger than 8 MiB are rejected before framework parsing, text is limited to 65,536
characters, and canonical metadata is limited to 262,144 UTF-8 bytes.

MindBridge has no user authentication, rate limiter, quota, key rotation store, or audit log. The
gateway must provide the identity, authorization, TLS, rate-limit, and audit controls required by
the deployment.

### MCP

The MCP server uses stdio and inherits the privileges of its host process. It has no network
listener or separate authentication layer. Configure the MCP client so only the intended local
principal can launch it and access its `data_dir`.

### Model data boundary

`add` and `search` send routed text and media through the supplied embedding backend. Audio fallback
also sends media through the supplied transcription backend. `ask` sends the question and retrieved
content, timestamps, metadata, and media through the supplied generation backend. Choose provider
clients and retention policies appropriate for that data, and use HTTPS outside a trusted local
network.

Stored text is untrusted model input. MindBridge separates it from the system instruction and asks
the model to treat it as evidence, but it cannot guarantee that a model will resist prompt
injection or hallucination. Applications should display or inspect `AnswerResult.hits` for
high-impact decisions.

### Storage, deletion, and backups

SQLite is authoritative; Zvec is a rebuildable search projection, not an access-control layer.
Public search hydrates Zvec IDs from SQLite, so stale index entries are not returned after logical
deletion.

`delete` is logical deletion, not guaranteed secure erasure from SQLite free pages, WAL files,
filesystem snapshots, or backups. Use encrypted storage and a documented backup-retention process
when secure deletion matters. Stop the owning process before copying or restoring a directory.

### Credentials and logs

Keep gateway credentials separate from provider credentials. Provider SDK clients own credential
loading and redaction; MindBridge does not copy model keys into its configuration. Public
model/storage errors omit provider bodies and filesystem details. Do not log request bodies,
authorization headers, memory content, or metadata in surrounding application code.

## Deployment hardening

| Control | Recommendation |
| --- | --- |
| Process | One process and one `Memory` owner per directory; do not pre-fork an open instance |
| Filesystem | Dedicated account, local durable filesystem, restrictive parent permissions |
| Encryption | Encrypt the host volume and backups when memories are sensitive |
| REST auth | Enforce identity and authorization at the gateway or ASGI middleware boundary |
| TLS | Terminate TLS before the unauthenticated ASGI app |
| Network | Restrict inbound access and use HTTPS for remote model endpoints |
| MCP | Keep stdio local to the intended principal |
| Backups | Stop the owner, protect the complete directory, and test restore into a new path |

## Out of scope

The following are not security guarantees provided by MindBridge:

- Per-user authorization, tenant isolation, quotas, or rate limiting inside one directory.
- Secure erasure from storage media or backups.
- Confidentiality from the configured model provider.
- Correctness or prompt-injection resistance of model output.
- Safety when another process can read or modify the data directory.
- Availability under intentionally expensive but valid model or indexing workloads.

## Dependencies

Dependencies are resolved in `uv.lock`. Report a vulnerable direct or transitive dependency through
the same private channel.
