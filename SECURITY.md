# Security policy

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report it through GitHub private vulnerability reporting: on this repository, open
**Security → Advisories → Report a vulnerability**. That keeps the report private until a
fix is available.

Include what you can: affected version or commit, component (API, worker, edge, SDK, MCP),
reproduction steps, and impact. A partial report is better than a delayed one.

Expect an acknowledgement within a few days, an assessment with a severity and a rough timeline
after triage, and a security advisory crediting you when the fix ships — unless you prefer
otherwise.

Please give us a reasonable window to fix before public disclosure.

## Supported versions

MindBridge is pre-1.0. Security fixes land on `master`; there are no maintained release branches
yet. Track `master` if you deploy this.

## Security model

Knowing what MindBridge does and does not defend against is the difference between a safe
deployment and a surprising one.

### Tenant isolation

Two independent layers, deliberately not one.

**Database.** Migration `0005` enables **forced** row-level security on every table carrying a
`tenant_id`, through the non-login `mindbridge_runtime` role. Each store transaction sets one
tenant locally. This is not a `WHERE` clause a missing filter could defeat — the database refuses
to return another tenant's rows even to a query that asks for them.

**API.** Every key is bound to an explicit tenant allowlist. Every `/v1` operation rejects a body
or query `tenant_id` outside it with `tenant_access_denied`, whether or not the row exists.

> **Never grant the runtime login `SUPERUSER` or `BYPASSRLS`.** Either one silently disables
> tenant isolation entirely while the system continues to look completely healthy. This is the
> single highest-impact misconfiguration available.

### Credentials

Credentials come from the environment, never from a CLI flag — so a shell history, a process list,
and a systemd unit never carry one.

API keys must be at least 32 characters; shorter ones fail at startup. MindBridge stores no
plaintext copy, only a digest, and comparison is constant-time. Multiple keys per tenant exist so
you can rotate without downtime: add the new key, deploy, move clients, remove the old one.

Object storage credentials are never copied into MindBridge configuration. Boto3's standard chain
resolves them, so one AWS configuration serves MindBridge and everything else on the host.

### Media access

Media is reached through short-lived signed URLs scoped to
`s3://<bucket>/tenants/<tenant_id>/<key>`. A URI outside its tenant prefix is refused before it
can become a path traversal. Buckets should not be public.

`MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL` moves **every** signed URL, including derived-clip
URLs. Signatures cover the host, so both names must reach the same bucket.

### Edge privacy boundary

The device is treated as the more sensitive side, and the boundary is narrow by construction.

**Never leaves the device:** raw face and voice embeddings, and the device encryption key. Samples
are AES-256-GCM encrypted in a local mode-`0600` SQLite store, with a 32-byte key from the device
TPM or a secret manager.

**Crosses the boundary:** anonymous identity IDs, time ranges, optional transcripts, identity
scope, normalized face boxes, and the media the deployment chose to upload.

AWS credentials and the MindBridge API key are never written to that SQLite file.

### Deletion

`forget()` is transitive and durable. Deleting an observation removes everything derived from it,
including identity samples the edge learned from that source. Tombstones are content-free by
construction and survive physical erasure, so an offline device can reconcile on reconnect and a
restore can be reconciled against them.

`propagation_state: complete` means deletion finished in central PostgreSQL and object storage.
It is not an acknowledgement from an offline edge device; verify that device's tombstone
reconciliation separately.

Two obligations that must be planned together: a backup outliving a deletion reintroduces deleted
content on restore. Decide your backup retention window and your deletion propagation window at
the same time, and rehearse restore-then-reconcile.

### Telemetry

MindBridge captures no authorization headers, request bodies, prompts, memory text, or media.
Span attributes are bounded enumerations and counts. `trace_id` values contain no user data.

### Model input

The generator receives memory content and media by design — that is the product. Two consequences
worth stating plainly:

- **Your model provider sees your users' memory content.** Choose the endpoint accordingly, or
  self-host it.
- **Content in memory reaches a model that produces answers.** Prompt-injection resistance is a
  property of the model and prompts you configure. MindBridge grounds answers in evidence and
  returns that evidence for verification; it does not claim to make an untrusted-content pipeline
  safe.

## Deployment hardening

| Control | Recommendation |
| --- | --- |
| Database login | Grant `mindbridge_runtime`. Never `SUPERUSER` or `BYPASSRLS`. |
| API keys | ≥ 32 random characters. Rotate through the multi-key list. |
| Object storage | Private buckets. Signed URLs only. |
| MCP | stdio behind process isolation, or an authenticated gateway. Never an open HTTP listener. |
| AML routes | Leave `MINDBRIDGE_AML_API_KEY` unset in production. |
| TLS | Terminate in front of the API. MindBridge does not manage certificates. |
| Network | Datastores and model endpoints on private networking. |
| Edge key | From TPM or a secret manager. Never on disk in plaintext. |
| Backups | Encrypt at rest, and reconcile against tombstones on restore. |

## Out of scope

Not vulnerabilities in this project:

- Missing per-tenant quotas. There is no rate limiting or ingest quota; use a gateway.
- Model behaviour, including hallucination and prompt injection through memory content.
- Denial of service through legitimate expensive operations such as large `enumerate` scopes.
- Vulnerabilities in the model endpoints you configure.
- Anything requiring `SUPERUSER` or `BYPASSRLS` on the database login, which is documented as
  unsupported.

## Dependencies

Dependencies are pinned in `uv.lock`. Report a vulnerable transitive dependency through the same
private channel; a fix is usually a lock update.
