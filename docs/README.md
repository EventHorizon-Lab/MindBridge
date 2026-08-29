# MindBridge documentation

MindBridge is an embedded multimodal memory library for text, image, video, audio, and combined
`omni` inputs. Start with the short path, then use the reference pages for exact contracts.

## Start here

- [Quick start](quickstart.md) — install, add text and media, retrieve, and upgrade.
- [Core concepts](concepts.md) — content, assets, identity, isolation, models, and indexing.
- [Memory types, time, and decay](memory-types-time-and-decay.md) — cognitive roles, temporal
  retrieval, decay, research basis, and limits.
- [Configuration](configuration.md) — explicit composition, provider ownership, and durable model spaces.

## APIs

- [Python API](api/python-sdk.md) — `Memory`, content values, return values, and exceptions.
- [REST API](api/rest.md) — ordered content parts, endpoints, deployment boundary, and errors.
- [MCP API](api/mcp.md) — five typed stdio tools over one local memory.
- [Command-line usage](api/cli.md) — benchmark command and application-owned service startup.

## Build and run

- [Architecture](architecture.md) — components, model routing, and consistency.
- [Technical architecture](technical-architecture.md) — SQLite, CAS, Zvec, trust boundaries, and concurrency.
- [Deployment](deployment.md) — one-process embedded and REST deployment.
- [Operations](operations.md) — backup, restore, rebuild, and observability.
- [Troubleshooting](troubleshooting.md) — startup, asset, model, and index failures.
- [Benchmarking](benchmarking.md) — evaluation tasks, uncertainty, isolation, and local-index metrics.

## Scope notes

- [Edge status](edge.md) — local media memory on constrained devices.

Historical service and account-scoping documents were removed because they describe contracts
that no longer exist. A `data_dir`, not a tenant or user field, is the isolation boundary.
