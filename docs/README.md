# MindBridge documentation

MindBridge is an embedded multimodal memory library for text, image, video, audio, and combined
`omni` inputs. Start with the short path, then use the reference pages for exact contracts.

## Start here

- [Quick start](quickstart.md) — install, add text and media, retrieve, and upgrade.
- [Core concepts](concepts.md) — content, assets, identity, isolation, models, and indexing.
- [Configuration](configuration.md) — endpoints, capabilities, durable model spaces, and media policy.

## APIs

- [Python API](api/python-sdk.md) — `Memory`, content values, return values, and exceptions.
- [REST API](api/rest.md) — ordered content parts, endpoints, authentication, and errors.
- [MCP API](api/mcp.md) — five typed stdio tools over one local memory.
- [Command-line usage](api/cli.md) — server, MCP, reindex, and benchmark commands.

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
