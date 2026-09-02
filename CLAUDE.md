# CLAUDE.md

Follow [AGENTS.md](AGENTS.md). It contains the binding repository structure, storage invariants,
public-contract rules, coding style, test requirements, and quality gates.

## Sources of truth

Do not maintain another copy of MindBridge's API inventory in this file. Verify behavior against:

- `src/mindbridge/__init__.py`, `memory.py`, `types.py`, and `exceptions.py` for the public Python
  surface;
- `src/mindbridge/api/app.py` and `api/mcp.py` for REST and the fourteen MCP tools;
- `src/mindbridge/cli.py` and `benchmarks/cli.py` for console entry points;
- [the architecture guide](docs/architecture.md) for storage, routing, isolation, and extension
  boundaries;
- [the documentation index](docs/README.md) for user and operator references.

Tests are executable contracts. If prose and code disagree, establish the intended behavior from
the public tests and implementation, then update the owning reference instead of copying a
correction into several pages.

## Working rules

- Keep `Memory` as the execution plane; REST, MCP, and CLI translate inputs and outputs only.
- Route models by declared capability and inject provider clients explicitly.
- Keep SQLite authoritative and Zvec rebuildable. One physical `data_dir` has one live owner.
- Keep product modules independent of benchmark modules; behavior benchmarks use the public SDK.
- Update tests and the owning documentation in the same patch when a public contract changes.
- Run the repository gates in `AGENTS.md`; documentation changes also run the pinned Markdown and
  link checks in [CONTRIBUTING.md](CONTRIBUTING.md).
