# Repository Guidelines

## Project structure

MindBridge is a Python package under `src/mindbridge/`.

- Keep public values and exceptions in small top-level modules such as `types.py` and
  `exceptions.py`.
- Keep the developer-facing orchestration in `memory.py`.
- Keep durable local storage and search adapters in `infrastructure/local/`.
- Keep model clients in `models/` and protocol adapters in `api/`.
- Keep benchmark harnesses in `benchmarks/` or `src/mindbridge/benchmarks/`.
- Keep tests in `tests/unit/`, `tests/contracts/`, `tests/integration/`, and
  `tests/benchmarks/`.
- Keep user documentation in `docs/`, indexed by `docs/README.md`.

Product modules must not import benchmark modules. Dataset and behavior benchmarks should call the
public SDK. A narrowly scoped storage microbenchmark may use the local adapters when its purpose is
to measure SQLite or Zvec directly; do not let that exception become an alternate product API.

## Build and test

MindBridge supports Python 3.10 through 3.14 and uses `uv` with `pyproject.toml` and `uv.lock`.
Install every development group and optional surface with:

```bash
uv sync --locked --default-index https://pypi.org/simple --all-groups --all-extras
```

Run the required quality gates before submitting a change:

```bash
uv lock --check --default-index https://pypi.org/simple
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest -W error
git diff --check
```

Documentation changes must also pass the pinned Markdown and link commands in
`CONTRIBUTING.md`. Ruff formats Python code blocks in Markdown, so documentation examples are part
of the formatting gate.

## Storage and isolation rules

One physical `data_dir` is one running MindBridge instance. Never introduce logical account,
request, or benchmark scope into the product contract as a substitute for physical isolation.
Parallel tests and benchmark cases must each allocate a distinct temporary directory.

SQLite is authoritative for records, embeddings, store metadata, and the durable search-index
outbox. Zvec is derived and rebuildable. A durable write must commit SQLite before changing Zvec,
and an outbox operation must be acknowledged only after the Zvec flush succeeds.

Do not add a database service, worker queue, or object store for behavior the embedded runtime can
provide directly. New dependencies need a demonstrated public requirement and must live in the
narrowest relevant optional extra.

## Coding style

Use UTF-8, LF endings, and a trailing newline. Prefer deletion and reuse over compatibility layers
or speculative abstractions. Keep names conventional, public contracts explicit, and comments
focused on constraints that are not obvious from the code.

Use `apply_patch` for hand edits. Preserve unrelated work in a dirty tree. Do not run destructive
Git commands to discard another contributor's changes.

## Tests

New behavior needs the smallest test that fails when it regresses:

- Unit tests for local behavior and failure mapping.
- Contract tests for stable Python and REST values.
- Integration tests only for real external model services.
- Benchmark tests for deterministic isolation and metric calculation.

Use a fresh `tmp_path` for every local store. Test that a second owner of the same directory fails
immediately and that different directories work concurrently. Retrieval tests must verify that
SQLite hydration drops stale Zvec IDs and that a missing index rebuilds without re-embedding
stored content.

## Public contracts

The supported Python imports come from `mindbridge`; the supported HTTP surface is under `/v1`,
and the supported MCP surface is the seven tools in `mindbridge.api.mcp`. Changing a signature,
response type, exception, endpoint, tool schema, error code, on-disk schema, or console entry point
is a breaking change and needs tests and documentation in the same patch.

The product API has no implicit scoping identifiers. Metadata is application data, not an access
control or isolation boundary.

## Commits and pull requests

Use a concise imperative commit subject such as `Document local deployment`. Keep each commit
focused. Pull requests should explain what changed and why, list the validation actually run, and
call out dependency, configuration, schema, or compatibility changes.
