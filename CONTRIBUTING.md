# Contributing to MindBridge

Thanks for helping improve MindBridge. This guide covers local setup, quality gates, storage
invariants, and review expectations. Automated contributors must also follow
[AGENTS.md](AGENTS.md).

## Setup

MindBridge supports Python 3.10 through 3.14. The project uses
[uv](https://docs.astral.sh/uv/) and the checked-in `uv.lock`.

```bash
git clone https://github.com/EventHorizon-Lab/MindBridge.git
cd MindBridge
uv sync --locked --default-index https://pypi.org/simple --all-groups --all-extras
```

For core and REST work without the optional MCP transport:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --all-groups --extra local --extra openai --extra server
```

`uv sync` is exact. Add extras to the same command instead of running a second, narrower sync that
would remove packages from the first environment.

Use `uv add` and `uv remove` for dependency changes, then normalize and verify the shared lockfile
against PyPI even when your shell configures a private mirror:

```bash
uv lock --default-index https://pypi.org/simple
uv lock --check --default-index https://pypi.org/simple
```

No external database, cache, queue, or object store is required. Unit tests allocate isolated
temporary SQLite and Zvec directories. Integration tests that contact a real model endpoint are
marked `integration` and require credentials for that endpoint.

## Quality gates

Run all required gates before submitting:

```bash
uv lock --check --default-index https://pypi.org/simple
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest -W error
git diff --check
```

Mypy runs in strict mode over `src` and `tests`. Ruff holds the project to Python 3.10 syntax and
formats Python code blocks inside Markdown. Pytest treats warnings as errors.

Run a focused test while iterating, then run the complete gate before handoff. For example:

```bash
uv run --frozen pytest -W error tests/unit/test_memory_api.py
uv run --frozen pytest -W error tests/unit/api
```

### Markdown

Documentation changes must pass the same pinned tools and arguments as CI:

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input \
  --exclude '^https://penfieldlabs\.substack\.com/' \
  './*.md' './docs/**/*.md' './.github/**/*.md'
```

Do not add a host exclusion for a transient outage. Fix invalid local links and replace links that
no longer identify a stable source.

## Test standards

New behavior needs the smallest test that proves the intended contract can fail:

| Kind | Location | Purpose |
| --- | --- | --- |
| Unit | `tests/unit/` | Local logic and adapter behavior |
| Contract | `tests/contracts/` | Stable public values and schemas |
| Integration | `tests/integration/` | Explicit real-service boundaries |
| Benchmark | `tests/benchmarks/` | Deterministic quality fixtures |

Every local store test must use a distinct temporary directory. When testing concurrency, assert
that the same directory rejects a second owner and that different directories operate in parallel.
Never simulate isolation by adding a hidden scope field.

Warnings fail the suite. Close `Memory`, SQLite, Zvec, and HTTP resources deterministically by
using their context managers or an explicit `close()`.

There is no numeric coverage threshold. A new test should fail when the implementation is
deliberately broken; assertions that only restate the implementation are not useful evidence.

## Architecture rules

The supported local path is intentionally direct:

```text
Memory -> SQLite transaction -> durable outbox -> Zvec flush
       -> provider SDK adapter
```

- SQLite owns memory records, canonical embeddings, configuration metadata, and pending index
  operations.
- Zvec is a disposable search projection. It must be rebuildable from SQLite.
- A failed Zvec write must leave its outbox work pending.
- Stored embedding identity or dimension changes must fail at open time rather than mixing spaces.
- One data directory has one live owner. Server deployments use one worker.
- Metadata is payload, never an isolation or authorization mechanism.

Keep the public API small. Reuse existing values and errors before adding layers. Do not introduce
a repository interface, worker service, compatibility shim, or provider registry without a current
public use case that cannot be expressed by the existing boundary.

Benchmark packages are leaves: no product module may import them. Dataset and behavior runners use
the public SDK. The local-index microbenchmark is the narrow exception that measures the storage
adapters directly.

## Public contracts

Stable Python imports are re-exported from `mindbridge`. Stable REST routes are under `/v1`, with
one error envelope documented in [the REST reference](docs/api/rest.md). The MCP adapter exposes
fourteen tools documented in [the MCP reference](docs/api/mcp.md). Update tests and docs in the
same change whenever any surface changes.

The embedded release is a breaking line from the former service architecture. Do not restore
removed scoping parameters, service configuration, or compatibility aliases. Applications that
need separate memory domains allocate separate directories or separate processes.

## Documentation style

Use short sections, descriptive headings, fenced code blocks for multi-line examples, and backticks
for commands and paths. Use UTF-8, LF endings, and a trailing newline. Examples must be executable
against the current public API. Keep `docs/README.md` current when adding, renaming, or removing a
page.

## Pull requests

Use a concise imperative commit subject such as `Add index recovery test`. A pull request should:

- Explain what changed and why.
- List the exact validation performed and any skips.
- Link the relevant issue.
- Call out new dependencies, configuration, schema changes, or follow-up work.
- Include screenshots only for a visible interface change.

## Reporting issues

Open an issue with the MindBridge version, expected behavior, actual behavior, and the smallest
reproduction. Include the REST `trace_id` when available; it identifies a failing request without
containing memory content.

Report security vulnerabilities through [SECURITY.md](SECURITY.md), not the public issue tracker.

## License

Contributions are licensed under the [Apache License 2.0](LICENSE).
