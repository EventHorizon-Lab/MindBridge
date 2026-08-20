# Repository Guidelines

## Project Structure & Module Organization

MindBridge is a Python package under `src/mindbridge/`. Keep domain types in `core/`, use cases and ports in `application/`, external adapters in `infrastructure/` and `models/`, protocol entry points in `api/`, Jetson/robot code in `edge/`, and official dataset adapters plus their runners in `benchmarks/`. `benchmarks/` may only call the public SDK and contracts; no product module may import it. Tests live in `tests/unit/`, `tests/contracts/`, and `tests/integration/`; deterministic benchmark fixtures live in `tests/benchmarks/`. Numbered PostgreSQL migrations are in `migrations/`, reproducibility manifests are in `benchmarks/manifests/`, and documentation is in `docs/`, indexed by `docs/README.md`. Keep root files limited to project-wide documentation, dependency, deployment, and tooling configuration.

## Build, Test, and Development Commands

MindBridge supports Python 3.10 and 3.11 and uses `uv` with the checked-in `pyproject.toml` and `uv.lock`. Install the same set CI installs — development groups plus the Edge, media, and Server extras — with:

```bash
uv sync --all-groups --extra edge --extra media --extra server
```

`uv sync` is an exact sync: it uninstalls anything the named extras do not cover, so extend this command rather than running a second one for another extra.

Run the required quality gates before submitting changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
git diff --check
```

Markdown changes must also pass the pinned Markdown and link-check commands documented in
`CONTRIBUTING.md`. Note that `ruff format` also formats Python code blocks inside Markdown, so a
documentation change can fail the formatting gate. Deployment-specific installation commands are
in `docs/deployment.md`, and PostgreSQL integration setup is in `CONTRIBUTING.md`.

## Coding Style & Naming Conventions

Use UTF-8 text, LF line endings, and a trailing newline. Write Markdown with short sections, descriptive headings, fenced code blocks for multi-line examples, and backticks for commands and paths. Prefer clear, conventional names: uppercase names for standard root documents (for example, `CONTRIBUTING.md`) and the chosen language's established naming rules for source files. Avoid generated artifacts and editor-specific settings unless the project explicitly adopts them.

## Testing Guidelines

Tests use pytest with pytest-asyncio. New behavior should include the smallest test that fails when the behavior regresses: unit tests for local logic, contract tests for public schemas, and integration tests for PostgreSQL/pgvector paths. Mark database-dependent tests with `pytest.mark.integration`; configure their disposable database as documented in `CONTRIBUTING.md`. `tests/benchmarks/golden_recall.json` is the deterministic production-path recall gate. It only runs with `MINDBRIDGE_TEST_DATABASE_URL` configured, so any change to recall, consolidation, or deletion must be validated with `MINDBRIDGE_REQUIRE_INTEGRATION=1 uv run pytest -W error`, which turns a missing database into a failure instead of a skip. There is no numeric coverage threshold, but all required quality gates above must pass. Documentation changes should also be checked for accurate commands, valid relative links, and readable rendered Markdown.

## Commit & Pull Request Guidelines

Use a concise, imperative commit subject such as `Add contributor guide`, and keep each commit focused. Pull requests should explain what changed and why, list validation performed, and link relevant issues. Include screenshots only for visible UI changes, and call out new dependencies, configuration, or follow-up work explicitly.
