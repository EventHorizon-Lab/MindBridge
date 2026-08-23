# Contributing to MindBridge

Thanks for considering a contribution. This document covers setup, the quality gates, and the
standards a change is reviewed against.

Repository conventions that apply to automated agents as well as people are in
[AGENTS.md](AGENTS.md); this file is the human-facing expansion of it.

## Setup

MindBridge supports Python 3.10 through 3.14, and CI runs the whole gate on every one of
them. 3.10 is the compatibility floor because several edge platform images — JetPack,
D-Robotics RDK, Rockchip RKNN — still ship it. The ceiling is the newest release the matrix
covers, so raising it means adding a leg and reading what it says, not editing one number.

Test 3.14 against a released build. A `uv` old enough to offer only `3.14.0b3` fails the whole
suite at collection — b3's private `typing._eval_type` has no `prefer_fwd_module` parameter, which
Pydantic passes — and that says nothing about this tree. `uv python install 3.14` on a current uv,
or a conda-forge `python=3.14.x`, both work.

The project uses [uv](https://docs.astral.sh/uv/) with a checked-in `uv.lock`. That lockfile is
authoritative; `pip install -e .` will not reproduce it.

```bash
git clone https://github.com/EventHorizon-Lab/MindBridge.git
cd MindBridge
uv sync --all-groups --extra edge --extra media --extra server
```

That is the set CI installs, so the clipping tests run here rather than skipping. `uv sync` is an
exact sync: to add the local Jina embedder, extend that same command with `--extra cloud-models`
rather than syncing it alone, which would uninstall everything above. It pulls torch, so skip it
unless you need it.

For everything at once — every scenario plus the benchmark harness and the local models:

```bash
uv sync --all-groups --all-extras
```

### A database for the tests that need one

```bash
docker compose up -d postgres redis
for migration in migrations/*.sql; do
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U mindbridge -d mindbridge < "$migration"
done
docker compose exec postgres createdb -U mindbridge mindbridge_test
export MINDBRIDGE_TEST_DATABASE_URL=postgresql://mindbridge:mindbridge@localhost:5432/mindbridge_test
```

The integration fixture refuses to rebuild a database whose name does not end in `_test`.

## Quality gates

These four must pass, and they are exactly what CI runs:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
git diff --check
```

`mypy` runs in **strict** mode over `src` and `tests`, targeting whichever interpreter invokes
it rather than a pinned version, so each matrix leg checks the branches its own version takes.
Ruff's `target-version = "py310"` is what holds the tree to syntax the floor can parse.

Ruff enforces `ANN`, `ASYNC`, `B`, `C4`, `C90`, `DTZ`, `E4`, `E7`, `E9`, `F`, `I`, `RUF`, `SIM`,
and `UP`, with a McCabe complexity ceiling of 10 and a 100-character line length.

`ANN401` bans `Any` in signatures. When you genuinely need to accept a structural type, write a
`Protocol` shim rather than reaching for `Any`.

### Markdown

Documentation changes must pass the same tool versions CI uses. The pinned Docker images matter —
`npx markdownlint-cli2` is old enough to miss rules the pinned version enforces:

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input './*.md' './docs/**/*.md'
```

### The integration gate

This is the one that catches people out.

Without `MINDBRIDGE_TEST_DATABASE_URL`, the **entire** integration suite skips — including
`tests/benchmarks/golden_recall.json`, the deterministic retrieval gate that exercises dense
evidence recall, exact text recall, temporal exclusion, and unsupported-query abstention through
the production kernel and the PostgreSQL/pgvector path.

A green `pytest` run may therefore never have touched the production store. Any change affecting
recall, consolidation, or deletion must be validated with the gate that turns a missing database
into a failure:

```bash
MINDBRIDGE_REQUIRE_INTEGRATION=1 uv run pytest -W error
```

CI provides a PostgreSQL service and sets `MINDBRIDGE_TEST_DATABASE_URL`, so the integration
suite does run there. It does not currently set `MINDBRIDGE_REQUIRE_INTEGRATION=1`, which means a
misconfigured service would degrade to a skip rather than a failure — run the required form
locally rather than relying on CI to catch it.

Run the gates again **after** merging your base branch, not only before. A clean merge can still
produce a broken tree: two branches each adding migration `00NN` merge without a conflict and
collide on the primary key at apply time.

## Standards

### Tests

pytest with pytest-asyncio in auto mode. New behaviour needs the smallest test that fails when
the behaviour regresses:

| Kind | Location | For |
| --- | --- | --- |
| Unit | `tests/unit/` | Local logic. |
| Contract | `tests/contracts/` | Public schemas. |
| Integration | `tests/integration/` | PostgreSQL and pgvector paths. Mark `pytest.mark.integration`. |
| Benchmark fixtures | `tests/benchmarks/` | Deterministic recall gates. |

`pytest -W error` is how CI runs, so a warning is a failure. The one that catches people is
SQLite: `with sqlite3.connect(...)` ends the transaction on exit and leaves the handle open, which
3.13 and later report as a `ResourceWarning`. Wrap it — `with closing(sqlite3.connect(path)) as
connection, connection:` — or use `mindbridge.edge._sqlite.connect`, which does both.

There is no numeric coverage threshold. There is a stronger requirement: **prove the test can
fail.** Break the code deliberately and watch it go red before you trust it. Assertions that
restate the implementation, ablations that are secretly no-ops, and barriers too weak to catch the
race they name have all shipped here and passed review. A test that cannot fail is worse than no
test, because it reads as coverage.

### Module boundaries

Dependencies point inward: `core` → `application` → `infrastructure`/`models` → `api`.

`benchmarks/` is a leaf. It may call only the public SDK and contracts, and **no product module may
import it**. An AST guard enforces this, including against string references — which is why
`mindbridge` and `mindbridge-bench` are separate console scripts. The duplicated argument parser on
the benchmarks side is the price of that rule; do not consolidate it.

### Naming and style

Follow the surrounding code. Match its comment density, naming, and idiom rather than importing a
different house style into one file.

Comments should explain *why*, not *what*. The existing codebase is unusually good about this —
several comments record a constraint discovered the hard way, and those are load-bearing. Do not
delete a comment because it looks like prose.

Markdown: short sections, descriptive headings, fenced code blocks for multi-line examples,
backticks for commands and paths. UTF-8, LF endings, trailing newline.

### Public contracts

Everything in `mindbridge.contracts` is a published contract shared by REST, MCP, and the Python
SDK. Changing one changes all three.

Entry points in `pyproject.toml` are a documented external contract too. If you change packaging,
**verify the built artifact** — `[project.scripts]` entries are written into `entry_points.txt`
independently of build `exclude` rules, so it is entirely possible to produce six green CI jobs and
a wheel that raises `ModuleNotFoundError` on install.

An OpenAPI snapshot test records drift. Note what it does and does not do: it *records* drift, it
does not *prevent* it. An invariant that matters needs a guard beyond the snapshot.

### Migrations

Numbered SQL in `migrations/`, applied in order, never edited once merged. Take the next free
number — and re-check it after merging your base branch, since two branches can each take `00NN`
and merge cleanly.

New tenant-scoped tables must satisfy the RLS gate, which checks for exact set equality against the
expected policy set. Copy the pattern from `0015` rather than improvising.

## Pull requests

Commits use a concise imperative subject: `Add contributor guide`, not `added contributor guide`
or `feat: contributor guide`. Keep each commit focused.

A pull request should say what changed and why, list the validation actually performed, and link
relevant issues. Call out new dependencies, configuration, and follow-up work explicitly.
Screenshots only for visible UI changes.

"Validation performed" means commands you ran and their outcome. If the integration suite skipped,
say so — an unstated skip reads as a pass.

### Review will reject

- `Any` in a signature where a `Protocol` would do.
- A test that cannot fail.
- Recall, consolidation, or deletion changes with no integration evidence.
- A product module importing `benchmarks/`.
- New configuration surface that could have been a `*_CONFIG_JSON` key. Fallback environment
  variables are reserved for credentials and model identity.
- Silent failure. Refuse loudly instead: a wrong configuration should fail startup, not degrade a
  request an hour later.
- Deleting a rationale comment along with the code it explained.

## Reporting bugs

Open an issue with the version, what you expected, what happened, and the smallest reproduction
you have. Include the `trace_id` from a failing response if there is one — it identifies the whole
request in telemetry and carries no user content.

**Security vulnerabilities do not go in the issue tracker.** See [SECURITY.md](SECURITY.md).

## Proposing changes

For anything structural, open an issue first. Design documents live in `docs/superpowers/specs/`
and plans in `docs/superpowers/plans/`; a substantial change is easier to review with one of those
in front of it than as a large diff.

Reversing a decision in [the ADR log](docs/technical-architecture.md#17-关键架构决策记录) is an
architecture change. Those decisions each name the cost they accept and the condition that would
justify revisiting them — argue against the condition, not around it.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
