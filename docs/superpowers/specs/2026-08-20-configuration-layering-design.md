# Configuration layering

Split MindBridge's 33-name environment contract into a committed file for structure and a
short environment for credentials, and give an operator one command that reports what is
missing before a process starts.

Status: approved design, not yet implemented.

## Problem

`docs/quickstart.md` step 4 is sixteen `export` lines. They live in one shell. Closing the
terminal loses all sixteen, and nothing in the repository persists them: `.gitignore` has
ignored `.env` and `.env.local` since the bootstrap, but no code loads either file, no
`.env.example` exists, and no invocation passes `--env-file`.

Four distinct costs fall out of that, all reported by the maintainer:

1. **No home.** Configuration survives only as shell state.
2. **No discoverability.** Which of the 33 names a given role needs is recoverable only by
   reading the six-column matrix in `docs/configuration.md` and reasoning backwards.
3. **No pre-flight.** The only validator is starting the process. It fails on the *first*
   missing value, so an operator missing five discovers them one restart at a time.
4. **JSON in a shell variable.** `MINDBRIDGE_GENERATOR_CONFIG_JSON` and its three siblings
   carry multi-line JSON objects as strings — no comments, no sections, no diff worth
   reading, and a typo surfaces as a startup failure.

### The asymmetry underneath

Of the 33 names, **7 are credentials**, **1 is a debug switch**, and **25 are non-secret
structure**. Every one of those 25 pays the cost of the environment-only contract —
ephemerality, invisibility, JSON-as-string — while protecting nothing.

| Kind | Count | Names |
| --- | --: | --- |
| Credential | 7 | `DATABASE_URL`, `TASK_BROKER_URL`, `GENERATOR_API_KEY`, `EMBEDDER_API_KEY`, `TENANT_API_KEYS_JSON`, `API_KEY`, `AML_API_KEY` |
| Structure | 25 | `*_PLUGIN`, `*_ENDPOINT`, `*_MODEL_ID`, `*_MODEL_REVISION`, `*_CONFIG_JSON`, `EMBEDDING_*`, `OBJECT_STORAGE_*`, `MEDIA_SAMPLING_CONFIG_JSON`, `MEDIA_EMBEDDER_DEVICE`, `DATABASE_MAX_POOL_SIZE`, `MINIMUM_EMBEDDING_SIMILARITY`, `AML_TENANT_PREFIX` |
| Debug | 1 | `TRACEBACK` |

`DATABASE_URL` and `TASK_BROKER_URL` count as credentials because both carry a password in
their DSN. `AWS_*` is out of scope in both directions: boto3's own chain resolves it and
MindBridge holds no copy. Of the seven credentials, five are what a local API deployment must supply: `API_KEY` is read only by `edge sync`, and `AML_API_KEY` is an opt-in benchmark switch.

### What already works, and why it is not enough

Three properties of today's design are load-bearing and this change keeps all three:

- **Credentials never touch disk and never enter `ps` output.** `docs/configuration.md`
  opens by stating this, and it is why the 2026-08-11 bootstrap deleted the previous
  `config/*.yaml` tree. Nothing here puts a credential back on disk.
- **`from_environment(environ=...)` is injectable.** All four settings classes accept a
  `Mapping` so unit tests configure a process without touching the real environment.
- **Each variable is read in exactly one place.** `src/mindbridge/models/defaults.py:70-78`
  documents this deliberately: three copies of the plugin config builders had drifted, and
  collapsing them is what stopped it.

What is missing is not a different mechanism. It is a persistence layer, a template, and a
validator on top of the mechanism that already works.

## Success criteria

1. Quickstart step 4 drops from 16 `export` lines to `cp .env.example .env` plus filling
   five values. The remaining 25 settings arrive from a committed `mindbridge.toml`.
2. `mindbridge config check --role api` reports **every** missing or invalid setting in one
   pass, names the source each resolved value came from, and prints no secret value.
3. With no `mindbridge.toml` present, every process behaves exactly as it does today. Proven
   by test, not asserted.
4. A credential key placed in `mindbridge.toml` **fails the load** with an error naming the
   key. It is never silently accepted, and never merely warned about.
5. The built wheel imports and runs on Python 3.10, verified against the artifact rather
   than against a passing CI job.

## Approved decisions

**Structure moves to `mindbridge.toml`; credentials stay in the environment.** TOML rather
than YAML or JSON: YAML would re-add a dependency the bootstrap removed and reverse a
deliberate deletion, and a JSON file solves persistence while leaving criterion 4's real
complaint — no comments, no sections — untouched.

**Precedence is environment over file over built-in default. One rule, no exceptions.**
Environment wins so that a container image or a CI job overrides a committed value without a
rebuild, and so that a deployment with no file is bit-for-bit unchanged.

**The file is discovered explicitly, never by searching upward.** `MINDBRIDGE_CONFIG_FILE`
names it if set — and a set-but-missing path is an error, not a silent fallback. Otherwise
`./mindbridge.toml` is used if it exists. Otherwise there is no file. No parent-directory
walk, no XDG lookup: a config file found somewhere the operator did not name is worse than
no config file.

**The file is flattened into the existing variable keyspace, not modelled as a new object.**
This is the decision that keeps the change small — see Architecture.

**Credential keys are rejected inside the file.** The rejected set is derived from the same
seven credential names the Problem section tabulates, mapped through the one key-mapping rule
below, so the guard cannot fall behind a name the contract adds.

### Why not a nested settings object

The obvious shape is a Pydantic model mirroring the TOML tree, threaded through the four
settings classes. It would rewrite all four `from_environment` methods, all three plugin
config builders in `models/defaults.py`, and `plugin_configuration()`, and it would give
every one of the 25 settings two spellings — a variable name and a model field — to keep in
agreement forever. Flattening costs one line per call site and gives them one spelling.

### Why not a `.env` loader in Python

Every deployment target already loads env files natively: `uv run --env-file`, Compose's
`env_file:`, systemd's `EnvironmentFile=`. A Python dotenv parser would reimplement all
three, add a dependency or a quoting bug, and introduce implicit file reads into library
code that four test suites currently rely on being pure. `config check` reports "found
`.env` but it is not loaded" instead, which is a message rather than a mechanism.

### Why not `mindbridge init`

A committed `mindbridge.toml` carrying working defaults plus a five-line `.env.example`
makes `cp` the whole onboarding step. A wizard is worth building only if that is still too
high a bar, which is not yet known.

## Architecture

One new public function in `src/mindbridge/configuration.py`, one new CLI command, two new
files at the repository root. `models/defaults.py`, `plugin_configuration()`, and the three plugin
config builders are untouched.

### The flattening

`configuration_source()` returns a flat `Mapping[str, str]` in which the environment layers
over the flattened file. Every existing reader keeps working unchanged, because every existing
reader already takes a `Mapping[str, str]`:

```python
def configuration_source(
    environ: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
) -> Mapping[str, str]:
    """The environment layered over the flattened config file, if there is one."""
```

Each of the four settings classes changes one line, from

```python
source = os.environ if environ is None else environ
```

to

```python
source = configuration_source(environ)
```

With no file, the result is the environment mapping itself and behaviour is identical. With
an explicit `environ=`, injectability is preserved: tests pass a dict and, unless they also
pass `path=`, no file is read.

Scalar keys resolve by plain lookup order. The four `*_CONFIG_JSON` keys cannot: their values
are objects assembled during flattening by the section rule below, so the mapping is built
once rather than layered lazily.

### One key-mapping rule, and one section rule that composes with it

Every key in the file maps to a variable name mechanically: a key `k` in section `s` is
`MINDBRIDGE_<S>_<K>`, uppercased. That derivation is the whole mapping — a hand-written table
would drift from the loader the first time either one changed.

**Scalar sections** — `[database]`, `[object_storage]`, `[embedding]`, `[aml]` — need nothing
further. Their keys are named values MindBridge owns, read today by
`require_environment_value()` and friends, and the derived name is already the variable those
readers look up.

**Plugin sections** — `[generator]`, `[embedder]`, `[media_embedder]`, `[media_sampling]` —
additionally serialise to their `*_CONFIG_JSON` variable, because that is how
`plugin_configuration()` reads a plugin's config today: as one opaque object whose schema
belongs to the plugin. The section is assembled lowest-precedence-first:

```text
MINDBRIDGE_GENERATOR_CONFIG_JSON = {
    **file_section,                     # [generator] in mindbridge.toml
    **individual_environment_overrides, # every MINDBRIDGE_GENERATOR_<KEY> that is set
}
```

The credential is not a special case here — `api_key` arrives through
`MINDBRIDGE_GENERATOR_API_KEY` by exactly the mechanism that lets
`MINDBRIDGE_GENERATOR_MODEL_ID` override a committed `model_id`. That generality is required,
not incidental: `plugin_configuration()` short-circuits on `*_CONFIG_JSON` and never calls
`openai_generator_config()`, so a section that spliced in only the credential would make
every other individual variable in the environment silently dead the moment a file existed —
contradicting the precedence rule for the variables a container or CI job is most likely to
override.

`MINDBRIDGE_GENERATOR_CONFIG_JSON` set in the environment still wins wholesale over both, as
it does today: it is an object, and a half-overridden opaque object is not a thing a plugin
schema can validate.

`plugin` is reserved inside a plugin section. It flattens to the scalar
`MINDBRIDGE_GENERATOR_PLUGIN` and is excluded from the serialised object, because the
selector is read separately from the config it selects.

The four plugin section names live in one tuple constant that the loader, the credential
guard, and the documentation generator all read.

### The two new root files

`mindbridge.toml` is committed and carries working defaults. `.env.example` is committed and
carries the five values a local deployment must supply; `.env` itself stays ignored.

### `mindbridge config check`

One entry in the `COMMANDS` table in `src/mindbridge/cli.py`, following the existing lazy
import and `guarded` failure contract. `--role` selects one of the six roles the matrix in
`docs/configuration.md` already names: `api`, `mcp`, `worker`, `consolidate`, `lifecycle`,
`edge-sync`. It resolves that role's settings class against `configuration_source()`,
collects every failure rather than raising on the first, and prints one line per setting
naming its source — `environment`, `mindbridge.toml`, or `default`.

Values are never printed. A credential is reported as present or missing, nothing more.

## Data flow

```text
MINDBRIDGE_CONFIG_FILE ─┐
      ./mindbridge.toml ─┴─→ read TOML ─→ reject credential keys ─→ flatten ─┐
                                                                            ├─→ ChainMap ─→ Settings.from_environment()
os.environ (or injected environ=) ──────────────────────────────────────────┘      (environment layer wins)
```

## Error handling

| Condition | Behaviour |
| --- | --- |
| `MINDBRIDGE_CONFIG_FILE` set, path missing | Error naming the path. Never falls back to `./mindbridge.toml` or to no file. |
| File is not valid TOML | Error naming the file and the parse position. |
| File contains a credential key | Error naming the key and the variable it belongs in. Load fails. |
| File has an unknown section or key | Error naming it, consistent with `PluginConfigModel`'s `extra="forbid"`. A typo that is ignored is a value that silently reverts to a default. |
| Both file and environment set a value | Environment wins, silently. `config check` shows which source won. |
| A plugin section in the file and its `*_CONFIG_JSON` in the environment | The environment object wins wholesale; the file section is not merged into it. |
| No file at all | Not an error. Today's behaviour exactly. |

Every message names the setting. No message quotes a value, because the same code path
handles credentials and structure.

## Cost

**One new runtime dependency.** `tomllib` is 3.11+; `requires-python` floors at 3.10 because
JetPack, RDK, and RKNN edge images still ship it. So `tomli>=2,<3; python_version < '3.11'`
joins `[project] dependencies` — the base install, not an extra, since the loader is in
`configuration.py` which every role imports. It is currently declared only in
`[dependency-groups] dev`, whose own comment records that "the tests passed before anything
declared it": a dev-group package imported from `src/` produces a green CI and a wheel that
raises `ModuleNotFoundError` on install. The dependency is pure Python with no transitive
requirements, but criterion 5 exists because declaring it correctly is not the same as
having verified it.

**A breaking change to a documented deployment contract.** The 25 structure variables keep
their names and meanings but become overrides of a file rather than the sole source. Six
roles, the six-column matrix in `docs/configuration.md`, `docs/quickstart.md`,
`docs/deployment.md`, and `docs/api/cli.md` all describe the old contract and all must
change. `docs/configuration.md`'s opening sentence — "There is no config file" — becomes
false and must be rewritten; the half of that sentence about credentials never reaching disk
or `ps` stays true and should be stated more prominently, since it is the property the split
is built to preserve.

**One inconsistency this surfaces.** `src/mindbridge/edge/sync_cli.py:92` reads
`os.environ.get("MINDBRIDGE_API_KEY")` directly rather than through an injected mapping, so
it is the one role that cannot be configured by injection today. `config check --role
edge-sync` needs it routed through `configuration_source()` like the other five.

## Testing

Beyond per-behaviour unit tests, four checks earn their place because each covers something
a passing suite would otherwise hide:

1. **No-file equivalence.** With no `mindbridge.toml`, each of the four settings classes
   built from `configuration_source(environ)` equals the same class built from `environ`
   directly. This is criterion 3, and it is the regression that would otherwise reach a
   deployment.
2. **Credential rejection, with synthetic inputs.** The guard scans file keys, and a scanning
   guard whose corpus contains no violating sample executes zero times while reporting
   success. A committed `mindbridge.toml` will never contain an `api_key`, so the test must
   supply one — one synthetic file per credential name, asserting the error names that key.
3. **Precedence.** File-only, environment-only, and both-set resolve to file, environment,
   and environment respectively — asserted on all three paths, because a scalar key, an
   individual override merged into a plugin section, and a wholesale `*_CONFIG_JSON` are
   three different code paths. The second is the one that regresses silently: without it,
   an existing file makes every individual plugin variable in the environment dead.
4. **Mutation check on each of the above.** Breaking the implementation must turn each test
   red. A test that passes against a deliberately broken loader is testing nothing; this has
   caught self-referential assertions in this repository before.

Criterion 5 is verified against the built artifact: build the wheel, install it into a clean
3.10 environment with no dev group, and import `mindbridge.configuration`. A wheel whose
`ModuleNotFoundError` only appears after installation has passed six CI jobs here before.

## Out of scope

- `mindbridge init` and any interactive wizard.
- Any credential in any file. The guard exists to keep this out of scope permanently.
- YAML, or any second file format.
- Per-tenant configuration, and configuration reload without a restart.
- `AWS_*` and `OTEL_*`, which belong to boto3's and OpenTelemetry's own contracts.
- The 25 old variable names. They keep working as overrides; removing any of them is a
  separate decision with its own deprecation cost.
