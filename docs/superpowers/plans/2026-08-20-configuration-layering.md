# Configuration Layering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MindBridge's 25 non-secret settings out of the environment into a committed
`mindbridge.toml`, keep the 7 credentials in the environment, and add `mindbridge config check`
so an operator sees every missing setting before a process starts.

**Architecture:** One new public function, `configuration_source()`, returns a flat
`Mapping[str, str]` in which the environment layers over the flattened config file. Every
existing reader already takes a `Mapping[str, str]`, so each of the four settings classes
changes one line and `models/defaults.py`, `plugin_configuration()`, and the three plugin
config builders are untouched.

**Tech Stack:** Python 3.10–3.11, `tomllib` (3.11) / `tomli` (3.10), pydantic 2, pytest, argparse.

**Spec:** `docs/superpowers/specs/2026-08-20-configuration-layering-design.md`

## Global Constraints

- Python floor is **3.10** (`requires-python = ">=3.10,<3.12"`). `tomllib` is 3.11+, so the
  `tomli` backport is required on 3.10. mypy is pinned at `python_version = "3.10"`.
- Ruff: line length **100**, McCabe ceiling **10**, rule sets `C90 DTZ E4 E7 E9 F I RUF SIM UP`.
  `ANN401` bans `Any` in signatures — write a `Protocol` shim instead.
- Every function and test needs full type annotations, including `-> None` on tests.
- `ruff format --check .` **includes Markdown files** and formats Python blocks inside them.
- Markdown gate: `MD013` (line length) is off; `MD040` (fenced code needs a language) is on.
  Run the pinned images, not `npx`:
  `docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"`
- The repository is private, so a self-referencing GitHub URL 404s and reddens the
  Documentation job. Reference repository files as inline code or relative links, never as
  `https://github.com/EventHorizon-Lab/...`.
- **No credential may ever be readable from a file.** This is the property the whole split
  exists to preserve; Task 2's guard is not optional.
- Never print a configuration *value* from `config check`. Report presence, not contents.
- The full gate is `uv run --frozen ruff format --check .`, `ruff check .`, `mypy`,
  `pytest -W error`, plus the Documentation job's markdownlint and lychee.

## The seven credentials

Referenced by several tasks. These stay in the environment and are rejected inside the file:

```text
MINDBRIDGE_API_KEY              MINDBRIDGE_GENERATOR_API_KEY
MINDBRIDGE_AML_API_KEY          MINDBRIDGE_TASK_BROKER_URL
MINDBRIDGE_DATABASE_URL         MINDBRIDGE_TENANT_API_KEYS_JSON
MINDBRIDGE_EMBEDDER_API_KEY
```

---

### Task 1: The `tomli` dependency and file discovery

**Files:**

- Modify: `pyproject.toml:12-15` (base `dependencies`)
- Modify: `src/mindbridge/configuration.py` (add imports and discovery)
- Test: `tests/unit/test_configuration.py`

**Interfaces:**

- Produces: `variable_name(key: str, section: str | None = None) -> str`;
  `CONFIG_FILE_VARIABLE: str`; `DEFAULT_CONFIG_FILE: str`;
  `_configuration_document(environ: Mapping[str, str], path: Path | None) -> dict[str, object] | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_configuration.py`:

```python
def test_file_keys_derive_their_variable_names() -> None:
    assert variable_name("max_pool_size", "database") == "MINDBRIDGE_DATABASE_MAX_POOL_SIZE"
    assert variable_name("bucket", "object_storage") == "MINDBRIDGE_OBJECT_STORAGE_BUCKET"
    assert variable_name("model_id", "media_embedder") == "MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID"
    assert (
        variable_name("minimum_embedding_similarity") == "MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY"
    )


def test_configuration_file_is_found_only_where_it_is_named(tmp_path: Path) -> None:
    named = tmp_path / "named.toml"
    named.write_text("[database]\nmax_pool_size = 8\n", encoding="utf-8")

    assert _configuration_document({"MINDBRIDGE_CONFIG_FILE": str(named)}, None) == {
        "database": {"max_pool_size": 8}
    }
    assert _configuration_document({}, named) == {"database": {"max_pool_size": 8}}
    # No file named and none in the working directory is not an error.
    assert _configuration_document({}, tmp_path / "absent.toml") is None
    with pytest.raises(ValueError, match="which is not a file"):
        _configuration_document({"MINDBRIDGE_CONFIG_FILE": str(tmp_path / "absent.toml")}, None)


def test_a_malformed_configuration_file_names_itself(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[database\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain valid TOML"):
        _configuration_document({}, broken)
```

Add to the test module's imports:

```python
from pathlib import Path

from mindbridge.configuration import (
    CONFIG_FILE_VARIABLE,
    DEFAULT_CONFIG_FILE,
    _configuration_document,
    variable_name,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -x -q`
Expected: FAIL with `ImportError: cannot import name 'variable_name'`

- [ ] **Step 3: Declare the dependency**

In `pyproject.toml`, change the base `dependencies` block to:

```toml
dependencies = [
    "httpx>=0.28,<1",
    "pydantic>=2.10,<3",
    # mindbridge.configuration reads mindbridge.toml, and every role imports it. tomllib is
    # 3.11+; the floor is 3.10 because JetPack, RDK, and RKNN edge images still ship it. This
    # is a runtime dependency, not the dev-group entry tests/test_package.py uses.
    "tomli>=2,<3; python_version < '3.11'",
]
```

Then relock: `uv lock`

Inspect the lock diff before continuing — `git diff uv.lock` should touch only `tomli`'s
markers. `uv.lock` pins Aliyun mirror URLs from the local `uv.toml`; a diff that rewrites
hundreds of URLs means the wrong index was used, so stop and report rather than committing it.

- [ ] **Step 4: Write the implementation**

In `src/mindbridge/configuration.py`, add to the imports at the top:

```python
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 floor, not on the mypy pin
    import tomli as tomllib
```

Then add, after the existing `optional_environment_value`:

```python
CONFIG_FILE_VARIABLE = "MINDBRIDGE_CONFIG_FILE"
"""Names the configuration file explicitly. A set-but-missing path is an error."""

DEFAULT_CONFIG_FILE = "mindbridge.toml"
"""Read from the working directory when nothing names a file.

There is deliberately no parent-directory walk and no XDG lookup: a configuration file found
somewhere the operator did not name is worse than no configuration file.
"""


def variable_name(key: str, section: str | None = None) -> str:
    """Derive the one variable a file key configures.

    The mapping is this function rather than a table, so nothing can fall behind the loader:
    key `k` under section `s` configures `MINDBRIDGE_<S>_<K>`, and a key at the top level of
    the document configures `MINDBRIDGE_<K>`.
    """
    if section is None:
        return f"MINDBRIDGE_{key.upper()}"
    return f"MINDBRIDGE_{section.upper()}_{key.upper()}"


def _configuration_document(
    environ: Mapping[str, str],
    path: Path | None,
) -> dict[str, object] | None:
    """Locate and parse the configuration file, or report that there is none."""
    located = path if path is not None else _located_path(environ)
    if located is None or not located.is_file():
        if path is None and optional_environment_value(environ, CONFIG_FILE_VARIABLE):
            raise ValueError(f"{CONFIG_FILE_VARIABLE} names {located}, which is not a file")
        return None
    try:
        text = located.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{located} could not be read") from error
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{located} must contain valid TOML: {error}") from error


def _located_path(environ: Mapping[str, str]) -> Path | None:
    """Resolve which file to read without searching anywhere the operator did not name."""
    named = optional_environment_value(environ, CONFIG_FILE_VARIABLE)
    if named is not None:
        return Path(named)
    default = Path(DEFAULT_CONFIG_FILE)
    return default if default.is_file() else None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q`
Expected: PASS

- [ ] **Step 6: Mutation-check the discovery rule**

Temporarily change `_located_path` to fall back to the default when
`MINDBRIDGE_CONFIG_FILE` names a missing file:

```python
    if named is not None:
        candidate = Path(named)
        if candidate.is_file():
            return candidate
```

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q`
Expected: FAIL on `test_configuration_file_is_found_only_where_it_is_named`. If it passes, the
test is not asserting the rule — fix the test. Revert the mutation either way.

- [ ] **Step 7: Verify the built artifact on the 3.10 floor**

CI green does not prove an installable wheel: `[project.scripts]` and the build `exclude` list
do not interact, and this repository has shipped a wheel that raised `ModuleNotFoundError`
with six green jobs. Check the artifact:

```bash
uv build --wheel
uv run --python 3.10 --with dist/mindbridge-0.1.0-py3-none-any.whl --no-project \
  python -c "import mindbridge.configuration as c; print(c.variable_name('bucket', 'object_storage'))"
```

Expected: prints `MINDBRIDGE_OBJECT_STORAGE_BUCKET`. `--no-project` keeps the source tree and
the dev group off `sys.path`, so a missing runtime declaration fails here rather than passing
by accident.

- [ ] **Step 8: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add pyproject.toml uv.lock src/mindbridge/configuration.py tests/unit/test_configuration.py
git commit -m "Find the configuration file only where an operator named it"
```

---

### Task 2: Flatten scalar sections, and reject credentials in the file

**Files:**

- Modify: `src/mindbridge/configuration.py`
- Test: `tests/unit/test_configuration.py`

**Interfaces:**

- Consumes: `variable_name()`, `_configuration_document()` from Task 1
- Produces: `CREDENTIAL_VARIABLES: frozenset[str]`; `PLUGIN_SECTIONS: tuple[str, ...]`;
  `_flattened_scalars(document: Mapping[str, object]) -> dict[str, str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_scalar_sections_and_top_level_keys_flatten_to_their_variables() -> None:
    document: dict[str, object] = {
        "minimum_embedding_similarity": 0.25,
        "database": {"max_pool_size": 32},
        "object_storage": {"bucket": "mindbridge-media", "endpoint_url": "http://minio:9000"},
        "embedding": {"dimension": 1024, "space_id": "jina-v5"},
    }

    assert _flattened_scalars(document) == {
        "MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY": "0.25",
        "MINDBRIDGE_DATABASE_MAX_POOL_SIZE": "32",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "mindbridge-media",
        "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL": "http://minio:9000",
        "MINDBRIDGE_EMBEDDING_DIMENSION": "1024",
        "MINDBRIDGE_EMBEDDING_SPACE_ID": "jina-v5",
    }


def test_a_plugin_section_contributes_no_scalar_of_its_own() -> None:
    # A plugin section's body is one config object; only its `plugin` selector is a scalar.
    flattened = _flattened_scalars({"generator": {"plugin": "openai", "model_id": "qwen3.8-max"}})

    assert flattened == {"MINDBRIDGE_GENERATOR_PLUGIN": "openai"}


def test_every_credential_is_refused_inside_the_file() -> None:
    # One synthetic document per credential: a committed file will never carry one, so a guard
    # tested only against real fixtures would run zero times and still report success.
    refused = {
        "MINDBRIDGE_DATABASE_URL": {"database": {"url": "postgresql://u:p@h/d"}},
        "MINDBRIDGE_TASK_BROKER_URL": {"task": {"broker_url": "redis://h:6379/0"}},
        "MINDBRIDGE_GENERATOR_API_KEY": {"generator": {"api_key": "sk-secret"}},
        "MINDBRIDGE_EMBEDDER_API_KEY": {"embedder": {"api_key": "sk-secret"}},
        "MINDBRIDGE_TENANT_API_KEYS_JSON": {"tenant": {"api_keys_json": "{}"}},
        "MINDBRIDGE_API_KEY": {"api_key": "sk-secret"},
        "MINDBRIDGE_AML_API_KEY": {"aml": {"api_key": "sk-secret"}},
    }
    assert set(refused) == set(CREDENTIAL_VARIABLES), "a credential gained no rejection test"

    for variable, document in refused.items():
        with pytest.raises(ValueError, match=f"{variable} is a credential"):
            _flattened_scalars(document)


def test_the_file_rejects_shapes_no_variable_could_carry() -> None:
    with pytest.raises(ValueError, match="must not nest"):
        _flattened_scalars({"database": {"pool": {"max_size": 4}}})
    with pytest.raises(ValueError, match="must be text, a number, or a boolean"):
        _flattened_scalars({"object_storage": {"bucket": ["a", "b"]}})
    with pytest.raises(ValueError, match="not a known configuration section"):
        _flattened_scalars({"databse": {"max_pool_size": 8}})
    with pytest.raises(ValueError, match="not a known key"):
        _flattened_scalars({"database": {"max_poool_size": 8}})


def test_the_known_keys_cannot_fall_behind_what_the_code_reads() -> None:
    # KNOWN_SCALAR_KEYS is the one table in the loader, so it needs a guard outside itself.
    # Every MINDBRIDGE_<SECTION>_* name the source reads must be derivable from it, or a
    # variable exists that no file key can reach.
    read: set[str] = set()
    for module in Path("src").rglob("*.py"):
        read |= set(re.findall(r"MINDBRIDGE_[A-Z0-9_]+", module.read_text(encoding="utf-8")))

    for section, keys in KNOWN_SCALAR_KEYS.items():
        prefix = f"MINDBRIDGE_{section.upper()}_"
        # Bare `MINDBRIDGE_<SECTION>_` hits are glob notation inside comments, not variables.
        reachable = {name for name in read if name.startswith(prefix) and name != prefix}
        derived = {variable_name(key, section) for key in keys}
        assert reachable - CREDENTIAL_VARIABLES == derived, f"[{section}] drifted"
```

Extend the test imports with `re`, `CREDENTIAL_VARIABLES`, `KNOWN_SCALAR_KEYS`,
`PLUGIN_SECTIONS`, `TOP_LEVEL_KEYS`, and `_flattened_scalars`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -x -q`
Expected: FAIL with `ImportError: cannot import name '_flattened_scalars'`

- [ ] **Step 3: Write the implementation**

Add to `src/mindbridge/configuration.py`:

```python
PLUGIN_SECTIONS: tuple[str, ...] = ("generator", "embedder", "media_embedder", "media_sampling")
"""Sections whose body is one plugin's config object rather than a set of named scalars.

`plugin_configuration()` reads a plugin's config as one opaque object whose schema belongs to
the plugin, so these sections serialise to their `*_CONFIG_JSON` variable instead of
contributing one variable per key. `plugin` is the exception: the selector is read separately
from the config it selects.
"""

KNOWN_SCALAR_KEYS: Mapping[str, tuple[str, ...]] = {
    "database": ("max_pool_size",),
    "object_storage": ("bucket", "endpoint_url", "public_endpoint_url"),
    "embedding": ("dimension", "space_id", "space_revision"),
    "aml": ("tenant_prefix",),
}
"""Sections holding named values MindBridge owns, one variable per key.

Spelled out rather than derived because nothing in the code enumerates them, and an unlisted
key has to be an error: a typo that flattens to a variable no reader looks up is a value that
silently reverts to its default, which is the failure `extra="forbid"` already prevents inside
a plugin config. `test_the_known_keys_cannot_fall_behind_what_the_code_reads` is the guard that
keeps this table honest.
"""

TOP_LEVEL_KEYS: tuple[str, ...] = ("minimum_embedding_similarity",)
"""Keys configuring one deployment-wide value that belongs to no section."""

PLUGIN_SELECTOR_KEY = "plugin"
"""Reserved inside a plugin section: it names the plugin rather than configuring it."""

CREDENTIAL_VARIABLES: frozenset[str] = frozenset(
    {
        "MINDBRIDGE_API_KEY",
        "MINDBRIDGE_AML_API_KEY",
        "MINDBRIDGE_DATABASE_URL",
        "MINDBRIDGE_EMBEDDER_API_KEY",
        "MINDBRIDGE_GENERATOR_API_KEY",
        "MINDBRIDGE_TASK_BROKER_URL",
        "MINDBRIDGE_TENANT_API_KEYS_JSON",
    }
)
"""The variables that may never be read from a file.

Keeping credentials out of every file is the property this split exists to preserve, so a
credential key is an error rather than a warning: a warning that is ignored puts a secret on
disk just as effectively as no check at all.
"""


def _flattened_scalars(document: Mapping[str, object]) -> dict[str, str]:
    """Flatten every file key that configures one named variable."""
    flattened: dict[str, str] = {}
    for name, value in document.items():
        if not isinstance(value, dict):
            if name not in TOP_LEVEL_KEYS:
                raise ValueError(f"{name} is not a known top-level configuration key")
            flattened.update(_scalar(variable_name(name), value))
            continue
        if name in PLUGIN_SECTIONS:
            selector = value.get(PLUGIN_SELECTOR_KEY)
            if selector is not None:
                flattened.update(_scalar(variable_name(PLUGIN_SELECTOR_KEY, name), selector))
            _reject_credentials_in(name, value)
            continue
        known = KNOWN_SCALAR_KEYS.get(name)
        if known is None:
            raise ValueError(f"{name} is not a known configuration section")
        for key, entry in value.items():
            if isinstance(entry, dict):
                raise ValueError(f"{name}.{key} must not nest another table")
            if key not in known:
                raise ValueError(f"{key} is not a known key of [{name}]")
            flattened.update(_scalar(variable_name(key, name), entry))
    return flattened


def _reject_credentials_in(section: str, body: Mapping[str, object]) -> None:
    """Refuse a plugin section that carries its own credential."""
    for key in body:
        _reject_credential(variable_name(key, section))


def _scalar(variable: str, value: object) -> dict[str, str]:
    """Render one file scalar as the string an environment reader would have received."""
    _reject_credential(variable)
    if isinstance(value, bool):
        return {variable: "true" if value else "false"}
    if isinstance(value, str | int | float):
        return {variable: str(value)}
    raise ValueError(f"{variable} must be text, a number, or a boolean")


def _reject_credential(variable: str) -> None:
    """Keep every credential out of every file, in one place both flatteners call."""
    if variable in CREDENTIAL_VARIABLES:
        raise ValueError(
            f"{variable} is a credential and must not appear in the configuration file. "
            f"Set it in the environment instead."
        )
```

Note the `"api_key"` top-level case: `variable_name("api_key")` is `MINDBRIDGE_API_KEY`, which
`_scalar` rejects through the same guard. No separate branch is needed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q`
Expected: PASS

- [ ] **Step 5: Prove the credential guard actually executes**

A scanning guard whose corpus contains no violating sample runs zero times and still reports
success. Confirm the loop body is reached:

```bash
uv run --frozen python -c "
from mindbridge.configuration import _reject_credential
import mindbridge.configuration as c
seen = []
original = c._reject_credential
c._reject_credential = lambda v: (seen.append(v), original(v))[1]
c._flattened_scalars({'generator': {'plugin': 'openai', 'model_id': 'm', 'endpoint': 'e'}})
print(f'guard ran {len(seen)} times: {seen}')
"
```

Expected: at least 4 calls, including `MINDBRIDGE_GENERATOR_MODEL_ID` and
`MINDBRIDGE_GENERATOR_ENDPOINT`. Zero or one call means plugin-section keys are not being
scanned and a credential could slip through.

- [ ] **Step 6: Mutation-check the guard**

Temporarily make `_reject_credential` a no-op (`return`). Run
`uv run --frozen pytest tests/unit/test_configuration.py -q` and expect
`test_every_credential_is_refused_inside_the_file` to FAIL. Revert.

- [ ] **Step 7: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add src/mindbridge/configuration.py tests/unit/test_configuration.py
git commit -m "Flatten the file's named settings and refuse its credentials"
```

---

### Task 3: Assemble each plugin section into its config object

**Files:**

- Modify: `src/mindbridge/configuration.py`
- Test: `tests/unit/test_configuration.py`

**Interfaces:**

- Consumes: `PLUGIN_SECTIONS`, `PLUGIN_SELECTOR_KEY`, `variable_name()`, `_reject_credential()`
- Produces: `_flattened_plugins(document: Mapping[str, object], environ: Mapping[str, str]) -> dict[str, str]`

Three behaviours, each required by a failure verified against the running code:

1. A section serialises to its `*_CONFIG_JSON`, since that is the only channel
   `plugin_configuration()` reads a plugin config through.
2. An individual `MINDBRIDGE_<SECTION>_<KEY>` in the environment overrides the file's key, in
   **the type the file declared**. `_GeneratorConfig.request_timeout_seconds` is
   `StrictFloat | StrictInt` and `_MediaSamplingConfig.generation_proxy` is `StrictBool`; both
   reject the strings the environment carries. `bool` is checked before `int`, because `bool`
   is a subclass of `int` and `bool("false")` is `True`.
3. `[embedding]` folds into `[embedder]` and `[media_embedder]`. `_embedding_space_config()`
   runs inside `openai_embedder_config()`, which a produced `*_CONFIG_JSON` skips — and
   `space_id` and `space_revision` are required with no default, so a file naming `[embedder]`
   without them would fail validation at startup.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_plugin_section_becomes_its_config_object() -> None:
    flattened = _flattened_plugins(
        {"generator": {"plugin": "openai", "endpoint": "https://g/v1", "model_id": "qwen3.8-max"}},
        {"MINDBRIDGE_GENERATOR_API_KEY": "sk-live"},
    )

    assert json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"]) == {
        "api_key": "sk-live",
        "endpoint": "https://g/v1",
        "model_id": "qwen3.8-max",
    }
    assert "MINDBRIDGE_GENERATOR_PLUGIN" not in flattened


def test_an_environment_override_is_read_in_the_type_the_file_declared() -> None:
    flattened = _flattened_plugins(
        {"generator": {"request_timeout_seconds": 1800, "max_retries": 2, "model_id": "a"}},
        {
            "MINDBRIDGE_GENERATOR_REQUEST_TIMEOUT_SECONDS": "900",
            "MINDBRIDGE_GENERATOR_MAX_RETRIES": "5",
            "MINDBRIDGE_GENERATOR_MODEL_ID": "b",
        },
    )
    generator = json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"])

    # Strict pydantic fields reject "900" and "5" as surely as they reject a missing key.
    assert generator == {"request_timeout_seconds": 900, "max_retries": 5, "model_id": "b"}
    assert isinstance(generator["request_timeout_seconds"], int)


def test_a_boolean_override_is_not_read_as_a_non_empty_string() -> None:
    flattened = _flattened_plugins(
        {"media_sampling": {"generation_proxy": True}},
        {"MINDBRIDGE_MEDIA_SAMPLING_GENERATION_PROXY": "false"},
    )

    # bool("false") is True; bool must be matched before int, which it subclasses.
    assert json.loads(flattened["MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON"]) == {
        "generation_proxy": False
    }


def test_the_embedding_space_reaches_every_encoder_from_one_place() -> None:
    document: dict[str, object] = {
        "embedding": {"dimension": 1024, "space_id": "jina-v5", "space_revision": "omni@abc"},
        "embedder": {"endpoint": "https://e/v1"},
        "media_embedder": {"device": "cuda:0"},
        "generator": {"endpoint": "https://g/v1"},
    }
    flattened = _flattened_plugins(document, {})

    for section in ("embedder", "media_embedder"):
        encoded = json.loads(flattened[f"MINDBRIDGE_{section.upper()}_CONFIG_JSON"])
        assert encoded["space_id"] == "jina-v5"
        assert encoded["space_revision"] == "omni@abc"
        assert encoded["dimension"] == 1024
    # The generator shares no embedding space.
    assert "space_id" not in json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"])


def test_an_override_of_a_key_the_file_omits_arrives_as_text() -> None:
    flattened = _flattened_plugins(
        {"embedder": {"endpoint": "https://e/v1"}},
        {"MINDBRIDGE_EMBEDDER_MODEL_REVISION": "abc123"},
    )

    assert json.loads(flattened["MINDBRIDGE_EMBEDDER_CONFIG_JSON"])["model_revision"] == "abc123"
```

Add `import json` and `_flattened_plugins` to the test module's imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -x -q`
Expected: FAIL with `ImportError: cannot import name '_flattened_plugins'`

- [ ] **Step 3: Write the implementation**

Add to `src/mindbridge/configuration.py`:

```python
ENCODER_SECTIONS: tuple[str, ...] = ("embedder", "media_embedder")
"""Plugin sections that must share the one deployment-wide embedding space."""

EMBEDDING_SECTION = "embedding"
"""The section whose keys every encoder section inherits."""


def _flattened_plugins(
    document: Mapping[str, object],
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Serialise each plugin section into the `*_CONFIG_JSON` its factory reads."""
    space = document.get(EMBEDDING_SECTION)
    shared = space if isinstance(space, dict) else {}
    flattened: dict[str, str] = {}
    for section in PLUGIN_SECTIONS:
        body = document.get(section)
        if not isinstance(body, dict):
            continue
        assembled: dict[str, object] = {}
        if section in ENCODER_SECTIONS:
            assembled.update(shared)
        assembled.update({key: value for key, value in body.items() if key != PLUGIN_SELECTOR_KEY})
        assembled.update(_overrides(section, assembled, environ))
        flattened[variable_name("config_json", section)] = json.dumps(
            assembled, sort_keys=True, allow_nan=False
        )
    return flattened


def _overrides(
    section: str,
    assembled: Mapping[str, object],
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Read every individual variable that overrides one key of this plugin section.

    The environment wins per key rather than per section. Splicing in only the credential would
    make every other individual variable silently dead the moment a file existed, because
    `plugin_configuration()` short-circuits on `*_CONFIG_JSON` and never calls the builder that
    reads them.
    """
    prefix = f"MINDBRIDGE_{section.upper()}_"
    reserved = {
        variable_name("config_json", section),
        variable_name(PLUGIN_SELECTOR_KEY, section),
    }
    overrides: dict[str, object] = {}
    for name, text in environ.items():
        if not name.startswith(prefix) or name in reserved:
            continue
        key = name[len(prefix) :].lower()
        overrides[key] = _as_declared(assembled.get(key), text)
    return overrides


def _as_declared(declared: object, text: str) -> object:
    """Read an environment override in the type the file declared for that key.

    A validated plugin config rejects `"1800"` where it wants a number and `"false"` where it
    wants a boolean, so an override cannot arrive as text wherever the file said otherwise. A
    key the file omits has no declared type and arrives as text, which is what every individual
    plugin variable in the contract today already is.
    """
    if isinstance(declared, bool):
        if text.lower() not in {"true", "false"}:
            raise ValueError(f"{text!r} must be 'true' or 'false'")
        return text.lower() == "true"
    if isinstance(declared, int):
        return int(text)
    if isinstance(declared, float):
        return float(text)
    return text
```

`bool` precedes `int` in `_as_declared` deliberately: `isinstance(True, int)` is `True`, so the
reverse order would turn `"false"` into `int("false")` and raise, or worse, turn a `"1"` into a
truthy int that pydantic's `StrictBool` then rejects.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q`
Expected: PASS

- [ ] **Step 5: Verify against the real plugin schemas, not just the flattener**

The point of the typing rules is that the produced object validates. Check it end to end:

```bash
uv run --frozen python -c "
import json
from mindbridge.configuration import _flattened_plugins
from mindbridge.models.openai import _GeneratorConfig
flat = _flattened_plugins(
    {'generator': {'endpoint': 'https://g/v1', 'model_revision': 'r', 'request_timeout_seconds': 1800}},
    {'MINDBRIDGE_GENERATOR_API_KEY': 'sk', 'MINDBRIDGE_GENERATOR_REQUEST_TIMEOUT_SECONDS': '900'},
)
config = _GeneratorConfig.model_validate(json.loads(flat['MINDBRIDGE_GENERATOR_CONFIG_JSON']))
print(f'validated, timeout={config.request_timeout_seconds}')
"
```

Expected: `validated, timeout=900.0`. A `ValidationError` here means the coercion is wrong even
though the flattener's own tests pass.

- [ ] **Step 6: Mutation-check the bool ordering and the space fold**

Two mutations, run separately:

1. Swap the `bool` and `int` branches in `_as_declared`. Expect
   `test_a_boolean_override_is_not_read_as_a_non_empty_string` to FAIL.
2. Delete the `if section in ENCODER_SECTIONS:` block. Expect
   `test_the_embedding_space_reaches_every_encoder_from_one_place` to FAIL.

Revert both.

- [ ] **Step 7: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add src/mindbridge/configuration.py tests/unit/test_configuration.py
git commit -m "Assemble each plugin section into the object its factory reads"
```

---

### Task 4: Layer the file under the environment at every call site

**Files:**

- Modify: `src/mindbridge/configuration.py` (add `configuration_source`)
- Modify: `src/mindbridge/api/runtime.py:112`
- Modify: `src/mindbridge/worker.py:155`
- Modify: `src/mindbridge/consolidation_cli.py:114`
- Modify: `src/mindbridge/infrastructure/s3.py:123`
- Modify: `src/mindbridge/infrastructure/postgres.py:87`
- Modify: `src/mindbridge/lifecycle_cli.py:134`
- Modify: `src/mindbridge/edge/sync_cli.py:92`
- Test: `tests/unit/test_configuration.py`

**Do not skip `postgres.py:87`.** `resolve_database_max_pool_size()` is called with no
argument from `worker.py:368`, `consolidation_cli.py:201`, and `lifecycle_cli.py:184`, so those
three read `os.environ` directly while `api/runtime.py:144` passes its `source`. Leaving it
alone means `[database] max_pool_size` in the file is honoured by the API and ignored by the
other three — which is precisely the bug the comment above that function says was already found
and fixed once: "previously only the API read it, so lowering it left the other three at the
default and the deployment still exceeded its server."

**Interfaces:**

- Consumes: `_configuration_document()`, `_flattened_scalars()`, `_flattened_plugins()`
- Produces: `configuration_source(environ: Mapping[str, str] | None = None, *, path: Path | None = None) -> Mapping[str, str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_with_no_file_the_source_is_the_environment_unchanged(tmp_path: Path) -> None:
    environ = {"MINDBRIDGE_DATABASE_URL": "postgresql://u:p@h/d", "OTHER": "kept"}

    resolved = configuration_source(environ, path=tmp_path / "absent.toml")

    assert dict(resolved) == environ


def test_the_environment_wins_over_the_file_key_by_key(tmp_path: Path) -> None:
    config = tmp_path / "mindbridge.toml"
    config.write_text(
        "[database]\nmax_pool_size = 32\n[object_storage]\nbucket = 'from-file'\n",
        encoding="utf-8",
    )

    resolved = configuration_source({"MINDBRIDGE_OBJECT_STORAGE_BUCKET": "from-env"}, path=config)

    assert resolved["MINDBRIDGE_OBJECT_STORAGE_BUCKET"] == "from-env"
    assert resolved["MINDBRIDGE_DATABASE_MAX_POOL_SIZE"] == "32"


def test_an_environment_config_object_wins_over_the_whole_file_section(tmp_path: Path) -> None:
    config = tmp_path / "mindbridge.toml"
    config.write_text("[generator]\nendpoint = 'https://from-file/v1'\n", encoding="utf-8")

    resolved = configuration_source(
        {"MINDBRIDGE_GENERATOR_CONFIG_JSON": '{"endpoint":"https://from-env/v1"}'},
        path=config,
    )

    # A half-overridden opaque object is not something a plugin schema can validate.
    assert json.loads(resolved["MINDBRIDGE_GENERATOR_CONFIG_JSON"]) == {
        "endpoint": "https://from-env/v1"
    }


def test_the_pool_ceiling_reaches_the_processes_that_pass_no_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Worker, consolidate, and lifecycle call this with no argument; only the API passes a
    # source. A file honoured by one of the four and ignored by three is the exact failure the
    # comment on this function records as already found once.
    config = tmp_path / "mindbridge.toml"
    config.write_text("[database]\nmax_pool_size = 7\n", encoding="utf-8")
    monkeypatch.setenv("MINDBRIDGE_CONFIG_FILE", str(config))

    assert resolve_database_max_pool_size() == 7
    monkeypatch.setenv("MINDBRIDGE_DATABASE_MAX_POOL_SIZE", "9")
    assert resolve_database_max_pool_size() == 9


def test_every_settings_class_reads_the_same_layered_source(tmp_path: Path) -> None:
    from mindbridge.api.runtime import Settings

    config = tmp_path / "mindbridge.toml"
    config.write_text(
        "[object_storage]\nbucket = 'mindbridge-media'\n"
        "[embedding]\ndimension = 1024\nspace_id = 's'\nspace_revision = 'r'\n"
        "[generator]\nendpoint = 'https://g/v1'\nmodel_revision = 'gr'\n"
        "[embedder]\nendpoint = 'https://e/v1'\nmodel_id = 'm'\nmodel_revision = 'er'\n",
        encoding="utf-8",
    )
    environ = {
        "MINDBRIDGE_CONFIG_FILE": str(config),
        "MINDBRIDGE_DATABASE_URL": "postgresql://u:p@h/d",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://h:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "sk-generator",
        "MINDBRIDGE_EMBEDDER_API_KEY": "sk-embedder",
        "MINDBRIDGE_TENANT_API_KEYS_JSON": '{"tenant_01":["' + "a" * 48 + '"]}',
    }

    settings = Settings.from_environment(environ)

    assert settings.object_storage.bucket == "mindbridge-media"
    assert settings.generator_config["endpoint"] == "https://g/v1"
    assert settings.embedder_config["api_key"] == "sk-embedder"
    assert settings.embedding_dimension == 1024
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -x -q`
Expected: FAIL with `ImportError: cannot import name 'configuration_source'`

- [ ] **Step 3: Write `configuration_source`**

Add to `src/mindbridge/configuration.py`:

```python
def configuration_source(
    environ: Mapping[str, str] | None = None,
    *,
    path: Path | None = None,
) -> Mapping[str, str]:
    """The environment layered over the flattened configuration file, if there is one.

    Every reader in the codebase already takes a `Mapping[str, str]`, so layering here is what
    lets one file configure all of them without any reader learning a second shape. With no
    file the environment mapping is returned unchanged, which is why a deployment that has none
    behaves exactly as it did before this existed.

    Scalar keys resolve by lookup order. The four `*_CONFIG_JSON` keys cannot: their values are
    objects assembled from the file and the environment together, so the mapping is built once
    rather than layered lazily.
    """
    source = os.environ if environ is None else environ
    document = _configuration_document(source, path)
    if document is None:
        return source
    flattened = _flattened_scalars(document)
    flattened.update(_flattened_plugins(document, source))
    flattened.update(source)
    return flattened
```

`configuration.py` needs `import os` added to its imports.

- [ ] **Step 4: Change the five call sites**

`src/mindbridge/api/runtime.py:112`, `src/mindbridge/worker.py:155`, and
`src/mindbridge/consolidation_cli.py:114` each read:

```python
        source = os.environ if environ is None else environ
```

Replace each with:

```python
        source = configuration_source(environ)
```

`src/mindbridge/infrastructure/s3.py:123` reads:

```python
        return cls(object_storage_from_environment(os.environ if environ is None else environ))
```

Replace with:

```python
        return cls(object_storage_from_environment(configuration_source(environ)))
```

`src/mindbridge/edge/sync_cli.py:92` reads `os.environ` directly, which is why it is the one
role that cannot be configured by injection. It reads:

```text
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
```

Replace with:

```text
        api_key=configuration_source().get("MINDBRIDGE_API_KEY"),
```

`src/mindbridge/infrastructure/postgres.py:87`, inside
`resolve_database_max_pool_size`, reads:

```python
    source = os.environ if environ is None else environ
```

Replace with:

```python
    source = configuration_source(environ)
```

`src/mindbridge/lifecycle_cli.py:134` reads `os.environ` directly:

```text
            require_environment_value(os.environ, "MINDBRIDGE_DATABASE_URL"),
```

Replace with:

```text
            require_environment_value(configuration_source(), "MINDBRIDGE_DATABASE_URL"),
```

In each of the seven modules, add `configuration_source` to the existing
`from mindbridge.configuration import (...)` block, keeping the list alphabetical for ruff's
`I` rules. In `s3.py` and `sync_cli.py`, check whether `import os` is now unused — ruff `F401`
will say so.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q && uv run --frozen pytest tests/unit -q`

Import `resolve_database_max_pool_size` from `mindbridge.infrastructure.postgres` in the test
module.
Expected: PASS. The existing suites are the real check here: they build settings from injected
dicts, and every one of them must still pass untouched.

- [ ] **Step 6: Mutation-check the precedence direction**

Swap the last two lines of `configuration_source` so the file updates over the environment:

```python
    flattened.update(source)
    flattened.update(_flattened_plugins(document, source))
```

Run: `uv run --frozen pytest tests/unit/test_configuration.py -q`
Expected: FAIL on `test_the_environment_wins_over_the_file_key_by_key`. Revert.

- [ ] **Step 7: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add src/mindbridge/configuration.py src/mindbridge/api/runtime.py src/mindbridge/worker.py \
  src/mindbridge/consolidation_cli.py src/mindbridge/lifecycle_cli.py \
  src/mindbridge/infrastructure/s3.py src/mindbridge/infrastructure/postgres.py \
  src/mindbridge/edge/sync_cli.py tests/unit/test_configuration.py
git commit -m "Read one layered source in every process that reads configuration"
```

---

### Task 5: `mindbridge config check`

**Files:**

- Create: `src/mindbridge/config_cli.py`
- Modify: `src/mindbridge/configuration.py` (typed missing-value error)
- Modify: `src/mindbridge/cli.py:96-119` (`COMMANDS` table)
- Test: `tests/unit/test_config_cli.py`

**Interfaces:**

- Consumes: `configuration_source()`, `CREDENTIAL_VARIABLES`
- Produces: `MissingConfigurationError`; `main(argv: Sequence[str], *, prog: str) -> int`

The command must report **every** missing setting in one pass. The settings classes raise on
the first one, so `config check` learns a role's requirements by asking the class rather than
from a table that would drift: it builds the settings, catches the typed error naming the
missing variable, substitutes a plausible value, and retries until the class stops complaining.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config_cli.py`:

```python
"""Tests for the pre-flight configuration report."""

from pathlib import Path

import pytest

from mindbridge.config_cli import ROLES, main


def test_every_documented_role_is_addressable() -> None:
    assert set(ROLES) == {"api", "mcp", "worker", "consolidate", "lifecycle", "edge-sync"}


def test_a_complete_configuration_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "mindbridge.toml"
    config.write_text(
        "[object_storage]\nbucket = 'mindbridge-media'\n"
        "[embedding]\ndimension = 1024\nspace_id = 's'\nspace_revision = 'r'\n"
        "[generator]\nendpoint = 'https://g/v1'\nmodel_revision = 'gr'\n"
        "[embedder]\nendpoint = 'https://e/v1'\nmodel_id = 'm'\nmodel_revision = 'er'\n",
        encoding="utf-8",
    )
    for name, value in {
        "MINDBRIDGE_CONFIG_FILE": str(config),
        "MINDBRIDGE_DATABASE_URL": "postgresql://u:p@h/d",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://h:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "sk-generator",
        "MINDBRIDGE_EMBEDDER_API_KEY": "sk-embedder",
        "MINDBRIDGE_TENANT_API_KEYS_JSON": '{"tenant_01":["' + "a" * 48 + '"]}',
    }.items():
        monkeypatch.setenv(name, value)

    assert main(["--role", "api"], prog="mindbridge config check") == 0
    assert "ready" in capsys.readouterr().out


def test_all_missing_settings_are_reported_in_one_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)  # away from the repository's own mindbridge.toml
    for name in list(os.environ):
        if name.startswith("MINDBRIDGE_"):
            monkeypatch.delenv(name, raising=False)

    assert main(["--role", "api"], prog="mindbridge config check") == 1

    reported = capsys.readouterr().out
    # One restart per missing value is the cost this command exists to remove.
    for expected in (
        "MINDBRIDGE_DATABASE_URL",
        "MINDBRIDGE_TASK_BROKER_URL",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET",
        "MINDBRIDGE_GENERATOR_API_KEY",
        "MINDBRIDGE_TENANT_API_KEYS_JSON",
    ):
        assert expected in reported


def test_no_configuration_value_is_ever_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINDBRIDGE_DATABASE_URL", "postgresql://user:s3cr3t-do-not-print@h/d")
    monkeypatch.setenv("MINDBRIDGE_GENERATOR_API_KEY", "sk-do-not-print")

    main(["--role", "api"], prog="mindbridge config check")

    printed = capsys.readouterr().out
    assert "s3cr3t-do-not-print" not in printed
    assert "sk-do-not-print" not in printed


def test_an_unloaded_env_file_is_named_as_a_likely_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MINDBRIDGE_DATABASE_URL=postgresql://u:p@h/d\n", encoding="utf-8"
    )
    for name in list(os.environ):
        if name.startswith("MINDBRIDGE_"):
            monkeypatch.delenv(name, raising=False)

    main(["--role", "api"], prog="mindbridge config check")

    assert "--env-file" in capsys.readouterr().out
```

Add `import os` to the test module's imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen pytest tests/unit/test_config_cli.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mindbridge.config_cli'`

- [ ] **Step 3: Give the missing-value failure a name to catch**

In `src/mindbridge/configuration.py`, add the exception and raise it from
`require_environment_value`:

```python
class MissingConfigurationError(ValueError):
    """Raised when a required setting is absent, carrying the variable it names.

    A `ValueError` subclass so every existing caller and test keeps working, and named so the
    pre-flight report can learn a role's requirements from the role itself instead of from a
    table that would drift away from it.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} must be configured")
        self.name = name
```

Then change `require_environment_value`'s raise from
`raise ValueError(f"{name} must be configured")` to `raise MissingConfigurationError(name)`.

- [ ] **Step 4: Write the command**

Create `src/mindbridge/config_cli.py`:

```python
"""The pre-flight configuration report: what a role still needs, in one pass.

Starting a process is the only validator MindBridge had, and it fails on the first missing
value — so an operator missing five discovered them one restart at a time. This command asks a
role's own settings class what it requires, substituting a placeholder for each complaint and
retrying, so the list of requirements comes from the class rather than from a table here that
would drift away from it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mindbridge.cli import parser
from mindbridge.configuration import (
    CREDENTIAL_VARIABLES,
    MissingConfigurationError,
    configuration_source,
    require_environment_value,
)

_PROBE_CEILING = 64
"""Bound on the ask-and-retry loop, so a non-missing failure cannot spin."""

_PLACEHOLDERS: Mapping[str, str] = {
    "MINDBRIDGE_DATABASE_URL": "postgresql://placeholder:placeholder@localhost:5432/placeholder",
    "MINDBRIDGE_TASK_BROKER_URL": "redis://localhost:6379/0",
    "MINDBRIDGE_TENANT_API_KEYS_JSON": '{"placeholder":["' + "0" * 48 + '"]}',
}
"""Values that satisfy a format check without being usable.

A bare non-empty string is enough for most variables, but a DSN, a broker URL, and the tenant
key map are parsed as they are read, so the probe would stop at the format error instead of
continuing to the next missing name.
"""


def _settings_probe(role: str) -> Callable[[Mapping[str, str]], object]:
    """Return the callable that builds one role's settings, imported only when asked."""
    if role in {"api", "mcp"}:
        from mindbridge.api.runtime import Settings

        return Settings.from_environment
    if role == "worker":
        from mindbridge.worker import WorkerSettings

        return WorkerSettings.from_environment
    if role == "consolidate":
        from mindbridge.consolidation_cli import ConsolidationSettings

        return ConsolidationSettings.from_environment
    if role == "lifecycle":
        from mindbridge.infrastructure.s3 import S3MediaAccess

        # The sweep reads MINDBRIDGE_DATABASE_URL at lifecycle_cli.py:134, separately from the
        # media access it builds only for --reclaim-orphan-clips. Probing the storage alone
        # would report a lifecycle run as ready with no database configured.
        def _lifecycle(source: Mapping[str, str]) -> object:
            require_environment_value(source, "MINDBRIDGE_DATABASE_URL")
            return S3MediaAccess.from_environment(source)

        return _lifecycle
    return lambda source: require_environment_value(source, "MINDBRIDGE_API_KEY")


ROLES: tuple[str, ...] = ("api", "mcp", "worker", "consolidate", "lifecycle", "edge-sync")
"""The six roles `docs/configuration.md` documents, addressable by name."""


def _missing(role: str, resolved: Mapping[str, str]) -> tuple[list[str], str | None]:
    """List every required variable this role cannot resolve, plus any other failure."""
    build = _settings_probe(role)
    probed = dict(resolved)
    missing: list[str] = []
    for _ in range(_PROBE_CEILING):
        try:
            build(probed)
        except MissingConfigurationError as error:
            missing.append(error.name)
            probed[error.name] = _PLACEHOLDERS.get(error.name, "placeholder" * 6)
            continue
        except Exception as error:  # noqa: BLE001 - reported, not handled
            return missing, f"{type(error).__name__}: {error}"
        return missing, None
    return missing, f"more than {_PROBE_CEILING} settings are missing"


def main(argv: Sequence[str], *, prog: str) -> int:
    """Report what one role still needs before it could start."""
    built = parser(
        prog=prog,
        description="Report whether one role's configuration is complete.",
        epilog=(
            "Values are never printed: a credential is reported as present or missing and\n"
            "nothing more, because the same code path handles credentials and structure."
        ),
    )
    built.add_argument(
        "--role",
        required=True,
        choices=ROLES,
        help="the process whose configuration to check",
    )
    arguments = built.parse_args(list(argv))
    resolved = configuration_source()
    missing, failure = _missing(arguments.role, resolved)

    for name in sorted(missing):
        kind = "credential" if name in CREDENTIAL_VARIABLES else "setting"
        print(f"missing  {name}  ({kind}, required)")
    for name in sorted(name for name in resolved if name.startswith("MINDBRIDGE_")):
        if name in missing:
            continue
        origin = "environment" if name in os.environ else "mindbridge.toml"
        print(f"present  {name}  (from {origin})")
    if failure is not None:
        print(f"invalid  {failure}")
    if missing and Path(".env").is_file():
        print("\nFound .env, which nothing here loads. Try `uv run --env-file .env ...`.")
    if not missing and failure is None:
        print(f"\n{arguments.role} is ready.")
        return 0
    print(f"\n{arguments.role} is not ready: {len(missing)} missing.")
    return 1
```

- [ ] **Step 5: Register the command**

In `src/mindbridge/cli.py`, add to the `COMMANDS` dict, keeping it beside the other one-token
commands:

```text
    ("config", "check"): Command(
        "mindbridge.config_cli",
        "Report whether one role's configuration is complete",
    ),
```

No `extra=` — the check must run on a bare install, which is exactly when an operator needs it.
`GROUPS` derives from `COMMANDS`, so `config` becomes a group with no further change.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --frozen pytest tests/unit/test_config_cli.py -q`
Expected: PASS

- [ ] **Step 7: Verify the report by hand, on an empty environment**

```bash
env -u MINDBRIDGE_DATABASE_URL -u MINDBRIDGE_TASK_BROKER_URL \
  uv run --frozen mindbridge config check --role api
```

Expected: several `missing` lines in one pass — not one. A single line means the probe loop is
exiting early and criterion 2 is unmet.

- [ ] **Step 8: Mutation-check the one-pass claim**

Replace the loop body's `continue` with `return missing, None` so the probe reports only the
first failure. Run `uv run --frozen pytest tests/unit/test_config_cli.py -q` and expect
`test_all_missing_settings_are_reported_in_one_pass` to FAIL. Revert.

- [ ] **Step 9: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add src/mindbridge/config_cli.py src/mindbridge/configuration.py src/mindbridge/cli.py \
  tests/unit/test_config_cli.py
git commit -m "Report every missing setting before a role starts, not one per restart"
```

---

### Task 6: The two root files, and the documentation that describes them

**Files:**

- Create: `mindbridge.toml`
- Create: `.env.example`
- Modify: `docs/configuration.md`
- Modify: `docs/quickstart.md:66-88`
- Modify: `docs/deployment.md`
- Modify: `docs/api/cli.md`
- Test: `tests/unit/test_configuration.py`

- [ ] **Step 1: Write the failing test**

The committed file must be loadable and must not carry a credential — the guard is worth
nothing if the shipped file was never run through it.

```python
def test_the_committed_configuration_file_loads_and_carries_no_credential() -> None:
    resolved = configuration_source({}, path=Path("mindbridge.toml"))

    assert resolved["MINDBRIDGE_OBJECT_STORAGE_BUCKET"] == "mindbridge-media"
    assert resolved["MINDBRIDGE_EMBEDDING_DIMENSION"] == "1024"
    assert not CREDENTIAL_VARIABLES & set(resolved)
    # Every credential the example names must be one the loader would refuse in the file.
    named = {
        line.split("=", 1)[0]
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert named <= CREDENTIAL_VARIABLES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -k committed -q`
Expected: FAIL — `mindbridge.toml` does not exist

- [ ] **Step 3: Write `mindbridge.toml`**

```toml
# MindBridge's non-secret configuration. This file is committed; credentials are not in it.
#
# Every key here is overridden by its environment variable, so a container or a CI job can
# change one value without rebuilding anything. `mindbridge config check --role <role>` reports
# which source won for each setting. Credentials live in the environment only -- see
# .env.example -- and this file is refused at load time if it carries one.

# Cosine floor applied to retrieved evidence. 0.0 keeps everything the index returns.
minimum_embedding_similarity = 0.0

[database]
max_pool_size = 32

[object_storage]
bucket = "mindbridge-media"
# Unset for AWS S3, which needs no endpoint. MinIO and other S3-compatible stores need one.
endpoint_url = "http://localhost:9000"

# One search space and one vector width shared by the index and every encoder. Both encoder
# sections below inherit these three keys, so the deployment states them once.
[embedding]
dimension = 1024
space_id = "jinaai/jina-embeddings-v5-omni-small-retrieval-1024"
space_revision = "omni@12949877f0092093f366c6450340011320152a05"

[generator]
plugin = "openai"
endpoint = "http://localhost:8001/v1"
model_id = "qwen3.8-max"
model_revision = "deployment-2026-08-11"
# The Worker sizes its Celery budget from this, so a slow generator needs both to agree. A
# perception clip can take 16-25 minutes, which is why the default is not a web timeout.
request_timeout_seconds = 1800

[embedder]
plugin = "openai"
endpoint = "http://localhost:8002/v1"
model_id = "jinaai/jina-embeddings-v5-omni-small-retrieval"
model_revision = "12949877f0092093f366c6450340011320152a05"

# The Worker's local image, video, and audio encoder. model_id here is a Hugging Face
# repository ID; the embedder's model_id above is an endpoint's alias for a served model.
[media_embedder]
plugin = "jina"
model_id = "jinaai/jina-embeddings-v5-omni-small-retrieval"
model_revision = "12949877f0092093f366c6450340011320152a05"
```

- [ ] **Step 4: Write `.env.example`**

```bash
# Credentials only. Copy to .env and fill in; .env is gitignored and never committed.
#
# Nothing in MindBridge loads this file -- every deployment target already does it natively:
#   uv run --env-file .env <command>
#   docker compose:  env_file: [.env]
#   systemd:         EnvironmentFile=/etc/mindbridge.env
#
# Everything that is not a credential lives in mindbridge.toml, which is committed.
# API keys must be at least 32 characters; `openssl rand -hex 24` gives 48.

MINDBRIDGE_DATABASE_URL=postgresql://mindbridge:mindbridge@localhost:5432/mindbridge
MINDBRIDGE_TASK_BROKER_URL=redis://localhost:6379/0
MINDBRIDGE_GENERATOR_API_KEY=
MINDBRIDGE_EMBEDDER_API_KEY=
MINDBRIDGE_TENANT_API_KEYS_JSON={"tenant_01":["replace-with-48-hex-characters"]}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run --frozen pytest tests/unit/test_configuration.py -k committed -q`
Expected: PASS

- [ ] **Step 6: Rewrite `docs/configuration.md`'s opening and matrix**

The opening currently reads "Every process is configured entirely through environment
variables. There is no config file, and credentials are never accepted as CLI flags". The first
half is now false; the second half is the property the split preserves and should lead. Replace
the first paragraph with:

```markdown
Credentials are configured entirely through environment variables, and are never accepted as a
CLI flag or read from a file — so a recorded invocation, a process list, a systemd unit, and
the repository itself never carry a secret. Everything that is not a credential lives in
`mindbridge.toml`, which is committed, commented, and diffable.

Each setting's environment variable overrides the file, so a container or a CI job changes one
value without rebuilding anything. `mindbridge config check --role <role>` reports which source
won for each setting, and every setting a role still needs, in one pass.

Configuration is validated at startup, not at first request. A deployment with a wrong value
fails to start rather than failing one call an hour later.
```

Then, in the "Which process reads what" matrix, add a `Source` column marking each variable
`file` or `env`, and add the `MINDBRIDGE_CONFIG_FILE` row. Keep the six role columns as they
are — role coverage has not changed.

- [ ] **Step 7: Rewrite `docs/quickstart.md` step 4**

Replace the sixteen `export` lines with:

````markdown
Copy the credential template and fill in the two API keys:

```bash
cp .env.example .env
```

Everything that is not a credential is already in the committed `mindbridge.toml` — endpoints,
model IDs, revisions, the embedding space, and the pool size. Edit that file rather than
exporting variables; `mindbridge config check` will tell you which source a value came from.

Generate a tenant key and put it in `.env`:

```bash
openssl rand -hex 24
```

Two things bite first-time users:

- **API keys must be at least 32 characters.** A shorter one fails at startup, not at the first
  request. `openssl rand -hex 24` gives 48.
- **`MINDBRIDGE_TENANT_API_KEYS_JSON` is required.** The REST factory refuses to build without
  it. There is no anonymous mode; only `/healthz` is public.

Nothing in MindBridge loads `.env` — pass it to whatever runs the process:

```bash
uv run --env-file .env mindbridge config check --role api
```
````

Then update step 5's launch command to `uv run --env-file .env --extra server uvicorn ...`.

- [ ] **Step 8: Sweep the remaining docs**

```bash
grep -rn 'export MINDBRIDGE_' docs/ README.md
```

Every hit describes the old contract. For each, decide whether the variable is a credential
(keep it as an environment variable, shown via `.env`) or structure (move it into a
`mindbridge.toml` snippet). `docs/deployment.md` and `docs/api/cli.md` both carry blocks that
need this treatment.

- [ ] **Step 9: Run the Documentation gate**

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input './*.md' './docs/**/*.md'
```

Expected: 0 errors from each. `MD040` catches a fenced block with no language, and `ruff format`
reformats Python blocks inside Markdown — so run the full gate too, not just these two.

- [ ] **Step 10: Run the full gate and commit**

```bash
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
git add mindbridge.toml .env.example docs/ tests/unit/test_configuration.py
git commit -m "Ship the file the deployment contract now describes"
```

---

## Verification against the success criteria

Run after Task 6, before opening the pull request. Each line maps to one criterion in the spec.

- [ ] **Criterion 1** — `wc -l .env.example` shows 5 assignments, and `grep -c '^export' docs/quickstart.md` is 0.
- [ ] **Criterion 2** — `uv run --frozen mindbridge config check --role api` on an empty environment lists more than one missing setting, and no value appears in the output.
- [ ] **Criterion 3** — `git stash` the two root files, run `uv run --frozen pytest -W error -q`, and confirm the suite is green with no config file present. Unstash.
- [ ] **Criterion 4** — `printf '[generator]\napi_key = "sk"\n' > /tmp/c.toml && MINDBRIDGE_CONFIG_FILE=/tmp/c.toml uv run --frozen mindbridge config check --role api` fails naming `MINDBRIDGE_GENERATOR_API_KEY`.
- [ ] **Criterion 5** — the Task 1 Step 7 wheel check, re-run against the final tree.

## Rebase before the gate

Merge or rebase `master` first, then re-run the whole gate. A clean merge still produces a
broken tree when two branches touch the same contract, and this branch touches a file every
role imports.

```bash
git fetch origin && git merge origin/master
uv run --frozen ruff format --check . && uv run --frozen ruff check . && uv run --frozen mypy && uv run --frozen pytest -W error -q
```

Scan for duplicated symbols after any hand-resolved conflict: a merge resolved by hand can
duplicate a definition outside the conflict markers, and `configuration.py` gains eight new
names in this branch.
