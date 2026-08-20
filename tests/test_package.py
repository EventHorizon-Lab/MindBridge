"""Package-level smoke tests."""

import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    # tomllib landed in 3.11. On the 3.10 half of the matrix mypy itself requires the
    # tomli backport, so it is present wherever the dev group that runs this test is.
    import tomli as tomllib

import mindbridge
from mindbridge.benchmarks.cli import RUNNERS

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "mindbridge"


def test_package_can_be_imported() -> None:
    """The installed package is importable through the src layout."""
    assert mindbridge.__name__ == "mindbridge"


def test_edge_identity_import_does_not_load_network_or_server_stack() -> None:
    """SQLite identity must stay independent from network and server adapters."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import mindbridge.edge.identity; "
                "excluded = {'boto3', 'botocore', 'celery', 'fastapi', 'mcp', "
                "'openai', 'pgvector', 'psycopg'}; "
                "print(json.dumps(sorted(excluded.intersection(sys.modules))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_model_capabilities_do_not_import_provider_adapters() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import mindbridge.models; "
                "excluded = {'openai', 'sentence_transformers', 'transformers'}; "
                "print(json.dumps(sorted(excluded.intersection(sys.modules))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_application_does_not_depend_on_model_adapters() -> None:
    assert _modules_importing(SOURCE / "application", "mindbridge.models") == []


def test_product_code_does_not_depend_on_the_benchmark_harness() -> None:
    """Evaluation adapters consume the public contracts; nothing may consume them back."""
    assert (
        _modules_importing(
            SOURCE,
            "mindbridge.benchmarks",
            exclude=SOURCE / "benchmarks",
        )
        == []
    )


def test_product_code_does_not_address_the_benchmark_harness_by_name() -> None:
    """A table of module strings is a dependency the import guard above cannot see.

    `mindbridge.cli` dispatches by importing module paths it holds as text, so an entry for
    a benchmark runner would couple the product entry point to the harness while leaving
    every `ast.Import` check green. Prose may explain the boundary; code may not cross it,
    so docstrings are exempt and any other string literal is not.
    """
    assert (
        _modules_named_in_code(
            SOURCE,
            "mindbridge.benchmarks",
            exclude=SOURCE / "benchmarks",
        )
        == []
    )


def _modules_named_in_code(directory: Path, prefix: str, *, exclude: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*.py")
        if not path.is_relative_to(exclude)
        and any(name.startswith(prefix) for name in _string_constants(path))
    )


def _string_constants(path: Path) -> set[str]:
    """Every string literal in one module except the docstrings that document it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    documented = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in documented
    }


def test_every_declared_console_script_ships_in_the_wheel() -> None:
    """A `[project.scripts]` target must survive the wheel build that declares it.

    The wheel records its console scripts from `[project.scripts]` whatever the build
    excludes, so aiming one at an excluded package installs a command that raises
    `ModuleNotFoundError` on every invocation -- and does it silently, because building
    and installing both succeed. The leaf checks above cannot catch that: they assert
    shipped modules never *import* the harness, which stays true exactly when the harness
    is the thing missing. `uv run` hides it too, by installing editable so the `.pth`
    maps the whole tree.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = [
        Path(pattern)
        for pattern in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"].get("exclude", ())
    ]

    unshipped = sorted(
        f"{name} -> {target}"
        for name, target in pyproject["project"]["scripts"].items()
        if not _ships(target.partition(":")[0], excluded)
    )

    assert unshipped == []


def _ships(module: str, excluded: list[Path]) -> bool:
    """Report whether an entry point's module is on disk and inside the built wheel."""
    relative = Path("src", *module.split("."))
    if any(relative.is_relative_to(root) for root in excluded):
        return False
    return (ROOT / relative.with_suffix(".py")).is_file() or (
        ROOT / relative / "__init__.py"
    ).is_file()


def _modules_importing(directory: Path, prefix: str, *, exclude: Path | None = None) -> list[str]:
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*.py")
        if (exclude is None or not path.is_relative_to(exclude))
        and any(name.startswith(prefix) for name in _imported_modules(path))
    )


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


# The scenarios a MindBridge install can serve, as `extra name -> the modules that install
# is responsible for`. The empty name is the bare install. Every module under `src/mindbridge`
# belongs to exactly one of them, so a new module without a scenario fails the partition test
# below rather than quietly relying on whatever extra a developer happened to have synced.
SCENARIOS: dict[str, tuple[str, ...]] = {
    "": ("__init__", "configuration", "contracts", "core", "file_integrity", "prompts", "sdk"),
    "edge": ("edge",),
    "server": (
        "api",
        "application",
        "celery_app",
        "cli",
        "consolidation_cli",
        "infrastructure",
        "lifecycle_cli",
        "models",
        "server",
        "telemetry",
        "worker",
    ),
    "benchmarks": ("benchmarks",),
    "media": ("media",),
}

ARTIFACT_EXTRAS = frozenset({"cloud-models"})
"""Extras that carry weights and decoders rather than serving a subtree of their own.

`cloud-models` is reached only through lazy imports -- `models/jina.py` and
`media/clipping.py` raise `ModelUnavailableError` naming it -- so no module is import-broken
without it and it owns no scenario.
"""

PROVIDERS: dict[str, str] = {
    "av": "av",
    "boto3": "boto3",
    "botocore": "botocore",
    "celery": "celery",
    "cryptography": "cryptography",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "huggingface_hub": "huggingface-hub",
    "mcp": "mcp",
    "mcp_types": "mcp-types",
    "mypy_boto3_s3": "boto3-stubs",
    "numpy": "numpy",
    "openai": "openai",
    # Longest prefix wins, so the instrumentation and exporter packages are told apart from
    # the API even though they all import as `opentelemetry.*`.
    "opentelemetry": "opentelemetry-api",
    "opentelemetry.exporter": "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry.instrumentation.botocore": "opentelemetry-instrumentation-botocore",
    "opentelemetry.instrumentation.celery": "opentelemetry-instrumentation-celery",
    "opentelemetry.instrumentation.fastapi": "opentelemetry-instrumentation-fastapi",
    "opentelemetry.instrumentation.httpx": "opentelemetry-instrumentation-httpx",
    "opentelemetry.instrumentation.psycopg": "opentelemetry-instrumentation-psycopg",
    "opentelemetry.sdk": "opentelemetry-sdk",
    "pgvector": "pgvector",
    "PIL": "pillow",
    "psycopg": "psycopg",
    "psycopg_pool": "psycopg",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "pytest": "pytest",
    "sentence_transformers": "sentence-transformers",
    "soundfile": "soundfile",
    "starlette": "starlette",
    "torch": "torch",
    "transformers": "transformers",
}
"""Import name to the distribution that provides it, spelled as `pyproject.toml` spells it."""

PLATFORM_SUPPLIED = frozenset({"cv2", "funasr", "insightface", "onnxruntime"})
"""Vendor accelerator SDKs that arrive with the platform image, never from an extra.

`docs/edge.md` installs these from JetPack, OpenExplorer, RKNN or OpenVINO. Declaring them
would pull the generic PyPI wheel over the vendor build on four of the five supported
platforms, so `edge/identity_inference.py` and `edge/identity_diarization.py` import them
lazily and raise `ModelUnavailableError` naming the platform requirement.
"""

RUNTIME_ONLY = frozenset({"torchvision", "uvicorn"})
"""Declared dependencies that nothing in this repository imports, and why they stay.

`uvicorn` is the ASGI server `docs/deployment.md` runs as a command. `torchvision` is loaded
by Jina Omni's own Qwen3-VL processor, which swallows the ImportError and embeds text-only.
"""


@cache
def _extras() -> Mapping[str, list[str]]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras: Mapping[str, list[str]] = pyproject["project"]["optional-dependencies"]
    return extras


@cache
def _base_dependencies() -> tuple[str, ...]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared: list[str] = pyproject["project"]["dependencies"]
    return tuple(declared)


def _distribution(requirement: str) -> str:
    """The distribution one requirement names, without its extras, marker or version."""
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]


@cache
def _declared(extra: str) -> frozenset[str]:
    """Every distribution a scenario install puts on disk, following self-references."""
    declared = {_distribution(requirement) for requirement in _base_dependencies()}
    pending = [extra] if extra else []
    while pending:
        for requirement in _extras()[pending.pop()]:
            name = _distribution(requirement)
            if name == "mindbridge":
                pending.extend(re.findall(r"[\w-]+", requirement.partition("[")[2]))
                continue
            declared.add(name)
    return frozenset(declared)


def _provider(module: str) -> str | None:
    """The distribution providing an imported module, by longest matching prefix."""
    parts = module.split(".")
    for size in range(len(parts), 0, -1):
        found = PROVIDERS.get(".".join(parts[:size]))
        if found is not None:
            return found
    return None


def _scenario_of(path: Path) -> str:
    """Which scenario owns one module under `src/mindbridge`."""
    area = path.relative_to(SOURCE).parts[0].removesuffix(".py")
    for extra, areas in SCENARIOS.items():
        if area in areas:
            return extra
    raise AssertionError(f"{path} belongs to no scenario in SCENARIOS")


def _third_party(modules: Iterable[str]) -> set[str]:
    return {
        module
        for module in modules
        if module.partition(".")[0] not in sys.stdlib_module_names
        and not module.startswith("mindbridge")
        and module.partition(".")[0] not in PLATFORM_SUPPLIED
    }


def _lazily_imported_modules(path: Path) -> set[str]:
    """Modules named as strings to `import_module`, `__import__` or `importorskip`.

    An `ast.Import` check cannot see these, and they are how every heavy dependency in the
    tree is reached -- torch, the decoders, pyarrow -- so leaving them out would let the
    declaration checks pass on an install that cannot run.
    """
    deferred = {"import_module", "__import__", "importorskip"}
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        called = called or getattr(node.func, "id", "")
        named = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        if called in deferred and named:
            modules.add(named[0])
        if called == "version" and named:
            # `importlib.metadata.version` raises without the distribution installed, so a
            # version probe is as real a requirement as an import.
            modules.add(named[0].replace("-", "_"))
    return modules


@cache
def _eagerly_imported_modules(path: Path) -> frozenset[str]:
    """The imports that run when the module is imported, so the ones an install must satisfy.

    Function bodies are deferred until called, and `if TYPE_CHECKING:` blocks never run at
    all -- `mypy_boto3_s3` and the OpenTelemetry SDK types are only ever needed there.
    """
    modules: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if isinstance(child, ast.If) and _tests_type_checking(child.test):
                continue
            if isinstance(child, ast.Import):
                modules.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0 and child.module is not None:
                    modules.add(child.module)
            else:
                visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return frozenset(modules)


def _tests_type_checking(test: ast.expr) -> bool:
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _modules(*directories: Path) -> list[Path]:
    return sorted(
        path
        for directory in directories
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_scenarios_partition_the_package() -> None:
    """Every shipped module belongs to one scenario, and every extra to one role.

    A module outside the table would be installable only by luck, and an extra outside it
    would be one nothing above knows how to check.
    """
    # Derived from the modules themselves: the audit receipts the protect-mcp tooling drops
    # under `src/` carry no Python and are not this package's to place.
    areas = {path.relative_to(SOURCE).parts[0].removesuffix(".py") for path in _modules(SOURCE)}
    assigned = {area for areas_of in SCENARIOS.values() for area in areas_of}
    unassigned = areas - assigned

    assert unassigned == set()
    assert set(_extras()) == (set(SCENARIOS) - {""}) | ARTIFACT_EXTRAS


def _module_name(path: Path) -> str:
    parts = path.relative_to(SOURCE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@cache
def _module_paths(module: str) -> tuple[Path, ...]:
    """Every file Python executes to import one first-party module, parents included."""
    parts = module.split(".")
    found = []
    for size in range(1, len(parts) + 1):
        candidate = SOURCE.parent.joinpath(*parts[:size])
        for path in (candidate / "__init__.py", candidate.with_suffix(".py")):
            if path.is_file():
                found.append(path)
    return tuple(found)


@cache
def _import_graph() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Each first-party module's eager imports, split into outside and first-party edges."""
    outside: dict[str, frozenset[str]] = {}
    inside: dict[str, frozenset[str]] = {}
    for path in _modules(SOURCE):
        module = _module_name(path)
        imported: set[str] = set()
        for parent in _module_paths(module):
            imported |= _eagerly_imported_modules(parent)
        outside[module] = frozenset(_third_party(imported))
        inside[module] = frozenset(
            name for name in imported if name.startswith("mindbridge") and _module_paths(name)
        )
    return outside, inside


def _reached_third_party(module: str) -> dict[str, str]:
    """Third-party modules importing this one runs, mapped to the module that names them.

    The walk follows first-party imports because that is how the real failure happened: no
    benchmark module mentions OpenTelemetry, but every one of them reaches
    `mindbridge.models.jina`, which reaches `mindbridge.telemetry`, which imports it at module
    scope. A per-file check calls that scenario complete; an import of it does not.
    """
    outside, inside = _import_graph()
    reached: dict[str, str] = {}
    seen: set[str] = set()
    pending = [module]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for name in sorted(outside.get(current, ())):
            reached.setdefault(name, current)
        pending.extend(inside.get(current, ()))
    return reached


def test_every_scenario_declares_what_its_own_modules_import() -> None:
    """Importing a scenario's modules must work on that scenario's install alone.

    This is the check `mindbridge-bench jina` failed: the harness reaches
    `mindbridge.telemetry`, and nothing in the benchmarks extra carried the OpenTelemetry API,
    so the runners were importable only next to an unrelated extra that happened to bring it.
    """
    missing = sorted(
        f"[{extra or 'core'}] needs {_provider(name)}:"
        f" {_module_name(path)} reaches {name} through {owner}"
        for extra in SCENARIOS
        for path in _modules(SOURCE)
        if _scenario_of(path) == extra
        for name, owner in _reached_third_party(_module_name(path)).items()
        if _provider(name) not in _declared(extra)
    )

    assert missing == []


def test_every_import_in_the_tree_is_declared_by_some_extra() -> None:
    """A dependency reached lazily still has to be installable by name.

    Nothing here says which extra -- `models/jina.py` is a server module reaching
    cloud-models -- only that some documented install puts it on disk. Without this, a
    transitive gift like starlette or huggingface_hub looks declared until the package that
    brought it drops it.
    """
    installable = {name for extra in _extras() for name in _declared(extra)}
    undeclared = sorted(
        f"{path.relative_to(ROOT)} imports {module}"
        for path in _modules(SOURCE)
        for module in _third_party(_eagerly_imported_modules(path) | _lazily_imported_modules(path))
        if _provider(module) is None or _provider(module) not in installable
    )

    assert undeclared == []


def test_no_extra_declares_a_dependency_nothing_uses() -> None:
    """The other direction: paying to install something no code reaches.

    `openai` sat in the edge extra this way -- edge injects a `Generator` Protocol and never
    loads an adapter -- so every Jetson image carried the client for nothing.
    """
    used = {
        module
        for path in _modules(SOURCE, ROOT / "tests")
        for module in _imported_modules(path) | _lazily_imported_modules(path)
    }
    provided = {_provider(module) for module in used} - {None}
    unused = sorted(
        f"{name} in [{extra}]"
        for extra, requirements in _extras().items()
        for name in map(_distribution, requirements)
        if name not in provided and name != "mindbridge" and name not in RUNTIME_ONLY
    )

    assert unused == []


def test_the_installability_matrix_probes_every_scenario() -> None:
    """CI must install each scenario on its own and import everything it owns.

    The matrix used to install three scenarios and import one module from each, so an extra
    that could not import its own subtree -- benchmarks -- passed. Holding the rows to
    SCENARIOS keeps the job honest as modules move, and keeps a renamed extra from becoming
    a row that installs the bare package: an unknown extra is a warning, not an error, in
    both uv and pip.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    rows = {
        requirement: probe.split()
        for _, requirement, probe in re.findall(
            r"- name: (\S+)\n\s+requirement: (\S+)\n\s+probe: (.+)", workflow
        )
    }
    expected = {
        f".[{extra}]" if extra else ".": sorted(set(areas) - {"__init__"})
        for extra, areas in SCENARIOS.items()
    }

    assert {name: sorted(probe) for name, probe in rows.items()} == expected


def test_every_extra_named_outside_pyproject_exists() -> None:
    """An extra nobody declares installs the bare package and says so in a warning.

    `uv pip install '.[typo]'` and pip alike warn and exit 0, so a renamed or mistyped extra
    is a silent no-op -- in a runbook, in this workflow, and in the message the benchmark
    dispatcher prints when a runner cannot import.
    """
    named: set[str] = set()
    sources = [ROOT / ".github" / "workflows" / "ci.yml", *ROOT.glob("*.md"), *_documentation()]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        # An extra name starts alphanumeric, so a `--extra ...` placeholder matches nothing.
        named.update(re.findall(r"--extra ([A-Za-z0-9][A-Za-z0-9._-]*)", text))
        for group in re.findall(r"(?:\.|mindbridge)\[([A-Za-z0-9][A-Za-z0-9,._-]*)\]", text):
            named.update(group.split(","))
    named.update(runner.extra for runner in RUNNERS.values() if runner.extra is not None)

    assert sorted(named - set(_extras())) == []


def _documentation() -> list[Path]:
    return sorted(path for path in (ROOT / "docs").rglob("*.md"))
