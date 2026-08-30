"""Small packaging contracts for the supported local product."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from typing import cast

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "mindbridge"
DOCUMENT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = cast(dict[str, object], DOCUMENT["project"])
DEPENDENCIES = cast(list[str], PROJECT["dependencies"])
EXTRAS = cast(dict[str, list[str]], PROJECT["optional-dependencies"])
SCRIPTS = cast(dict[str, str], PROJECT["scripts"])


def test_base_import_does_not_load_optional_protocols() -> None:
    code = (
        "import json, sys; import mindbridge; "
        "blocked = {'fastapi', 'mcp', 'openai', 'sentence_transformers', 'torch', "
        "'transformers', 'uvicorn', 'vllm'}; "
        "print(json.dumps(sorted(blocked.intersection(sys.modules))))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == []


def test_full_modal_contract_is_exported_from_the_package_root() -> None:
    package = importlib.import_module("mindbridge")
    names = {
        "Blob",
        "AssetRef",
        "ContentAtom",
        "ContentInput",
        "MemoryType",
        "Modality",
        "EmbeddingBackend",
        "GenerationBackend",
        "JinaOmniEmbedder",
        "ModelInput",
        "OpenAIModels",
        "EmbedTask",
        "SentenceTransformersEmbedder",
        "SpeakerNotFoundError",
        "StreamingGenerationBackend",
        "TranscriptionBackend",
    }

    assert names <= set(package.__all__)
    assert all(hasattr(package, name) for name in names)


def test_product_does_not_address_benchmarks() -> None:
    benchmark_root = SOURCE / "benchmarks"
    violations = {
        str(path.relative_to(SOURCE)): references
        for path in SOURCE.rglob("*.py")
        if not path.is_relative_to(benchmark_root) and (references := _benchmark_references(path))
    }
    assert violations == {}


def test_imported_distributions_are_declared() -> None:
    """Every third-party root a product module imports must be declared by this project.

    The walk behind ``_benchmark_references`` already collected these imports and then threw the
    third-party ones away. Keeping them turns the packaging tests around:
    ``test_dependency_surface_is_exact`` is a tripwire against *adding* a dependency and cannot
    see one that is imported but undeclared. This catches the direct case; a transitive one --
    an upstream package importing what it never declares -- needs the isolated CI legs.
    """
    declared = {
        canonicalize_name(_name(item))
        for item in [*DEPENDENCIES, *(item for items in EXTRAS.values() for item in items)]
    }
    provided = _installed_roots()
    undeclared = {
        f"{path.relative_to(SOURCE)}:{line}": root
        for path in sorted(SOURCE.rglob("*.py"))
        for root, line in sorted(_imports(path).third_party)
        if not provided.get(root, frozenset()) & declared
    }
    assert undeclared == {}, (
        f"imported but declared in no extra: {undeclared}; add each distribution to the "
        "narrowest optional extra that reaches the import"
    )


def _installed_roots() -> dict[str, frozenset[str]]:
    """Map each importable root to the installed distributions that provide it.

    ``importlib.metadata.packages_distributions`` reads only ``top_level.txt`` before Python
    3.12, and modern wheels omit that file, so on 3.10 and 3.11 it reports almost nothing and
    the declaration check above would call every third-party import undeclared. Reading the
    recorded files instead is what 3.12 does internally and behaves the same on every
    supported interpreter.
    """
    roots: dict[str, set[str]] = {}
    for distribution in distributions():
        name = distribution.metadata["Name"]
        if not name:
            continue
        canonical = canonicalize_name(name)
        for recorded in distribution.files or ():
            root = recorded.parts[0]
            if root.startswith((".", "_distutils_hack")) or root.endswith(".dist-info"):
                continue
            if root.endswith(".py"):
                root = root[: -len(".py")]
            elif "." in root:
                continue
            roots.setdefault(root, set()).add(canonical)
    return {root: frozenset(names) for root, names in roots.items()}


def test_deferred_and_conditional_imports_are_classified(tmp_path: Path) -> None:
    """Pin the classification the declaration check depends on, on inputs the source lacks.

    Every deferred backend in this package loads through ``import_module``, so a scan that only
    read import statements would run its new branch over a corpus with nothing to find.
    """
    module = tmp_path / "sample.py"
    module.write_text(
        "import os\n"
        "from . import sibling\n"
        "from mindbridge.types import Modality\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import lazyquiet\n"
        "PROMPT = 'answer using only the supplied memories'\n"
        "def load() -> object:\n"
        "    find_spec('probed')\n"
        "    return import_module('deferred.submodule')\n",
        encoding="utf-8",
    )
    found = _imports(module)
    # Deferred and TYPE_CHECKING imports are declarations; stdlib, relative, first-party, and
    # the prose constant are not, and prose reaching this set would fail every product module.
    assert dict(found.third_party) == {"lazyquiet": 6, "deferred": 10, "probed": 9}
    assert {"sibling", "os", "answer using only the supplied memories"} <= found.names


def _benchmark_references(path: Path) -> set[str]:
    return {
        name
        for name in _imports(path).names
        if name == "benchmarks" or name.startswith(("benchmarks.", "mindbridge.benchmarks"))
    }


@dataclass(frozen=True, slots=True)
class _Imports:
    """Module names a file references, and the third-party roots among them with line numbers."""

    names: frozenset[str]
    third_party: frozenset[tuple[str, int]]


_FIRST_PARTY = frozenset({"mindbridge", *sys.stdlib_module_names})


def _imports(path: Path) -> _Imports:
    """Collect module names imported statically, lazily, or named as a bare string constant.

    ``names`` keeps every string constant so a product module cannot reach the benchmarks by
    spelling the path instead of importing it. ``third_party`` keeps only names a real import
    statement or a deferred ``import_module``/``find_spec`` call resolves, because every heavy
    backend here loads that way and an unfiltered constant sweep would return application text.
    Relative imports name no distribution, so they stay out of ``third_party``.
    """
    names: set[str] = set()
    third_party: set[tuple[str, int]] = set()

    def add(name: str, line: int, *, resolvable: bool) -> None:
        names.add(name)
        root = name.split(".", 1)[0]
        if resolvable and root and root not in _FIRST_PARTY:
            third_party.add((root, line))

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, node.lineno, resolvable=True)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            add(module, node.lineno, resolvable=not node.level)
            for alias in node.names:
                add(f"{module}.{alias.name}".lstrip("."), node.lineno, resolvable=False)
        elif isinstance(node, ast.Call) and (loaded := _loader_argument(node)) is not None:
            add(loaded, node.lineno, resolvable=True)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            add(node.value, node.lineno, resolvable=False)
    return _Imports(frozenset(names), frozenset(third_party))


def _loader_argument(node: ast.Call) -> str | None:
    """Return the module named by a single-argument ``import_module``/``find_spec`` call."""
    called = node.func
    name = called.attr if isinstance(called, ast.Attribute) else None
    name = called.id if isinstance(called, ast.Name) else name
    if name not in {"import_module", "find_spec"} or len(node.args) != 1:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def test_console_script_targets_exist() -> None:
    assert set(SCRIPTS) == {"mindbridge", "mindbridge-bench"}
    for target in SCRIPTS.values():
        module_name, attribute = target.split(":", 1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_the_product_cli_cannot_reach_the_benchmark_family(tmp_path: Path) -> None:
    """Two console scripts exist because one dispatcher cannot satisfy the guard above.

    ``test_product_does_not_address_benchmarks`` already covers every product module, but it passes
    trivially while the CLI is small. This pins the reason the layout is what it is: the scan reads
    string constants, so even a lazy ``import_module`` of the other family trips it.
    """
    product_cli = SOURCE / "cli.py"
    assert product_cli.exists()
    assert _benchmark_references(product_cli) == set()

    dispatcher = tmp_path / "single_tree.py"
    dispatcher.write_text(
        "from importlib import import_module\n"
        "def main() -> int:\n"
        "    return import_module('mindbridge.benchmarks.cli').main()\n",
        encoding="utf-8",
    )
    assert _benchmark_references(dispatcher) == {"mindbridge.benchmarks.cli"}


def test_dependency_surface_is_exact() -> None:
    assert {_name(item) for item in DEPENDENCIES} == {
        "opentelemetry-api",
        "pydantic",
        "zvec",
    }
    assert {extra: {_name(item) for item in items} for extra, items in EXTRAS.items()} == {
        "benchmarks": {"httpx", "huggingface-hub", "nltk", "opentelemetry-sdk", "pyarrow"},
        "observability": {"opentelemetry-sdk"},
        "openai": {"openai"},
        "local": {
            "cairosvg",
            "funasr",
            "librosa",
            "numpy",
            "sentence-transformers",
            "soundfile",
            "torch",
            "torchaudio",
        },
        "server": {"fastapi", "starlette", "uvicorn"},
        "mcp": {"mcp"},
    }
    declared = set(EXTRAS) | {_name(item) for item in DEPENDENCIES}
    declared |= {_name(item) for items in EXTRAS.values() for item in items}
    forbidden = (
        "postgres",
        "pgvector",
        "psycopg",
        "celery",
        "redis",
        "boto",
        "edge",
        "media",
        "cloud-models",
    )
    assert {name for name in declared if any(word in name for word in forbidden)} == set()


def _name(requirement: str) -> str:
    return Requirement(requirement).name.lower()


@pytest.mark.parametrize(
    "modules",
    [
        ("mindbridge", "mindbridge.memory", "mindbridge.infrastructure.local"),
        ("mindbridge.api.app", "mindbridge.api.errors"),
        ("mindbridge.api.mcp",),
    ],
    ids=("base", "server", "mcp"),
)
def test_supported_modules_import(modules: tuple[str, ...]) -> None:
    for module in modules:
        importlib.import_module(module)
