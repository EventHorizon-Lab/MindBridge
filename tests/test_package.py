"""Small packaging contracts for the supported local product."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from packaging.requirements import Requirement

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
        "blocked = {'fastapi', 'mcp', 'sentence_transformers', 'torch', "
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
        "URL",
        "Blob",
        "AssetRef",
        "ContentAtom",
        "ContentInput",
        "MemoryType",
        "Modality",
        "EmbeddingBackend",
        "JinaOmniEmbedder",
        "ModelBackend",
        "ModelCapabilities",
        "ModelInput",
        "OpenAIHTTP",
        "EmbedTask",
        "SentenceTransformersEmbedder",
        "SpeakerNotFoundError",
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


def _benchmark_references(path: Path) -> set[str]:
    references: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            references.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            references.add(module)
            references.update(f"{module}.{alias.name}".lstrip(".") for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            references.add(node.value)
    return {
        name
        for name in references
        if name == "benchmarks" or name.startswith(("benchmarks.", "mindbridge.benchmarks"))
    }


def test_console_script_targets_exist() -> None:
    assert set(SCRIPTS) == {"mindbridge", "mindbridge-bench", "mindbridge-mcp"}
    for target in SCRIPTS.values():
        module_name, attribute = target.split(":", 1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_dependency_surface_is_exact() -> None:
    assert {_name(item) for item in DEPENDENCIES} == {"httpx", "pydantic", "zvec"}
    assert {extra: {_name(item) for item in items} for extra, items in EXTRAS.items()} == {
        "local": {"funasr", "librosa", "numpy", "sentence-transformers", "soundfile"},
        "vllm": {"vllm"},
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
        "opentelemetry",
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
        ("mindbridge.api.app", "mindbridge.api.errors", "mindbridge.server"),
        ("mindbridge.api.mcp",),
    ],
    ids=("base", "server", "mcp"),
)
def test_supported_modules_import(modules: tuple[str, ...]) -> None:
    for module in modules:
        importlib.import_module(module)
