"""Package-level smoke tests."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import mindbridge

SOURCE = Path(__file__).parents[1] / "src" / "mindbridge"


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
