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
