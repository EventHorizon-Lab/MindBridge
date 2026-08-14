"""Package-level smoke tests."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import mindbridge


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
    application = Path(__file__).parents[1] / "src" / "mindbridge" / "application"
    violations = []
    for path in application.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("mindbridge.models") for alias in node.names
            ):
                violations.append(str(path.relative_to(application)))
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "mindbridge.models"
            ):
                violations.append(str(path.relative_to(application)))

    assert violations == []
