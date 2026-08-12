"""Package-level smoke tests."""

import json
import subprocess
import sys

import mindbridge


def test_package_can_be_imported() -> None:
    """The installed package is importable through the src layout."""
    assert mindbridge.__name__ == "mindbridge"


def test_edge_identity_import_does_not_load_server_stack() -> None:
    """An edge-only import must stay independent from server dependencies."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; import mindbridge.edge.identity; "
                "server = {'celery', 'fastapi', 'mcp', 'openai', 'pgvector', 'psycopg'}; "
                "print(json.dumps(sorted(server.intersection(sys.modules))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
