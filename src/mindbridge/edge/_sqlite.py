"""Shared SQLite connection settings for the durable edge stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a durable connection: WAL for readers, FULL sync for power loss.

    Closes it on the way out, which `with sqlite3.connect(...)` does not do on its own:
    `Connection.__exit__` commits or rolls back and leaves the handle open. Every store here
    opens one per operation, so leaving the close to the garbage collector costs a file
    descriptor per call on a device that runs for weeks. 3.13 made that visible by reporting
    each unclosed connection as a `ResourceWarning`, which `pytest -W error` fails on.

    The transaction still spans the caller's whole block, so the `with connect(...)` sites
    that predate this keep committing exactly once, on the same condition, as before.
    """
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    try:
        with connection:
            yield connection
    finally:
        connection.close()
