"""Shared SQLite connection settings for the durable edge stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(database_path: Path) -> sqlite3.Connection:
    """Open a durable connection: WAL for readers, FULL sync for power loss."""
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection
