"""Disposable PostgreSQL fixtures shared by integration checks."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio
from psycopg import AsyncConnection

from mindbridge.infrastructure import PostgresMemoryStore

DATABASE_URL = os.getenv("MINDBRIDGE_TEST_DATABASE_URL")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def database_url() -> AsyncIterator[str]:
    """Rebuild only an explicitly named disposable test database."""
    if DATABASE_URL is None:
        pytest.skip("MINDBRIDGE_TEST_DATABASE_URL is not configured")
    connection = await AsyncConnection.connect(DATABASE_URL, autocommit=True)
    async with connection:
        row = await (await connection.execute("SELECT current_database()")).fetchone()
        database_name = cast(tuple[str], row)[0]
        if not database_name.endswith("_test"):
            raise RuntimeError("integration database name must end with _test")
        await connection.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public", prepare=False)
        migration = Path(__file__).parents[2] / "migrations" / "0001_initial.sql"
        await connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
    yield DATABASE_URL


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def store(database_url: str) -> AsyncIterator[PostgresMemoryStore]:
    """Open the production adapter once for the integration suite."""
    postgres_store = PostgresMemoryStore(database_url, max_pool_size=4)
    await postgres_store.open()
    yield postgres_store
    await postgres_store.close()
