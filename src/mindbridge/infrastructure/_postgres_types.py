"""Concrete Psycopg connection types and tenant-scoped checkout."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeAlias

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

DatabaseConnection: TypeAlias = AsyncConnection[TupleRow]
DatabasePool: TypeAlias = AsyncConnectionPool[DatabaseConnection]


@asynccontextmanager
async def tenant_connection(
    pool: DatabasePool,
    tenant_id: str,
) -> AsyncIterator[DatabaseConnection]:
    """Set the transaction-local tenant consumed by PostgreSQL RLS policies."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    async with pool.connection() as connection:
        await connection.execute(
            "SELECT set_config('mindbridge.tenant_id', %s, true)",
            (tenant_id,),
        )
        yield connection
