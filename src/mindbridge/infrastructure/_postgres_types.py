"""Concrete Psycopg connection types and tenant-scoped checkout."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TypeAlias

from psycopg import AsyncConnection, OperationalError
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from mindbridge.core import DatabaseUnavailableError

DatabaseConnection: TypeAlias = AsyncConnection[TupleRow]
DatabasePool: TypeAlias = AsyncConnectionPool[DatabaseConnection]
_TRANSIENT_SQLSTATES = frozenset(
    {"40000", "40001", "40003", "40P01", "53300", "57P01", "57P02", "57P03"}
)


@asynccontextmanager
async def tenant_connection(
    pool: DatabasePool,
    tenant_id: str,
) -> AsyncIterator[DatabaseConnection]:
    """Set the transaction-local tenant consumed by PostgreSQL RLS policies."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with translate_transient_database_errors():
        async with pool.connection() as connection:
            await connection.execute(
                "SELECT set_config('mindbridge.tenant_id', %s, true)",
                (tenant_id,),
            )
            yield connection


@contextmanager
def translate_transient_database_errors() -> Iterator[None]:
    """Hide and classify only database failures safe for whole-operation retry."""
    try:
        yield
    except OperationalError as error:
        sqlstate = error.sqlstate
        if sqlstate is not None and not (
            sqlstate.startswith("08") or sqlstate in _TRANSIENT_SQLSTATES
        ):
            raise
        raise DatabaseUnavailableError("database is temporarily unavailable") from error
