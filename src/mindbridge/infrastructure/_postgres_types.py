"""Concrete Psycopg connection types shared by PostgreSQL adapters."""

from typing import TypeAlias

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

DatabaseConnection: TypeAlias = AsyncConnection[TupleRow]
DatabasePool: TypeAlias = AsyncConnectionPool[DatabaseConnection]
