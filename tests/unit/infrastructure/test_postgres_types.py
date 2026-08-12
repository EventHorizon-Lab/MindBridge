"""Checks for PostgreSQL tenant context setup."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from psycopg import OperationalError
from psycopg.errors import ConnectionFailure, InvalidPassword, QueryCanceled, SerializationFailure
from psycopg_pool import PoolTimeout

from mindbridge.core import DatabaseUnavailableError
from mindbridge.infrastructure._postgres_types import DatabasePool, tenant_connection


async def test_tenant_connection_sets_transaction_local_rls_context() -> None:
    connection = AsyncMock()
    checkout = MagicMock()
    checkout.__aenter__ = AsyncMock(return_value=connection)
    checkout.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = checkout

    async with tenant_connection(cast(DatabasePool, pool), "tenant_01") as selected:
        assert selected is connection

    connection.execute.assert_awaited_once_with(
        "SELECT set_config('mindbridge.tenant_id', %s, true)",
        ("tenant_01",),
    )


async def test_tenant_connection_rejects_an_empty_identity() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        async with tenant_connection(cast(DatabasePool, MagicMock()), " "):
            pass


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("socket secret"),
        ConnectionFailure("socket secret"),
        PoolTimeout("secret"),
        QueryCanceled("statement timeout secret"),
        SerializationFailure("secret"),
    ],
)
async def test_tenant_connection_sanitizes_transient_failures(error: OperationalError) -> None:
    checkout = MagicMock()
    checkout.__aenter__ = AsyncMock(side_effect=error)
    pool = MagicMock()
    pool.connection.return_value = checkout

    with pytest.raises(DatabaseUnavailableError, match="temporarily") as raised:
        async with tenant_connection(cast(DatabasePool, pool), "tenant_01"):
            pass

    assert "secret" not in str(raised.value)


async def test_tenant_connection_does_not_retry_permanent_errors() -> None:
    error = InvalidPassword("permanent secret")
    checkout = MagicMock()
    checkout.__aenter__ = AsyncMock(side_effect=error)
    pool = MagicMock()
    pool.connection.return_value = checkout

    with pytest.raises(type(error), match="permanent secret"):
        async with tenant_connection(cast(DatabasePool, pool), "tenant_01"):
            pass
