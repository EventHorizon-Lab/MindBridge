"""Checks for PostgreSQL tenant context setup."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

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
