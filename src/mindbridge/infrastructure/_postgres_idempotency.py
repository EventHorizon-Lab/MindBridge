"""Transactional idempotency claims shared by PostgreSQL writes."""

from typing import cast

from mindbridge.core import IdempotencyConflictError, MemoryIntegrityError, TenantId
from mindbridge.infrastructure._postgres_types import DatabaseConnection


async def claim_idempotency_key(
    connection: DatabaseConnection,
    *,
    tenant_id: TenantId,
    operation: str,
    idempotency_key: str,
    content_digest: str,
    resource_id: str,
) -> str | None:
    """Claim a key, or return its existing resource after digest verification."""
    cursor = await connection.execute(
        """
        INSERT INTO idempotency_keys (
            tenant_id, operation, idempotency_key, content_digest, resource_id
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING resource_id
        """,
        (tenant_id, operation, idempotency_key, content_digest, resource_id),
    )
    if await cursor.fetchone() is not None:
        return None
    cursor = await connection.execute(
        """
        SELECT content_digest, resource_id
        FROM idempotency_keys
        WHERE tenant_id = %s AND operation = %s AND idempotency_key = %s
        """,
        (tenant_id, operation, idempotency_key),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("idempotency key disappeared during transaction")
    stored_digest, stored_resource_id = cast(tuple[str, str], row)
    if stored_digest != content_digest:
        raise IdempotencyConflictError("idempotency key already stores different content")
    return stored_resource_id
