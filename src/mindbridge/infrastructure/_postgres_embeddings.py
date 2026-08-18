"""PostgreSQL pgvector persistence and cosine retrieval."""

from typing import cast

from pgvector import Vector

from mindbridge.application.ports import EmbeddingMatch, EmbeddingSearch
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    MemoryIntegrityError,
    TenantId,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    PostgresStoreOperations,
    tenant_connection,
)

# Permit ~1e-4 normalized GPU jitter without masking model or input changes.
_RETRY_MINIMUM_COSINE_SIMILARITY = 0.999_999


async def read_embedding_column_dimension(pool: DatabasePool) -> int:
    """Return the width the pgvector column actually enforces."""
    async with pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'embeddings'::regclass AND attname = 'embedding'
            """
        )
        row = await cursor.fetchone()
    if row is None or not str(row[0]).startswith("vector("):
        raise MemoryIntegrityError("embeddings.embedding is not a dimensioned pgvector column")
    return int(str(row[0]).removeprefix("vector(").removesuffix(")"))


async def write_embedding(pool: DatabasePool, embedding: EmbeddingRecord) -> bool:
    """Insert one immutable model-versioned vector; return false for a retry."""
    async with tenant_connection(pool, embedding.tenant_id) as connection:
        return await write_embedding_on_connection(connection, embedding)


async def write_embedding_on_connection(
    connection: DatabaseConnection,
    embedding: EmbeddingRecord,
) -> bool:
    """Write a vector inside a caller-owned transaction."""
    vector = Vector(list(embedding.values))
    cursor = await connection.execute(
        """
        INSERT INTO embeddings (
            tenant_id, embedding_id, object_type, object_id,
            model_id, model_revision, space_id, space_revision, task,
            dimension, normalized, embedding, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING embedding_id
        """,
        (
            embedding.tenant_id,
            embedding.embedding_id,
            embedding.object_type.value,
            embedding.object_id,
            embedding.model_reference.model_id,
            embedding.model_reference.revision,
            embedding.space_reference.space_id,
            embedding.space_reference.revision,
            embedding.task,
            embedding.dimension,
            embedding.normalized,
            vector,
            embedding.created_at,
        ),
    )
    if await cursor.fetchone() is not None:
        return True
    cursor = await connection.execute(
        """
        SELECT embedding_id,
               normalized,
               1 - (embedding <=> %s) >= %s
        FROM embeddings
        WHERE tenant_id = %s
          AND object_type = %s
          AND object_id = %s
          AND model_id = %s
          AND model_revision = %s
          AND space_id = %s
          AND space_revision = %s
          AND task = %s
        """,
        (
            vector,
            _RETRY_MINIMUM_COSINE_SIMILARITY,
            embedding.tenant_id,
            embedding.object_type.value,
            embedding.object_id,
            embedding.model_reference.model_id,
            embedding.model_reference.revision,
            embedding.space_reference.space_id,
            embedding.space_reference.revision,
            embedding.task,
        ),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("embedding conflict could not be resolved")
    existing_id, normalized, same_values = cast(tuple[str, bool, bool], row)
    # An entity's embedding_id hashes the casefolded name the vector encodes, so any stored
    # vector for that ID encodes the same text and re-encoding it is expected: a later
    # observation that mentions the entity again batches the name with different neighbouring
    # texts, and that padding alone moves the vector well past this comparison's tolerance.
    # Every other object type is only ever re-encoded by replaying one identical batch.
    if (
        existing_id == embedding.embedding_id
        and normalized == embedding.normalized
        and (same_values or embedding.object_type is EmbeddedObjectType.ENTITY)
    ):
        return False
    raise DomainInvariantError("embedding version already stores different vector content")


async def search_embeddings(
    pool: DatabasePool,
    search: EmbeddingSearch,
    *,
    expected_dimension: int,
) -> tuple[EmbeddingMatch, ...]:
    """Return nearest objects only from the requested compatible space and task."""
    if len(search.values) != expected_dimension:
        raise DomainInvariantError(
            f"query vector must have {expected_dimension} values to match the index"
        )
    vector = Vector(list(search.values))
    async with tenant_connection(pool, search.tenant_id) as connection:
        await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        cursor = await connection.execute(
            """
            SELECT embedding_id,
                   object_type,
                   object_id,
                   1 - (embedding <=> %s) AS similarity
            FROM embeddings
            WHERE tenant_id = %s
              AND space_id = %s
              AND space_revision = %s
              AND task = %s
              AND object_type = ANY(%s)
              AND 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s, embedding_id
            LIMIT %s
            """,
            (
                vector,
                search.tenant_id,
                search.space_reference.space_id,
                search.space_reference.revision,
                search.document_task,
                [object_type.value for object_type in search.object_types],
                vector,
                search.minimum_similarity,
                vector,
                search.limit,
            ),
        )
        return tuple(
            [
                EmbeddingMatch(
                    embedding_id=embedding_id,
                    object_type=EmbeddedObjectType(object_type),
                    object_id=object_id,
                    similarity=similarity,
                )
                async for row in cursor
                for embedding_id, object_type, object_id, similarity in (
                    (cast(tuple[str, str, str, float], row)),
                )
            ]
        )


class EmbeddingReadOperations(PostgresStoreOperations):
    async def has_embedding(
        self,
        tenant_id: TenantId,
        embedding_id: EmbeddingId,
    ) -> bool:
        """Return whether one tenant already has the immutable vector."""
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                "SELECT 1 FROM embeddings WHERE tenant_id = %s AND embedding_id = %s",
                (tenant_id, embedding_id),
            )
            return await cursor.fetchone() is not None

    async def unreachable_embedded_object_types(
        self,
        tenant_id: TenantId,
        space_reference: EmbeddingSpaceReference,
    ) -> tuple[EmbeddedObjectType, ...]:
        """Return the object types a tenant stored that no query in this space can match.

        Row-level security hides other tenants, so this is asked per tenant. Holding vectors
        in several spaces is legitimate while re-embedding; holding none in the configured
        space is not, because recall then returns empty results instead of failing. Writers
        do not share object types — the Worker owns evidence, events, and claims while the
        server owns memory records — so this is asked per object type. A whole-tenant probe
        would let the server's own memory records mask everything the Worker stranded.
        """
        async with tenant_connection(self._pool, tenant_id) as connection:
            cursor = await connection.execute(
                """
                SELECT candidate.object_type
                FROM unnest(%s::text[]) AS candidate(object_type)
                WHERE EXISTS (
                          SELECT 1 FROM embeddings
                          WHERE tenant_id = %s AND object_type = candidate.object_type
                      )
                  AND NOT EXISTS (
                          SELECT 1 FROM embeddings
                          WHERE tenant_id = %s AND object_type = candidate.object_type
                            AND space_id = %s AND space_revision = %s
                      )
                ORDER BY candidate.object_type
                """,
                (
                    [object_type.value for object_type in EmbeddedObjectType],
                    tenant_id,
                    tenant_id,
                    space_reference.space_id,
                    space_reference.revision,
                ),
            )
            return tuple(
                [
                    EmbeddedObjectType(object_type)
                    async for row in cursor
                    for object_type in (cast(tuple[str], row)[0],)
                ]
            )
