"""PostgreSQL pgvector persistence and cosine retrieval."""

from typing import cast

from pgvector import Vector

from mindbridge.application.ports import EmbeddingMatch, EmbeddingSearch
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingRecord,
    MemoryIntegrityError,
)
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

CLOUD_EMBEDDING_DIMENSION = 1_024
_RETRY_MINIMUM_COSINE_SIMILARITY = 0.999_999


async def write_embedding(pool: DatabasePool, embedding: EmbeddingRecord) -> bool:
    """Insert one immutable model-versioned vector; return false for a retry."""
    async with tenant_connection(pool, embedding.tenant_id) as connection:
        return await write_embedding_on_connection(connection, embedding)


async def write_embedding_on_connection(
    connection: DatabaseConnection,
    embedding: EmbeddingRecord,
) -> bool:
    """Write a vector inside a caller-owned transaction."""
    _require_cloud_dimension(embedding.dimension)
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
    if existing_id == embedding.embedding_id and normalized == embedding.normalized and same_values:
        return False
    raise DomainInvariantError("embedding version already stores different vector content")


async def search_embeddings(
    pool: DatabasePool,
    search: EmbeddingSearch,
) -> tuple[EmbeddingMatch, ...]:
    """Return nearest objects only from the requested compatible space and task."""
    _require_cloud_dimension(len(search.values))
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


def _require_cloud_dimension(dimension: int) -> None:
    if dimension != CLOUD_EMBEDDING_DIMENSION:
        raise DomainInvariantError(f"cloud embedding dimension must be {CLOUD_EMBEDDING_DIMENSION}")
