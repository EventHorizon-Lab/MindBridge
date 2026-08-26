"""PostgreSQL pgvector persistence and cosine retrieval."""

from typing import cast

from pgvector import Vector

from mindbridge.application.ports import EmbeddingMatch, EmbeddingSearch
from mindbridge.core import (
    EMBEDDING_ID_RECIPE_VERSION,
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


async def write_embedding(
    pool: DatabasePool,
    embedding: EmbeddingRecord,
    *,
    allow_reencoding: bool = False,
) -> bool:
    """Insert one immutable model-versioned vector; return false for a retry."""
    async with tenant_connection(pool, embedding.tenant_id) as connection:
        return await write_embedding_on_connection(
            connection, embedding, allow_reencoding=allow_reencoding
        )


async def write_embedding_on_connection(
    connection: DatabaseConnection,
    embedding: EmbeddingRecord,
    *,
    allow_reencoding: bool = False,
) -> bool:
    """Write a vector inside a caller-owned transaction.

    `allow_reencoding` is the caller asserting that this `embedding_id` can only ever
    encode one text, so a stored vector that differs is encoder noise rather than
    content drift. Only a caller can know that: the guarantee comes from how the ID was
    derived, which this layer cannot see. It defaults to false, so a caller that has not
    thought about it still gets the strict comparison.
    """
    vector = Vector(list(embedding.values))
    cursor = await connection.execute(
        """
        INSERT INTO embeddings (
            tenant_id, embedding_id, object_type, object_id,
            model_id, space_id, task,
            dimension, normalized, embedding, created_at, embedding_id_recipe
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING embedding_id
        """,
        (
            embedding.tenant_id,
            embedding.embedding_id,
            embedding.object_type.value,
            embedding.object_id,
            embedding.model_reference.model_id,
            embedding.space_reference.space_id,
            embedding.task,
            embedding.dimension,
            embedding.normalized,
            vector,
            embedding.created_at,
            EMBEDDING_ID_RECIPE_VERSION,
        ),
    )
    if await cursor.fetchone() is not None:
        return True
    cursor = await connection.execute(
        """
        SELECT embedding_id,
               normalized,
               1 - (embedding <=> %s) >= %s,
               embedding_id_recipe
        FROM embeddings
        WHERE tenant_id = %s
          AND object_type = %s
          AND object_id = %s
          AND model_id = %s
          AND space_id = %s
          AND task = %s
        """,
        (
            vector,
            _RETRY_MINIMUM_COSINE_SIMILARITY,
            embedding.tenant_id,
            embedding.object_type.value,
            embedding.object_id,
            embedding.model_reference.model_id,
            embedding.space_reference.space_id,
            embedding.task,
        ),
    )
    row = await cursor.fetchone()
    if row is None:
        raise MemoryIntegrityError("embedding conflict could not be resolved")
    existing_id, normalized, same_values, stored_recipe = cast(tuple[str, bool, bool, int], row)
    # Two independent reasons a differing vector can be legitimate, each known at a
    # different layer. An entity's embedding_id hashes the casefolded name the vector
    # encodes, so any stored vector for that ID encodes the same text and re-encoding it is
    # expected: a later observation that mentions the entity again batches the name with
    # different neighbouring texts, and that padding alone moves the vector well past this
    # comparison's tolerance -- that is a property of the object type, true everywhere.
    # `allow_reencoding` is the same guarantee arrived at per call site, for IDs whose text
    # is pinned upstream. Anything else re-encoding into a different vector is content drift.
    if normalized != embedding.normalized or not (
        same_values or allow_reencoding or embedding.object_type is EmbeddedObjectType.ENTITY
    ):
        raise DomainInvariantError("embedding version already stores different vector content")
    if existing_id != embedding.embedding_id:
        await _adopt_embedding_id(connection, embedding, existing_id, stored_recipe=stored_recipe)
    return False


async def _adopt_embedding_id(
    connection: DatabaseConnection,
    embedding: EmbeddingRecord,
    existing_id: str,
    *,
    stored_recipe: int,
) -> None:
    """Re-key a stored vector whose ID an older `derive_embedding_id` recipe produced.

    `embedding_id` is content-addressed, so changing what it hashes makes every ID already
    written unreachable from the same inputs -- while the object key this row was just matched
    on, tenant and object and model and space and task, is unchanged. A row carrying an ID an
    older recipe produced, with equivalent content, is not drift; it is that row still under
    its old name.

    Re-keying rather than tolerating the mismatch is what makes it stop happening: the ID is
    what `has_embedding` looks up, so a row left under its old name fails the `skip_existing`
    check on every future pass and pays for an encode it already has. Healing one row the
    first time anything touches it is cheaper than rewriting the column for every tenant in a
    migration, and does not require reproducing a Python digest in SQL.

    `stored_recipe` is what keeps this from becoming a general amnesty, and it is a fact the
    writer recorded rather than one inferred here. Two things were wrong with inferring it from
    `created_at < applied_at(0021)`: that timestamp is supplied by the caller -- `kernel.py`
    passes the memory's own creation time, not the write's -- so a replayed or backfilled record
    could claim the amnesty and have a real content disagreement silently re-keyed; and a bound
    naming one migration cannot survive a second recipe change without another number beside it.
    A row already at the current recipe is refused exactly as it always was.
    """
    if stored_recipe >= EMBEDDING_ID_RECIPE_VERSION:
        raise DomainInvariantError("embedding version already stores different vector content")
    cursor = await connection.execute(
        """
        UPDATE embeddings
        SET embedding_id = %s, embedding_id_recipe = %s
        WHERE tenant_id = %s AND embedding_id = %s
          AND embedding_id_recipe < %s
          AND NOT EXISTS (
                  SELECT 1 FROM embeddings AS claimed
                  WHERE claimed.tenant_id = %s AND claimed.embedding_id = %s
              )
        RETURNING embedding_id
        """,
        (
            embedding.embedding_id,
            EMBEDDING_ID_RECIPE_VERSION,
            embedding.tenant_id,
            existing_id,
            EMBEDDING_ID_RECIPE_VERSION,
            embedding.tenant_id,
            embedding.embedding_id,
        ),
    )
    if await cursor.fetchone() is None:
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
        # Inert while no HNSW index exists: migration 0018 dropped the one this deployment
        # shipped with, because RLS makes every query tenant-selective and the planner
        # always reaches a tenant's vectors through embeddings_space_search_idx and sorts
        # them exactly. Kept because it is what makes a filtered approximate search return
        # a full LIMIT instead of silently fewer rows, so a deployment that adds the index
        # back for one very large tenant is already correct.
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
                            AND space_id = %s
                      )
                ORDER BY candidate.object_type
                """,
                (
                    [object_type.value for object_type in EmbeddedObjectType],
                    tenant_id,
                    tenant_id,
                    space_reference.space_id,
                ),
            )
            return tuple(
                [
                    EmbeddedObjectType(object_type)
                    async for row in cursor
                    for object_type in (cast(tuple[str], row)[0],)
                ]
            )
