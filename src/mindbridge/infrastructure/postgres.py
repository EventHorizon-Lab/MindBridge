"""Psycopg adapter for the PostgreSQL system of record."""

from __future__ import annotations

from collections.abc import Iterable

from pgvector.psycopg import register_vector_async

from mindbridge.application.episodes import EpisodeWrite
from mindbridge.application.observation_processing import (
    ObservationProcessingOutput,
)
from mindbridge.application.ports import (
    EmbeddingMatch,
    EmbeddingSearch,
)
from mindbridge.application.semantic_claims import (
    ClaimConsolidationCommit,
    ClaimConsolidationWrite,
)
from mindbridge.application.summary_consolidation import (
    SummaryWrite,
)
from mindbridge.core import (
    DEFAULT_EMBEDDING_DIMENSION,
    DeletionTombstone,
    DomainInvariantError,
    EmbeddingRecord,
    JobId,
    MemoryIntegrityError,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
    TombstoneId,
)
from mindbridge.infrastructure._postgres_claim_consolidation import ClaimCandidateOperations
from mindbridge.infrastructure._postgres_claim_writes import commit_claim_consolidation
from mindbridge.infrastructure._postgres_consolidation import (
    EpisodeCandidateOperations,
    commit_episode_consolidation,
)
from mindbridge.infrastructure._postgres_embeddings import (
    EmbeddingReadOperations,
    read_embedding_column_dimension,
    search_embeddings,
    write_embedding,
)
from mindbridge.infrastructure._postgres_evidence import EvidenceReadOperations
from mindbridge.infrastructure._postgres_feedback import FeedbackOperations
from mindbridge.infrastructure._postgres_forget import (
    ForgetOperations,
    read_deletion_tombstone,
)
from mindbridge.infrastructure._postgres_jobs import ObservationJobOperations
from mindbridge.infrastructure._postgres_lifecycle import LifecycleOperations
from mindbridge.infrastructure._postgres_memories import MemoryOperations
from mindbridge.infrastructure._postgres_observation_reads import ObservationReadOperations
from mindbridge.infrastructure._postgres_observations import ObservationWriteOperations
from mindbridge.infrastructure._postgres_processing import commit_observation_processing
from mindbridge.infrastructure._postgres_summary_consolidation import SummaryCandidateOperations
from mindbridge.infrastructure._postgres_summary_writes import commit_summary_consolidation
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    translate_transient_database_errors,
)

# One recall alone peaks near ten pooled connections: a sparse search runs concurrently
# with three vector searches, then four memory searches, and a reflection round runs
# several such waves at once. The previous ceiling of ten therefore let a single recall
# occupy the whole pool while a second queued behind it. `min_size` stays 1, so a higher
# ceiling costs nothing until load asks for it; deployments whose PostgreSQL has a lower
# `max_connections` lower this through MINDBRIDGE_DATABASE_MAX_POOL_SIZE.
DEFAULT_DATABASE_MAX_POOL_SIZE = 32


class PostgresMemoryStore(
    ObservationWriteOperations,
    ObservationReadOperations,
    ObservationJobOperations,
    MemoryOperations,
    LifecycleOperations,
    FeedbackOperations,
    ForgetOperations,
    EmbeddingReadOperations,
    EvidenceReadOperations,
    EpisodeCandidateOperations,
    ClaimCandidateOperations,
    SummaryCandidateOperations,
):
    """Transactional PostgreSQL implementation of the memory store boundary."""

    def __init__(
        self,
        database_url: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = DEFAULT_DATABASE_MAX_POOL_SIZE,
        statement_timeout_ms: int = 30_000,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        if min_pool_size < 0 or max_pool_size < max(1, min_pool_size):
            raise ValueError("pool sizes must satisfy 0 <= min_pool_size <= max_pool_size")
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be positive")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        self._embedding_dimension = embedding_dimension
        self._schema_refusal: str | None = None
        self._pool = DatabasePool(
            database_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            open=False,
            kwargs={
                "options": f"-c timezone=UTC -c statement_timeout={statement_timeout_ms}",
            },
            configure=_configure_connection,
        )

    async def open(self) -> None:
        """Open the pool and refuse a schema that cannot hold configured vectors."""
        if self._schema_refusal is not None:
            # Repeat the actionable message; the pool is already closed.
            raise MemoryIntegrityError(self._schema_refusal)
        with translate_transient_database_errors():
            await self._pool.open(wait=True)
            column_dimension = await read_embedding_column_dimension(self._pool)
        if column_dimension != self._embedding_dimension:
            self._schema_refusal = (
                f"embeddings.embedding is vector({column_dimension}) but this deployment is "
                f"configured for {self._embedding_dimension}; changing the width invalidates "
                "every stored vector, so declare a new embedding space, run "
                f"ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector"
                f"({self._embedding_dimension}) on an empty table, and re-index"
            )
            await self._pool.close()
            raise MemoryIntegrityError(self._schema_refusal)

    def _require_index_dimension(self, embeddings: Iterable[EmbeddingRecord]) -> None:
        """Reject vectors the configured pgvector index cannot store."""
        for embedding in embeddings:
            if embedding.dimension != self._embedding_dimension:
                raise DomainInvariantError(
                    f"embedding dimension must be {self._embedding_dimension} "
                    "to match the configured index"
                )

    async def close(self) -> None:
        """Close pooled connections cleanly during application shutdown."""
        await self._pool.close()

    async def commit_summary_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[SummaryWrite, ...],
    ) -> int:
        """Atomically persist disjoint Summary parents and their aligned vectors."""
        self._require_index_dimension(write.embedding for write in writes)
        return await commit_summary_consolidation(self._pool, tenant_id, writes)

    async def commit_claim_consolidation(
        self,
        tenant_id: TenantId,
        write: ClaimConsolidationWrite,
    ) -> ClaimConsolidationCommit:
        """Atomically persist verified Semantic Claims and version decisions."""
        self._require_index_dimension(claim.embedding for claim in write.semantic_claims)
        return await commit_claim_consolidation(self._pool, tenant_id, write)

    async def commit_episode_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[EpisodeWrite, ...],
    ) -> int:
        """Atomically claim child Events and persist verified Episodes."""
        self._require_index_dimension(write.embedding for write in writes)
        return await commit_episode_consolidation(self._pool, tenant_id, writes)

    async def read_deletion_tombstone(
        self,
        tenant_id: TenantId,
        tombstone_id: str,
    ) -> DeletionTombstone:
        """Read one tenant-owned deletion propagation state."""
        return await read_deletion_tombstone(self._pool, tenant_id, TombstoneId(tombstone_id))

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool:
        """Persist one immutable vector version."""
        self._require_index_dimension((embedding,))
        return await write_embedding(self._pool, embedding)

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        """Search one explicit frozen embedding space by cosine similarity."""
        return await search_embeddings(
            self._pool, search, expected_dimension=self._embedding_dimension
        )

    async def commit_observation_processing(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        output: ObservationProcessingOutput,
    ) -> ObservationProcessingJob:
        """Atomically persist derived memory and successful job state."""
        self._require_index_dimension(output.embeddings)
        return await commit_observation_processing(
            self._pool,
            tenant_id,
            observation_id,
            job_id,
            attempt=attempt,
            output=output,
        )


async def _configure_connection(connection: DatabaseConnection) -> None:
    await register_vector_async(connection)
    await connection.execute("SET ROLE mindbridge_runtime")
    await connection.commit()
