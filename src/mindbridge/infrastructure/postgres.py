"""Psycopg adapter for the PostgreSQL system of record."""

from __future__ import annotations

from pgvector.psycopg import register_vector_async

from mindbridge.application import (
    EmbeddingMatch,
    EmbeddingSearch,
    MemoryWriteResult,
    ObservationBatch,
    ObservationWriteResult,
)
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    EmbeddingRecord,
    EvidenceId,
    EvidenceSpan,
    JobId,
    MediaObject,
    MediaObjectId,
    MemoryRecord,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
)
from mindbridge.infrastructure._postgres_embeddings import (
    search_embeddings,
    write_embedding,
)
from mindbridge.infrastructure._postgres_jobs import (
    claim_observation_processing_job,
    mark_observation_processing_failed,
    mark_observation_processing_succeeded,
)
from mindbridge.infrastructure._postgres_memories import (
    read_evidence,
    search_memories,
    write_memory,
)
from mindbridge.infrastructure._postgres_observations import read_media_objects, write_observation
from mindbridge.infrastructure._postgres_types import DatabaseConnection, DatabasePool


class PostgresMemoryStore:
    """Transactional PostgreSQL implementation of the memory store boundary."""

    def __init__(
        self,
        database_url: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        statement_timeout_ms: int = 30_000,
    ) -> None:
        if min_pool_size < 0 or max_pool_size < max(1, min_pool_size):
            raise ValueError("pool sizes must satisfy 0 <= min_pool_size <= max_pool_size")
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be positive")
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
        """Open the pool explicitly, as required by current Psycopg."""
        await self._pool.open(wait=True)

    async def close(self) -> None:
        """Close pooled connections cleanly during application shutdown."""
        await self._pool.close()

    async def write_observation(
        self,
        batch: ObservationBatch,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ObservationWriteResult:
        """Atomically persist media, observation, and exact evidence spans."""
        return await write_observation(
            self._pool,
            batch,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )

    async def write_memory(
        self,
        memory: MemoryRecord,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> MemoryWriteResult:
        """Persist one explicit memory and its evidence links atomically."""
        return await write_memory(
            self._pool,
            memory,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )

    async def search_memories(self, request: RecallRequest) -> tuple[MemoryRecord, ...]:
        """Apply exact filters and PostgreSQL full-text candidate retrieval."""
        return await search_memories(self._pool, request)

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        """Read exact evidence spans while preserving caller order."""
        return await read_evidence(self._pool, tenant_id, evidence_ids)

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        """Read media metadata for evidence resolution in caller order."""
        return await read_media_objects(self._pool, tenant_id, media_object_ids)

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool:
        """Persist one immutable vector version."""
        return await write_embedding(self._pool, embedding)

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]:
        """Search one explicit frozen embedding space by cosine similarity."""
        return await search_embeddings(self._pool, search)

    async def claim_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationJobClaim:
        """Claim one ready job or report its current durable state."""
        return await claim_observation_processing_job(self._pool, tenant_id, observation_id, job_id)

    async def mark_observation_processing_succeeded(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
    ) -> ObservationProcessingJob:
        """Commit successful observation processing state."""
        return await mark_observation_processing_succeeded(
            self._pool, tenant_id, observation_id, job_id, attempt=attempt
        )

    async def mark_observation_processing_failed(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        error_code: str,
    ) -> ObservationProcessingJob:
        """Record a sanitized failure that remains eligible for retry."""
        return await mark_observation_processing_failed(
            self._pool,
            tenant_id,
            observation_id,
            job_id,
            attempt=attempt,
            error_code=error_code,
        )


async def _configure_connection(connection: DatabaseConnection) -> None:
    await register_vector_async(connection)
    await connection.commit()
