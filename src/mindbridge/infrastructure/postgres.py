"""Psycopg adapter for the PostgreSQL system of record."""

from __future__ import annotations

from datetime import datetime

from pgvector.psycopg import register_vector_async

from mindbridge.application import (
    ClaimCandidatePage,
    ClaimCandidateRequest,
    ClaimConsolidationCommit,
    ClaimConsolidationWrite,
    EmbeddingMatch,
    EmbeddingSearch,
    EpisodeCandidatePage,
    EpisodeCandidateRequest,
    EpisodeWrite,
    FeedbackWriteResult,
    ForgetPlan,
    MemoryLifecycleChange,
    MemoryWriteResult,
    ObservationBatch,
    ObservationProcessingOutput,
    ObservationWriteResult,
    SummaryCandidatePage,
    SummaryCandidateRequest,
)
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DeletionTombstone,
    EmbeddingRecord,
    EvidenceId,
    EvidenceSpan,
    JobId,
    MediaObject,
    MediaObjectId,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
    TombstoneId,
)
from mindbridge.infrastructure._postgres_claim_consolidation import list_claim_candidates
from mindbridge.infrastructure._postgres_claim_writes import commit_claim_consolidation
from mindbridge.infrastructure._postgres_consolidation import (
    commit_episode_consolidation,
    list_episode_candidates,
)
from mindbridge.infrastructure._postgres_embeddings import (
    search_embeddings,
    write_embedding,
)
from mindbridge.infrastructure._postgres_evidence import read_evidence
from mindbridge.infrastructure._postgres_feedback import record_feedback
from mindbridge.infrastructure._postgres_forget import (
    complete_forget,
    list_deletion_tombstones,
    mark_forget_failed,
    prepare_forget,
    read_deletion_tombstone,
)
from mindbridge.infrastructure._postgres_jobs import (
    claim_observation_processing_job,
    mark_observation_processing_failed,
    mark_observation_processing_succeeded,
    read_observation_processing_job,
)
from mindbridge.infrastructure._postgres_lifecycle import (
    list_memories_for_lifecycle,
    record_memory_accesses,
    update_memory_lifecycles,
)
from mindbridge.infrastructure._postgres_memories import (
    list_memories_for_enumeration,
    read_memory,
    search_memories,
    search_memories_by_evidence,
    search_memories_by_graph_objects,
    search_memories_by_ids,
    write_memory,
)
from mindbridge.infrastructure._postgres_observation_reads import (
    read_media_objects,
    read_observation_batch,
)
from mindbridge.infrastructure._postgres_observations import write_observation
from mindbridge.infrastructure._postgres_processing import commit_observation_processing
from mindbridge.infrastructure._postgres_summary_consolidation import list_summary_candidates
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

    async def list_memories_for_lifecycle(
        self,
        tenant_id: TenantId,
        *,
        after_memory_id: MemoryId | None,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Read one stable page eligible for automatic state evolution."""
        return await list_memories_for_lifecycle(
            self._pool,
            tenant_id,
            after_memory_id=after_memory_id,
            limit=limit,
        )

    async def list_episode_candidates(
        self,
        request: EpisodeCandidateRequest,
    ) -> EpisodeCandidatePage:
        """Discover a stable bounded page for Omni episode verification."""
        return await list_episode_candidates(self._pool, request)

    async def list_claim_candidates(
        self,
        request: ClaimCandidateRequest,
    ) -> ClaimCandidatePage:
        """Discover a stable bounded page for semantic Claim verification."""
        return await list_claim_candidates(self._pool, request)

    async def list_summary_candidates(
        self,
        request: SummaryCandidateRequest,
    ) -> SummaryCandidatePage:
        """Discover a stable bounded page for hierarchical Summary verification."""
        return await list_summary_candidates(self._pool, request)

    async def commit_claim_consolidation(
        self,
        tenant_id: TenantId,
        write: ClaimConsolidationWrite,
    ) -> ClaimConsolidationCommit:
        """Atomically persist verified Semantic Claims and version decisions."""
        return await commit_claim_consolidation(self._pool, tenant_id, write)

    async def commit_episode_consolidation(
        self,
        tenant_id: TenantId,
        writes: tuple[EpisodeWrite, ...],
    ) -> int:
        """Atomically claim child Events and persist verified Episodes."""
        return await commit_episode_consolidation(self._pool, tenant_id, writes)

    async def update_memory_lifecycles(
        self,
        changes: tuple[MemoryLifecycleChange, ...],
    ) -> int:
        """Optimistically persist automatic score and state changes."""
        return await update_memory_lifecycles(self._pool, changes)

    async def read_memory(
        self,
        tenant_id: TenantId,
        memory_id: MemoryId,
    ) -> MemoryRecord:
        """Read one tenant-owned memory or raise the stable not-found error."""
        return await read_memory(self._pool, tenant_id, memory_id)

    async def record_feedback(
        self,
        feedback: MemoryFeedback,
        corrected_memory: MemoryRecord | None,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> FeedbackWriteResult:
        """Atomically retain feedback and evolve its target memory."""
        return await record_feedback(
            self._pool,
            feedback,
            corrected_memory,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )

    async def prepare_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ForgetPlan:
        """Persist the deletion barrier and return external media work."""
        return await prepare_forget(
            self._pool,
            tombstone,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )

    async def complete_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        completed_at: datetime,
    ) -> DeletionTombstone:
        """Erase database derivatives after external media deletion."""
        return await complete_forget(self._pool, tombstone, completed_at=completed_at)

    async def mark_forget_failed(
        self,
        tombstone: DeletionTombstone,
        *,
        error_code: str,
    ) -> DeletionTombstone:
        """Keep a recoverable sanitized deletion failure."""
        return await mark_forget_failed(self._pool, tombstone, error_code=error_code)

    async def read_deletion_tombstone(
        self,
        tenant_id: TenantId,
        tombstone_id: str,
    ) -> DeletionTombstone:
        """Read one tenant-owned deletion propagation state."""
        return await read_deletion_tombstone(self._pool, tenant_id, TombstoneId(tombstone_id))

    async def list_deletion_tombstones(
        self,
        tenant_id: TenantId,
        *,
        after_tombstone_id: str | None,
        limit: int,
    ) -> tuple[DeletionTombstone, ...]:
        """List deletion barriers in stable propagation order."""
        return await list_deletion_tombstones(
            self._pool,
            tenant_id,
            after_tombstone_id=after_tombstone_id,
            limit=limit,
        )

    async def search_memories(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Apply exact filters and PostgreSQL full-text candidate retrieval."""
        return await search_memories(self._pool, request, limit=limit)

    async def list_memories_for_enumeration(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Scan an exact structured-filter scope in chronological order."""
        return await list_memories_for_enumeration(self._pool, request, limit=limit)

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Map ranked semantic evidence hits to filtered memory candidates."""
        return await search_memories_by_evidence(
            self._pool,
            request,
            ranked_evidence_ids,
            limit=limit,
        )

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Resolve ranked memory-vector hits with exact recall filters."""
        return await search_memories_by_ids(
            self._pool,
            request,
            ranked_memory_ids,
            limit=limit,
        )

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        """Follow ranked Event/Claim representation edges with exact filters."""
        return await search_memories_by_graph_objects(
            self._pool,
            request,
            ranked_objects,
            limit=limit,
        )

    async def record_memory_accesses(
        self,
        tenant_id: TenantId,
        memory_ids: tuple[MemoryId, ...],
        *,
        accessed_at: datetime,
    ) -> tuple[MemoryRecord, ...]:
        """Record selected recall results and return their current rows in input order."""
        return await record_memory_accesses(
            self._pool,
            tenant_id,
            memory_ids,
            accessed_at=accessed_at,
        )

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

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        """Read one tenant-owned observation processing state."""
        return await read_observation_processing_job(self._pool, tenant_id, job_id)

    async def read_observation_batch(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
    ) -> ObservationBatch:
        """Read one immutable observation with its complete source evidence."""
        return await read_observation_batch(self._pool, tenant_id, observation_id)

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
