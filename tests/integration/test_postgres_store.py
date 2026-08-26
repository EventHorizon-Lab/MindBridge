"""Integration checks for migrations and the real PostgreSQL adapter."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation

from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.observation_processing import (
    ObservationBatch,
    ObservationProcessingOutput,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    GeneratedAnswer,
    PresignedMediaDownload,
    ResolvedQueryMedia,
)
from mindbridge.contracts import (
    FeedbackRequest,
    IdentityObservationInput,
    MediaObjectInput,
    MemoryResult,
    ObservationStatus,
    ObserveRequest,
    RecallFilters,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import (
    DeviceId,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    FeedbackType,
    IdempotencyConflictError,
    IdentityKind,
    JobId,
    JobNotFoundError,
    JobState,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelReference,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure._postgres_jobs import (
    OBSERVATION_JOB_STALE_AFTER_SECONDS,
    observation_job_accounting,
    tenant_scope_required,
    unreachable_observation_jobs,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.task_queue import create_task_queue
from mindbridge.jobs_cli import queue_depth, reconcile
from mindbridge.models import Embedding, EmbedRequest, EmbedResult
from mindbridge.telemetry import operation_span, set_current_span_attributes

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

pytestmark = pytest.mark.integration


class FirstCandidateAnswerer:
    """Deterministic answerer for persistence-path verification."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        return GeneratedAnswer(answer=memories[0].summary, confidence=0.9)

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        return tuple(memory.memory_id for memory in memories)


class DeterministicMediaUrlSigner:
    """Keeps database integration independent from object storage."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )

    async def delete_media(self, media_object: MediaObject) -> None:
        return None


class DiscardingObservationJobPublisher:
    """Keeps store integration independent from the Redis delivery adapter."""

    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None:
        return None


class FixedEmbedder:
    """Keeps persistence integration independent from the embedding service."""

    model_reference = ModelReference(model_id="jina-omni")
    space_reference = EmbeddingSpaceReference(space_id="jina-v5")

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        vector = (1.0,) + (0.0,) * 1_023
        embedding = Embedding(vector, self.model_reference, self.space_reference)
        return EmbedResult((embedding,) * len(request.inputs))


async def test_migration_installs_complete_phase_zero_schema(database_url: str) -> None:
    """The initial migration creates pgvector and every Phase 0 system table."""
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        extension = await (
            await connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        tables = await (
            await connection.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                """
            )
        ).fetchall()
        row_secured_tables = await (
            await connection.execute(
                """
                SELECT column_info.table_name
                FROM information_schema.columns AS column_info
                JOIN pg_class AS relation ON relation.relname = column_info.table_name
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE column_info.table_schema = 'public'
                  AND column_info.column_name = 'tenant_id'
                  AND namespace.nspname = 'public'
                  AND relation.relrowsecurity
                  AND relation.relforcerowsecurity
                """
            )
        ).fetchall()
        policy_tables = await (
            await connection.execute(
                """
                SELECT tablename FROM pg_policies
                WHERE schemaname = 'public' AND policyname = 'tenant_isolation'
                """
            )
        ).fetchall()

    assert cast(tuple[str], extension)[0] == "0.8.2"
    assert {
        "claims",
        "embeddings",
        "events",
        "evidence_clips",
        "evidence_spans",
        "media_objects",
        "memory_records",
        "observations",
    } <= {cast(tuple[str], row)[0] for row in tables}
    tenant_tables = {
        "claim_evidence",
        "claims",
        "deletion_tombstones",
        "embeddings",
        "entities",
        "entity_mentions",
        "entity_resolution_verdicts",
        "event_evidence",
        "event_observations",
        "events",
        "evidence_clips",
        "evidence_spans",
        "idempotency_keys",
        "jobs",
        "media_objects",
        "memory_evidence",
        "memory_feedback",
        "memory_records",
        "observation_media",
        "observations",
        "relations",
    }
    assert {cast(tuple[str], row)[0] for row in row_secured_tables} == tenant_tables
    assert {cast(tuple[str], row)[0] for row in policy_tables} == tenant_tables
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        versions = await (
            await connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ).fetchall()
    assert [cast(tuple[int], row)[0] for row in versions] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ]


async def test_access_time_migration_repairs_legacy_clock_rollback(database_url: str) -> None:
    migration = Path(__file__).parents[2] / "migrations/0012_memory_activity_time.sql"
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute("DROP SCHEMA IF EXISTS migration_0012_legacy CASCADE")
        await connection.execute("CREATE SCHEMA migration_0012_legacy")
        await connection.execute("SET search_path TO migration_0012_legacy")
        await connection.execute(
            """
            CREATE TABLE memory_records (
                created_at timestamptz NOT NULL,
                last_accessed_at timestamptz
            );
            CREATE TABLE schema_migrations (version integer PRIMARY KEY);
            INSERT INTO memory_records VALUES (
                '2026-08-12 12:00:00+00',
                '2026-08-12 11:59:59+00'
            );
            """,
            prepare=False,
        )
        await connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
        repaired = await (
            await connection.execute("SELECT created_at, last_accessed_at FROM memory_records")
        ).fetchone()
        await connection.execute("RESET search_path")
        await connection.execute("DROP SCHEMA migration_0012_legacy CASCADE")

    assert repaired is not None
    assert repaired[1] == repaired[0]


async def test_dropping_model_revisions_dedupes_vectors_and_rewrites_stored_identities(
    database_url: str,
) -> None:
    """The suite's own replay starts empty, so these two branches would never run.

    Both only do work on rows written before migration 0021: two vectors that differed only by
    revision now collide on the narrowed unique key, and a stored identity span carrying
    `model_revision` would fail the reader's own contract, which forbids unknown fields.
    """
    migration = Path(__file__).parents[2] / "migrations/0021_drop_model_revisions.sql"
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute("DROP SCHEMA IF EXISTS migration_0021_legacy CASCADE")
        await connection.execute("CREATE SCHEMA migration_0021_legacy")
        await connection.execute("SET search_path TO migration_0021_legacy")
        await connection.execute(
            """
            CREATE TABLE events (model_id text, model_revision text);
            CREATE TABLE claims (model_id text, model_revision text);
            CREATE TABLE memory_records (model_id text, model_revision text);
            CREATE TABLE embeddings (
                tenant_id text NOT NULL,
                embedding_id text NOT NULL,
                object_type text NOT NULL,
                object_id text NOT NULL,
                model_id text NOT NULL,
                model_revision text NOT NULL,
                space_id text NOT NULL,
                space_revision text NOT NULL,
                task text NOT NULL,
                created_at timestamptz NOT NULL,
                UNIQUE (tenant_id, object_type, object_id, model_id, model_revision, task)
            );
            CREATE INDEX embeddings_space_search_idx
                ON embeddings (tenant_id, space_id, space_revision, task, object_type);
            CREATE INDEX embeddings_object_lookup_idx
                ON embeddings (tenant_id, object_type, object_id, space_id, space_revision, task);
            CREATE TABLE observations (identity_observations jsonb NOT NULL);
            CREATE TABLE schema_migrations (version integer PRIMARY KEY);
            INSERT INTO embeddings VALUES
                ('t', 'first', 'event', 'e', 'm', 'r1', 's', 'r1', 'document', now()),
                ('t', 'second', 'event', 'e', 'm', 'r2', 's', 'r2', 'document', now() + '1s');
            INSERT INTO observations VALUES ('[{"model_id": "m", "model_revision": "r1"}]'::jsonb);
            """,
            prepare=False,
        )
        await connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
        surviving = await (
            await connection.execute("SELECT embedding_id FROM embeddings")
        ).fetchall()
        identities = await (
            await connection.execute("SELECT identity_observations FROM observations")
        ).fetchone()
        await connection.execute("RESET search_path")
        await connection.execute("DROP SCHEMA migration_0021_legacy CASCADE")

    assert [cast(tuple[str], row)[0] for row in surviving] == ["first"]
    assert cast(tuple[list[dict[str, str]]], identities)[0] == [{"model_id": "m"}]


async def test_widening_the_embedding_key_restores_space_coexistence(
    database_url: str,
) -> None:
    """Migration 0021 left `space_id` out of the key that replaced the revision-keyed one.

    The revision was what let one object hold two vectors while a re-embedding ran, which
    `docs/configuration.md` and `docs/troubleshooting.md` both document as supported. Without
    `space_id` in the key those two rows collide, so this asserts the second space is accepted
    and that the widened key still rejects a genuine duplicate within one space.
    """
    migration = Path(__file__).parents[2] / (
        "migrations/0025_embedding_space_key_and_stale_digests.sql"
    )
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute("DROP SCHEMA IF EXISTS migration_0025_legacy CASCADE")
        await connection.execute("CREATE SCHEMA migration_0025_legacy")
        await connection.execute("SET search_path TO migration_0025_legacy")
        await connection.execute(
            """
            CREATE TABLE embeddings (
                tenant_id text NOT NULL,
                embedding_id text NOT NULL,
                object_type text NOT NULL,
                object_id text NOT NULL,
                model_id text NOT NULL,
                space_id text NOT NULL,
                task text NOT NULL,
                created_at timestamptz NOT NULL,
                -- Production's primary key, without which this table cannot show what the
                -- widened unique key does and does not buy: `embedding_id` is derived, and
                -- only the recipe in `kernel.py` hashes `space_id`, so for every other object
                -- type the second vector arrives under the same ID and collides here rather
                -- than on the key this migration widens.
                PRIMARY KEY (tenant_id, embedding_id),
                CONSTRAINT embeddings_object_model_task_key
                    UNIQUE (tenant_id, object_type, object_id, model_id, task)
            );
            CREATE TABLE idempotency_keys (
                tenant_id text NOT NULL,
                operation text NOT NULL,
                idempotency_key text NOT NULL,
                content_digest text NOT NULL,
                created_at timestamptz NOT NULL
            );
            CREATE TABLE observations (
                observation_id text NOT NULL,
                identity_observations jsonb NOT NULL,
                content_digest char(64) NOT NULL,
                ingested_at timestamptz NOT NULL
            );
            CREATE TABLE schema_migrations (
                version integer PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            );
            INSERT INTO schema_migrations VALUES (21, now());
            INSERT INTO embeddings VALUES
                ('t', 'first', 'memory_record', 'm1', 'model', 'space-v1', 'document', now()),
                ('t', 'claim-hash', 'claim', 'c1', 'model', 'space-v1', 'document', now());
            INSERT INTO idempotency_keys VALUES
                ('t', 'observe', 'key-1', 'stale', now() - interval '1 day'),
                ('t', 'observe', 'key-3', 'still-valid', now() + interval '1 day'),
                ('t', 'remember', 'key-2', 'still-valid', now() - interval '1 day');
            INSERT INTO observations VALUES
                ('with_spans', '[{"model_id": "m"}]'::jsonb, repeat('a', 64),
                 now() - interval '1 day'),
                ('spans_after_0021', '[{"model_id": "m"}]'::jsonb, repeat('c', 64),
                 now() + interval '1 day'),
                ('without_spans', '[]'::jsonb, repeat('b', 64), now() - interval '1 day');
            """,
            prepare=False,
        )
        await connection.execute(migration.read_text(encoding="utf-8"), prepare=False)

        # The state the narrowed key made impossible: one object, two spaces, mid-re-embed.
        await connection.execute(
            """
            INSERT INTO embeddings VALUES
                ('t', 'second', 'memory_record', 'm1', 'model', 'space-v2', 'document', now())
            """,
            prepare=False,
        )
        spaces = await (
            await connection.execute(
                "SELECT space_id FROM embeddings WHERE object_id = 'm1' ORDER BY space_id"
            )
        ).fetchall()
        duplicate_rejected = False
        try:
            await connection.execute(
                """
                INSERT INTO embeddings VALUES
                    ('t', 'third', 'memory_record', 'm1', 'model', 'space-v2', 'document', now())
                """,
                prepare=False,
            )
        except UniqueViolation:
            duplicate_rejected = True
        # The half the widened key does not reach. A claim's `embedding_id` recipe
        # (`semantic_claims.py`) hashes tenant, object type, object id, model and task but not
        # `space_id`, so re-embedding it under a second space derives the same ID and collides
        # on the primary key instead. Asserted so the limitation cannot be mistaken for fixed.
        space_blind_recipe_still_collides = False
        try:
            await connection.execute(
                """
                INSERT INTO embeddings VALUES
                    ('t', 'claim-hash', 'claim', 'c1', 'model', 'space-v2', 'document', now())
                """,
                prepare=False,
            )
        except UniqueViolation:
            space_blind_recipe_still_collides = True
        claims = await (
            await connection.execute("SELECT operation FROM idempotency_keys")
        ).fetchall()
        digests = await (
            await connection.execute("SELECT observation_id, content_digest FROM observations")
        ).fetchall()
        await connection.execute("RESET search_path")
        await connection.execute("DROP SCHEMA migration_0025_legacy CASCADE")

    assert [cast(tuple[str], row)[0] for row in spaces] == ["space-v1", "space-v2"]
    assert duplicate_rejected
    assert space_blind_recipe_still_collides
    # An `observe` claim recorded before 0021 can never match again and is dropped; one recorded
    # after it digests what the current recipe digests, and a `remember` claim never moved.
    assert sorted(cast(tuple[str], row)[0] for row in claims) == ["observe", "remember"]
    # Only an observation that both carries identity spans -- where the removed field lived --
    # and predates 0021 loses its digest. One without spans digests exactly what it always did,
    # and one written after 0021 already carries a digest the current recipe reproduces, so
    # nulling it would open a window for a different body to be accepted as a duplicate.
    assert dict(cast(list[tuple[str, str | None]], digests)) == {
        "with_spans": None,
        "spans_after_0021": "c" * 64,
        "without_spans": "b" * 64,
    }


def test_every_migration_version_is_claimed_by_exactly_one_file() -> None:
    """Two files claiming one version merge cleanly and then break the apply, silently.

    Filenames differing anywhere but the number do not conflict in git, so two branches can
    each add `00NN_*.sql` and both survive a merge. The runner globs and sorts, so the first
    one applies and commits and the second aborts on work already done -- and the documented
    apply loop has no `set -e`, so it prints that error, continues, and exits 0. The version
    list above cannot see it, because it asserts the set of versions reached, not that each
    was reached once. This is the assertion that can.
    """
    directory = Path(__file__).parents[2] / "migrations"
    numbers = [path.name[:4] for path in sorted(directory.glob("[0-9][0-9][0-9][0-9]_*.sql"))]

    assert sorted(set(numbers)) == numbers


async def test_a_resend_digested_by_the_retired_recipe_is_accepted_once(
    database_url: str,
) -> None:
    """Migration 0021 moved the digest of a request whose bytes did not change.

    `_request_digest` hashes the whole `ObserveRequest`, so removing a field from it changed the
    digest while `Observation.idempotency_key` -- which hashes only device, boot and sequence --
    stayed stable. A device retrying an observation the server already accepted therefore failed
    the digest comparison and got a conflict forever, for a byte-identical resend. This drives
    the real store: a NULL digest is accepted once and written back, so the very next genuinely
    different body for that sequence is refused again rather than the guard being dropped.
    """
    store = PostgresMemoryStore(database_url, max_pool_size=2)
    await store.open()
    try:
        first = await store.write_observation(
            _stale_digest_batch(),
            idempotency_key="stale_digest_first",
            content_digest=f"{0xA:064x}",
        )
        assert first.created is True

        # Exactly what migration 0025 does to a row the retired recipe digested.
        connection = await AsyncConnection.connect(database_url, autocommit=True)
        async with connection:
            await connection.execute(
                "UPDATE observations SET content_digest = NULL WHERE observation_id = %s",
                (STALE_DIGEST_OBSERVATION,),
            )

        resend = await store.write_observation(
            _stale_digest_batch(),
            idempotency_key="stale_digest_resent",
            content_digest=f"{0xB:064x}",
        )
        assert resend.created is False
        assert resend.observation.observation_id == STALE_DIGEST_OBSERVATION

        with pytest.raises(IdempotencyConflictError, match="different observation content"):
            await store.write_observation(
                _stale_digest_batch(),
                idempotency_key="stale_digest_third",
                content_digest=f"{0xC:064x}",
            )
    finally:
        await store.close()


STALE_DIGEST_TENANT = TenantId("tenant_stale_digest")
STALE_DIGEST_OBSERVATION = ObservationId("observation_stale_digest")


def _stale_digest_batch() -> ObservationBatch:
    media_object_id = MediaObjectId("media_stale_digest")
    return ObservationBatch(
        media_objects=(
            MediaObject(
                media_object_id=media_object_id,
                tenant_id=STALE_DIGEST_TENANT,
                kind=MediaKind.VIDEO,
                uri="s3://memory/tenants/tenant_stale_digest/clip.mp4",
                sha256=f"{9:064x}",
                size_bytes=100,
                created_at=NOW,
                duration_ms=4_000,
            ),
        ),
        observation=Observation(
            observation_id=STALE_DIGEST_OBSERVATION,
            tenant_id=STALE_DIGEST_TENANT,
            device_id=DeviceId("device_01"),
            boot_id="boot_01",
            sequence=1,
            sensor=SensorKind.CAMERA,
            media_object_ids=(media_object_id,),
            occurred_at=NOW,
            ended_at=NOW + timedelta(seconds=4),
            observed_at=NOW,
            clock_offset_ms=0,
        ),
        evidence_spans=(),
    )


async def test_the_runtime_role_can_read_the_migration_ledger(database_url: str) -> None:
    """The re-key bound reads `schema_migrations`, and the runtime role is not the owner.

    Migration 0005 grants per-table access only to tables carrying a `tenant_id`, and this one
    has none, so `write_embedding_on_connection` would fail with "permission denied for table
    schema_migrations" in any deployment while every test here passed -- the fixture connects
    as the owner. That is the gap this asserts against, so it switches role explicitly.
    """
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute("SET ROLE mindbridge_runtime")
        row = await (
            await connection.execute("SELECT applied_at FROM schema_migrations WHERE version = 21")
        ).fetchone()
        await connection.execute("RESET ROLE")

    assert row is not None


async def test_postgres_vertical_path_is_idempotent_and_evidence_first(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The production store runs observe, remember, and recall without a side path."""
    kernel = _kernel(store)
    request = _observe_request(tenant_id="tenant_roundtrip")

    first = await kernel.observe(request)
    retry = await kernel.observe(request)
    job = await store.read_observation_processing_job(
        TenantId("tenant_roundtrip"),
        JobId(first.processing_job_id),
    )
    stored_batch = await store.read_observation_batch(
        TenantId("tenant_roundtrip"),
        ObservationId(first.observation_id),
    )
    evidence_id = first.evidence_ids[0]
    memory = await kernel.remember(
        _remember_request(tenant_id="tenant_roundtrip", evidence_id=evidence_id)
    )
    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_roundtrip",
            query=RecallQuery(text="red screwdriver"),
            filters=RecallFilters(device_ids=("device_01",)),
        )
    )

    assert first.status is ObservationStatus.ACCEPTED
    assert retry.status is ObservationStatus.DUPLICATE
    assert first.processing_job_id == retry.processing_job_id
    assert retry.evidence_ids == first.evidence_ids
    assert job.state is JobState.PENDING
    assert job.observation_id == first.observation_id
    assert stored_batch.observation.observation_id == first.observation_id
    assert stored_batch.media_objects[0].media_object_id == "media_01"
    assert stored_batch.evidence_spans[0].end_ms == 4_000
    assert stored_batch.observation.identity_observations[0].identity_id == "person_device_01"
    assert stored_batch.observation.identity_observations[0].model_reference.model_id == (
        "insightface/buffalo_l"
    )
    assert await _processing_job_count(database_url, "tenant_roundtrip") == 1
    assert memory.verification_status is VerificationStatus.ATTESTED
    assert result.answer == "The robot put the red screwdriver beside the blue toolbox."
    assert result.evidence[0].evidence_id == evidence_id
    assert result.evidence[0].end_ms == 4_000
    assert result.evidence[0].media_url.startswith("https://objects.example.test/media_")

    with pytest.raises(JobNotFoundError):
        await store.read_observation_processing_job(
            TenantId("other_tenant"),
            JobId(first.processing_job_id),
        )


async def test_postgres_enumeration_scans_filters_without_full_text_truncation(
    store: PostgresMemoryStore,
) -> None:
    kernel = _kernel(store)
    tenant_id = "tenant_enumeration"
    for ordinal in reversed(range(3)):
        await kernel.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=f"Attested occurrence {ordinal}",
                memory_type=MemoryType.EPISODIC,
                occurred_at=NOW + timedelta(minutes=ordinal),
                idempotency_key=f"enumeration_{ordinal}",
            )
        )

    result = await kernel.recall(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="text absent from every stored summary"),
            filters=RecallFilters(occurred_after=NOW + timedelta(minutes=1)),
            mode=RecallMode.ENUMERATE,
            include_evidence=False,
        )
    )

    assert result.answer == "2"
    assert [memory.summary for memory in result.memories] == [
        "Attested occurrence 1",
        "Attested occurrence 2",
    ]


async def test_postgres_occurred_before_excludes_equal_boundary(
    store: PostgresMemoryStore,
) -> None:
    kernel = _kernel(store)
    tenant_id = "tenant_before_boundary"
    before = await kernel.remember(
        RememberRequest(
            tenant_id=tenant_id,
            summary="Completed before the question boundary.",
            memory_type=MemoryType.EPISODIC,
            occurred_at=NOW,
        )
    )
    at_boundary = await kernel.remember(
        RememberRequest(
            tenant_id=tenant_id,
            summary="Started exactly at the question boundary.",
            memory_type=MemoryType.EPISODIC,
            occurred_at=NOW + timedelta(minutes=1),
        )
    )

    memories = await store.search_memories_by_ids(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="boundary"),
            filters=RecallFilters(occurred_before=NOW + timedelta(minutes=1)),
        ),
        (MemoryId(before.memory_id), MemoryId(at_boundary.memory_id)),
        limit=10,
    )

    assert [memory.memory_id for memory in memories] == [before.memory_id]


async def test_postgres_filtered_recall_widens_past_ineligible_dense_prefix(
    store: PostgresMemoryStore,
) -> None:
    """Four filtered vectors cannot hide the valid fifth result behind the first LIMIT."""
    tenant_id = TenantId("tenant_filtered_dense_prefix")
    model = ModelReference(model_id="jina")
    space = EmbeddingSpaceReference(space_id="jina-v5")
    memories = tuple(
        MemoryRecord(
            memory_id=MemoryId(f"memory_dense_{ordinal}"),
            tenant_id=tenant_id,
            memory_type=(MemoryType.SEMANTIC if ordinal < 4 else MemoryType.EPISODIC),
            summary=f"Dense candidate {ordinal}",
            evidence_ids=(),
            occurred_at=NOW,
            ended_at=NOW,
            created_at=NOW,
            verification_status=VerificationStatus.ATTESTED,
        )
        for ordinal in range(5)
    )
    for ordinal, memory in enumerate(memories):
        similarity = 1.0 - ordinal / 100
        await store.write_memory(
            memory,
            idempotency_key=f"dense_{ordinal}",
            content_digest=str(ordinal + 1) * 64,
        )
        await store.write_embedding(
            EmbeddingRecord(
                embedding_id=EmbeddingId(f"embedding_dense_{ordinal}"),
                tenant_id=tenant_id,
                object_type=EmbeddedObjectType.MEMORY_RECORD,
                object_id=memory.memory_id,
                values=(similarity, math.sqrt(1.0 - similarity**2)) + (0.0,) * 1_022,
                model_reference=model,
                space_reference=space,
                task="retrieval_document",
                dimension=1_024,
                normalized=True,
                created_at=NOW,
            )
        )

    result = await _kernel(store).recall(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text="words absent from every summary"),
            filters=RecallFilters(memory_types=(MemoryType.EPISODIC,)),
            mode=RecallMode.SEARCH,
            include_evidence=False,
            limit=1,
        )
    )

    assert [memory.memory_id for memory in result.memories] == [memories[-1].memory_id]


async def test_postgres_round_trips_attested_source_memory(store: PostgresMemoryStore) -> None:
    """Explicit source text survives persistence and can ground a reported answer."""
    kernel = _kernel(store)
    memory = await kernel.remember(
        RememberRequest(
            tenant_id="tenant_attested",
            summary="Caroline said she plans to become a counselor.",
            memory_type=MemoryType.SEMANTIC,
            occurred_at=NOW,
        )
    )

    result = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="plans to become a counselor"),
        )
    )

    assert memory.verification_status is VerificationStatus.ATTESTED
    assert result.answer == "Caroline said she plans to become a counselor."
    assert result.evidence == ()

    ranked_memory_ids = (MemoryId(memory.memory_id),)
    matched = await store.search_memories_by_ids(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="words do not constrain dense candidates"),
            filters=RecallFilters(memory_types=(MemoryType.SEMANTIC,)),
        ),
        ranked_memory_ids,
        limit=10,
    )
    filtered = await store.search_memories_by_ids(
        RecallRequest(
            tenant_id="tenant_attested",
            query=RecallQuery(text="words do not constrain dense candidates"),
            filters=RecallFilters(memory_types=(MemoryType.EPISODIC,)),
        ),
        ranked_memory_ids,
        limit=10,
    )
    assert [item.memory_id for item in matched] == [memory.memory_id]
    assert filtered == ()

    found = await kernel.get_memory("tenant_attested", memory.memory_id)
    assert (
        found.model_copy(
            update={
                "trace_id": memory.trace_id,
            }
        )
        # `remember` reports the write's own `status` beside the memory and a read has no such
        # field, so what the two must agree on is the memory they share.
        == MemoryResult.model_validate(memory.model_dump(exclude={"status"}))
    )
    assert found.trace_id.startswith("trace_")
    assert found.useful_access_count == 0
    assert found.last_accessed_at is None
    with pytest.raises(MemoryNotFoundError):
        await kernel.get_memory("other_tenant", memory.memory_id)


async def test_postgres_rejects_idempotency_key_reuse(store: PostgresMemoryStore) -> None:
    """Conflicting retries roll back rather than aliasing two observations."""
    kernel = _kernel(store)
    await kernel.observe(
        _observe_request(tenant_id="tenant_conflict", idempotency_key="request_01")
    )

    with pytest.raises(IdempotencyConflictError):
        await kernel.observe(
            _observe_request(
                tenant_id="tenant_conflict",
                sequence=2,
                idempotency_key="request_01",
            )
        )


async def test_postgres_feedback_evolves_and_versions_memory_atomically(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    kernel = _kernel(store)
    original = await kernel.remember(
        RememberRequest(
            tenant_id="tenant_feedback",
            summary="The screwdriver is on the blue toolbox.",
            memory_type=MemoryType.EPISODIC,
            occurred_at=NOW,
            idempotency_key="remember_original",
        )
    )
    useful_request = FeedbackRequest(
        tenant_id="tenant_feedback",
        feedback_type=FeedbackType.USEFUL,
        memory_id=original.memory_id,
        idempotency_key="feedback_useful",
    )
    useful = await kernel.record_feedback(useful_request)
    useful_retry = await kernel.record_feedback(useful_request)
    correction_request = FeedbackRequest(
        tenant_id="tenant_feedback",
        feedback_type=FeedbackType.CORRECTION,
        memory_id=original.memory_id,
        correction_summary="The screwdriver is inside the green drawer.",
        idempotency_key="feedback_correction",
    )
    correction = await kernel.record_feedback(correction_request)
    correction_retry = await kernel.record_feedback(correction_request)

    old = await kernel.get_memory("tenant_feedback", original.memory_id)
    corrected = await kernel.get_memory(
        "tenant_feedback", cast(str, correction.corrected_memory_id)
    )
    recalled = await kernel.recall(
        RecallRequest(
            tenant_id="tenant_feedback",
            query=RecallQuery(text="green drawer"),
        )
    )

    assert useful.feedback_id == useful_retry.feedback_id
    assert useful.resulting_state is MemoryState.STRENGTHENED
    assert correction.corrected_memory_id == correction_retry.corrected_memory_id
    assert old.superseded_at == NOW
    assert old.positive_feedback_count == 1
    assert old.negative_feedback_count == 1
    assert corrected.supersedes_memory_id == original.memory_id
    assert corrected.verification_status is VerificationStatus.ATTESTED
    assert [memory.memory_id for memory in recalled.memories] == [corrected.memory_id]

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT count(*) FROM memory_feedback
                WHERE tenant_id = 'tenant_feedback'
                """
            )
        ).fetchone()
    assert cast(tuple[int], row)[0] == 2

    with pytest.raises(IdempotencyConflictError):
        await kernel.record_feedback(
            FeedbackRequest(
                tenant_id="tenant_feedback",
                feedback_type=FeedbackType.WRONG,
                memory_id=corrected.memory_id,
                idempotency_key="feedback_useful",
            )
        )


async def test_postgres_deduplicates_media_by_content_hash(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """Different device aliases for identical bytes share one media row."""
    kernel = _kernel(store)
    await kernel.observe(
        _observe_request(
            tenant_id="tenant_dedup",
            sequence=1,
            media_object_id="media_alias_01",
        )
    )
    await kernel.observe(
        _observe_request(
            tenant_id="tenant_dedup",
            sequence=2,
            media_object_id="media_alias_02",
        )
    )

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                "SELECT count(*) FROM media_objects WHERE tenant_id = 'tenant_dedup'"
            )
        ).fetchone()

    assert cast(tuple[int], row)[0] == 1


async def test_observation_job_state_is_atomic_and_retryable(
    store: PostgresMemoryStore,
) -> None:
    """Concurrent deliveries own one attempt and a failed job can later succeed."""
    receipt = await _kernel(store).observe(_observe_request(tenant_id="tenant_job_state"))
    tenant_id = TenantId("tenant_job_state")
    observation_id = ObservationId(receipt.observation_id)
    job_id = JobId(receipt.processing_job_id)

    claims = await asyncio.gather(
        *(
            store.claim_observation_processing_job(tenant_id, observation_id, job_id)
            for _ in range(5)
        )
    )

    assert sum(claim.acquired for claim in claims) == 1
    assert {claim.job.state for claim in claims} == {JobState.RUNNING}
    assert {claim.job.attempt for claim in claims} == {1}
    failed = await store.mark_observation_processing_failed(
        tenant_id,
        observation_id,
        job_id,
        attempt=1,
        error_code="model_unavailable",
    )
    retry = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
    with pytest.raises(MemoryIntegrityError, match="not running"):
        await store.mark_observation_processing_failed(
            tenant_id,
            observation_id,
            job_id,
            attempt=1,
            error_code="stale_attempt",
        )
    succeeded = await store.commit_observation_processing(
        tenant_id,
        observation_id,
        job_id,
        attempt=2,
        output=ObservationProcessingOutput(
            evidence_spans=(),
            events=(),
            entities=(),
            entity_mentions=(),
            claims=(),
            memories=(),
            relations=(),
            embeddings=(),
        ),
    )
    duplicate = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)

    assert failed.state is JobState.FAILED
    assert failed.error_code == "model_unavailable"
    assert retry.acquired is True
    assert retry.job.attempt == 2
    assert retry.job.error_code is None
    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.memory_ids == ()
    assert duplicate.acquired is False
    assert duplicate.job.state is JobState.SUCCEEDED


async def _processing_job_count(database_url: str, tenant_id: str) -> int:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT count(*) FROM jobs
                WHERE tenant_id = %s AND job_type = 'process_observation'
                """,
                (tenant_id,),
            )
        ).fetchone()
    return cast(tuple[int], row)[0]


async def test_job_row_separates_queue_wait_from_work_and_accumulates_cost(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """created_at and updated_at alone answered neither "how long did it wait" nor "cost"."""
    receipt = await _kernel(store).observe(_observe_request(tenant_id="tenant_job_cost"))
    tenant_id = TenantId("tenant_job_cost")
    observation_id = ObservationId(receipt.observation_id)
    job_id = JobId(receipt.processing_job_id)
    empty_output = ObservationProcessingOutput(
        evidence_spans=(),
        events=(),
        entities=(),
        entity_mentions=(),
        claims=(),
        memories=(),
        relations=(),
        embeddings=(),
    )

    async with operation_span("mindbridge.test.first_attempt"):
        await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
        set_current_span_attributes(
            {"mindbridge.model.input_tokens": 120, "mindbridge.model.output_tokens": 8}
        )
        await store.mark_observation_processing_failed(
            tenant_id,
            observation_id,
            job_id,
            attempt=1,
            error_code="model_output_invalid",
        )
    first_attempt = await _job_timing(database_url, tenant_id, job_id)
    async with operation_span("mindbridge.test.second_attempt"):
        await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
        set_current_span_attributes({"mindbridge.model.input_tokens": 30})
        await store.commit_observation_processing(
            tenant_id,
            observation_id,
            job_id,
            attempt=2,
            output=empty_output,
        )
    second_attempt = await _job_timing(database_url, tenant_id, job_id)

    started_at, created_at, input_tokens, output_tokens = first_attempt
    assert started_at is not None and started_at >= created_at
    assert (input_tokens, output_tokens) == (120, 8)
    # The retry reports its own wait, and the tokens the failed attempt burned are still owed.
    assert second_attempt[0] is not None and second_attempt[0] > started_at
    assert second_attempt[2:] == (150, 8)


async def test_the_ledger_scan_finds_only_jobs_a_worker_would_still_claim(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The repair has to match the claim: republishing anything else pays for nothing."""
    kernel = _kernel(store)
    tenant_id = TenantId("tenant_reconcile")
    jobs = []
    for sequence in (1, 2, 3, 4):
        receipt = await kernel.observe(
            _observe_request(
                tenant_id=tenant_id,
                sequence=sequence,
                media_object_id=f"media_reconcile_{sequence}",
            )
        )
        jobs.append((ObservationId(receipt.observation_id), JobId(receipt.processing_job_id)))
    left_pending, failed, stale, succeeded = jobs
    await store.claim_observation_processing_job(tenant_id, *failed)
    await store.mark_observation_processing_failed(
        tenant_id, *failed, attempt=1, error_code="model_output_invalid"
    )
    await store.claim_observation_processing_job(tenant_id, *stale)
    await store.claim_observation_processing_job(tenant_id, *succeeded)
    await store.commit_observation_processing(
        tenant_id,
        *succeeded,
        attempt=1,
        output=ObservationProcessingOutput(
            evidence_spans=(),
            events=(),
            entities=(),
            entity_mentions=(),
            claims=(),
            memories=(),
            relations=(),
            embeddings=(),
        ),
    )

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        # What a lost worker leaves behind: still running, but past the window the claim
        # treats as abandoned. Nothing else in the ledger records that it is unreachable.
        await connection.execute(
            """
            UPDATE jobs SET updated_at = now() - make_interval(secs => %s)
            WHERE tenant_id = %s AND job_id = %s
            """,
            (OBSERVATION_JOB_STALE_AFTER_SECONDS + 60, tenant_id, stale[1]),
        )
        # A job whose observation is gone: the worker would only fail it again.
        await connection.execute(
            """
            INSERT INTO jobs (
                tenant_id, job_id, job_type, state, payload, created_at, updated_at
            )
            VALUES (%s, 'job_process_obs_deleted', 'process_observation', 'pending',
                    '{"observation_id": "obs_deleted"}', now(), now())
            """,
            (tenant_id,),
        )
        await connection.commit()
        claimable = await unreachable_observation_jobs(connection, tenant_id=tenant_id)
        with_failed = await unreachable_observation_jobs(
            connection, tenant_id=tenant_id, include_failed=True
        )
        other_tenant = await unreachable_observation_jobs(
            connection, tenant_id=TenantId("tenant_job_cost")
        )

    assert [job_id for _, _, job_id in claimable] == [left_pending[1], stale[1]]
    assert [job_id for _, _, job_id in with_failed] == [left_pending[1], failed[1], stale[1]]
    assert [job_id for _, _, job_id in other_tenant] == []


async def test_the_reconciler_republishes_what_the_ledger_still_owes(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A row with no message behind it is invisible until the two are compared."""
    tenant_id = TenantId("tenant_republish")
    await _kernel(store).observe(_observe_request(tenant_id=tenant_id))

    report = await reconcile(
        database_url,
        "memory://",
        tenant_id=tenant_id,
        include_failed=False,
        republish=True,
    )
    # Counted from the broker, not from the report: an in-process transport shares its queues
    # by name, so this is the message actually delivered rather than the number claimed.
    delivered = queue_depth(create_task_queue("memory://"))

    assert delivered == 1
    assert report["queue"] == "mindbridge"
    # Zero because the publisher the kernel was given here discards, which is the divergence
    # this repairs; the depth is read before the repair so it describes what was found.
    assert (report["queue_depth"], report["claimable"], report["republished"]) == (0, 1, 1)
    tenants = cast(list[dict[str, object]], report["tenants"])
    # Split out because it is now a real measured duration rather than a constant: a job that has
    # only ever waited has its wait counted from `updated_at`, where the previous expression
    # derived it from a `started_at` that is still NULL until something claims the job -- so this
    # read 0.0 for every pending job no matter how long it had been queued.
    waited = cast(float, tenants[0].pop("queue_wait_seconds"))
    assert waited > 0
    assert tenants == [
        {
            "tenant_id": tenant_id,
            "jobs": 1,
            "pending": 1,
            "running": 0,
            "failed": 0,
            "succeeded": 0,
            "work_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    ]


async def test_the_ledger_charges_the_attempt_in_flight_and_every_attempt_before_it(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """Two ways the ledger understated worker time, both structural rather than approximate.

    The claim stamps `started_at` and `updated_at` together, so `updated_at - started_at` was
    exactly zero for a *running* job -- and this summary exists to answer "who is consuming the
    worker" and orders by that column, so the tenant holding a worker right now contributed
    nothing and sorted last. A retry then moved `started_at` forward, leaving the difference
    covering only the final attempt while the token columns beside it counted every attempt.

    Both are pinned by making the first attempt the slow one: under the old expressions the
    retried tenant's work would collapse to its short second attempt while its "wait" absorbed
    the long first one, so `wait > work`. It is the other way around.
    """
    kernel = _kernel(store)
    slow_seconds = 0.4

    running_tenant = TenantId("tenant_accounting_running")
    receipt = await kernel.observe(
        _observe_request(
            tenant_id=running_tenant, sequence=1, media_object_id="media_accounting_running"
        )
    )
    running_job = (ObservationId(receipt.observation_id), JobId(receipt.processing_job_id))
    await store.claim_observation_processing_job(running_tenant, *running_job)
    await asyncio.sleep(slow_seconds)

    retried_tenant = TenantId("tenant_accounting_retried")
    receipt = await kernel.observe(
        _observe_request(
            tenant_id=retried_tenant, sequence=1, media_object_id="media_accounting_retried"
        )
    )
    retried_job = (ObservationId(receipt.observation_id), JobId(receipt.processing_job_id))
    await store.claim_observation_processing_job(retried_tenant, *retried_job)
    await asyncio.sleep(slow_seconds)
    await store.mark_observation_processing_failed(
        retried_tenant, *retried_job, attempt=1, error_code="model_output_invalid"
    )
    await store.claim_observation_processing_job(retried_tenant, *retried_job)
    await store.commit_observation_processing(
        retried_tenant,
        *retried_job,
        attempt=2,
        output=ObservationProcessingOutput(
            evidence_spans=(),
            events=(),
            entities=(),
            entity_mentions=(),
            claims=(),
            relations=(),
            memories=(),
            embeddings=(),
        ),
    )

    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        rows = {row.tenant_id: row for row in await observation_job_accounting(connection)}

    running = rows[running_tenant]
    assert running.running == 1
    # The defect: this was 0 for the whole time the job held a worker.
    assert running.work_seconds >= slow_seconds

    # An abandoned attempt stops accruing at the stale window. Past it the claim treats the row
    # as reclaimable, so whatever held it is gone; without the cap a worker that died would grow
    # its tenant's total forever and sort every live tenant below a corpse.
    abandoned = await AsyncConnection.connect(database_url, autocommit=True)
    async with abandoned:
        await abandoned.execute(
            # `created_at` moves with it: migration 0022's CHECK forbids a start before creation,
            # which is that constraint doing its job on a hand-built row.
            "UPDATE jobs SET created_at = now() - make_interval(secs => %s),"
            " started_at = now() - make_interval(secs => %s) WHERE tenant_id = %s",
            (
                OBSERVATION_JOB_STALE_AFTER_SECONDS * 6,
                OBSERVATION_JOB_STALE_AFTER_SECONDS * 5,
                running_tenant,
            ),
        )
        stale = {row.tenant_id: row for row in await observation_job_accounting(abandoned)}[
            running_tenant
        ]
    assert stale.work_seconds == pytest.approx(OBSERVATION_JOB_STALE_AFTER_SECONDS, rel=0.01)

    retried = rows[retried_tenant]
    assert retried.succeeded == 1
    # Both attempts are charged, not just the last -- which is what makes this consistent with
    # the token columns, where 0022 already counted every attempt.
    assert retried.work_seconds >= slow_seconds
    # And the long first attempt is work, not waiting. Reversed under the old expressions.
    assert retried.queue_wait_seconds < retried.work_seconds


async def test_reclaiming_a_stale_attempt_charges_it_once_and_stops_at_the_stale_window(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A dead worker's abandoned interval was written to both duration columns, uncapped.

    The claim's SET list reads the pre-update row, where a running job has `started_at` equal to
    `updated_at` because the previous claim stamped both in one statement. So the same interval
    was added to `queue_wait_seconds` and to `work_seconds`, and the wait column absorbed time
    the job spent running. The reader's cap only guards the interval still open; a reclaim wrote
    the uncapped value permanently, so `--republish` after a worker died sorted the tenant with
    the deadest worker first in the summary that answers "who is consuming the worker".
    """
    tenant_id = TenantId("tenant_stale_reclaim")
    receipt = await _kernel(store).observe(
        _observe_request(tenant_id=tenant_id, media_object_id="media_stale_reclaim")
    )
    job = (ObservationId(receipt.observation_id), JobId(receipt.processing_job_id))
    await store.claim_observation_processing_job(tenant_id, *job)

    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute(
            "SELECT set_config('mindbridge.tenant_id', %s, false)", (tenant_id,)
        )
        # Age the claimed row past the stale window, exactly as a worker that died would leave
        # it. `created_at` moves too because migration 0022 forbids a start before creation.
        await connection.execute(
            """
            UPDATE jobs
            SET created_at = now() - make_interval(secs => %s),
                started_at = now() - make_interval(secs => %s),
                updated_at = now() - make_interval(secs => %s)
            WHERE tenant_id = %s
            """,
            (
                OBSERVATION_JOB_STALE_AFTER_SECONDS * 3,
                OBSERVATION_JOB_STALE_AFTER_SECONDS * 2,
                OBSERVATION_JOB_STALE_AFTER_SECONDS * 2,
                tenant_id,
            ),
        )
        waited_before = await _job_durations(connection, tenant_id)

        claim = await store.claim_observation_processing_job(tenant_id, *job)
        wait_seconds, work_seconds = await _job_durations(connection, tenant_id)

    assert claim.acquired and claim.job.attempt == 2
    # The abandoned attempt was running, not queued, so it owes the wait column nothing.
    assert wait_seconds == waited_before[0]
    # And it stops accruing where the reader stops counting it, for the same reason: past the
    # stale window whatever held it is gone.
    assert work_seconds == pytest.approx(OBSERVATION_JOB_STALE_AFTER_SECONDS, rel=0.01)


async def _job_durations(connection: AsyncConnection, tenant_id: TenantId) -> tuple[float, float]:
    """Read the columns as written, which is what a reclaim makes permanent."""
    cursor = await connection.execute(
        "SELECT queue_wait_seconds, work_seconds FROM jobs WHERE tenant_id = %s", (tenant_id,)
    )
    return cast(tuple[float, float], await cursor.fetchone())


async def test_a_tenant_confined_role_is_refused_a_ledger_wide_scan(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """Under FORCE row-level security an unscoped scan reports an empty ledger, not an error."""
    receipt = await _kernel(store).observe(_observe_request(tenant_id="tenant_rls_scan"))
    tenant_id = TenantId("tenant_rls_scan")

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        await connection.execute("SET ROLE mindbridge_runtime")
        confined = await tenant_scope_required(connection)
        blind = await unreachable_observation_jobs(connection)
        await connection.execute(
            "SELECT set_config('mindbridge.tenant_id', %s, false)",
            (tenant_id,),
        )
        scoped = await unreachable_observation_jobs(connection, tenant_id=tenant_id)
        await connection.execute("RESET ROLE")
        unconfined = await tenant_scope_required(connection)

    assert confined is True
    assert blind == ()
    assert [job_id for _, _, job_id in scoped] == [JobId(receipt.processing_job_id)]
    assert unconfined is False


async def _job_timing(
    database_url: str,
    tenant_id: str,
    job_id: str,
) -> tuple[datetime | None, datetime, int | None, int | None]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT started_at, created_at, input_tokens, output_tokens
                FROM jobs WHERE tenant_id = %s AND job_id = %s
                """,
                (tenant_id, job_id),
            )
        ).fetchone()
    return cast("tuple[datetime | None, datetime, int | None, int | None]", row)


def _observe_request(
    *,
    tenant_id: str,
    sequence: int = 1,
    media_object_id: str = "media_01",
    idempotency_key: str | None = None,
) -> ObserveRequest:
    return ObserveRequest(
        tenant_id=tenant_id,
        device_id="device_01",
        boot_id="boot_01",
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id=media_object_id,
                kind=MediaKind.VIDEO,
                uri=f"s3://memories/{media_object_id}.mp4",
                sha256="a" * 64,
                size_bytes=100,
                duration_ms=4_000,
                created_at=NOW,
            ),
        ),
        occurred_at=NOW + timedelta(seconds=sequence),
        ended_at=NOW + timedelta(seconds=sequence + 4),
        observed_at=NOW,
        identity_observations=(
            IdentityObservationInput(
                identity_id="person_device_01",
                kind=IdentityKind.FACE,
                start_ms=500,
                end_ms=3_500,
                confidence=0.91,
                model_id="insightface/buffalo_l",
            ),
        ),
        idempotency_key=idempotency_key,
    )


def _remember_request(*, tenant_id: str, evidence_id: str) -> RememberRequest:
    return RememberRequest(
        tenant_id=tenant_id,
        summary="The robot put the red screwdriver beside the blue toolbox.",
        memory_type=MemoryType.EPISODIC,
        occurred_at=NOW,
        evidence_ids=(evidence_id,),
    )


def _kernel(store: PostgresMemoryStore) -> MemoryKernel:
    media_access = DeterministicMediaUrlSigner()
    answerer = FirstCandidateAnswerer()
    return MemoryKernel(
        store,
        answerer,
        answerer,
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=DiscardingObservationJobPublisher(),
        embedder=FixedEmbedder(),
        clock=lambda: NOW,
    )


async def test_postgres_lexical_recall_matches_questions_and_identity_tokens(
    store: PostgresMemoryStore,
) -> None:
    """Lexical recall must survive whole-sentence queries and bracketed identity tokens."""
    kernel = _kernel(store)
    tenant_id = "tenant_lexical"
    summaries = {
        "target": "<voice_0> explains that the meat was prepared according to Islamic rules.",
        "decoy": "A cyclist repairs a punctured tyre beside the road.",
    }
    for name, summary in summaries.items():
        await kernel.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=summary,
                memory_type=MemoryType.EPISODIC,
                occurred_at=NOW,
                idempotency_key=f"lexical_{name}",
            )
        )

    def _request(text: str) -> RecallRequest:
        return RecallRequest(tenant_id=tenant_id, query=RecallQuery(text=text))

    question = await store.search_memories(
        _request("How was the meat prepared according to Islamic rules?"), limit=5
    )
    identity = await store.search_memories(_request("What did <voice_0> say?"), limit=5)
    unrelated = await store.search_memories(_request("What colour is the moon rock?"), limit=5)
    # Mapping `tag` into the configuration is what lets a lexeme carry a backslash, and the
    # tsquery the query side builds has to quote it as tsquery does rather than as SQL does.
    # Escaping this wrong raises a syntax error the caller sees as a bare 500.
    backslash = await store.search_memories(_request('see <img src="a\\b.png"/> please'), limit=5)
    # The substring arm takes caller text, not a pattern. Read as LIKE, either of these is a
    # wildcard matching every summary the tenant owns; read literally, `%` appears in neither
    # summary and `_` appears only inside <voice_0>. Asserting the underscore case this way
    # separates "escaped correctly" from "stopped matching at all".
    percent = await store.search_memories(_request("%"), limit=5)
    underscore = await store.search_memories(_request("_"), limit=5)

    # A question is not a conjunction of every one of its words, and <voice_0> is a term the
    # parser would otherwise discard as an HTML tag on both the document and the query side.
    # Each match is asserted whole: ranking the target first is not enough, because a query
    # that also drags in the decoy through a shared stopword still ranks the target first.
    # The unrelated probe is itself a question, so it fails if stopwords ever match on their own.
    assert [memory.summary for memory in question] == [summaries["target"]]
    assert [memory.summary for memory in identity] == [summaries["target"]]
    assert [memory.summary for memory in unrelated] == []
    assert [memory.summary for memory in backslash] == []
    assert [memory.summary for memory in percent] == []
    assert [memory.summary for memory in underscore] == [summaries["target"]]
