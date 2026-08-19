"""PostgreSQL checks for automatic lifecycle pages and concurrent feedback."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from psycopg import AsyncConnection
from psycopg.errors import CheckViolation, InsufficientPrivilege

from mindbridge.application.lifecycle import (
    EvolveMemoryLifecycle,
    LifecycleSweepRequest,
    MemoryLifecycleChange,
)
from mindbridge.core import (
    FeedbackId,
    FeedbackType,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryStrengthPolicy,
    MemoryType,
    TenantId,
    VerificationStatus,
    evolve_memory_strength,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore

NOW = datetime.now(timezone.utc) + timedelta(days=1)
POLICY = MemoryStrengthPolicy(age_decay_weight=1.0, cold_below=0.0)
pytestmark = pytest.mark.integration


async def test_postgres_lifecycle_sweep_pages_and_cools_unused_memory(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_page")
    old = _memory(tenant_id, "memory_01", NOW - timedelta(days=30))
    recent = _memory(tenant_id, "memory_02", NOW)
    await _write_memory(store, old)
    await _write_memory(store, recent)
    use_case = EvolveMemoryLifecycle(store, POLICY)

    first = await use_case.run(
        LifecycleSweepRequest(tenant_id=tenant_id, evaluated_at=NOW, limit=1)
    )
    second = await use_case.run(
        LifecycleSweepRequest(
            tenant_id=tenant_id,
            evaluated_at=NOW,
            after_memory_id=first.next_cursor,
            limit=1,
        )
    )

    assert first.evaluated_count == 1
    assert first.updated_count == 1
    assert first.next_cursor == "memory_01"
    assert second.evaluated_count == 1
    assert second.updated_count == 0
    assert second.next_cursor is None
    assert (await store.read_memory(tenant_id, old.memory_id)).state is MemoryState.COLD
    assert (await store.read_memory(tenant_id, recent.memory_id)).state is MemoryState.ACTIVE


async def test_postgres_lifecycle_does_not_overwrite_concurrent_feedback(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_concurrent")
    memory = _memory(tenant_id, "memory_01", NOW - timedelta(days=30))
    await _write_memory(store, memory)
    stale_change = MemoryLifecycleChange(
        previous=memory,
        evolved=evolve_memory_strength(memory, NOW, POLICY),
    )
    await store.record_feedback(
        MemoryFeedback(
            feedback_id=FeedbackId("feedback_01"),
            tenant_id=tenant_id,
            feedback_type=FeedbackType.USEFUL,
            memory_id=memory.memory_id,
            created_at=NOW,
        ),
        None,
        idempotency_key="feedback_01",
        content_digest="f" * 64,
    )

    updated_count = await store.update_memory_lifecycles(
        (stale_change,),
        evaluated_at=NOW,
    )
    stored = await store.read_memory(tenant_id, memory.memory_id)

    assert updated_count == 0
    assert stored.positive_feedback_count == 1
    assert stored.state is MemoryState.STRENGTHENED


async def test_postgres_lifecycle_snapshot_skips_later_writes(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_snapshot")
    snapshot_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
    created_at = snapshot_at - timedelta(days=30)
    feedback_memory = _memory(tenant_id, "memory_feedback", created_at)
    access_memory = _memory(tenant_id, "memory_access", created_at)
    delayed_memory = _memory(tenant_id, "memory_delayed", created_at)
    await _write_memory(store, feedback_memory)
    await _write_memory(store, access_memory)
    await _write_memory(store, delayed_memory)
    later = snapshot_at + timedelta(minutes=1)
    await store.record_feedback(
        MemoryFeedback(
            feedback_id=FeedbackId("feedback_later"),
            tenant_id=tenant_id,
            feedback_type=FeedbackType.USEFUL,
            memory_id=feedback_memory.memory_id,
            created_at=later,
        ),
        None,
        idempotency_key="feedback_later",
        content_digest="c" * 64,
    )
    await store.record_memory_accesses(
        tenant_id,
        (access_memory.memory_id,),
        accessed_at=later,
    )

    snapshot = await EvolveMemoryLifecycle(store, POLICY).run(
        LifecycleSweepRequest(tenant_id=tenant_id, evaluated_at=snapshot_at, limit=10)
    )
    next_snapshot = await EvolveMemoryLifecycle(store, POLICY).run(
        LifecycleSweepRequest(
            tenant_id=tenant_id,
            evaluated_at=datetime(2100, 1, 1, tzinfo=timezone.utc),
            limit=10,
        )
    )

    assert snapshot.evaluated_count == 0
    assert snapshot.updated_count == 0
    assert next_snapshot.evaluated_count == 3


async def test_postgres_recall_access_reactivates_cold_memory_without_rewinding_time(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_access")
    memory = _memory(tenant_id, "memory_access", NOW - timedelta(days=30))
    await _write_memory(store, memory)
    await EvolveMemoryLifecycle(store, POLICY).run(
        LifecycleSweepRequest(tenant_id=tenant_id, evaluated_at=NOW, limit=10)
    )

    first = await store.record_memory_accesses(
        tenant_id,
        (memory.memory_id,),
        accessed_at=NOW + timedelta(minutes=1),
    )
    second = await store.record_memory_accesses(
        tenant_id,
        (memory.memory_id,),
        accessed_at=NOW,
    )

    assert first[0].state is MemoryState.ACTIVE
    assert first[0].useful_access_count == 1
    assert second[0].useful_access_count == 2
    assert second[0].last_accessed_at == NOW + timedelta(minutes=1)


async def test_postgres_concurrent_overlapping_accesses_do_not_deadlock(
    store: PostgresMemoryStore,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_overlapping_access")
    memories = tuple(
        _memory(tenant_id, MemoryId(f"memory_{index:02d}"), NOW) for index in range(20)
    )
    for memory in memories:
        await _write_memory(store, memory)

    await asyncio.gather(
        *(
            store.record_memory_accesses(
                tenant_id,
                tuple(memory.memory_id for memory in memories[offset:] + memories[:offset]),
                accessed_at=NOW,
            )
            for offset in range(8)
        )
    )

    assert (await store.read_memory(tenant_id, memories[0].memory_id)).useful_access_count == 8


async def test_postgres_clamps_access_before_memory_creation(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_id = TenantId("tenant_lifecycle_access_time")
    memory = _memory(tenant_id, "memory_access_time", NOW)
    await _write_memory(store, memory)
    accessed = await store.record_memory_accesses(
        tenant_id,
        (memory.memory_id,),
        accessed_at=NOW - timedelta(seconds=1),
    )

    assert accessed[0].useful_access_count == 1
    assert accessed[0].last_accessed_at == memory.created_at

    connection = await AsyncConnection.connect(database_url, autocommit=True)

    async with connection:
        with pytest.raises(CheckViolation):
            await connection.execute(
                """
                UPDATE memory_records
                SET last_accessed_at = created_at - interval '1 second'
                WHERE tenant_id = %s AND memory_id = %s
                """,
                (tenant_id, memory.memory_id),
            )


async def test_postgres_runtime_role_enforces_tenant_row_security(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    tenant_a = TenantId("tenant_rls_a")
    tenant_b = TenantId("tenant_rls_b")
    memory_a = _memory(tenant_a, "memory_rls_a", NOW)
    memory_b = _memory(tenant_b, "memory_rls_b", NOW)
    await _write_memory(store, memory_a)
    await _write_memory(store, memory_b)

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        await connection.execute("SET ROLE mindbridge_runtime")
        await connection.execute(
            "SELECT set_config('mindbridge.tenant_id', %s, true)",
            (tenant_a,),
        )
        rows = await (
            await connection.execute(
                "SELECT tenant_id FROM memory_records ORDER BY tenant_id, memory_id"
            )
        ).fetchall()

        assert {row[0] for row in rows} == {tenant_a}
        with pytest.raises(InsufficientPrivilege):
            await connection.execute(
                "UPDATE memory_records SET tenant_id = %s WHERE tenant_id = %s AND memory_id = %s",
                (tenant_b, tenant_a, memory_a.memory_id),
            )


async def test_postgres_clip_purge_only_touches_fully_compressed_evidence(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The EXISTS guard is what keeps a not-yet-cited evidence span from losing its clips."""
    tenant_id = TenantId("tenant_clip_purge")
    await _seed_clip_graph(database_url, tenant_id)

    purged = await store.purge_compressed_clips(tenant_id, limit=100)

    connection = await AsyncConnection.connect(database_url)
    async with connection:
        surviving = await (
            await connection.execute(
                "SELECT evidence_id FROM evidence_clips WHERE tenant_id = %s ORDER BY evidence_id",
                (tenant_id,),
            )
        ).fetchall()
        media = await (
            await connection.execute(
                "SELECT media_object_id FROM media_objects "
                "WHERE tenant_id = %s ORDER BY media_object_id",
                (tenant_id,),
            )
        ).fetchall()
        spans = await (
            await connection.execute(
                "SELECT count(*) FROM evidence_spans WHERE tenant_id = %s",
                (tenant_id,),
            )
        ).fetchone()

    assert purged == 1
    # Only the wholly compressed span loses its clip. The span shared with an active memory keeps
    # its clip, and so does the span no memory cites yet.
    assert [row[0] for row in surviving] == ["evidence_shared", "evidence_unlinked"]
    # The purged clip's derived media row is gone, which is what orphans its storage key; the
    # source object and every evidence span stay put.
    assert "media_clip_compressed" not in {row[0] for row in media}
    assert "media_source" in {row[0] for row in media}
    assert spans is not None and spans[0] == 3


async def _seed_clip_graph(database_url: str, tenant_id: TenantId) -> None:
    """Build one compressed, one shared, and one uncited evidence span, each with a clip."""
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute(
            "INSERT INTO observations (tenant_id, observation_id, device_id, boot_id, sequence,"
            " sensor, occurred_at, ended_at, observed_at, clock_offset_ms, content_digest)"
            " VALUES (%s, 'observation_01', 'd', 'b', 1, 'camera', %s, %s, %s, 0, %s)",
            (tenant_id, NOW, NOW, NOW, "f" * 64),
        )
        media = (
            "media_source",
            "media_clip_compressed",
            "media_clip_shared",
            "media_clip_unlinked",
        )
        for digest, media_object_id in enumerate(media, start=1):
            await connection.execute(
                "INSERT INTO media_objects (tenant_id, media_object_id, kind, uri, sha256,"
                " size_bytes, created_at, derived_from_media_object_id)"
                " VALUES (%s, %s, 'video', %s, %s, 1024, %s, %s)",
                (
                    tenant_id,
                    media_object_id,
                    f"s3://bucket/{media_object_id}.mp4",
                    str(digest) * 64,
                    NOW,
                    None if media_object_id == "media_source" else "media_source",
                ),
            )
        for evidence_id, clip_media in zip(
            ("evidence_compressed", "evidence_shared", "evidence_unlinked"), media[1:], strict=True
        ):
            await connection.execute(
                "INSERT INTO evidence_spans (tenant_id, evidence_id, observation_id,"
                " media_object_id, start_ms, end_ms, created_at)"
                " VALUES (%s, %s, 'observation_01', 'media_source', 0, 1000, %s)",
                (tenant_id, evidence_id, NOW),
            )
            await connection.execute(
                "INSERT INTO evidence_clips (tenant_id, evidence_id, ordinal, media_object_id,"
                " start_ms, end_ms, created_at) VALUES (%s, %s, 0, %s, 0, 1000, %s)",
                (tenant_id, evidence_id, clip_media, NOW),
            )
        # memory_active also cites the shared span, so that span's clip must survive;
        # evidence_unlinked is cited by nobody, which is what the EXISTS guard protects.
        for digest, (memory_id, state, evidence_ids) in enumerate(
            (
                ("memory_compressed", "compressed", ("evidence_compressed", "evidence_shared")),
                ("memory_active", "active", ("evidence_shared",)),
            ),
            start=1,
        ):
            await connection.execute(
                "INSERT INTO memory_records (tenant_id, memory_id, memory_type, summary,"
                " verification_status, state, occurred_at, ended_at, content_digest, created_at,"
                " lifecycle_changed_at)"
                " VALUES (%s, %s, 'episodic', 'seeded', 'unverified', %s, %s, %s, %s, %s, %s)",
                (tenant_id, memory_id, state, NOW, NOW, str(digest) * 64, NOW, NOW),
            )
            for evidence_id in evidence_ids:
                await connection.execute(
                    "INSERT INTO memory_evidence (tenant_id, memory_id, evidence_id)"
                    " VALUES (%s, %s, %s)",
                    (tenant_id, memory_id, evidence_id),
                )


async def _write_memory(store: PostgresMemoryStore, memory: MemoryRecord) -> None:
    await store.write_memory(
        memory,
        idempotency_key=memory.memory_id,
        content_digest=("a" if memory.memory_id == "memory_01" else "b") * 64,
    )


def _memory(tenant_id: TenantId, memory_id: str, created_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=tenant_id,
        memory_type=MemoryType.EPISODIC,
        summary=f"Retained event {memory_id}",
        evidence_ids=(),
        occurred_at=created_at,
        ended_at=created_at,
        created_at=created_at,
        verification_status=VerificationStatus.ATTESTED,
    )
