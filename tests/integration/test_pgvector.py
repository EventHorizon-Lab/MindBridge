"""Integration checks for the versioned pgvector index."""

from datetime import datetime, timezone

import pytest
from pgvector import Vector

from mindbridge.application.ports import EmbeddingSearch
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    ModelReference,
    TenantId,
    derive_embedding_id,
)
from mindbridge.infrastructure._postgres_media import write_media_object
from mindbridge.infrastructure._postgres_types import tenant_connection
from mindbridge.infrastructure.postgres import PostgresMemoryStore

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
VECTOR_TENANT = TenantId("tenant_vectors")

pytestmark = pytest.mark.integration


async def test_pgvector_separates_search_spaces(
    store: PostgresMemoryStore,
) -> None:
    """The cloud index retrieves only vectors written into the requested space."""
    model = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
    space = EmbeddingSpaceReference(space_id="jina-v5")
    first = _embedding_record(
        embedding_id="embedding_01",
        object_id="memory_near",
        values=(1.0,) + (0.0,) * 1_023,
        model=model,
        space=space,
    )
    second = _embedding_record(
        embedding_id="embedding_02",
        object_id="memory_far",
        values=(0.0, 1.0) + (0.0,) * 1_022,
        model=model,
        space=space,
    )

    assert await store.write_embedding(first) is True
    assert await store.write_embedding(first) is False
    assert (
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_01",
                object_id="memory_near",
                values=(0.999_999_995, 0.000_1) + (0.0,) * 1_022,
                model=model,
                space=space,
            )
        )
        is False
    )
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_01",
                object_id="memory_near",
                values=(0.999_5, 0.031_618_824) + (0.0,) * 1_022,
                model=model,
                space=space,
            )
        )
    assert await store.write_embedding(second) is True
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_01",
                object_id="memory_near",
                values=second.values,
                model=model,
                space=space,
            )
        )
    matches = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=VECTOR_TENANT,
            values=first.values,
            space_reference=space,
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )
    other_space = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=VECTOR_TENANT,
            values=first.values,
            space_reference=EmbeddingSpaceReference(space_id="jina-v5-text-matching"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )
    thresholded = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=VECTOR_TENANT,
            values=first.values,
            space_reference=space,
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
            minimum_similarity=0.5,
        )
    )

    assert [match.object_id for match in matches] == ["memory_near", "memory_far"]
    assert matches[0].similarity == pytest.approx(1.0)
    assert [match.object_id for match in thresholded] == ["memory_near"]
    assert other_space == ()


async def test_unreachable_probe_is_scoped_per_tenant_and_per_object_type(
    store: PostgresMemoryStore,
) -> None:
    """A reachable memory record must not vouch for evidence stranded in another space."""
    # A dedicated tenant: this writes into the session-scoped database that every other
    # integration test shares, and tenant_vectors already owns fixture vectors.
    tenant_id = TenantId("tenant_space_probe")
    space = EmbeddingSpaceReference(space_id="jina-v5")
    drifted = EmbeddingSpaceReference(space_id="jina-v5-text-matching")
    model = ModelReference(model_id="jina-omni")

    assert await store.unreachable_embedded_object_types(tenant_id, space) == ()
    assert await store.write_embedding(
        _embedding_record(
            embedding_id="embedding_probe_memory",
            object_id="memory_probe",
            values=(0.0, 0.0, 1.0) + (0.0,) * 1_021,
            model=model,
            space=space,
            tenant_id=tenant_id,
        )
    )
    assert await store.write_embedding(
        _embedding_record(
            embedding_id="embedding_probe_evidence",
            object_id="evidence_probe",
            values=(0.0, 0.0, 0.0, 1.0) + (0.0,) * 1_020,
            model=model,
            space=drifted,
            tenant_id=tenant_id,
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
        )
    )

    # The memory record sits in `space`, so a whole-tenant probe would report nothing wrong.
    assert await store.unreachable_embedded_object_types(tenant_id, space) == (
        EmbeddedObjectType.EVIDENCE_SPAN,
    )
    assert await store.unreachable_embedded_object_types(tenant_id, drifted) == (
        EmbeddedObjectType.MEMORY_RECORD,
    )
    assert await store.unreachable_embedded_object_types(TenantId("tenant_empty"), drifted) == ()


async def test_open_refuses_a_dimension_the_index_cannot_store(database_url: str) -> None:
    """A shrunk Matryoshka width must fail at startup, not on the first write."""
    mismatched = PostgresMemoryStore(database_url, embedding_dimension=256)

    with pytest.raises(MemoryIntegrityError, match=r"vector\(1024\).*configured for 256"):
        await mismatched.open()


async def test_pgvector_keeps_one_vector_for_a_re_encoded_entity_name(
    store: PostgresMemoryStore,
) -> None:
    """An entity name is re-encoded in later batches, so encoder noise must not fail the write."""
    model = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
    space = EmbeddingSpaceReference(space_id="jina-v5")
    stored = _embedding_record(
        embedding_id="embedding_entity_01",
        object_id="entity_red_tool",
        values=(1.0,) + (0.0,) * 1_023,
        model=model,
        space=space,
        object_type=EmbeddedObjectType.ENTITY,
    )
    # 0.999 930 against the stored vector: the measured drift from batching one short name
    # beside long event descriptions, and well outside the identical-replay tolerance.
    re_encoded = _embedding_record(
        embedding_id="embedding_entity_01",
        object_id="entity_red_tool",
        values=(0.999_93, 0.011_832) + (0.0,) * 1_022,
        model=model,
        space=space,
        object_type=EmbeddedObjectType.ENTITY,
    )

    assert await store.write_embedding(stored) is True
    assert await store.write_embedding(re_encoded) is False
    # The same drift on any other object type is still a conflict: only an entity's text is
    # recoverable from its object ID, so only an entity may be re-encoded from a new batch.
    assert (
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_memory_01",
                object_id="memory_red_tool",
                values=stored.values,
                model=model,
                space=space,
            )
        )
        is True
    )
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_memory_01",
                object_id="memory_red_tool",
                values=re_encoded.values,
                model=model,
                space=space,
            )
        )


async def test_pgvector_accepts_a_reencoded_vector_only_when_the_caller_allows_it(
    store: PostgresMemoryStore,
) -> None:
    """A batch's vectors depend on the batch's composition, so two concurrent
    `remember` batches that share one memory encode it beside different neighbours.
    The memory's `embedding_id` pins its text -- `write_memory` has already refused the
    idempotency key if it carried a different content digest -- so that difference is
    encoder noise and the caller says so. Without the flag the same write is still a
    conflict, which is what keeps Summary consolidation honest: `_summary_memory`
    derives its ID from the source IDs and timestamp, NOT from the generated summary
    text, so there a differing vector really can mean differing content.
    """
    model = ModelReference(model_id="jina")
    space = EmbeddingSpaceReference(space_id="jina-v5")
    # Written after this database was migrated, which is what makes the last assertion below
    # about the guard rather than about the one-time re-key: only a row predating migration
    # 0021 can have had its ID derived by the recipe that migration changed. The session
    # fixture applies every migration before any test body runs, so this is ordered, not raced.
    written_after_0021 = datetime.now(timezone.utc)
    stored = _embedding_record(
        embedding_id="embedding_memory_reencode",
        object_id="memory_reencode",
        values=(1.0,) + (0.0,) * 1_023,
        model=model,
        space=space,
        created_at=written_after_0021,
    )
    # 0.999 93 against the stored vector: the measured drift from encoding one text alone
    # versus mid-batch, and well outside the identical-replay tolerance.
    reencoded = _embedding_record(
        embedding_id="embedding_memory_reencode",
        object_id="memory_reencode",
        values=(0.999_93, 0.011_832) + (0.0,) * 1_022,
        model=model,
        space=space,
    )

    assert await store.write_embedding(stored) is True
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(reencoded)
    assert await store.write_embedding(reencoded, allow_reencoding=True) is False
    # The flag skips the vector comparison outright, exactly as the entity carve-out does,
    # rather than widening it to some larger tolerance: its premise is that this ID can
    # only ever encode one text, and under that premise any difference is encoder noise.
    # So it accepts an arbitrarily different vector too, and that is the cost of the flag.
    assert (
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_memory_reencode",
                object_id="memory_reencode",
                values=(0.0, 1.0) + (0.0,) * 1_022,
                model=model,
                space=space,
            ),
            allow_reencoding=True,
        )
        is False
    )
    # What the flag does NOT forgive: a row whose stored embedding_id is a different
    # version. embedding_id is not part of the unique key, so this is the case where the
    # same object really is indexed under another version, and it still refuses. The stored
    # row here postdates migration 0021, so it is not eligible for the one-time re-key that
    # heals IDs derived by the recipe 0021 changed.
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_memory_reencode_v2",
                object_id="memory_reencode",
                values=reencoded.values,
                model=model,
                space=space,
                created_at=written_after_0021,
            ),
            allow_reencoding=True,
        )


async def test_a_vector_stranded_under_its_old_identifier_is_re_keyed(
    store: PostgresMemoryStore,
) -> None:
    """`embedding_id` is content-addressed, so changing what it hashes strands every stored ID.

    The ID is unreachable from the same inputs while the object key -- tenant, object, model,
    space, task -- is unchanged, so the writer meets a stored row with equivalent content under
    a different ID. That is not content drift and must not be reported as it. Re-keying rather
    than merely tolerating is what stops it recurring: the ID is what `has_embedding` looks up,
    so a row left under its old name pays for an encode it already has on every future pass.

    The legacy row is inserted with `embedding_id_recipe = 1` rather than written through the
    store, because that column is what makes it legacy. Simulating it with a backdated
    `created_at` is what the previous bound did, and `created_at` is caller-supplied.
    """
    model = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
    space = EmbeddingSpaceReference(space_id="jina-v5-rekey")
    values = (1.0, 0.0) + (0.0,) * 1_022
    legacy = _embedding_record(
        embedding_id="embedding_derived_by_recipe_one",
        object_id="memory_stranded",
        values=values,
        model=model,
        space=space,
    )
    current = _embedding_record(
        embedding_id="embedding_derived_by_recipe_two",
        object_id="memory_stranded",
        values=values,
        model=model,
        space=space,
    )

    await _insert_at_recipe(store, legacy, recipe=1)
    assert await store.has_embedding(VECTOR_TENANT, current.embedding_id) is False

    assert await store.write_embedding(current) is False

    # Healed, and healed exactly once: one row, under the name the current recipe derives.
    assert await store.has_embedding(VECTOR_TENANT, current.embedding_id) is True
    assert await store.has_embedding(VECTOR_TENANT, legacy.embedding_id) is False
    async with tenant_connection(store._pool, VECTOR_TENANT) as connection:
        row = await (
            await connection.execute(
                "SELECT count(*) FROM embeddings WHERE tenant_id = %s AND object_id = %s",
                (VECTOR_TENANT, "memory_stranded"),
            )
        ).fetchone()
    assert row is not None and row[0] == 1

    # A genuinely different vector under the same object key is still drift, not a re-key.
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_third_name",
                object_id="memory_stranded",
                values=(0.0, 1.0) + (0.0,) * 1_022,
                model=model,
                space=space,
            )
        )

    # And an ID disagreement on a row already at the current recipe is refused, which is the
    # half a general amnesty would have thrown away: the row was written by this recipe, so a
    # different ID for the same object key is not a stranded name.
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_a_fourth_name",
                object_id="memory_stranded",
                values=values,
                model=model,
                space=space,
            )
        )


async def test_one_object_holds_a_vector_in_each_space_while_re_embedding(
    store: PostgresMemoryStore,
) -> None:
    """The workflow migration 0025 widened the unique key for, and could not actually run.

    `docs/configuration.md` tells an operator that vectors in several spaces are accepted while
    a re-embedding is in progress. Widening the key was necessary and not sufficient: the table
    is also `PRIMARY KEY (tenant_id, embedding_id)`, and five of the six recipes deriving that
    ID did not hash `space_id`, so the second vector derived the *same* ID, collided on the
    primary key, and `write_embedding` raised "embedding conflict could not be resolved" --
    naming a conflict it could not see. A claim is used because `kernel.py`'s memory-record
    recipe was the one that already worked, and testing that one proves nothing.
    """
    model = ModelReference(model_id="jinaai/jina-embeddings-v5-omni-small-retrieval")
    claim_id = "claim_being_re_embedded"
    first, second = (
        _embedding_record(
            embedding_id=derive_embedding_id(
                VECTOR_TENANT,
                EmbeddedObjectType.CLAIM.value,
                claim_id,
                model_id=model.model_id,
                space_id=space_id,
                task="retrieval_document",
            ),
            object_id=claim_id,
            object_type=EmbeddedObjectType.CLAIM,
            values=values,
            model=model,
            space=EmbeddingSpaceReference(space_id=space_id),
        )
        for space_id, values in (
            ("jina-v5-1024", (1.0, 0.0) + (0.0,) * 1_022),
            ("jina-v6-1024", (0.0, 1.0) + (0.0,) * 1_022),
        )
    )

    assert first.embedding_id != second.embedding_id, "the recipe must vary with space_id"
    assert await store.write_embedding(first) is True
    assert await store.write_embedding(second) is True

    async with tenant_connection(store._pool, VECTOR_TENANT) as connection:
        rows = await (
            await connection.execute(
                "SELECT space_id FROM embeddings WHERE tenant_id = %s AND object_id = %s"
                " ORDER BY space_id",
                (VECTOR_TENANT, claim_id),
            )
        ).fetchall()

    assert [row[0] for row in rows] == ["jina-v5-1024", "jina-v6-1024"]


async def _insert_at_recipe(
    store: PostgresMemoryStore,
    embedding: EmbeddingRecord,
    *,
    recipe: int,
) -> None:
    """Store one vector as an older recipe wrote it, which the writer will no longer do."""
    async with tenant_connection(store._pool, embedding.tenant_id) as connection:
        await connection.execute(
            """
            INSERT INTO embeddings (
                tenant_id, embedding_id, object_type, object_id, model_id, space_id, task,
                dimension, normalized, embedding, created_at, embedding_id_recipe
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                Vector(list(embedding.values)),
                embedding.created_at,
                recipe,
            ),
        )


def _embedding_record(
    *,
    embedding_id: str,
    object_id: str,
    values: tuple[float, ...],
    model: ModelReference,
    space: EmbeddingSpaceReference,
    tenant_id: TenantId = VECTOR_TENANT,
    object_type: EmbeddedObjectType = EmbeddedObjectType.MEMORY_RECORD,
    created_at: datetime = NOW,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(embedding_id),
        tenant_id=tenant_id,
        object_type=object_type,
        object_id=object_id,
        values=values,
        model_reference=model,
        space_reference=space,
        task="retrieval_document",
        dimension=1_024,
        normalized=True,
        created_at=created_at,
    )


async def test_reading_a_clip_keeps_its_provenance_link(
    store: PostgresMemoryStore,
) -> None:
    """A derived clip read back must still know which raw object it was cut from."""
    tenant_id = TenantId("tenant_provenance")
    source = MediaObject(
        media_object_id=MediaObjectId("provenance_source_01"),
        tenant_id=tenant_id,
        kind=MediaKind.VIDEO,
        uri="s3://memory/tenants/tenant_provenance/source.mp4",
        sha256="7" * 64,
        size_bytes=8_192,
        created_at=NOW,
    )
    clip = MediaObject(
        media_object_id=MediaObjectId("provenance_clip_01"),
        tenant_id=tenant_id,
        kind=MediaKind.VIDEO,
        uri="s3://memory/tenants/tenant_provenance/clips/" + "8" * 64 + ".mp4",
        sha256="8" * 64,
        size_bytes=1_024,
        created_at=NOW,
        duration_ms=3_000,
        derived_from_media_object_id=source.media_object_id,
    )
    async with tenant_connection(store._pool, tenant_id) as connection:
        for media_object in (source, clip):
            await write_media_object(connection, media_object)

    read_back = await store.read_media_objects(
        tenant_id, (source.media_object_id, clip.media_object_id)
    )

    assert read_back[0].derived_from_media_object_id is None
    assert read_back[1].derived_from_media_object_id == source.media_object_id
    assert read_back[1].duration_ms == 3_000


async def test_known_clip_digests_separate_orphans_from_registered_clips(
    store: PostgresMemoryStore,
) -> None:
    """An upload whose transaction rolled back has no row, so its digest is unknown."""
    tenant_id = TenantId("tenant_clips")
    registered_digest = "d" * 64
    orphan_digest = "e" * 64
    source = MediaObject(
        media_object_id=MediaObjectId("clip_source_01"),
        tenant_id=tenant_id,
        kind=MediaKind.AUDIO,
        uri="s3://memory/tenants/tenant_clips/source.wav",
        sha256="c" * 64,
        size_bytes=2_048,
        created_at=NOW,
    )
    registered = MediaObject(
        media_object_id=MediaObjectId("clip_registered_01"),
        tenant_id=tenant_id,
        kind=MediaKind.AUDIO,
        uri=f"s3://memory/tenants/tenant_clips/clips/{registered_digest}.wav",
        sha256=registered_digest,
        size_bytes=512,
        created_at=NOW,
        duration_ms=1_000,
        derived_from_media_object_id=source.media_object_id,
    )
    async with tenant_connection(store._pool, tenant_id) as connection:
        for media_object in (source, registered):
            await write_media_object(connection, media_object)

    known = await store.list_known_clip_digests(tenant_id, (registered_digest, orphan_digest))

    assert known == frozenset({registered_digest})
