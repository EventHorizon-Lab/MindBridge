"""Integration checks for the versioned pgvector index."""

from datetime import datetime, timezone

import pytest

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
    stored = _embedding_record(
        embedding_id="embedding_memory_reencode",
        object_id="memory_reencode",
        values=(1.0,) + (0.0,) * 1_023,
        model=model,
        space=space,
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
    # same object really is indexed under another version, and it still refuses.
    with pytest.raises(DomainInvariantError, match="different vector content"):
        await store.write_embedding(
            _embedding_record(
                embedding_id="embedding_memory_reencode_v2",
                object_id="memory_reencode",
                values=reencoded.values,
                model=model,
                space=space,
            ),
            allow_reencoding=True,
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
        created_at=NOW,
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
