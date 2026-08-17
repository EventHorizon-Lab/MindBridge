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


async def test_pgvector_separates_space_revisions(
    store: PostgresMemoryStore,
) -> None:
    """The cloud index retrieves only the requested frozen model version."""
    model = ModelReference(
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        revision="abcdef0",
    )
    space = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
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
    other_revision = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=VECTOR_TENANT,
            values=first.values,
            space_reference=EmbeddingSpaceReference(space_id=space.space_id, revision="different"),
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
    assert other_revision == ()


async def test_unreachable_probe_is_scoped_per_tenant_and_per_object_type(
    store: PostgresMemoryStore,
) -> None:
    """A reachable memory record must not vouch for evidence stranded in another space."""
    # A dedicated tenant: this writes into the session-scoped database that every other
    # integration test shares, and tenant_vectors already owns fixture vectors.
    tenant_id = TenantId("tenant_space_probe")
    space = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")
    drifted = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v2")
    model = ModelReference(model_id="jina-omni", revision="abcdef0")

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
