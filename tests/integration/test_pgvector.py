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
    ModelReference,
    TenantId,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
VECTOR_TENANT = TenantId("tenant_vectors")

pytestmark = pytest.mark.integration


async def test_pgvector_mix_aligned_encoders_but_separates_space_revisions(
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
        model=ModelReference(
            model_id="jinaai/jina-embeddings-v5-text-small-retrieval",
            revision="text-revision",
        ),
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
