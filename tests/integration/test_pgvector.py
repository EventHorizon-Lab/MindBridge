"""Integration checks for the versioned pgvector index."""

from datetime import datetime, timezone

import pytest

from mindbridge.application import EmbeddingSearch
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    ModelReference,
    TenantId,
)
from mindbridge.infrastructure import PostgresMemoryStore

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

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
            tenant_id=TenantId("tenant_vectors"),
            values=first.values,
            space_reference=space,
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )
    other_revision = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=TenantId("tenant_vectors"),
            values=first.values,
            space_reference=EmbeddingSpaceReference(space_id=space.space_id, revision="different"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.MEMORY_RECORD,),
            limit=2,
        )
    )
    thresholded = await store.search_embeddings(
        EmbeddingSearch(
            tenant_id=TenantId("tenant_vectors"),
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


def _embedding_record(
    *,
    embedding_id: str,
    object_id: str,
    values: tuple[float, ...],
    model: ModelReference,
    space: EmbeddingSpaceReference,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId(embedding_id),
        tenant_id=TenantId("tenant_vectors"),
        object_type=EmbeddedObjectType.MEMORY_RECORD,
        object_id=object_id,
        values=values,
        model_reference=model,
        space_reference=space,
        task="retrieval_document",
        dimension=1_024,
        normalized=True,
        created_at=NOW,
    )
