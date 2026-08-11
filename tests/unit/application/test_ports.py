"""Tests for validated external-boundary values."""

import pytest

from mindbridge.application import EmbeddingSearch
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    ModelReference,
    TenantId,
)


def test_embedding_search_requires_unique_object_types() -> None:
    """Duplicate filters cannot inflate or ambiguate semantic retrieval."""
    with pytest.raises(DomainInvariantError, match="non-empty and unique"):
        EmbeddingSearch(
            tenant_id=TenantId("tenant_01"),
            values=(1.0,),
            model_reference=ModelReference(model_id="model", revision="revision"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.EVENT, EmbeddedObjectType.EVENT),
            limit=20,
        )


def test_embedding_search_rejects_invalid_similarity_threshold() -> None:
    with pytest.raises(DomainInvariantError, match="minimum_similarity"):
        EmbeddingSearch(
            tenant_id=TenantId("tenant_01"),
            values=(1.0,),
            model_reference=ModelReference(model_id="model", revision="revision"),
            document_task="retrieval_document",
            object_types=(EmbeddedObjectType.EVENT,),
            limit=20,
            minimum_similarity=1.1,
        )
