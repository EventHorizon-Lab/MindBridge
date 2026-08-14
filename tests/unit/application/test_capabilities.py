"""Invariant checks for application-owned model capabilities."""

import pytest

from mindbridge.application.capabilities import Embedding
from mindbridge.core import (
    DomainInvariantError,
    EmbeddingSpaceReference,
    ModelReference,
)


@pytest.mark.parametrize("values", [(0.0, 0.0), (2.0, 0.0)])
def test_embedding_requires_a_normalized_vector(values: tuple[float, ...]) -> None:
    with pytest.raises(DomainInvariantError, match="L2-normalized"):
        Embedding(
            values,
            ModelReference("test/model", "1"),
            EmbeddingSpaceReference("test-space", "1"),
        )
