"""Invariant checks for application-owned model capabilities."""

import pytest

from mindbridge.application.capabilities import Embedder, Embedding, EmbedRequest, EmbedResult
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
            ModelReference("test/model"),
            EmbeddingSpaceReference("test-space"),
        )


def test_an_embedder_without_a_space_fails_the_capability_check() -> None:
    """A structural isinstance cannot reject an explicit subclass, so the member must not be None."""

    class Incomplete(Embedder):
        async def embed(self, request: EmbedRequest) -> EmbedResult:
            return EmbedResult(())

    # mypy rejects this class too; the runtime guard covers plugins published without type checking.
    incomplete = Incomplete()  # type: ignore[abstract]
    with pytest.raises(NotImplementedError, match="declare its embedding space"):
        isinstance(incomplete, Embedder)
