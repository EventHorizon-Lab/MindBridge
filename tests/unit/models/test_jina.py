"""Tests for the Jina Sentence Transformers boundary."""

import pytest

from mindbridge.core import ModelOutputError, ModelReference
from mindbridge.models.jina import JinaOmniEmbedder


class Matrix:
    """Minimal ndarray-shaped result used by the adapter boundary."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class RecordingEncoder:
    """Records whether query/document semantics reach the official API."""

    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.calls: list[str] = []

    def encode_query(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("query")
        return Matrix(self.values)

    def encode_document(
        self,
        sentences: list[object],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int,
    ) -> Matrix:
        self.calls.append("document")
        return Matrix(self.values)


async def test_jina_uses_distinct_query_and_document_methods() -> None:
    """Retrieval prefixes remain owned by Sentence Transformers."""
    encoder = RecordingEncoder([[1.0, 0.0]])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    query = await embedder.encode_queries(("Where is the screwdriver?",))
    document = await embedder.encode_documents((b"image-bytes",))

    assert query == ((1.0, 0.0),)
    assert document == query
    assert encoder.calls == ["query", "document"]


async def test_jina_rejects_invalid_model_output() -> None:
    """Malformed upstream vectors cannot enter the semantic index."""
    embedder = JinaOmniEmbedder(
        RecordingEncoder([[1.0, 1.0]]),
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    with pytest.raises(ModelOutputError, match="L2-normalized"):
        await embedder.encode_queries(("query",))


async def test_jina_skips_model_call_for_empty_batch() -> None:
    """An empty batch has no model cost or ambiguous output shape."""
    encoder = RecordingEncoder([])
    embedder = JinaOmniEmbedder(
        encoder,
        ModelReference(model_id="jina", revision="revision"),
        dimension=2,
    )

    assert await embedder.encode_documents(()) == ()
    assert encoder.calls == []
