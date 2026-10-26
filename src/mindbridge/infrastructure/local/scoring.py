"""Score bounded candidate parents from authoritative persisted vectors."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import islice
from operator import mul
from sys import float_info
from typing import cast

import numpy as np
from numpy.typing import NDArray

_BATCH_SIZE = 128
_BATCH_MIN_QUERIES = 4
_BATCH_MIN_COMPONENTS = 8192


def _fallback_dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return cast(float, sum(map(mul, left, right)))


# Python 3.10 and 3.11 lack the standard-library native dot product.
_dot_product: Callable[[Sequence[float], Sequence[float]], float] = getattr(
    math, "sumprod", _fallback_dot_product
)


def max_cosine_scores(
    query_vectors: Sequence[Sequence[float]],
    document_vectors: Iterable[tuple[str, Sequence[float]]],
) -> dict[str, tuple[float, float]]:
    """Return parent relevance and confidence on the dense index's cosine scale.

    Each persisted part competes against every query key. The caller can complete a
    missing lexical candidate score or replace a quantized score without model calls.
    Small workloads retain the native scalar dot product; multiple query keys share
    a bounded NumPy batch without invoking an uncontrolled BLAS thread pool.
    """
    measured = _max_cosines(
        query_vectors, ((memory_id, None, values) for memory_id, values in document_vectors)
    )
    return {
        memory_id: (max(0.0, similarity), (1.0 + similarity) / 2.0)
        for memory_id, (_, similarity) in measured.items()
    }


def max_cosine_part_scores(
    query_vectors: Sequence[Sequence[float]],
    document_parts: Iterable[tuple[str, str, Sequence[float]]],
    *,
    preferred_embeddings: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, float, float]]:
    """Return each parent's exact winning embedding, relevance and confidence.

    Preserve the existing match on exact ties so equally good parts do not arbitrarily change
    compilation excerpts. Otherwise the input stream's stable part order breaks ties.
    """
    measured = _max_cosines(
        query_vectors, document_parts, preferred_embeddings=preferred_embeddings
    )
    return {
        memory_id: (cast(str, embedding_id), max(0.0, similarity), (1.0 + similarity) / 2.0)
        for memory_id, (embedding_id, similarity) in measured.items()
    }


def _max_cosines(
    query_vectors: Sequence[Sequence[float]],
    document_parts: Iterable[tuple[str, str | None, Sequence[float]]],
    *,
    preferred_embeddings: Mapping[str, str] | None = None,
) -> dict[str, tuple[str | None, float]]:
    if not query_vectors:
        return {}

    dimension = len(query_vectors[0])
    queries = tuple(_unit_vector(vector, dimension) for vector in query_vectors)
    similarities: dict[str, tuple[str | None, float]] = {}
    documents = iter(document_parts)
    query_matrix = (
        np.asarray(queries, dtype=np.float64) if len(queries) >= _BATCH_MIN_QUERIES else None
    )
    while batch := tuple(islice(documents, _BATCH_SIZE)):
        measured: Iterable[tuple[str, str | None, float]]
        if query_matrix is not None and len(batch) * dimension >= _BATCH_MIN_COMPONENTS:
            matrix = _normalized_matrix(tuple(values for _, _, values in batch), dimension)
            scores = np.einsum("ij,kj->ik", matrix, query_matrix, optimize=False).max(axis=1)
            measured = (
                (memory_id, embedding_id, float(score))
                for (memory_id, embedding_id, _), score in zip(batch, scores, strict=True)
            )
        else:
            measured = (
                (memory_id, embedding_id, _max_similarity(queries, values, dimension))
                for memory_id, embedding_id, values in batch
            )
        for memory_id, embedding_id, raw_similarity in measured:
            similarity = min(1.0, max(-1.0, float(raw_similarity)))
            previous = similarities.get(memory_id)
            if (
                previous is None
                or similarity > previous[1]
                or (
                    similarity == previous[1]
                    and preferred_embeddings is not None
                    and embedding_id == preferred_embeddings.get(memory_id)
                )
            ):
                similarities[memory_id] = (embedding_id, similarity)
    return similarities


def _unit_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    norm = _vector_norm(values, dimension)
    if norm < float_info.min or not math.isfinite(norm):
        # Scaling first preserves direction for subnormals and finite vectors whose norm
        # overflows. Dividing by their rounded raw norm would bias relevance thresholds.
        scale = max(map(abs, values))
        scaled = tuple(value / scale for value in values)
        norm = math.hypot(*scaled)
        return tuple(value / norm for value in scaled)
    return tuple(value / norm for value in values)


def _max_similarity(
    queries: Sequence[Sequence[float]], values: Sequence[float], dimension: int
) -> float:
    norm = _vector_norm(values, dimension)
    if norm < float_info.min or not math.isfinite(norm):
        unit = _unit_vector(values, dimension)
        return max(_dot_product(query, unit) for query in queries)
    return max(_dot_product(query, values) / norm for query in queries)


def _normalized_matrix(vectors: Sequence[Sequence[float]], dimension: int) -> NDArray[np.float64]:
    if not dimension or any(len(values) != dimension for values in vectors):
        raise ValueError("Cosine scoring requires nonempty vectors of equal dimension")
    matrix = np.asarray(vectors, dtype=np.float64)
    scales = np.max(np.abs(matrix), axis=1)
    if not np.isfinite(matrix).all() or (scales == 0).any():
        raise ValueError("Cosine scoring requires finite, nonzero vectors")
    matrix = matrix / scales[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, optimize=False))
    return np.asarray(matrix / norms[:, None], dtype=np.float64)


def _vector_norm(values: Sequence[float], dimension: int) -> float:
    if not dimension or len(values) != dimension:
        raise ValueError("Cosine scoring requires nonempty vectors of equal dimension")
    norm = math.hypot(*values)
    if norm == 0.0 or (
        not math.isfinite(norm) and any(not math.isfinite(value) for value in values)
    ):
        raise ValueError("Cosine scoring requires finite, nonzero vectors")
    return norm
