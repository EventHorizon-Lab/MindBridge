"""Complete hybrid candidate scores from authoritative persisted vectors."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from operator import mul
from typing import cast


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

    A missing ANN hit is an unmeasured similarity, not a zero similarity. Hybrid
    retrieval can use this bounded completion step for parents discovered only
    by its lexical route. Every persisted part competes against every query key,
    preserving the index's maximum-over-parts semantics without model calls.
    """
    if not query_vectors:
        return {}

    dimension = len(query_vectors[0])
    queries = tuple(_unit_vector(vector, dimension) for vector in query_vectors)
    similarities: dict[str, float] = {}
    for memory_id, values in document_vectors:
        norm = _vector_norm(values, dimension)
        similarity = max(_dot_product(query, values) / norm for query in queries)
        # Floating-point dot products of unit vectors can slightly exceed one.
        similarity = min(1.0, max(-1.0, similarity))
        previous = similarities.get(memory_id)
        if previous is None or similarity > previous:
            similarities[memory_id] = similarity

    return {
        memory_id: (max(0.0, similarity), (1.0 + similarity) / 2.0)
        for memory_id, similarity in similarities.items()
    }


def _unit_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    norm = _vector_norm(values, dimension)
    return tuple(value / norm for value in values)


def _vector_norm(values: Sequence[float], dimension: int) -> float:
    if not dimension or len(values) != dimension:
        raise ValueError("Cosine scoring requires nonempty vectors of equal dimension")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Cosine scoring requires finite, nonzero vectors")
    return norm
