from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from mindbridge.infrastructure.local import scoring
from mindbridge.infrastructure.local.scoring import max_cosine_scores


def test_max_cosine_scores_normalizes_inputs_and_preserves_negative_confidence() -> None:
    scores = max_cosine_scores(
        ((2.0, 0.0),),
        (
            ("positive", (30.0, 0.0)),
            ("negative", (-4.0, 0.0)),
        ),
    )

    assert scores == {
        "positive": (1.0, 1.0),
        "negative": (0.0, 0.0),
    }


def test_max_cosine_scores_uses_the_best_query_and_part_for_each_parent() -> None:
    scores = max_cosine_scores(
        ((1.0, 0.0), (0.0, 3.0)),
        (
            ("first", (-1.0, 0.0)),
            ("first", (0.0, 7.0)),
            ("second", (1.0, 1.0)),
        ),
    )

    assert scores["first"] == pytest.approx((1.0, 1.0))
    expected = 1.0 / math.sqrt(2.0)
    assert scores["second"] == pytest.approx((expected, (1.0 + expected) / 2.0))


@pytest.mark.parametrize(
    ("queries", "documents"),
    (
        (((0.0, 0.0),), (("memory", (1.0, 0.0)),)),
        (((1.0, 0.0),), (("memory", (0.0, 0.0)),)),
        (((1.0, 0.0),), (("memory", (1.0,)),)),
        (((1.0, math.inf),), (("memory", (1.0, 0.0)),)),
        (((1.0, 0.0),), (("memory", (1.0, math.nan)),)),
    ),
)
def test_max_cosine_scores_rejects_invalid_vectors(
    queries: tuple[tuple[float, ...], ...],
    documents: tuple[tuple[str, tuple[float, ...]], ...],
) -> None:
    with pytest.raises(ValueError, match="Cosine scoring requires"):
        max_cosine_scores(queries, documents)


def test_max_cosine_scores_does_not_consume_documents_for_an_empty_query() -> None:
    def documents() -> Iterator[tuple[str, tuple[float, ...]]]:
        raise AssertionError("documents should not be consumed")
        yield "memory", (1.0, 0.0)

    assert max_cosine_scores((), documents()) == {}


def test_python_310_fallback_preserves_the_public_scoring_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scoring, "_dot_product", scoring._fallback_dot_product)

    scores = max_cosine_scores(
        ((2.0, 0.0), (0.0, -5.0)),
        (
            ("multi-part", (-3.0, 0.0)),
            ("multi-part", (0.0, -11.0)),
            ("negative", (-7.0, 0.0)),
        ),
    )

    assert scores == {
        "multi-part": (1.0, 1.0),
        "negative": (0.0, 0.5),
    }
