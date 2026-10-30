from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from mindbridge.infrastructure.local import scoring
from mindbridge.infrastructure.local.scoring import max_cosine_scores


def test_exact_part_scores_preserve_the_winner_and_prefer_an_equal_existing_match() -> None:
    result = scoring.max_cosine_part_scores(
        ((1.0, 0.0),),
        (
            ("parent", "approximate", (0.0, 1.0)),
            ("parent", "precise", (2.0, 0.0)),
            ("tied", "first", (1.0, 0.0)),
            ("tied", "preferred", (3.0, 0.0)),
            ("negative", "opposite", (-1.0, 0.0)),
        ),
        preferred_embeddings={"tied": "preferred"},
    )
    assert result == {
        "parent": ("precise", 1.0, 1.0),
        "tied": ("preferred", 1.0, 1.0),
        "negative": ("opposite", 0.0, 0.0),
    }


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


@pytest.mark.parametrize("scale", (5e-324, 1e308))
def test_cosine_scores_do_not_depend_on_finite_vector_magnitude(scale: float) -> None:
    scores = max_cosine_scores(
        ((scale, 0.0),),
        (("diagonal", (scale, scale)), ("opposite", (-scale, 0.0))),
    )
    expected = 1.0 / math.sqrt(2.0)
    assert scores["diagonal"] == pytest.approx((expected, (1.0 + expected) / 2.0))
    assert scores["opposite"] == (0.0, 0.0)


def test_cosine_accepts_finite_vectors_whose_unscaled_norm_overflows() -> None:
    scores = max_cosine_scores(
        ((1e308, 1e308, 1e308, 1e308),),
        (("axis", (1e308, 0.0, 0.0, 0.0)),),
    )
    assert scores["axis"] == pytest.approx((0.5, 0.75))


def test_batched_scoring_keeps_parent_maxima_across_batches_and_extreme_scales() -> None:
    queries = tuple(tuple(float(component == axis) for component in range(64)) for axis in range(4))
    outside = (0.0,) * 4 + (1.0,) + (0.0,) * 59
    documents = (
        *(("parent", outside) for _ in range(128)),
        ("parent", (5e-324,) + (0.0,) * 63),
        ("diagonal", (1e308,) * 4 + (0.0,) * 60),
        ("negative", (-1.0,) * 4 + (0.0,) * 60),
        *((f"tail-{number}", outside) for number in range(125)),
    )
    scores = max_cosine_scores(queries, iter(documents))
    assert scores["parent"] == pytest.approx((1.0, 1.0))
    assert scores["diagonal"] == pytest.approx((0.5, 0.75))
    assert scores["negative"] == pytest.approx((0.0, 0.25))
    assert scores["tail-124"] == (0.0, 0.5)


@pytest.mark.parametrize(
    "invalid",
    ((0.0,) * 64, (math.nan,) + (1.0,) * 63, (math.inf,) + (1.0,) * 63, (1.0,) * 63),
)
def test_multikey_batches_reject_invalid_vectors(invalid: tuple[float, ...]) -> None:
    queries = tuple(tuple(float(component == axis) for component in range(64)) for axis in range(4))
    documents = (("invalid", invalid), *((str(i), (1.0,) * 64) for i in range(127)))
    with pytest.raises(ValueError, match="Cosine scoring requires"):
        max_cosine_scores(queries, iter(documents))


def test_python_311_fallback_preserves_the_public_scoring_contract(
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
