"""Edge cases checked against Mem-Gallery's a93959e1e978 evaluator."""

from __future__ import annotations

import pytest

from mindbridge.benchmarks.official_scorers import local_scores


def _scores(
    prediction: str,
    reference: str,
    clues: tuple[str, ...] = (),
    retrieved: tuple[str, ...] = (),
) -> dict[str, float]:
    return local_scores(
        "mem-gallery",
        score_kind="text",
        prediction=prediction,
        references=(reference,),
        parsed_choice=None,
        expected_choice=None,
        question="",
        metadata={"clue_ids": clues},
        evidence_source_ids=retrieved,
    )


@pytest.mark.parametrize("retrieved", [(), ("r1",)])
def test_empty_clues_contribute_zero_to_official_retrieval_means(
    retrieved: tuple[str, ...],
) -> None:
    scores = _scores("Not mentioned.", "Not mentioned.", retrieved=retrieved)
    assert {name: score for name, score in scores.items() if name.startswith("retrieval_")} == {
        "retrieval_precision@10": 0.0,
        "retrieval_recall@10": 0.0,
        "retrieval_hit@10": 0.0,
    }


def test_retrieval_truncates_before_deduplicating() -> None:
    scores = _scores("yes", "yes", ("gold",), ("miss",) * 10 + ("gold",))
    assert scores["retrieval_recall@10"] == 0.0


@pytest.mark.parametrize(
    ("prediction", "reference", "f1", "exact_match"),
    [
        ("the\u0301", "\u0301", 0.0, 0.0),
        ("3.14, IMG_001 and IMG_002", "3.14 IMG_001 IMG_002", 1.0, 1.0),
        ("Yes—Leo says he has learned guitar before.", "Yes.", 0.0, 0.0),
        ("running running", "runs", 2 / 3, 0.0),
        ("[insufficient_evidence]", "Not mentioned.", 0.0, 0.0),
        ("", "", 0.0, 1.0),
    ],
)
def test_answer_normalization_preserves_official_behavior(
    prediction: str, reference: str, f1: float, exact_match: float
) -> None:
    scores = _scores(prediction, reference)
    assert scores["f1"] == pytest.approx(f1)
    assert scores["exact_match"] == exact_match
