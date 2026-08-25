"""Video-MME-v2 grouped non-linear scoring, checked against the released evaluator.

The rating is the number this benchmark exists to report, and reimplementing it is the one
place a plausible-looking mistake would be invisible: every cell would still be a float in
range, and only a side-by-side comparison with the official code shows a wrong one. So the
released scorer is vendored below verbatim and the adapter is compared against it over every
outcome a group can have, rather than over a few hand-picked cases.
"""

from __future__ import annotations

import ast
import itertools

import pytest

from mindbridge.benchmarks.video_mme_v2 import (
    GROUP_SIZE,
    VideoMMEV2GroupResult,
    VideoMMEV2GroupType,
    VideoMMEV2Level,
    VideoMMEV2QuestionResult,
    evaluate_video_mme_v2,
    logic_structure,
    parse_video_mme_v2_option,
    score_group,
)

_STRUCTURES = ("[1,2,3,4]", "[1,[2,3],4]", "[[1,2],3,4]")
_OUTCOMES = tuple(itertools.product((0, 1), repeat=GROUP_SIZE))


# ---------------------------------------------------------------------------
# Vendored verbatim from MME-Benchmarks/Video-MME-v2 evaluation/test_video_mme_v2.py.
# Do not tidy: its value is being an independent copy of the released arithmetic, so any
# edit that makes it agree with the adapter for the wrong reason destroys the test.
# ---------------------------------------------------------------------------
def _official_relevance_rating(scores: list[int]) -> float:
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0)


def _official_logic_rating(scores: list[int], group_structure: str) -> float:
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break
    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"Unknown group_structure_list: {group_structure_list}")
    return score_map.get(last_correct_idx + 1, 0.0)


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_relevance_group_score_matches_the_released_scorer(outcome: tuple[int, ...]) -> None:
    """Every way four relevance questions can land must score what the release says."""
    result = _group_result("relevance", "4", outcome)

    assert score_group(result) == pytest.approx(_official_relevance_rating(list(outcome)))


@pytest.mark.parametrize("structure", _STRUCTURES)
@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_logic_group_score_matches_the_released_scorer(
    structure: str, outcome: tuple[int, ...]
) -> None:
    """All 48 chain outcomes, including the two parallel-pair adjustments."""
    result = _group_result("logic", structure, outcome)

    assert score_group(result) == pytest.approx(_official_logic_rating(list(outcome), structure))


def test_scoring_is_non_linear_in_the_way_the_benchmark_intends() -> None:
    """Two half-right groups must score below one whole-right group plus one wrong.

    This is the property the benchmark was rebuilt around, so it is asserted directly rather
    than left implicit in the table above: it is what the vendored comparison would still pass
    if both implementations were linear.
    """
    scattered = (
        _group_result("relevance", "4", (1, 1, 0, 0)),
        _group_result("relevance", "4", (1, 1, 0, 0)),
    )
    concentrated = (
        _group_result("relevance", "4", (1, 1, 1, 1)),
        _group_result("relevance", "4", (0, 0, 0, 0)),
    )

    scattered_metrics = evaluate_video_mme_v2(scattered)
    concentrated_metrics = evaluate_video_mme_v2(concentrated)

    assert scattered_metrics.accuracy.overall == concentrated_metrics.accuracy.overall
    assert scattered_metrics.rating.overall == pytest.approx(25.0)
    assert concentrated_metrics.rating.overall == pytest.approx(50.0)


def test_logic_chain_denies_credit_for_a_correct_tail_after_a_broken_head() -> None:
    """Answering 2-4 right after missing question 1 earns the chain's floor, not most of it."""
    broken_head = _group_result("logic", "[1,2,3,4]", (0, 1, 1, 1))

    assert score_group(broken_head) == pytest.approx(0.0)


def test_parallel_pair_credits_a_sibling_but_does_not_resume_the_chain() -> None:
    """The released adjustments are one-shot; reproducing that is the point."""
    sibling_only = _group_result("logic", "[[1,2],3,4]", (0, 1, 1, 1))

    assert score_group(sibling_only) == pytest.approx(100.0 / 10)


def test_rating_cells_key_on_the_last_question_of_each_group() -> None:
    """`level`, `second_head` and `third_head` vary inside a group; the release reads `group[-1]`.

    A run keyed on the first question would produce a plausible, differently-wrong radar, so
    the fixture deliberately gives its four questions four different levels.
    """
    result = _group_result(
        "relevance",
        "4",
        (1, 1, 1, 1),
        levels=("1", "2", "3", "3"),
        second_heads=("Order", "Change", "Order", "Action & Motion"),
    )

    metrics = evaluate_video_mme_v2((result,))

    assert metrics.rating.by_level == {"level_3": pytest.approx(100.0)}
    assert metrics.rating.by_second_head == {"Action & Motion": pytest.approx(100.0)}
    # The accuracy view keys per question instead, so all four cells survive there.
    assert set(metrics.accuracy.by_level) == {"level_1", "level_2", "level_3"}
    assert set(metrics.accuracy.by_second_head) == {"Order", "Change", "Action & Motion"}


def test_unanswered_questions_are_wrong_everywhere_but_the_answered_only_figure() -> None:
    """The release folds an unparseable response into zero for both artifacts it writes."""
    result = _group_result("relevance", "4", (1, 1, 0, 0), unanswered=(2, 3))

    metrics = evaluate_video_mme_v2((result,))

    assert metrics.accuracy.question_count == 4
    assert metrics.accuracy.answered_count == 2
    assert metrics.accuracy.correct_count == 2
    assert metrics.accuracy.overall == pytest.approx(50.0)
    assert metrics.accuracy.answered_accuracy == pytest.approx(100.0)
    assert metrics.rating.overall == pytest.approx(100.0 * 4 / 16)


def test_error_count_may_not_exceed_the_unanswered_questions() -> None:
    """A question that carries an error code cannot also carry a parsed letter."""
    with pytest.raises(ValueError, match="must not carry a parsed answer"):
        evaluate_video_mme_v2((_group_result("relevance", "4", (1, 1, 1, 1), error_codes=(0,)),))


def test_group_counts_are_the_ratings_denominator() -> None:
    metrics = evaluate_video_mme_v2(
        (
            _group_result("relevance", "4", (1, 1, 1, 1), video_id="001"),
            _group_result("logic", "[1,2,3,4]", (0, 0, 0, 0), video_id="002"),
        )
    )

    assert metrics.rating.group_count == 2
    assert metrics.accuracy.question_count == 8
    assert metrics.rating.by_group_type == {
        "logic": pytest.approx(0.0),
        "relevance": pytest.approx(100.0),
    }


def test_rescoring_a_logic_group_with_an_unknown_structure_is_refused() -> None:
    """Results deserialized from a predictions file skip the annotation's own validation."""
    tampered = VideoMMEV2GroupResult.model_validate(
        _group_result("logic", "[1,2,3,4]", (1, 1, 1, 1)).model_dump()
        | {"group_structure": "[4,3,2,1]"}
    )

    with pytest.raises(ValueError, match="unknown structure"):
        score_group(tampered)


def test_evaluating_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_video_mme_v2(())


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The best answer is F.", "F"),
        ("Final Answer: H", "H"),
        ("Answer: A", "A"),
        ("G", "G"),
        ("Insufficient information to answer.", None),
        ("", None),
        (None, None),
    ],
)
def test_option_parsing_follows_the_released_first_letter_rule(
    response: str | None, expected: str | None
) -> None:
    assert parse_video_mme_v2_option(response) == expected


def test_option_parsing_reaches_the_letters_video_mme_never_had() -> None:
    """E through H are new in v2; a parser inherited from v1 would silently miss them."""
    assert [parse_video_mme_v2_option(letter) for letter in "EFGH"] == ["E", "F", "G", "H"]


def test_logic_structure_ignores_spacing_and_rejects_the_relevance_placeholder() -> None:
    """`relevance` groups carry the scalar `"4"` in the structure column."""
    assert logic_structure("[1, 2, 3, 4]") == "[1,2,3,4]"
    assert logic_structure("[1,[2,3],4]") == "[1,[2,3],4]"
    assert logic_structure("4") is None
    assert logic_structure("[9,9,9,9]") is None


def _group_result(
    group_type: VideoMMEV2GroupType,
    group_structure: str,
    outcome: tuple[int, ...],
    *,
    video_id: str = "001",
    levels: tuple[VideoMMEV2Level, ...] = ("1", "2", "3", "1"),
    second_heads: tuple[str, ...] = ("Order", "Order", "Order", "Order"),
    unanswered: tuple[int, ...] = (),
    error_codes: tuple[int, ...] = (),
) -> VideoMMEV2GroupResult:
    """Build a scored group whose questions land exactly as `outcome` says."""
    questions = tuple(
        VideoMMEV2QuestionResult(
            question_id=f"{video_id}-{position}",
            position=position,
            question=f"Question {position}?",
            options=("A. First.", "B. Second."),
            answer="A",
            level=levels[position - 1],
            second_head=second_heads[position - 1],
            third_head="Causal Reasoning",
            response=_response(position - 1, correct, unanswered),
            mindbridge_model_answer="",
            mindbridge_confidence=0.0,
            mindbridge_memory_ids=(),
            mindbridge_evidence_ids=(),
            mindbridge_trace_id=f"trace_{video_id}_{position}",
            mindbridge_error_code=("model_request_failed" if position - 1 in error_codes else None),
        )
        for position, correct in enumerate(outcome, start=1)
    )
    return VideoMMEV2GroupResult(
        video_id=video_id,
        group_type=group_type,
        group_structure=group_structure,
        questions=questions,
    )


def _response(index: int, correct: int, unanswered: tuple[int, ...]) -> str:
    if index in unanswered:
        return ""
    return "A" if correct else "B"
