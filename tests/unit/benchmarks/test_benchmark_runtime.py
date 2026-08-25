"""Checks for shared production benchmark runtime behavior."""

import asyncio

import pytest

from mindbridge.benchmarks.runtime import (
    answer_failure_trace_id,
    benchmark_tenant_id,
    multiple_choice_query,
    parse_option_ranking,
    settle_answers,
)
from mindbridge.sdk import MindBridgeError


def test_benchmark_tenant_requires_an_isolated_run_id() -> None:
    assert benchmark_tenant_id("benchmark", "subject_1", "run_01") == ("benchmark_subject_1_run_01")
    with pytest.raises(ValueError, match="run_id"):
        benchmark_tenant_id("benchmark", "subject_1", " ")


def test_multiple_choice_query_contains_only_inference_inputs() -> None:
    query = multiple_choice_query(
        "Where is the mug?",
        ("On the table", "In the sink", "In a drawer", "Unknown"),
        rank_all=True,
    )

    assert "A. On the table" in query
    assert query.endswith("best to worst, separated by commas.")


def test_option_parser_accepts_constrained_outputs_and_rejects_prose() -> None:
    choices = ("On the table", "In the sink", "In a drawer", "Unknown")

    assert parse_option_ranking("Ranking: B, A, D, C", choices) == (1, 0, 3, 2)
    assert parse_option_ranking("In the sink", choices) == (1,)
    assert parse_option_ranking("B because the memory says so", choices) == ()
    assert parse_option_ranking("B, B, A, C", choices) == ()


def test_multiple_choice_runtime_supports_egomem_reason_a_to_j() -> None:
    choices = tuple(f"Choice {index}" for index in range(10))

    query = multiple_choice_query("Pick one", choices, rank_all=False)

    assert "J. Choice 9" in query
    assert parse_option_ranking("J", choices) == (9,)


def test_settle_answers_replaces_only_the_question_that_failed() -> None:
    outcomes = [
        "answered q0",
        MindBridgeError("gateway timeout", code="model_unavailable"),
        RuntimeError("connection dropped"),
    ]

    settled = settle_answers(
        ("q0", "q1", "q2"),
        outcomes,
        lambda question, code: f"{question} failed with {code}",
    )

    assert settled == (
        "answered q0",
        "q1 failed with model_unavailable",
        "q2 failed with RuntimeError",
    )


def test_settle_answers_refuses_to_score_a_cancelled_run() -> None:
    """A cancelled cohort is a run that ended, not a cohort of wrong answers."""
    with pytest.raises(asyncio.CancelledError):
        settle_answers(("q0",), [asyncio.CancelledError()], lambda question, code: "recorded")


def test_settle_answers_rejects_outcomes_that_do_not_line_up_with_their_questions() -> None:
    with pytest.raises(ValueError, match="shorter"):
        settle_answers(("q0", "q1"), ["answered q0"], lambda question, code: "recorded")


def test_answer_failure_trace_id_stays_inside_the_identifier_limit() -> None:
    assert len(answer_failure_trace_id("q" * 400)) == 255
