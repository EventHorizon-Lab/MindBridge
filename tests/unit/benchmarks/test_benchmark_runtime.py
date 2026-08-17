"""Checks for shared production benchmark runtime behavior."""

import pytest

from mindbridge.benchmarks.runtime import (
    benchmark_tenant_id,
    multiple_choice_query,
    parse_option_ranking,
)


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
