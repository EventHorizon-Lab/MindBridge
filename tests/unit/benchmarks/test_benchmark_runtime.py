"""Checks for shared production benchmark runtime behavior."""

from mindbridge.benchmarks import multiple_choice_query, parse_option_ranking


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
