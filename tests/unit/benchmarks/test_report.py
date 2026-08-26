"""What the results table prints, and where each number in it came from.

The sweep's own behaviour is covered where the sweep is. What is checked here is the rendering:
that a figure is attributed to the artifact claiming it, that a benchmark scored by four
different shapes of metric all read, and that a task with no numbers says so rather than
printing an empty row that reads like a zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.report import METRIC_ROW_LIMIT, render, render_directory

SUMMARY_FILENAME = "suite-summary.json"


def test_a_runner_that_scores_itself_has_its_numbers_attributed_to_the_runner(
    tmp_path: Path,
) -> None:
    """Four benchmarks here are exact-match, so their own runner pins the score in the manifest."""
    _task(tmp_path, "video-mme", manifest={"metrics": {"accuracy": 0.6129}})

    table = render(_summary(tmp_path, "video-mme"), directory=tmp_path)

    assert "accuracy" in table
    assert "0.6129" in table
    assert "runner" in table
    assert "official" not in table


def test_an_external_scorers_numbers_are_attributed_to_the_scorer(tmp_path: Path) -> None:
    """Most benchmarks here are scored outside MindBridge; a table must not blur the two."""
    _task(tmp_path, "locomo-refined", score={"metrics": {"llm_judge": 0.581}})

    table = render(_summary(tmp_path, "locomo-refined"), directory=tmp_path)

    assert "llm_judge" in table
    assert "official" in table


def test_both_sources_appear_on_one_task_without_being_confused_for_each_other(
    tmp_path: Path,
) -> None:
    """Video-MME scores itself and can also be scored officially; the two are different claims."""
    _task(
        tmp_path,
        "video-mme",
        manifest={"metrics": {"accuracy": 0.61}},
        score={"metrics": {"official_accuracy": 0.59}},
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert [row for row in rows if "accuracy " in row and "official" not in row]
    assert [row for row in rows if "official_accuracy" in row and row.endswith("official")]


def test_a_task_with_no_numbers_says_so_and_names_the_command_that_would_give_it_some(
    tmp_path: Path,
) -> None:
    """An empty cell reads like a zero, which is the one thing a results table must never do."""
    _task(tmp_path, "memlens-32k")

    table = render(_summary(tmp_path, "memlens-32k"), directory=tmp_path)

    # The row itself, not the footer: a bare dash in the source column reads as "no such thing"
    # rather than "no number yet", and the footer's advice would then be the only sign of either.
    assert "not scored" in _rows(table)[0]
    assert "mindbridge-bench score --predictions" in table
    assert "memlens-32k/predictions.jsonl" in table


def test_a_failed_task_is_not_reported_as_merely_unscored(tmp_path: Path) -> None:
    """It has no numbers for a different reason, and telling someone to score it is wrong."""
    _task(tmp_path, "atm-main")

    table = render(_summary(tmp_path, "atm-main", status="failed"), directory=tmp_path)

    assert "failed" in table
    assert "not scored" not in table


def test_the_headline_figure_comes_before_the_counts_it_was_computed_from(tmp_path: Path) -> None:
    """Ordered as declared, `accuracy` sat on the sixth row of its own task behind four counts."""
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "metrics": {
                "question_count": 900,
                "answered_count": 881,
                "correct_count": 540,
                "accuracy": 0.6129,
            }
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert "accuracy" in rows[0]
    assert "question_count" in rows[1]


def test_a_breakdown_is_labelled_by_its_full_path_so_it_reads_as_a_cell_not_a_total(
    tmp_path: Path,
) -> None:
    """`0.49` under a bare `accuracy` would be indistinguishable from the overall figure."""
    _task(
        tmp_path,
        "video-mme",
        manifest={"metrics": {"accuracy": 0.61, "by_duration": {"long": {"accuracy": 0.49}}}},
    )

    table = render(_summary(tmp_path, "video-mme"), directory=tmp_path)

    assert "by_duration.long.accuracy" in table


def test_a_list_of_scored_groups_is_keyed_by_the_label_each_group_carries(tmp_path: Path) -> None:
    """EgoLifeQA pins its per-category metrics as a list, and `categories.0` names nothing."""
    _task(
        tmp_path,
        "egolife",
        manifest={
            "metrics": {
                "categories": [
                    {"question_type": "EventRecall", "accuracy": 0.83},
                    {"question_type": "HabitInsight", "accuracy": 0.58},
                ]
            }
        },
    )

    table = render(_summary(tmp_path, "egolife"), directory=tmp_path)

    assert "categories.EventRecall.accuracy" in table
    assert "categories.HabitInsight.accuracy" in table


def test_a_denominator_repeated_inside_every_taxonomy_cell_is_not_printed(tmp_path: Path) -> None:
    """Video-MME-v2 pins four taxonomies twice over; their counts would bury the figures."""
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "metrics": {
                "question_count": 900,
                "by_duration": {"long": {"question_count": 300, "accuracy": 0.49}},
            }
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert any("question_count" in row for row in rows), "the run's own total belongs in the table"
    assert not any("by_duration.long.question_count" in row for row in rows)


def test_too_many_metrics_are_capped_with_a_count_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    """A terminal summary forty rows long for one task is a file, not a summary.

    What the cap drops has to be visible, and it has to be a breakdown: the headline sits at the
    top level and the cells below it, so ordering by depth is what makes that true.
    """
    cells = {f"cell_{index}": index / 100 for index in range(METRIC_ROW_LIMIT + 5)}
    _task(tmp_path, "video-mme", manifest={"metrics": {"overall": 0.5, "by_head": cells}})
    dropped = 1 + len(cells) - METRIC_ROW_LIMIT

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert "overall" in rows[0], "the headline must survive the cap"
    assert len(rows) == METRIC_ROW_LIMIT + 1, "the shown metrics, plus the row saying what is not"
    assert f"+{dropped} more" in rows[-1]


def test_a_results_directory_moved_off_the_machine_that_produced_it_still_renders(
    tmp_path: Path,
) -> None:
    """A scored run usually reaches its reader by being copied, which breaks every recorded path."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _task(elsewhere, "memlens-32k", manifest={"metrics": {"accuracy": 0.4}})
    summary = _summary(Path("/gone/results/sweep-01"), "memlens-32k")
    (elsewhere / SUMMARY_FILENAME).write_text(summary, encoding="utf-8")

    table = render_directory(elsewhere)

    assert "0.4000" in table


def test_a_malformed_artifact_costs_its_own_row_and_not_the_whole_table(tmp_path: Path) -> None:
    """The run that wrote a truncated manifest may be the one that died; the rest are why we look."""
    _task(tmp_path, "broken")
    (tmp_path / "broken" / "predictions.jsonl.manifest.json").write_text("{ne", encoding="utf-8")
    _task(tmp_path, "fine", manifest={"metrics": {"accuracy": 0.9}})

    table = render(_summary(tmp_path, "broken", "fine"), directory=tmp_path)

    assert "0.9000" in table
    assert "not scored" in table


def test_a_directory_holding_no_sweep_summary_says_which_file_it_wanted(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=SUMMARY_FILENAME):
        render_directory(tmp_path)


def test_a_count_prints_as_a_count_and_a_rate_as_a_rate(tmp_path: Path) -> None:
    """A run of 900 questions reported as `900.0000` is a rate; 0.61 reported as `0.6` is rounded."""
    _task(tmp_path, "video-mme", manifest={"metrics": {"question_count": 900, "accuracy": 0.6129}})

    table = render(_summary(tmp_path, "video-mme"), directory=tmp_path)

    assert " 900 " in table or table.count(" 900\n") or "  900  " in table
    assert "900.0000" not in table
    assert "0.6129" in table


def _task(
    directory: Path,
    name: str,
    *,
    manifest: dict[str, object] | None = None,
    score: dict[str, object] | None = None,
) -> Path:
    """Write what one task leaves behind: predictions, and whichever sidecars it has."""
    (directory / name).mkdir(parents=True, exist_ok=True)
    predictions = directory / name / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    if manifest is not None:
        (directory / name / "predictions.jsonl.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    if score is not None:
        (directory / name / "predictions.jsonl.score.json").write_text(
            json.dumps(score), encoding="utf-8"
        )
    return predictions


def _summary(directory: Path, *names: str, status: str = "completed") -> str:
    """Serialize a sweep summary naming those tasks, as the sweep itself writes one."""
    return json.dumps(
        {
            "run_id": "sweep-01",
            "tasks": [
                {
                    "name": name,
                    "benchmark": name.title(),
                    "run_id": f"sweep-01-{name}",
                    "output_path": str(directory / name / "predictions.jsonl"),
                    "arguments": ["--dataset", "release.json"],
                    "status": status,
                    "exit_code": 0 if status == "completed" else 1,
                    "duration_seconds": 61.0,
                }
                for name in names
            ],
        }
    )


def _rows(table: str) -> list[str]:
    """The metric rows of a rendered table: everything between the rule and the blank line."""
    lines = table.splitlines()
    body = lines[lines.index(next(line for line in lines if line.startswith("─"))) + 1 :]
    return list(body[: body.index("")])


def test_a_declared_metric_carries_the_arrow_saying_which_way_it_points(tmp_path: Path) -> None:
    """lmms-eval prints `↑`/`↓` beside the metric; a bare number does not say what good means."""
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "scoring": {
                "mode": "runner",
                "metrics": {"accuracy": 0.61, "error_rate": 0.04},
                "higher_is_better": {"accuracy": True, "error_rate": False},
            }
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert [row for row in rows if "accuracy" in row and "↑" in row]
    assert [row for row in rows if "error_rate" in row and "↓" in row]


def test_a_count_gets_no_arrow_even_where_a_direction_was_declared(tmp_path: Path) -> None:
    """`question_count` rising says the run covered more of the release, not that it did better."""
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "scoring": {
                "mode": "runner",
                "metrics": {"question_count": 900},
                "higher_is_better": {"question_count": True},
            }
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    assert "question_count" in rows[0]
    assert "↑" not in rows[0]


def test_a_judged_number_is_not_labelled_as_an_exact_match(tmp_path: Path) -> None:
    """The judge model was MindBridge's choice, which a reader has to see from the table."""
    _task(
        tmp_path,
        "locomo-refined",
        manifest={
            "scoring": {
                "mode": "judge",
                "metrics": {"llm_judge": 0.58},
                "judge_model": "gpt-4o-2024-11-20",
            }
        },
    )

    rows = _rows(render(_summary(tmp_path, "locomo-refined"), directory=tmp_path))

    assert rows[0].endswith("judge")
    assert "runner" not in rows[0]


def test_a_judge_that_could_not_be_read_is_named_under_the_table(tmp_path: Path) -> None:
    """The floored answers are 0.0 in the mean, and nothing in the column can show that."""
    _task(
        tmp_path,
        "locomo-refined",
        manifest={
            "scoring": {
                "mode": "judge",
                "metrics": {"llm_judge": 0.31},
                "judge_model": "gpt-4o-2024-11-20",
                "judge_failure_count": 47,
            }
        },
    )

    table = render(_summary(tmp_path, "locomo-refined"), directory=tmp_path)

    assert "47 answers scored 0.0" in table
    assert "a floor, not a measurement" in table
    assert "gpt-4o-2024-11-20" in table


def test_the_bypass_sentinel_is_called_a_sentinel_rather_than_left_to_look_like_a_score(
    tmp_path: Path,
) -> None:
    """999 sits in the same column as an accuracy, which is exactly why it needs saying."""
    _task(
        tmp_path,
        "memlens-32k",
        manifest={"scoring": {"mode": "bypass", "metrics": {"bypass": 999.0}}},
    )

    table = render(_summary(tmp_path, "memlens-32k"), directory=tmp_path)

    assert "999" in table
    assert "--predict-only" in table
    assert "has not been evaluated" in table


def test_a_predict_only_run_does_not_label_a_measured_number_as_the_sentinel(
    tmp_path: Path,
) -> None:
    """`--predict-only` suppresses the declared metric; the runner still pins its own breakdown.

    Labelling both blocks with the one `scoring.mode` printed a real measured accuracy sourced
    `bypass`, underneath a footer saying the run had not been evaluated. The runner's arithmetic
    is the runner's whatever scored the headline.
    """
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "scoring": {"mode": "bypass", "metrics": {"bypass": 999.0}},
            "metrics": {"accuracy": 0.61},
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    sentinel = [row for row in rows if "bypass" in row]
    measured = [row for row in rows if "accuracy" in row]
    assert sentinel and sentinel[0].strip().endswith("bypass")
    assert measured and measured[0].strip().endswith("runner"), measured


def test_one_unreadable_number_does_not_take_the_whole_table_with_it(tmp_path: Path) -> None:
    """`json.loads` accepts bare NaN, which a 0/0 category in a hand-made sidecar carries.

    `int(nan)` raises, so one cell aborted `render` before it printed anything -- the failure
    `_load` is deliberately defensive about, one step further on.
    """
    _task(
        tmp_path,
        "video-mme",
        manifest={"scoring": {"mode": "runner", "metrics": {"accuracy": 0.61}}},
    )
    sidecar = tmp_path / "video-mme" / "predictions.jsonl.score.json"
    sidecar.write_text('{"metrics": {"strict_accuracy": NaN}}', encoding="utf-8")

    table = render(_summary(tmp_path, "video-mme"), directory=tmp_path)

    assert "0.6100" in table
    assert "nan" in table.lower()


def test_a_headline_declared_and_also_pinned_in_the_breakdown_is_printed_once(
    tmp_path: Path,
) -> None:
    """Two rows for one number would read as the run disagreeing with itself."""
    _task(
        tmp_path,
        "video-mme",
        manifest={
            "scoring": {"mode": "runner", "metrics": {"accuracy": 0.61}},
            "metrics": {"accuracy": 0.61, "by_duration": {"long": {"accuracy": 0.49}}},
        },
    )

    rows = _rows(render(_summary(tmp_path, "video-mme"), directory=tmp_path))

    # The first row carries the task identity before its metric, so match on the cell instead.
    headline = [row for row in rows if "accuracy" in row and "by_duration" not in row]
    assert len(headline) == 1, headline
    assert [row for row in rows if "by_duration.long.accuracy" in row]
