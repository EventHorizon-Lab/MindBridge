"""The experiment must penalize false corroboration rather than reward visibility."""

from pathlib import Path

from mindbridge.benchmarks.corroboration import Scenario, Step, _ablation, run_case, summarize


def test_learning_and_false_corroboration_have_separate_denominators() -> None:
    rows = [
        {"expected_visible": True, "visible": True, "compiled": True},
        {"expected_visible": True, "visible": False, "compiled": False},
        {"expected_visible": False, "visible": True, "compiled": True},
        {"expected_visible": False, "visible": False, "compiled": False},
        {"expected_visible": False, "visible": False, "compiled": False},
    ]
    result = summarize(rows)
    assert result["learning_rate"] == 0.5
    assert result["false_corroboration_rate"] == 1 / 3
    assert result["accuracy"] == 3 / 5
    assert result["compiled_accuracy"] == 3 / 5


def test_absent_class_is_unmeasured_not_a_perfect_score() -> None:
    result = summarize([{"expected_visible": False, "visible": False, "compiled": False}])
    assert result["learning_rate"] is None
    assert result["false_corroboration_rate"] == 0.0


def test_no_provenance_control_still_deduplicates_one_assessment() -> None:
    case = Scenario(
        "permuted",
        "dev",
        ("a", "b"),
        (Step("target", ("r0", "r1")), Step("target", ("r1", "r0"))),
        False,
    )
    assert not _ablation(case, union_alternatives=False)


def test_union_control_reads_final_ancestry_after_late_alternative() -> None:
    case = Scenario(
        "late",
        "dev",
        ("a", "b"),
        (Step("x", ("r0",)), Step("target", ("x",)), Step("target", ("r1",)), Step("x", ("r1",))),
        True,
    )
    assert not _ablation(case, union_alternatives=True)


def test_omni_fixture_survives_real_asset_storage(tmp_path: Path) -> None:
    case = Scenario("joint_single", "dev", ("a", "b"), (Step("target", ("r0", "r1")),), False)
    row = run_case(case, "omni", 7100, "B", tmp_path)
    assert row["visible"] is False
    assert row["budget_ok"]
