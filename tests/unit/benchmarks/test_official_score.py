"""Binding an external official scorer's numbers to the exact run it scored."""

import json
from pathlib import Path

import pytest

import mindbridge.benchmarks.official_score as official_score
from mindbridge.benchmarks.official_score import (
    OfficialScore,
    build_official_score,
    parse_metric_assignment,
    parse_metric_assignments,
    score_sidecar_path,
)
from mindbridge.file_integrity import sha256_file


def test_official_score_binds_metrics_to_the_scored_predictions(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text("[]\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, predictions_path)
    scorer_output_path = tmp_path / "scorer-stdout.json"
    scorer_output_path.write_text('{"f1": 45.65}\n', encoding="utf-8")

    score = build_official_score(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        scorer_output_path=scorer_output_path,
        scorer_repository="snap-research/locomo",
        scorer_revision="3eb6f2c5",
        scorer_command="python evaluation/evaluate.py --data predictions.json",
        judge_model="gpt-4o-mini",
        answer_backbone="qwen3.8-max",
        scored_question_count=1_540,
        metrics={"f1": 45.65},
    )

    assert score.benchmark == "LoCoMo"
    assert score.run_id == "run_01"
    assert score.predictions_sha256 == sha256_file(predictions_path)
    assert score.scorer_output_sha256 == sha256_file(scorer_output_path)
    assert score.metrics == {"f1": 45.65}


def test_official_score_rejects_predictions_the_manifest_did_not_produce(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text("[]\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, predictions_path)
    scorer_output_path = tmp_path / "scorer-stdout.json"
    scorer_output_path.write_text("{}\n", encoding="utf-8")
    predictions_path.write_text('[{"edited": true}]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="predictions"):
        build_official_score(
            manifest_path=manifest_path,
            predictions_path=predictions_path,
            scorer_output_path=scorer_output_path,
            scorer_repository="snap-research/locomo",
            scorer_revision="3eb6f2c5",
            scorer_command="python evaluation/evaluate.py",
            judge_model="gpt-4o-mini",
            answer_backbone=None,
            scored_question_count=1_540,
            metrics={"f1": 45.65},
        )


def test_official_score_requires_at_least_one_metric(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text("[]\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, predictions_path)
    scorer_output_path = tmp_path / "scorer-stdout.json"
    scorer_output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_official_score(
            manifest_path=manifest_path,
            predictions_path=predictions_path,
            scorer_output_path=scorer_output_path,
            scorer_repository="snap-research/locomo",
            scorer_revision="3eb6f2c5",
            scorer_command="python evaluation/evaluate.py",
            judge_model=None,
            answer_backbone=None,
            scored_question_count=1_540,
            metrics={},
        )


def test_metric_assignment_parses_name_and_value() -> None:
    assert parse_metric_assignment("f1=45.65") == ("f1", 45.65)
    with pytest.raises(ValueError, match="name=value"):
        parse_metric_assignment("f1")
    with pytest.raises(ValueError):
        parse_metric_assignment("f1=not-a-number")


def test_score_sidecar_sits_next_to_its_predictions() -> None:
    assert score_sidecar_path(Path("run/predictions.json")) == Path(
        "run/predictions.json.score.json"
    )


def test_official_score_round_trips_through_json(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text("[]\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, predictions_path)
    scorer_output_path = tmp_path / "scorer-stdout.json"
    scorer_output_path.write_text("{}\n", encoding="utf-8")

    score = build_official_score(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        scorer_output_path=scorer_output_path,
        scorer_repository="MM-Lifelong/MM-Lifelong",
        scorer_revision="248aa820",
        scorer_command="python eval.py",
        judge_model="gpt-5",
        answer_backbone=None,
        scored_question_count=300,
        metrics={"accuracy": 18.62, "ref_at_300": 15.46},
    )

    assert OfficialScore.model_validate_json(score.model_dump_json()) == score


def _write_manifest(directory: Path, predictions_path: Path) -> Path:
    manifest_path = directory / "predictions.json.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark": "LoCoMo",
                "run_id": "run_01",
                "predictions_sha256": sha256_file(predictions_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_repeated_metric_names_are_refused_rather_than_silently_collapsed() -> None:
    assert parse_metric_assignments(["f1=45.65", "judge_accuracy=0.881"]) == {
        "f1": 45.65,
        "judge_accuracy": 0.881,
    }
    with pytest.raises(ValueError, match="more than once"):
        parse_metric_assignments(["f1=45.65", "f1=41.20"])


def test_the_metric_flag_is_required_so_omitting_it_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "official_score",
            "--predictions",
            "p.json",
            "--manifest",
            "m.json",
            "--scorer-output",
            "s.json",
            "--scorer-repository",
            "snap-research/locomo",
            "--scorer-revision",
            "3eb6f2c5",
            "--scorer-command",
            "python evaluation/evaluate.py",
            "--scored-question-count",
            "1540",
        ],
    )

    with pytest.raises(SystemExit):
        official_score._parse_arguments()
