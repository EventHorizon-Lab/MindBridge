"""Artifact and selection checks for the reproducible LoCoMo-Refined CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.artifacts import (
    load_deployment_snapshot,
    require_writable_output_pair,
    sidecar_manifest_path,
)
from mindbridge.benchmarks.locomo_refined import (
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.benchmarks.locomo_refined_cli import (
    LOCOMO_REFINED_RUNNER_VERSION,
    LoCoMoRefinedRunManifest,
    _Arguments,
    _write_artifacts,
)
from mindbridge.benchmarks.locomo_refined_runner import LoCoMoRefinedPrediction

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_locomo_refined_artifacts_pin_source_system_code_and_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo_refined.json"
    dataset_path.write_text("official-input", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.jsonl"
    conversation = _conversation()
    arguments = _arguments(dataset_path, output_path)

    _write_artifacts(
        arguments,
        (conversation,),
        (_prediction("conv-26#q0000", "Hello"),),
        load_deployment_snapshot(arguments.deployment_config_path),
    )

    rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line
    ]
    manifest = LoCoMoRefinedRunManifest.model_validate_json(
        sidecar_manifest_path(output_path).read_text(encoding="utf-8")
    )
    # `mem-eval-suite/LoCoMo_refined`'s own evaluator indexes predictions by `qa_id` and
    # reads `predicted_answer`; everything else on the row is diagnostics it ignores.
    assert rows[0]["qa_id"] == "conv-26#q0000"
    assert rows[0]["predicted_answer"] == "Hello"
    assert rows[0]["mindbridge_prediction_context"] == ["D1:1"]
    assert manifest.benchmark == "LoCoMo-Refined"
    assert manifest.source_repository == "mem-eval-suite/LoCoMo_refined"
    assert manifest.source_revision == "official-revision"
    assert manifest.code_revision == "mindbridge-commit"
    assert manifest.deployment.server_generator.config["model_revision"] == ("serving-fingerprint")
    assert (
        manifest.runner_version
        == LOCOMO_REFINED_RUNNER_VERSION
        == "locomo_refined_production_api_v1"
    )
    assert manifest.run_id == "run_01"
    assert manifest.request_timeout_seconds == 1_800.0
    assert manifest.memory_item_count == 1
    assert manifest.question_count == 1
    assert manifest.unanswered_question_count == 0
    assert manifest.category_question_counts == {1: 1}
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        require_writable_output_pair(output_path, overwrite=False)


def test_locomo_refined_manifest_separates_silence_from_wrong_answers(tmp_path: Path) -> None:
    """An empty prediction scores as a miss, so a run has to say how many were empty."""
    dataset_path = tmp_path / "locomo_refined.json"
    dataset_path.write_text("official-input", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.jsonl"
    conversation = _conversation().model_copy(
        update={
            "questions": tuple(
                LoCoMoRefinedQuestion(
                    question_id=f"conv-26#q{index:04d}",
                    question=f"Question {category}?",
                    reference_answers=("Hello",),
                    evidence_dialog_ids=("D1:1",),
                    category=category,
                    is_multi_modality=False,
                )
                for index, category in enumerate((1, 1, 2, 4))
            )
        }
    )
    predictions = (
        _prediction("conv-26#q0000", "Hello"),
        _prediction("conv-26#q0001", ""),
        _prediction("conv-26#q0002", ""),
        _prediction("conv-26#q0003", "Hello"),
    )
    arguments = _arguments(dataset_path, output_path)

    _write_artifacts(
        arguments,
        (conversation,),
        predictions,
        load_deployment_snapshot(arguments.deployment_config_path),
    )

    manifest = LoCoMoRefinedRunManifest.model_validate_json(
        sidecar_manifest_path(output_path).read_text(encoding="utf-8")
    )
    assert manifest.question_count == 4
    assert manifest.unanswered_question_count == 2
    # 802 of the release's 1,382 questions are category 4, so a subset run that skews the
    # mix reports a number the whole-release number cannot be compared against.
    assert manifest.category_question_counts == {1: 2, 2: 1, 4: 1}


def _prediction(qa_id: str, answer: str) -> LoCoMoRefinedPrediction:
    return LoCoMoRefinedPrediction(
        qa_id=qa_id,
        predicted_answer=answer,
        mindbridge_answered=bool(answer),
        mindbridge_confidence=0.9 if answer else 0.0,
        mindbridge_prediction_context=("D1:1",),
        mindbridge_trace_id="trace_01",
    )


def _arguments(dataset_path: Path, output_path: Path) -> _Arguments:
    deployment_path = dataset_path.parent / "deployment.json"
    _write_deployment(deployment_path)
    return _Arguments(
        dataset_path=dataset_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        source_revision="official-revision",
        code_revision="mindbridge-commit",
        deployment_config_path=deployment_path,
        run_id="run_01",
        tenant_prefix="benchmark_locomo_refined",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        sample_ids=(),
        overwrite=False,
        quiet=True,
    )


def _write_deployment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "server_generator": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {
                        "model_id": "qwen3.8-max",
                        "model_revision": "serving-fingerprint",
                        "reasoning_effort": "low",
                    },
                },
                "server_embedder": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"space_id": "jina-v5", "space_revision": "space-v1"},
                },
            }
        ),
        encoding="utf-8",
    )


def _conversation() -> LoCoMoRefinedConversation:
    return LoCoMoRefinedConversation(
        sample_id="conv-26",
        turns=(
            LoCoMoRefinedTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="Hello",
                occurred_at=NOW,
            ),
        ),
        questions=(
            LoCoMoRefinedQuestion(
                question_id="conv-26#q0000",
                question="What happened?",
                reference_answers=("Hello",),
                evidence_dialog_ids=("D1:1",),
                category=1,
                is_multi_modality=False,
            ),
        ),
    )
