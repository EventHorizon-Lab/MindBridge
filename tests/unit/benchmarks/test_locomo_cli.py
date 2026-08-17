"""Artifact and selection checks for the reproducible LoCoMo CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.artifacts import load_deployment_snapshot, require_writable_output_pair
from mindbridge.benchmarks.locomo import LoCoMoConversation, LoCoMoQuestion, LoCoMoTurn
from mindbridge.benchmarks.locomo_cli import (
    LOCOMO_RUNNER_VERSION,
    LoCoMoRunManifest,
    _Arguments,
    _write_artifacts,
)
from mindbridge.benchmarks.locomo_runner import (
    LoCoMoOfficialConversationResult,
    LoCoMoOfficialQuestionResult,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_locomo_artifacts_pin_source_system_code_and_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo10.json"
    dataset_path.write_text("official-input", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.json"
    conversation = _conversation()
    result = LoCoMoOfficialConversationResult(
        sample_id=conversation.sample_id,
        qa=(
            LoCoMoOfficialQuestionResult(
                question="What happened?",
                answer="Hello",
                evidence=("D1:1",),
                category=1,
                mindbridge_prediction="Hello",
                mindbridge_abstained=False,
                mindbridge_confidence=0.9,
                mindbridge_prediction_context=("D1:1",),
                mindbridge_trace_id="trace_01",
            ),
        ),
    )
    arguments = _arguments(dataset_path, output_path)

    _write_artifacts(
        arguments,
        (conversation,),
        (result,),
        load_deployment_snapshot(arguments.deployment_config_path),
    )

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    manifest_path = output_path.with_suffix(".json.manifest.json")
    manifest = LoCoMoRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert predictions[0]["qa"][0]["mindbridge_prediction"] == "Hello"
    assert predictions[0]["qa"][0]["mindbridge_prediction_context"] == ["D1:1"]
    assert manifest.source_revision == "official-revision"
    assert manifest.code_revision == "mindbridge-commit"
    assert manifest.deployment.server_generator.config["model_revision"] == ("serving-fingerprint")
    assert manifest.runner_version == LOCOMO_RUNNER_VERSION == "locomo_production_api_v9"
    assert manifest.run_id == "run_01"
    assert manifest.request_timeout_seconds == 1_800.0
    assert manifest.memory_item_count == 1
    assert manifest.question_count == 1
    assert manifest.abstained_question_count == 0
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        require_writable_output_pair(output_path, overwrite=False)


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
        tenant_prefix="benchmark_locomo",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        sample_ids=(),
        overwrite=False,
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


def _conversation() -> LoCoMoConversation:
    return LoCoMoConversation(
        sample_id="conv-01",
        turns=(
            LoCoMoTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="Hello",
                occurred_at=NOW,
            ),
        ),
        questions=(
            LoCoMoQuestion(
                question_id="conv-01_Q0001",
                question="What happened?",
                reference_answers=("Hello",),
                evidence_dialog_ids=("D1:1",),
                category=1,
            ),
        ),
    )
