"""Artifact and selection checks for the reproducible LoCoMo CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.artifacts import require_writable_output_pair
from mindbridge.benchmarks.locomo import LoCoMoConversation, LoCoMoQuestion, LoCoMoTurn
from mindbridge.benchmarks.locomo_cli import (
    LoCoMoRunManifest,
    _Arguments,
    _select_conversations,
    _write_artifacts,
)
from mindbridge.benchmarks.locomo_runner import (
    LoCoMoOfficialConversationResult,
    LoCoMoOfficialQuestionResult,
)
from mindbridge.models.jina import DEFAULT_JINA_OMNI_MODEL_ID, DEFAULT_JINA_OMNI_REVISION

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
                mindbridge_confidence=0.9,
                mindbridge_retrieved_dialog_ids=("D1:1",),
                mindbridge_trace_id="trace_01",
            ),
        ),
    )
    arguments = _arguments(dataset_path, output_path)

    _write_artifacts(arguments, (conversation,), (result,))

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    manifest_path = output_path.with_suffix(".json.manifest.json")
    manifest = LoCoMoRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert predictions[0]["qa"][0]["mindbridge_prediction"] == "Hello"
    assert manifest.source_revision == "official-revision"
    assert manifest.code_revision == "mindbridge-commit"
    assert manifest.answer_model_revision == "serving-fingerprint"
    assert manifest.run_id == "run_01"
    assert manifest.memory_item_count == 1
    assert manifest.question_count == 1
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        require_writable_output_pair(output_path, overwrite=False)


def test_locomo_subset_selection_rejects_unknown_samples() -> None:
    conversation = _conversation()
    assert _select_conversations((conversation,), ("conv-01",)) == (conversation,)
    with pytest.raises(ValueError, match="unknown"):
        _select_conversations((conversation,), ("missing",))


def _arguments(dataset_path: Path, output_path: Path) -> _Arguments:
    return _Arguments(
        dataset_path=dataset_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        source_revision="official-revision",
        code_revision="mindbridge-commit",
        answer_model_id="qwen3.8-max",
        answer_model_revision="serving-fingerprint",
        embedding_model_id=DEFAULT_JINA_OMNI_MODEL_ID,
        embedding_model_revision=DEFAULT_JINA_OMNI_REVISION,
        run_id="run_01",
        tenant_prefix="benchmark_locomo",
        recall_limit=20,
        request_concurrency=4,
        sample_ids=(),
        overwrite=False,
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
