from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mindbridge import (
    EmbedTask,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryKind,
    MemoryNotFoundError,
    Modality,
    ModelInput,
)


class _FixtureEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "semantic-support-fixture"
    embedding_space = "semantic-support-fixture:4:v1"
    embedding_dimension = 4

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


def _audit(claims: list[dict[str, Any]], **fields: object) -> dict[str, Any]:
    from mindbridge.benchmarks.semantic_support import AuditRun, score_audit

    return score_audit(
        AuditRun.model_validate(
            {
                "fixture_version": "test-v1",
                "reviewer_revision": "scripted-test-v1",
                "claims": claims,
                **fields,
            }
        )
    )


def test_unsupported_rate_uses_admitted_gold_claims_not_verifier_agreement() -> None:
    report = _audit(
        [
            {"claim_id": "a", "gold": "entailment", "predicted": "unknown", "admitted": False},
            {"claim_id": "b", "gold": "contradiction", "predicted": "entailment", "admitted": True},
            {"claim_id": "c", "gold": "unknown", "predicted": "unknown", "admitted": True},
            {"claim_id": "d", "gold": "entailment", "predicted": "entailment", "admitted": True},
        ]
    )
    assert report["unsupported_assertion_rate"] == pytest.approx(2 / 3)
    assert report["supported_retention_rate"] == 0.5
    assert report["contradiction_recall"] == 0.0
    assert report["classification_accuracy"] == 0.5
    assert report["write_cost"] is None
    assert report["verification_cost"] is None


def test_missing_judgments_and_failed_writes_are_not_successful_rejections() -> None:
    report = _audit(
        [
            {"claim_id": "a", "gold": "entailment", "admitted": None, "error_code": "model_failed"},
            {"claim_id": "b", "gold": "contradiction", "admitted": False},
        ]
    )
    assert report["unsupported_assertion_rate"] is None
    assert report["supported_retention_rate"] == 0.0
    assert report["classification_accuracy"] is None
    assert report["contradiction_recall"] is None
    assert report["classification_coverage"] == 0.0
    assert report["admission_coverage"] == 0.5
    assert report["failed_claims"] == 1


@pytest.mark.parametrize("kind,asset", [("text", None), ("frame", "video-1"), ("audio", "audio-1")])
def test_localization_requires_complete_joint_evidence_and_correct_source(
    kind: str, asset: str | None
) -> None:
    first = {"source_id": "s1", "kind": kind, "start": 0, "end": 1, "asset_id": asset}
    second = {**first, "source_id": "s2"}
    report = _audit(
        [
            {
                "claim_id": "partial",
                "gold": "entailment",
                "admitted": True,
                "gold_locations": [[first, second]],
                "predicted_locations": [first],
            },
            {
                "claim_id": "wrong-source",
                "gold": "contradiction",
                "admitted": False,
                "gold_locations": [[first]],
                "predicted_locations": [second],
            },
            {
                "claim_id": "alternative",
                "gold": "entailment",
                "admitted": True,
                "gold_locations": [[first, second], [second]],
                "predicted_locations": [second],
            },
        ]
    )
    assert report["localization_exact_match_rate"] == pytest.approx(1 / 3)
    assert report["localization_coverage"] == 1.0


def test_withdrawal_excludes_soft_forgetting_and_preserves_valid_or_alternatives() -> None:
    base = {"gold": "entailment", "admitted": True}
    report = _audit(
        [
            {
                **base,
                "claim_id": "residue",
                "lifecycle": {"kind": "withdrawal", "expected_absent": True, "visible_after": True},
            },
            {
                **base,
                "claim_id": "removed",
                "lifecycle": {
                    "kind": "withdrawal",
                    "expected_absent": True,
                    "visible_after": False,
                },
            },
            {
                **base,
                "claim_id": "or",
                "lifecycle": {
                    "kind": "withdrawal",
                    "expected_absent": False,
                    "visible_after": True,
                },
            },
            {
                **base,
                "claim_id": "forgotten",
                "lifecycle": {
                    "kind": "soft_forgetting",
                    "expected_absent": False,
                    "visible_after": True,
                },
            },
            {
                **base,
                "claim_id": "unobserved",
                "lifecycle": {"kind": "withdrawal", "expected_absent": True},
            },
        ]
    )
    assert report["withdrawal_residual_rate"] == 0.5
    assert report["withdrawal_coverage"] == pytest.approx(2 / 3)
    assert report["withdrawal_preservation_rate"] == 1.0
    assert report["soft_forgetting_cases"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"admitted": "false"},
        {"gold": "supported"},
        {"error_code": "model_failed"},
        {"predicted_locations": [{"source_id": "s", "kind": "text", "start": True, "end": 2}]},
        {"gold_locations": [[]]},
        {"predicted_locations": [{"source_id": "s", "kind": "audio", "start": 0, "end": 1}]},
    ],
)
def test_invalid_review_inputs_cannot_silently_improve_metrics(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _audit([{"claim_id": "a", "gold": "unknown", "admitted": True, **change}])


def test_duplicate_claims_and_unpinned_judgments_are_rejected() -> None:
    row = {"claim_id": "a", "gold": "unknown", "admitted": True}
    with pytest.raises(ValidationError):
        _audit([row, row])
    with pytest.raises(ValidationError):
        _audit([{**row, "predicted": "unknown"}], reviewer_revision=None)


def test_partial_usage_keeps_unknown_measurements_and_counts_shared_requests_once() -> None:
    report = _audit(
        [
            {"claim_id": "a", "gold": "entailment", "admitted": True},
            {"claim_id": "b", "gold": "unknown", "admitted": False},
        ],
        write_cost={"requests": 1, "latency_ms": 25.0},
        verification_cost={"requests": 2, "input_tokens": 120, "output_tokens": 15},
    )
    assert report["write_cost"] == {
        "requests": 1,
        "latency_ms": 25.0,
        "input_tokens": None,
        "output_tokens": None,
    }
    assert report["verification_cost"] == {
        "requests": 2,
        "latency_ms": None,
        "input_tokens": 120,
        "output_tokens": 15,
    }
    for cost in ({"requests": True}, {"input_tokens": -1}, {"latency_ms": float("nan")}):
        with pytest.raises(ValidationError):
            _audit([{"claim_id": "a", "gold": "unknown", "admitted": True}], write_cost=cost)


def test_cli_scores_without_echoing_private_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from mindbridge.benchmarks.semantic_support import main

    source = tmp_path / "audit.json"
    output = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "fixture_version": "test-v1",
                "claims": [{"claim_id": "private-id", "gold": "unknown", "admitted": True}],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["unsupported_assertion_rate"] == 1.0
    assert "private-id" not in capsys.readouterr().out
    source.write_text('{"private-memory-content": "secret"}', encoding="utf-8")
    assert main(["--input", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "secret" not in captured.err
    assert "private-memory-content" not in captured.err


def test_formation_provenance_does_not_establish_semantic_entailment(tmp_path: Path) -> None:
    class Former:
        formation_capabilities = _FixtureEmbedder.embedding_capabilities
        formation_model = "scripted-semantic-boundary"
        formation_space = "scripted-semantic-boundary:v1"

        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert len(inputs) == 1
            return (
                tuple(
                    FormationProposal(kind=MemoryKind.EVENT, content=claim)
                    for claim in (
                        "Lin drank tea.",
                        "Lin drank coffee.",
                        "Lin owns a bicycle.",
                    )
                ),
            )

        def close(self) -> None:
            pass

    with Memory(
        tmp_path, embedder=_FixtureEmbedder(), former=Former(), minimum_relevance=0
    ) as memory:
        source = memory.add("Lin drank tea and did not drink coffee.")
        derived = [record for record in memory.list().items if record.id != source.id]
        assert {record.content for record in derived} == {
            "Lin drank tea.",
            "Lin drank coffee.",
            "Lin owns a bicycle.",
        }
        assert all(
            record.context and record.context.evidence_ids == (source.id,) for record in derived
        )
        # Cognitive forgetting leaves existing support standing; deletion withdraws it.
        memory.forget((source.id,))
        for record in derived:
            standing = memory.get(record.id).context
            assert standing is not None and standing.visible
        memory.delete(source.id)
        for record in derived:
            with pytest.raises(MemoryNotFoundError):
                memory.get(record.id)
