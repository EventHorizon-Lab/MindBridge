from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from mindbridge.benchmarks import rejudge_icm
from mindbridge.benchmarks.eval_config import _JudgeConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _saved(**updates: object) -> dict[str, object]:
    return {
        "sample_id": "icm-bench/timeline/q1",
        "task": "icm-bench",
        "unit_id": "timeline",
        "question_id": "q1",
        "dataset_sha256": "original-corpus",
        "prediction": "Ada",
        "references": ["Ada"],
        "source_question": "Who left?",
        "score": 0.0,
        "scorer_protocol": "old_protocol",
        **updates,
    }


def _annotations(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Who left?",
                "reference_answer": "Ada",
                "category": "Identity Recall",
                "target_character_ids": ["p1"],
                "evidence_video_ids": ["clip_000"],
                "before_clip": "clip_000",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_rejudge_preserves_old_predictions_and_scores_and_repairs_old_judge_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = rejudge_icm.attach_questions(
        (_saved(error_code="JudgeError"),), _annotations(tmp_path / "qa.jsonl")
    )
    calls = []

    async def judge(*args: object, **kwargs: object) -> tuple[dict[str, float], str, bool]:
        calls.append(args[1])
        return {"accuracy": 1.0}, "Yes.", False

    monkeypatch.setattr(rejudge_icm, "_judge_call", judge)
    (result,) = asyncio.run(
        rejudge_icm.rejudge(
            rows,
            client=cast("AsyncOpenAI", object()),
            config=_JudgeConfig(model="test-judge", base_url="https://judge.invalid/v1"),
        )
    )
    assert result["prediction"] == rows[0]["prediction"] == "Ada"
    assert result["old_score"] == rows[0]["score"] == 0.0
    assert result["score"] == 1.0
    assert result["old_scorer_protocol"] == "old_protocol"
    assert result["scorer_protocol"] != "old_protocol"
    assert result["official_judge_model"] is False
    assert len(calls) == 1
    messages = calls[0]
    assert isinstance(messages, tuple)
    assert len(messages) == 1 and messages[0].role == "user"


@pytest.mark.parametrize("updates", [{"references": ["wrong"]}, {"source_question": "Wrong?"}])
def test_rejudge_refuses_annotation_mismatch_before_any_request(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="disagree"):
        rejudge_icm.attach_questions((_saved(**updates),), _annotations(tmp_path / "qa.jsonl"))


def test_rejudge_failures_are_unscored_and_failed_generation_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def failed(*args: object, **kwargs: object) -> tuple[dict[str, float], str, bool]:
        calls.append(args)
        raise ValueError("secret endpoint credentials")

    monkeypatch.setattr(rejudge_icm, "_judge_call", failed)
    results = asyncio.run(
        rejudge_icm.rejudge(
            (_saved(), _saved(error_code="ModelError")),
            client=cast("AsyncOpenAI", object()),
            config=_JudgeConfig(model="test-judge", base_url="https://judge.invalid/v1"),
        )
    )
    assert len(calls) == 1
    assert results[0]["status"] == "judge_failed"
    assert results[0]["score"] is None
    assert "secret" not in json.dumps(results)
    assert results[1]["status"] == "skipped_failed_sample"


def test_duplicate_predictions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text((json.dumps(_saved()) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        rejudge_icm.load_predictions(path)
