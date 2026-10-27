"""The pinned dataset uses both absolute and local question-span annotations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.supermemory_vqa import (
    SUPERMEMORY_UNANSWERABLE_CHOICE,
    load_supermemory_vqa,
)


@pytest.mark.parametrize(
    "evidence",
    (
        {
            "time_spans": [
                {
                    "start_time": 2,
                    "end_time": 5,
                    "video_id": "video-1",
                    "video_start_time_unix": 1000,
                }
            ],
        },
        {
            "time_span": {"start_time": "00:02", "end_time": "00:05"},
            "video_id": "video-1",
            "start_time": 1000,
        },
    ),
    ids=("absolute-span", "local-span"),
)
def test_current_annotation_shapes_resolve_the_question_end(
    tmp_path: Path, evidence: dict[str, object]
) -> None:
    annotation = tmp_path / "all_qa.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_id": 1,
                    "subject": 1,
                    "question": "What color is the cup?",
                    "choices": ["red", "blue", "green", SUPERMEMORY_UNANSWERABLE_CHOICE],
                    "correct_answer": "red",
                    "correct_option_index": 0,
                    "choice_types": ["correct", "incorrect", "incorrect", "unanswerable"],
                    "metadata": {"skill": "color", "primary_video_id": "video-1"},
                    "video_ids": ["video-1"],
                    "question_evidence": evidence,
                    "is_answerable": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    (question,) = load_supermemory_vqa(annotation)

    assert question.question_video_id == "video-1"
    assert question.question_ended_at == datetime.fromtimestamp(1005, tz=timezone.utc)
