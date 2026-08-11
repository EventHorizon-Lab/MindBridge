"""Schema checks for official memory benchmark adapters."""

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks import load_locomo, load_m3_bench


def test_locomo_adapter_orders_sessions_and_normalizes_adversarial_qa(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo10.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-01",
                    "conversation": {
                        "speaker_a": "Caroline",
                        "speaker_b": "Melanie",
                        "session_2_date_time": "2:00 pm on 9 May, 2023",
                        "session_2": [
                            {"speaker": "Melanie", "dia_id": "D2:1", "text": "Later turn"}
                        ],
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                        "session_1": [
                            {
                                "speaker": "Caroline",
                                "dia_id": "D1:1",
                                "text": "Earlier turn",
                                "img_url": ["data:image/jpeg;base64," + "a" * 3_000],
                                "blip_caption": "a sunrise",
                            }
                        ],
                    },
                    "qa": [
                        {
                            "question": "What happened first?",
                            "answer": "Earlier turn",
                            "evidence": ["D1:1; D2:1"],
                            "category": 2,
                        },
                        {
                            "question": "What was never discussed?",
                            "answer": "A distractor",
                            "evidence": [],
                            "category": 5,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    conversation = load_locomo(dataset_path)[0]

    assert [turn.dialog_id for turn in conversation.turns] == ["D1:1", "D2:1"]
    assert conversation.turns[0].occurred_at.isoformat() == "2023-05-08T13:56:00+00:00"
    assert conversation.turns[0].image_sources[0].startswith("data:image/jpeg;base64,")
    assert conversation.questions[0].evidence_dialog_ids == ("D1:1", "D2:1")
    assert conversation.questions[1].reference_answers == ("Not mentioned in the conversation",)


def test_m3_bench_adapter_preserves_multimodal_question_metadata(tmp_path: Path) -> None:
    annotation_path = tmp_path / "robot.json"
    annotation_path.write_text(
        json.dumps(
            {
                "living_room_06": {
                    "video_path": "data/videos/robot/living_room_06.mp4",
                    "mem_path": "data/memory_graphs/robot/living_room_06.pkl",
                    "qa_list": [
                        {
                            "question": "Where is the yoga mat?",
                            "answer": "Inside the storage room",
                            "question_id": "living_room_06_Q09",
                            "reasoning": "Ground-truth rationale",
                            "timestamp": "16:10",
                            "type": ["Cross-Modal Reasoning", "Multi-Detail Reasoning"],
                            "before_clip": 31,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    video = load_m3_bench(annotation_path)[0]

    assert video.video_id == "living_room_06"
    assert video.questions[0].question_types == (
        "Cross-Modal Reasoning",
        "Multi-Detail Reasoning",
    )
    assert video.questions[0].timestamp_seconds == 970
    assert video.questions[0].before_clip_index == 31


def test_m3_bench_adapter_rejects_empty_annotations(tmp_path: Path) -> None:
    annotation_path = tmp_path / "robot.json"
    annotation_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        load_m3_bench(annotation_path)
