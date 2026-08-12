"""Schema checks for official memory benchmark adapters."""

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks import (
    load_egolife_qa,
    load_locomo,
    load_m3_bench,
    load_supermemory_vqa,
)


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


def test_egolife_adapter_keeps_queries_but_discards_retrieval_hints(tmp_path: Path) -> None:
    annotation_path = tmp_path / "EgoLifeQA_A1_JAKE.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "ID": "1",
                    "query_time": {"date": "DAY2", "time": "11210250"},
                    "type": "EntityLog",
                    "need_audio": True,
                    "need_name": True,
                    "last_time": False,
                    "question": "Who used the screwdriver first?",
                    "choice_a": "Tasha",
                    "choice_b": "Alice",
                    "choice_c": "Shure",
                    "choice_d": "Lucia",
                    "answer": "B",
                    "target_time": {"date": "DAY1", "time": "11152408"},
                    "keywords": "SECRET RETRIEVAL HINT",
                    "reason": "SECRET ANSWER EVIDENCE",
                }
            ]
        ),
        encoding="utf-8",
    )

    question = load_egolife_qa(annotation_path)[0]

    assert question.query_offset_ms == 127_264_500
    assert question.correct_option == "B"
    assert question.choices == ("Tasha", "Alice", "Shure", "Lucia")
    assert "SECRET" not in question.model_dump_json()


def test_egolife_adapter_rejects_invalid_query_clock(tmp_path: Path) -> None:
    annotation_path = tmp_path / "bad.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "ID": "1",
                    "query_time": {"date": "DAY1", "time": "11600200"},
                    "type": "EventRecall",
                    "need_audio": False,
                    "need_name": False,
                    "last_time": False,
                    "question": "What happened?",
                    "choice_a": "A",
                    "choice_b": "B",
                    "choice_c": "C",
                    "choice_d": "D",
                    "answer": "A",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid EgoLifeQA query time"):
        load_egolife_qa(annotation_path)


def test_supermemory_adapter_supports_current_and_legacy_question_end_times(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "all_qa.json"
    annotation_path.write_text(
        json.dumps(
            [
                _supermemory_question(
                    1,
                    {
                        "time_spans": [
                            {
                                "start_time": 600,
                                "end_time": 636,
                                "video_id": "Person_1_session_8",
                                "video_start_time_unix": 1_773_180_268,
                            }
                        ]
                    },
                ),
                _supermemory_question(
                    2,
                    {
                        "time_span": {"start_time": "01:29:28", "end_time": "01:30:54"},
                        "video_id": "Person_1_session_8",
                        "start_time": 1_773_180_268,
                    },
                ),
            ]
        ),
        encoding="utf-8",
    )

    questions = load_supermemory_vqa(annotation_path)

    assert questions[0].question_ended_at.timestamp() == 1_773_180_904
    assert questions[1].question_ended_at.timestamp() == 1_773_185_722
    assert questions[0].question_video_id == "Person_1_session_8"
    assert questions[0].unanswerable_option_index == 0
    assert "SECRET ANSWER EVIDENCE" not in questions[0].model_dump_json()


def test_supermemory_adapter_rejects_inconsistent_answerability(tmp_path: Path) -> None:
    annotation_path = tmp_path / "bad.json"
    row = _supermemory_question(
        1,
        {
            "time_spans": [
                {
                    "start_time": 0,
                    "end_time": 1,
                    "video_id": "Person_1_session_8",
                    "video_start_time_unix": 1_773_180_268,
                }
            ]
        },
    )
    row["is_answerable"] = False
    annotation_path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent answerability"):
        load_supermemory_vqa(annotation_path)


def _supermemory_question(
    question_id: int, question_evidence: dict[str, object]
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": "What did B say he cooks?",
        "choices": [
            "This question can not be answered.",
            "Beef",
            "Chicken",
            "Meat",
        ],
        "correct_answer": "Beef",
        "correct_option_index": 1,
        "choice_types": ["incorrect", "correct", "incorrect", "vague"],
        "subject": 1,
        "metadata": {
            "skill": "conversational_memory",
            "primary_video_id": "Person_1_session_8",
        },
        "video_ids": ["Person_1_session_8"],
        "start_time": 1_773_180_268,
        "question_evidence": question_evidence,
        "is_answerable": True,
        "answer_evidence": {"text": "SECRET ANSWER EVIDENCE"},
    }
