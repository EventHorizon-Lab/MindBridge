"""Schema checks for official memory benchmark adapters."""

import json
from pathlib import Path
from typing import ClassVar

import pytest

from mindbridge.benchmarks import (
    MMLifelongSplit,
    load_egolife_qa,
    load_egomem_reason,
    load_egotempo,
    load_locomo,
    load_m3_bench,
    load_memlens,
    load_memlens_agent_subset,
    load_mm_lifelong,
    load_supermemory_vqa,
    load_video_mme,
)


class _FakeArrowTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return self.rows


class _FakeParquet:
    rows: ClassVar[list[dict[str, object]]] = []

    @classmethod
    def read_table(cls, source: Path) -> _FakeArrowTable:
        assert source.name == "test.parquet"
        return _FakeArrowTable(cls.rows)


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


def test_video_mme_adapter_loads_official_parquet_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation_path = tmp_path / "test.parquet"
    _FakeParquet.rows = [
        {
            "video_id": "001",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "Humanity & History",
            "url": "https://www.youtube.com/watch?v=fFjv93ACGo8",
            "videoID": "fFjv93ACGo8",
            "question_id": f"001-{index}",
            "task_type": "Counting Problem",
            "question": "Which decoration appears most?",
            "options": ["A. Apples.", "B. Candles.", "C. Berries.", "D. Equal."],
            "answer": "C",
        }
        for index in range(1, 4)
    ]
    monkeypatch.setattr("mindbridge.benchmarks.video_mme.import_module", lambda name: _FakeParquet)

    video = load_video_mme(annotation_path)[0]

    assert video.video_id == "001"
    assert video.source_video_id == "fFjv93ACGo8"
    assert len(video.questions) == 3
    assert video.questions[0].options[2] == "C. Berries."


def test_egotempo_adapter_parses_official_clip_boundaries(tmp_path: Path) -> None:
    annotation_path = tmp_path / "egotempo_openQA.json"
    annotation_path.write_text(
        json.dumps(
            {
                "info": {"release date": "19.03.2025", "version": "1.0"},
                "annotations": [
                    {
                        "question_id": "video_-0.1_180.0_0",
                        "clip_id": "video_-0.1_180.0",
                        "question_type": "action-specific object",
                        "question": "What did the person pick up?",
                        "answer": "A spoon.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    question = load_egotempo(annotation_path)[0]

    assert question.source_video_id == "video"
    assert question.clip_start_seconds == -0.1
    assert question.clip_end_seconds == 180.0
    assert question.reference_answer == "A spoon."


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


def test_egomem_reason_adapter_preserves_reshuffled_option_order(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations_public.jsonl"
    annotation_path.write_text(
        json.dumps(
            {
                "example_id": 1,
                "p_id": "A1_JAKE_DAY7_19_00_00_q001",
                "identity": "A1_JAKE",
                "query_time": "DAY7, 19:00:00",
                "question": "What do I most often eat?",
                "options": {
                    "A": "Rice",
                    "B": "Dumplings",
                    "C": "Burger",
                    "D": "Pancake",
                    "E": "Noodles",
                },
                "query_type": "Activity Pattern",
                "correct_answer": "SECRET",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    question = load_egomem_reason(annotation_path)[0]

    assert question.choices == ("Rice", "Dumplings", "Burger", "Pancake", "Noodles")
    assert question.query_offset_ms == 586_800_000
    assert "answer" not in question.model_dump()
    assert "SECRET" not in question.model_dump_json()


def test_memlens_adapter_keeps_session_order_and_agent_subset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_32k.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_4106e113",
                    "question_type": "multi_session_reasoning",
                    "question": "How much did I spend?",
                    "answer": "$260.00",
                    "question_date": "2024/05/31 (Fri) 07:58",
                    "haystack_dates": [
                        "2024/05/06 (Mon) 17:17",
                        "2024/05/07 (Tue) 21:42",
                    ],
                    "haystack_session_ids": ["sess_first", "sess_second"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I paid $150. <image>",
                                "images": [
                                    {
                                        "file": "needle_images/receipt.jpg",
                                        "image_url": "",
                                        "blip_caption": "a receipt",
                                    }
                                ],
                            }
                        ],
                        [{"role": "AI Assistant", "content": "Noted.", "images": []}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    subset_path = tmp_path / "agent_subset_195.json"
    subset_path.write_text(
        json.dumps({"n_questions": 1, "question_ids": ["q_4106e113"]}),
        encoding="utf-8",
    )

    question = load_memlens(dataset_path)[0]

    assert [session.session_id for session in question.sessions] == [
        "sess_first",
        "sess_second",
    ]
    assert question.sessions[0].turns[0].images[0].source_file == ("needle_images/receipt.jpg")
    assert question.sessions[0].turns[0].images[0].source_url is None
    assert question.sessions[1].turns[0].role == "assistant"
    assert load_memlens_agent_subset(subset_path) == ("q_4106e113",)


@pytest.mark.parametrize(
    ("split", "clue_key", "clues"),
    [
        ("day_test", "clue_intervals", [[10, 20]]),
        (
            "week_test",
            "clue_intervals",
            [{"video_id": "day5", "intervals": [[10, 20]]}],
        ),
        (
            "month_val",
            "clue_interval",
            [{"video_id": "14", "intervals": [[10, 20]]}],
        ),
    ],
)
def test_mm_lifelong_adapter_normalizes_official_split_schemas(
    tmp_path: Path,
    split: MMLifelongSplit,
    clue_key: str,
    clues: object,
) -> None:
    annotation_path = tmp_path / f"{split}.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "question": "What happened?",
                    "answer": "A meeting",
                    "question_type": "Event Recognition",
                    "temporal_certificate": "Short",
                    clue_key: clues,
                    "total_intervals": [[100, 110]],
                }
            ]
        ),
        encoding="utf-8",
    )

    question = load_mm_lifelong(annotation_path, split)[0]

    assert question.reference_intervals == ((100.0, 110.0),)
    assert question.clue_interval_count == 1


def test_mm_lifelong_adapter_retains_reversed_intervals_skipped_by_official_scorer(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "day.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "index": 17,
                    "question": "What happened?",
                    "answer": "Nothing",
                    "question_type": "Event Recognition",
                    "temporal_certificate": "Short",
                    "clue_intervals": [[11867, 11842]],
                    "total_intervals": [[11867, 11842]],
                }
            ]
        ),
        encoding="utf-8",
    )

    question = load_mm_lifelong(annotation_path, "day_test")[0]

    assert question.reference_intervals == ((11867.0, 11842.0),)


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
