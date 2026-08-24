"""Schema checks for official memory benchmark adapters."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import pytest

from mindbridge.benchmarks.atm_bench import (
    atm_email_block,
    atm_evidence_kind,
    atm_memory_chunks,
    atm_sgm_block,
    load_atm_bench,
    load_atm_emails,
    load_atm_niah_pool,
    load_atm_sgm,
)
from mindbridge.benchmarks.egolife_qa import load_egolife_qa
from mindbridge.benchmarks.egomem_reason import load_egomem_reason
from mindbridge.benchmarks.egotempo import load_egotempo
from mindbridge.benchmarks.locomo_refined import load_locomo_refined
from mindbridge.benchmarks.m3_bench import load_m3_bench
from mindbridge.benchmarks.mem_gallery import load_mem_gallery, load_mem_gallery_topic
from mindbridge.benchmarks.memlens import load_memlens, load_memlens_agent_subset
from mindbridge.benchmarks.mm_lifelong import MMLifelongSplit, load_mm_lifelong
from mindbridge.benchmarks.supermemory_vqa import load_supermemory_vqa
from mindbridge.benchmarks.video_mme import load_video_mme
from mindbridge.core import MediaKind


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


def test_locomo_refined_adapter_orders_sessions_and_uses_official_question_ids(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "locomo_refined.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-26",
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
                            "answer": ["Earlier turn"],
                            "evidence": ["D1:1; D2:1"],
                            "category": 2,
                            "is_multi_modality": True,
                        },
                        {
                            "question": "When did Melanie paint a sunrise?",
                            "answer": [2022],
                            "evidence": ["D2:1"],
                            "category": 2,
                            "is_multi_modality": False,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    conversation = load_locomo_refined(dataset_path)[0]

    assert [turn.dialog_id for turn in conversation.turns] == ["D1:1", "D2:1"]
    assert conversation.turns[0].occurred_at.isoformat() == "2023-05-08T13:56:00+00:00"
    assert conversation.turns[0].image_sources[0].startswith("data:image/jpeg;base64,")
    assert conversation.questions[0].evidence_dialog_ids == ("D1:1", "D2:1")
    assert conversation.questions[0].is_multi_modality is True
    # `{sample_id}#q{index:04d}`, zero-based, is the release's own `qa_id` -- the key its
    # evaluator joins predictions on, so a drift here silently scores nothing.
    assert [question.question_id for question in conversation.questions] == [
        "conv-26#q0000",
        "conv-26#q0001",
    ]
    # `data/public/questions.jsonl` publishes the six numeric golds as strings.
    assert conversation.questions[1].reference_answers == ("2022",)


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


# U+2028 LINE SEPARATOR is legal inside a JSON string and is not a JSON line
# delimiter, but `str.splitlines()` breaks on it. The shipped EgoMemReason
# v1.1 public release happens to carry none today, so no fixture built from
# the real corpus can exercise this -- hence a synthetic annotation that does.
_EGOMEM_SEPARATOR = "\u2028"
_EGOMEM_SEPARATED_QUESTION = f"What do I eat{_EGOMEM_SEPARATOR}most often?"


def test_egomem_reason_adapter_keeps_a_unicode_line_separator_in_a_question(
    tmp_path: Path,
) -> None:
    """Splitting the annotations JSONL on Unicode line boundaries shreds records.

    Under `str.splitlines()` this one annotation becomes two truncated
    pieces, so `model_validate_json` raises `Invalid JSON` and the loader
    fails outright. Asserting the round-tripped question text (not just that
    loading succeeded) also rules out a rule that split the line and silently
    dropped or rejoined the character.
    """
    annotation_path = tmp_path / "annotations_public.jsonl"
    annotation_path.write_text(
        json.dumps(
            {
                "example_id": 1,
                "p_id": "A1_JAKE_DAY7_19_00_00_q001",
                "identity": "A1_JAKE",
                "query_time": "DAY7, 19:00:00",
                "question": _EGOMEM_SEPARATED_QUESTION,
                "options": {"A": "Rice", "B": "Dumplings", "C": "Burger", "D": "Pancake"},
                "query_type": "Activity Pattern",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    [question] = load_egomem_reason(annotation_path)

    assert question.question == _EGOMEM_SEPARATED_QUESTION


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


def _atm_question(**overrides: object) -> dict[str, object]:
    question = {
        "id": "1defb7d5-aab4-4244-8b3c-971a36376b04",
        "question": "How much did I pay for accommodation for BMVC 2024?",
        "answer": "£799.74",
        "notes": "",
        "evidence_ids": ["email202411160004", "20250223_130249"],
        "qtype": "number",
    }
    return question | overrides


def test_atm_bench_adapter_reads_questions_and_classifies_evidence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "atm-bench.json"
    dataset_path.write_text(json.dumps([_atm_question()]), encoding="utf-8")

    questions = load_atm_bench(dataset_path)

    assert len(questions) == 1
    assert questions[0].question_id == "1defb7d5-aab4-4244-8b3c-971a36376b04"
    assert questions[0].qtype == "number"
    assert questions[0].evidence_ids == ("email202411160004", "20250223_130249")
    assert questions[0].niah_evidence_ids == ()
    assert atm_evidence_kind("email202411160004") == "email"
    assert atm_evidence_kind("20250223_130249") == "media"


def test_atm_bench_adapter_refuses_duplicate_ids_and_missing_evidence(tmp_path: Path) -> None:
    duplicated = tmp_path / "duplicated.json"
    duplicated.write_text(json.dumps([_atm_question(), _atm_question()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question IDs"):
        load_atm_bench(duplicated)

    empty_evidence = tmp_path / "empty_evidence.json"
    empty_evidence.write_text(json.dumps([_atm_question(evidence_ids=[])]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_atm_bench(empty_evidence)

    empty_release = tmp_path / "empty.json"
    empty_release.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_atm_bench(empty_release)


def test_atm_niah_pool_requires_every_gold_evidence_in_the_pool(tmp_path: Path) -> None:
    complete = tmp_path / "niah25.json"
    complete.write_text(
        json.dumps(
            [
                _atm_question(
                    niah_evidence_ids=[
                        "email202411160004",
                        "20250223_130249",
                        "20220430_132212",
                    ]
                )
            ]
        ),
        encoding="utf-8",
    )
    pool = load_atm_niah_pool(complete)
    assert len(pool[0].niah_evidence_ids) == 3

    truncated = tmp_path / "broken.json"
    truncated.write_text(
        json.dumps([_atm_question(niah_evidence_ids=["20250223_130249"])]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pool must contain every gold evidence"):
        load_atm_niah_pool(truncated)


def test_atm_email_adapter_reads_naive_timestamps_as_utc(tmp_path: Path) -> None:
    emails_path = tmp_path / "emails.json"
    emails_path.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel confirmation",
                    "detail": "Total £799.74 for four nights.",
                }
            ]
        ),
        encoding="utf-8",
    )

    emails = load_atm_emails(emails_path)

    assert emails[0].email_id == "email202411160004"
    assert emails[0].occurred_at == datetime(2024, 11, 16, 9, 12, tzinfo=timezone.utc)
    assert emails[0].summary == "Hotel confirmation"


def test_atm_email_block_reproduces_the_official_field_order(tmp_path: Path) -> None:
    emails_path = tmp_path / "emails.json"
    emails_path.write_text(
        json.dumps(
            [
                {
                    "id": "email202411160004",
                    "timestamp": "2024-11-16 09:12:00",
                    "short_summary": "Hotel confirmation",
                    "detail": "Total £799.74 for four nights.",
                }
            ]
        ),
        encoding="utf-8",
    )

    block = atm_email_block(load_atm_emails(emails_path)[0])

    assert block.splitlines() == [
        "ID: email202411160004",
        "Timestamp: 2024-11-16 09:12:00",
        "Summary: Hotel confirmation",
        "Detail: Total £799.74 for four nights.",
    ]


def test_atm_sgm_adapter_takes_capture_time_from_the_stem_not_the_timestamp(
    tmp_path: Path,
) -> None:
    batch_path = tmp_path / "video_batch_results.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "video_path": "data/raw_memory/video/20220502_172850.mp4",
                    "timestamp": "2022-05-02 16:28:54+00:00",
                    "location_name": "Fellows' Garden, Cambridge, United Kingdom",
                    "city": "Cambridge, United Kingdom",
                    "short_caption": "A blackbird forages on a lawn.",
                    "caption": "A solitary blackbird moves across a green lawn.",
                    "ocr_text": "",
                    "tags": ["blackbird", "garden"],
                    "duration": 3.300756,
                    "file_size": 790569,
                    "entities": [{"entity": "bird", "type": "other"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_atm_sgm(batch_path)

    assert records[0].media_id == "20220502_172850"
    assert records[0].media_kind is MediaKind.VIDEO
    # The stem is local wall clock; the record's own timestamp is UTC and an hour earlier.
    assert records[0].occurred_at == datetime(2022, 5, 2, 17, 28, 50, tzinfo=timezone.utc)
    assert records[0].duration_seconds == pytest.approx(3.300756)
    assert records[0].size_bytes == 790569
    assert records[0].tags == ("blackbird", "garden")


def test_atm_sgm_adapter_tolerates_a_disambiguated_media_stem(tmp_path: Path) -> None:
    # 28 of the release's 3,759 real image filenames carry a disambiguating suffix that its
    # export tooling appends to a duplicate capture, e.g. `20221212_115316_001.jpg` or
    # `20220627_155122(0).jpg`. Only the leading 15 characters are the timestamp.
    batch_path = tmp_path / "image_batch_results.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "image_path": "data/raw_memory/image/20221212_115316_001.jpg",
                    "timestamp": "2022-12-12 11:53:16",
                    "location_name": "",
                    "city": "",
                    "short_caption": "",
                    "caption": "",
                    "ocr_text": "",
                    "tags": [],
                    "file_size": 12_345,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_atm_sgm(batch_path)

    assert records[0].media_id == "20221212_115316_001"
    assert records[0].occurred_at == datetime(2022, 12, 12, 11, 53, 16, tzinfo=timezone.utc)


def test_atm_sgm_block_reproduces_the_official_field_order(tmp_path: Path) -> None:
    batch_path = tmp_path / "image_batch_results.json"
    batch_path.write_text(
        json.dumps(
            [
                {
                    "image_path": "data/raw_memory/image/20220703_210745.jpg",
                    "timestamp": "2022-07-03 21:07:45",
                    "location_name": "West Quay Road, Southampton, United Kingdom",
                    "city": "Southampton, United Kingdom",
                    "short_caption": "A small airplane against a clear sky.",
                    "caption": "A solitary small aircraft streaks across a cloudless sky.",
                    "ocr_text": "There is no text visible in the image.",
                    "tags": ["airplane", "sky"],
                    "file_size": 100686,
                }
            ]
        ),
        encoding="utf-8",
    )

    block = atm_sgm_block(load_atm_sgm(batch_path)[0])

    assert block.splitlines() == [
        "ID: 20220703_210745",
        "Type: image",
        "Timestamp: 2022-07-03 21:07:45",
        "Location: West Quay Road, Southampton, United Kingdom",
        "Short Caption: A small airplane against a clear sky.",
        "Caption: A solitary small aircraft streaks across a cloudless sky.",
        "OCR: There is no text visible in the image.",
        "Tags: airplane, sky",
    ]


def test_atm_memory_chunks_keep_every_chunk_addressable_and_within_the_limit() -> None:
    short_block = "ID: 20220703_210745\nType: image\n"
    assert atm_memory_chunks(short_block, "20220703_210745") == (short_block,)

    long_block = "ID: 20220703_210745\n" + "x" * 9_000
    chunks = atm_memory_chunks(long_block, "20220703_210745")

    assert len(chunks) == 5
    assert all(len(chunk) <= 2_048 for chunk in chunks)
    assert all(chunk.startswith("ID: 20220703_210745\n") for chunk in chunks)
    assert "Part 1/5" in chunks[0]


def _mem_gallery_topic_payload() -> dict[str, object]:
    return {
        "character_profile": {
            "name": "Maya",
            "persona_summary": "A part-time librarian who took up baking.",
            "traits": ["curious", "earnest"],
            "conversation_style": "Inquisitive and earnest.",
        },
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2024-06-24",
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "Can you tell me the basics of handmade baking?",
                        "assistant": "Start with an oven of 30 litres or more.",
                    },
                    {
                        "round": "D1:2",
                        "user": "What is in this picture?",
                        "assistant": "A tray of shortbread.",
                        "image_id": ["D1:IMG_001"],
                        "input_image": ["../image/Baking/D1_IMG_001.jpg"],
                        "image_caption": ["A tray of pale shortbread fingers."],
                    },
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "point": "FR",
                "question": "What oven size was recommended?",
                "answer": "30 litres or more.",
                "session_id": ["D1"],
                "clue": ["D1:1"],
            },
            {
                "point": "TTL",
                "question": "What species of plant is shown in the picture?",
                "question_image": "../image/Baking/QA_IMG_001.jpg",
                "answer": "Foxglove",
                "session_id": ["D1"],
                "clue": ["D1:2"],
                "image_caption": "Cluster of purple bell-shaped flowers.",
            },
        ],
    }


def test_mem_gallery_adapter_reads_sessions_rounds_and_question_images(tmp_path: Path) -> None:
    topic_path = tmp_path / "Baking_Dessert_Daily_Life_Skill.json"
    topic_path.write_text(json.dumps(_mem_gallery_topic_payload()), encoding="utf-8")

    topic = load_mem_gallery_topic(topic_path)

    assert topic.topic == "Baking_Dessert_Daily_Life_Skill"
    assert topic.profile.name == "Maya"
    assert topic.sessions[0].session_id == "D1"
    assert topic.sessions[0].occurred_at == datetime(2024, 6, 24, tzinfo=timezone.utc)
    assert topic.sessions[0].rounds[0].image_id is None
    assert topic.sessions[0].rounds[1].image_id == "D1:IMG_001"
    assert topic.sessions[0].rounds[1].image_path == "../image/Baking/D1_IMG_001.jpg"
    assert topic.questions[0].question_id == "Baking_Dessert_Daily_Life_Skill:1"
    assert topic.questions[0].point == "FR"
    assert topic.questions[0].clue_round_ids == ("D1:1",)
    assert topic.questions[1].question_image_path == "../image/Baking/QA_IMG_001.jpg"
    assert topic.questions[1].question_image_caption == "Cluster of purple bell-shaped flowers."


def test_mem_gallery_adapter_refuses_unknown_points_and_dangling_clues(tmp_path: Path) -> None:
    unknown_point = _mem_gallery_topic_payload()
    qas = unknown_point["human-annotated QAs"]
    assert isinstance(qas, list)
    qas[0]["point"] = "ZZ"
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown_point), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown Mem-Gallery point"):
        load_mem_gallery_topic(unknown_path)

    dangling = _mem_gallery_topic_payload()
    dangling_qas = dangling["human-annotated QAs"]
    assert isinstance(dangling_qas, list)
    dangling_qas[0]["clue"] = ["D9:7"]
    dangling_path = tmp_path / "dangling.json"
    dangling_path.write_text(json.dumps(dangling), encoding="utf-8")
    with pytest.raises(ValueError, match="clue names an unknown round"):
        load_mem_gallery_topic(dangling_path)


def test_mem_gallery_adapter_refuses_a_round_carrying_more_than_one_image(
    tmp_path: Path,
) -> None:
    payload = _mem_gallery_topic_payload()
    sessions = payload["multi_session_dialogues"]
    assert isinstance(sessions, list)
    sessions[0]["dialogues"][1]["input_image"] = ["a.jpg", "b.jpg"]
    sessions[0]["dialogues"][1]["image_id"] = ["D1:IMG_001", "D1:IMG_002"]
    path = tmp_path / "two_images.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one image"):
        load_mem_gallery_topic(path)


def test_mem_gallery_directory_loader_keeps_sorted_topic_order(tmp_path: Path) -> None:
    directory = tmp_path / "dialog"
    directory.mkdir()
    for name in ("Zebra_Topic", "Apple_Topic"):
        (directory / f"{name}.json").write_text(
            json.dumps(_mem_gallery_topic_payload()), encoding="utf-8"
        )

    topics = load_mem_gallery(directory)

    assert tuple(topic.topic for topic in topics) == ("Apple_Topic", "Zebra_Topic")
