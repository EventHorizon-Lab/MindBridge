"""Tests for the LongMemEval -> AML case loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindbridge.benchmarks.aml.loaders.longmemeval import load

_FIXTURE: list[dict[str, Any]] = [
    {
        "question_id": "e47becba",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": [
            "2023/05/20 (Sat) 02:21",
            "2023/05/21 (Sun) 10:05",
        ],
        "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "distractor session, irrelevant"},
                {"role": "assistant", "content": "sure, here you go"},
            ],
            [
                {"role": "user", "content": "I graduated with a degree in Business Administration"},
                {"role": "assistant", "content": "Congratulations!"},
            ],
        ],
        "answer_session_ids": ["85a1be56_1"],
    },
    {
        "question_id": "a1b2c3d4",
        "question_type": "temporal-reasoning",
        "question": "How many cats do I have?",
        "answer": 3,
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_dates": ["2023/05/15 (Mon) 08:00"],
        "haystack_session_ids": ["sess_0"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I have three cats"},
            ],
        ],
        "answer_session_ids": ["sess_0"],
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "longmemeval_s"
    path.write_text(json.dumps(_FIXTURE))
    return path


def test_load_makes_one_case_per_question_with_its_own_haystack(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))

    assert len(cases) == 2
    case1, case2 = cases

    assert case1.user_id == "longmemeval:e47becba"
    assert len(case1.questions) == 1
    assert case2.user_id == "longmemeval:a1b2c3d4"


def test_load_flattens_sessions_in_order_with_session_level_timestamps(
    tmp_path: Path,
) -> None:
    cases = load(_write_fixture(tmp_path))
    case1 = cases[0]

    assert [message["content"] for message in case1.messages] == [
        "distractor session, irrelevant",
        "sure, here you go",
        "I graduated with a degree in Business Administration",
        "Congratulations!",
    ]
    assert [message["role"] for message in case1.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # Epoch milliseconds, UTC -- "2023/05/20 (Sat) 02:21" / "2023/05/21 (Sun)
    # 10:05" (AML's wire contract requires `int`, not free text).
    assert case1.messages[0]["timestamp"] == 1684549260000
    assert case1.messages[2]["timestamp"] == 1684663500000


def test_load_renames_answer_to_gold_answer_and_sets_id(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))
    case1, case2 = cases

    [question1] = case1.questions
    assert question1.question_id == "e47becba"
    assert question1.question == "What degree did I graduate with?"
    assert question1.payload == {"id": "e47becba", "gold_answer": "Business Administration"}

    [question2] = case2.questions
    assert question2.payload == {"id": "a1b2c3d4", "gold_answer": 3}


def test_load_omits_speaker_2_keys(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))
    [question] = cases[0].questions

    assert "speaker_2_name" not in question.payload
    assert "speaker_2_memories" not in question.payload
