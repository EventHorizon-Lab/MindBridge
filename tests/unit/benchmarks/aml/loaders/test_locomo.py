"""Tests for the LoCoMo -> AML case loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindbridge.benchmarks.aml.loaders.locomo import load

_FIXTURE: list[dict[str, Any]] = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            # Sessions 2 and 10 only (no 1..9): a lexical sort would put
            # "session_10" before "session_2" since "1" < "2" character-wise.
            "session_2_date_time": "10:00 am on 1 January, 2024",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "hi from alice"},
                {"speaker": "Bob", "dia_id": "D2:2", "text": "hi from bob"},
            ],
            "session_10_date_time": "11:00 am on 2 January, 2024",
            "session_10": [
                {"speaker": "Alice", "dia_id": "D10:1", "text": "final from alice"},
            ],
        },
        "qa": [
            {"question": "Q1?", "answer": "A1", "evidence": ["D2:1"], "category": 1},
            {"question": "Q2?", "answer": 42, "evidence": ["D10:1"], "category": 2},
            # Adversarial (category 5) questions in the real corpus drop
            # "answer" entirely and carry "adversarial_answer" instead.
            {
                "question": "Q3?",
                "adversarial_answer": "not mentioned",
                "evidence": [],
                "category": 5,
            },
        ],
    },
    {
        "sample_id": "conv-2",
        "conversation": {
            "speaker_a": "Carol",
            "speaker_b": "Dave",
            "session_1_date_time": "9:00 am on 3 January, 2024",
            "session_1": [
                {"speaker": "Dave", "dia_id": "D1:1", "text": "hi from dave"},
            ],
        },
        "qa": [
            {"question": "Q1?", "answer": "A1", "evidence": ["D1:1"], "category": 1},
        ],
    },
]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(_FIXTURE))
    return path


def test_load_orders_sessions_numerically_and_maps_roles(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))

    assert len(cases) == 2
    conv1, conv2 = cases

    assert conv1.user_id == "locomo:conv-1"
    assert [message["content"] for message in conv1.messages] == [
        "hi from alice",
        "hi from bob",
        "final from alice",
    ]
    assert [message["role"] for message in conv1.messages] == ["user", "assistant", "user"]
    assert conv1.messages[0]["timestamp"] == "10:00 am on 1 January, 2024"
    assert conv1.messages[2]["timestamp"] == "11:00 am on 2 January, 2024"

    assert conv2.user_id == "locomo:conv-2"
    assert [message["role"] for message in conv2.messages] == ["assistant"]


def test_load_synthesizes_stable_question_ids_and_renames_gold_answer(
    tmp_path: Path,
) -> None:
    cases = load(_write_fixture(tmp_path))
    conv1 = cases[0]

    assert [question.question_id for question in conv1.questions] == [
        "conv-1:qa0000",
        "conv-1:qa0001",
        "conv-1:qa0002",
    ]
    assert conv1.questions[0].question == "Q1?"
    assert conv1.questions[0].payload == {"gold_answer": "A1"}
    assert conv1.questions[1].payload == {"gold_answer": 42}
    # Category-5 questions with no "answer" key fall back to "adversarial_answer".
    assert conv1.questions[2].payload == {"gold_answer": "not mentioned"}
