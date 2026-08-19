"""Tests for the LoCoMo-Refined -> AML case loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mindbridge.benchmarks.aml.loaders.locomo_refined import load

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
            {"question": "Q1?", "answer": ["A1"], "evidence": ["D2:1"], "category": 1},
            # Six golds in the real release are published as numbers.
            {"question": "Q2?", "answer": [2022], "evidence": ["D10:1"], "category": 2},
            # Multi-candidate golds are alternatives, not a required set.
            {
                "question": "Q3?",
                "answer": ["A3", "also A3"],
                "evidence": [],
                "category": 4,
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
            {"question": "Q1?", "answer": ["A1"], "evidence": ["D1:1"], "category": 1},
        ],
    },
]


def _write_fixture(tmp_path: Path, records: list[dict[str, Any]] | None = None) -> Path:
    path = tmp_path / "locomo_refined.json"
    path.write_text(json.dumps(records if records is not None else _FIXTURE))
    return path


def test_load_orders_sessions_numerically_and_maps_roles(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))

    assert len(cases) == 2
    conv1, conv2 = cases

    assert conv1.user_id == "locomo-refined:conv-1"
    assert [message["content"] for message in conv1.messages] == [
        "hi from alice",
        "hi from bob",
        "final from alice",
    ]
    assert [message["role"] for message in conv1.messages] == ["user", "assistant", "user"]
    # Epoch milliseconds, UTC -- "10:00 am on 1 January, 2024" / "11:00 am on
    # 2 January, 2024" (AML's wire contract requires `int`, not free text).
    assert conv1.messages[0]["timestamp"] == 1704103200000
    assert conv1.messages[2]["timestamp"] == 1704193200000

    assert conv2.user_id == "locomo-refined:conv-2"
    assert [message["role"] for message in conv2.messages] == ["assistant"]


def test_load_carries_the_official_qa_id_and_a_single_gold_answer(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path))
    conv1 = cases[0]

    # `{sample_id}#q{index:04d}` is the release's own `qa_id`, so a row this run writes
    # joins straight onto `data/public/questions.jsonl` -- and `driver.row_id` prefers
    # the payload `id` over its own format, which the CLI's resume check reads back.
    assert [question.question_id for question in conv1.questions] == [
        "conv-1#q0000",
        "conv-1#q0001",
        "conv-1#q0002",
    ]
    assert conv1.questions[0].question == "Q1?"
    assert conv1.questions[0].payload == {"id": "conv-1#q0000", "gold_answer": "A1"}
    # Numeric golds are stringified, the way `data/public/questions.jsonl` publishes them.
    assert conv1.questions[1].payload == {"id": "conv-1#q0001", "gold_answer": "2022"}
    # The vendored pipeline renders a list gold by joining it with newlines, which its own
    # judge reads as "cover all of these"; alternatives would score as a required set, so
    # only the first candidate is handed over.
    assert conv1.questions[2].payload == {"id": "conv-1#q0002", "gold_answer": "A3"}


def test_load_refuses_a_question_with_no_gold_answer(tmp_path: Path) -> None:
    """LoCoMo-Refined dropped the adversarial category, so an empty gold is corruption."""
    records = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "10:00 am on 1 January, 2024",
                "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "hi"}],
            },
            "qa": [{"question": "Q1?", "answer": [], "evidence": [], "category": 1}],
        }
    ]

    with pytest.raises(ValidationError):
        load(_write_fixture(tmp_path, records))
