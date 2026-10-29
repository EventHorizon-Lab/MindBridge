"""ATM numeric and temporal questions use the shared official instruction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.eval_adapters import load_task
from mindbridge.benchmarks.task_catalog import TASKS


@pytest.mark.parametrize(
    "question",
    (
        "What is the deadline for the application?",
        "Which day is the workshop?",
        "At what time is pickup?",
        "How many tickets did I buy?",
        "What is the price range, in pounds?",
    ),
)
def test_atm_number_query_preserves_the_complete_requested_value(
    tmp_path: Path, question: str
) -> None:
    spec = TASKS["atm-bench-main"]
    dataset = spec.dataset_path(tmp_path)
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": question,
                    "answer": "reference must remain scorer-only",
                    "qtype": "number",
                    "evidence_ids": ["email0001"],
                }
            ]
        ),
        encoding="utf-8",
    )
    emails = tmp_path / spec.auxiliary[0]
    emails.parent.mkdir(parents=True)
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email0001",
                    "timestamp": "2024-01-01 00:00:00",
                    "short_summary": "Fixture",
                    "detail": "Unrelated memory.",
                }
            ]
        ),
        encoding="utf-8",
    )
    media = tmp_path / "fixture-media"
    media.mkdir()
    (media / "20240101_000000.jpg").write_bytes(b"fixture")

    loaded = load_task(spec, root=tmp_path, media_root=media, verify_digest=False)

    (unit,) = loaded.units
    (adapted,) = unit.questions
    assert adapted.content == (
        question,
        "You are a QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with "
        "the corresponding evidence IDs only, comma-separated, with no extra text.",
    )
    assert adapted.source_question == question
    assert adapted.references == ("reference must remain scorer-only",)
    assert adapted.metadata == {"qtype": "number", "evidence_ids": ("email0001",)}
