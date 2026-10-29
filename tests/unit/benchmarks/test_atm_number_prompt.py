"""ATM's `number` label includes complete temporal and numeric values."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.eval_adapters import load_task
from mindbridge.benchmarks.prompts import ATM_BENCH_NUMBER_FORMAT_PROMPT
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
    assert ATM_BENCH_NUMBER_FORMAT_PROMPT.version == "atm_bench_number_format_v2"
    assert adapted.content == (
        question,
        "Answer with the complete value requested by the question: number, date, time, or "
        "range, with any required unit or currency symbol. Preserve all requested components; "
        "do not add unrequested components or explanations.",
    )
    assert adapted.source_question == question
    assert adapted.references == ("reference must remain scorer-only",)
    assert adapted.metadata == {"qtype": "number", "evidence_ids": ("email0001",)}
