"""Durability and recovery checks for the opt-in benchmark result journal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from mindbridge.benchmarks.eval_journal import (
    DurableResultJournal,
    load_journal,
    merge_finalized_samples,
)


def _sample(
    question_id: str, prediction: str = "A", *, error: str | None = None
) -> dict[str, object]:
    return {
        "schema_version": 14,
        "sample_id": f"egolifeqa/A1_JAKE/{question_id}",
        "arm": "mindbridge",
        "task": "egolifeqa",
        "benchmark": "EgoLifeQA",
        "dataset_sha256": "dataset",
        "evaluation_sha256": "evaluation",
        "unit_id": "A1_JAKE",
        "question_id": question_id,
        "prediction": "" if error else prediction,
        "parsed_choice": None,
        "score": None,
        "exact_match": None,
        "latency_ms": 12.5,
        "confidence": 0.5,
        "memory_ids": [],
        "error_code": error,
        "error_reason": "fixture" if error else None,
        "error_stage": "generate" if error else None,
        "error_cause_type": "RuntimeError" if error else None,
        "metadata": {
            "day": 1,
            "question_type": "EventRecall",
            "needs_audio": False,
            "choices": ["one", "two", "three", "four"],
        },
    }


def _record_line(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    wrapped = {**payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}
    return (
        json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )


def test_journal_fsyncs_each_unique_generation_and_error(tmp_path: Path) -> None:
    usage_counts: list[int] = []

    def usage(_sample: object, completed: int) -> dict[str, object]:
        usage_counts.append(completed)
        return {"request_count": completed}

    path = tmp_path / "generation.journal.jsonl"
    journal = DurableResultJournal(
        path,
        run_id="run-1",
        attempt_id="attempt-1",
        usage_snapshot=usage,
    )
    journal.append(_sample("1"), expected_label="A")
    journal.append(_sample("2", error="model_error"), expected_label="B")

    records = load_journal(path)
    assert usage_counts == [1, 2]
    assert [record.payload["sequence"] for record in records] == [1, 2]
    assert [record.payload["outcome"] for record in records] == ["generation", "error"]
    assert [record.payload["expected_label"] for record in records] == ["A", "B"]
    assert records[0].payload["roster"] == {
        "question_id": "1",
        "expected_label": "A",
        "day": 1,
        "question_type": "EventRecall",
        "needs_audio": False,
    }
    assert records[1].payload["cumulative_usage"] == {"request_count": 2}
    assert records[0].identity == (
        "attempt-1",
        "run-1",
        "egolifeqa",
        "A1_JAKE",
        "1",
        "mindbridge",
    )
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="duplicate journal sample identity"):
        journal.append(_sample("1"), expected_label="A")


def test_loader_ignores_only_one_unterminated_tail(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = DurableResultJournal(path, run_id="run", attempt_id="attempt")
    journal.append(_sample("1"), expected_label="A")
    path.write_bytes(path.read_bytes() + b'{"schema_version":1')

    records = load_journal(path)

    assert [record.identity[4] for record in records] == ["1"]


@pytest.mark.parametrize("corruption", [b"not-json\n", b'{"schema_version":1}\n'])
def test_loader_rejects_complete_corruption_before_a_truncated_tail(
    tmp_path: Path, corruption: bytes
) -> None:
    path = tmp_path / "journal.jsonl"
    journal = DurableResultJournal(path, run_id="run", attempt_id="attempt")
    journal.append(_sample("1"), expected_label="A")
    path.write_bytes(path.read_bytes() + corruption + b"truncated")

    with pytest.raises(ValueError, match="journal"):
        load_journal(path)


def test_loader_rejects_even_checksum_valid_identical_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = DurableResultJournal(path, run_id="run", attempt_id="attempt")
    journal.append(_sample("1"), expected_label="A")
    first = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    first.pop("payload_sha256")
    first["sequence"] = 2
    path.write_bytes(path.read_bytes() + _record_line(first))

    with pytest.raises(ValueError, match="duplicate journal sample identity at line 2"):
        load_journal(path)


def test_finalized_sample_wins_only_when_raw_prediction_and_error_agree(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = DurableResultJournal(path, run_id="run", attempt_id="attempt")
    journal.append(_sample("1"), expected_label="A")
    records = load_journal(path)
    finalized = {**_sample("1"), "parsed_choice": "A", "score": 1.0}

    assert merge_finalized_samples(records, (finalized,), run_id="run", attempt_id="attempt") == (
        finalized,
    )

    with pytest.raises(ValueError, match="finalized sample conflicts with journal"):
        merge_finalized_samples(
            records,
            ({**finalized, "prediction": "B"},),
            run_id="run",
            attempt_id="attempt",
        )
    with pytest.raises(ValueError, match="duplicate finalized sample identity"):
        merge_finalized_samples(
            (),
            (finalized, finalized),
            run_id="run",
            attempt_id="attempt",
        )
