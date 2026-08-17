"""Tests for the PersonaMem v2 -> AML case loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.aml.loaders.personamem_v2 import load

_HISTORY_LINK_1 = "data/chat_history_32k/chat_history_250911_persona1.json"
_HISTORY_LINK_2 = "data/chat_history_32k/chat_history_250911_persona2.json"

_HISTORIES: dict[str, list[dict[str, str]]] = {
    _HISTORY_LINK_1: [
        {"role": "system", "content": "You are helping persona 1."},
        {"role": "user", "content": "I love hiking on weekends."},
        {"role": "assistant", "content": "Noted, hiking is a favorite."},
    ],
    _HISTORY_LINK_2: [
        {"role": "user", "content": "I collect vintage cameras."},
        {"role": "assistant", "content": "Got it, vintage cameras."},
    ],
}

# Two rows for persona "1" (sharing one history), one row for persona "2" --
# mirrors the real corpus, where every row for a persona_id points at the
# same chat_history_32k_link (measured: 0/200 personas differ).
_ROWS: list[dict[str, str]] = [
    {
        "persona_id": "1",
        "chat_history_32k_link": _HISTORY_LINK_1,
        "user_query": "{'role': 'user', 'content': 'What outdoor activity do I enjoy?'}",
        "correct_answer": "hiking",
        "incorrect_answers": json.dumps(["running", "swimming", "cycling"]),
        "preference": "Enjoys hiking on weekends",
    },
    {
        "persona_id": "1",
        "chat_history_32k_link": _HISTORY_LINK_1,
        "user_query": "{'role': 'user', 'content': 'When do I like to hike?'}",
        "correct_answer": "weekends",
        "incorrect_answers": json.dumps(["weekdays", "mornings only", "never"]),
        "preference": "Do not remember 'favorite hiking trail'",
    },
    {
        "persona_id": "2",
        "chat_history_32k_link": _HISTORY_LINK_2,
        "user_query": "{'role': 'user', 'content': 'What do I collect?'}",
        "correct_answer": "vintage cameras",
        "incorrect_answers": json.dumps(["stamps", "coins", "comic books"]),
        "preference": "Collects vintage cameras",
    },
]


def _write_histories(data_root: Path, histories: dict[str, list[dict[str, str]]]) -> None:
    for link, messages in histories.items():
        history_path = data_root / link
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps({"metadata": {}, "chat_history": messages}))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path
    _write_histories(data_root, _HISTORIES)
    benchmark_csv = tmp_path / "benchmark.csv"
    _write_csv(benchmark_csv, _ROWS)
    return benchmark_csv, data_root


def test_load_scopes_one_case_per_persona(tmp_path: Path) -> None:
    """Measured on the real corpus: 0/200 personas have >1 distinct
    chat_history_32k_link within benchmark.csv, so persona_id alone is a
    safe (and correct) retrieval scope here -- no snapshot suffix needed.
    """
    benchmark_csv, data_root = _write_fixture(tmp_path)
    cases = load(benchmark_csv, data_root)

    assert len(cases) == 2
    user_ids = sorted(case.user_id for case in cases)
    assert user_ids == ["personamem-v2:1", "personamem-v2:2"]

    persona1 = next(case for case in cases if case.user_id == "personamem-v2:1")
    persona2 = next(case for case in cases if case.user_id == "personamem-v2:2")
    assert len(persona1.questions) == 2
    assert len(persona2.questions) == 1


def test_load_resolves_history_through_the_csv_link_column(tmp_path: Path) -> None:
    benchmark_csv, data_root = _write_fixture(tmp_path)
    cases = load(benchmark_csv, data_root)

    persona1 = next(case for case in cases if case.user_id == "personamem-v2:1")
    assert [message["content"] for message in persona1.messages] == [
        "You are helping persona 1.",
        "I love hiking on weekends.",
        "Noted, hiking is a favorite.",
    ]
    assert [message["role"] for message in persona1.messages] == [
        "system",
        "user",
        "assistant",
    ]
    # No per-message timestamps exist anywhere in this dataset.
    assert all(message["timestamp"] is None for message in persona1.messages)


def test_load_synthesizes_stable_ids_from_file_read_order(tmp_path: Path) -> None:
    benchmark_csv, data_root = _write_fixture(tmp_path)
    cases = load(benchmark_csv, data_root)

    persona1 = next(case for case in cases if case.user_id == "personamem-v2:1")
    question_ids = [question.question_id for question in persona1.questions]
    # Global CSV row order: persona "1"'s two rows are rows 0 and 1.
    assert question_ids == ["persona1-0", "persona1-1"]
    assert all(question.payload["id"] == question.question_id for question in persona1.questions)

    persona2 = next(case for case in cases if case.user_id == "personamem-v2:2")
    [question] = persona2.questions
    # Persona "2"'s only row is CSV row 2.
    assert question.question_id == "persona2-2"


def test_load_builds_the_official_payload(tmp_path: Path) -> None:
    benchmark_csv, data_root = _write_fixture(tmp_path)
    cases = load(benchmark_csv, data_root)

    persona1 = next(case for case in cases if case.user_id == "personamem-v2:1")
    [first, second] = persona1.questions

    assert first.question == "What outdoor activity do I enjoy?"
    assert first.payload == {
        "id": "persona1-0",
        "user_query": "{'role': 'user', 'content': 'What outdoor activity do I enjoy?'}",
        "correct_answer": "hiking",
        "incorrect_answers": ["running", "swimming", "cycling"],
        "persona_id": "1",
        "preference": "Enjoys hiking on weekends",
    }
    assert isinstance(first.payload["incorrect_answers"], list)

    # evaluate-narrow picks its judge prompt off whether this starts with
    # "do not" -- dropping this key would silently change the grading rubric.
    assert second.payload["preference"] == "Do not remember 'favorite hiking trail'"


def test_load_raises_if_a_persona_ever_points_at_two_different_histories(
    tmp_path: Path,
) -> None:
    """The real corpus never hits this (measured: 0/200 personas), but if a
    future snapshot does, silently picking one link would leak one
    snapshot's content into another's questions -- fail loudly instead.
    """
    rows = [dict(_ROWS[0]), dict(_ROWS[0])]
    rows[1]["chat_history_32k_link"] = _HISTORY_LINK_2

    data_root = tmp_path
    _write_histories(data_root, _HISTORIES)
    benchmark_csv = tmp_path / "benchmark.csv"
    _write_csv(benchmark_csv, rows)

    with pytest.raises(ValueError, match="more than one"):
        load(benchmark_csv, data_root)
