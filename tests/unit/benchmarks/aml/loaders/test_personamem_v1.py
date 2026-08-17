"""Tests for the PersonaMem v1 -> AML case loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mindbridge.benchmarks.aml.loaders.personamem_v1 import load

# Persona "ctx-1" has 5 messages; its two questions were generated against
# different prefixes of that same shared context (2 messages, then 4).
_CONTEXTS = {
    "ctx-1": [
        {"role": "user", "content": "User: hi, I'm planning a trip"},
        {"role": "assistant", "content": "Assistant: sounds fun, where to?"},
        {"role": "user", "content": "User: maybe Japan"},
        {"role": "assistant", "content": "Assistant: great choice"},
        {"role": "user", "content": "User: what should I pack"},
    ],
    "ctx-2": [
        {"role": "user", "content": "User: I like hiking"},
        {"role": "assistant", "content": "Assistant: noted"},
    ],
}

_QUESTIONS = [
    {
        "persona_id": "persona-1",
        "question_id": "q-0001",
        "user_question_or_message": "Where is the user planning to travel?",
        "correct_answer": "(c)",
        "all_options": "(a) France\n(b) Italy\n(c) Japan\n(d) Germany",
        "shared_context_id": "ctx-1",
        "end_index_in_shared_context": "2",
    },
    {
        "persona_id": "persona-1",
        "question_id": "q-0002",
        "user_question_or_message": "What is the assistant helping the user pack for?",
        "correct_answer": "(a)",
        "all_options": "(a) a trip to Japan\n(b) a hike\n(c) a party\n(d) work",
        "shared_context_id": "ctx-1",
        "end_index_in_shared_context": "4",
    },
    {
        "persona_id": "persona-2",
        "question_id": "q-0003",
        "user_question_or_message": "What hobby does the user like?",
        "correct_answer": "(b)",
        "all_options": "(a) swimming\n(b) hiking\n(c) chess\n(d) cooking",
        "shared_context_id": "ctx-2",
        "end_index_in_shared_context": "1",
    },
]


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    contexts_path = tmp_path / "shared_contexts_32k.jsonl"
    with contexts_path.open("w") as handle:
        for shared_context_id, turns in _CONTEXTS.items():
            handle.write(json.dumps({shared_context_id: turns}) + "\n")

    questions_path = tmp_path / "questions_32k.csv"
    with questions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_QUESTIONS[0].keys()))
        writer.writeheader()
        writer.writerows(_QUESTIONS)

    return questions_path, contexts_path


def test_load_groups_questions_by_shared_context_id(tmp_path: Path) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)

    assert len(cases) == 2
    case1, case2 = cases

    assert case1.user_id == "personamem-v1:ctx-1"
    assert len(case1.questions) == 2
    assert case2.user_id == "personamem-v1:ctx-2"
    assert len(case2.questions) == 1


def test_load_carries_the_full_untruncated_shared_context_per_case(tmp_path: Path) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case1 = cases[0]

    # The case-level history is the FULL persona conversation, not truncated to
    # any single question's end_index -- see the module docstring for why.
    assert [message["content"] for message in case1.messages] == [
        "User: hi, I'm planning a trip",
        "Assistant: sounds fun, where to?",
        "User: maybe Japan",
        "Assistant: great choice",
        "User: what should I pack",
    ]
    assert [message["role"] for message in case1.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    # PersonaMem messages carry no timestamps at all.
    assert all(message["timestamp"] is None for message in case1.messages)


def test_load_builds_the_official_payload_and_keeps_all_options_a_string(
    tmp_path: Path,
) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case1 = cases[0]

    q1, q2 = case1.questions
    assert q1.question_id == "q-0001"
    assert q1.question == "Where is the user planning to travel?"
    assert q1.payload["id"] == "q-0001"
    assert q1.payload["correct_answer"] == "(c)"
    # This must stay a plain string -- the vendored pipeline raises TypeError
    # if `all_options` is anything else (e.g. a list of option strings).
    assert q1.payload["all_options"] == "(a) France\n(b) Italy\n(c) Japan\n(d) Germany"
    assert isinstance(q1.payload["all_options"], str)
    assert not isinstance(q1.payload["all_options"], list)

    # Different questions in the same scope truncate the shared context at
    # different points -- that per-question cut point rides along in the
    # payload so a driver can slice `case.messages` correctly per question.
    assert q1.payload["end_index_in_shared_context"] == 2
    assert q2.payload["end_index_in_shared_context"] == 4


def test_load_scopes_second_persona_independently(tmp_path: Path) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case2 = cases[1]

    assert [message["content"] for message in case2.messages] == [
        "User: I like hiking",
        "Assistant: noted",
    ]
    [question] = case2.questions
    assert question.payload == {
        "id": "q-0003",
        "all_options": "(a) swimming\n(b) hiking\n(c) chess\n(d) cooking",
        "correct_answer": "(b)",
        "end_index_in_shared_context": 1,
    }
