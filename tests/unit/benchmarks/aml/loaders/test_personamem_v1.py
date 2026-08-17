"""Tests for the PersonaMem v1 -> AML case loader."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mindbridge.benchmarks.aml.loaders.personamem_v1 import load

# Persona "ctx-1" has 5 messages; its two questions were generated against
# different prefixes of that same shared context (2 messages, then 4) -- they
# must land in two separate cases, one per truncation point.
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


def test_load_splits_one_scope_into_one_case_per_truncation_point(tmp_path: Path) -> None:
    """Would fail under the old "one case per shared_context_id" grouping.

    Under that grouping, ctx-1's two questions (end indices 2 and 4) would
    collapse into a single case, and whichever message list won would either
    starve the shorter question of nothing extra or leak later messages into
    it. Splitting by (shared_context_id, end_index) keeps them apart.
    """
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)

    assert len(cases) == 3
    ctx1_cases = [case for case in cases if case.user_id.startswith("personamem-v1:ctx-1:")]
    assert len(ctx1_cases) == 2

    short_case, long_case = sorted(ctx1_cases, key=lambda case: len(case.messages))
    assert short_case.user_id == "personamem-v1:ctx-1:2"
    assert long_case.user_id == "personamem-v1:ctx-1:4"
    assert short_case.user_id != long_case.user_id
    assert len(short_case.messages) == 2
    assert len(long_case.messages) == 4

    # The shorter case's messages are an exact prefix of the longer case's.
    short_contents = [message["content"] for message in short_case.messages]
    long_contents = [message["content"] for message in long_case.messages]
    assert long_contents[: len(short_contents)] == short_contents

    [short_question] = short_case.questions
    [long_question] = long_case.questions
    assert short_question.question_id == "q-0001"
    assert long_question.question_id == "q-0002"


def test_load_truncates_messages_to_the_case_group_end_index(tmp_path: Path) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case = next(case for case in cases if case.user_id == "personamem-v1:ctx-1:2")

    assert [message["content"] for message in case.messages] == [
        "User: hi, I'm planning a trip",
        "Assistant: sounds fun, where to?",
    ]
    assert [message["role"] for message in case.messages] == ["user", "assistant"]
    # PersonaMem messages carry no timestamps at all.
    assert all(message["timestamp"] is None for message in case.messages)


def test_load_builds_the_official_payload_and_keeps_all_options_a_string(
    tmp_path: Path,
) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case = next(case for case in cases if case.user_id == "personamem-v1:ctx-1:2")

    [question] = case.questions
    assert question.question_id == "q-0001"
    assert question.question == "Where is the user planning to travel?"
    assert question.payload["id"] == "q-0001"
    assert question.payload["correct_answer"] == "(c)"
    # This must stay a plain string -- the vendored pipeline raises TypeError
    # if `all_options` is anything else (e.g. a list of option strings).
    assert question.payload["all_options"] == "(a) France\n(b) Italy\n(c) Japan\n(d) Germany"
    assert isinstance(question.payload["all_options"], str)
    assert not isinstance(question.payload["all_options"], list)

    # end_index_in_shared_context is now expressed by the case split (the
    # user_id / message truncation), not duplicated into the payload.
    assert "end_index_in_shared_context" not in question.payload
    assert question.payload == {
        "id": "q-0001",
        "all_options": "(a) France\n(b) Italy\n(c) Japan\n(d) Germany",
        "correct_answer": "(c)",
    }


def test_load_scopes_second_persona_independently(tmp_path: Path) -> None:
    questions_csv, contexts_jsonl = _write_fixture(tmp_path)
    cases = load(questions_csv, contexts_jsonl)
    case = next(case for case in cases if case.user_id == "personamem-v1:ctx-2:1")

    assert [message["content"] for message in case.messages] == ["User: I like hiking"]
    [question] = case.questions
    assert question.payload == {
        "id": "q-0003",
        "all_options": "(a) swimming\n(b) hiking\n(c) chess\n(d) cooking",
        "correct_answer": "(b)",
    }
