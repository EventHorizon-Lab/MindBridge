"""Tests for the CL-Bench -> AML case loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.aml.loaders.clbench import load

# Record 1: a clean 2-message record whose final (only) user turn has one
# blank-line paragraph break -- the common case (measured: 69.6% of the real
# corpus splits this way).
_RECORD_1 = {
    "messages": [
        {"role": "system", "content": "You are FactBot."},
        {
            "role": "user",
            "content": (
                "Freedonia is a fictional country used in economics textbooks. "
                "It has a long history of trade disputes.\n\n"
                "What is Freedonia primarily used to illustrate in economics textbooks?"
            ),
        },
    ],
    "rubrics": ["Mentions trade disputes or an economics illustration"],
    "metadata": {
        "task_id": "task-1",
        "context_id": "ctx-1",
        "context_category": "Trivia",
        "sub_category": "Facts",
    },
}

# Record 2: a 4-message record (system, user, assistant, user) whose final
# user turn is a short one-sentence follow-up with no blank-line break at
# all -- the documented fallback path (the whole turn becomes the question,
# harmless here because it is already short).
_RECORD_2 = {
    "messages": [
        {"role": "system", "content": "Persona: helpful research assistant."},
        {
            "role": "user",
            "content": "Zorblatt Inc. was founded in 2001 by Jane Doe in Springfield.",
        },
        {"role": "assistant", "content": "Understood, I'll remember that about Zorblatt Inc."},
        {"role": "user", "content": "Who founded Zorblatt Inc.?"},
    ],
    "rubrics": ["Names Jane Doe"],
    "metadata": {
        "task_id": "task-2",
        "context_id": "ctx-2",
        "context_category": "Business",
        "sub_category": "Corporate History",
    },
}

# Record 3: a 2-message record whose final user turn has a *long* history
# section -- three blank-line-separated paragraphs -- preceding a short
# query, per the fixture policy ("one whose last user turn has a long
# history section preceding a short query").
_RECORD_3 = {
    "messages": [
        {"role": "system", "content": "You are DocQA."},
        {
            "role": "user",
            "content": (
                "Section 1: The city of Lumenport was founded in 1750 by merchant "
                "sailors.\n\n"
                "Section 2: Its main export was spiced glassware, prized across "
                "three continents.\n\n"
                "Section 3: The Great Fire of 1812 destroyed much of the old "
                "harbor district.\n\n"
                "When was Lumenport founded?"
            ),
        },
    ],
    "rubrics": ["States 1750"],
    "metadata": {
        "task_id": "task-3",
        "context_id": "ctx-3",
        "context_category": "History",
        "sub_category": "Cities",
    },
}


# Record 4: a 2-message record whose final (only) user turn is long
# (>= 2,000 characters) with no blank-line break anywhere -- the genuine
# failure mode measured at 2.9% (55/1,899) of the real corpus: no cheap,
# safe signal to slice on, so the whole turn falls back to being the
# question and gets flagged via `question_unsliced`.
_RECORD_4_LONG_TURN = "The unbroken reference passage. " * 70 + "What year is described above?"
_RECORD_4 = {
    "messages": [
        {"role": "system", "content": "You are DocQA."},
        {"role": "user", "content": _RECORD_4_LONG_TURN},
    ],
    "rubrics": ["States the year"],
    "metadata": {
        "task_id": "task-4",
        "context_id": "ctx-4",
        "context_category": "History",
        "sub_category": "Unstructured",
    },
}


def _write_fixture(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "CL-bench.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def test_load_produces_one_case_per_line_with_one_question_each(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path, _RECORD_1, _RECORD_2, _RECORD_3))

    assert len(cases) == 3
    assert [case.user_id for case in cases] == [
        "clbench:task-1",
        "clbench:task-2",
        "clbench:task-3",
    ]
    assert all(len(case.questions) == 1 for case in cases)


def test_load_splits_the_last_user_turn_at_its_final_blank_line_paragraph(
    tmp_path: Path,
) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_1))

    assert [m["content"] for m in case.messages] == [
        "Freedonia is a fictional country used in economics textbooks. "
        "It has a long history of trade disputes.",
    ]
    assert [m["role"] for m in case.messages] == ["user"]
    [question] = case.questions
    assert question.question == (
        "What is Freedonia primarily used to illustrate in economics textbooks?"
    )


def test_load_preserves_prior_turns_and_falls_back_to_the_whole_turn_when_no_break(
    tmp_path: Path,
) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_2))

    # No blank-line break in the final turn -> nothing more to fold in as
    # history beyond the real prior turns.
    assert [m["content"] for m in case.messages] == [
        "Zorblatt Inc. was founded in 2001 by Jane Doe in Springfield.",
        "Understood, I'll remember that about Zorblatt Inc.",
    ]
    assert [m["role"] for m in case.messages] == ["user", "assistant"]
    [question] = case.questions
    assert question.question == "Who founded Zorblatt Inc.?"


def test_load_folds_a_long_multi_paragraph_history_into_one_message(tmp_path: Path) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_3))

    [message] = case.messages
    assert message["role"] == "user"
    assert message["content"] == (
        "Section 1: The city of Lumenport was founded in 1750 by merchant sailors.\n\n"
        "Section 2: Its main export was spiced glassware, prized across three "
        "continents.\n\n"
        "Section 3: The Great Fire of 1812 destroyed much of the old harbor district."
    )
    [question] = case.questions
    assert question.question == "When was Lumenport founded?"


def test_load_extracts_system_prompt_from_the_system_role_message(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path, _RECORD_1, _RECORD_2))

    [q1] = cases[0].questions
    [q2] = cases[1].questions
    assert q1.payload["system_prompt"] == "You are FactBot."
    assert q2.payload["system_prompt"] == "Persona: helpful research assistant."


def test_load_sets_id_from_metadata_task_id_and_preserves_metadata_and_rubrics(
    tmp_path: Path,
) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_1))
    [question] = case.questions

    assert question.question_id == "task-1"
    assert question.payload["id"] == "task-1"
    assert question.payload["rubrics"] == ["Mentions trade disputes or an economics illustration"]
    assert question.payload["metadata"] == {
        "task_id": "task-1",
        "context_id": "ctx-1",
        "context_category": "Trivia",
        "sub_category": "Facts",
    }


def test_load_omits_timestamp_field_entirely(tmp_path: Path) -> None:
    cases = load(_write_fixture(tmp_path, _RECORD_1, _RECORD_2, _RECORD_3))

    for case in cases:
        for message in case.messages:
            assert "timestamp" not in message


def test_load_flags_question_unsliced_when_a_long_final_turn_has_no_blank_line(
    tmp_path: Path,
) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_4))

    assert len(_RECORD_4_LONG_TURN) >= 2000
    # No blank-line break anywhere -> nothing folds into history, and the
    # whole turn becomes the question.
    assert case.messages == ()
    [question] = case.questions
    assert question.question == _RECORD_4_LONG_TURN
    assert question.payload["question_unsliced"] is True


def test_load_sets_question_unsliced_false_for_a_cleanly_split_record(tmp_path: Path) -> None:
    [case] = load(_write_fixture(tmp_path, _RECORD_1))

    [question] = case.questions
    assert question.payload["question_unsliced"] is False


def test_load_raises_on_a_record_with_an_empty_rubrics_list(tmp_path: Path) -> None:
    bad_record = {**_RECORD_1, "rubrics": []}
    path = _write_fixture(tmp_path, bad_record)

    with pytest.raises(ValueError, match="rubrics"):
        load(path)
