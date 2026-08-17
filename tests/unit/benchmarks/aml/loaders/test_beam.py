"""Tests for the BEAM -> AML case loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.aml.loaders.beam import load

# Batch 1 has a two-turn pair: a "user" turn with a time_anchor, and an
# "assistant" reply with none (matches the real corpus: time_anchor typically
# only appears on the user turn that opens a batch). Batch 2 has a four-turn
# pair (real turn-pairs are not always exactly 2 long) with no timestamps at
# all, to exercise the "omit the key" path on every turn in the group.
_CHAT = [
    {
        "batch_number": 1,
        "turns": [
            [
                {
                    "role": "user",
                    "id": 0,
                    "time_anchor": "March-15-2024",
                    "content": "I'm planning a Flask app.",
                },
                {"role": "assistant", "id": 1, "content": "Great, let's start."},
            ]
        ],
    },
    {
        "batch_number": 2,
        "turns": [
            [
                {"role": "user", "id": 2, "content": "Add a login page."},
                {"role": "assistant", "id": 3, "content": "Sure, here's a plan."},
                {"role": "user", "id": 4, "content": "Also add logout."},
                {"role": "assistant", "id": 5, "content": "Done."},
            ]
        ],
    },
]

# "information_extraction" carries an inner "question_type" field in the real
# corpus (e.g. "context_date/time") that is a sub-classification, NOT the
# category -- the payload's question_type must come from the outer dict key
# ("information_extraction"), not this inner field, or event_ordering-style
# category-based scoring would silently read the wrong thing.
_QUESTIONS = {
    "event_ordering": [
        {
            "question": "In what order did I mention login, then logout?",
            "answer": "1) login 2) logout",
            "rubric": ["login", "logout"],
        }
    ],
    "information_extraction": [
        {
            "question": "What framework is the app built with?",
            "question_type": "context_date/time",
            "answer": "Flask",
            "rubric": ["Flask"],
        }
    ],
    "abstention": [
        {
            "question": "What database migrations have I run?",
            "ideal_response": "There is no information about database migrations.",
            "rubric": ["no information about database migrations"],
        }
    ],
}


def _write_fixture(tmp_path: Path, size: str = "100K", conv_name: str = "1") -> tuple[Path, Path]:
    # Real BEAM layout nests conversations two levels deep:
    # `.benchmarks/beam/chats/{size}/{conv_id}/...` -- conv_id restarts from
    # "1" in every size bucket, so fixtures must reproduce both levels or
    # they can't catch a loader that only reads one of them.
    conv_dir = tmp_path / size / conv_name
    conv_dir.mkdir(parents=True)
    chat_path = conv_dir / "chat.json"
    chat_path.write_text(json.dumps(_CHAT))

    pq_dir = conv_dir / "probing_questions"
    pq_dir.mkdir()
    questions_path = pq_dir / "probing_questions.json"
    questions_path.write_text(json.dumps(_QUESTIONS))

    return chat_path, questions_path


def test_load_returns_one_case_named_after_the_conversation_directory(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path, size="100K", conv_name="1")
    cases = load(chat_path, questions_path)

    assert len(cases) == 1
    [case] = cases
    assert case.user_id == "beam:100K:1"


def test_load_flattens_batches_and_turn_pairs_of_any_length(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path)
    [case] = load(chat_path, questions_path)

    assert [m["content"] for m in case.messages] == [
        "I'm planning a Flask app.",
        "Great, let's start.",
        "Add a login page.",
        "Sure, here's a plan.",
        "Also add logout.",
        "Done.",
    ]
    assert [m["role"] for m in case.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_load_omits_timestamp_key_rather_than_inventing_one(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path)
    [case] = load(chat_path, questions_path)

    # First turn has a real time_anchor, parsed to epoch milliseconds (UTC) --
    # AML's wire contract requires `int`, not free text.
    assert case.messages[0]["timestamp"] == 1710460800000
    # Every other turn in this fixture has no time_anchor at all -- the key
    # must be absent, not present-with-None (that would look like a real,
    # deliberately-null timestamp instead of "we never had one").
    for message in case.messages[1:]:
        assert "timestamp" not in message


def test_load_builds_the_official_payload_and_keeps_rubric_verbatim(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path)
    [case] = load(chat_path, questions_path)

    by_question = {q.question: q for q in case.questions}
    extraction = by_question["What framework is the app built with?"]
    assert extraction.payload == {
        "id": extraction.payload["id"],
        "rubric": ["Flask"],
        "question_type": "information_extraction",
    }
    # The inner "question_type": "context_date/time" sub-classification must
    # not leak through as the payload's question_type.
    assert extraction.payload["question_type"] != "context_date/time"


def test_load_propagates_event_ordering_category_verbatim(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path)
    [case] = load(chat_path, questions_path)

    [ordering] = [q for q in case.questions if q.payload["question_type"] == "event_ordering"]
    assert ordering.question == "In what order did I mention login, then logout?"
    assert ordering.payload["rubric"] == ["login", "logout"]


def test_load_synthesizes_unique_nonempty_ids_across_categories(tmp_path: Path) -> None:
    chat_path, questions_path = _write_fixture(tmp_path)
    [case] = load(chat_path, questions_path)

    ids = [q.payload["id"] for q in case.questions]
    assert len(case.questions) == 3
    assert all(isinstance(question_id, str) and question_id for question_id in ids)
    assert len(set(ids)) == len(ids)
    question_ids = [q.question_id for q in case.questions]
    assert len(set(question_ids)) == len(question_ids)


def test_load_scopes_a_second_conversation_independently(tmp_path: Path) -> None:
    chat_path_1, questions_path_1 = _write_fixture(tmp_path, size="100K", conv_name="1")
    chat_path_2, questions_path_2 = _write_fixture(tmp_path, size="100K", conv_name="2")

    [case_1] = load(chat_path_1, questions_path_1)
    [case_2] = load(chat_path_2, questions_path_2)

    assert case_1.user_id == "beam:100K:1"
    assert case_2.user_id == "beam:100K:2"
    # Same fixture content in both dirs -- ids must still not collide once a
    # driver mixes cases from multiple conversations into one answers file.
    ids_1 = {q.payload["id"] for q in case_1.questions}
    ids_2 = {q.payload["id"] for q in case_2.questions}
    assert ids_1.isdisjoint(ids_2)


def test_load_scopes_conversations_with_the_same_name_across_size_buckets(
    tmp_path: Path,
) -> None:
    # BEAM's real corpus restarts conversation directory names from "1" in
    # every size bucket: `.benchmarks/beam/chats/{100K,500K,1M,10M}/1/` all
    # exist simultaneously. A loader that only reads the conversation
    # directory name (dropping the size bucket) would collide these into one
    # `user_id` -- merging two unrelated conversations into the same
    # MindBridge retrieval scope and contaminating recall for both.
    chat_path_100k, questions_path_100k = _write_fixture(tmp_path, size="100K", conv_name="1")
    chat_path_500k, questions_path_500k = _write_fixture(tmp_path, size="500K", conv_name="1")

    [case_100k] = load(chat_path_100k, questions_path_100k)
    [case_500k] = load(chat_path_500k, questions_path_500k)

    assert case_100k.user_id != case_500k.user_id

    ids_100k = {q.payload["id"] for q in case_100k.questions}
    ids_500k = {q.payload["id"] for q in case_500k.questions}
    assert ids_100k.isdisjoint(ids_500k)

    question_ids_100k = {q.question_id for q in case_100k.questions}
    question_ids_500k = {q.question_id for q in case_500k.questions}
    assert question_ids_100k.isdisjoint(question_ids_500k)


def test_load_raises_on_a_question_with_an_empty_rubric(tmp_path: Path) -> None:
    conv_dir = tmp_path / "1"
    conv_dir.mkdir()
    chat_path = conv_dir / "chat.json"
    chat_path.write_text(json.dumps(_CHAT))

    pq_dir = conv_dir / "probing_questions"
    pq_dir.mkdir()
    questions_path = pq_dir / "probing_questions.json"
    bad_questions = {"abstention": [{"question": "No rubric here?", "rubric": []}]}
    questions_path.write_text(json.dumps(bad_questions))

    with pytest.raises(ValueError, match="rubric"):
        load(chat_path, questions_path)
