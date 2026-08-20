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

# The flat message stream every tier must land on, whatever its `chat.json`
# nesting looks like: batches concatenated, turn-pairs flattened, in document
# order.
_EXPECTED_CONTENTS = [
    "I'm planning a Flask app.",
    "Great, let's start.",
    "Add a login page.",
    "Sure, here's a plan.",
    "Also add logout.",
    "Done.",
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


def _chat_document(size: str) -> list[object]:
    """One tier's real top-level `chat.json` shape: 10M wraps batches in plans.

    Emitting the flat shape under a `10M/` path is exactly what let the
    plan-wrapped schema go unnoticed -- the derived `user_id` reads "10M" while
    the bytes are a 100K document. Each batch goes into its own single-key
    element, matching the real corpus (measured: every 10M element carries
    exactly one plan key), so element order alone fixes message order.
    """
    if size != "10M":
        return [*_CHAT]
    return [{"plan-1": [_CHAT[0]]}, {"plan-2": [_CHAT[1]]}]


def _write_fixture(tmp_path: Path, size: str = "100K", conv_name: str = "1") -> tuple[Path, Path]:
    # Real BEAM layout nests conversations two levels deep:
    # `.benchmarks/beam/chats/{size}/{conv_id}/...` -- conv_id restarts from
    # "1" in every size bucket, so fixtures must reproduce both levels or
    # they can't catch a loader that only reads one of them.
    conv_dir = tmp_path / size / conv_name
    conv_dir.mkdir(parents=True)
    chat_path = conv_dir / "chat.json"
    chat_path.write_text(json.dumps(_chat_document(size)))

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

    assert [m["content"] for m in case.messages] == _EXPECTED_CONTENTS
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


@pytest.mark.parametrize("size", ["100K", "500K", "1M", "10M"])
def test_load_parses_every_size_bucket_to_the_same_message_stream(
    tmp_path: Path, size: str
) -> None:
    # Asserting only on `user_id` here proves nothing about a tier's schema:
    # the id is derived from the path, so it comes out "beam:10M:1" even when
    # the bytes under that path are another tier's shape entirely. The content
    # assertions are what make this test tier-specific.
    chat_path, questions_path = _write_fixture(tmp_path, size=size)
    [case] = load(chat_path, questions_path)

    assert case.user_id == f"beam:{size}:1"
    assert [m["content"] for m in case.messages] == _EXPECTED_CONTENTS
    assert case.messages[0]["timestamp"] == 1710460800000
    assert len(case.questions) == 3


def test_load_concatenates_10m_plan_groups_in_document_order(tmp_path: Path) -> None:
    # The 10M tier's `chat.json` is a list of single-key `{"plan-N": [...batches
    # ...]}` dicts; the other three tiers put bare batch objects at that level.
    # Plans are consecutive stretches of one conversation (turn ids ascend
    # straight across the boundaries in the real corpus), so the batches under
    # them concatenate into one message stream in document order.
    chat_path, questions_path = _write_fixture(tmp_path, size="10M")

    # Guard the fixture itself. A 10M fixture that silently emitted the flat
    # shape would leave the plan branch running zero times while every
    # assertion below still passed.
    document = json.loads(chat_path.read_text())
    assert [sorted(element) for element in document] == [["plan-1"], ["plan-2"]]

    [case] = load(chat_path, questions_path)

    # Both plans' batches survive, in the document order of the plan elements --
    # the second plan's turns must not lead, and neither plan may be dropped.
    assert [m["content"] for m in case.messages] == _EXPECTED_CONTENTS
    assert len(case.messages) == 6


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
