"""Regression checks for the LongMemEval, CL-Bench, BEAM and PersonaMem-v3 adapters.

Each case pins a shape the pinned releases actually contain and an earlier
version of these loaders rejected or mishandled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.beam import BEAM_CATEGORIES, load_beam
from mindbridge.benchmarks.clbench import (
    OVERSIZED_QUESTION_CHARACTERS,
    load_clbench,
    split_question,
)
from mindbridge.benchmarks.download import _patterns
from mindbridge.benchmarks.longmemeval import load_longmemeval
from mindbridge.benchmarks.personamem_v3 import (
    load_personamem_v3,
    render_candidates,
    render_event,
)
from mindbridge.benchmarks.task_catalog import TASKS


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _longmemeval_question(**overrides: object) -> dict[str, object]:
    question: dict[str, object] = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I studied business."}]],
        "answer_session_ids": ["s1"],
    }
    question.update(overrides)
    return question


def test_longmemeval_keeps_numeric_answers_and_repeated_sessions(tmp_path: Path) -> None:
    dataset = _write(
        tmp_path / "longmemeval_s",
        [
            _longmemeval_question(question_id="numeric", answer=1300),
            _longmemeval_question(
                question_id="repeat",
                haystack_dates=["2023/05/20 (Sat) 02:21", "2023/05/24 (Wed) 06:24"],
                haystack_session_ids=["shared", "shared"],
                haystack_sessions=[
                    [{"role": "user", "content": "same turns"}],
                    [{"role": "user", "content": "same turns"}],
                ],
                answer_session_ids=["shared"],
            ),
            _longmemeval_question(
                question_id="empty",
                haystack_sessions=[[]],
                answer_session_ids=[],
            ),
        ],
    )

    questions = {question.question_id: question for question in load_longmemeval(dataset)}

    # 32 of the 500 released answers are JSON numbers rather than strings.
    assert questions["numeric"].reference_answer == "1300"
    # 15 questions plant one session ID twice at different dates; both copies
    # are kept and their turns stay individually addressable.
    repeated = questions["repeat"]
    assert [session.session_id for session in repeated.sessions] == ["shared", "shared"]
    assert [session.position for session in repeated.sessions] == [0, 1]
    turn_ids = [turn.turn_id for session in repeated.sessions for turn in session.turns]
    assert len(set(turn_ids)) == len(turn_ids)
    # 1,230 released sessions carry no turns at all.
    assert questions["empty"].sessions[0].turns == ()


def test_longmemeval_reads_dates_without_a_locale_and_flags_abstention(tmp_path: Path) -> None:
    dataset = _write(
        tmp_path / "longmemeval_s",
        [
            _longmemeval_question(question_id="plain"),
            _longmemeval_question(question_id="q2_abs"),
        ],
    )

    questions = {question.question_id: question for question in load_longmemeval(dataset)}

    # The weekday is matched by pattern; `strptime("%a")` would read the
    # process locale and reject every date under a non-English one.
    assert questions["plain"].question_date.isoformat() == "2023-05-30T23:40:00+00:00"
    assert questions["plain"].sessions[0].occurred_at.isoformat() == "2023-05-20T02:21:00+00:00"
    # Upstream derives the abstention judge from the ID, not from a field.
    assert questions["plain"].abstention is False
    assert questions["q2_abs"].abstention is True

    bad = _write(
        tmp_path / "bad",
        [_longmemeval_question(question_date="2023-05-30 23:40")],
    )
    with pytest.raises(ValueError, match="invalid LongMemEval date"):
        load_longmemeval(bad)


def test_clbench_splits_the_question_off_its_reference_document(tmp_path: Path) -> None:
    document = "PARA ONE\n\nPARA TWO"
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a rules referee."},
                {"role": "user", "content": f"{document}\n\nWhat do Sighting cards do?"},
            ],
            "rubrics": ["The response should define a Sighting card."],
            "metadata": {
                "task_id": "t1",
                "context_id": "c1",
                "context_category": "Rule System Application",
                "sub_category": "Game Mechanics",
            },
        },
        {
            # A record whose final turn has a blank-line break but whose
            # trailing paragraph is still oversized: 75 released records look
            # like this, and they carry the same flag as an unsplit one.
            "messages": [
                {"role": "user", "content": "lead\n\n" + "x" * OVERSIZED_QUESTION_CHARACTERS}
            ],
            "rubrics": ["ok"],
            "metadata": {
                "task_id": "t2",
                "context_id": "c1",
                "context_category": "Rule System Application",
                "sub_category": "Game Mechanics",
            },
        },
    ]
    # A bare U+2028 inside a JSON string is legal JSON, but `str.splitlines()`
    # treats it as a line break and cuts the record in half so it no longer
    # parses. The pinned release carries 343 of them.
    dataset = tmp_path / "CL-bench.jsonl"
    encoded = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    encoded = encoded.replace("PARA ONE", "PARA\u2028ONE")
    assert len(encoded.splitlines()) > len(encoded.split("\n"))
    dataset.write_text(encoded + "\n", encoding="utf-8")

    tasks = {task.task_id: task for task in load_clbench(dataset)}

    first = tasks["t1"]
    assert first.question == "What do Sighting cards do?"
    assert first.question_unsliced is False
    assert first.system_prompt == "You are a rules referee."
    assert [turn.content for turn in first.turns] == ["PARA\u2028ONE\n\nPARA TWO"]
    assert tasks["t2"].question_unsliced is True
    assert split_question("only one paragraph") == ("", "only one paragraph", False)


def test_beam_flattens_the_ten_million_tier_and_resolves_each_reference_key(
    tmp_path: Path,
) -> None:
    def turn(index: int, anchor: str | None = None) -> dict[str, object]:
        item: dict[str, object] = {
            "role": "user" if index % 2 == 0 else "assistant",
            "id": index,
            "content": f"turn {index}",
        }
        if anchor is not None:
            item["time_anchor"] = anchor
        return item

    directory = tmp_path / "10M" / "3"
    _write(
        directory / "chat.json",
        [
            {"plan-1": [{"batch_number": 1, "turns": [[turn(0, "March-15-2024"), turn(1)]]}]},
            {"plan-2": [{"batch_number": 1, "turns": [[turn(2), turn(3)]]}]},
        ],
    )
    _write(
        directory / "probing_questions" / "probing_questions.json",
        {
            "abstention": [{"question": "q", "rubric": ["r"], "ideal_response": "no information"}],
            "contradiction_resolution": [
                {"question": "q", "rubric": ["r"], "ideal_answer": "resolved"}
            ],
            "summarization": [{"question": "q", "rubric": ["r"], "ideal_summary": "summary"}],
            "temporal_reasoning": [{"question": "q", "rubric": ["r"], "answer": "three weeks"}],
            # Two categories publish no reference answer at all.
            "instruction_following": [{"question": "q", "rubric": ["r"]}],
        },
    )

    conversations = load_beam(tmp_path / "10M", "10M")

    assert len(conversations) == 1
    conversation = conversations[0]
    # Plans are consecutive stretches of one conversation, so the directory is
    # still one retrieval scope.
    assert [turn_.turn_id for turn_ in conversation.turns] == [
        "3_T000000",
        "3_T000001",
        "3_T000002",
        "3_T000003",
    ]
    # `%B` would read the process locale; the month table does not.
    assert conversation.turns[0].occurred_at is not None
    assert conversation.turns[0].occurred_at.isoformat() == "2024-03-15T00:00:00+00:00"
    assert conversation.turns[1].occurred_at is None

    references = {
        question.category: question.reference_answer for question in conversation.questions
    }
    assert references == {
        "abstention": "no information",
        "contradiction_resolution": "resolved",
        "summarization": "summary",
        "temporal_reasoning": "three weeks",
        "instruction_following": "",
    }
    # Questions carry no ID upstream; it comes from the tier, conversation,
    # category key and list position, in catalog category order.
    assert conversation.questions[0].question_id == "10M-3-abstention-0000"
    assert [question.category for question in conversation.questions] == [
        category for category in BEAM_CATEGORIES if category in references
    ]

    _write(directory / "probing_questions" / "probing_questions.json", {"mystery": []})
    with pytest.raises(ValueError, match="unknown categories: mystery"):
        load_beam(tmp_path / "10M", "10M")


def _personamem_event(object_id: str, timestamp: int, **extra: object) -> dict[str, object]:
    event: dict[str, object] = {
        "source_object_id": object_id,
        "source_timestamp": timestamp,
        "formatted_timestamp": "03:41, 04/03/2026",
        "source_hashtags": ["#ankara"],
        "source_interaction_type": "explicit_positive",
        "interaction_format": {"app": "Instagram", "action_label": "Posted image"},
        "content": {"caption": "clean lines"},
    }
    event.update(extra)
    return event


def _personamem_query(**overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "query_id": "8:0001:aaa",
        "task_family": "chatbot_response",
        "task_type": "chatbot_personalized_response",
        "query_kind": "user_query",
        "expected_behavior": "personalize",
        "ts": 2_000,
        "user_query": "what should i cook",
        "example_response": "pepper stew",
        "groundtruth_preference": "likes stew",
        "rubric_tags": ["(+) reflect preferences"],
    }
    query.update(overrides)
    return query


def test_personamem_v3_reads_every_released_row_shape(tmp_path: Path) -> None:
    persona = tmp_path / "backend" / "8"
    _write(
        persona / "instagram.json",
        [
            _personamem_event("a", 1_000),
            # A direct-message thread names its parts `sender`/`text`.
            _personamem_event(
                "dm",
                1_500,
                is_dm=True,
                messages=[{"msg_id": "m0", "sender": "friend_5", "text": "your kind of vibe"}],
            ),
        ],
    )
    _write(
        persona / "calendar.json",
        {
            "modifications": [
                {
                    "mod_id": "mod_001",
                    "ts": 1_200,
                    "action": "added",
                    "entry": {
                        "title": "Grant review",
                        "type": "work",
                        # Booked before it starts, so the scheduled window is
                        # not derivable from the modification's own timestamp.
                        "start_ts": 9_000,
                        "end_ts": 16_200,
                        "location": {"city": "Osaka", "country": "Japan"},
                    },
                },
                # An `updated` modification carries no `entry` at all, only the
                # entry ID and a per-field diff.
                {
                    "mod_id": "mod_002",
                    "ts": 1_300,
                    "action": "updated",
                    "entry_id": "cal_006",
                    "diff": {
                        "end_ts": {"from": 16_200, "to": 18_000},
                        "notes": {"from": "", "to": "Ran long."},
                    },
                },
                {
                    "mod_id": "mod_003",
                    "ts": 1_400,
                    "action": "removed",
                    "entry_id": "cal_017",
                    "removal_reason": "canceled: organizer moved it",
                },
            ]
        },
    )
    _write(
        persona / "test.json",
        [
            _personamem_query(),
            # The sycophancy rows publish a bare string where every other
            # family publishes a list.
            _personamem_query(
                query_id="8:0002:syco",
                task_type="over_personalization_sycophancy",
                rubric_tags="sycophancy_resistance",
            ),
            # Cluster-scored rows are dropped: their runner threads each
            # response into the next prompt.
            _personamem_query(
                query_id="8:0003:rep",
                task_type="over_personalization_repetition_chatbot",
            ),
            _personamem_query(
                query_id="8:0004:rank",
                task_type="personalized_recommendation",
                user_query="[system prompt] Recommend feed items.",
                instance_full={
                    "candidates": [
                        {"title": "A", "source_app": "facebook", "_held_out_persona_item": "LEAK"},
                        {"title": "B", "source_app": "threads"},
                    ],
                    "held_out_idx": 0,
                    "hard_negative_idxs": [1],
                },
            ),
        ],
    )

    personas = load_personamem_v3(tmp_path / "backend")

    assert len(personas) == 1
    events = personas[0].events
    assert [event.event_id for event in events] == [
        "instagram:a",
        "calendar:mod_001",
        "calendar:mod_002",
        "calendar:mod_003",
        "instagram:dm",
    ]
    assert events[4].conversation[0].role == "friend_5"
    assert "friend_5: your kind of vibe" in render_event(events[4])

    queries = {query.query_id: query for query in personas[0].queries}
    assert "8:0003:rep" not in queries
    assert queries["8:0002:syco"].rubric_tags == ("sycophancy_resistance",)
    ranking = queries["8:0004:rank"]
    assert ranking.positive_indexes == (0,)
    assert ranking.negative_indexes == (1,)
    # The slate must never show the scorer-side identity of the held-out item.
    slate = render_candidates(ranking.candidates)
    assert "LEAK" not in slate
    assert "_held_out_persona_item" not in slate
    assert "title='A'" in slate
    # `profile.json` is scorer-side and is never read as memory.
    assert not any(event.app == "Profile" for event in events)


def test_personamem_v3_keeps_what_each_calendar_modification_says(tmp_path: Path) -> None:
    """A calendar memory has to carry the appointment, not just the edit."""
    persona = tmp_path / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "calendar.json",
        {
            "modifications": [
                {
                    "mod_id": "mod_001",
                    "ts": 1_200,
                    "action": "added",
                    "entry": {
                        "title": "Grant review",
                        "type": "work",
                        "start_ts": 9_000,
                        "end_ts": 16_200,
                        "location": {"city": "Osaka", "country": "Japan"},
                    },
                },
                {
                    "mod_id": "mod_002",
                    "ts": 1_300,
                    "action": "updated",
                    "entry_id": "cal_006",
                    "diff": {
                        "end_ts": {"from": 16_200, "to": 18_000},
                        "notes": {"from": "", "to": "Ran long."},
                    },
                },
                {
                    "mod_id": "mod_003",
                    "ts": 1_400,
                    "action": "removed",
                    "entry_id": "cal_017",
                    "removal_reason": "canceled: organizer moved it",
                },
            ]
        },
    )
    _write(persona / "test.json", [_personamem_query()])

    events = {
        event.event_id: event
        for event in load_personamem_v3(tmp_path / "backend")[0].events
        if event.app == "Calendar"
    }

    added = events["calendar:mod_001"]
    # The appointment's window is a different moment from the edit's, and the
    # duration appears nowhere else, so neither is recoverable from `ts`.
    assert added.occurred_at.isoformat() == "1970-01-01T00:20:00+00:00"
    assert added.scheduled_start is not None
    assert added.scheduled_start.isoformat() == "1970-01-01T02:30:00+00:00"
    assert added.scheduled_end is not None
    assert added.scheduled_end.isoformat() == "1970-01-01T04:30:00+00:00"
    rendered = render_event(added)
    assert "Scheduled: 1970-01-01T02:30:00+00:00 to 1970-01-01T04:30:00+00:00" in rendered
    assert "Location: Osaka, Japan" in rendered
    assert "Type: work" in rendered

    # An `updated` modification has no `entry`; without its diff the memory
    # would be a bare header line with nothing retrievable in it.
    updated = render_event(events["calendar:mod_002"])
    assert "end_ts 1970-01-01T04:30:00+00:00 -> 1970-01-01T05:00:00+00:00" in updated
    assert "notes (none) -> Ran long." in updated
    assert "cal_006" in updated

    assert "canceled: organizer moved it" in render_event(events["calendar:mod_003"])
    # No calendar memory may be a header line on its own.
    assert all(len(render_event(event).splitlines()) > 1 for event in events.values())


def test_download_patterns_cover_extensionless_files_and_pinned_directories() -> None:
    root = Path(".benchmarks")

    def patterns_for(name: str) -> tuple[str, ...]:
        spec = TASKS[name]
        dataset = spec.dataset_path(root)
        release = root / Path(spec.dataset).parts[0]
        return _patterns(spec, dataset, dataset, dataset.relative_to(release).as_posix())

    # `longmemeval_s` is a 278 MB JSON file with no extension. Asking only for
    # `longmemeval_s/*` matched nothing while the download reported success.
    assert patterns_for("longmemeval-s") == ("longmemeval_s", "longmemeval_s/*")
    assert patterns_for("clbench") == ("CL-bench.jsonl",)
    # A directory input may pin exactly the files its loader opens.
    assert patterns_for("beam-100k") == (
        "chats/100K/*/chat.json",
        "chats/100K/*/probing_questions/probing_questions.json",
    )
    personamem = patterns_for("personamem-v3")
    assert "backend/*/test.json" in personamem
    assert not any("persona.html" in pattern for pattern in personamem)
    assert not any("profile.json" in pattern for pattern in personamem)
    # A directory with no pinned patterns keeps the old both-spellings default.
    assert patterns_for("mem-gallery") == ("data/dialog", "data/dialog/*")


def test_beam_keys_turns_by_position_when_the_published_id_restarts(tmp_path: Path) -> None:
    """4 of the 35 conversations in the 1M tier restart their turn counter."""

    def turn(identifier: int, text: str) -> dict[str, object]:
        return {"role": "user", "id": identifier, "content": text}

    directory = tmp_path / "1M" / "5"
    _write(
        directory / "chat.json",
        [
            {"batch_number": 1, "turns": [[turn(0, "first thing"), turn(1, "reply")]]},
            # The counter restarts here with different content under the same
            # IDs, so `id` does not identify a turn within the conversation.
            {"batch_number": 10, "turns": [[turn(0, "much later"), turn(1, "later reply")]]},
        ],
    )
    _write(
        directory / "probing_questions" / "probing_questions.json",
        {"abstention": [{"question": "q", "rubric": ["r"], "ideal_response": "none"}]},
    )

    turns = load_beam(tmp_path / "1M", "1M")[0].turns

    assert [item.turn_id for item in turns] == [
        "5_T000000",
        "5_T000001",
        "5_T000002",
        "5_T000003",
    ]
    # Document order is preserved and no turn is lost to an ID collision.
    assert [item.content for item in turns] == [
        "first thing",
        "reply",
        "much later",
        "later reply",
    ]
    # The published ID is still carried, because probing questions cite it.
    assert [item.source_id for item in turns] == [0, 1, 0, 1]


def test_personamem_v3_reads_the_flattened_row_spelling(tmp_path: Path) -> None:
    """Six released rows publish a different spelling of the same record."""
    persona = tmp_path / "backend" / "26"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            {
                "query_id": "26:0043:sensitive_row00_q0",
                "task_family": "over_personalization",
                "task_type": "over_personalization_sensitive_event",
                "ts": 2_000,
                # `query_text`, not `user_query`; no `query_kind` at all;
                # `expected_response_kind` in place of `expected_behavior`;
                # semicolon-joined rubric tags; and the instance as a JSON
                # string rather than an object.
                "query_text": "is that just burnout or something else?",
                "expected_response_kind": "text",
                "rubric_tags": "avoid_overpersonalization;telegraph_avoidance",
                "instance_json": '{"test_id": "row00", "arm": "sensitive_event"}',
            },
            # `user_query` present but null, with the question in the instance.
            {
                **_personamem_query(query_id="26:0044:nulled"),
                "user_query": None,
                "instance_full": {"user_query": "what should i cook"},
            },
            # No question text under any spelling: dropped rather than stored
            # as a memory-less question.
            {
                **_personamem_query(query_id="26:0045:empty"),
                "user_query": None,
                "instance_full": {},
            },
        ],
    )

    queries = {
        query.query_id: query for query in load_personamem_v3(tmp_path / "backend")[0].queries
    }

    assert set(queries) == {"26:0043:sensitive_row00_q0", "26:0044:nulled"}
    flattened = queries["26:0043:sensitive_row00_q0"]
    assert flattened.user_query == "is that just burnout or something else?"
    assert flattened.expected_behavior == "text"
    assert flattened.query_kind == ""
    assert flattened.rubric_tags == ("avoid_overpersonalization", "telegraph_avoidance")
    assert flattened.judge_evidence["arm"] == "sensitive_event"
    assert queries["26:0044:nulled"].user_query == "what should i cook"


def test_personamem_v3_never_records_the_question_as_its_own_reference(tmp_path: Path) -> None:
    """857 released queries publish no `example_response`."""
    from mindbridge.benchmarks.eval_adapters import load_task
    from mindbridge.benchmarks.task_catalog import TASKS

    persona = tmp_path / "personamem-v3" / "backend" / "8"
    _write(persona / "instagram.json", [_personamem_event("a", 1_000)])
    _write(
        persona / "test.json",
        [
            _personamem_query(query_id="8:0001:gold"),
            {
                **_personamem_query(query_id="8:0002:no-gold"),
                "example_response": "",
                "groundtruth_preference": "Enjoys couple-life content.",
            },
            {
                **_personamem_query(query_id="8:0003:nothing"),
                "example_response": "",
                "groundtruth_preference": "",
            },
        ],
    )

    unit = load_task(TASKS["personamem-v3"], root=tmp_path, verify_digest=False).units[0]
    references = {q.question_id: q.references[0] for q in unit.questions}

    assert references["8:0001:gold"] == "pepper stew"
    # Falls back to label material, never to the prompt itself.
    assert references["8:0002:no-gold"] == "Enjoys couple-life content."
    assert "what should i cook" not in references["8:0003:nothing"]
    assert references["8:0003:nothing"] == "(no reference answer published for this row)"


def test_text_memories_split_passages_the_product_would_reject() -> None:
    """An oversized part is not just dropped -- it voids its whole unit.

    `memory.add` rejects a part over `_MAX_TEXT_CHARACTERS`, the runner counts
    an ingest failure, and `_apply_judges` then skips every question in that
    unit. BEAM turns reach 348,864 characters and LongMemEval has one of
    76,594, so unsplit storage silently left 70% of BEAM's 10M tier unjudged.
    """
    from mindbridge.benchmarks.eval_adapters import _text_memories
    from mindbridge.memory import _MAX_TEXT_CHARACTERS

    short = _text_memories("turn", "one short line")
    assert len(short) == 1
    assert short[0].source_id == "turn"
    assert short[0].content == ("one short line",)

    long_turn = "word " * 120_000
    assert len(long_turn) > _MAX_TEXT_CHARACTERS
    parts = _text_memories("turn", long_turn, end_seconds=1_775_000_000.0)
    assert len(parts) > 1
    assert all(
        len(text) <= _MAX_TEXT_CHARACTERS
        for item in parts
        for text in item.content
        if isinstance(text, str)
    )
    # Split parts stay individually addressable and keep the causal cutoff.
    assert [item.source_id for item in parts[:2]] == ["turn_B0000", "turn_B0001"]
    assert len({item.source_id for item in parts}) == len(parts)
    assert all(item.end_seconds == 1_775_000_000.0 for item in parts)
    # Nothing is dropped.
    rebuilt = "".join(text for item in parts for text in item.content if isinstance(text, str))
    assert rebuilt.replace("\n\n", " ").split() == long_turn.split()
