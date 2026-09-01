"""Thin adapter for the official LongMemEval haystack releases."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

LONGMEMEVAL_ADAPTER_VERSION = "longmemeval_official_v1"

# The six official question types. `evaluate_qa.py` selects a different judge
# template per type and raises `NotImplementedError` on anything else, so an
# unrecognised value is a corpus problem rather than something to pass through.
LongMemEvalQuestionType = Literal[
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

# Session dates are published as "2023/05/20 (Sat) 02:21". The weekday is
# parsed by pattern rather than by `%a` on purpose: `strptime` resolves `%a`
# through the process locale, so a non-English locale rejects every date in
# the release. MEMLENS publishes the same format and guards it the same way.
_DATE_PATTERN = re.compile(r"^(\d{4}/\d{2}/\d{2}) \([A-Za-z]{3}\) (\d{2}:\d{2})$")

# `evaluate_qa.py` selects the abstention judge template with
# `abstention='_abs' in entry['question_id']`, so the marker lives in the ID
# itself rather than in a field of its own.
_ABSTENTION_MARKER = "_abs"


class LongMemEvalTurn(ContractModel):
    """One user or assistant turn inside a dated haystack session."""

    turn_id: Identifier
    role: Literal["user", "assistant"]
    content: str
    has_answer: bool = False


class LongMemEvalSession(ContractModel):
    """One dated session of the question's private haystack."""

    # Position in the haystack. 15 of the 500 `longmemeval_s` questions plant
    # the same `session_id` twice -- same turns, different dates -- so the
    # published ID does not identify a session on its own. Upstream keeps both
    # copies because its haystack is a list; this keeps both and disambiguates
    # them by where they sit.
    position: int = Field(ge=0)
    session_id: Identifier
    occurred_at: AwareDatetime
    is_answer_session: bool
    # Not `min_length=1`: the release ships sessions with no turns at all, and
    # dropping the question they belong to would change the question set.
    turns: tuple[LongMemEvalTurn, ...] = ()


class LongMemEvalQuestion(ContractModel):
    """One question with the haystack it owns and its official labels."""

    question_id: Identifier
    question_type: LongMemEvalQuestionType
    question: NonEmptyString
    reference_answer: NonEmptyString
    question_date: AwareDatetime
    abstention: bool
    sessions: tuple[LongMemEvalSession, ...] = Field(min_length=1)


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str
    has_answer: bool = False


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    question_type: str
    question: str
    # 32 of the 500 `longmemeval_s` answers are JSON numbers rather than
    # strings (`3`, `120`, `1300`). Upstream never notices: its judge prompt
    # interpolates the value with `str.format`, which stringifies whatever it
    # is handed. Accepting both and normalising keeps those 32 questions.
    answer: str | int | float
    question_date: str
    haystack_dates: list[str] = Field(min_length=1)
    haystack_session_ids: list[str] = Field(min_length=1)
    haystack_sessions: list[list[_RawTurn]] = Field(min_length=1)
    answer_session_ids: list[str] = Field(default_factory=list)


_QUESTIONS = TypeAdapter(list[_RawQuestion])


def load_longmemeval(dataset_path: Path) -> tuple[LongMemEvalQuestion, ...]:
    """Load one official release file (`longmemeval_s`, `_m`, or `_oracle`).

    The release files carry no extension, so the caller's path -- not a suffix
    -- decides which split is read.
    """
    raw_questions = _QUESTIONS.validate_python(json.loads(dataset_path.read_bytes()))
    questions = tuple(_question(raw) for raw in raw_questions)
    if not questions:
        raise ValueError("LongMemEval annotations must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("LongMemEval annotations contain duplicate question IDs")
    return questions


def _question(raw: _RawQuestion) -> LongMemEvalQuestion:
    count = len(raw.haystack_sessions)
    if len(raw.haystack_dates) != count or len(raw.haystack_session_ids) != count:
        raise ValueError(f"LongMemEval question {raw.question_id} has a misaligned haystack")
    answer_sessions = set(raw.answer_session_ids)
    sessions = tuple(
        _session(position, session_id, date, turns, session_id in answer_sessions)
        for position, (session_id, date, turns) in enumerate(
            zip(
                raw.haystack_session_ids,
                raw.haystack_dates,
                raw.haystack_sessions,
                strict=True,
            )
        )
    )
    return LongMemEvalQuestion(
        question_id=raw.question_id,
        question_type=_question_type(raw.question_id, raw.question_type),
        question=raw.question,
        reference_answer=str(raw.answer),
        question_date=_date(raw.question_date),
        abstention=_ABSTENTION_MARKER in raw.question_id,
        sessions=sessions,
    )


def _session(
    position: int,
    session_id: str,
    date: str,
    turns: list[_RawTurn],
    is_answer_session: bool,
) -> LongMemEvalSession:
    return LongMemEvalSession(
        position=position,
        session_id=session_id,
        occurred_at=_date(date),
        is_answer_session=is_answer_session,
        turns=tuple(
            LongMemEvalTurn(
                turn_id=f"S{position:04d}_{session_id}_T{index:04d}",
                role=_role(turn.role),
                content=turn.content,
                has_answer=turn.has_answer,
            )
            for index, turn in enumerate(turns)
        ),
    )


def _question_type(question_id: str, value: str) -> LongMemEvalQuestionType:
    normalized = value.strip()
    if normalized not in {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    }:
        raise ValueError(f"LongMemEval question {question_id} has an unknown type: {value}")
    return normalized  # type: ignore[return-value]


def _role(value: str) -> Literal["user", "assistant"]:
    normalized = value.strip().casefold()
    if normalized in {"user", "assistant"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid LongMemEval dialogue role: {value}")


def _date(value: str) -> datetime:
    match = _DATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid LongMemEval date: {value}")
    try:
        return datetime.strptime(" ".join(match.groups()), "%Y/%m/%d %H:%M").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"invalid LongMemEval date: {value}") from error
