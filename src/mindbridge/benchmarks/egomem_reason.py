"""Thin adapter for the official EgoMemReason public JSONL release."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

OPTION_LABELS = tuple("ABCDEFGHIJ")

EGOMEM_REASON_ADAPTER_VERSION = "egomem_reason_official_v1"
_QUERY_TIME_PATTERN = re.compile(r"^DAY([1-9][0-9]*), ([0-9]{2}):([0-9]{2}):([0-9]{2})$")


class EgoMemReasonQuestion(ContractModel):
    """One answer-key-free EgoMemReason multiple-choice question."""

    example_id: int = Field(gt=0)
    question_id: Identifier
    identity: Identifier
    query_time: NonEmptyString
    query_offset_ms: int = Field(ge=0)
    question: NonEmptyString
    choices: tuple[NonEmptyString, ...] = Field(min_length=4, max_length=10)
    query_type: NonEmptyString


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    example_id: int = Field(gt=0)
    p_id: str
    identity: str
    query_time: str
    question: str
    options: dict[str, str]
    query_type: str


def load_egomem_reason(annotation_path: Path) -> tuple[EgoMemReasonQuestion, ...]:
    """Load the v1.1 public JSONL without inventing or retaining answer keys."""
    raw_questions = tuple(
        _RawQuestion.model_validate_json(line)
        for line in annotation_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    )
    questions = tuple(_question(raw) for raw in raw_questions)
    if not questions:
        raise ValueError("EgoMemReason annotations must not be empty")
    if len({question.example_id for question in questions}) != len(questions):
        raise ValueError("EgoMemReason annotations contain duplicate example IDs")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("EgoMemReason annotations contain duplicate question IDs")
    return questions


def _question(raw: _RawQuestion) -> EgoMemReasonQuestion:
    labels = tuple(raw.options)
    if labels != OPTION_LABELS[: len(labels)]:
        raise ValueError(
            f"EgoMemReason question {raw.example_id} options must be consecutive A-J labels"
        )
    return EgoMemReasonQuestion(
        example_id=raw.example_id,
        question_id=raw.p_id,
        identity=raw.identity,
        query_time=raw.query_time,
        query_offset_ms=_query_offset_ms(raw.query_time),
        question=raw.question,
        choices=tuple(raw.options.values()),
        query_type=raw.query_type,
    )


def _query_offset_ms(value: str) -> int:
    match = _QUERY_TIME_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid EgoMemReason query time: {value}")
    day, hours, minutes, seconds = map(int, match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid EgoMemReason query time: {value}")
    return ((day - 1) * 86_400 + hours * 3_600 + minutes * 60 + seconds) * 1_000
