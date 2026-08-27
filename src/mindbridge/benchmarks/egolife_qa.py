"""Thin adapter for the official EgoLifeQA multiple-choice release."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

EGOLIFE_QA_ADAPTER_VERSION = "egolife_qa_official_v2"
EGOLIFE_VIDEO_FPS = 20
EgoLifeOption = Literal["A", "B", "C", "D"]
_DAY_PATTERN = re.compile(r"DAY([1-9][0-9]*)", re.IGNORECASE)
_TIMECODE_PATTERN = re.compile(r"[0-9]{8}")


class EgoLifeQuestion(ContractModel):
    """One causal EgoLifeQA query and its four official choices."""

    question_id: Identifier
    question: NonEmptyString
    choices: tuple[NonEmptyString, ...] = Field(min_length=4, max_length=4)
    correct_option: EgoLifeOption
    query_day: int = Field(ge=1)
    query_timecode: str = Field(pattern=r"^[0-9]{8}$")
    query_offset_ms: int = Field(ge=0)
    question_type: NonEmptyString
    needs_audio: bool
    needs_name: bool
    asks_last_time: bool


class _RawMoment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    time: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(alias="ID")
    query_time: _RawMoment
    type: str
    need_audio: bool
    need_name: bool
    last_time: bool
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    answer: EgoLifeOption


def load_egolife_qa(annotation_path: Path) -> tuple[EgoLifeQuestion, ...]:
    """Load the official QA JSON without retaining retrieval-hint annotations."""
    raw_questions = TypeAdapter(list[_RawQuestion]).validate_json(annotation_path.read_bytes())
    questions = tuple(_question(question) for question in raw_questions)
    if not questions:
        raise ValueError("EgoLifeQA annotations must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("EgoLifeQA annotations contain duplicate question IDs")
    return questions


def _question(raw: _RawQuestion) -> EgoLifeQuestion:
    day, offset_ms = _moment_offset_ms(raw.query_time)
    return EgoLifeQuestion(
        question_id=raw.question_id,
        question=raw.question,
        choices=(raw.choice_a, raw.choice_b, raw.choice_c, raw.choice_d),
        correct_option=raw.answer,
        query_day=day,
        query_timecode=raw.query_time.time,
        query_offset_ms=offset_ms,
        question_type=raw.type,
        needs_audio=raw.need_audio,
        needs_name=raw.need_name,
        asks_last_time=raw.last_time,
    )


def _moment_offset_ms(moment: _RawMoment) -> tuple[int, int]:
    day_match = _DAY_PATTERN.fullmatch(moment.date)
    if day_match is None:
        raise ValueError(f"invalid EgoLifeQA query time: {moment.date} {moment.time}")
    day = int(day_match.group(1))
    try:
        offset_ms = egolife_timecode_offset_ms(day, moment.time)
    except ValueError as error:
        raise ValueError(f"invalid EgoLifeQA query time: {moment.date} {moment.time}") from error
    return day, offset_ms


def egolife_timecode_offset_ms(day: int, timecode: str) -> int:
    """Convert the release's 20 FPS ``HHMMSSFF`` clock to milliseconds."""
    if _TIMECODE_PATTERN.fullmatch(timecode) is None:
        raise ValueError(f"invalid EgoLife timecode: {timecode}")
    hours, minutes, seconds, frames = (
        int(timecode[0:2]),
        int(timecode[2:4]),
        int(timecode[4:6]),
        int(timecode[6:8]),
    )
    if day < 1 or hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid EgoLife timecode: {timecode}")
    seconds_since_day_one = (day - 1) * 86_400 + hours * 3_600 + minutes * 60 + seconds
    return seconds_since_day_one * 1_000 + frames * 1_000 // EGOLIFE_VIDEO_FPS
