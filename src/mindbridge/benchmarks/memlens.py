"""Thin adapter for the official MEMLENS long-context JSON releases."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString

MEMLENS_ADAPTER_VERSION = "memlens_official_v1"
_DATE_PATTERN = re.compile(r"^(\d{4}/\d{2}/\d{2}) \([A-Za-z]{3}\) (\d{2}:\d{2})$")


class MemLensImage(ContractModel):
    """One official image reference attached to its original dialogue turn."""

    source_file: NonEmptyString
    source_url: NonEmptyString | None = None
    caption: NonEmptyString | None = None


class MemLensTurn(ContractModel):
    """One user or assistant turn in a dated MEMLENS session."""

    turn_id: Identifier
    role: Literal["user", "assistant"]
    content: str
    images: tuple[MemLensImage, ...] = ()

    @model_validator(mode="after")
    def require_content_or_image(self) -> MemLensTurn:
        if not self.content.strip() and not self.images:
            raise ValueError("MEMLENS turns require text or an image")
        if len({image.source_file for image in self.images}) != len(self.images):
            raise ValueError("MEMLENS turn image references must be unique")
        return self


class MemLensSession(ContractModel):
    """One dated conversation session retained in official source order."""

    session_id: Identifier
    occurred_at: AwareDatetime
    turns: tuple[MemLensTurn, ...]


class MemLensQuestion(ContractModel):
    """One question-specific memory context and official reference answer."""

    question_id: Identifier
    question_type: NonEmptyString
    question_subtype: NonEmptyString | None = None
    question: NonEmptyString
    reference_answer: NonEmptyString
    question_date: AwareDatetime
    old_answer: NonEmptyString | None = None
    sessions: tuple[MemLensSession, ...] = Field(min_length=1)


class _RawImage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file: str
    image_url: str | None = None
    blip_caption: str | None = None


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str
    images: list[_RawImage] = Field(default_factory=list)


class _RawSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str
    session: list[_RawTurn] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    question_type: str
    question_subtype: str | None = None
    question: str
    answer: str
    question_date: str
    old_answer: str | None = None
    haystack_dates: list[str] = Field(default_factory=list)
    haystack_session_ids: list[str] = Field(default_factory=list)
    haystack_sessions: list[list[_RawTurn] | _RawSession] = Field(min_length=1)


class _RawAgentSubset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_questions: int = Field(gt=0)
    question_ids: list[str] = Field(min_length=1)


def load_memlens(dataset_path: Path) -> tuple[MemLensQuestion, ...]:
    """Load one of the official 32K/64K/128K/256K JSON files."""
    raw = _load_raw_questions(dataset_path)
    questions = tuple(_question(item) for item in raw)
    if not questions:
        raise ValueError("MEMLENS annotations must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("MEMLENS annotations contain duplicate question IDs")
    return questions


def load_memlens_agent_subset(index_path: Path) -> tuple[Identifier, ...]:
    """Load the official fixed 195-question memory-agent index."""
    subset = _RawAgentSubset.model_validate_json(index_path.read_bytes())
    if subset.n_questions != len(subset.question_ids):
        raise ValueError("MEMLENS agent subset count does not match its question IDs")
    if len(set(subset.question_ids)) != len(subset.question_ids):
        raise ValueError("MEMLENS agent subset contains duplicate question IDs")
    return tuple(subset.question_ids)


def _load_raw_questions(path: Path) -> list[_RawQuestion]:
    import json

    payload = json.loads(path.read_bytes())
    if isinstance(payload, dict):
        payload = payload.get("data", payload)
    if not isinstance(payload, list):
        raise ValueError("MEMLENS dataset root must be a list or contain a data list")
    return [_RawQuestion.model_validate(item) for item in payload]


def _question(raw: _RawQuestion) -> MemLensQuestion:
    session_count = len(raw.haystack_sessions)
    if raw.haystack_dates and len(raw.haystack_dates) != session_count:
        raise ValueError(f"MEMLENS question {raw.question_id} has misaligned session dates")
    if raw.haystack_session_ids and len(raw.haystack_session_ids) != session_count:
        raise ValueError(f"MEMLENS question {raw.question_id} has misaligned session IDs")

    sessions = tuple(
        _session(raw, index, session) for index, session in enumerate(raw.haystack_sessions)
    )
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError(f"MEMLENS question {raw.question_id} has duplicate session IDs")
    return MemLensQuestion(
        question_id=raw.question_id,
        question_type=raw.question_type,
        question_subtype=raw.question_subtype,
        question=raw.question,
        reference_answer=raw.answer,
        question_date=_date(raw.question_date),
        old_answer=raw.old_answer,
        sessions=sessions,
    )


def _session(
    question: _RawQuestion,
    index: int,
    raw: list[_RawTurn] | _RawSession,
) -> MemLensSession:
    if isinstance(raw, _RawSession):
        date = raw.date
        turns = raw.session
    else:
        if index >= len(question.haystack_dates):
            raise ValueError(f"MEMLENS question {question.question_id} is missing a session date")
        date = question.haystack_dates[index]
        turns = raw
    session_id = (
        question.haystack_session_ids[index]
        if index < len(question.haystack_session_ids)
        else f"session_{index:04d}"
    )
    return MemLensSession(
        session_id=session_id,
        occurred_at=_date(date),
        turns=tuple(
            _turn(session_id, turn_index, turn)
            for turn_index, turn in enumerate(turns)
            if turn.content.strip() or turn.images
        ),
    )


def _turn(session_id: str, index: int, raw: _RawTurn) -> MemLensTurn:
    return MemLensTurn(
        turn_id=f"{session_id}_T{index:04d}",
        role=_role(raw.role),
        content=raw.content,
        images=tuple(
            MemLensImage(
                source_file=image.file,
                source_url=_nonempty_or_none(image.image_url),
                caption=_nonempty_or_none(image.blip_caption),
            )
            for image in raw.images
        ),
    )


def _role(value: str) -> Literal["user", "assistant"]:
    normalized = value.strip().casefold()
    if normalized == "user":
        return "user"
    if normalized in {"assistant", "ai assistant"}:
        return "assistant"
    raise ValueError(f"invalid MEMLENS dialogue role: {value}")


def _nonempty_or_none(value: str | None) -> str | None:
    return value.strip() if value is not None and value.strip() else None


def _date(value: str) -> datetime:
    match = _DATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid MEMLENS date: {value}")
    try:
        return datetime.strptime(" ".join(match.groups()), "%Y/%m/%d %H:%M").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"invalid MEMLENS date: {value}") from error
