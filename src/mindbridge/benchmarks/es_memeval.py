"""Thin adapter for the official ES-MemEval EvoEmo QA release."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

ES_MEMEVAL_ADAPTER_VERSION = "es_memeval_qa_v1"


class EsMemEvalTurn(ContractModel):
    """One visible seeker/supporter turn in an EvoEmo dialogue session."""

    source_id: Identifier
    role: Literal["seeker", "supporter"]
    content: NonEmptyString


class EsMemEvalSession(ContractModel):
    """One dated dialogue session supplied to the evaluated agent."""

    session_id: Identifier
    occurred_at: AwareDatetime
    turns: tuple[EsMemEvalTurn, ...] = Field(min_length=1)


class EsMemEvalQuestion(ContractModel):
    """One EvoEmo QA item and its scorer-side labels."""

    question_id: Identifier
    group_id: Identifier
    capability: Literal[
        "information extraction",
        "temporal reasoning",
        "conflict detection",
        "abstention",
        "user modeling",
    ]
    question: NonEmptyString
    answer: NonEmptyString
    evidence: tuple[Identifier, ...] = ()


class EsMemEvalSeeker(ContractModel):
    """One independently evaluated person, dialogue corpus, and question set."""

    seeker_id: Identifier
    name: Identifier
    sessions: tuple[EsMemEvalSession, ...] = Field(min_length=1)
    questions: tuple[EsMemEvalQuestion, ...] = Field(min_length=1)


class _RawBasicInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    idx: int = Field(ge=1)
    role: str
    content: str


class _RawSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    timestamp: str
    dialogue: list[_RawTurn] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    idx: int = Field(ge=1)
    capability: str
    question: str
    answer: str
    evidence: list[str] = Field(default_factory=list)


class _RawQuestionGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    questions: list[_RawQuestion] = Field(min_length=1)


class _RawSeeker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    basic_info: _RawBasicInfo
    dialog_history: list[_RawSession] = Field(min_length=1)
    questions: list[_RawQuestionGroup] = Field(min_length=1)


_SEEKERS = TypeAdapter(list[_RawSeeker])
_CAPABILITIES = {
    "information extraction",
    "temporal reasoning",
    "conflict detection",
    "abstention",
    "user modeling",
}


def load_es_memeval(dataset_path: Path) -> tuple[EsMemEvalSeeker, ...]:
    """Load the pinned ``data/evo_emo.json`` QA annotations.

    ES-MemEval also publishes summarization and dialogue-generation labels. The
    unified ``mindbridge-bench eval`` contract is question/answer based, so this
    adapter deliberately exposes the paper's QA task only.
    """
    raw_seekers = _SEEKERS.validate_python(json.loads(dataset_path.read_bytes()))
    seekers = tuple(_seeker(raw) for raw in raw_seekers)
    if not seekers:
        raise ValueError("ES-MemEval annotations must not be empty")
    if len({seeker.seeker_id for seeker in seekers}) != len(seekers):
        raise ValueError("ES-MemEval annotations contain duplicate seeker IDs")
    return seekers


def _seeker(raw: _RawSeeker) -> EsMemEvalSeeker:
    sessions = tuple(_session(raw.id, session) for session in raw.dialog_history)
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError(f"ES-MemEval seeker {raw.id} contains duplicate session IDs")
    questions = tuple(
        _question(raw.id, group.id, question)
        for group in raw.questions
        for question in group.questions
    )
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError(f"ES-MemEval seeker {raw.id} contains duplicate question IDs")
    return EsMemEvalSeeker(
        seeker_id=raw.id,
        name=raw.basic_info.name,
        sessions=sessions,
        questions=questions,
    )


def _session(seeker_id: str, raw: _RawSession) -> EsMemEvalSession:
    turns = tuple(
        EsMemEvalTurn(
            source_id=f"{raw.id}:{turn.idx}",
            role=_role(turn.role),
            content=turn.content,
        )
        for turn in raw.dialogue
    )
    if len({turn.source_id for turn in turns}) != len(turns):
        raise ValueError(
            f"ES-MemEval seeker {seeker_id} session {raw.id} contains duplicate turn IDs"
        )
    return EsMemEvalSession(
        session_id=raw.id,
        occurred_at=_date(raw.timestamp),
        turns=turns,
    )


def _question(seeker_id: str, group_id: str, raw: _RawQuestion) -> EsMemEvalQuestion:
    capability = raw.capability.strip().casefold()
    if capability not in _CAPABILITIES:
        raise ValueError(
            f"ES-MemEval question {seeker_id}/{group_id}/{raw.idx} has an unknown capability: "
            f"{raw.capability}"
        )
    return EsMemEvalQuestion(
        question_id=f"{group_id}:{raw.idx}",
        group_id=group_id,
        capability=capability,  # type: ignore[arg-type]
        question=raw.question,
        answer=raw.answer,
        evidence=tuple(raw.evidence),
    )


def _role(value: str) -> Literal["seeker", "supporter"]:
    normalized = value.strip().casefold()
    if normalized in {"seeker", "supporter"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid ES-MemEval dialogue role: {value}")


def _date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ValueError(f"invalid ES-MemEval session date: {value}") from error
