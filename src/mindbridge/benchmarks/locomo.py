"""Thin, deterministic adapter for the official LoCoMo JSON release."""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString

LOCOMO_ADAPTER_VERSION = "locomo_official_v1"
_SESSION_TIME_FORMAT = "%I:%M %p on %d %B, %Y"
_BenchmarkSource = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LoCoMoTurn(ContractModel):
    """One original dialogue turn retained as benchmark source memory."""

    dialog_id: Identifier
    speaker: NonEmptyString
    text: NonEmptyString
    occurred_at: AwareDatetime
    image_sources: tuple[_BenchmarkSource, ...] = ()
    image_caption: NonEmptyString | None = None


class LoCoMoQuestion(ContractModel):
    """One official QA item with its source dialogue references."""

    question_id: Identifier
    question: NonEmptyString
    reference_answers: tuple[NonEmptyString, ...] = Field(min_length=1)
    evidence_dialog_ids: tuple[Identifier, ...]
    category: int = Field(ge=1, le=5)


class LoCoMoConversation(ContractModel):
    """One complete long-horizon conversation and its evaluation questions."""

    sample_id: Identifier
    turns: tuple[LoCoMoTurn, ...] = Field(min_length=1)
    questions: tuple[LoCoMoQuestion, ...] = Field(min_length=1)


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str
    dia_id: str
    text: str
    img_url: str | list[str] | None = None
    blip_caption: str | None = None


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str | int | float | list[str | int | float] | None = None
    evidence: list[str] = Field(default_factory=list)
    category: int


class _RawConversationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    conversation: dict[str, object]
    qa: list[_RawQuestion]


def load_locomo(
    dataset_path: Path,
    *,
    source_timezone: tzinfo = timezone.utc,
) -> tuple[LoCoMoConversation, ...]:
    """Load the official `locomo10.json` without copying its evaluation logic."""
    records = TypeAdapter(list[_RawConversationRecord]).validate_json(dataset_path.read_bytes())
    return tuple(
        _adapt_locomo_record(record, source_timezone=source_timezone) for record in records
    )


def _adapt_locomo_record(
    record: _RawConversationRecord,
    *,
    source_timezone: tzinfo = timezone.utc,
) -> LoCoMoConversation:
    """Normalize one dynamic-session record into stable chronological contracts."""
    session_numbers = sorted(
        int(key.removeprefix("session_"))
        for key in record.conversation
        if key.startswith("session_") and key.removeprefix("session_").isdigit()
    )
    turns: list[LoCoMoTurn] = []
    for session_number in session_numbers:
        session_key = f"session_{session_number}"
        timestamp_key = f"{session_key}_date_time"
        timestamp = _session_datetime(record.conversation.get(timestamp_key), source_timezone)
        raw_turns = TypeAdapter(list[_RawTurn]).validate_python(record.conversation[session_key])
        turns.extend(_turn(turn, occurred_at=timestamp) for turn in raw_turns)
    if len({turn.dialog_id for turn in turns}) != len(turns):
        raise ValueError(f"LoCoMo sample {record.sample_id} contains duplicate dialogue IDs")

    questions = tuple(
        _question(record.sample_id, ordinal, question)
        for ordinal, question in enumerate(record.qa, start=1)
    )
    return LoCoMoConversation(
        sample_id=record.sample_id,
        turns=tuple(turns),
        questions=questions,
    )


def _turn(turn: _RawTurn, *, occurred_at: datetime) -> LoCoMoTurn:
    image_sources = (turn.img_url,) if isinstance(turn.img_url, str) else tuple(turn.img_url or ())
    return LoCoMoTurn(
        dialog_id=turn.dia_id,
        speaker=turn.speaker,
        text=turn.text,
        occurred_at=occurred_at,
        image_sources=image_sources,
        image_caption=turn.blip_caption,
    )


def _question(
    sample_id: str,
    ordinal: int,
    question: _RawQuestion,
) -> LoCoMoQuestion:
    answers = (
        ("Not mentioned in the conversation",)
        if question.category == 5
        else _reference_answers(question.answer)
    )
    evidence_ids = tuple(
        dict.fromkeys(
            part.strip()
            for evidence in question.evidence
            for part in evidence.split(";")
            if part.strip()
        )
    )
    return LoCoMoQuestion(
        question_id=f"{sample_id}_Q{ordinal:04d}",
        question=question.question,
        reference_answers=answers,
        evidence_dialog_ids=evidence_ids,
        category=question.category,
    )


def _reference_answers(
    answer: str | int | float | list[str | int | float] | None,
) -> tuple[str, ...]:
    if answer is None:
        raise ValueError("non-adversarial LoCoMo question is missing its answer")
    values = answer if isinstance(answer, list) else [answer]
    return tuple(str(value) for value in values)


def _session_datetime(value: object, source_timezone: tzinfo) -> datetime:
    if not isinstance(value, str):
        raise ValueError("LoCoMo session is missing its date_time string")
    return datetime.strptime(value, _SESSION_TIME_FORMAT).replace(tzinfo=source_timezone)
