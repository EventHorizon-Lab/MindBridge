"""Thin, deterministic adapter for the official LoCoMo-Refined JSON release.

Reads `data/raw/locomo_refined.json` from `mem-eval-suite/LoCoMo_refined`, which
keeps the original LoCoMo record layout (`sample_id`, a `session_N` /
`session_N_date_time` conversation, and a `qa` list) but recalibrates what is in
it: 337 of the 1,382 questions were revised, every `answer` is a list of
complete gold candidates, and the adversarial category 5 is gone entirely.

`question_id` is the release's own `qa_id` (`{sample_id}#q{index:04d}`, the same
value `data/public/questions.jsonl` publishes), because that is the key the
official evaluator joins predictions on.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

LOCOMO_REFINED_ADAPTER_VERSION = "locomo_refined_v1"
_SESSION_TIME_FORMAT = "%I:%M %p on %d %B, %Y"
_Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
_Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
_Source = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LoCoMoRefinedTurn(BaseModel):
    """One original dialogue turn retained as benchmark source memory."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dialog_id: _Identifier
    speaker: _Text
    text: _Text
    occurred_at: AwareDatetime
    image_sources: tuple[_Source, ...] = ()
    image_caption: _Text | None = None


class LoCoMoRefinedQuestion(BaseModel):
    """One official QA item with its source dialogue references."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    question_id: _Identifier
    question: _Text
    reference_answers: tuple[_Text, ...] = Field(min_length=1)
    evidence_dialog_ids: tuple[_Identifier, ...]
    # LoCoMo-Refined dropped LoCoMo's adversarial category 5 outright, so there is no
    # abstention protocol left to model and no four-versus-five-category ambiguity in
    # what a reported score covers.
    category: int = Field(ge=1, le=4)
    is_multi_modality: bool


class LoCoMoRefinedConversation(BaseModel):
    """One complete long-horizon conversation and its evaluation questions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sample_id: _Identifier
    turns: tuple[LoCoMoRefinedTurn, ...] = Field(min_length=1)
    questions: tuple[LoCoMoRefinedQuestion, ...] = Field(min_length=1)


class _RawTurn(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    speaker: str
    dia_id: str
    text: str
    img_url: str | list[str] | None = None
    blip_caption: str | None = None


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    question: str
    # Always a list in this release; the six numeric golds ("2022", "3") are published as
    # strings in `questions.jsonl`, so they are stringified here to match.
    answer: list[str | int | float] = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    category: int
    is_multi_modality: bool = False


class _RawConversationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    sample_id: str
    conversation: dict[str, object]
    qa: list[_RawQuestion]


def load_locomo_refined(
    dataset_path: Path,
    *,
    source_timezone: tzinfo = timezone.utc,
) -> tuple[LoCoMoRefinedConversation, ...]:
    """Load the official `locomo_refined.json` without copying its evaluation logic."""
    records = TypeAdapter(list[_RawConversationRecord]).validate_json(dataset_path.read_bytes())
    return tuple(_adapt_record(record, source_timezone=source_timezone) for record in records)


def _adapt_record(
    record: _RawConversationRecord,
    *,
    source_timezone: tzinfo = timezone.utc,
) -> LoCoMoRefinedConversation:
    """Normalize one dynamic-session record into stable chronological contracts."""
    session_numbers = sorted(
        int(key.removeprefix("session_"))
        for key in record.conversation
        if key.startswith("session_") and key.removeprefix("session_").isdigit()
    )
    turns: list[LoCoMoRefinedTurn] = []
    for session_number in session_numbers:
        session_key = f"session_{session_number}"
        timestamp_key = f"{session_key}_date_time"
        timestamp = _session_datetime(record.conversation.get(timestamp_key), source_timezone)
        raw_turns = TypeAdapter(list[_RawTurn]).validate_python(record.conversation[session_key])
        turns.extend(_turn(turn, occurred_at=timestamp) for turn in raw_turns)
    if len({turn.dialog_id for turn in turns}) != len(turns):
        raise ValueError(f"LoCoMo-Refined sample {record.sample_id} has duplicate dialogue IDs")

    questions = tuple(
        _question(record.sample_id, index, question) for index, question in enumerate(record.qa)
    )
    return LoCoMoRefinedConversation(
        sample_id=record.sample_id,
        turns=tuple(turns),
        questions=questions,
    )


def _turn(turn: _RawTurn, *, occurred_at: datetime) -> LoCoMoRefinedTurn:
    image_sources = (turn.img_url,) if isinstance(turn.img_url, str) else tuple(turn.img_url or ())
    return LoCoMoRefinedTurn(
        dialog_id=turn.dia_id,
        speaker=turn.speaker,
        text=turn.text,
        occurred_at=occurred_at,
        image_sources=image_sources,
        image_caption=turn.blip_caption,
    )


def _question(sample_id: str, index: int, question: _RawQuestion) -> LoCoMoRefinedQuestion:
    evidence_ids = tuple(
        dict.fromkeys(
            part.strip()
            for evidence in question.evidence
            for part in evidence.split(";")
            if part.strip()
        )
    )
    return LoCoMoRefinedQuestion(
        question_id=official_qa_id(sample_id, index),
        question=question.question,
        reference_answers=tuple(str(value) for value in question.answer),
        evidence_dialog_ids=evidence_ids,
        category=question.category,
        is_multi_modality=question.is_multi_modality,
    )


def official_qa_id(sample_id: str, index: int) -> str:
    """Build the release's own `qa_id`, which its evaluator joins predictions on."""
    return f"{sample_id}#q{index:04d}"


def _session_datetime(value: object, source_timezone: tzinfo) -> datetime:
    if not isinstance(value, str):
        raise ValueError("LoCoMo-Refined session is missing its date_time string")
    return datetime.strptime(value, _SESSION_TIME_FORMAT).replace(tzinfo=source_timezone)
