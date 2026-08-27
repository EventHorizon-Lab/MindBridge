"""Thin adapter for the official SuperMemory-VQA test annotations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

SUPERMEMORY_VQA_ADAPTER_VERSION = "supermemory_vqa_official_v3"
SUPERMEMORY_UNANSWERABLE_CHOICE = "This question can not be answered."


class SuperMemoryQuestion(ContractModel):
    """One causal multiple-choice query without answer-evidence annotations."""

    question_id: int = Field(gt=0)
    subject: int = Field(gt=0)
    question: NonEmptyString
    choices: tuple[NonEmptyString, ...] = Field(min_length=4, max_length=4)
    correct_option_index: int = Field(ge=0, lt=4)
    unanswerable_option_index: int = Field(ge=0, lt=4)
    is_answerable: bool
    skill: Identifier
    source_video_ids: tuple[Identifier, ...] = Field(min_length=1)
    question_video_id: Identifier
    question_ended_at: AwareDatetime


class _RawMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    skill: str
    primary_video_id: str


class _RawSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_time: int | float
    end_time: int | float
    video_id: str
    video_start_time_unix: int | float


class _RawLegacySpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_time: str | int | float
    end_time: str | int | float


class _RawQuestionEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    time_spans: list[_RawSpan] | None = None
    time_span: _RawLegacySpan | None = None
    video_id: str | None = None
    start_time: int | float | None = None


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: int = Field(gt=0)
    question: str
    choices: list[str]
    correct_answer: str
    correct_option_index: int
    choice_types: list[str]
    subject: int = Field(gt=0)
    metadata: _RawMetadata
    video_ids: list[str]
    question_evidence: _RawQuestionEvidence
    is_answerable: bool


def load_supermemory_vqa(annotation_path: Path) -> tuple[SuperMemoryQuestion, ...]:
    """Load the official JSON while discarding its answer-evidence field."""
    raw_questions = TypeAdapter(list[_RawQuestion]).validate_json(annotation_path.read_bytes())
    questions = tuple(_question(question) for question in raw_questions)
    if not questions:
        raise ValueError("SuperMemory-VQA annotations must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("SuperMemory-VQA annotations contain duplicate question IDs")
    return questions


def _question(raw: _RawQuestion) -> SuperMemoryQuestion:
    if len(raw.choices) != 4 or len(raw.choice_types) != 4:
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} must have four choices")
    if not 0 <= raw.correct_option_index < len(raw.choices):
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} has an invalid answer index")
    if raw.choices[raw.correct_option_index] != raw.correct_answer:
        raise ValueError(
            f"SuperMemory-VQA question {raw.question_id} has inconsistent answer fields"
        )
    if raw.choice_types[raw.correct_option_index] != "correct":
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} has an invalid correct type")
    unanswerable = [
        index
        for index, choice in enumerate(raw.choices)
        if choice == SUPERMEMORY_UNANSWERABLE_CHOICE
    ]
    if len(unanswerable) != 1:
        raise ValueError(
            f"SuperMemory-VQA question {raw.question_id} must have one unanswerable choice"
        )
    if raw.is_answerable == (raw.correct_option_index == unanswerable[0]):
        raise ValueError(
            f"SuperMemory-VQA question {raw.question_id} has inconsistent answerability"
        )
    if not raw.video_ids or len(set(raw.video_ids)) != len(raw.video_ids):
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} has invalid video IDs")
    if raw.metadata.primary_video_id not in raw.video_ids:
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} omits its primary video")
    question_video_id, question_ended_at = _question_end(raw)
    if question_video_id not in raw.video_ids:
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} omits its question video")
    return SuperMemoryQuestion(
        question_id=raw.question_id,
        subject=raw.subject,
        question=raw.question,
        choices=tuple(raw.choices),
        correct_option_index=raw.correct_option_index,
        unanswerable_option_index=unanswerable[0],
        is_answerable=raw.is_answerable,
        skill=raw.metadata.skill,
        source_video_ids=tuple(raw.video_ids),
        question_video_id=question_video_id,
        question_ended_at=datetime.fromtimestamp(question_ended_at, tz=timezone.utc),
    )


def _question_end(raw: _RawQuestion) -> tuple[str, float]:
    evidence = raw.question_evidence
    if evidence.time_spans:
        if len(evidence.time_spans) != 1:
            raise ValueError(
                f"SuperMemory-VQA question {raw.question_id} has multiple question spans"
            )
        span = evidence.time_spans[0]
        if span.start_time < 0 or span.end_time < span.start_time:
            raise ValueError(f"SuperMemory-VQA question {raw.question_id} has an invalid time span")
        return span.video_id, float(span.video_start_time_unix) + float(span.end_time)
    if evidence.time_span is None or evidence.start_time is None or evidence.video_id is None:
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} has no question time")
    start = _seconds(evidence.time_span.start_time)
    end = _seconds(evidence.time_span.end_time)
    if start < 0 or end < start:
        raise ValueError(f"SuperMemory-VQA question {raw.question_id} has an invalid time span")
    return evidence.video_id, float(evidence.start_time) + end


def _seconds(value: str | int | float) -> float:
    if not isinstance(value, str):
        return float(value)
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"invalid SuperMemory-VQA local timestamp: {value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"invalid SuperMemory-VQA local timestamp: {value}") from error
    hours, minutes, seconds = (0.0, *numbers) if len(numbers) == 2 else numbers
    if minutes >= 60 or seconds >= 60 or min(numbers) < 0:
        raise ValueError(f"invalid SuperMemory-VQA local timestamp: {value}")
    return hours * 3_600 + minutes * 60 + seconds
