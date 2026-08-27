"""Thin adapter for the official MM-Lifelong Day, Week, and Month splits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks._contracts import ContractModel, NonEmptyString

MM_LIFELONG_ADAPTER_VERSION = "mm_lifelong_official_v1"
MMLifelongSplit = Literal["day_test", "week_test", "month_train", "month_val"]


class MMLifelongQuestion(ContractModel):
    """One open-ended question plus labels retained only for offline evaluation."""

    index: int = Field(ge=0)
    split: MMLifelongSplit
    question: NonEmptyString
    reference_answer: NonEmptyString
    question_type: NonEmptyString
    temporal_certificate: NonEmptyString
    clue_interval_count: int = Field(gt=0)
    reference_intervals: tuple[tuple[float, float], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_reference_intervals(self) -> MMLifelongQuestion:
        if any(start < 0 or end < 0 for start, end in self.reference_intervals):
            raise ValueError("MM-Lifelong reference intervals must be non-negative")
        return self


class _RawClueSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    intervals: list[tuple[float, float]] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0)
    question: str
    answer: str
    question_type: str
    temporal_certificate: str
    clue_intervals: list[object] | None = None
    clue_interval: list[object] | None = None
    total_intervals: list[tuple[float, float]] = Field(min_length=1)


_INTERVALS = TypeAdapter(list[tuple[float, float]])
_SOURCES = TypeAdapter(list[_RawClueSource])


def load_mm_lifelong(
    annotation_path: Path,
    split: MMLifelongSplit,
) -> tuple[MMLifelongQuestion, ...]:
    """Load one official split while rejecting a mismatched split/schema flag."""
    raw_questions = TypeAdapter(list[_RawQuestion]).validate_json(annotation_path.read_bytes())
    questions = tuple(_question(raw, split) for raw in raw_questions)
    if not questions:
        raise ValueError("MM-Lifelong annotations must not be empty")
    if len({question.index for question in questions}) != len(questions):
        raise ValueError("MM-Lifelong annotations contain duplicate indices")
    return questions


def _question(raw: _RawQuestion, split: MMLifelongSplit) -> MMLifelongQuestion:
    clue_count = _clue_count(raw, split)
    return MMLifelongQuestion(
        index=raw.index,
        split=split,
        question=raw.question,
        reference_answer=raw.answer,
        question_type=raw.question_type,
        temporal_certificate=raw.temporal_certificate,
        clue_interval_count=clue_count,
        reference_intervals=tuple(raw.total_intervals),
    )


def _clue_count(raw: _RawQuestion, split: MMLifelongSplit) -> int:
    if split == "day_test":
        if raw.clue_intervals is None or raw.clue_interval is not None:
            raise ValueError("MM-Lifelong day_test requires clue_intervals")
        intervals = _INTERVALS.validate_python(raw.clue_intervals)
        _validate_intervals(intervals)
        return len(intervals)
    if split == "week_test":
        if raw.clue_intervals is None or raw.clue_interval is not None:
            raise ValueError("MM-Lifelong week_test requires source-grouped clue_intervals")
        sources = _SOURCES.validate_python(raw.clue_intervals)
    else:
        if raw.clue_interval is None or raw.clue_intervals is not None:
            raise ValueError("MM-Lifelong month splits require clue_interval")
        sources = _SOURCES.validate_python(raw.clue_interval)
    if not sources:
        raise ValueError("MM-Lifelong clue sources must not be empty")
    for source in sources:
        _validate_intervals(source.intervals)
    return sum(len(source.intervals) for source in sources)


def _validate_intervals(intervals: list[tuple[float, float]]) -> None:
    if not intervals or any(start < 0 or end < 0 for start, end in intervals):
        raise ValueError("MM-Lifelong clue intervals must be non-negative")
