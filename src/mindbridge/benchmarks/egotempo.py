"""Official EgoTempo adapter and production-path runner."""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mindbridge.benchmarks._contracts import ContractModel, Identifier, NonEmptyString


class EgoTempoQuestion(ContractModel):
    """One open question over an official Ego4D temporal clip."""

    question_id: Identifier
    clip_id: Identifier
    source_video_id: Identifier
    clip_start_seconds: float = Field(allow_inf_nan=False)
    clip_end_seconds: float = Field(gt=0, allow_inf_nan=False)
    question_type: NonEmptyString
    question: NonEmptyString
    reference_answer: NonEmptyString

    @model_validator(mode="after")
    def require_ordered_clip(self) -> EgoTempoQuestion:
        if self.clip_end_seconds <= self.clip_start_seconds:
            raise ValueError("EgoTempo clip end must follow its start")
        return self


class _ReleaseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_date: str = Field(alias="release date")
    version: str


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    clip_id: str
    question_type: str
    question: str
    answer: str


class _RawDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    info: _ReleaseInfo
    annotations: list[_RawQuestion] = Field(min_length=1)


def load_egotempo(annotation_path: Path) -> tuple[EgoTempoQuestion, ...]:
    """Load the released EgoTempo JSON without coupling to its notebook runtime."""
    raw = _RawDataset.model_validate_json(annotation_path.read_bytes())
    questions = tuple(_question(item) for item in raw.annotations)
    question_ids = tuple(question.question_id for question in questions)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("EgoTempo annotations contain duplicate question IDs")
    return questions


def _question(raw: _RawQuestion) -> EgoTempoQuestion:
    try:
        source_video_id, start_text, end_text = raw.clip_id.rsplit("_", 2)
        start_seconds = float(start_text)
        end_seconds = float(end_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid EgoTempo clip_id: {raw.clip_id}") from error
    if not source_video_id or not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise ValueError(f"invalid EgoTempo clip_id: {raw.clip_id}")
    return EgoTempoQuestion(
        question_id=raw.question_id,
        clip_id=raw.clip_id,
        source_video_id=source_video_id,
        clip_start_seconds=start_seconds,
        clip_end_seconds=end_seconds,
        question_type=raw.question_type,
        question=raw.question,
        reference_answer=raw.answer,
    )
