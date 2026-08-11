"""Thin adapter for the official M3-Bench robot and web annotations."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mindbridge.contracts import ContractModel, Identifier, NonEmptyString

M3_BENCH_ADAPTER_VERSION = "m3_bench_official_v1"


class M3BenchQuestion(ContractModel):
    """One open-ended question and the official reference answer."""

    question_id: Identifier
    question: NonEmptyString
    reference_answer: NonEmptyString
    question_types: tuple[NonEmptyString, ...] = Field(min_length=1)
    timestamp_seconds: int | None = Field(default=None, ge=0)
    before_clip_index: int | None = Field(default=None, ge=0)


class M3BenchVideo(ContractModel):
    """One long-video source and all questions evaluated against its memory."""

    video_id: Identifier
    video_path: NonEmptyString
    video_url: NonEmptyString | None = None
    questions: tuple[M3BenchQuestion, ...] = Field(min_length=1)


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    answer: str
    question_id: str
    type: list[str] = Field(min_length=1)
    timestamp: str | None = None
    before_clip: int | None = Field(default=None, ge=0)


class _RawVideo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_path: str
    video_url: str | None = None
    qa_list: list[_RawQuestion] = Field(min_length=1)


def load_m3_bench(annotation_path: Path) -> tuple[M3BenchVideo, ...]:
    """Load `robot.json` or `web.json` while preserving official source paths."""
    records = TypeAdapter(dict[str, _RawVideo]).validate_json(annotation_path.read_bytes())
    videos = tuple(_video(video_id, record) for video_id, record in sorted(records.items()))
    if not videos:
        raise ValueError("M3-Bench annotations must not be empty")
    question_ids = [question.question_id for video in videos for question in video.questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("M3-Bench annotations contain duplicate question IDs")
    return videos


def _video(video_id: str, record: _RawVideo) -> M3BenchVideo:
    if PurePosixPath(record.video_path).stem != video_id:
        raise ValueError(f"M3-Bench video path does not match video ID {video_id}")
    return M3BenchVideo(
        video_id=video_id,
        video_path=record.video_path,
        video_url=record.video_url,
        questions=tuple(
            M3BenchQuestion(
                question_id=question.question_id,
                question=question.question,
                reference_answer=question.answer,
                question_types=tuple(question.type),
                timestamp_seconds=_timestamp_seconds(question.timestamp),
                before_clip_index=question.before_clip,
            )
            for question in record.qa_list
        ),
    )


def _timestamp_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid M3-Bench timestamp: {value}")
    hours, minutes, seconds = (0, *map(int, parts)) if len(parts) == 2 else map(int, parts)
    if seconds >= 60 or (len(parts) == 3 and minutes >= 60):
        raise ValueError(f"invalid M3-Bench timestamp: {value}")
    return hours * 3_600 + minutes * 60 + seconds
