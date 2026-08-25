"""Official Video-MME adapter and production-path runner."""

from __future__ import annotations

import asyncio
import re
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mindbridge.benchmarks.prompts import VIDEO_MME_QUERY_PROMPT
from mindbridge.benchmarks.runtime import (
    PreparedVideo,
    answer_failure_trace_id,
    benchmark_tenant_id,
    ingest_prepared_video,
    prepared_video_end,
    settle_answers,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    NonEmptyString,
    RecallFilters,
    RecallQuery,
    RecallRequest,
)
from mindbridge.sdk import MindBridge, MindBridgeError

VIDEO_MME_ADAPTER_VERSION = "video_mme_official_v1"
VideoMMEDuration = Literal["short", "medium", "long"]
VideoMMEOption = Literal["A", "B", "C", "D"]
_OPTION_LABELS = tuple("ABCD")
_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
    "The correct answer",
    "The correct option",
)


class VideoMMEQuestion(ContractModel):
    """One official multiple-choice question and its offline label."""

    question_id: Identifier
    task_type: NonEmptyString
    question: NonEmptyString
    options: tuple[NonEmptyString, ...] = Field(min_length=4, max_length=4)
    answer: VideoMMEOption

    @model_validator(mode="after")
    def require_official_option_labels(self) -> VideoMMEQuestion:
        if any(
            re.match(rf"^{label}\.\s+\S", option) is None
            for label, option in zip(_OPTION_LABELS, self.options, strict=True)
        ):
            raise ValueError("Video-MME options must be labelled A. through D. in order")
        return self


class VideoMMEVideo(ContractModel):
    """The three questions and source identity for one official video."""

    video_id: Identifier
    duration: VideoMMEDuration
    domain: NonEmptyString
    sub_category: NonEmptyString
    source_url: NonEmptyString
    source_video_id: Identifier
    questions: tuple[VideoMMEQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_questions(self) -> VideoMMEVideo:
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Video-MME question IDs must be unique per video")
        return self


class VideoMMEQuestionResult(ContractModel):
    """One official evaluator row plus MindBridge diagnostics."""

    question_id: Identifier
    task_type: NonEmptyString
    question: NonEmptyString
    options: tuple[NonEmptyString, ...] = Field(min_length=4, max_length=4)
    answer: VideoMMEOption
    response: str
    mindbridge_model_answer: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    mindbridge_error_code: NonEmptyString | None = None
    # Every question over one video carries the same count, unlike the per-cutoff counts in the
    # runners that ingest and answer in interleaved cohorts: this runner ingests the whole video
    # before answering anything, so any failed segment is missing from every answer here.
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


class VideoMMEVideoResult(ContractModel):
    """Official nested result object for one Video-MME video."""

    video_id: Identifier
    duration: VideoMMEDuration
    domain: NonEmptyString
    sub_category: NonEmptyString
    questions: tuple[VideoMMEQuestionResult, ...] = Field(min_length=1)


class VideoMMEMetrics(ContractModel):
    """Both the released evaluator's denominator and the full-question-set floor.

    `accuracy` reproduces the official parser, which drops rows whose response does not
    contain an option letter. Abstentions and API failures land in exactly that bucket, so
    `strict_accuracy` and `error_count` are reported alongside it: a run may not quote the
    official number without the count of questions it never answered.

    `by_duration` carries the short/medium/long cells the leaderboard reports separately. The
    overall number on this benchmark is saturated, so the long cell is the one a memory system
    is actually judged on.
    """

    question_count: int = Field(gt=0)
    answered_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    strict_accuracy: float = Field(ge=0.0, le=1.0)
    by_duration: dict[VideoMMEDuration, VideoMMEMetrics] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_consistent_counts(self) -> VideoMMEMetrics:
        if self.correct_count > self.answered_count or self.answered_count > self.question_count:
            raise ValueError("Video-MME metric counts are inconsistent")
        if self.error_count > self.question_count - self.answered_count:
            raise ValueError("Video-MME failed questions must not carry a parsed answer")
        if any(cell.by_duration for cell in self.by_duration.values()):
            raise ValueError("Video-MME duration cells must not nest further")
        if sum(cell.question_count for cell in self.by_duration.values()) not in {
            0,
            self.question_count,
        }:
            raise ValueError("Video-MME duration cells must cover every scored question")
        return self


class _RawQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    duration: VideoMMEDuration
    domain: str
    sub_category: str
    url: str
    source_video_id: str = Field(alias="videoID")
    question_id: str
    task_type: str
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer: VideoMMEOption


def load_video_mme(annotation_path: Path) -> tuple[VideoMMEVideo, ...]:
    """Load the official Hugging Face Parquet release without materializing media."""
    try:
        parquet = cast(Any, import_module("pyarrow.parquet"))
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("pyarrow"):
            raise
        raise RuntimeError(
            "Video-MME Parquet support requires `uv sync --group benchmarks`"
        ) from error
    rows = TypeAdapter(list[_RawQuestion]).validate_python(
        parquet.read_table(annotation_path).to_pylist()
    )
    if not rows:
        raise ValueError("Video-MME annotations must not be empty")

    grouped: dict[str, list[_RawQuestion]] = {}
    for row in rows:
        grouped.setdefault(row.video_id, []).append(row)
    videos = tuple(_video(group) for group in grouped.values())
    question_ids = tuple(question.question_id for video in videos for question in video.questions)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Video-MME annotations contain duplicate question IDs")
    return videos


async def run_video_mme_video(
    memory: MindBridge,
    annotation: VideoMMEVideo,
    prepared: PreparedVideo,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_video_mme",
    device_id: str = "video_mme_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> VideoMMEVideoResult:
    """Ingest one source video and answer its official questions."""
    if annotation.video_id != prepared.video_id:
        raise ValueError("Video-MME annotation and prepared video IDs must match")
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    tenant_id = benchmark_tenant_id(tenant_prefix, annotation.video_id, run_id)
    ingest_failures = await ingest_prepared_video(
        memory,
        tenant_id,
        device_id,
        prepared,
        adapter_version=VIDEO_MME_ADAPTER_VERSION,
        request_concurrency=request_concurrency,
        poll_interval_seconds=poll_interval_seconds,
        processing_timeout_seconds=processing_timeout_seconds,
    )

    semaphore = asyncio.Semaphore(request_concurrency)
    answers: list[VideoMMEQuestionResult] = []
    cutoff = prepared_video_end(prepared)
    for offset in range(0, len(annotation.questions), request_concurrency):
        cohort = annotation.questions[offset : offset + request_concurrency]
        answered = await asyncio.gather(
            *(
                _answer_question(
                    memory,
                    tenant_id,
                    question,
                    cutoff,
                    recall_limit,
                    semaphore,
                    ingest_failures,
                )
                for question in cohort
            ),
            return_exceptions=True,
        )
        answers.extend(
            settle_answers(
                cohort, answered, partial(_failed_result, ingest_failures=ingest_failures)
            )
        )
    return VideoMMEVideoResult(
        video_id=annotation.video_id,
        duration=annotation.duration,
        domain=annotation.domain,
        sub_category=annotation.sub_category,
        questions=tuple(answers),
    )


def evaluate_video_mme(results: tuple[VideoMMEVideoResult, ...]) -> VideoMMEMetrics:
    """Compute the official accuracy, its unanswered counts, and the duration cells."""
    if not results:
        raise ValueError("Video-MME results must not be empty")
    grouped: dict[VideoMMEDuration, list[VideoMMEVideoResult]] = {}
    for video in results:
        grouped.setdefault(video.duration, []).append(video)
    # Built through the constructor, not `model_copy(update=...)`: the latter skips validators,
    # which would leave `require_consistent_counts` unenforced on the only object that ever
    # carries duration cells, and a manifest could then be written that cannot be read back.
    return _duration_metrics(
        results,
        by_duration={
            duration: _duration_metrics(tuple(group)) for duration, group in sorted(grouped.items())
        },
    )


def _duration_metrics(
    results: tuple[VideoMMEVideoResult, ...],
    *,
    by_duration: dict[VideoMMEDuration, VideoMMEMetrics] | None = None,
) -> VideoMMEMetrics:
    questions = tuple(question for video in results for question in video.questions)
    if not questions:
        raise ValueError("Video-MME results must not be empty")
    answered = tuple(question for question in questions if question.response in _OPTION_LABELS)
    correct_count = sum(question.response == question.answer for question in answered)
    return VideoMMEMetrics(
        question_count=len(questions),
        answered_count=len(answered),
        correct_count=correct_count,
        error_count=sum(question.mindbridge_error_code is not None for question in questions),
        accuracy=correct_count / len(answered) if answered else 0.0,
        strict_accuracy=correct_count / len(questions),
        by_duration=by_duration or {},
    )


def parse_video_mme_option(response: str | None) -> VideoMMEOption | None:
    """Normalize a response using the released evaluator's first-letter rule."""
    if response is None:
        return None
    normalized = response.strip()
    for prefix in _ANSWER_PREFIXES:
        normalized = normalized.replace(prefix, "")
    match = re.search(r"[ABCD]", normalized)
    return cast(VideoMMEOption, match.group()) if match is not None else None


def _video(rows: list[_RawQuestion]) -> VideoMMEVideo:
    first = rows[0]
    metadata = (
        first.duration,
        first.domain,
        first.sub_category,
        first.url,
        first.source_video_id,
    )
    if any(
        (row.duration, row.domain, row.sub_category, row.url, row.source_video_id) != metadata
        for row in rows
    ):
        raise ValueError(f"Video-MME video {first.video_id} has inconsistent metadata")
    return VideoMMEVideo(
        video_id=first.video_id,
        duration=first.duration,
        domain=first.domain,
        sub_category=first.sub_category,
        source_url=first.url,
        source_video_id=first.source_video_id,
        questions=tuple(
            VideoMMEQuestion(
                question_id=row.question_id,
                task_type=row.task_type,
                question=row.question,
                options=tuple(row.options),
                answer=row.answer,
            )
            for row in rows
        ),
    )


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: VideoMMEQuestion,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    ingest_failures: int,
) -> VideoMMEQuestionResult:
    query = VIDEO_MME_QUERY_PROMPT.text.format(
        question=question.question,
        options="\n".join(question.options),
    )
    try:
        async with semaphore:
            result = await memory.recall(
                RecallRequest(
                    tenant_id=tenant_id,
                    query=RecallQuery(text=query),
                    filters=RecallFilters(occurred_before=cutoff),
                    limit=recall_limit,
                )
            )
    except MindBridgeError as error:
        if error.code not in {"model_output_invalid", "model_request_failed"}:
            raise
        return _question_result(
            question,
            model_answer="",
            confidence=0.0,
            memory_ids=(),
            evidence_ids=(),
            trace_id=error.trace_id or f"trace_model_error_{question.question_id}",
            error_code=error.code,
            ingest_failures=ingest_failures,
        )
    return _question_result(
        question,
        model_answer=result.answer or "",
        confidence=result.confidence,
        memory_ids=tuple(item.memory_id for item in result.memories),
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        trace_id=result.trace_id,
        ingest_failures=ingest_failures,
    )


def _failed_result(
    question: VideoMMEQuestion,
    error_code: str,
    *,
    ingest_failures: int,
) -> VideoMMEQuestionResult:
    """One row for a question whose recall raised, so its cohort still answers all of them."""
    return _question_result(
        question,
        model_answer="",
        confidence=0.0,
        memory_ids=(),
        evidence_ids=(),
        trace_id=answer_failure_trace_id(question.question_id),
        error_code=error_code,
        ingest_failures=ingest_failures,
    )


def _question_result(
    question: VideoMMEQuestion,
    *,
    model_answer: str,
    confidence: float,
    memory_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    trace_id: str,
    ingest_failures: int,
    error_code: str | None = None,
) -> VideoMMEQuestionResult:
    option = parse_video_mme_option(model_answer)
    return VideoMMEQuestionResult(
        question_id=question.question_id,
        task_type=question.task_type,
        question=question.question,
        options=question.options,
        answer=question.answer,
        response=option or "",
        mindbridge_model_answer=model_answer,
        mindbridge_confidence=confidence,
        mindbridge_memory_ids=memory_ids,
        mindbridge_evidence_ids=evidence_ids,
        mindbridge_trace_id=trace_id,
        mindbridge_error_code=error_code,
        mindbridge_ingest_failure_count=ingest_failures,
    )
