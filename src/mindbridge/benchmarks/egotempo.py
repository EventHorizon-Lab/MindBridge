"""Official EgoTempo adapter and production-path runner."""

from __future__ import annotations

import asyncio
import math
from functools import partial
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from mindbridge.benchmarks.prompts import EGOTEMPO_QUERY_PROMPT
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

EGOTEMPO_ADAPTER_VERSION = "egotempo_official_v1"


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


class EgoTempoQuestionResult(ContractModel):
    """Official judge input fields plus retrieval diagnostics."""

    clip_id: Identifier = Field(serialization_alias="V")
    question: NonEmptyString = Field(serialization_alias="Q")
    formatted_prompt: NonEmptyString = Field(serialization_alias="QA")
    model_answer: str = Field(serialization_alias="A")
    reference_answer: NonEmptyString = Field(serialization_alias="C")
    question_type: NonEmptyString = Field(serialization_alias="M")
    question_id: Identifier
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    mindbridge_error_code: NonEmptyString | None = None
    # Every question over one clip carries the same count, unlike the per-cutoff counts in the
    # runners that ingest and answer in interleaved cohorts: this runner ingests the whole clip
    # before answering anything, so any failed segment is missing from every answer here.
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


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


async def run_egotempo_clip(
    memory: MindBridge,
    questions: tuple[EgoTempoQuestion, ...],
    prepared: PreparedVideo,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_egotempo",
    device_id: str = "egotempo_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[EgoTempoQuestionResult, ...]:
    """Ingest one trimmed clip and answer every official question attached to it."""
    if not questions:
        raise ValueError("EgoTempo questions must not be empty")
    if {question.clip_id for question in questions} != {prepared.video_id}:
        raise ValueError("EgoTempo questions and prepared clip IDs must match")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("EgoTempo question IDs must be unique")
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    if (
        len({(question.clip_start_seconds, question.clip_end_seconds) for question in questions})
        != 1
    ):
        raise ValueError("EgoTempo questions for one clip have inconsistent boundaries")

    tenant_id = benchmark_tenant_id(tenant_prefix, prepared.video_id, run_id)
    ingest_failures = await ingest_prepared_video(
        memory,
        tenant_id,
        device_id,
        prepared,
        adapter_version=EGOTEMPO_ADAPTER_VERSION,
        request_concurrency=request_concurrency,
        poll_interval_seconds=poll_interval_seconds,
        processing_timeout_seconds=processing_timeout_seconds,
    )
    semaphore = asyncio.Semaphore(request_concurrency)
    answers: list[EgoTempoQuestionResult] = []
    cutoff = prepared_video_end(prepared)
    for offset in range(0, len(questions), request_concurrency):
        cohort = questions[offset : offset + request_concurrency]
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
    return tuple(answers)


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


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: EgoTempoQuestion,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    ingest_failures: int,
) -> EgoTempoQuestionResult:
    prompt = EGOTEMPO_QUERY_PROMPT.text.format(question=question.question)
    try:
        async with semaphore:
            result = await memory.recall(
                RecallRequest(
                    tenant_id=tenant_id,
                    query=RecallQuery(text=prompt),
                    filters=RecallFilters(occurred_before=cutoff),
                    limit=recall_limit,
                )
            )
    except MindBridgeError as error:
        # `model_unavailable` is the deployment's 60 s gateway timeout under load (~5.5% of
        # recalls once nine benchmarks share the endpoint). Fatal here discarded every answer
        # the sweep had already produced; recoverable records the code against this one
        # question instead, which is counted as infrastructure and never as a wrong answer.
        if error.code not in {
            "model_output_invalid",
            "model_request_failed",
            "model_unavailable",
        }:
            raise
        return _question_result(
            question,
            prompt,
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
        prompt,
        model_answer=result.answer or "",
        confidence=result.confidence,
        memory_ids=tuple(item.memory_id for item in result.memories),
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        trace_id=result.trace_id,
        ingest_failures=ingest_failures,
    )


def _failed_result(
    question: EgoTempoQuestion,
    error_code: str,
    *,
    ingest_failures: int,
) -> EgoTempoQuestionResult:
    """One row for a question whose recall raised, so its cohort still answers all of them."""
    return _question_result(
        question,
        EGOTEMPO_QUERY_PROMPT.text.format(question=question.question),
        model_answer="",
        confidence=0.0,
        memory_ids=(),
        evidence_ids=(),
        trace_id=answer_failure_trace_id(question.question_id),
        error_code=error_code,
        ingest_failures=ingest_failures,
    )


def _question_result(
    question: EgoTempoQuestion,
    prompt: str,
    *,
    model_answer: str,
    confidence: float,
    memory_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    trace_id: str,
    ingest_failures: int,
    error_code: str | None = None,
) -> EgoTempoQuestionResult:
    return EgoTempoQuestionResult(
        clip_id=question.clip_id,
        question=question.question,
        formatted_prompt=prompt,
        model_answer=model_answer,
        reference_answer=question.reference_answer,
        question_type=question.question_type,
        question_id=question.question_id,
        mindbridge_confidence=confidence,
        mindbridge_memory_ids=memory_ids,
        mindbridge_evidence_ids=evidence_ids,
        mindbridge_trace_id=trace_id,
        mindbridge_error_code=error_code,
        mindbridge_ingest_failure_count=ingest_failures,
    )
