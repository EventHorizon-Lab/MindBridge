"""Grounded answer and exact-occurrence pipelines."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from mindbridge.application.capabilities import (
    GenerateRequest,
    Generator,
    InputPart,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.application.ports import GeneratedAnswer, ResolvedQueryMedia
from mindbridge.contracts import RecallRequest
from mindbridge.core import MemoryId, MemoryRecord, ModelOutputError
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, SELECT_OCCURRENCES_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_NonEmptyAnswer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_RetrievalQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]


class _AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: _NonEmptyAnswer | None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    retrieval_queries: Annotated[tuple[_RetrievalQuery, ...], Field(max_length=2)] = ()
    temporal_order: Literal["relevance", "newest", "oldest"] = "relevance"

    @model_validator(mode="after")
    def require_zero_confidence_for_abstention(self) -> _AnswerOutput:
        if self.answer is None and self.confidence != 0.0:
            raise ValueError("confidence must be zero when answer is null")
        if len(set(self.retrieval_queries)) != len(self.retrieval_queries):
            raise ValueError("retrieval_queries must be unique")
        return self


class _OccurrenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_ids: tuple[_NonEmptyAnswer, ...]

    @model_validator(mode="after")
    def require_unique_memory_ids(self) -> _OccurrenceOutput:
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("memory_ids must be unique")
        return self


class AnswerPipeline:
    """Turn a Generator into MindBridge's evidence-grounded answer policy."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 2_048) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.answer")
    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        parts = _recall_parts(
            request,
            memories,
            evidence,
            query_media=query_media,
            attempted_retrieval_queries=attempted_retrieval_queries,
        )
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=ANSWER_FROM_EVIDENCE_PROMPT.text,
                input=ModelInput(parts),
                max_output_tokens=self._max_output_tokens,
            ),
            _parse_answer,
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": ANSWER_FROM_EVIDENCE_PROMPT.version,
                "mindbridge.memory.count": len(memories),
                "mindbridge.evidence.count": len(evidence),
                "mindbridge.query.media_count": len(query_media),
            }
        )
        return GeneratedAnswer(
            answer=output.answer,
            confidence=output.confidence,
            retrieval_queries=output.retrieval_queries,
            temporal_order=output.temporal_order,
        )


class OccurrencePipeline:
    """Turn a Generator into exhaustive candidate verification."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 2_048) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.occurrences")
    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        candidate_ids = {str(memory.memory_id) for memory in memories}
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=SELECT_OCCURRENCES_PROMPT.text,
                input=ModelInput(
                    _recall_parts(
                        request,
                        memories,
                        evidence,
                        query_media=query_media,
                        attempted_retrieval_queries=(),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            lambda content: _parse_occurrences(content, candidate_ids),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": SELECT_OCCURRENCES_PROMPT.version,
                "mindbridge.memory.count": len(memories),
                "mindbridge.evidence.count": len(evidence),
                "mindbridge.query.media_count": len(query_media),
            }
        )
        selected_ids = tuple(MemoryId(memory_id) for memory_id in output.memory_ids)
        return selected_ids


def _recall_parts(
    request: RecallRequest,
    memories: tuple[MemoryRecord, ...],
    evidence: tuple[ResolvedEvidence, ...],
    *,
    query_media: tuple[ResolvedQueryMedia, ...],
    attempted_retrieval_queries: tuple[str, ...],
) -> tuple[InputPart, ...]:
    parts: list[InputPart] = [
        TextPart(
            "<recall_context>\n"
            f"{_recall_context(request, memories, evidence, attempted_retrieval_queries)}\n"
            "</recall_context>"
        )
    ]
    seen_media_object_ids: set[str] = set()
    for item in query_media:
        media_object_id = item.media_object.media_object_id
        seen_media_object_ids.add(media_object_id)
        parts.extend(
            (
                TextPart(f"Query media_object_id={media_object_id} follows."),
                MediaPart(
                    kind=item.media_object.kind,
                    url=item.media_url,
                    source_uri=item.media_object.uri,
                ),
            )
        )
    parts.extend(evidence_parts(evidence, excluded_media_object_ids=seen_media_object_ids))
    final_input = json.dumps(
        {
            "question": request.query.text,
            "query_media_object_ids": request.query.media_object_ids,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    parts.append(
        TextPart(
            f"<final_task_input>{final_input}</final_task_input>\n"
            "Complete the system task now and return only its required JSON object."
        )
    )
    return tuple(parts)


def _recall_context(
    request: RecallRequest,
    memories: tuple[MemoryRecord, ...],
    evidence: tuple[ResolvedEvidence, ...],
    attempted_retrieval_queries: tuple[str, ...],
) -> str:
    return json.dumps(
        {
            "question": request.query.text,
            "query_media_object_ids": request.query.media_object_ids,
            "attempted_retrieval_queries": attempted_retrieval_queries,
            "candidate_memories": [
                {
                    "memory_id": memory.memory_id,
                    "summary": memory.summary,
                    "verification_status": memory.verification_status.value,
                    "occurred_at": memory.occurred_at.isoformat(),
                    "ended_at": memory.ended_at.isoformat(),
                    "evidence_ids": memory.evidence_ids,
                }
                for memory in memories
            ],
            "evidence_spans": [
                {
                    "evidence_id": item.evidence_span.evidence_id,
                    "media_object_id": item.media_object.media_object_id,
                    "media_kind": item.media_object.kind.value,
                    "start_ms": item.evidence_span.start_ms,
                    "end_ms": item.evidence_span.end_ms,
                }
                for item in evidence
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_answer(content: str) -> _AnswerOutput:
    if not content.strip():
        raise ModelOutputError("answer pipeline returned empty content")
    try:
        return _AnswerOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("answer pipeline returned invalid structured output") from error


def _parse_occurrences(content: str, candidate_ids: set[str]) -> _OccurrenceOutput:
    if not content.strip():
        raise ModelOutputError("occurrence pipeline returned empty content")
    try:
        output = _OccurrenceOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("occurrence pipeline returned invalid structured output") from error
    if not set(output.memory_ids) <= candidate_ids:
        raise ModelOutputError("occurrence pipeline returned an unknown memory ID")
    return output
