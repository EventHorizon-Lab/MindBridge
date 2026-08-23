"""Grounded answer and exact-occurrence pipelines."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypeVar, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
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
from mindbridge.application.pipelines.evidence import (
    DEFAULT_MAX_EVIDENCE_MEDIA_PARTS,
    evidence_parts,
)
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.application.ports import GeneratedAnswer, ResolvedQueryMedia
from mindbridge.contracts import RecallRequest
from mindbridge.core import MemoryId, MemoryRecord, ModelOutputError
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, SELECT_OCCURRENCES_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_MODEL_OUTPUT_CONFIG = ConfigDict(extra="ignore", frozen=True)
"""Everything the prompt asked for, and nothing about what it did not.

Same boundary and same reason as the perception pipeline's: this is a model-output boundary, not
a trust boundary, and forbidding extras here threw away a finished answer over a key nothing
reads. On the read path that is measurably the more expensive half -- 14 of 55 M3-Bench robot
predictions (25.5%) were lost to `model_output_invalid`, more than every wrong answer that
benchmark made combined, clustered on the widest recall payloads.

A renamed field still fails, because the field it replaced is then missing.
"""

_TemporalOrder = Literal["relevance", "newest", "oldest"]
_DEFAULT_TEMPORAL_ORDER: _TemporalOrder = "relevance"
# What GeneratedAnswer accepts, and what the prompt asks for: at most two follow-up queries.
_MAX_RETRIEVAL_QUERIES = 2

# Query media is attached as the caller uploaded it. Nothing is substituted for it: a derived clip
# is cut per evidence span, and an object asked *with* is not a span of anything, so what goes on
# the wire is a full-resolution source -- ~12.3k prompt tokens against ~1.65k for a clip, and four
# of them in one call is the measured 60 s gateway timeout. So it is charged what it costs against
# the one ceiling the whole request shares, rather than a second ceiling of its own: at most three
# query objects are attached, and each one attached is eight evidence clips that are not.
QUERY_MEDIA_PART_COST = 8

_NonEmptyAnswer = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_RetrievalQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_048),
]
_Item = TypeVar("_Item")


def _valid_items(adapter: TypeAdapter[_Item], value: object) -> tuple[_Item, ...] | object:
    """Keep the list entries that validate, in the order first mentioned, dropping repeats.

    An entry the model got wrong is one entry, so it costs one entry. Something that is not a
    list at all is handed on untouched, because the field's own validation is what should judge
    that.
    """
    if not isinstance(value, list):
        return value
    kept: list[_Item] = []
    for item in value:
        try:
            kept.append(adapter.validate_python(item))
        except ValidationError:
            continue
    return tuple(dict.fromkeys(kept))


_QUERY_ADAPTER: TypeAdapter[str] = TypeAdapter(_RetrievalQuery)
_MEMORY_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(_NonEmptyAnswer)


def _asked_queries(value: object) -> object:
    """Take the first two distinct follow-up queries and drop the rest.

    A third query, or a repeat of the first, says nothing the first two do not -- and losing the
    answer they came attached to costs a whole recall, including the queries themselves, which
    are what a second retrieval round would have used.
    """
    kept = _valid_items(_QUERY_ADAPTER, value)
    return kept[:_MAX_RETRIEVAL_QUERIES] if isinstance(kept, tuple) else kept


def _known_temporal_order(value: object) -> object:
    """Fall back to the documented default rather than losing the answer over its ordering hint."""
    return value if value in get_args(_TemporalOrder) else _DEFAULT_TEMPORAL_ORDER


class _AnswerOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    answer: _NonEmptyAnswer | None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    retrieval_queries: Annotated[tuple[_RetrievalQuery, ...], BeforeValidator(_asked_queries)] = ()
    temporal_order: Annotated[_TemporalOrder, BeforeValidator(_known_temporal_order)] = (
        _DEFAULT_TEMPORAL_ORDER
    )

    @model_validator(mode="after")
    def keep_abstention_at_zero_confidence(self) -> _AnswerOutput:
        """A null answer is an abstention; a confidence contradicting it is the part that is wrong.

        Zeroing it keeps the abstention exactly as the model gave it, which is the reason this is
        a repair and not a change of answering behaviour. `confidence` is otherwise never
        rewritten: with an answer present it is a required number carrying meaning of its own, so
        an out-of-range one is rejected rather than clamped into a calibration nobody reported.
        """
        if self.answer is None and self.confidence != 0.0:
            return self.model_copy(update={"confidence": 0.0})
        return self


class _OccurrenceOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    memory_ids: Annotated[
        tuple[_NonEmptyAnswer, ...],
        BeforeValidator(lambda value: _valid_items(_MEMORY_ID_ADAPTER, value)),
    ]


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
    attached_query_media = query_media[: DEFAULT_MAX_EVIDENCE_MEDIA_PARTS // QUERY_MEDIA_PART_COST]
    for item in attached_query_media:
        parts.extend(
            (
                TextPart(f"Query media_object_id={item.media_object.media_object_id} follows."),
                MediaPart(
                    kind=item.media_object.kind,
                    url=item.media_url,
                    source_uri=item.media_object.uri,
                ),
            )
        )
    parts.extend(
        evidence_parts(
            evidence,
            # Excluding the query object's *id* would drop the derived clips of every span cut
            # from it, which on a media query is exactly the evidence being asked about. What
            # must not be sent twice is one set of bytes.
            excluded_media_urls={item.media_url for item in attached_query_media},
            max_media_parts=(
                DEFAULT_MAX_EVIDENCE_MEDIA_PARTS - QUERY_MEDIA_PART_COST * len(attached_query_media)
            ),
        )
    )
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
    # An ID that was never a candidate cannot name an occurrence, so it is dropped rather than
    # taking the batch's real selections with it. Enumeration verifies memories in batches, and
    # one invented ID used to fail the whole count.
    selected = tuple(memory_id for memory_id in output.memory_ids if memory_id in candidate_ids)
    if len(selected) != len(output.memory_ids):
        set_current_span_attributes(
            {
                "mindbridge.occurrences.dropped_id_count": len(output.memory_ids) - len(selected),
            }
        )
    return output.model_copy(update={"memory_ids": selected})
