"""Provider-neutral hierarchical summary consolidation pipeline."""

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

from mindbridge.application.capabilities import GenerateRequest, Generator, ModelInput, TextPart
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.application.summary_consolidation import (
    SummaryCandidate,
    SummaryConsolidation,
    SummaryProposal,
    SummaryScope,
)
from mindbridge.core import DomainInvariantError, MemoryId, ModelOutputError
from mindbridge.prompts import CONSOLIDATE_SUMMARIES_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_SummaryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
_MemoryIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]
_SummaryScope = Literal["session", "day", "person", "place", "topic"]


class _SummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_memory_ids: Annotated[tuple[_MemoryIdentifier, ...], Field(min_length=2, max_length=32)]
    scope: _SummaryScope
    summary: _SummaryText
    salience: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def require_unique_sources(self) -> _SummaryOutput:
        if len(set(self.source_memory_ids)) != len(self.source_memory_ids):
            raise ValueError("summary source memory IDs must be unique")
        return self


class _ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summaries: Annotated[tuple[_SummaryOutput, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def require_disjoint_summaries(self) -> _ConsolidationOutput:
        memory_ids = [
            memory_id for summary in self.summaries for memory_id in summary.source_memory_ids
        ]
        if len(memory_ids) > 64:
            raise ValueError("summary output exceeds the candidate memory limit")
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("one memory cannot appear in multiple summaries")
        return self


class SummaryPipeline:
    """Turn a Generator into evidence-first hierarchy verification."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 8_192) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.summaries")
    async def propose_summaries(
        self,
        candidates: tuple[SummaryCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> SummaryConsolidation:
        _require_candidate_evidence(candidates, evidence)
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=CONSOLIDATE_SUMMARIES_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(
                            f"<candidate_context>\n{_context(candidates, evidence)}\n"
                            "</candidate_context>"
                        ),
                        *evidence_parts(evidence),
                        TextPart(
                            "<final_task>Propose supported hierarchy summaries now. Omit any "
                            "summary with fewer than two unique source_memory_ids.</final_task>"
                        ),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            lambda content: _parse_output(content, candidates),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.prompt.version": CONSOLIDATE_SUMMARIES_PROMPT.version,
                "mindbridge.memory.count": len(candidates),
                "mindbridge.evidence.count": len(evidence),
            }
        )
        return SummaryConsolidation(
            summaries=tuple(
                SummaryProposal(
                    source_memory_ids=tuple(
                        MemoryId(value) for value in proposal.source_memory_ids
                    ),
                    scope=SummaryScope(proposal.scope),
                    summary=proposal.summary,
                    salience=proposal.salience,
                )
                for proposal in output.summaries
            ),
            model_reference=result.model_reference,
            prompt_version=CONSOLIDATE_SUMMARIES_PROMPT.version,
        )


def _context(
    candidates: tuple[SummaryCandidate, ...],
    evidence: tuple[ResolvedEvidence, ...],
) -> str:
    return json.dumps(
        {
            "memories": [
                {
                    "memory_id": candidate.memory.memory_id,
                    "memory_type": candidate.memory.memory_type.value,
                    "summary": candidate.memory.summary,
                    "verification_status": candidate.memory.verification_status.value,
                    "occurred_at": candidate.memory.occurred_at.isoformat(),
                    "ended_at": candidate.memory.ended_at.isoformat(),
                    "evidence_ids": candidate.memory.evidence_ids,
                    "entity_ids": candidate.entity_ids,
                }
                for candidate in candidates
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


def _parse_output(
    content: str,
    candidates: tuple[SummaryCandidate, ...],
) -> _ConsolidationOutput:
    if not content.strip():
        raise ModelOutputError("summary pipeline returned empty content")
    try:
        output = _ConsolidationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("summary pipeline returned invalid structured output") from error
    candidate_ids = {str(candidate.memory.memory_id) for candidate in candidates}
    if any(
        memory_id not in candidate_ids
        for proposal in output.summaries
        for memory_id in proposal.source_memory_ids
    ):
        raise ModelOutputError("summary pipeline referenced an unknown memory")
    return output


def _require_candidate_evidence(
    candidates: tuple[SummaryCandidate, ...],
    evidence: tuple[ResolvedEvidence, ...],
) -> None:
    memories = tuple(candidate.memory for candidate in candidates)
    if not 2 <= len(memories) <= 64 or len({memory.memory_id for memory in memories}) != len(
        memories
    ):
        raise DomainInvariantError("summary consolidation requires 2 to 64 unique memories")
    tenant_ids = {memory.tenant_id for memory in memories} | {
        item.evidence_span.tenant_id for item in evidence
    }
    if len(tenant_ids) != 1:
        raise DomainInvariantError("summary candidates and evidence must belong to one tenant")
    expected = {evidence_id for memory in memories for evidence_id in memory.evidence_ids}
    actual = {item.evidence_span.evidence_id for item in evidence}
    if expected != actual or len(actual) != len(evidence):
        raise DomainInvariantError(
            "summary consolidation requires each exact candidate evidence span"
        )
