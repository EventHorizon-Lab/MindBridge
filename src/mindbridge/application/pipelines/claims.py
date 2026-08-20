"""Provider-neutral semantic claim consolidation pipeline."""

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
from mindbridge.application.claim_consolidation import ClaimCandidate
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import (
    generate_json,
    output_schema,
    unwrap_json_code_fence,
)
from mindbridge.application.semantic_claims import (
    ClaimConsolidation,
    ClaimRelationshipProposal,
    SemanticClaimProposal,
)
from mindbridge.core import ClaimId, DomainInvariantError, ModelOutputError, RelationType
from mindbridge.prompts import CONSOLIDATE_CLAIMS_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_Statement = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
_ClaimIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]


class _SemanticClaimOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_claim_ids: Annotated[tuple[_ClaimIdentifier, ...], Field(min_length=2, max_length=32)]
    statement: _Statement
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def require_unique_sources(self) -> _SemanticClaimOutput:
        if len(set(self.source_claim_ids)) != len(self.source_claim_ids):
            raise ValueError("semantic claim source IDs must be unique")
        return self


class _RelationshipOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_claim_id: _ClaimIdentifier
    relation_type: Literal["contradicts", "supersedes"]
    target_claim_id: _ClaimIdentifier

    @model_validator(mode="after")
    def reject_self_relation(self) -> _RelationshipOutput:
        if self.source_claim_id == self.target_claim_id:
            raise ValueError("claim relationship cannot point to itself")
        return self


class _ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_claims: Annotated[tuple[_SemanticClaimOutput, ...], Field(max_length=32)]
    relationships: Annotated[tuple[_RelationshipOutput, ...], Field(max_length=64)]

    @model_validator(mode="after")
    def require_unambiguous_claim_usage(self) -> _ConsolidationOutput:
        support_ids = [
            claim_id for proposal in self.semantic_claims for claim_id in proposal.source_claim_ids
        ]
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("one claim cannot support multiple semantic claims")
        support_id_set = set(support_ids)
        pairs = [
            frozenset((relationship.source_claim_id, relationship.target_claim_id))
            for relationship in self.relationships
        ]
        if len(set(pairs)) != len(pairs):
            raise ValueError("a claim pair can have only one semantic relationship")
        if any(
            claim_id in support_id_set
            for relationship in self.relationships
            for claim_id in (relationship.source_claim_id, relationship.target_claim_id)
        ):
            raise ValueError("supporting claims cannot also have direct decisions")
        return self


_CLAIM_CONSOLIDATION_SCHEMA = output_schema("claim_consolidation", _ConsolidationOutput)


class ClaimPipeline:
    """Turn a Generator into evidence-first semantic claim verification."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 8_192) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.claims")
    async def propose_claims(
        self,
        candidates: tuple[ClaimCandidate, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> ClaimConsolidation:
        _require_candidate_evidence(candidates, evidence)
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=CONSOLIDATE_CLAIMS_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(
                            f"<candidate_context>\n{_context(candidates, evidence)}\n"
                            "</candidate_context>"
                        ),
                        *evidence_parts(evidence),
                        TextPart(
                            "<final_task>Propose supported claim decisions now. Omit any "
                            "semantic_claim with fewer than two unique source_claim_ids."
                            "</final_task>"
                        ),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
                output_schema=_CLAIM_CONSOLIDATION_SCHEMA,
            ),
            lambda content: _parse_output(content, candidates),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": CONSOLIDATE_CLAIMS_PROMPT.version,
                "mindbridge.claim.count": len(candidates),
                "mindbridge.evidence.count": len(evidence),
            }
        )
        return ClaimConsolidation(
            semantic_claims=tuple(
                SemanticClaimProposal(
                    source_claim_ids=tuple(ClaimId(value) for value in proposal.source_claim_ids),
                    statement=proposal.statement,
                    confidence=proposal.confidence,
                )
                for proposal in output.semantic_claims
            ),
            relationships=tuple(
                ClaimRelationshipProposal(
                    source_claim_id=ClaimId(relationship.source_claim_id),
                    relation_type=RelationType(relationship.relation_type),
                    target_claim_id=ClaimId(relationship.target_claim_id),
                )
                for relationship in output.relationships
            ),
            model_reference=result.model_reference,
            prompt_version=CONSOLIDATE_CLAIMS_PROMPT.version,
        )


def _context(
    candidates: tuple[ClaimCandidate, ...],
    evidence: tuple[ResolvedEvidence, ...],
) -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": candidate.claim.claim_id,
                    "claim_type": candidate.claim.claim_type.value,
                    "statement": candidate.claim.statement,
                    "confidence": candidate.claim.confidence,
                    "valid_from": candidate.claim.valid_from.isoformat(),
                    "valid_to": (
                        candidate.claim.valid_to.isoformat()
                        if candidate.claim.valid_to is not None
                        else None
                    ),
                    "evidence_ids": candidate.claim.evidence_ids,
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
    candidates: tuple[ClaimCandidate, ...],
) -> _ConsolidationOutput:
    if not content.strip():
        raise ModelOutputError("claim pipeline returned empty content")
    try:
        output = _ConsolidationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("claim pipeline returned invalid structured output") from error
    _require_candidate_output(candidates, output)
    return output


def _require_candidate_output(
    candidates: tuple[ClaimCandidate, ...],
    output: _ConsolidationOutput,
) -> None:
    candidate_by_id = {str(candidate.claim.claim_id): candidate.claim for candidate in candidates}
    referenced_ids = {
        claim_id for proposal in output.semantic_claims for claim_id in proposal.source_claim_ids
    } | {
        claim_id
        for relationship in output.relationships
        for claim_id in (relationship.source_claim_id, relationship.target_claim_id)
    }
    if not referenced_ids <= set(candidate_by_id):
        raise ModelOutputError("claim pipeline referenced an unknown claim")
    if any(
        len({candidate_by_id[claim_id].claim_type for claim_id in proposal.source_claim_ids}) != 1
        for proposal in output.semantic_claims
    ):
        raise ModelOutputError("semantic claim sources must share one claim type")
    if any(
        (
            candidate_by_id[relationship.source_claim_id].valid_from,
            relationship.source_claim_id,
        )
        <= (
            candidate_by_id[relationship.target_claim_id].valid_from,
            relationship.target_claim_id,
        )
        for relationship in output.relationships
    ):
        raise ModelOutputError("claim relationship source must be the later claim")


def _require_candidate_evidence(
    candidates: tuple[ClaimCandidate, ...],
    evidence: tuple[ResolvedEvidence, ...],
) -> None:
    claims = tuple(candidate.claim for candidate in candidates)
    if not 2 <= len(claims) <= 64 or len({claim.claim_id for claim in claims}) != len(claims):
        raise DomainInvariantError("claim consolidation requires 2 to 64 unique claims")
    tenant_ids = {claim.tenant_id for claim in claims} | {
        item.evidence_span.tenant_id for item in evidence
    }
    if len(tenant_ids) != 1:
        raise DomainInvariantError("claim candidates and evidence must belong to one tenant")
    expected = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    actual = {item.evidence_span.evidence_id for item in evidence}
    if expected != actual or len(actual) != len(evidence):
        raise DomainInvariantError(
            "claim consolidation requires each exact candidate evidence span"
        )
