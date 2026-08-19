"""Provider-neutral multimodal event perception pipeline."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from mindbridge.application.capabilities import GenerateRequest, Generator, ModelInput, TextPart
from mindbridge.application.perception import (
    MAX_PERCEIVED_CLAIMS_PER_EVENT,
    MAX_PERCEIVED_ENTITIES_PER_EVENT,
    MAX_PERCEPTION_CLAIMS,
    MAX_PERCEPTION_ENTITIES,
    MAX_PERCEPTION_EVENTS,
    EventPerception,
    PerceivedClaim,
    PerceivedEntity,
    PerceivedEvent,
    ResolvedEvidence,
    time_ranges_overlap,
)
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.core import (
    ClaimType,
    DomainInvariantError,
    EntityType,
    EvidenceId,
    ModelOutputError,
    Observation,
)
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_Description = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
_EvidenceIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class _EntityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: EntityType
    canonical_name: _Description
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: Annotated[tuple[_EvidenceIdentifier, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_evidence(self) -> _EntityOutput:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("entity evidence_ids must be unique")
        return self


class _ClaimOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_type: ClaimType
    statement: _Description
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: Annotated[tuple[_EvidenceIdentifier, ...], Field(min_length=1)]
    valid_from_ms: Annotated[int, Field(ge=0)]
    valid_to_ms: Annotated[int, Field(ge=0)] | None
    entity_indices: Annotated[tuple[Annotated[int, Field(ge=0)], ...], Field(max_length=32)] = ()

    @model_validator(mode="after")
    def require_valid_references(self) -> _ClaimOutput:
        if self.valid_to_ms is not None and self.valid_to_ms < self.valid_from_ms:
            raise ValueError("valid_to_ms must not precede valid_from_ms")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("claim evidence_ids must be unique")
        if len(set(self.entity_indices)) != len(self.entity_indices):
            raise ValueError("claim entity_indices must be unique")
        return self


class _EventOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=0)]
    description: _Description
    salience: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: Annotated[tuple[_EvidenceIdentifier, ...], Field(min_length=1)]
    entities: Annotated[
        tuple[_EntityOutput, ...], Field(max_length=MAX_PERCEIVED_ENTITIES_PER_EVENT)
    ] = ()
    claims: Annotated[
        tuple[_ClaimOutput, ...], Field(max_length=MAX_PERCEIVED_CLAIMS_PER_EVENT)
    ] = ()

    @model_validator(mode="after")
    def require_ordered_range(self) -> _EventOutput:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not precede start_ms")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        return self


class _PerceptionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: Annotated[tuple[_EventOutput, ...], Field(max_length=MAX_PERCEPTION_EVENTS)]

    @model_validator(mode="after")
    def require_bounded_details(self) -> _PerceptionOutput:
        if sum(len(event.entities) for event in self.events) > MAX_PERCEPTION_ENTITIES:
            raise ValueError("total perception entity count exceeds the processing limit")
        if sum(len(event.claims) for event in self.events) > MAX_PERCEPTION_CLAIMS:
            raise ValueError("total perception claim count exceeds the processing limit")
        return self


class PerceptionPipeline:
    """Turn a Generator into bounded, evidence-linked semantic intervals."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 8_192) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.perception")
    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception:
        _require_observation_evidence(observation, evidence)
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=PERCEIVE_EVENTS_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(
                            f"<observation_context>\n{_context(observation, evidence)}\n"
                            "</observation_context>"
                        ),
                        *evidence_parts(evidence),
                        TextPart(
                            "<final_task>Produce the grounded events JSON for the observation "
                            "and media above.</final_task>"
                        ),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            lambda content: _parse_output(content, observation, evidence),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": PERCEIVE_EVENTS_PROMPT.version,
                "mindbridge.evidence.count": len(evidence),
            }
        )
        return EventPerception(
            events=tuple(
                PerceivedEvent(
                    start_ms=event.start_ms,
                    end_ms=event.end_ms,
                    description=event.description,
                    salience=event.salience,
                    evidence_ids=tuple(EvidenceId(value) for value in event.evidence_ids),
                    entities=tuple(
                        PerceivedEntity(
                            entity_type=entity.entity_type,
                            canonical_name=entity.canonical_name,
                            confidence=entity.confidence,
                            evidence_ids=tuple(EvidenceId(value) for value in entity.evidence_ids),
                        )
                        for entity in event.entities
                    ),
                    claims=tuple(
                        PerceivedClaim(
                            claim_type=claim.claim_type,
                            statement=claim.statement,
                            confidence=claim.confidence,
                            evidence_ids=tuple(EvidenceId(value) for value in claim.evidence_ids),
                            valid_from_ms=claim.valid_from_ms,
                            valid_to_ms=claim.valid_to_ms,
                            entity_indices=claim.entity_indices,
                        )
                        for claim in event.claims
                    ),
                )
                for event in output.events
            ),
            model_reference=result.model_reference,
            prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        )


def _context(observation: Observation, evidence: tuple[ResolvedEvidence, ...]) -> str:
    return json.dumps(
        {
            "observation_id": observation.observation_id,
            "duration_ms": round(
                (observation.ended_at - observation.occurred_at).total_seconds() * 1000
            ),
            "sensor": observation.sensor.value,
            "identity_observations": [
                {
                    "identity_id": identity.identity_id,
                    "kind": identity.kind.value,
                    "start_ms": identity.start_ms,
                    "end_ms": identity.end_ms,
                    "confidence": identity.confidence,
                    "model_id": identity.model_reference.model_id,
                    "model_revision": identity.model_reference.revision,
                    "scope": identity.scope.value,
                    "transcript": identity.transcript,
                    "visual_bbox_xyxy": identity.visual_bbox_xyxy,
                }
                for identity in observation.identity_observations
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
    observation: Observation,
    evidence: tuple[ResolvedEvidence, ...],
) -> _PerceptionOutput:
    if not content.strip():
        raise ModelOutputError("perception pipeline returned empty content")
    try:
        output = _PerceptionOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("perception pipeline returned invalid structured output") from error
    _require_grounded_output(observation, evidence, output)
    return output


def _require_observation_evidence(
    observation: Observation,
    evidence: tuple[ResolvedEvidence, ...],
) -> None:
    if not evidence:
        raise DomainInvariantError("event perception requires evidence")
    if any(
        item.evidence_span.tenant_id != observation.tenant_id
        or item.evidence_span.observation_id != observation.observation_id
        for item in evidence
    ):
        raise DomainInvariantError("event perception evidence must belong to its observation")


def _require_grounded_output(
    observation: Observation,
    evidence: tuple[ResolvedEvidence, ...],
    output: _PerceptionOutput,
) -> None:
    duration_ms = round((observation.ended_at - observation.occurred_at).total_seconds() * 1000)
    evidence_by_id = {str(item.evidence_span.evidence_id): item.evidence_span for item in evidence}
    for event in output.events:
        if event.end_ms > duration_ms:
            raise ModelOutputError("perception event exceeds observation duration")
        try:
            spans = tuple(evidence_by_id[evidence_id] for evidence_id in event.evidence_ids)
        except KeyError as error:
            raise ModelOutputError("perception event references unknown evidence") from error
        if any(
            not time_ranges_overlap(span.start_ms, span.end_ms, event.start_ms, event.end_ms)
            for span in spans
        ):
            raise ModelOutputError("perception event does not overlap its evidence")
        event_evidence_ids = set(event.evidence_ids)
        if any(
            not set(entity.evidence_ids) <= event_evidence_ids for entity in event.entities
        ) or any(not set(claim.evidence_ids) <= event_evidence_ids for claim in event.claims):
            raise ModelOutputError("perception detail references evidence outside its event")
        if any(
            claim.valid_from_ms < event.start_ms
            or claim.valid_from_ms > event.end_ms
            or (claim.valid_to_ms is not None and claim.valid_to_ms > event.end_ms)
            for claim in event.claims
        ):
            raise ModelOutputError("perception claim validity exceeds its event")
        if any(
            entity_index >= len(event.entities)
            for claim in event.claims
            for entity_index in claim.entity_indices
        ):
            raise ModelOutputError("perception claim references an unknown entity")
