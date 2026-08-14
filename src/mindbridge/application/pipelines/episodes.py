"""Provider-neutral episode consolidation pipeline."""

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
from mindbridge.application.episodes import EpisodeConsolidation, EpisodeProposal
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.core import DomainInvariantError, Event, EventId, ModelOutputError
from mindbridge.prompts import CONSOLIDATE_EPISODES_PROMPT
from mindbridge.telemetry import set_current_span_attributes, trace_operation

_Description = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
_EventIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]


class _EpisodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_ids: Annotated[tuple[_EventIdentifier, ...], Field(min_length=2, max_length=32)]
    description: _Description
    salience: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def require_unique_events(self) -> _EpisodeOutput:
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("episode event_ids must be unique")
        return self


class _ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episodes: Annotated[tuple[_EpisodeOutput, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def require_disjoint_episodes(self) -> _ConsolidationOutput:
        event_ids = [event_id for episode in self.episodes for event_id in episode.event_ids]
        if len(event_ids) > 64:
            raise ValueError("episode output exceeds the candidate event limit")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("one event cannot appear in multiple episodes")
        return self


class EpisodePipeline:
    """Turn a Generator into evidence-first episode verification."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 4_096) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @trace_operation("mindbridge.pipeline.episodes")
    async def propose_episodes(
        self,
        events: tuple[Event, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EpisodeConsolidation:
        _require_candidate_evidence(events, evidence)
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=CONSOLIDATE_EPISODES_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(
                            f"<candidate_context>\n{_context(events, evidence)}\n"
                            "</candidate_context>"
                        ),
                        *evidence_parts(evidence),
                        TextPart(
                            "<final_task>Propose supported episode groupings now.</final_task>"
                        ),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            lambda content: _parse_output(content, events),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": CONSOLIDATE_EPISODES_PROMPT.version,
                "mindbridge.event.count": len(events),
                "mindbridge.evidence.count": len(evidence),
            }
        )
        return EpisodeConsolidation(
            episodes=tuple(
                EpisodeProposal(
                    event_ids=tuple(EventId(value) for value in episode.event_ids),
                    description=episode.description,
                    salience=episode.salience,
                )
                for episode in output.episodes
            ),
            model_reference=result.model_reference,
            prompt_version=CONSOLIDATE_EPISODES_PROMPT.version,
        )


def _context(events: tuple[Event, ...], evidence: tuple[ResolvedEvidence, ...]) -> str:
    return json.dumps(
        {
            "events": [
                {
                    "event_id": event.event_id,
                    "occurred_at": event.occurred_at.isoformat(),
                    "ended_at": event.ended_at.isoformat(),
                    "description": event.description,
                    "evidence_ids": event.evidence_ids,
                }
                for event in events
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


def _parse_output(content: str, events: tuple[Event, ...]) -> _ConsolidationOutput:
    if not content.strip():
        raise ModelOutputError("episode pipeline returned empty content")
    try:
        output = _ConsolidationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("episode pipeline returned invalid structured output") from error
    candidate_ids = {str(event.event_id) for event in events}
    if any(
        event_id not in candidate_ids
        for episode in output.episodes
        for event_id in episode.event_ids
    ):
        raise ModelOutputError("episode pipeline referenced an unknown event")
    return output


def _require_candidate_evidence(
    events: tuple[Event, ...],
    evidence: tuple[ResolvedEvidence, ...],
) -> None:
    if not 2 <= len(events) <= 64 or len({event.event_id for event in events}) != len(events):
        raise DomainInvariantError("episode consolidation requires 2 to 64 unique events")
    tenant_ids = {event.tenant_id for event in events} | {
        item.evidence_span.tenant_id for item in evidence
    }
    if len(tenant_ids) != 1:
        raise DomainInvariantError("episode candidates and evidence must belong to one tenant")
    expected = {evidence_id for event in events for evidence_id in event.evidence_ids}
    actual = {item.evidence_span.evidence_id for item in evidence}
    if expected != actual or len(actual) != len(evidence):
        raise DomainInvariantError(
            "episode consolidation requires each exact candidate evidence span"
        )
