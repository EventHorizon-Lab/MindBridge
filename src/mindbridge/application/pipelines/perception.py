"""Provider-neutral multimodal event perception pipeline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from mindbridge.application.capabilities import GenerateRequest, Generator, ModelInput, TextPart
from mindbridge.application.perception import (
    MAX_PERCEIVED_CLAIMS_PER_EVENT,
    MAX_PERCEIVED_COUNT,
    MAX_PERCEIVED_ENTITIES_PER_EVENT,
    MAX_PERCEPTION_CLAIMS,
    MAX_PERCEPTION_ENTITIES,
    MAX_PERCEPTION_EVENTS,
    EventPerception,
    PerceivedClaim,
    PerceivedCount,
    PerceivedEntity,
    PerceivedEvent,
    ResolvedEvidence,
    time_ranges_overlap,
)
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import (
    generate_json,
    output_schema,
    unwrap_json_code_fence,
)
from mindbridge.core import (
    ClaimType,
    DomainInvariantError,
    EntityType,
    EvidenceId,
    EvidenceSpan,
    ModelOutputError,
    Observation,
)
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT
from mindbridge.telemetry import (
    operation_span,
    record_output_repairs,
    set_current_span_attributes,
)

_MODEL_OUTPUT_CONFIG = ConfigDict(extra="ignore", frozen=True)
"""What these models accept: everything the prompt asked for, and nothing about what it did not.

This is a model-output boundary, not a trust boundary. Forbidding extras here rejected the whole
observation over one key the model invented and nothing reads -- a run lost every event, entity,
and claim in a clip, after minutes of generation, because the JSON also carried
`"start_ms_note": null`. On a slow generator that is the most expensive failure in the pipeline,
and it aborts the caller rather than degrading.

Ignoring an unknown key costs no safety. Every field below is still validated exactly as before,
and a *renamed* field still fails, because the field it replaced is then missing. Extras are
forbidden where a caller could smuggle something past a contract -- the Celery message schema and
the public request models -- not where a language model padded its own answer.
"""

_T = TypeVar("_T")
_Element = TypeVar("_Element", bound=BaseModel)


def _deduplicated(values: tuple[_T, ...]) -> tuple[_T, ...]:
    """Collapse repeats the model emitted, keeping the order it first mentioned them in.

    A repeated evidence id or entity index says nothing a single mention does not, so rejecting
    the observation over one is throwing away minutes of generation to punish a typo. Repair what
    carries no meaning; refuse to repair what does -- an unknown evidence id, an event that ends
    before it starts, a claim whose validity leaves its event. Those values are not coerced into
    something acceptable; `_kept` drops the element that carries them.
    """
    return tuple(dict.fromkeys(values))


def _kept(model: type[_Element], payload: object) -> _Element | None:
    """Validate one element on its own, so a wrong one costs only itself.

    Validating the whole answer as a unit made one bad field cost every other element in the same
    generation: 28 of 61 write-path job failures in the 2026-08-21 evaluation were
    `model_output_invalid`, all of them from a single rejected value inside output that was
    otherwise usable, and each one had already been paid for in full -- minutes of generation per
    clip. An element is the unit a model gets wrong, so it is the unit that gets dropped.

    Retry is not the alternative here. The three clips that gated EgoLifeQA each failed at
    `attempt=3` with the identical rejection, so a re-ask buys the same failure at full price.
    """
    try:
        return model.model_validate(payload)
    except ValidationError:
        return None


_Description = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
_EvidenceIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
_EvidenceIdentifiers = Annotated[
    tuple[_EvidenceIdentifier, ...], Field(min_length=1), AfterValidator(_deduplicated)
]
_EntityIndices = Annotated[
    tuple[Annotated[int, Field(ge=0)], ...], Field(max_length=32), AfterValidator(_deduplicated)
]
_CountSubject = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]


class _CountOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    subject: _CountSubject
    value: Annotated[int, Field(ge=0, le=MAX_PERCEIVED_COUNT)]


def _usable_count(value: object) -> object:
    """Drop a malformed count instead of the claim it was attached to.

    Same rule as everywhere else in this module: the element a model got wrong is the element
    that goes. A claim carrying an unusable count still carries a statement, a window, and its
    evidence, and losing all of that over an optional field would make adding the field a net
    loss on exactly the clips it was added for.
    """
    if value is None or isinstance(value, _CountOutput):
        return value
    return _kept(_CountOutput, value)


class _EntityOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    entity_type: EntityType
    canonical_name: _Description
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: _EvidenceIdentifiers


class _ClaimOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    claim_type: ClaimType
    statement: _Description
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: _EvidenceIdentifiers
    valid_from_ms: Annotated[int, Field(ge=0)]
    valid_to_ms: Annotated[int, Field(ge=0)] | None
    entity_indices: _EntityIndices = ()
    exact_count: Annotated[_CountOutput | None, BeforeValidator(_usable_count)] = None

    @model_validator(mode="after")
    def require_valid_references(self) -> _ClaimOutput:
        if self.valid_to_ms is not None and self.valid_to_ms < self.valid_from_ms:
            raise ValueError("valid_to_ms must not precede valid_from_ms")
        return self


class _EventOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=0)]
    description: _Description
    salience: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: _EvidenceIdentifiers
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
        return self


class _RawPerceptionOutput(BaseModel):
    """The outer shape only: every event is judged, and kept or dropped, on its own."""

    model_config = _MODEL_OUTPUT_CONFIG

    events: tuple[object, ...]


class _PerceptionOutput(BaseModel):
    model_config = _MODEL_OUTPUT_CONFIG

    events: Annotated[tuple[_EventOutput, ...], Field(max_length=MAX_PERCEPTION_EVENTS)]

    @model_validator(mode="after")
    def require_bounded_details(self) -> _PerceptionOutput:
        if sum(len(event.entities) for event in self.events) > MAX_PERCEPTION_ENTITIES:
            raise ValueError("total perception entity count exceeds the processing limit")
        if sum(len(event.claims) for event in self.events) > MAX_PERCEPTION_CLAIMS:
            raise ValueError("total perception claim count exceeds the processing limit")
        return self


_PERCEPTION_SCHEMA = output_schema("perception_events", _PerceptionOutput)


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
                output_schema=_PERCEPTION_SCHEMA,
            ),
            lambda content: _parse_output(content, observation, evidence),
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
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
                            exact_count=_perceived_count(claim.exact_count),
                        )
                        for claim in event.claims
                    ),
                )
                for event in output.events
            ),
            model_reference=result.model_reference,
            prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        )


def _perceived_count(count: _CountOutput | None) -> PerceivedCount | None:
    if count is None:
        return None
    return PerceivedCount(subject=count.subject, value=count.value)


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
        raw = _RawPerceptionOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("perception pipeline returned invalid structured output") from error
    duration_ms = round((observation.ended_at - observation.occurred_at).total_seconds() * 1000)
    spans = {str(item.evidence_span.evidence_id): item.evidence_span for item in evidence}
    dropped: Counter[str] = Counter()
    budget = _DetailBudget()
    events: list[_EventOutput] = []
    for item in raw.events:
        if len(events) >= MAX_PERCEPTION_EVENTS:
            dropped["event_over_cap"] += 1
            continue
        event = _usable_event(item, duration_ms, spans, budget, dropped)
        if event is not None:
            events.append(event)
    if dropped:
        # Partial output that nobody can see is just quiet data loss, so what was discarded is
        # recorded even though the observation goes on to commit. The two kinds are counted
        # apart because they call for opposite responses: a model that got a value wrong is not
        # the same problem as a processing limit that has become too small for the prompt.
        record_output_repairs(
            {
                "mindbridge.perception.dropped_event_count": dropped["event"],
                "mindbridge.perception.dropped_entity_count": dropped["entity"],
                "mindbridge.perception.dropped_claim_count": dropped["claim"],
                "mindbridge.perception.over_cap_event_count": dropped["event_over_cap"],
                "mindbridge.perception.over_cap_entity_count": dropped["entity_over_cap"],
                "mindbridge.perception.over_cap_claim_count": dropped["claim_over_cap"],
            }
        )
    # Perceiving nothing is a legitimate answer; having every event dropped is not the same thing.
    # Committing an observation whose entire content was discarded would report success for work
    # that produced none, so that stays a failure the caller sees.
    if raw.events and not events:
        raise ModelOutputError("perception pipeline returned no usable event")
    try:
        # Every bound above is applied while building, so this can only fail if one of them was
        # missed -- which is worth failing loudly for rather than writing an unbounded batch.
        return _PerceptionOutput(events=tuple(events))
    except ValidationError as error:
        raise ModelOutputError("perception pipeline returned invalid structured output") from error


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


@dataclass(slots=True)
class _DetailBudget:
    """What is left of one observation's entity and claim allowance, spent in event order.

    The totals used to be a validator that raised, so an observation carrying one claim past the
    limit lost every event it had -- the same failure the rest of this module exists to stop, with
    a processing bound as the trigger instead of a wrong value. Nothing about the 257th claim is
    invalid; the system simply declines to store that many, which is a reason to drop the surplus
    and not the observation. The per-event limits behave the same way, and the frame ceiling in
    `evidence_clips` already set the precedent: too much degrades, it does not fail.

    ponytail: spent first-come, so a clip that overruns loses the claims of its last events while
    keeping their descriptions. Fair shares would need the totals up front and a second pass; the
    over-cap counters on the span are what would say the limit is worth raising instead.
    """

    entities: int = MAX_PERCEPTION_ENTITIES
    claims: int = MAX_PERCEPTION_CLAIMS


def _usable_event(
    raw: object,
    duration_ms: int,
    spans: Mapping[str, EvidenceSpan],
    budget: _DetailBudget,
    dropped: Counter[str],
) -> _EventOutput | None:
    """Keep one event, carrying only the details its own window, evidence, and budget allow.

    The event is validated twice: once bare, because its window and evidence list are what the
    details are judged against, and once with the surviving details.
    """
    if not isinstance(raw, Mapping):
        dropped["event"] += 1
        return None
    bare = _kept(_EventOutput, {**raw, "entities": (), "claims": ()})
    if bare is None or not _grounded_event(bare, duration_ms, spans):
        dropped["event"] += 1
        return None
    entities, positions = _grounded_entities(
        raw.get("entities", ()),
        bare,
        min(budget.entities, MAX_PERCEIVED_ENTITIES_PER_EVENT),
        dropped,
    )
    claims = _grounded_claims(
        raw.get("claims", ()),
        bare,
        positions,
        min(budget.claims, MAX_PERCEIVED_CLAIMS_PER_EVENT),
        dropped,
    )
    event = _kept(_EventOutput, {**raw, "entities": entities, "claims": claims})
    if event is None:
        dropped["event"] += 1
        return None
    # Charged from what survived, so an event that was dropped never spends another event's room.
    budget.entities -= len(event.entities)
    budget.claims -= len(event.claims)
    return event


def _grounded_event(
    event: _EventOutput,
    duration_ms: int,
    spans: Mapping[str, EvidenceSpan],
) -> bool:
    """An event has to sit inside the observation and overlap every span it cites.

    Fabricated provenance is not repairable: dropping the unknown id would leave an event claiming
    support it never had, and widening the event to the span it names would invent the support
    instead. The event goes.
    """
    if event.end_ms > duration_ms:
        return False
    cited = tuple(spans.get(evidence_id) for evidence_id in event.evidence_ids)
    return all(
        span is not None
        and time_ranges_overlap(span.start_ms, span.end_ms, event.start_ms, event.end_ms)
        for span in cited
    )


def _grounded_entities(
    raw: object,
    event: _EventOutput,
    limit: int,
    dropped: Counter[str],
) -> tuple[object, Mapping[int, int]]:
    """Keep the entities the event's own evidence supports, and where each one ended up.

    Claims address entities by position, so a dropped entity has to be remembered as a moved
    position. Renumbering silently would leave a surviving claim describing its neighbour, which
    is a wrong memory rather than a lost one.
    """
    if not isinstance(raw, list):
        # Not a list of entities at all: the event's own validation is what should judge that.
        return raw, {}
    kept: list[_EntityOutput] = []
    positions: dict[int, int] = {}
    for position, item in enumerate(raw):
        if len(kept) >= limit:
            dropped["entity_over_cap"] += 1
            continue
        entity = _kept(_EntityOutput, item)
        if entity is None or not set(entity.evidence_ids) <= set(event.evidence_ids):
            dropped["entity"] += 1
            continue
        positions[position] = len(kept)
        kept.append(entity)
    return tuple(kept), positions


def _grounded_claims(
    raw: object,
    event: _EventOutput,
    positions: Mapping[int, int],
    limit: int,
    dropped: Counter[str],
) -> object:
    """Keep the claims the event supports, pointed at where their entities actually are."""
    if not isinstance(raw, list):
        return raw
    kept: list[_ClaimOutput] = []
    for item in raw:
        if len(kept) >= limit:
            dropped["claim_over_cap"] += 1
            continue
        claim = _kept(_ClaimOutput, item)
        if claim is None or not _grounded_claim(claim, event, positions):
            dropped["claim"] += 1
            continue
        kept.append(
            claim.model_copy(
                update={"entity_indices": tuple(positions[index] for index in claim.entity_indices)}
            )
        )
    return tuple(kept)


def _grounded_claim(
    claim: _ClaimOutput,
    event: _EventOutput,
    positions: Mapping[int, int],
) -> bool:
    """A claim has to cite the event's evidence, stay inside its window, and name a kept entity.

    Validity outside the window is not clamped into it: the window is what the evidence covers, so
    a claim reaching past it is asserting something the clip cannot show. This rejection was
    measured once in the 2026-08-21 run, and it cost the whole observation.
    """
    return (
        set(claim.evidence_ids) <= set(event.evidence_ids)
        and event.start_ms <= claim.valid_from_ms <= event.end_ms
        and (claim.valid_to_ms is None or claim.valid_to_ms <= event.end_ms)
        and set(claim.entity_indices) <= set(positions)
    )
