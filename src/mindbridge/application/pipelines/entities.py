"""Provider-neutral adjudication of whether two entity records are one entity."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from mindbridge.application.capabilities import GenerateRequest, Generator, ModelInput, TextPart
from mindbridge.application.entity_resolution import EntityAdjudication, EntityPair
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.core import ModelOutputError
from mindbridge.prompts import RESOLVE_ENTITIES_PROMPT
from mindbridge.telemetry import operation_span, set_current_span_attributes

_Cue = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)]


class _AdjudicationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    same_entity: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    # Required in both directions. A verdict that cannot name what it rested on is not a
    # verdict, and this pair is the one place an unexamined "yes" would fuse two histories.
    discriminating_cue: _Cue


class EntityResolutionPipeline:
    """Turn a Generator into a pairwise, evidence-first identity judgement."""

    def __init__(self, generator: Generator, *, max_output_tokens: int = 512) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._generator = generator
        self._max_output_tokens = max_output_tokens

    @operation_span("mindbridge.pipeline.entity_resolution")
    async def adjudicate(
        self,
        pair: EntityPair,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EntityAdjudication:
        """Inspect both records' original media and judge them the same entity or not."""
        output, result = await generate_json(
            self._generator,
            GenerateRequest(
                system_prompt=RESOLVE_ENTITIES_PROMPT.text,
                input=ModelInput(
                    (
                        TextPart(_context(pair, evidence)),
                        *evidence_parts(evidence),
                        TextPart(
                            "<final_task>Judge these two records now. Answer false unless the "
                            "media shows an observation that holds them together."
                            "</final_task>"
                        ),
                    )
                ),
                max_output_tokens=self._max_output_tokens,
            ),
            _parse_output,
        )
        set_current_span_attributes(
            {
                "mindbridge.model.id": result.model_reference.model_id,
                "mindbridge.model.revision": result.model_reference.revision,
                "mindbridge.prompt.version": RESOLVE_ENTITIES_PROMPT.version,
                "mindbridge.entity.same_entity": output.same_entity,
                "mindbridge.evidence.count": len(evidence),
            }
        )
        return EntityAdjudication(
            same_entity=output.same_entity,
            confidence=output.confidence,
            discriminating_cue=output.discriminating_cue,
        )


def _context(pair: EntityPair, evidence: tuple[ResolvedEvidence, ...]) -> str:
    """Name each record, say which spans it cites, and bound every span in time.

    Without this the judge sees two names and a bag of whole recordings: two records citing
    different moments of one clip collapse to a single attachment, and nothing says which
    moment belongs to which record. The claim and summary pipelines join their candidates to
    their spans the same way.
    """
    return json.dumps(
        {
            "records": [
                {
                    "label": label,
                    "entity_type": candidate.entity.entity_type.value,
                    "canonical_name": candidate.entity.canonical_name,
                    "evidence_ids": list(candidate.evidence_ids),
                }
                for label, candidate in (("record_a", pair.left), ("record_b", pair.right))
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
        sort_keys=True,
    )


def _parse_output(content: str) -> _AdjudicationOutput:
    try:
        return _AdjudicationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("entity adjudication violated its schema") from error
