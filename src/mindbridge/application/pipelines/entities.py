"""Provider-neutral adjudication of whether two entity records are one entity."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from mindbridge.application.capabilities import GenerateRequest, Generator, ModelInput, TextPart
from mindbridge.application.entity_resolution import EntityAdjudication, EntityPair
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.pipelines.structured import generate_json, unwrap_json_code_fence
from mindbridge.core import ModelOutputError
from mindbridge.prompts import RESOLVE_ENTITIES_PROMPT
from mindbridge.telemetry import set_current_span_attributes, trace_operation

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

    @trace_operation("mindbridge.pipeline.entity_resolution")
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
                        TextPart(_context(pair)),
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


def _context(pair: EntityPair) -> str:
    return (
        "<entity_pair>\n"
        f"<record_a type={pair.left.entity.entity_type.value!r}>"
        f"{pair.left.entity.canonical_name}</record_a>\n"
        f"<record_b type={pair.right.entity.entity_type.value!r}>"
        f"{pair.right.entity.canonical_name}</record_b>\n"
        "</entity_pair>"
    )


def _parse_output(content: str) -> _AdjudicationOutput:
    try:
        return _AdjudicationOutput.model_validate_json(unwrap_json_code_fence(content))
    except ValidationError as error:
        raise ModelOutputError("entity adjudication violated its schema") from error
