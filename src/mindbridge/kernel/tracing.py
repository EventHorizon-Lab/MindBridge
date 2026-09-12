"""Span helpers shared by every kernel plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager

from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    IDENTITY_CACHED,
    IDENTITY_CREATED,
    IDENTITY_IDENTITIES,
    IDENTITY_MATCHED,
    IDENTITY_OBSERVATIONS,
    MODEL_MODULE,
    SPAN_KIND,
    model_span,
    operation_span,
    traced_span,
)
from mindbridge.exceptions import MindBridgeError
from mindbridge.types import Modality

# Model span modules mapped onto the failure stage a caller sees.
_MODEL_STAGES = {
    "consolidation": "consolidate",
    "embedding": "embed",
    "face": "recognize",
    "formation": "form",
    "generation": "generate",
    "transcription": "transcribe",
    "vision": "describe",
}


@contextmanager
def _staged(span: AbstractContextManager[Span], stage: str) -> Iterator[Span]:
    """Name the failing stage on public errors so an agent never parses prose to find it.

    The innermost span wins: an adapter that already classified its own failure keeps its stage.
    """
    with span as opened:
        try:
            yield opened
        except MindBridgeError as error:
            if error.stage is None:
                error.stage = stage
            raise


class Traced:
    """Base for every plane: one tracer, the same span helpers."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def _trace(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, AttributeValue] | None = None,
        failure_stage: str | None = None,
    ) -> AbstractContextManager[Span]:
        values = dict(attributes or {})
        values[SPAN_KIND] = kind
        if kind == "operation":
            return operation_span(self._tracer, name, attributes=values)
        return _staged(
            traced_span(self._tracer, name, attributes=values),
            failure_stage or name.removeprefix("mindbridge."),
        )

    def _model_trace(
        self,
        module: str,
        operation: str,
        *,
        model: str | None,
        batch_size: int,
        modalities: Iterable[Modality],
    ) -> AbstractContextManager[Span]:
        attributes: dict[str, AttributeValue] = {
            MODEL_MODULE: module,
            "gen_ai.operation.name": operation,
            "mindbridge.model.batch_size": batch_size,
            "mindbridge.input.modalities": tuple(
                sorted({modality.value for modality in modalities})
            ),
        }
        if model is not None:
            attributes["gen_ai.request.model"] = model
        attributes[SPAN_KIND] = "model"
        return _staged(
            model_span(self._tracer, f"mindbridge.model.{module}", attributes=attributes),
            _MODEL_STAGES[module],
        )

    def _trace_identity_yield(
        self,
        name: str,
        resolved: Sequence[tuple[str | None, float | None]],
        *,
        cached: bool,
    ) -> None:
        """Record one asset's recognizer yield under stable ``mindbridge.identity.*`` names.

        ``resolved`` pairs every observation's identity with its match score. All five attributes
        are emitted even for an empty asset: a recognizer whose detection threshold suits another
        domain runs, costs time, and yields nothing, which is otherwise indistinguishable from
        having configured no recognizer at all. ``cached`` separates re-read observations from a
        fresh analysis so that cheap-because-cached never reads as cheap-because-empty.

        ``observations`` and ``matched_existing`` count observations; ``identities`` and ``created``
        count distinct identities. The store reports ``identity_score`` only where an observation
        matched an existing identity by similarity, so its absence is the only available signal for
        a new identity. A cross-modal adoption, where one asset's lone face takes the identity of
        its lone voice, also arrives without a score and therefore counts as created here.
        """
        identities = {identity for identity, _ in resolved if identity is not None}
        created = {
            identity for identity, score in resolved if identity is not None and score is None
        }
        with self._trace(name, kind="stage") as span:
            span.set_attribute(IDENTITY_OBSERVATIONS, len(resolved))
            span.set_attribute(IDENTITY_IDENTITIES, len(identities))
            span.set_attribute(
                IDENTITY_MATCHED,
                sum(1 for _, score in resolved if score is not None),
            )
            span.set_attribute(IDENTITY_CREATED, len(created))
            span.set_attribute(IDENTITY_CACHED, cached)
