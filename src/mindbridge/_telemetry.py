"""Shared OpenTelemetry names and model-usage recording helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

TRACER_NAME = "mindbridge"
SPAN_KIND = "mindbridge.span.kind"
MODEL_MODULE = "mindbridge.model.module"
MODEL_REQUEST_COUNT = "mindbridge.model.request_count"
TOKEN_EXPECTED_REQUEST_COUNT = "mindbridge.token_usage.expected_request_count"
TOKEN_REPORTED_REQUEST_COUNT = "mindbridge.token_usage.reported_request_count"
TOKEN_TOTAL = "mindbridge.token_usage.total_tokens"
TOKEN_COMPLETE = "mindbridge.token_usage.complete"
TOKEN_AUDIO_SECONDS = "mindbridge.token_usage.audio_seconds"
TOKEN_CACHED_INPUT = "gen_ai.usage.cache_read.input_tokens"
TOKEN_REASONING_OUTPUT = "gen_ai.usage.reasoning.output_tokens"
TOKEN_INPUT_COMPLETE = "mindbridge.token_usage.input_tokens.complete"
TOKEN_OUTPUT_COMPLETE = "mindbridge.token_usage.output_tokens.complete"
TOKEN_CACHED_INPUT_COMPLETE = "mindbridge.token_usage.cached_input_tokens.complete"
TOKEN_REASONING_OUTPUT_COMPLETE = "mindbridge.token_usage.reasoning_output_tokens.complete"
MODEL_TTFT = "mindbridge.model.time_to_first_token"
OPERATION_TTFT = "mindbridge.operation.time_to_first_token_ms"
GEN_AI_TTFC = "gen_ai.response.time_to_first_chunk"
GEN_AI_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "gen_ai.openai.response.system_fingerprint"
MODEL_RESPONSE_MODELS = "mindbridge.model.response_models"
MODEL_RESPONSE_SYSTEM_FINGERPRINTS = "mindbridge.model.response_system_fingerprints"
EMBEDDING_TASK = "mindbridge.embedding.task"
ASYNC_QUEUE_TIME = "mindbridge.async.queue_time_ms"
CAPTURE_SETTLED = "mindbridge.capture.records_settled"
CAPTURE_FAILED = "mindbridge.capture.records_failed"
# The capture-to-searchable interval `docs/context-os.md` requires measured separately from the
# capture acknowledgement and from settle duration. Reported as the batch maximum: one span
# attribute answers "how stale was the oldest record this pass made searchable".
CAPTURE_TIME_TO_SEARCHABLE = "mindbridge.capture.max_time_to_searchable_ms"
EMBEDDING_PARTS_ELIDED = "mindbridge.embedding.elided_parts"
EMBEDDING_VIDEO_SAMPLED = "mindbridge.embedding.video_sampled_inputs"
GROUNDING_MEDIA_ELIDED = "mindbridge.grounding.media_elided_hits"
GROUNDING_HITS_DROPPED = "mindbridge.grounding.dropped_hits"
# Proposals the model adapter could not read, counted on the model span, and proposals the kernel
# refused for how they were grounded, accumulated over a whole operation. They answer different
# questions -- a backend returning garbage, against a backend grounding an opinion wrongly -- so
# they stay separate names.
FORMATION_PROPOSALS_DROPPED = "mindbridge.formation.dropped_proposals"
FORMATION_PROPOSALS_REFUSED = "mindbridge.formation.refused_proposals"
VISION_BATCHES_FAILED = "mindbridge.vision.failed_batches"
IDENTITY_OBSERVATIONS = "mindbridge.identity.observations"
IDENTITY_MATCHED = "mindbridge.identity.matched_existing"
IDENTITY_IDENTITIES = "mindbridge.identity.identities"
IDENTITY_CREATED = "mindbridge.identity.created"
IDENTITY_CACHED = "mindbridge.identity.cached"
IDENTITY_EVIDENCE_ASSETS = "mindbridge.identity.evidence_assets"
IDENTITY_EVIDENCE_REQUIRED = "mindbridge.identity.evidence_required"
IDENTITY_LINKED = "mindbridge.identity.linked"

TOKEN_MODALITIES = ("text", "image", "video", "audio", "unattributed")


@dataclass(slots=True)
class _ModelUsage:
    request_count: int = 0
    expected_requests: int = 0
    reported_requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    input_complete: bool | None = None
    output_complete: bool | None = None
    cached_input_complete: bool | None = None
    reasoning_output_complete: bool | None = None
    input_by_modality: Mapping[str, int] = field(default_factory=dict)
    output_by_modality: Mapping[str, int] = field(default_factory=dict)
    audio_seconds: float | None = None
    response_models: set[str] = field(default_factory=set)
    response_system_fingerprints: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _OperationUsage:
    lock: Lock = field(default_factory=Lock)
    request_count: int = 0
    expected_requests: int = 0
    reported_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    input_seen: bool = False
    output_seen: bool = False
    cached_input_seen: bool = False
    reasoning_output_seen: bool = False
    input_complete: bool = True
    output_complete: bool = True
    cached_input_complete: bool = True
    reasoning_output_complete: bool = True
    total_complete: bool = True
    input_by_modality: dict[str, int] = field(default_factory=dict)
    output_by_modality: dict[str, int] = field(default_factory=dict)
    audio_seconds: float = 0.0

    def add(self, usage: _ModelUsage) -> None:
        with self.lock:
            self.request_count += usage.request_count
            self.expected_requests += usage.expected_requests
            self.reported_requests += usage.reported_requests
            if usage.input_tokens is not None:
                self.input_tokens += usage.input_tokens
                self.input_seen = True
            if usage.output_tokens is not None:
                self.output_tokens += usage.output_tokens
                self.output_seen = True
            if usage.cached_input_tokens is not None:
                self.cached_input_tokens += usage.cached_input_tokens
                self.cached_input_seen = True
            if usage.reasoning_output_tokens is not None:
                self.reasoning_output_tokens += usage.reasoning_output_tokens
                self.reasoning_output_seen = True
            if usage.expected_requests:
                self.input_complete &= usage.input_complete is True
                self.output_complete &= usage.output_complete is True
                self.cached_input_complete &= usage.cached_input_complete is True
                self.reasoning_output_complete &= usage.reasoning_output_complete is True
            self.total_tokens += usage.total_tokens or 0
            self.total_complete &= usage.reported_requests == 0 or usage.total_tokens is not None
            _sum_modalities(self.input_by_modality, usage.input_by_modality)
            _sum_modalities(self.output_by_modality, usage.output_by_modality)
            self.audio_seconds += usage.audio_seconds or 0.0

    def write(self, span: Span) -> None:
        complete = self.expected_requests == self.reported_requests and self.total_complete
        span.set_attribute(MODEL_REQUEST_COUNT, self.request_count)
        span.set_attribute(TOKEN_EXPECTED_REQUEST_COUNT, self.expected_requests)
        span.set_attribute(TOKEN_REPORTED_REQUEST_COUNT, self.reported_requests)
        span.set_attribute(TOKEN_COMPLETE, complete)
        if complete or (self.reported_requests and self.total_complete):
            span.set_attribute(TOKEN_TOTAL, self.total_tokens)
        span.set_attribute(
            TOKEN_INPUT_COMPLETE,
            bool(self.expected_requests and self.input_seen and self.input_complete),
        )
        span.set_attribute(
            TOKEN_OUTPUT_COMPLETE,
            bool(self.expected_requests and self.output_seen and self.output_complete),
        )
        span.set_attribute(
            TOKEN_CACHED_INPUT_COMPLETE,
            bool(self.expected_requests and self.cached_input_seen and self.cached_input_complete),
        )
        span.set_attribute(
            TOKEN_REASONING_OUTPUT_COMPLETE,
            bool(
                self.expected_requests
                and self.reasoning_output_seen
                and self.reasoning_output_complete
            ),
        )
        if self.input_seen:
            span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
        if self.output_seen:
            span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
        if self.cached_input_seen:
            span.set_attribute(TOKEN_CACHED_INPUT, self.cached_input_tokens)
        if self.reasoning_output_seen:
            span.set_attribute(TOKEN_REASONING_OUTPUT, self.reasoning_output_tokens)
        if self.audio_seconds:
            span.set_attribute(TOKEN_AUDIO_SECONDS, self.audio_seconds)
        _record_modalities(span, "input", self.input_by_modality)
        _record_modalities(span, "output", self.output_by_modality)


@dataclass(slots=True)
class _FormationRefusals:
    lock: Lock = field(default_factory=Lock)
    count: int = 0
    formed: bool = False

    def add(self, count: int) -> None:
        with self.lock:
            self.count += count
            self.formed = True

    def write(self, span: Span) -> None:
        # Only once formation actually ran: the absence of the attribute is otherwise
        # indistinguishable from a pass that refused nothing, which is what it exists to tell
        # apart. Every operation opens one of these, and most never form anything.
        if self.formed:
            span.set_attribute(FORMATION_PROPOSALS_REFUSED, self.count)


_CURRENT_MODEL_USAGE: ContextVar[_ModelUsage | None] = ContextVar(
    "mindbridge_model_usage", default=None
)
_CURRENT_OPERATION_USAGE: ContextVar[_OperationUsage | None] = ContextVar(
    "mindbridge_operation_usage", default=None
)
_CURRENT_FORMATION_REFUSALS: ContextVar[_FormationRefusals | None] = ContextVar(
    "mindbridge_formation_refusals", default=None
)
_RETRIEVAL_OBSERVER: ContextVar[Callable[[object], None] | None] = ContextVar(
    "mindbridge_retrieval_observer", default=None
)


@contextmanager
def _observe_retrieval_results(observer: Callable[[object], None]) -> Iterator[None]:
    """Let an in-process benchmark observe the ranked list an answer already computed."""
    token = _RETRIEVAL_OBSERVER.set(observer)
    try:
        yield
    finally:
        _RETRIEVAL_OBSERVER.reset(token)


def _record_retrieval_results(results: object) -> None:
    observer = _RETRIEVAL_OBSERVER.get()
    if observer is None:
        return
    # Observability must not change a product answer; the harness marks the missing list.
    with suppress(Exception):
        observer(results)


def token_modality_attribute(direction: str, modality: str) -> str:
    """Return the stable attribute name for one exact modality token count."""
    return f"mindbridge.token_usage.{direction}_tokens.{modality}"


@contextmanager
def traced_span(
    tracer: Tracer,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue],
) -> Iterator[Span]:
    """Create a span that records only a generic error status, never exception details."""
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException:
            span.set_status(StatusCode.ERROR)
            raise


@contextmanager
def operation_span(
    tracer: Tracer,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue],
) -> Iterator[Span]:
    """Trace one public operation and attach exact descendant model totals."""
    with traced_span(tracer, name, attributes=attributes) as span:
        if not span.is_recording():
            yield span
            return
        usage = _OperationUsage()
        refusals = _FormationRefusals()
        token = _CURRENT_OPERATION_USAGE.set(usage)
        refusals_token = _CURRENT_FORMATION_REFUSALS.set(refusals)
        try:
            yield span
        finally:
            try:
                usage.write(span)
                refusals.write(span)
            finally:
                _CURRENT_OPERATION_USAGE.reset(token)
                _CURRENT_FORMATION_REFUSALS.reset(refusals_token)


@contextmanager
def model_span(
    tracer: Tracer,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue],
) -> Iterator[Span]:
    """Trace one model boundary and roll its final usage into the public operation."""
    with traced_span(tracer, name, attributes=attributes) as span:
        operation = _CURRENT_OPERATION_USAGE.get()
        usage = _ModelUsage()
        token = _CURRENT_MODEL_USAGE.set(usage)
        try:
            yield span
        finally:
            try:
                _write_model_provenance(span, usage)
                if operation is not None:
                    operation.add(usage)
            finally:
                _CURRENT_MODEL_USAGE.reset(token)


def mark_model_requests(count: int, *, token_usage_expected: int | None = None) -> None:
    """Record requests issued by the current model span before provider I/O."""
    expected = count if token_usage_expected is None else token_usage_expected
    usage = _CURRENT_MODEL_USAGE.get()
    if usage is not None:
        usage.request_count = count
        usage.expected_requests = expected
        usage.reported_requests = 0
        usage.input_tokens = None
        usage.output_tokens = None
        usage.total_tokens = None
        usage.cached_input_tokens = None
        usage.reasoning_output_tokens = None
        usage.input_complete = None
        usage.output_complete = None
        usage.cached_input_complete = None
        usage.reasoning_output_complete = None
        usage.input_by_modality = {}
        usage.output_by_modality = {}
        usage.audio_seconds = None
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute(MODEL_REQUEST_COUNT, count)
    span.set_attribute(
        TOKEN_EXPECTED_REQUEST_COUNT,
        expected,
    )
    span.set_attribute(TOKEN_REPORTED_REQUEST_COUNT, 0)
    span.set_attribute(TOKEN_COMPLETE, expected == 0)
    for name in (
        TOKEN_INPUT_COMPLETE,
        TOKEN_OUTPUT_COMPLETE,
        TOKEN_CACHED_INPUT_COMPLETE,
        TOKEN_REASONING_OUTPUT_COMPLETE,
    ):
        span.set_attribute(name, False)


def record_formation_refusals(count: int) -> None:
    """Add one formation pass's refused proposals to the current operation's total.

    Called even at zero, because one operation forms many records: `settle` runs a pass per
    captured row, and publishing per pass let the last row overwrite what the first one lost.
    """
    refusals = _CURRENT_FORMATION_REFUSALS.get()
    if refusals is not None:
        refusals.add(count)


def current_model_request_count() -> int:
    """Return requests issued in the current MindBridge model boundary."""
    usage = _CURRENT_MODEL_USAGE.get()
    return 0 if usage is None else usage.request_count


def record_model_provenance(*, response_model: str | None, system_fingerprint: str | None) -> None:
    """Attach every stable response identity without assigning a mixed batch to its last reply."""
    model = None if response_model is None else response_model.strip()
    fingerprint = None if system_fingerprint is None else system_fingerprint.strip()
    usage = _CURRENT_MODEL_USAGE.get()
    if usage is not None:
        if model:
            usage.response_models.add(model)
        if fingerprint:
            usage.response_system_fingerprints.add(fingerprint)
        return
    span = trace.get_current_span()
    if model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, model)
        span.set_attribute(MODEL_RESPONSE_MODELS, (model,))
    if fingerprint:
        span.set_attribute(GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT, fingerprint)
        span.set_attribute(MODEL_RESPONSE_SYSTEM_FINGERPRINTS, (fingerprint,))


def record_model_usage(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    cached_input_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
    input_by_modality: Mapping[str, int] | None = None,
    output_by_modality: Mapping[str, int] | None = None,
    request_count: int = 1,
    expected_requests: int = 1,
    reported_requests: int = 1,
    input_tokens_complete: bool | None = None,
    output_tokens_complete: bool | None = None,
    cached_input_tokens_complete: bool | None = None,
    reasoning_output_tokens_complete: bool | None = None,
    audio_seconds: float | None = None,
) -> None:
    """Attach provider-reported usage to the current model span without estimation."""
    reported_all = expected_requests == reported_requests
    component_states = (
        (
            TOKEN_INPUT_COMPLETE,
            input_tokens,
            input_tokens_complete,
        ),
        (
            TOKEN_OUTPUT_COMPLETE,
            output_tokens,
            output_tokens_complete,
        ),
        (
            TOKEN_CACHED_INPUT_COMPLETE,
            cached_input_tokens,
            cached_input_tokens_complete,
        ),
        (
            TOKEN_REASONING_OUTPUT_COMPLETE,
            reasoning_output_tokens,
            reasoning_output_tokens_complete,
        ),
    )
    resolved_states = tuple(
        explicit
        if explicit is not None
        else bool(expected_requests and reported_all and value is not None)
        for _name, value, explicit in component_states
    )
    usage = _CURRENT_MODEL_USAGE.get()
    if usage is not None:
        usage.request_count = request_count
        usage.expected_requests = expected_requests
        usage.reported_requests = reported_requests
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.total_tokens = total_tokens
        usage.cached_input_tokens = cached_input_tokens
        usage.reasoning_output_tokens = reasoning_output_tokens
        usage.input_complete = resolved_states[0]
        usage.output_complete = resolved_states[1]
        usage.cached_input_complete = resolved_states[2]
        usage.reasoning_output_complete = resolved_states[3]
        usage.input_by_modality = dict(input_by_modality or {})
        usage.output_by_modality = dict(output_by_modality or {})
        usage.audio_seconds = audio_seconds
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute(MODEL_REQUEST_COUNT, request_count)
    span.set_attribute(TOKEN_EXPECTED_REQUEST_COUNT, expected_requests)
    span.set_attribute(TOKEN_REPORTED_REQUEST_COUNT, reported_requests)
    span.set_attribute(
        TOKEN_COMPLETE,
        expected_requests == reported_requests
        and (reported_requests == 0 or total_tokens is not None),
    )
    for (name, _value, _explicit), complete in zip(component_states, resolved_states, strict=True):
        span.set_attribute(name, complete)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if total_tokens is not None:
        span.set_attribute(TOKEN_TOTAL, total_tokens)
    if cached_input_tokens is not None:
        span.set_attribute(TOKEN_CACHED_INPUT, cached_input_tokens)
    if reasoning_output_tokens is not None:
        span.set_attribute(TOKEN_REASONING_OUTPUT, reasoning_output_tokens)
    if audio_seconds is not None:
        span.set_attribute(TOKEN_AUDIO_SECONDS, audio_seconds)
    _record_modalities(span, "input", input_by_modality or {})
    _record_modalities(span, "output", output_by_modality or {})


def record_unmetered_model_usage(
    *, request_count: int = 1, audio_seconds: float | None = None
) -> None:
    """Mark successful local inference as not exposing token-billing units."""
    record_model_usage(
        input_tokens=None,
        output_tokens=None,
        total_tokens=0,
        request_count=request_count,
        expected_requests=0,
        reported_requests=0,
        audio_seconds=audio_seconds,
    )


def _record_modalities(span: Span, direction: str, values: Mapping[str, int]) -> None:
    for modality, count in values.items():
        if modality in TOKEN_MODALITIES:
            span.set_attribute(token_modality_attribute(direction, modality), count)


def _sum_modalities(target: dict[str, int], values: Mapping[str, int]) -> None:
    for modality, count in values.items():
        if modality in TOKEN_MODALITIES:
            target[modality] = target.get(modality, 0) + count


def _write_model_provenance(span: Span, usage: _ModelUsage) -> None:
    for scalar, plural, values in (
        (GEN_AI_RESPONSE_MODEL, MODEL_RESPONSE_MODELS, usage.response_models),
        (
            GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
            MODEL_RESPONSE_SYSTEM_FINGERPRINTS,
            usage.response_system_fingerprints,
        ),
    ):
        if not values:
            continue
        ordered = tuple(sorted(values))
        span.set_attribute(scalar, ordered[0] if len(ordered) == 1 else "mixed")
        span.set_attribute(plural, ordered)
