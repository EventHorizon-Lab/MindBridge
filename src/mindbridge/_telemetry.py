"""Shared OpenTelemetry names and model-usage recording helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
MODEL_TTFT = "mindbridge.model.time_to_first_token"
GEN_AI_TTFC = "gen_ai.response.time_to_first_chunk"
GEN_AI_FINISH_REASONS = "gen_ai.response.finish_reasons"
EMBEDDING_PARTS_ELIDED = "mindbridge.embedding.elided_parts"
EMBEDDING_VIDEO_SAMPLED = "mindbridge.embedding.video_sampled_inputs"
GROUNDING_MEDIA_ELIDED = "mindbridge.grounding.media_elided_hits"
GROUNDING_HITS_DROPPED = "mindbridge.grounding.dropped_hits"

TOKEN_MODALITIES = ("text", "image", "video", "audio", "unattributed")


@dataclass(slots=True)
class _ModelUsage:
    request_count: int = 0
    expected_requests: int = 0
    reported_requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_by_modality: Mapping[str, int] = field(default_factory=dict)
    output_by_modality: Mapping[str, int] = field(default_factory=dict)
    audio_seconds: float | None = None


@dataclass(slots=True)
class _OperationUsage:
    lock: Lock = field(default_factory=Lock)
    request_count: int = 0
    expected_requests: int = 0
    reported_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_seen: bool = False
    output_seen: bool = False
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
        if self.input_seen:
            span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
        if self.output_seen:
            span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
        if self.audio_seconds:
            span.set_attribute(TOKEN_AUDIO_SECONDS, self.audio_seconds)
        _record_modalities(span, "input", self.input_by_modality)
        _record_modalities(span, "output", self.output_by_modality)


_CURRENT_MODEL_USAGE: ContextVar[_ModelUsage | None] = ContextVar(
    "mindbridge_model_usage", default=None
)
_CURRENT_OPERATION_USAGE: ContextVar[_OperationUsage | None] = ContextVar(
    "mindbridge_operation_usage", default=None
)


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
        token = _CURRENT_OPERATION_USAGE.set(usage)
        try:
            yield span
        finally:
            try:
                usage.write(span)
            finally:
                _CURRENT_OPERATION_USAGE.reset(token)


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
        if operation is None:
            yield span
            return
        usage = _ModelUsage()
        token = _CURRENT_MODEL_USAGE.set(usage)
        try:
            yield span
        finally:
            try:
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


def current_model_request_count() -> int:
    """Return requests issued in the current MindBridge model boundary."""
    usage = _CURRENT_MODEL_USAGE.get()
    return 0 if usage is None else usage.request_count


def record_model_usage(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    input_by_modality: Mapping[str, int] | None = None,
    output_by_modality: Mapping[str, int] | None = None,
    request_count: int = 1,
    expected_requests: int = 1,
    reported_requests: int = 1,
    audio_seconds: float | None = None,
) -> None:
    """Attach provider-reported usage to the current model span without estimation."""
    usage = _CURRENT_MODEL_USAGE.get()
    if usage is not None:
        usage.request_count = request_count
        usage.expected_requests = expected_requests
        usage.reported_requests = reported_requests
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.total_tokens = total_tokens
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
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if total_tokens is not None:
        span.set_attribute(TOKEN_TOTAL, total_tokens)
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
