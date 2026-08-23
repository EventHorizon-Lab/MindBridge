"""Small OpenTelemetry boundary for runtime setup and domain spans.

Every measurement here has two sinks and one definition. `OTEL_TRACES_EXPORTER=console` and
`OTEL_METRICS_EXPORTER=console` render the same instruments into the process's own output, so a
deployment with no collector can still read its stage timings and its token counts; they cannot
disagree with what OTLP reports, because there is nothing measured twice.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import version
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricsData
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

_INSTRUMENTATION_VERSION = version("mindbridge")
_TRACER = trace.get_tracer("mindbridge", _INSTRUMENTATION_VERSION)
_METER = metrics.get_meter("mindbridge", _INSTRUMENTATION_VERSION)
_OPERATION_CALLS = _METER.create_counter(
    "mindbridge.operation.calls",
    unit="{call}",
    description="Completed MindBridge domain operations.",
)
_OPERATION_DURATION = _METER.create_histogram(
    "mindbridge.operation.duration",
    unit="s",
    description="MindBridge domain operation duration.",
)
_STAGE_DURATION = _METER.create_histogram(
    "mindbridge.stage.duration",
    unit="s",
    description="Elapsed time for low-cardinality MindBridge pipeline milestones.",
)
_MODEL_TOKENS = _METER.create_counter(
    "mindbridge.model.tokens",
    unit="{token}",
    description="Tokens a model charged for, attributed to the operation that spent them.",
)
MODEL_TOKEN_ATTRIBUTES = {
    "mindbridge.model.input_tokens": "input",
    "mindbridge.model.output_tokens": "output",
}
"""The span attributes a model adapter reports usage under, and the kind each one counts.

The adapter already measured these; nothing here measures them again. `total_tokens` is
deliberately absent: it is the sum, and the two halves are what price differently.
"""
_model_token_usage: ContextVar[dict[str, int] | None] = ContextVar(
    "mindbridge_model_token_usage",
    default=None,
)
_DURATION_BUCKET_BOUNDARIES_SECONDS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    30,
    60,
    120,
    300,
    600,
    1_200,
    1_800,
    3_600,
    7_200,
    14_400,
    28_800,
    86_400,
    259_200,
    604_800,
)
_CONFIGURATION_LOCK = Lock()
_configured_process: tuple[int, str, TelemetryProviders] | None = None


@dataclass(frozen=True, slots=True)
class TelemetryProviders:
    """Process-owned providers passed to official instrumentation libraries."""

    tracer: TracerProvider | None
    meter: MeterProvider | None

    @property
    def enabled(self) -> bool:
        return self.tracer is not None or self.meter is not None


@asynccontextmanager
async def operation_span(name: str) -> AsyncIterator[None]:
    """Trace and measure one domain operation, as an `async with` block or a decorator.

    `asynccontextmanager` returns an `AsyncContextDecorator`, so `@operation_span(...)` wraps
    an async operation without changing its typed signature.
    """
    if not name.strip():
        raise ValueError("operation name must not be empty")
    started_at = perf_counter()
    outcome = "error"
    # The outermost operation owns the token account, so a nested span cannot drain what its
    # parent is still spending and no total is counted twice on the way out.
    owns_account = _model_token_usage.get() is None
    if owns_account:
        _model_token_usage.set({})
    try:
        with _TRACER.start_as_current_span(name):
            yield
        outcome = "success"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    finally:
        attributes = {"operation": name, "outcome": outcome}
        _OPERATION_CALLS.add(1, attributes)
        _OPERATION_DURATION.record(max(0.0, perf_counter() - started_at), attributes)
        if owns_account:
            for kind, charged in model_token_usage().items():
                _MODEL_TOKENS.add(charged, {"operation": name, "kind": kind})
            # Closed by assignment rather than by resetting a token, because this block can run
            # in a context that never held one: `kernel.watch_observation_job` wraps a yield, so
            # asyncio may finalize it from a task of its own, and `ContextVar.reset` raises
            # there. The account is only ever opened where it was absent, so assigning that back
            # restores exactly what a reset would have.
            _model_token_usage.set(None)


def record_stage_duration(stage: str, duration_seconds: float) -> None:
    """Record one successful, code-defined pipeline milestone without content or IDs."""
    if not stage.strip():
        raise ValueError("stage must not be empty")
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("stage duration must be finite and non-negative")
    _STAGE_DURATION.record(duration_seconds, {"stage": stage})


def current_trace_id() -> str:
    """Return the active W3C trace identity or a standalone fallback identity."""
    context = trace.get_current_span().get_span_context()
    trace_hex = f"{context.trace_id:032x}" if context.is_valid else uuid4().hex
    return f"trace_{trace_hex}"


def set_current_span_attributes(attributes: Mapping[str, str | int | float | bool]) -> None:
    """Attach content-free domain diagnostics, and keep any token charge among them.

    The charge is kept whether or not a span is recording, because what it costs to process an
    observation is not a debugging detail that a sampling decision may drop.
    """
    usage = _model_token_usage.get()
    if usage is not None:
        for attribute, kind in MODEL_TOKEN_ATTRIBUTES.items():
            charged = attributes.get(attribute)
            if isinstance(charged, int) and not isinstance(charged, bool):
                usage[kind] = usage.get(kind, 0) + charged
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(attributes)


def model_token_usage() -> Mapping[str, int]:
    """Return the tokens charged so far inside the operation that owns the token account.

    A caller with somewhere durable to put this -- the job row of the observation being
    processed -- reads it here rather than measuring again. `operation_span` reports the same
    numbers to `mindbridge.model.tokens` when the operation ends, so the two agree by
    construction.
    """
    return dict(_model_token_usage.get() or {})


def configure_telemetry(default_service_name: str) -> TelemetryProviders:
    """Configure OTLP from standard environment variables once per process."""
    if not default_service_name.strip():
        raise ValueError("default_service_name must not be empty")
    if _sdk_disabled(os.environ):
        return TelemetryProviders(tracer=None, meter=None)

    trace_exporter = _selected_exporter(os.environ, "TRACES")
    metric_exporter = _selected_exporter(os.environ, "METRICS")
    if trace_exporter is None and metric_exporter is None:
        return TelemetryProviders(tracer=None, meter=None)

    service_name = os.environ.get("OTEL_SERVICE_NAME", default_service_name).strip()
    if not service_name:
        raise ValueError("OTEL_SERVICE_NAME must not be blank")
    process_id = os.getpid()
    with _CONFIGURATION_LOCK:
        global _configured_process
        if _configured_process is not None and _configured_process[0] == process_id:
            if _configured_process[1] != service_name:
                raise RuntimeError("OpenTelemetry service name changed inside one process")
            return _configured_process[2]

        from opentelemetry.sdk.resources import Resource

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": version("mindbridge"),
            }
        )
        meter_provider = (
            _configure_metrics(resource, metric_exporter) if metric_exporter is not None else None
        )
        tracer_provider = (
            _configure_traces(resource, meter_provider, trace_exporter)
            if trace_exporter is not None
            else None
        )
        providers = TelemetryProviders(tracer=tracer_provider, meter=meter_provider)
        _instrument_clients(providers)
        _configured_process = (process_id, service_name, providers)
        return providers


def instrument_fastapi(app: FastAPI, providers: TelemetryProviders) -> None:
    """Instrument one FastAPI app without collecting headers or body content."""
    if not providers.enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=providers.tracer,
        meter_provider=providers.meter,
        excluded_urls="healthz",
        exclude_spans=["receive", "send"],
    )


def _configure_traces(
    resource: Resource,
    meter_provider: MeterProvider | None,
    exporter: str,
) -> TracerProvider:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return existing
    provider = TracerProvider(resource=resource, meter_provider=meter_provider)
    provider.add_span_processor(BatchSpanProcessor(_span_exporter(exporter)))
    trace.set_tracer_provider(provider)
    return provider


def _configure_metrics(resource: Resource, exporter: str) -> MeterProvider:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

    existing = metrics.get_meter_provider()
    if isinstance(existing, MeterProvider):
        return existing
    reader = PeriodicExportingMetricReader(_metric_exporter(exporter))
    duration_buckets = ExplicitBucketHistogramAggregation(
        boundaries=_DURATION_BUCKET_BOUNDARIES_SECONDS
    )
    provider = MeterProvider(
        resource=resource,
        metric_readers=(reader,),
        views=(
            View(
                instrument_name="mindbridge.operation.duration",
                aggregation=duration_buckets,
            ),
            View(
                instrument_name="mindbridge.stage.duration",
                aggregation=duration_buckets,
            ),
        ),
    )
    metrics.set_meter_provider(provider)
    return provider


def _span_exporter(exporter: str) -> SpanExporter:
    """Build the span exporter one process was configured for."""
    if exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter(formatter=_format_span_line)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def _metric_exporter(exporter: str) -> MetricExporter:
    """Build the metric exporter one process was configured for."""
    if exporter == "console":
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter(formatter=_format_metric_lines)
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    return OTLPMetricExporter()


def _format_span_line(span: ReadableSpan) -> str:
    """Render one finished span as one line, so a console sink stays readable.

    The SDK's own formatter writes each span as a multi-line JSON document, which is fine for a
    file a machine reads and unusable for the case this exists for: an operator following a
    worker's output. Nothing is rendered here that the span does not already carry, and what a
    span carries is content-free by construction -- see the FastAPI test that asserts it.
    """
    started_at, ended_at = span.start_time, span.end_time
    elapsed_seconds = (ended_at - started_at) / 1e9 if started_at and ended_at else 0.0
    return f"{span.name} {elapsed_seconds:.3f}s{_format_attributes(span.attributes)}\n"


def _format_metric_lines(metrics_data: MetricsData) -> str:
    """Render one metric export as one line per data point.

    This is the sink that answers "how long did each stage take, and what did the tokens cost"
    without a collector: the durations are aggregated already, so one export is a few lines
    rather than one per operation.
    """
    return "".join(
        f"{metric.name}{_format_attributes(point.attributes)} {_format_measurement(point)}\n"
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        for point in metric.data.data_points
    )


def _format_attributes(attributes: Mapping[str, object] | None) -> str:
    return "".join(f" {name}={value}" for name, value in (attributes or {}).items())


def _format_measurement(point: object) -> str:
    """Describe one data point without asking which aggregation produced it.

    A counter carries a single value and a histogram carries a distribution; both arrive here,
    and neither the operation duration view nor a future one should have to be enumerated.
    """
    value = getattr(point, "value", None)
    if value is not None:
        return f"value={value}"
    return (
        f"count={getattr(point, 'count', 0)}"
        f" sum={getattr(point, 'sum', 0)}"
        f" max={getattr(point, 'max', 0)}"
    )


def _instrument_clients(providers: TelemetryProviders) -> None:
    keywords = {
        "tracer_provider": providers.tracer,
        "meter_provider": providers.meter,
    }
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

    BotocoreInstrumentor().instrument(**keywords)  # type: ignore[no-untyped-call]
    if "celery" in sys.modules:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument(**keywords)  # type: ignore[no-untyped-call]
    if "httpx" in sys.modules:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(**keywords)
    if "psycopg" in sys.modules:
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument(**keywords)


def _sdk_disabled(environ: Mapping[str, str]) -> bool:
    return environ.get("OTEL_SDK_DISABLED", "false").strip().lower() == "true"


def _selected_exporter(environ: Mapping[str, str], signal: str) -> str | None:
    """Name the exporter one signal is configured for, or None when that signal is off.

    `console` is OpenTelemetry's own value for "write it where this process writes", and it
    needs no endpoint. That is the whole gap this closes: this function used to answer only
    whether an OTLP endpoint was configured, so a box with no collector recorded every duration
    and every token count and then dropped all of it, including when the operator had asked for
    `console` by name. Any other exporter still needs an endpoint, because defaulting to
    localhost:4318 would make every deployment retry a connection nothing is listening on.
    """
    exporter = environ.get(f"OTEL_{signal}_EXPORTER", "otlp").strip().lower()
    if exporter == "console":
        return "console"
    if exporter == "none":
        return None
    endpoint_configured = bool(
        environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        or environ.get(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT", "").strip()
    )
    return "otlp" if endpoint_configured else None
