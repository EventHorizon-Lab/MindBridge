"""Small OpenTelemetry boundary for runtime setup and domain spans."""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
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
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

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
    """Attach content-free domain diagnostics when the active span is sampled."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(attributes)


def configure_telemetry(default_service_name: str) -> TelemetryProviders:
    """Configure OTLP from standard environment variables once per process."""
    if not default_service_name.strip():
        raise ValueError("default_service_name must not be empty")
    if _sdk_disabled(os.environ):
        return TelemetryProviders(tracer=None, meter=None)

    trace_enabled = _signal_enabled(os.environ, "TRACES")
    metrics_enabled = _signal_enabled(os.environ, "METRICS")
    if not trace_enabled and not metrics_enabled:
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
        meter_provider = _configure_metrics(resource) if metrics_enabled else None
        tracer_provider = _configure_traces(resource, meter_provider) if trace_enabled else None
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
) -> TracerProvider:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return existing
    provider = TracerProvider(resource=resource, meter_provider=meter_provider)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def _configure_metrics(resource: Resource) -> MeterProvider:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

    existing = metrics.get_meter_provider()
    if isinstance(existing, MeterProvider):
        return existing
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
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


def _signal_enabled(environ: Mapping[str, str], signal: str) -> bool:
    exporter = environ.get(f"OTEL_{signal}_EXPORTER", "otlp").strip().lower()
    return exporter != "none" and bool(
        environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        or environ.get(f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT", "").strip()
    )
