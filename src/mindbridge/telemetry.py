"""Small OpenTelemetry boundary for runtime setup and domain spans."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
from threading import Lock
from typing import TYPE_CHECKING, ParamSpec, TypeVar
from uuid import uuid4

from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_TRACER = trace.get_tracer("mindbridge", version("mindbridge"))
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


def trace_operation(
    name: str,
) -> Callable[
    [Callable[_Parameters, Coroutine[object, object, _Result]]],
    Callable[_Parameters, Coroutine[object, object, _Result]],
]:
    """Trace one async domain operation without changing its typed signature."""

    def decorate(
        operation: Callable[_Parameters, Coroutine[object, object, _Result]],
    ) -> Callable[_Parameters, Coroutine[object, object, _Result]]:
        @wraps(operation)
        async def traced(
            *args: _Parameters.args,
            **kwargs: _Parameters.kwargs,
        ) -> _Result:
            with _TRACER.start_as_current_span(name):
                return await operation(*args, **kwargs)

        return traced

    return decorate


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

    existing = metrics.get_meter_provider()
    if isinstance(existing, MeterProvider):
        return existing
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    provider = MeterProvider(resource=resource, metric_readers=(reader,))
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
