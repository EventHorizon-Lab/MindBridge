"""Observability boundary: structured logs, domain spans, and runtime setup.

One module owns structured logs and OpenTelemetry so entry points cannot wire one and forget the
other. Console exporters and completion logs keep measurements readable without a collector;
OTLP exports the same instruments when configured.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import math
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

from opentelemetry import metrics, trace

from mindbridge.configuration import configuration_source

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricsData
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

LogValue: TypeAlias = str | int | float | bool | None
"""What a structured log field may hold: no objects, so no content leaks by repr."""

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

_LOGGER_NAME = "mindbridge"
_FIELDS_ATTRIBUTE = "mindbridge_fields"
_LOG_FORMATS = ("json", "text")
_LOGGER = logging.getLogger(_LOGGER_NAME)
_operation_frame: ContextVar[_OperationFrame | None] = ContextVar(
    "mindbridge_operation_frame",
    default=None,
)
"""The innermost running operation of this task: its diagnostics and its nested time.

Domain code reports diagnostics once, through `set_current_span_attributes`. A span drops
them unless an SDK is installed and sampling, so the same call also files them here, which
is what lets the completion log line carry them with no exporter configured. Being a
context variable is what keeps concurrent operations from reading each other's: every
`asyncio` task starts from a copy of the context, and the frame object it copies is shared
with the parent, which is how a gathered child still bills its time upward.
"""
_TIMINGS_LOCK = Lock()
_timings: dict[str, _OperationTiming] = {}
_summary_registered = False


@dataclass(frozen=True, slots=True)
class TelemetryProviders:
    """Process-owned providers passed to official instrumentation libraries."""

    tracer: TracerProvider | None
    meter: MeterProvider | None

    @property
    def enabled(self) -> bool:
        return self.tracer is not None or self.meter is not None


@dataclass(slots=True)
class _OperationFrame:
    """One running operation's reported diagnostics and the time its children took."""

    fields: dict[str, LogValue]
    child_seconds: float = 0.0


@dataclass(slots=True)
class _OperationTiming:
    """Where one operation's wall clock went, accumulated for this process."""

    calls: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    self_seconds: float = 0.0
    max_seconds: float = 0.0

    def observe(self, duration_seconds: float, self_seconds: float, *, failed: bool) -> None:
        self.calls += 1
        self.failures += int(failed)
        self.total_seconds += duration_seconds
        self.self_seconds += self_seconds
        self.max_seconds = max(self.max_seconds, duration_seconds)


@dataclass(frozen=True, slots=True)
class OperationTotals:
    """One row of the process timing summary.

    `total_seconds` is inclusive and answers "how long does this take end to end", which is
    what an SLO is written against. `self_seconds` excludes nested instrumented operations
    and answers "where is the time actually spent", which is what an optimization needs:
    ranked by inclusive time, the outermost operation always wins and says nothing.

    An operation that only gathers other instrumented operations reports near-zero
    `self_seconds` by construction. That is the intended reading -- it has no time of its own
    to remove -- not a missing measurement.
    """

    operation: str
    calls: int
    failures: int
    total_seconds: float
    self_seconds: float
    max_seconds: float

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


@asynccontextmanager
async def operation_span(name: str) -> AsyncIterator[None]:
    """Trace, measure, and log one domain operation, as an `async with` block or a decorator.

    `asynccontextmanager` returns an `AsyncContextDecorator`, so `@operation_span(...)` wraps
    an async operation without changing its typed signature.

    Every instrumented operation reports its own duration and outcome on completion, so the
    write and read paths are attributable from stderr alone. That is the point of logging
    here rather than only recording a histogram: a histogram needs a collector to read, and
    the operation that is slow is the one nobody stood a collector up for yet.
    """
    if not name.strip():
        raise ValueError("operation name must not be empty")
    frame = _OperationFrame(fields={})
    parent_frame = _operation_frame.get()
    _operation_frame.set(frame)
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
        # An async generator may finish in a task that did not create this frame. Assignment is
        # context-independent; resetting the creator's ContextVar token is not.
        _operation_frame.set(parent_frame)
        duration_seconds = max(0.0, perf_counter() - started_at)
        if parent_frame is not None:
            parent_frame.child_seconds += duration_seconds
        # Children are billed by the sum of their durations, which overlap under
        # `asyncio.gather` and can exceed the parent's own wall clock. Clamping keeps self
        # time non-negative, at the price that a parent gathering more child-seconds than it
        # ran reports zero rather than the sliver of orchestration it really spent. That is
        # the right trade for finding a bottleneck -- the seconds are attributed to the
        # leaves that actually consumed them -- and `duration_ms` still carries the whole.
        self_seconds = max(0.0, duration_seconds - frame.child_seconds)
        attributes = {"operation": name, "outcome": outcome}
        _OPERATION_CALLS.add(1, attributes)
        _OPERATION_DURATION.record(duration_seconds, attributes)
        _observe_timing(name, duration_seconds, self_seconds, failed=outcome != "success")
        # Merged rather than splatted: a reported attribute happening to be named `operation`
        # would make `**` raise TypeError, and raising from this `finally` would replace the
        # operation's own failure with an unrelated one.
        _LOGGER.log(
            logging.INFO if outcome == "success" else logging.WARNING,
            "operation %s",
            outcome,
            extra={
                _FIELDS_ATTRIBUTE: {
                    **frame.fields,
                    "operation": name,
                    "outcome": outcome,
                    "duration_ms": round(duration_seconds * 1_000, 3),
                    "self_ms": round(self_seconds * 1_000, 3),
                }
            },
        )
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
    _LOGGER.info(
        "stage reached",
        extra=log_fields(stage=stage, duration_ms=round(duration_seconds * 1_000, 3)),
    )


def log_fields(**fields: LogValue) -> dict[str, object]:
    """Build the `extra` payload every MindBridge log record carries its diagnostics in.

    One reserved key holds them all, because a flat `extra` silently loses any field whose
    name collides with a `LogRecord` attribute -- `message` and `name` among them.
    """
    return {_FIELDS_ATTRIBUTE: dict(fields)}


def logger(name: str) -> logging.Logger:
    """Return the module logger for `name` inside the configured MindBridge namespace."""
    if not name.startswith(f"{_LOGGER_NAME}."):
        raise ValueError("MindBridge loggers must live under the mindbridge namespace")
    return logging.getLogger(name)


def timing_summary() -> tuple[OperationTotals, ...]:
    """Return this process's operations ordered by the self time they account for.

    Ordering by self rather than by total is the whole point: ranked by inclusive time the
    outermost operation always wins because it contains everything, so the row at the top
    would never be the one to go and optimize. Both columns are reported either way.
    """
    with _TIMINGS_LOCK:
        snapshot = tuple(
            OperationTotals(
                operation=name,
                calls=timing.calls,
                failures=timing.failures,
                total_seconds=timing.total_seconds,
                self_seconds=timing.self_seconds,
                max_seconds=timing.max_seconds,
            )
            for name, timing in _timings.items()
        )
    return tuple(sorted(snapshot, key=lambda row: row.self_seconds, reverse=True))


def reset_timings() -> None:
    """Discard accumulated timings, so one process can measure phases separately."""
    with _TIMINGS_LOCK:
        _timings.clear()


def flush_timing_summary() -> None:
    """Emit the summary now, for a process whose exit never reaches `atexit`.

    Celery's prefork children end in `os._exit` -- billiard calls it directly from
    `Worker._do_exit` -- so the handler registered at configuration time never runs there.
    The worker is the process whose timings matter most, being where the write path spends
    its wall clock, so leaving it to `atexit` is leaving it unmeasured. Unregisters as it
    goes, so a process that both flushes and exits normally still reports once.
    """
    global _summary_registered
    if not _summary_registered:
        return
    atexit.unregister(log_timing_summary)
    _summary_registered = False
    log_timing_summary()


def log_timing_summary() -> None:
    """Emit the accumulated timing summary as one log record per operation."""
    rows = timing_summary()
    if not rows:
        return
    measured_seconds = sum(row.self_seconds for row in rows)
    _LOGGER.info(
        "timing summary",
        extra=log_fields(
            operation_count=len(rows),
            measured_seconds=round(measured_seconds, 3),
        ),
    )
    for rank, row in enumerate(rows, start=1):
        _LOGGER.info(
            "timing",
            extra=log_fields(
                rank=rank,
                operation=row.operation,
                calls=row.calls,
                failures=row.failures,
                self_seconds=round(row.self_seconds, 3),
                total_seconds=round(row.total_seconds, 3),
                mean_ms=round(row.mean_seconds * 1_000, 3),
                max_ms=round(row.max_seconds * 1_000, 3),
                self_share=(
                    round(row.self_seconds / measured_seconds, 4) if measured_seconds > 0 else 0.0
                ),
            ),
        )


def _observe_timing(
    name: str,
    duration_seconds: float,
    self_seconds: float,
    *,
    failed: bool,
) -> None:
    with _TIMINGS_LOCK:
        _timings.setdefault(name, _OperationTiming()).observe(
            duration_seconds,
            self_seconds,
            failed=failed,
        )


def current_trace_id() -> str:
    """Return the active W3C trace identity or a standalone fallback identity."""
    context = trace.get_current_span().get_span_context()
    trace_hex = f"{context.trace_id:032x}" if context.is_valid else uuid4().hex
    return f"trace_{trace_hex}"


def set_current_span_attributes(attributes: Mapping[str, str | int | float | bool]) -> None:
    """Attach diagnostics to the span and completion log, retaining any token charge."""
    frame = _operation_frame.get()
    if frame is not None:
        frame.fields.update(attributes)
    usage = _model_token_usage.get()
    if usage is not None:
        for attribute, kind in MODEL_TOKEN_ATTRIBUTES.items():
            charged = attributes.get(attribute)
            if isinstance(charged, int) and not isinstance(charged, bool):
                usage[kind] = usage.get(kind, 0) + charged
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(attributes)


def configure_observability(
    default_service_name: str,
    environ: Mapping[str, str] | None = None,
) -> TelemetryProviders:
    """Install this process's logs and exporters together.

    Entry points call this instead of `configure_telemetry` so that a process can never
    export spans while staying silent on stderr.
    """
    configure_logging(default_service_name, environ)
    return configure_telemetry(default_service_name)


def configure_logging(
    default_service_name: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Install one stderr handler on the MindBridge logger namespace.

    Only the `mindbridge` namespace is configured, never the root logger: this package is
    imported as a library as well as run as a service, and taking the root logger would
    reformat the logs of whichever host process embedded it.
    """
    if not default_service_name.strip():
        raise ValueError("default_service_name must not be empty")
    source = configuration_source() if environ is None else environ
    service_name = source.get("OTEL_SERVICE_NAME", default_service_name).strip()
    if not service_name:
        raise ValueError("OTEL_SERVICE_NAME must not be blank")
    level = _log_level(source)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _JsonFormatter(service_name)
        if _log_format(source) == "json"
        else _TextFormatter(service_name)
    )
    handler.addFilter(_TraceContextFilter())
    for existing in tuple(_LOGGER.handlers):
        _LOGGER.removeHandler(existing)
        existing.close()
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)
    # Records stop here rather than reaching the root logger, which would print each line a
    # second time under any host that configured `basicConfig`.
    _LOGGER.propagate = False
    global _summary_registered
    if _summary_requested(source) and not _summary_registered:
        # Celery reconfigures every forked child, so registering per call would emit the
        # summary once per configuration rather than once per process.
        atexit.register(log_timing_summary)
        _summary_registered = True


def _log_level(source: Mapping[str, str]) -> int:
    name = source.get("MINDBRIDGE_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        raise ValueError(f"MINDBRIDGE_LOG_LEVEL is not a logging level: {name}")
    return level


def _log_format(source: Mapping[str, str]) -> str:
    configured = source.get("MINDBRIDGE_LOG_FORMAT", "").strip().lower()
    if not configured:
        # A terminal gets the readable form and a pipe gets the parseable one, because the
        # wrong guess is only ever a cosmetic surprise -- and the variable overrides it.
        return "text" if sys.stderr.isatty() else "json"
    if configured not in _LOG_FORMATS:
        raise ValueError(f"MINDBRIDGE_LOG_FORMAT must be one of {_LOG_FORMATS}: {configured}")
    return configured


def _summary_requested(source: Mapping[str, str]) -> bool:
    return source.get("MINDBRIDGE_TIMING_SUMMARY", "").strip().lower() in {"1", "true", "yes"}


class _TraceContextFilter(logging.Filter):
    """Stamp each record with the active W3C identity so logs join traces."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            record.trace_id = f"{context.trace_id:032x}"
            record.span_id = f"{context.span_id:016x}"
        return True


class _JsonFormatter(logging.Formatter):
    """Render one machine-readable object per record, without user content."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        # Reported fields go in first so the record's own identity overwrites them rather
        # than the other way round: `log_fields` keeps a domain key from colliding with a
        # `LogRecord` attribute, and this keeps one named `level` or `message` from
        # rewriting the line that carries it.
        payload: dict[str, object] = dict(_record_fields(record))
        payload.update(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "service": self._service_name,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )
        for attribute in ("trace_id", "span_id"):
            value = getattr(record, attribute, None)
            if value is not None:
                payload[attribute] = value
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)


class _TextFormatter(logging.Formatter):
    """Render one aligned line per record for a developer reading a terminal."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        fields = _record_fields(record)
        stamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        # Duration leads the rendered fields: reading a terminal, the number being looked for
        # is almost always how long the line's operation took.
        ordered = sorted(fields.items(), key=lambda item: (item[0] != "duration_ms", item[0]))
        rendered = " ".join(f"{key}={value}" for key, value in ordered)
        line = f"{stamp} {record.levelname:<7} {record.name} {record.getMessage()}"
        if rendered:
            line = f"{line} | {rendered}"
        if record.exc_info is not None:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _record_fields(record: logging.LogRecord) -> dict[str, LogValue]:
    fields = getattr(record, _FIELDS_ATTRIBUTE, None)
    return dict(fields) if isinstance(fields, dict) else {}


def record_output_repairs(attributes: Mapping[str, int]) -> None:
    """Report what a stage discarded or rewrote, on a sink no collector is needed to read.

    Span attributes are the wrong and only sink these had. A span keeps them while it is
    recording, and the deployment they matter on is the one that samples a tenth of its traces
    into a collector it does not run -- so nine of ten of these vanish and the tenth goes
    nowhere. What survived that is a job marked SUCCEEDED and a low census, which reads exactly
    the same whether the model said little or this process threw most of it away.

    So the counts also go to `logging`, at WARNING because that is the level an unconfigured
    process still prints: `logging.lastResort` writes it to stderr, and every deployed process
    here already has its output somewhere an operator looks. This is the token charge's rule
    (`set_current_span_attributes`) applied to the other measurement a sampling decision must
    not be allowed to lose.

    A zero still reaches the trace, because "nothing was discarded" is what makes a count that
    is not zero mean anything; only a repair that happened is worth waking a log up for.
    """
    set_current_span_attributes(attributes)
    if any(attributes.values()):
        _LOGGER.warning("model output repaired:%s", _format_attributes(attributes))


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
