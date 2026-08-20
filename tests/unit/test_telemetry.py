"""Observability configuration, timing attribution, and privacy-safe trace identity."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from io import StringIO
from typing import cast

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, TraceFlags, use_span

from mindbridge import telemetry
from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.application.kernel import MemoryKernel
from mindbridge.telemetry import (
    TelemetryProviders,
    current_trace_id,
    instrument_fastapi,
    operation_span,
    record_stage_duration,
)


class RecordingMetric:
    def __init__(self) -> None:
        self.measurements: list[tuple[float, dict[str, str]]] = []

    def add(self, value: int, attributes: Mapping[str, str] | None = None) -> None:
        self.measurements.append((value, dict(attributes or {})))

    def record(self, value: float, attributes: Mapping[str, str] | None = None) -> None:
        self.measurements.append((value, dict(attributes or {})))


async def test_domain_operation_returns_the_active_w3c_trace_identity() -> None:
    context = SpanContext(
        trace_id=0x1234,
        span_id=0x5678,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    @operation_span("mindbridge.test")
    async def operation() -> str:
        return current_trace_id()

    with use_span(NonRecordingSpan(context)):
        result = await operation()

    assert result == "trace_00000000000000000000000000001234"


async def test_domain_operation_records_content_free_slo_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = RecordingMetric()
    durations = RecordingMetric()
    monkeypatch.setattr(telemetry, "_OPERATION_CALLS", calls)
    monkeypatch.setattr(telemetry, "_OPERATION_DURATION", durations)

    @operation_span("mindbridge.test.metrics")
    async def succeed() -> str:
        return "private-memory-content"

    @operation_span("mindbridge.test.metrics")
    async def fail() -> None:
        raise RuntimeError("private-error-content")

    assert await succeed() == "private-memory-content"
    with pytest.raises(RuntimeError, match="private-error-content"):
        await fail()

    assert [attributes for _, attributes in calls.measurements] == [
        {"operation": "mindbridge.test.metrics", "outcome": "success"},
        {"operation": "mindbridge.test.metrics", "outcome": "error"},
    ]
    assert len(durations.measurements) == 2
    assert all(value >= 0 for value, _ in durations.measurements)
    assert "private" not in repr(calls.measurements + durations.measurements)


async def test_one_decorated_operation_serves_repeated_and_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`@operation_span` builds one context manager that every later call must reuse."""
    calls = RecordingMetric()
    monkeypatch.setattr(telemetry, "_OPERATION_CALLS", calls)

    @operation_span("mindbridge.test.reuse")
    async def operation(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    assert await operation(1) == 2
    assert await operation(2) == 4
    assert await asyncio.gather(*(operation(value) for value in range(4))) == [0, 2, 4, 6]
    assert len(calls.measurements) == 6


def test_stage_duration_records_only_a_code_defined_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durations = RecordingMetric()
    monkeypatch.setattr(telemetry, "_STAGE_DURATION", durations)

    record_stage_duration("recall.first_answer", 0.125)

    assert durations.measurements == [(0.125, {"stage": "recall.first_answer"})]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_stage_duration_rejects_invalid_measurements(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        record_stage_duration("recall.first_answer", value)


def test_duration_buckets_cover_multi_round_and_offline_tail() -> None:
    assert 7_200 in telemetry._DURATION_BUCKET_BOUNDARIES_SECONDS
    assert telemetry._DURATION_BUCKET_BOUNDARIES_SECONDS[-1] >= 7 * 24 * 60 * 60


def test_fastapi_returns_trace_identity_without_capturing_secret_content() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator(
            {"tenant_01": ("tenant-api-key-000000000000000000",)}
        ),
    )
    instrument_fastapi(app, TelemetryProviders(tracer=provider, meter=None))

    with TestClient(app) as client:
        response = client.post(
            "/v1/recall",
            headers={"Authorization": "Bearer tenant-api-key-000000000000000000"},
            content='{"query":"private-memory-content"}',
        )

    server_span = next(
        span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER
    )
    attributes = repr(server_span.attributes)
    assert response.status_code == 422
    assert response.json()["trace_id"] == f"trace_{server_span.context.trace_id:032x}"
    assert "tenant-api-key" not in attributes
    assert "private-memory-content" not in attributes
    provider.shutdown()


def test_otlp_runtime_configures_in_an_isolated_process() -> None:
    environment = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://127.0.0.1:9/v1/traces",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_SDK_DISABLED": "false",
        "OTEL_SERVICE_NAME": "mindbridge-test",
        "OTEL_TRACES_EXPORTER": "otlp",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mindbridge.worker; "
                "from opentelemetry.instrumentation.botocore import BotocoreInstrumentor; "
                "from mindbridge.telemetry import configure_telemetry; "
                "providers = configure_telemetry('mindbridge-test'); "
                "assert providers.tracer is not None and providers.meter is None; "
                "assert BotocoreInstrumentor().is_instrumented_by_opentelemetry"
            ),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def _configure_capture(
    monkeypatch: pytest.MonkeyPatch,
    **environment: str,
) -> StringIO:
    """Configure the MindBridge logger for one test and hand back what it writes to."""
    monkeypatch.setattr(telemetry, "_timings", {})
    telemetry.configure_logging("mindbridge-test", {"MINDBRIDGE_LOG_FORMAT": "json", **environment})
    captured = StringIO()
    monkeypatch.setattr(telemetry._LOGGER.handlers[0], "stream", captured)
    return captured


def _records(captured: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in captured.getvalue().splitlines() if line.strip()]


async def test_operation_logs_its_duration_and_reported_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completion line is the only reachable timing when no collector is configured."""
    captured = _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.logged")
    async def operation() -> None:
        telemetry.set_current_span_attributes({"mindbridge.event.count": 7})

    await operation()

    (record,) = _records(captured)
    assert record["level"] == "INFO"
    assert record["operation"] == "mindbridge.test.logged"
    assert record["outcome"] == "success"
    assert record["mindbridge.event.count"] == 7
    assert cast(float, record["duration_ms"]) >= 0.0


async def test_failed_operation_logs_at_warning_with_its_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.failing")
    async def operation() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await operation()

    (record,) = _records(captured)
    assert record["level"] == "WARNING"
    assert record["outcome"] == "error"


async def test_nested_operations_do_not_share_reported_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inner operation's attributes must not be attributed to its caller, or the reverse."""
    captured = _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.inner")
    async def inner() -> None:
        telemetry.set_current_span_attributes({"mindbridge.inner": 1})

    @operation_span("mindbridge.test.outer")
    async def outer() -> None:
        telemetry.set_current_span_attributes({"mindbridge.outer": 1})
        await inner()
        # Reporting after a nested call is the common shape -- `process_observation` counts
        # its events only once perception has returned -- so the caller's frame has to be
        # restored by then rather than still pointing at the operation that finished.
        telemetry.set_current_span_attributes({"mindbridge.after_inner": 1})

    await outer()

    inner_record, outer_record = _records(captured)
    assert inner_record["operation"] == "mindbridge.test.inner"
    assert "mindbridge.outer" not in inner_record
    assert outer_record["operation"] == "mindbridge.test.outer"
    assert "mindbridge.inner" not in outer_record
    assert outer_record["mindbridge.after_inner"] == 1


async def test_concurrent_operations_do_not_share_reported_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.concurrent")
    async def operation(index: int) -> None:
        telemetry.set_current_span_attributes({"mindbridge.index": index})
        await asyncio.sleep(0)

    await asyncio.gather(*(operation(index) for index in range(3)))

    assert sorted(cast(int, record["mindbridge.index"]) for record in _records(captured)) == [
        0,
        1,
        2,
    ]


async def test_self_time_excludes_nested_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ranking by inclusive time names the outermost operation, which explains nothing."""
    _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.leaf")
    async def leaf() -> None:
        await asyncio.sleep(0.02)

    @operation_span("mindbridge.test.orchestrator")
    async def orchestrator() -> None:
        await leaf()

    await orchestrator()

    summary = {row.operation: row for row in telemetry.timing_summary()}
    orchestration = summary["mindbridge.test.orchestrator"]
    assert orchestration.total_seconds >= summary["mindbridge.test.leaf"].total_seconds
    assert orchestration.self_seconds < summary["mindbridge.test.leaf"].self_seconds
    assert telemetry.timing_summary()[0].operation == "mindbridge.test.leaf"


async def test_timing_summary_counts_calls_and_failures_per_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.counted")
    async def operation(fail: bool) -> None:
        if fail:
            raise RuntimeError("boom")

    await operation(False)
    await operation(False)
    with pytest.raises(RuntimeError):
        await operation(True)

    (row,) = telemetry.timing_summary()
    assert (row.calls, row.failures) == (3, 1)


async def test_logged_operation_carries_the_active_trace_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reported trace ID is only useful if the logs carry the same one."""
    captured = _configure_capture(monkeypatch)
    context = SpanContext(
        trace_id=0xABC,
        span_id=0xDEF,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    @operation_span("mindbridge.test.correlated")
    async def operation() -> None:
        return None

    with use_span(NonRecordingSpan(context)):
        await operation()

    (record,) = _records(captured)
    assert record["trace_id"] == "00000000000000000000000000000abc"
    assert record["span_id"] == "0000000000000def"


def test_stage_duration_logs_the_milestone_it_records(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _configure_capture(monkeypatch)

    record_stage_duration("recall.first_answer", 0.25)

    (record,) = _records(captured)
    assert record["stage"] == "recall.first_answer"
    assert record["duration_ms"] == 250.0


def test_reconfiguring_logging_does_not_duplicate_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery reconfigures each forked child, so a second call must replace the handler."""
    _configure_capture(monkeypatch)
    captured = _configure_capture(monkeypatch)

    telemetry._LOGGER.warning("once")

    assert len(telemetry._LOGGER.handlers) == 1
    assert len(_records(captured)) == 1


def test_logging_leaves_the_root_logger_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """MindBridge is embedded as a library, so it must not reformat a host's own logs."""
    root_handlers = tuple(logging.getLogger().handlers)

    _configure_capture(monkeypatch)

    assert tuple(logging.getLogger().handlers) == root_handlers
    assert telemetry._LOGGER.propagate is False


def test_text_logs_lead_with_the_duration_a_reader_is_looking_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _configure_capture(monkeypatch, MINDBRIDGE_LOG_FORMAT="text")

    record_stage_duration("recall.answer_complete", 1.5)

    rendered = captured.getvalue()
    assert "| duration_ms=1500.0 stage=recall.answer_complete" in rendered


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("MINDBRIDGE_LOG_LEVEL", "CHATTY", "not a logging level"),
        ("MINDBRIDGE_LOG_FORMAT", "yaml", "must be one of"),
    ],
)
def test_logging_refuses_an_unusable_configuration(
    variable: str,
    value: str,
    message: str,
) -> None:
    """A silent fallback to INFO or JSON is worse to debug than refusing to start."""
    with pytest.raises(ValueError, match=message):
        telemetry.configure_logging("mindbridge-test", {variable: value})


def test_log_level_suppresses_the_operation_stream_it_excludes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _configure_capture(monkeypatch, MINDBRIDGE_LOG_LEVEL="WARNING")

    record_stage_duration("recall.first_answer", 0.25)
    telemetry._LOGGER.warning("kept")

    assert [record["message"] for record in _records(captured)] == ["kept"]


async def test_a_reported_attribute_named_like_a_log_field_does_not_break_the_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completion line is built in a `finally`, so a collision there would replace the
    operation's own outcome with an unrelated TypeError."""
    captured = _configure_capture(monkeypatch)

    @operation_span("mindbridge.test.colliding")
    async def operation() -> None:
        telemetry.set_current_span_attributes({"operation": "reported", "duration_ms": -1})

    await operation()

    (record,) = _records(captured)
    assert record["operation"] == "mindbridge.test.colliding"
    assert cast(float, record["duration_ms"]) >= 0.0


def test_the_timing_summary_is_registered_once_per_process_not_per_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery reconfigures every forked child, which would print one summary per call."""
    registered: list[object] = []
    monkeypatch.setattr("atexit.register", registered.append)
    monkeypatch.setattr(telemetry, "_summary_registered", False)
    environment = {"MINDBRIDGE_LOG_FORMAT": "json", "MINDBRIDGE_TIMING_SUMMARY": "1"}

    telemetry.configure_logging("mindbridge-test", environment)
    telemetry.configure_logging("mindbridge-test", environment)

    assert registered == [telemetry.log_timing_summary]


async def test_the_summary_is_reachable_without_atexit_for_a_child_that_os_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Celery's prefork child ends in `os._exit`, which runs no `atexit` handler.

    That child is the worker, where the write path spends its wall clock, so leaving the
    summary to `atexit` leaves the one process worth profiling unmeasured.
    """
    monkeypatch.setattr("atexit.register", lambda _handler: None)
    monkeypatch.setattr(telemetry, "_summary_registered", False)
    captured = _configure_capture(monkeypatch, MINDBRIDGE_TIMING_SUMMARY="1")

    @operation_span("mindbridge.test.flushed")
    async def operation() -> None:
        return None

    await operation()
    telemetry.flush_timing_summary()

    messages = [record["message"] for record in _records(captured)]
    assert "timing summary" in messages


def test_flushing_the_summary_twice_reports_it_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that flushes and then exits normally must not print the summary twice."""
    unregistered: list[object] = []
    monkeypatch.setattr("atexit.register", lambda _handler: None)
    monkeypatch.setattr("atexit.unregister", unregistered.append)
    monkeypatch.setattr(telemetry, "_summary_registered", True)
    emitted: list[int] = []
    monkeypatch.setattr(telemetry, "log_timing_summary", lambda: emitted.append(1))

    telemetry.flush_timing_summary()
    telemetry.flush_timing_summary()

    assert emitted == [1]
    assert unregistered == [telemetry.log_timing_summary]


def test_a_reported_field_cannot_rewrite_the_record_it_travels_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`log_fields` keeps a domain key off `LogRecord`; this keeps it off the payload too."""
    captured = _configure_capture(monkeypatch)

    telemetry.logger("mindbridge.test").info(
        "real message",
        extra=telemetry.log_fields(message="spoofed", level="DEBUG", service="elsewhere"),
    )

    (record,) = _records(captured)
    assert record["message"] == "real message"
    assert record["level"] == "INFO"
    assert record["service"] == "mindbridge-test"


def test_the_timing_summary_stays_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[object] = []
    monkeypatch.setattr("atexit.register", registered.append)
    monkeypatch.setattr(telemetry, "_summary_registered", False)

    telemetry.configure_logging("mindbridge-test", {"MINDBRIDGE_LOG_FORMAT": "json"})

    assert registered == []
