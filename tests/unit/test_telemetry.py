"""OpenTelemetry configuration and privacy-safe trace identity checks."""

import asyncio
import os
import subprocess
import sys
from collections.abc import Mapping
from typing import cast

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
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


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"OTEL_TRACES_EXPORTER": "console"}, "console"),
        ({"OTEL_TRACES_EXPORTER": "CONSOLE "}, "console"),
        ({"OTEL_TRACES_EXPORTER": "none", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://c:4318"}, None),
        ({"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://c:4318/v1/traces"}, "otlp"),
        # The gap this closes: no endpoint and no console asked for means nothing is recorded,
        # which is the only case where dropping a measurement is the operator's own choice.
        ({}, None),
    ],
)
def test_console_needs_no_endpoint_while_otlp_still_does(
    environ: dict[str, str],
    expected: str | None,
) -> None:
    assert telemetry._selected_exporter(environ, "TRACES") == expected


def test_console_span_line_carries_the_duration_and_the_attributes() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("mindbridge.test.console") as span:
        span.set_attribute("mindbridge.model.input_tokens", 41)

    line = telemetry._format_span_line(exporter.get_finished_spans()[0])

    assert line.count("\n") == 1
    assert line.startswith("mindbridge.test.console 0.")
    assert line.rstrip().endswith("mindbridge.model.input_tokens=41")
    provider.shutdown()


def test_console_metric_lines_describe_counters_and_histograms() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=(reader,))
    meter = provider.get_meter("test")
    meter.create_counter("mindbridge.test.tokens").add(7, {"kind": "input"})
    meter.create_histogram("mindbridge.test.duration").record(0.5, {"stage": "test.console"})
    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None

    lines = telemetry._format_metric_lines(metrics_data).splitlines()

    assert "mindbridge.test.tokens kind=input value=7" in lines
    assert "mindbridge.test.duration stage=test.console count=1 sum=0.5 max=0.5" in lines
    provider.shutdown()


_CONSOLE_RUNTIME_PROGRAM = """
import asyncio

from mindbridge.telemetry import configure_telemetry, operation_span, record_stage_duration

providers = configure_telemetry("mindbridge-test")
assert providers.tracer is not None and providers.meter is not None


async def main() -> None:
    async with operation_span("mindbridge.test.console"):
        record_stage_duration("test.console", 0.25)


asyncio.run(main())
providers.tracer.shutdown()
providers.meter.shutdown()
"""


def test_console_runtime_prints_measurements_with_no_collector_configured() -> None:
    """A box with no OTLP endpoint must still be able to read what it measured."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("OTEL_")
    } | {
        "OTEL_METRICS_EXPORTER": "console",
        "OTEL_SDK_DISABLED": "false",
        "OTEL_SERVICE_NAME": "mindbridge-test",
        "OTEL_TRACES_EXPORTER": "console",
    }
    result = subprocess.run(
        [sys.executable, "-c", _CONSOLE_RUNTIME_PROGRAM],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "mindbridge.test.console 0." in result.stdout
    assert "mindbridge.stage.duration stage=test.console count=1 sum=0.25" in result.stdout
    assert "mindbridge.operation.duration operation=mindbridge.test.console" in result.stdout


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
