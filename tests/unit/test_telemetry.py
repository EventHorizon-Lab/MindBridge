"""OpenTelemetry configuration and privacy-safe trace identity checks."""

import os
import subprocess
import sys
from typing import cast

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, TraceFlags, use_span

from mindbridge.api import TenantApiKeyAuthenticator, create_app
from mindbridge.application.kernel import MemoryKernel
from mindbridge.telemetry import (
    TelemetryProviders,
    current_trace_id,
    instrument_fastapi,
    trace_operation,
)


async def test_domain_operation_returns_the_active_w3c_trace_identity() -> None:
    context = SpanContext(
        trace_id=0x1234,
        span_id=0x5678,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    @trace_operation("mindbridge.test")
    async def operation() -> str:
        return current_trace_id()

    with use_span(NonRecordingSpan(context)):
        result = await operation()

    assert result == "trace_00000000000000000000000000001234"


def test_fastapi_returns_trace_identity_without_capturing_secret_content() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(
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
