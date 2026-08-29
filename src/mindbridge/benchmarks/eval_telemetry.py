"""Bounded in-process aggregation for evaluation telemetry spans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    GEN_AI_TTFC,
    MODEL_MODULE,
    MODEL_REQUEST_COUNT,
    MODEL_TTFT,
    SPAN_KIND,
    TOKEN_AUDIO_SECONDS,
    TOKEN_COMPLETE,
    TOKEN_EXPECTED_REQUEST_COUNT,
    TOKEN_MODALITIES,
    TOKEN_REPORTED_REQUEST_COUNT,
    TOKEN_TOTAL,
    TRACER_NAME,
    token_modality_attribute,
)

BENCHMARK_TASK = "mindbridge.benchmark.task"
BENCHMARK_TASK_SPAN = "mindbridge.benchmark.run"
BENCHMARK_JUDGE_SPAN = "mindbridge.benchmark.judge"


@dataclass(slots=True)
class _Durations:
    count: int = 0
    total_ns: int = 0
    ttft_count: int = 0
    ttft_seconds: float = 0.0
    ttfc_count: int = 0
    ttfc_seconds: float = 0.0

    def add(self, span: ReadableSpan) -> None:
        self.count += 1
        self.total_ns += _duration_ns(span)
        attributes = span.attributes or {}
        ttft = _float_attribute(attributes, MODEL_TTFT)
        if ttft is not None:
            self.ttft_count += 1
            self.ttft_seconds += ttft
        ttfc = _float_attribute(attributes, GEN_AI_TTFC)
        if ttfc is not None:
            self.ttfc_count += 1
            self.ttfc_seconds += ttfc

    def json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "count": self.count,
            "total_seconds": self.total_ns / 1_000_000_000,
            "average_ms": self.total_ns / self.count / 1_000_000,
        }
        if self.ttft_count:
            result["ttft_ms"] = {
                "count": self.ttft_count,
                "average": self.ttft_seconds / self.ttft_count * 1_000,
            }
        if self.ttfc_count:
            result["time_to_first_chunk_ms"] = {
                "count": self.ttfc_count,
                "average": self.ttfc_seconds / self.ttfc_count * 1_000,
            }
        return result


@dataclass(slots=True)
class _Tokens:
    complete: bool = True
    request_count: int = 0
    expected_request_count: int = 0
    reported_request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    audio_seconds: float = 0.0
    calls_by_input_modality: dict[str, int] = field(default_factory=dict)
    input_by_modality: dict[str, int] = field(default_factory=dict)
    output_by_modality: dict[str, int] = field(default_factory=dict)

    def add(self, attributes: Mapping[str, AttributeValue]) -> None:
        requests = _int_attribute(attributes, MODEL_REQUEST_COUNT) or 0
        expected = _int_attribute(attributes, TOKEN_EXPECTED_REQUEST_COUNT) or 0
        reported = _int_attribute(attributes, TOKEN_REPORTED_REQUEST_COUNT) or 0
        self.request_count += requests
        self.expected_request_count += expected
        self.reported_request_count += reported
        input_tokens = _int_attribute(attributes, "gen_ai.usage.input_tokens")
        output_tokens = _int_attribute(attributes, "gen_ai.usage.output_tokens")
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        total_tokens = _int_attribute(attributes, TOKEN_TOTAL)
        if total_tokens is not None:
            self.total_tokens += total_tokens
        elif input_tokens is not None and output_tokens is not None:
            self.total_tokens += input_tokens + output_tokens
        complete = attributes.get(TOKEN_COMPLETE)
        self.complete &= (
            complete
            if isinstance(complete, bool)
            else expected == reported
            and (
                reported == 0
                or total_tokens is not None
                or (input_tokens is not None and output_tokens is not None)
            )
        )
        self.audio_seconds += _float_attribute(attributes, TOKEN_AUDIO_SECONDS) or 0.0
        requested = attributes.get("mindbridge.input.modalities")
        if isinstance(requested, tuple) and all(isinstance(value, str) for value in requested):
            for modality in requested:
                self.calls_by_input_modality[modality] = (
                    self.calls_by_input_modality.get(modality, 0) + requests
                )
        self._add_modalities(attributes, "input", self.input_by_modality)
        self._add_modalities(attributes, "output", self.output_by_modality)

    def json(self, question_count: int) -> dict[str, object]:
        complete = self.complete and self.expected_request_count == self.reported_request_count
        return {
            "complete": complete,
            "request_count": self.request_count,
            "unreported_request_count": (self.expected_request_count - self.reported_request_count),
            "total_tokens": self.total_tokens if complete else None,
            "average_tokens": (
                self.total_tokens / question_count if complete and question_count else None
            ),
            "reported_total_tokens": self.total_tokens,
            "reported_average_tokens": (
                self.total_tokens / question_count if question_count else None
            ),
            "modality_breakdown_complete": (
                sum(self.input_by_modality.values()) + sum(self.output_by_modality.values())
                == self.total_tokens
                and not self.input_by_modality.get("unattributed")
                and not self.output_by_modality.get("unattributed")
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls_by_input_modality": dict(sorted(self.calls_by_input_modality.items())),
            "input_by_modality": dict(sorted(self.input_by_modality.items())),
            "output_by_modality": dict(sorted(self.output_by_modality.items())),
            "audio_seconds": self.audio_seconds,
        }

    @staticmethod
    def _add_modalities(
        attributes: Mapping[str, AttributeValue],
        direction: str,
        target: dict[str, int],
    ) -> None:
        for modality in TOKEN_MODALITIES:
            count = _int_attribute(
                attributes,
                token_modality_attribute(direction, modality),
            )
            if count is not None:
                target[modality] = target.get(modality, 0) + count


@dataclass(slots=True)
class _TaskTelemetry:
    run_ns: int = 0
    judge_started_ns: int | None = None
    judge_ended_ns: int | None = None
    nodes: dict[str, _Durations] = field(default_factory=dict)
    tokens: _Tokens = field(default_factory=_Tokens)
    tokens_by_module: dict[str, _Tokens] = field(default_factory=dict)

    def add(self, span: ReadableSpan) -> None:
        attributes = span.attributes or {}
        if span.name == BENCHMARK_TASK_SPAN:
            self.run_ns += _duration_ns(span)
        elif span.name == BENCHMARK_JUDGE_SPAN:
            start, end = span.start_time, span.end_time
            if start is not None and end is not None:
                self.judge_started_ns = (
                    start if self.judge_started_ns is None else min(self.judge_started_ns, start)
                )
                self.judge_ended_ns = (
                    end if self.judge_ended_ns is None else max(self.judge_ended_ns, end)
                )
        kind = _string_attribute(attributes, SPAN_KIND)
        if kind not in {"operation", "stage", "model"}:
            return
        self.nodes.setdefault(span.name, _Durations()).add(span)
        if kind == "model":
            self.tokens.add(attributes)
            module = _string_attribute(attributes, MODEL_MODULE) or "unknown"
            self.tokens_by_module.setdefault(module, _Tokens()).add(attributes)

    def json(self, question_count: int) -> dict[str, object]:
        judge_ns = (
            0
            if self.judge_started_ns is None or self.judge_ended_ns is None
            else self.judge_ended_ns - self.judge_started_ns
        )
        total_ns = self.run_ns + judge_ns
        return {
            "duration_seconds": {
                "total": total_ns / 1_000_000_000,
                "average": (total_ns / question_count / 1_000_000_000 if question_count else None),
                "mindbridge": self.run_ns / 1_000_000_000,
                "judge": judge_ns / 1_000_000_000,
            },
            "nodes": {name: value.json() for name, value in sorted(self.nodes.items())},
            "token_usage": {
                **self.tokens.json(question_count),
                "by_module": {
                    name: value.json(question_count)
                    for name, value in sorted(self.tokens_by_module.items())
                },
            },
        }


class EvaluationTelemetry(SpanProcessor):
    """Aggregate evaluation spans online without retaining per-request traces."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._span_tasks: dict[int, str] = {}
        self._tasks: dict[str, _TaskTelemetry] = {}
        self._provider = TracerProvider()
        self._provider.add_span_processor(self)
        self.tracer: Tracer = self._provider.get_tracer(TRACER_NAME)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        attributes = span.attributes or {}
        task = _string_attribute(attributes, BENCHMARK_TASK)
        parent_context_value = trace.get_current_span(parent_context).get_span_context()
        parent = 0 if parent_context_value is None else parent_context_value.span_id
        with self._lock:
            if task is None:
                task = self._span_tasks.get(parent)
            if task is not None:
                span_context = span.get_span_context()
                if span_context is not None:
                    self._span_tasks[span_context.span_id] = task

    def on_end(self, span: ReadableSpan) -> None:
        span_context = span.get_span_context()
        span_id = 0 if span_context is None else span_context.span_id
        attributes = span.attributes or {}
        with self._lock:
            task = self._span_tasks.pop(span_id, None) or _string_attribute(
                attributes, BENCHMARK_TASK
            )
            if task is not None:
                self._tasks.setdefault(task, _TaskTelemetry()).add(span)

    def result(self, task: str, *, question_count: int) -> Mapping[str, object]:
        with self._lock:
            values = self._tasks.get(task, _TaskTelemetry())
            return values.json(question_count)

    def shutdown(self) -> None:
        """Release the private provider; aggregation itself has no exporter."""

    def close(self) -> None:
        self._provider.shutdown()


def _duration_ns(span: ReadableSpan) -> int:
    if span.start_time is None or span.end_time is None:
        return 0
    return max(0, span.end_time - span.start_time)


def _int_attribute(attributes: Mapping[str, AttributeValue], name: str) -> int | None:
    value = attributes.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _float_attribute(attributes: Mapping[str, AttributeValue], name: str) -> float | None:
    value = attributes.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _string_attribute(attributes: Mapping[str, AttributeValue], name: str) -> str | None:
    value = attributes.get(name)
    return value if isinstance(value, str) else None
