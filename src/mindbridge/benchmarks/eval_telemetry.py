"""Bounded in-process aggregation for evaluation telemetry spans."""

from __future__ import annotations

import math
import os
import platform
import resource
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from types import TracebackType

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    CAPTURE_TIME_TO_SEARCHABLE,
    EMBEDDING_TASK,
    GEN_AI_TTFC,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    MODEL_MODULE,
    MODEL_REQUEST_COUNT,
    MODEL_RESPONSE_MODELS,
    MODEL_RESPONSE_SYSTEM_FINGERPRINTS,
    MODEL_TTFT,
    OPERATION_TTFT,
    SPAN_KIND,
    TOKEN_AUDIO_SECONDS,
    TOKEN_CACHED_INPUT,
    TOKEN_CACHED_INPUT_COMPLETE,
    TOKEN_COMPLETE,
    TOKEN_EXPECTED_REQUEST_COUNT,
    TOKEN_INPUT_COMPLETE,
    TOKEN_MODALITIES,
    TOKEN_OUTPUT_COMPLETE,
    TOKEN_REASONING_OUTPUT,
    TOKEN_REASONING_OUTPUT_COMPLETE,
    TOKEN_REPORTED_REQUEST_COUNT,
    TOKEN_TOTAL,
    TRACER_NAME,
    VISION_BATCHES_FAILED,
    token_modality_attribute,
)
from mindbridge.benchmarks.eval_statistics import percentile

BENCHMARK_TASK = "mindbridge.benchmark.task"
BENCHMARK_SAMPLE = "mindbridge.benchmark.sample"
BENCHMARK_ARM = "mindbridge.benchmark.arm"
BENCHMARK_PURPOSE = "mindbridge.benchmark.purpose"
BENCHMARK_PARENT_OPERATION = "mindbridge.benchmark.parent_operation"
BENCHMARK_TASK_SPAN = "mindbridge.benchmark.run"
BENCHMARK_ARM_SPAN = "mindbridge.benchmark.arm.run"
BENCHMARK_ANSWER_SPAN = "mindbridge.benchmark.answer"
BENCHMARK_DIAGNOSTIC_SPAN = "mindbridge.benchmark.retrieval_diagnostic"
BENCHMARK_JUDGE_SPAN = "mindbridge.benchmark.judge"
BENCHMARK_INGEST_SPAN = "mindbridge.benchmark.ingest"
BENCHMARK_INGEST_ITEMS = "mindbridge.benchmark.ingest.items"

# The fast plane's three product operations: `capture()` acknowledges after the SQLite commit,
# so its own span duration *is* capture-acknowledgement latency; `settle()`'s span duration is the
# model-dependent formation work one settle() call paid, i.e. formation lag; `compile()`'s span
# duration is compile latency. None of these run unless the harness actually calls them --
# `--ingest capture` for the first two, the `compile` answering arm for the third -- so a run that
# never exercises a path reports nothing under it, the same convention `SEARCH_SPAN` follows.
CAPTURE_SPAN = "mindbridge.capture"
SETTLE_SPAN = "mindbridge.settle"
COMPILE_SPAN = "mindbridge.compile"
# Harness-owned spans (like `BENCHMARK_INGEST_SPAN`) that carry a `compile()` bundle's size:
# `compile()` itself sets no such attribute, so the eval driver tags one stage span per compiled
# answer instead of asking product code to carry a benchmark-only measurement.
BENCHMARK_COMPILE_SPAN = "mindbridge.benchmark.compile"
BENCHMARK_COMPILE_CHARS = "mindbridge.benchmark.compile.chars"
BENCHMARK_COMPILE_ITEMS = "mindbridge.benchmark.compile.items"
BENCHMARK_COMPILE_MEDIA_ITEMS = "mindbridge.benchmark.compile.media_items"
DEFAULT_BENCHMARK_ARM = "mindbridge"
SHARED_BENCHMARK_ARM = "shared"
PRODUCT_PURPOSE = "product"
DIAGNOSTIC_PURPOSE = "diagnostic"
JUDGE_PURPOSE = "judge"

ANSWER_SPAN = "mindbridge.ask"
SEARCH_E2E_SPAN = "mindbridge.search"
ASK_RETRIEVAL_SPAN = "mindbridge.retrieve"
# Compatibility export for callers that imported the former ambiguous name. Result documents
# expose it only as a deprecated alias of ``ask_retrieval_core``.
SEARCH_SPAN = ASK_RETRIEVAL_SPAN
GENERATION_SPAN = "mindbridge.model.generation"
TRANSCRIPTION_SPAN = "mindbridge.model.transcription"
TRANSCRIPTION_MODULE = "transcription"
JUDGE_MODULE = "judge"

_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_MODEL_BATCH_SIZE = "mindbridge.model.batch_size"
_INPUT_MODALITIES = "mindbridge.input.modalities"


@dataclass(frozen=True, slots=True)
class SampleGrounding:
    """How much retrieved evidence one answer's inline context budget removed."""

    dropped_hits: int
    media_elided_hits: int


@dataclass(slots=True)
class _Durations:
    count: int = 0
    total_ns: int = 0
    ttft_count: int = 0
    ttft_total_ms: float = 0.0
    ttfc_count: int = 0
    ttfc_total_ms: float = 0.0
    durations_ns: list[int] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    ttfc_ms: list[float] = field(default_factory=list)
    intervals_ns: list[tuple[int, int]] = field(default_factory=list)
    first_start_ns: int | None = None
    last_end_ns: int | None = None

    def add(self, span: ReadableSpan) -> None:
        self.count += 1
        duration_ns = _duration_ns(span)
        self.total_ns += duration_ns
        self.durations_ns.append(duration_ns)
        if span.start_time is not None and span.end_time is not None:
            self.intervals_ns.append((span.start_time, max(span.start_time, span.end_time)))
            self.first_start_ns = (
                span.start_time
                if self.first_start_ns is None
                else min(self.first_start_ns, span.start_time)
            )
        if span.end_time is not None:
            self.last_end_ns = (
                span.end_time if self.last_end_ns is None else max(self.last_end_ns, span.end_time)
            )
        attributes = span.attributes or {}
        # Operation TTFT is recorded directly in milliseconds. Model TTFT follows the OpenAI
        # convention used by the runtime and is recorded in seconds.
        ttft = _float_attribute(attributes, OPERATION_TTFT)
        if ttft is None:
            model_ttft = _float_attribute(attributes, MODEL_TTFT)
            ttft = None if model_ttft is None else model_ttft * 1_000
        if ttft is not None:
            self.ttft_count += 1
            self.ttft_total_ms += ttft
            self.ttft_ms.append(ttft)
        ttfc = _float_attribute(attributes, GEN_AI_TTFC)
        if ttfc is not None:
            self.ttfc_count += 1
            ttfc_ms = ttfc * 1_000
            self.ttfc_total_ms += ttfc_ms
            self.ttfc_ms.append(ttfc_ms)

    def wall_seconds(self) -> float | None:
        """Return the elapsed time from the first span start to the last span end."""
        if self.first_start_ns is None or self.last_end_ns is None:
            return None
        return max(0, self.last_end_ns - self.first_start_ns) / 1_000_000_000

    def active_seconds(self) -> float | None:
        """Return the union of retained span intervals, excluding gaps and overlap."""
        if not self.intervals_ns:
            return None
        start, end = sorted(self.intervals_ns)[0]
        active_ns = 0
        for next_start, next_end in sorted(self.intervals_ns)[1:]:
            if next_start <= end:
                end = max(end, next_end)
                continue
            active_ns += end - start
            start, end = next_start, next_end
        return (active_ns + end - start) / 1_000_000_000

    def latency_ms(self) -> dict[str, object]:
        """Return exact quantiles over the retained per-span durations."""
        return _distribution_json(
            tuple(value / 1_000_000 for value in self.durations_ns),
            total_count=self.count,
            total=self.total_ns / 1_000_000,
        )

    def ttft_json(self) -> dict[str, object] | None:
        if not self.ttft_count:
            return None
        return _observed_distribution_json(
            self.ttft_ms,
            total_count=self.count,
            total=self.ttft_total_ms,
        )

    def ttfc_json(self) -> dict[str, object] | None:
        if not self.ttfc_count:
            return None
        return _observed_distribution_json(
            self.ttfc_ms,
            total_count=self.count,
            total=self.ttfc_total_ms,
        )

    def json(self) -> dict[str, object]:
        active_seconds = self.active_seconds()
        result: dict[str, object] = {
            "count": self.count,
            "total_seconds": self.total_ns / 1_000_000_000,
            "average_ms": self.total_ns / self.count / 1_000_000,
            "latency_ms": self.latency_ms(),
            "active_seconds": active_seconds,
            "observation_window_seconds": self.wall_seconds(),
            "throughput_per_active_second": (
                None if not active_seconds else self.count / active_seconds
            ),
        }
        if self.ttft_count:
            result["ttft_ms"] = self.ttft_json()
        if self.ttfc_count:
            result["time_to_first_chunk_ms"] = self.ttfc_json()
        return result


@dataclass(slots=True)
class _Samples:
    """Scalar observations retained in full for unbiased exact percentiles."""

    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    def json(self) -> dict[str, object]:
        return {
            "count": len(self.values),
            "average": None if not self.values else sum(self.values) / len(self.values),
            "p50": percentile(self.values, 0.50),
            "p95": percentile(self.values, 0.95),
            "p99": percentile(self.values, 0.99),
        }


@dataclass(slots=True)
class _Tokens:
    complete: bool = True
    request_count: int = 0
    expected_request_count: int = 0
    reported_request_count: int = 0
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
    audio_seconds: float = 0.0
    exact_call_tokens: list[float] = field(default_factory=list)
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
        self.input_seen |= input_tokens is not None
        self.output_seen |= output_tokens is not None
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        total_tokens = _int_attribute(attributes, TOKEN_TOTAL)
        resolved_total = total_tokens
        if total_tokens is not None:
            self.total_tokens += total_tokens
        elif input_tokens is not None and output_tokens is not None:
            resolved_total = input_tokens + output_tokens
            self.total_tokens += resolved_total
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
        cached_input_tokens = _int_attribute(attributes, TOKEN_CACHED_INPUT)
        reasoning_output_tokens = _int_attribute(attributes, TOKEN_REASONING_OUTPUT)
        self.cached_input_seen |= cached_input_tokens is not None
        self.reasoning_output_seen |= reasoning_output_tokens is not None
        self.cached_input_tokens += cached_input_tokens or 0
        self.reasoning_output_tokens += reasoning_output_tokens or 0
        if expected:
            reported_all = expected == reported
            fallbacks = (
                (TOKEN_INPUT_COMPLETE, reported_all and input_tokens is not None),
                (TOKEN_OUTPUT_COMPLETE, reported_all and output_tokens is not None),
                (
                    TOKEN_CACHED_INPUT_COMPLETE,
                    reported_all and cached_input_tokens is not None,
                ),
                (
                    TOKEN_REASONING_OUTPUT_COMPLETE,
                    reported_all and reasoning_output_tokens is not None,
                ),
            )
            states = tuple(
                value if isinstance(value := attributes.get(name), bool) else fallback
                for name, fallback in fallbacks
            )
            self.input_complete &= states[0]
            self.output_complete &= states[1]
            self.cached_input_complete &= states[2]
            self.reasoning_output_complete &= states[3]
        # A distribution is exact only when one traced call represents one provider request.
        # Batched totals stay in the aggregate and make this distribution explicitly incomplete.
        if requests == expected == reported == 1 and resolved_total is not None:
            self.exact_call_tokens.append(float(resolved_total))
        self.audio_seconds += _float_attribute(attributes, TOKEN_AUDIO_SECONDS) or 0.0
        requested = attributes.get("mindbridge.input.modalities")
        if isinstance(requested, tuple) and all(isinstance(value, str) for value in requested):
            for modality in requested:
                self.calls_by_input_modality[modality] = (
                    self.calls_by_input_modality.get(modality, 0) + requests
                )
        self._add_modalities(attributes, "input", self.input_by_modality)
        self._add_modalities(attributes, "output", self.output_by_modality)

    def json(
        self, question_count: int, *, compute_seconds: float | None = None
    ) -> dict[str, object]:
        complete = self.complete and self.expected_request_count == self.reported_request_count
        input_complete = bool(
            self.expected_request_count and self.input_seen and self.input_complete
        )
        output_complete = bool(
            self.expected_request_count and self.output_seen and self.output_complete
        )
        cached_complete = bool(
            self.expected_request_count and self.cached_input_seen and self.cached_input_complete
        )
        reasoning_complete = bool(
            self.expected_request_count
            and self.reasoning_output_seen
            and self.reasoning_output_complete
        )
        per_call: dict[str, object] | None = None
        if self.expected_request_count:
            exact_total = sum(self.exact_call_tokens)
            per_call = _distribution_json(
                self.exact_call_tokens,
                total_count=self.expected_request_count,
                total=exact_total,
            )
            per_call["retained_average"] = (
                None if not self.exact_call_tokens else exact_total / len(self.exact_call_tokens)
            )
            if not per_call["complete"]:
                per_call["average"] = None
        modality_complete = (
            complete
            and input_complete
            and output_complete
            and sum(self.input_by_modality.values()) + sum(self.output_by_modality.values())
            == self.total_tokens
            and not self.input_by_modality.get("unattributed")
            and not self.output_by_modality.get("unattributed")
        )
        return {
            "complete": complete,
            "request_count": self.request_count,
            "token_usage_expected_request_count": self.expected_request_count,
            "token_usage_reported_request_count": self.reported_request_count,
            "unreported_request_count": max(
                0, self.expected_request_count - self.reported_request_count
            ),
            "total_tokens": self.total_tokens if complete else None,
            "average_tokens": (
                self.total_tokens / question_count if complete and question_count else None
            ),
            "reported_total_tokens": self.total_tokens,
            "reported_average_tokens": (
                self.total_tokens / question_count if question_count else None
            ),
            "average_tokens_per_request": (
                self.total_tokens / self.expected_request_count
                if complete and self.expected_request_count
                else None
            ),
            "per_call_total_tokens": per_call,
            "modality_breakdown_complete": (
                modality_complete if self.expected_request_count else None
            ),
            "input_tokens": self.input_tokens if input_complete else None,
            "output_tokens": self.output_tokens if output_complete else None,
            "cached_input_tokens": self.cached_input_tokens if cached_complete else None,
            "reasoning_output_tokens": (
                self.reasoning_output_tokens if reasoning_complete else None
            ),
            "input_tokens_complete": input_complete,
            "output_tokens_complete": output_complete,
            "cached_input_tokens_complete": cached_complete,
            "reasoning_output_tokens_complete": reasoning_complete,
            "reported_input_tokens": self.input_tokens if self.input_seen else None,
            "reported_output_tokens": self.output_tokens if self.output_seen else None,
            "reported_cached_input_tokens": (
                self.cached_input_tokens if self.cached_input_seen else None
            ),
            "reported_reasoning_output_tokens": (
                self.reasoning_output_tokens if self.reasoning_output_seen else None
            ),
            "observed_output_tokens_per_second": (
                None
                if not complete or not output_complete or not compute_seconds
                else self.output_tokens / compute_seconds
            ),
            "calls_by_input_modality": dict(sorted(self.calls_by_input_modality.items())),
            "input_by_modality": dict(sorted(self.input_by_modality.items())),
            "output_by_modality": dict(sorted(self.output_by_modality.items())),
            "audio_seconds": self.audio_seconds,
        }

    def merge(self, other: _Tokens) -> None:
        self.complete &= other.complete
        self.request_count += other.request_count
        self.expected_request_count += other.expected_request_count
        self.reported_request_count += other.reported_request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.input_seen |= other.input_seen
        self.output_seen |= other.output_seen
        self.cached_input_seen |= other.cached_input_seen
        self.reasoning_output_seen |= other.reasoning_output_seen
        self.input_complete &= other.input_complete
        self.output_complete &= other.output_complete
        self.cached_input_complete &= other.cached_input_complete
        self.reasoning_output_complete &= other.reasoning_output_complete
        self.audio_seconds += other.audio_seconds
        self.exact_call_tokens.extend(other.exact_call_tokens)
        for source, target in (
            (other.calls_by_input_modality, self.calls_by_input_modality),
            (other.input_by_modality, self.input_by_modality),
            (other.output_by_modality, self.output_by_modality),
        ):
            for name, value in source.items():
                target[name] = target.get(name, 0) + value

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


@dataclass(frozen=True, slots=True)
class _NodeDimensions:
    requested_model: str | None
    response_model: str | None
    response_models: tuple[str, ...]
    system_fingerprint: str | None
    response_system_fingerprints: tuple[str, ...]
    embedding_task: str | None
    batch_size: int | None
    modalities: tuple[str, ...]
    status: str
    parent_operation: str | None
    purpose: str

    @classmethod
    def from_span(cls, span: ReadableSpan) -> _NodeDimensions:
        attributes = span.attributes or {}
        requested = attributes.get(_INPUT_MODALITIES)
        modalities = (
            tuple(sorted(requested))
            if isinstance(requested, tuple) and all(isinstance(value, str) for value in requested)
            else ()
        )
        return cls(
            requested_model=_string_attribute(attributes, _GEN_AI_REQUEST_MODEL),
            response_model=_string_attribute(attributes, "gen_ai.response.model"),
            response_models=_string_tuple_attribute(attributes, MODEL_RESPONSE_MODELS),
            system_fingerprint=_string_attribute(
                attributes, "gen_ai.openai.response.system_fingerprint"
            ),
            response_system_fingerprints=_string_tuple_attribute(
                attributes, MODEL_RESPONSE_SYSTEM_FINGERPRINTS
            ),
            embedding_task=_string_attribute(attributes, EMBEDDING_TASK),
            batch_size=_int_attribute(attributes, _MODEL_BATCH_SIZE),
            modalities=modalities,
            status=_span_status(span),
            parent_operation=_string_attribute(attributes, BENCHMARK_PARENT_OPERATION),
            purpose=_string_attribute(attributes, BENCHMARK_PURPOSE) or PRODUCT_PURPOSE,
        )

    def json(self) -> dict[str, object]:
        return {
            "model": self.requested_model,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "response_models": self.response_models,
            "system_fingerprint": self.system_fingerprint,
            "response_system_fingerprints": self.response_system_fingerprints,
            "embedding_task": self.embedding_task,
            "batch_size": self.batch_size,
            "modalities": self.modalities,
            "status": self.status,
            "parent_operation": self.parent_operation,
            "purpose": self.purpose,
        }


@dataclass(slots=True)
class _TaskTelemetry:
    run: _Durations = field(default_factory=_Durations)
    judge: _Durations = field(default_factory=_Durations)
    diagnostic_search: _Durations = field(default_factory=_Durations)
    successful_diagnostic_search: _Durations = field(default_factory=_Durations)
    product_samples: set[str] = field(default_factory=set)
    judge_samples: set[str] = field(default_factory=set)
    nodes: dict[str, _Durations] = field(default_factory=dict)
    successful_nodes: dict[str, _Durations] = field(default_factory=dict)
    diagnostic_nodes: dict[str, _Durations] = field(default_factory=dict)
    successful_diagnostic_nodes: dict[str, _Durations] = field(default_factory=dict)
    node_breakdowns: dict[str, dict[_NodeDimensions, _Durations]] = field(default_factory=dict)
    diagnostic_breakdowns: dict[str, dict[_NodeDimensions, _Durations]] = field(
        default_factory=dict
    )
    model_durations_by_module: dict[str, _Durations] = field(default_factory=dict)
    diagnostic_model_durations_by_module: dict[str, _Durations] = field(default_factory=dict)
    caller_answers: _Durations = field(default_factory=_Durations)
    successful_caller_answers: _Durations = field(default_factory=_Durations)
    tokens: _Tokens = field(default_factory=_Tokens)
    tokens_by_module: dict[str, _Tokens] = field(default_factory=dict)
    asr_ratio_spans: _Durations = field(default_factory=_Durations)
    asr_audio_seconds: float = 0.0
    diagnostic_tokens: _Tokens = field(default_factory=_Tokens)
    diagnostic_tokens_by_module: dict[str, _Tokens] = field(default_factory=dict)
    media_elided_hits: int = 0
    dropped_hits: int = 0
    vision_failed_batches: int = 0
    # Fast-plane and compiler measurements: populated only when the run actually exercises
    # `capture()`/`settle()` (the `--ingest capture` path) or the `compile` answering arm.
    time_to_searchable_ms: _Samples = field(default_factory=_Samples)
    compile_chars: _Samples = field(default_factory=_Samples)
    compile_items: _Samples = field(default_factory=_Samples)
    compile_media_items: _Samples = field(default_factory=_Samples)
    ingest_attempted_items: int = 0
    ingest_successful_items: int = 0
    ingest_error_count: int = 0

    def add(self, span: ReadableSpan) -> None:  # noqa: C901 - one pass classifies every dimension
        attributes = span.attributes or {}
        status = _span_status(span)
        purpose = _string_attribute(attributes, BENCHMARK_PURPOSE) or PRODUCT_PURPOSE
        diagnostic = purpose == DIAGNOSTIC_PURPOSE
        if span.name == BENCHMARK_ANSWER_SPAN and not diagnostic:
            self.caller_answers.add(span)
            sample = _string_attribute(attributes, BENCHMARK_SAMPLE)
            if sample is not None:
                self.product_samples.add(sample)
            if status == "ok":
                self.successful_caller_answers.add(span)
        if span.name == BENCHMARK_DIAGNOSTIC_SPAN and diagnostic:
            self.diagnostic_search.add(span)
            if status == "ok":
                self.successful_diagnostic_search.add(span)
        if span.name == BENCHMARK_ARM_SPAN:
            self.run.add(span)
        elif span.name == BENCHMARK_INGEST_SPAN:
            items = _int_attribute(attributes, BENCHMARK_INGEST_ITEMS) or 0
            self.run.add(span)
            self.ingest_attempted_items += items
            if status == "ok":
                self.ingest_successful_items += items
            else:
                self.ingest_error_count += 1
        elif span.name == SETTLE_SPAN:
            self._add_time_to_searchable(attributes)
        elif span.name == BENCHMARK_COMPILE_SPAN:
            self._add_compile_bundle(attributes)
        elif span.name == BENCHMARK_JUDGE_SPAN:
            self.judge.add(span)
            sample = _string_attribute(attributes, BENCHMARK_SAMPLE)
            if sample is not None:
                self.judge_samples.add(sample)
        kind = _string_attribute(attributes, SPAN_KIND)
        if kind not in {"operation", "stage", "model"}:
            return
        if kind == "model" and _int_attribute(attributes, MODEL_REQUEST_COUNT) == 0:
            return
        dimensions = _NodeDimensions.from_span(span)
        if diagnostic:
            self.diagnostic_nodes.setdefault(span.name, _Durations()).add(span)
            if status == "ok":
                self.successful_diagnostic_nodes.setdefault(span.name, _Durations()).add(span)
            self.diagnostic_breakdowns.setdefault(span.name, {}).setdefault(
                dimensions, _Durations()
            ).add(span)
            if kind == "model":
                self.diagnostic_tokens.add(attributes)
                module = _string_attribute(attributes, MODEL_MODULE) or "unknown"
                self.diagnostic_tokens_by_module.setdefault(module, _Tokens()).add(attributes)
                self.diagnostic_model_durations_by_module.setdefault(module, _Durations()).add(span)
            return
        self.nodes.setdefault(span.name, _Durations()).add(span)
        self.node_breakdowns.setdefault(span.name, {}).setdefault(dimensions, _Durations()).add(
            span
        )
        if status == "ok":
            self.successful_nodes.setdefault(span.name, _Durations()).add(span)
        if kind == "model":
            self.tokens.add(attributes)
            self.media_elided_hits += _int_attribute(attributes, GROUNDING_MEDIA_ELIDED) or 0
            self.dropped_hits += _int_attribute(attributes, GROUNDING_HITS_DROPPED) or 0
            self.vision_failed_batches += _int_attribute(attributes, VISION_BATCHES_FAILED) or 0
            module = _string_attribute(attributes, MODEL_MODULE) or "unknown"
            self.tokens_by_module.setdefault(module, _Tokens()).add(attributes)
            if status == "ok":
                audio_seconds = _float_attribute(attributes, TOKEN_AUDIO_SECONDS)
                if (
                    module == TRANSCRIPTION_MODULE
                    and audio_seconds is not None
                    and audio_seconds > 0
                ):
                    self.asr_ratio_spans.add(span)
                    self.asr_audio_seconds += audio_seconds
            # Token usage includes every billed attempt, so its elapsed denominator must include
            # failed model spans too. Pairing all-attempt tokens with success-only time inflated
            # observed throughput whenever a provider returned a metered malformed response.
            self.model_durations_by_module.setdefault(module, _Durations()).add(span)

    def _add_time_to_searchable(self, attributes: Mapping[str, AttributeValue]) -> None:
        value = _float_attribute(attributes, CAPTURE_TIME_TO_SEARCHABLE)
        if value is not None:
            self.time_to_searchable_ms.add(value)

    def _add_compile_bundle(self, attributes: Mapping[str, AttributeValue]) -> None:
        chars = _int_attribute(attributes, BENCHMARK_COMPILE_CHARS)
        items = _int_attribute(attributes, BENCHMARK_COMPILE_ITEMS)
        media_items = _int_attribute(attributes, BENCHMARK_COMPILE_MEDIA_ITEMS)
        if chars is not None:
            self.compile_chars.add(chars)
        if items is not None:
            self.compile_items.add(items)
        if media_items is not None:
            self.compile_media_items.add(media_items)

    def _product_tokens_json(self, question_count: int) -> dict[str, object]:
        modules = {
            name: value for name, value in self.tokens_by_module.items() if name != JUDGE_MODULE
        }
        combined = _Tokens()
        for value in modules.values():
            combined.merge(value)
        compute_seconds = (
            sum(
                value.total_ns
                for name, value in self.model_durations_by_module.items()
                if name != JUDGE_MODULE
            )
            / 1_000_000_000
        )
        return {
            "modules": sorted(modules),
            **combined.json(question_count, compute_seconds=compute_seconds),
        }

    def _ingest_json(self) -> dict[str, object]:
        attempts = self.nodes.get(BENCHMARK_INGEST_SPAN)
        successful = self.successful_nodes.get(BENCHMARK_INGEST_SPAN)
        active_seconds = None if successful is None else successful.active_seconds()
        successful_count = 0 if successful is None else successful.count
        return {
            "measures": (
                "successful add/add_many batches through durable SQLite commit, Zvec flush, and "
                "searchable outbox acknowledgement; active_seconds is the union of successful "
                "batch intervals"
            ),
            "span": BENCHMARK_INGEST_SPAN,
            "attempt_count": 0 if attempts is None else attempts.count,
            "success_count": successful_count,
            "error_count": self.ingest_error_count,
            "attempted_item_count": self.ingest_attempted_items,
            "accepted_item_count": self.ingest_successful_items,
            "attempt_latency_ms": None if attempts is None else attempts.latency_ms(),
            "successful_batch_latency_ms": (
                None if successful is None else successful.latency_ms()
            ),
            "successful_compute_seconds": (
                0.0 if successful is None else successful.total_ns / 1_000_000_000
            ),
            "active_seconds": active_seconds,
            "amortized_compute_ms_per_accepted_item": (
                None
                if successful is None or not self.ingest_successful_items
                else successful.total_ns / self.ingest_successful_items / 1_000_000
            ),
            "items_per_active_second": (
                None
                if not active_seconds or not self.ingest_successful_items
                else self.ingest_successful_items / active_seconds
            ),
            # Compatibility aliases. Their semantics are now explicit and failed attempts never
            # enter them. ``item_latency_ms`` was not a latency and is intentionally retired.
            "call_count": successful_count,
            "item_count": self.ingest_successful_items,
            "compute_seconds": (0.0 if successful is None else successful.total_ns / 1_000_000_000),
            "wall_seconds": active_seconds,
            "call_latency_ms": None if successful is None else successful.latency_ms(),
            "item_latency_ms": None,
            "items_per_second": (
                None
                if not active_seconds or not self.ingest_successful_items
                else self.ingest_successful_items / active_seconds
            ),
            "deprecated_fields": {
                "wall_seconds": "alias of active_seconds, not first-to-last observation window",
                "item_latency_ms": "removed; use successful_batch_latency_ms",
                "items_per_second": "alias of items_per_active_second",
            },
        }

    def _span_latency_json(self, span_name: str) -> dict[str, object]:
        attempts = self.nodes.get(span_name)
        successful = self.successful_nodes.get(span_name)
        return self._durations_latency_json(span_name, attempts, successful)

    @staticmethod
    def _durations_latency_json(
        span_name: str,
        attempts: _Durations | None,
        successful: _Durations | None,
    ) -> dict[str, object]:
        active_seconds = None if successful is None else successful.active_seconds()
        count = 0 if successful is None else successful.count
        attempt_count = 0 if attempts is None else attempts.count
        return {
            "span": span_name,
            "attempt_count": attempt_count,
            "count": count,
            "error_count": attempt_count - count,
            "active_seconds": active_seconds,
            "observation_window_seconds": None if successful is None else successful.wall_seconds(),
            "latency_ms": None if successful is None else successful.latency_ms(),
            "throughput_per_active_second": (
                None if not active_seconds or not count else count / active_seconds
            ),
            # Compatibility aliases with corrected, gap-free semantics.
            "wall_seconds": active_seconds,
            "throughput_per_second": (
                None if not active_seconds or not count else count / active_seconds
            ),
        }

    def _answer_json(self) -> dict[str, object]:
        caller_recorded = bool(self.caller_answers.count)
        result = (
            self._durations_latency_json(
                BENCHMARK_ANSWER_SPAN,
                self.caller_answers,
                self.successful_caller_answers if self.successful_caller_answers.count else None,
            )
            if caller_recorded
            else self._span_latency_json(ANSWER_SPAN)
        )
        answers = (
            self.successful_caller_answers
            if caller_recorded
            else self.successful_nodes.get(ANSWER_SPAN, _Durations())
        )
        generation = self.successful_nodes.get(GENERATION_SPAN)
        end_to_end = answers.ttft_json()
        generation_ttft = None if generation is None else generation.ttft_json()
        generation_ttfc = None if generation is None else generation.ttfc_json()
        result.update(
            streaming=generation_ttft is not None,
            end_to_end_time_to_first_token_ms=end_to_end,
            generation_time_to_first_token_ms=generation_ttft,
            generation_time_to_first_chunk_ms=generation_ttfc,
            # Compatibility only: this used to be presented as answer TTFT even though it starts
            # at generation. Consumers must migrate to one of the two explicit fields above.
            time_to_first_token_ms=generation_ttft,
            deprecated_fields={
                "time_to_first_token_ms": "alias of generation_time_to_first_token_ms"
            },
            sdk_operation=self._span_latency_json(ANSWER_SPAN),
        )
        return result

    def _asr_json(self) -> dict[str, object] | None:
        tokens = self.tokens_by_module.get(TRANSCRIPTION_MODULE)
        node = self.nodes.get(TRANSCRIPTION_SPAN)
        if tokens is None or node is None or not node.total_ns:
            return None
        successful = self.successful_nodes.get(TRANSCRIPTION_SPAN)
        success_count = 0 if successful is None else successful.count
        successful_compute_seconds = (
            0.0 if successful is None else successful.total_ns / 1_000_000_000
        )
        ratio_call_count = self.asr_ratio_spans.count
        compute_seconds = self.asr_ratio_spans.total_ns / 1_000_000_000
        audio_seconds = self.asr_audio_seconds
        subset_rtf = (
            None if not audio_seconds or not compute_seconds else compute_seconds / audio_seconds
        )
        subset_speedup = (
            None if not audio_seconds or not compute_seconds else audio_seconds / compute_seconds
        )
        ratio_complete = ratio_call_count == success_count
        return {
            "span": TRANSCRIPTION_SPAN,
            "invocation_count": node.count,
            "request_count": tokens.request_count,
            "call_count": node.count,
            "success_count": success_count,
            "error_count": node.count - success_count,
            "audio_seconds": audio_seconds,
            "compute_seconds": compute_seconds,
            "successful_compute_seconds": successful_compute_seconds,
            "ratio_call_count": ratio_call_count,
            "ratio_invocation_count": ratio_call_count,
            "audio_duration_missing_success_count": success_count - ratio_call_count,
            "ratio_complete": ratio_complete,
            "real_time_factor": subset_rtf if ratio_complete else None,
            "realtime_speedup": subset_speedup if ratio_complete else None,
            "reported_subset_real_time_factor": subset_rtf,
            "reported_subset_realtime_speedup": subset_speedup,
            "ratio_scope": "successful calls with reported audio duration",
            "inference_latency_ms": node.latency_ms(),
            "successful_inference_latency_ms": (
                None if successful is None else successful.latency_ms()
            ),
            "ratio_inference_latency_ms": (
                None if not ratio_call_count else self.asr_ratio_spans.latency_ms()
            ),
            "deprecated_fields": {"call_count": "alias of invocation_count"},
        }

    def _time_to_searchable_json(self) -> dict[str, object]:
        return {
            "measures": (
                "elapsed time from one record's capture() commit to the settle() call that made "
                "it searchable, taken from the batch-maximum wait settle() reports on its span; "
                "empty unless the run ingests through capture()+settle() (--ingest capture)"
            ),
            **self.time_to_searchable_ms.json(),
        }

    def _compile_json(self) -> dict[str, object]:
        latency = self._span_latency_json(COMPILE_SPAN)
        return {
            **latency,
            "measures": "Memory.compile latency and the size of the bundle it returned",
            "bundle_chars": self.compile_chars.json(),
            "bundle_items": self.compile_items.json(),
            "bundle_media_items": self.compile_media_items.json(),
        }

    def _nodes_json(
        self,
        nodes: Mapping[str, _Durations],
        breakdowns: Mapping[str, Mapping[_NodeDimensions, _Durations]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in sorted(nodes.items()):
            payload = value.json()
            successful_count = sum(
                durations.count
                for dimensions, durations in breakdowns.get(name, {}).items()
                if dimensions.status == "ok"
            )
            payload["success_count"] = successful_count
            payload["error_count"] = value.count - successful_count
            payload["breakdown"] = [
                {**dimensions.json(), **durations.json()}
                for dimensions, durations in sorted(
                    breakdowns.get(name, {}).items(),
                    key=lambda item: repr(item[0]),
                )
            ]
            result[name] = payload
        return result

    def json(self, question_count: int) -> dict[str, object]:
        run_seconds = self.run.active_seconds() or 0.0
        judge_seconds = self.judge.active_seconds() or 0.0
        total_seconds = run_seconds + judge_seconds
        judge_question_count = len(self.judge_samples)
        measured_samples = self.product_samples | self.judge_samples
        average_denominator = len(measured_samples) or question_count or None
        search_sdk_operation = self._durations_latency_json(
            SEARCH_E2E_SPAN,
            self.diagnostic_nodes.get(SEARCH_E2E_SPAN),
            self.successful_diagnostic_nodes.get(SEARCH_E2E_SPAN),
        )
        search_e2e = {
            **self._durations_latency_json(
                BENCHMARK_DIAGNOSTIC_SPAN,
                self.diagnostic_search,
                (
                    self.successful_diagnostic_search
                    if self.successful_diagnostic_search.count
                    else None
                ),
            ),
            "measurement": "post_answer_warm_store_replay",
            "sdk_operation": search_sdk_operation,
        }
        search_e2e["planned_count"] = search_e2e["attempt_count"]
        search_e2e["success_count"] = search_e2e["count"]
        search_e2e["complete"] = search_e2e["attempt_count"] == search_e2e["count"]
        ask_retrieval = self._span_latency_json(ASK_RETRIEVAL_SPAN)
        token_usage = self.tokens.json(average_denominator or 0)
        token_usage["average_denominator_question_count"] = average_denominator
        return {
            "duration_seconds": {
                "definition": "sum of gap-free active wall-time unions for product and judge",
                "measured_question_count": average_denominator,
                "measured_product_question_count": question_count,
                "measured_judge_question_count": judge_question_count,
                "average_denominator_question_count": average_denominator,
                "total": total_seconds,
                "average": (
                    None if average_denominator is None else total_seconds / average_denominator
                ),
                "mindbridge": run_seconds,
                "judge": judge_seconds,
                "mindbridge_average": (
                    None if not question_count else run_seconds / question_count
                ),
                "judge_average": (
                    None if not judge_question_count else judge_seconds / judge_question_count
                ),
            },
            "ingest": self._ingest_json(),
            "search_e2e": search_e2e,
            "ask_retrieval_core": ask_retrieval,
            "search": {
                **ask_retrieval,
                "deprecated": True,
                "replacement": "ask_retrieval_core",
            },
            "answer": self._answer_json(),
            "asr": self._asr_json(),
            "capture": self._span_latency_json(CAPTURE_SPAN),
            "time_to_searchable_ms": self._time_to_searchable_json(),
            "formation": self._span_latency_json(SETTLE_SPAN),
            "compile": self._compile_json(),
            "nodes": self._nodes_json(self.nodes, self.node_breakdowns),
            "diagnostic": {
                "excluded_from_product_metrics": True,
                "search_e2e": search_e2e,
                "sdk_operation": search_sdk_operation,
                "nodes": self._nodes_json(self.diagnostic_nodes, self.diagnostic_breakdowns),
                "token_usage": {
                    **self.diagnostic_tokens.json(self.diagnostic_search.count),
                    "average_denominator_question_count": self.diagnostic_search.count,
                    "by_module": {
                        name: value.json(
                            self.diagnostic_search.count,
                            compute_seconds=(
                                None
                                if (duration := self.diagnostic_model_durations_by_module.get(name))
                                is None
                                else duration.total_ns / 1_000_000_000
                            ),
                        )
                        for name, value in sorted(self.diagnostic_tokens_by_module.items())
                    },
                },
            },
            "token_usage": {
                **token_usage,
                "by_module": {
                    name: value.json(
                        judge_question_count if name == JUDGE_MODULE else question_count,
                        compute_seconds=(
                            None
                            if (duration := self.model_durations_by_module.get(name)) is None
                            else duration.total_ns / 1_000_000_000
                        ),
                    )
                    for name, value in sorted(self.tokens_by_module.items())
                },
                "product": self._product_tokens_json(question_count),
            },
            "grounding": {
                "media_elided_hits": self.media_elided_hits,
                "dropped_hits": self.dropped_hits,
            },
            "vision": {"failed_batches": self.vision_failed_batches},
        }


@dataclass(frozen=True, slots=True)
class _SpanScope:
    task: str | None
    arm: str | None
    sample: str | None
    purpose: str | None
    parent_operation: str | None
    current_operation: str | None


class EvaluationTelemetry(SpanProcessor):
    """Aggregate evaluation spans online without retaining per-request traces."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._span_scopes: dict[int, _SpanScope] = {}
        self._samples: dict[str, _TaskTelemetry] = {}
        self._tasks: dict[tuple[str, str], _TaskTelemetry] = {}
        self._provider = TracerProvider()
        self._provider.add_span_processor(self)
        self.tracer: Tracer = self._provider.get_tracer(TRACER_NAME)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        attributes = span.attributes or {}
        task = _string_attribute(attributes, BENCHMARK_TASK)
        arm = _string_attribute(attributes, BENCHMARK_ARM)
        sample = _string_attribute(attributes, BENCHMARK_SAMPLE)
        purpose = _string_attribute(attributes, BENCHMARK_PURPOSE)
        parent_operation = _string_attribute(attributes, BENCHMARK_PARENT_OPERATION)
        kind = _string_attribute(attributes, SPAN_KIND)
        parent_context_value = trace.get_current_span(parent_context).get_span_context()
        parent = 0 if parent_context_value is None else parent_context_value.span_id
        with self._lock:
            inherited = self._span_scopes.get(parent)
            if inherited is not None:
                task = task or inherited.task
                arm = arm or inherited.arm
                sample = sample or inherited.sample
                purpose = purpose or inherited.purpose
                parent_operation = parent_operation or inherited.current_operation
            current_operation = (
                span.name
                if kind == "operation"
                else (None if inherited is None else inherited.current_operation)
            )
            scope = _SpanScope(
                task=task,
                arm=arm,
                sample=sample,
                purpose=purpose,
                parent_operation=parent_operation,
                current_operation=current_operation,
            )
            span_context = span.get_span_context()
            if span_context is None:
                return
            self._span_scopes[span_context.span_id] = scope
        # Materialize inherited dimensions on every recorded span. This keeps exported traces
        # attributable even though this processor aggregates them in-process.
        for name, value in (
            (BENCHMARK_TASK, task),
            (BENCHMARK_ARM, arm),
            (BENCHMARK_SAMPLE, sample),
            (BENCHMARK_PURPOSE, purpose),
            (BENCHMARK_PARENT_OPERATION, parent_operation),
        ):
            if value is not None and name not in attributes:
                span.set_attribute(name, value)

    def on_end(self, span: ReadableSpan) -> None:
        span_context = span.get_span_context()
        span_id = 0 if span_context is None else span_context.span_id
        attributes = span.attributes or {}
        with self._lock:
            scope = self._span_scopes.pop(span_id, None)
            task = (None if scope is None else scope.task) or _string_attribute(
                attributes, BENCHMARK_TASK
            )
            arm = (None if scope is None else scope.arm) or _string_attribute(
                attributes, BENCHMARK_ARM
            )
            sample = (None if scope is None else scope.sample) or _string_attribute(
                attributes, BENCHMARK_SAMPLE
            )
            if task is not None:
                selected_arm = arm or DEFAULT_BENCHMARK_ARM
                self._tasks.setdefault((task, selected_arm), _TaskTelemetry()).add(span)
            if sample is not None:
                self._samples.setdefault(sample, _TaskTelemetry()).add(span)

    def result(
        self,
        task: str,
        *,
        arm: str = DEFAULT_BENCHMARK_ARM,
        question_count: int,
    ) -> Mapping[str, object]:
        with self._lock:
            values = self._tasks.get((task, arm), _TaskTelemetry())
            return values.json(question_count)

    def known_arms(self, task: str) -> tuple[str, ...]:
        """Return the arms that emitted at least one span for ``task``."""
        with self._lock:
            return tuple(sorted(arm for candidate, arm in self._tasks if candidate == task))

    def sample_grounding(self, sample_id: str) -> SampleGrounding | None:
        """Return one answer's budget loss, or None when no answer span was recorded."""
        with self._lock:
            values = self._samples.get(sample_id)
        if values is None:
            return None
        return SampleGrounding(
            dropped_hits=values.dropped_hits,
            media_elided_hits=values.media_elided_hits,
        )

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
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def _string_attribute(attributes: Mapping[str, AttributeValue], name: str) -> str | None:
    value = attributes.get(name)
    return value if isinstance(value, str) else None


def _string_tuple_attribute(attributes: Mapping[str, AttributeValue], name: str) -> tuple[str, ...]:
    value = attributes.get(name)
    return (
        tuple(value)
        if isinstance(value, tuple) and all(isinstance(item, str) for item in value)
        else ()
    )


def _span_status(span: ReadableSpan) -> str:
    status = getattr(span, "status", None)
    return "error" if getattr(status, "status_code", None) == StatusCode.ERROR else "ok"


def _distribution_json(
    values: Sequence[float],
    *,
    total_count: int,
    total: float,
) -> dict[str, object]:
    retained = tuple(values)
    return {
        "count": total_count,
        "retained_count": len(retained),
        "complete": total_count == len(retained),
        "average": None if not total_count else total / total_count,
        "p50": percentile(retained, 0.50),
        "p95": percentile(retained, 0.95),
        "p99": percentile(retained, 0.99),
    }


def _observed_distribution_json(
    values: Sequence[float], *, total_count: int, total: float
) -> dict[str, object]:
    result = _distribution_json(values, total_count=total_count, total=total)
    result["retained_average"] = None if not values else total / len(values)
    if not result["complete"]:
        result["average"] = None
    return result


def storage_bytes(root: Path) -> dict[str, int]:
    """Classify one benchmark data root into media, row, and vector bytes."""
    media = rows = vectors = other = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        parents = {parent.name for parent in path.parents}
        if "assets" in parents:
            media += size
        elif "zvec" in parents:
            vectors += size
        elif path.name.startswith("state.sqlite3"):
            rows += size
        else:
            other += size
    return {
        "media": media,
        "rows": rows,
        "vectors": vectors,
        "other": other,
        "total": media + rows + vectors + other,
    }


@dataclass(slots=True)
class _GpuSamples:
    readings: list[tuple[float, float, int, float | None]] = field(default_factory=list)

    def add(
        self,
        sampled_at: float,
        utilization: float,
        memory_used: int,
        power_watts: float | None,
    ) -> None:
        self.readings.append((sampled_at, utilization, memory_used, power_watts))

    def json(self, started: float, stopped: float) -> dict[str, object]:
        readings = sorted(reading for reading in self.readings if reading[0] <= stopped)
        if not readings:
            raise RuntimeError("GPU sample set is empty")
        utilization_area = memory_area = power_area = power_coverage = 0.0
        coverage = 0.0
        for first, second in pairwise(readings):
            left = max(started, first[0])
            right = min(stopped, second[0])
            seconds = max(0.0, right - left)
            if not seconds:
                continue
            coverage += seconds
            utilization_area += (first[1] + second[1]) * 0.5 * seconds
            memory_area += (first[2] + second[2]) * 0.5 * seconds
            if first[3] is not None and second[3] is not None:
                power_area += (first[3] + second[3]) * 0.5 * seconds
                power_coverage += seconds
        latest = readings[-1]
        power_values = tuple(reading[3] for reading in readings if reading[3] is not None)
        return {
            "average_utilization_percent": (utilization_area / coverage if coverage else latest[1]),
            "peak_utilization_percent": max(reading[1] for reading in readings),
            "average_memory_used_bytes": memory_area / coverage if coverage else latest[2],
            "peak_memory_used_bytes": max(reading[2] for reading in readings),
            "average_power_watts": (
                power_area / power_coverage
                if power_coverage
                else latest[3]
                if latest[3] is not None
                else None
            ),
            "peak_power_watts": max(power_values) if power_values else None,
            "estimated_energy_watt_hours": (power_area / 3_600 if power_coverage else None),
            "sample_count": len(readings),
            "power_sample_count": len(power_values),
            "sample_coverage_seconds": coverage,
            "power_coverage_seconds": power_coverage,
        }


class ResourceSampler:
    """Record client CPU, resident memory, storage growth, and sampled GPU load.

    CPU time and peak resident memory come from ``resource.getrusage``, so they need no
    sampling. GPU utilization, memory, and power are instantaneous, so one background poll runs
    only when ``nvidia-smi`` answers at all.
    """

    def __init__(
        self,
        *,
        storage_root: Path | None = None,
        storage_roots: Sequence[Path] = (),
        interval_seconds: float = 0.5,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sampling interval must be positive")
        if storage_root is not None and storage_roots:
            raise ValueError("pass storage_root or storage_roots, not both")
        self._storage_roots = tuple(storage_roots) or (
            () if storage_root is None else (storage_root,)
        )
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._gpu: dict[int, _GpuSamples] = {}
        self._started_cpu_seconds = 0.0
        self._stopped_cpu_seconds: float | None = None
        self._started_wall = 0.0
        self._stopped_wall: float | None = None
        self._started_storage: dict[str, int] | None = None
        self._stopped_storage: dict[str, int] | None = None
        self._gpu_available = False
        self._rapl_start: dict[str, tuple[int, int | None]] | None = None
        self._rapl_joules: float | None = None
        self._rapl_reason: str | None = None

    def __enter__(self) -> ResourceSampler:
        if self._storage_roots:
            self._started_storage = _combined_storage_bytes(self._storage_roots)
        initial = _nvidia_utilization()
        self._started_cpu_seconds = _cpu_seconds()
        self._rapl_start, self._rapl_reason = _rapl_energy_uj()
        self._started_wall = perf_counter()
        self._gpu_available = bool(initial)
        self._record_gpu(initial, sampled_at=self._started_wall)
        if self._gpu_available:
            self._thread = Thread(target=self._poll, name="mindbridge-bench-resources", daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stopped_wall = perf_counter()
        self._stopped_cpu_seconds = _cpu_seconds()
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._interval_seconds + 5.0)
        final = _nvidia_utilization()
        if final:
            self._gpu_available = True
            self._record_gpu(final, sampled_at=self._stopped_wall)
        if self._storage_roots:
            self._stopped_storage = _combined_storage_bytes(self._storage_roots)
        if self._rapl_start is not None:
            end, end_reason = _rapl_energy_uj()
            if end is None:
                self._rapl_reason = self._rapl_reason or end_reason
            else:
                joules_uj, wrap_reason = _rapl_delta_uj(self._rapl_start, end)
                if joules_uj is None:
                    self._rapl_reason = self._rapl_reason or wrap_reason
                else:
                    self._rapl_joules = joules_uj / 1_000_000

    def json(self, *, wall_seconds: float) -> dict[str, object]:
        """Return one JSON-ready resource record for the completed evaluation."""
        ended_wall = self._stopped_wall if self._stopped_wall is not None else perf_counter()
        measured_wall_seconds = (
            max(0.0, ended_wall - self._started_wall)
            if self._started_wall
            else max(0.0, wall_seconds)
        )
        ended_cpu = (
            self._stopped_cpu_seconds if self._stopped_cpu_seconds is not None else _cpu_seconds()
        )
        cpu_seconds = max(0.0, ended_cpu - self._started_cpu_seconds)
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        with self._lock:
            gpu = {
                str(index): samples.json(self._started_wall, ended_wall)
                for index, samples in sorted(self._gpu.items())
            }
        return {
            "measurement": {
                "scope": "client_process_and_system_devices",
                "phase": "product_execution_including_post_answer_search_replay",
                "exclusive_attribution": False,
                "wall_seconds": measured_wall_seconds,
                "storage_scope": "selected_run_directories",
                "storage_root_count": len(self._storage_roots),
                "gpu_sampling_interval_seconds": self._interval_seconds,
                "gpu_values": "sampled system-device values; peaks between polls may be missed",
                "gpu_average_method": "time-weighted trapezoidal integration",
                "gpu_energy_method": "time-integrated sampled power",
            },
            "cpu": {
                "seconds": cpu_seconds,
                "logical_cores": cores,
                "utilization_percent": (
                    None
                    if measured_wall_seconds <= 0
                    else 100.0 * cpu_seconds / measured_wall_seconds / cores
                ),
            },
            "memory": {
                "peak_resident_bytes": _peak_resident_bytes(),
                "scope": "process_lifetime_high_water_mark",
            },
            "storage": _storage_growth(self._started_storage, self._stopped_storage),
            "gpu": gpu if self._gpu_available else None,
            "energy": self._energy_json(gpu),
        }

    def _energy_json(self, gpu: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        """Report package and per-GPU energy, or exactly why neither is readable.

        Never a fabricated number: a value is reported only when its source (Intel RAPL for the
        package, ``nvidia-smi --query-gpu=power.draw`` integrated over the poll interval for the
        GPU) actually answered.
        """
        cpu_joules = self._rapl_joules
        measured_gpu_joules = {
            index: watt_hours * 3_600
            for index, values in gpu.items()
            if isinstance((watt_hours := values.get("estimated_energy_watt_hours")), (int, float))
        }
        gpu_joules = measured_gpu_joules or None
        reasons = [
            reason
            for reason in (
                self._rapl_reason if cpu_joules is None else None,
                None
                if gpu_joules is not None
                else (
                    "nvidia-smi did not report power.draw for any GPU"
                    if self._gpu_available
                    else "no GPU visible to nvidia-smi"
                ),
            )
            if reason
        ]
        available = cpu_joules is not None or gpu_joules is not None
        return {
            "cpu_package_joules": cpu_joules,
            "gpu_joules": gpu_joules,
            "available": available,
            "reason": None if available else "; ".join(reasons),
        }

    def _poll(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            readings = _nvidia_utilization()
            self._record_gpu(readings, sampled_at=perf_counter())

    def _record_gpu(
        self,
        readings: Sequence[tuple[int, float, int, float | None]],
        *,
        sampled_at: float,
    ) -> None:
        with self._lock:
            for index, utilization, memory_used, power_watts in readings:
                self._gpu.setdefault(index, _GpuSamples()).add(
                    sampled_at, utilization, memory_used, power_watts
                )


def _storage_growth(
    started: Mapping[str, int] | None, stopped: Mapping[str, int] | None
) -> dict[str, object] | None:
    if started is None or stopped is None:
        return None
    growth = {key: stopped.get(key, 0) - value for key, value in started.items()}
    total = growth.get("total", 0)
    return {
        "start_bytes": dict(started),
        "end_bytes": dict(stopped),
        "growth_bytes": growth,
        "media_share": None if total <= 0 else growth.get("media", 0) / total,
    }


def _combined_storage_bytes(roots: Sequence[Path]) -> dict[str, int]:
    totals = dict.fromkeys(("media", "rows", "vectors", "other", "total"), 0)
    for root in roots:
        for name, value in storage_bytes(root).items():
            totals[name] += value
    return totals


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _peak_resident_bytes() -> int:
    # getrusage reports ru_maxrss in kilobytes on Linux and in bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def _nvidia_utilization() -> tuple[tuple[int, float, int, float | None], ...]:
    """Return each visible GPU's (index, utilization%, memory bytes, power watts-or-None).

    Power is ``None`` on a card ``nvidia-smi`` cannot meter (reported as ``[N/A]``), which the
    caller must not turn into a fabricated zero.
    """
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode:
        return ()
    devices = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4 or not fields[0].isdecimal():
            continue
        try:
            power = None if fields[3].casefold() in {"", "n/a", "[n/a]"} else float(fields[3])
            devices.append(
                (
                    int(fields[0]),
                    float(fields[1]),
                    int(float(fields[2]) * 1_048_576),
                    power,
                )
            )
        except ValueError:
            continue
    return tuple(devices)


# Overridable so a test can point at a fake sysfs tree instead of the real machine.
_RAPL_ROOT = Path("/sys/class/powercap")


def _rapl_energy_uj(
    root: Path | None = None,
) -> tuple[dict[str, tuple[int, int | None]] | None, str | None]:
    """Read each Intel RAPL package's energy counter and the range it wraps at.

    Per package rather than one sum, because ``energy_uj`` wraps at that package's own
    ``max_energy_range_uj`` and a delta can only be corrected against the counter it came from.
    ``root`` resolves ``_RAPL_ROOT`` at call time (not as a bound default) so a test can point
    at a fake sysfs tree by monkeypatching the module attribute.
    """
    if root is None:
        root = _RAPL_ROOT
    try:
        packages = sorted(root.glob("intel-rapl:[0-9]*/energy_uj"))
    except OSError as error:
        return None, f"cannot list {root}: {error}"
    if not packages:
        return None, f"no intel-rapl packages under {root}"
    readings: dict[str, tuple[int, int | None]] = {}
    for path in packages:
        try:
            energy = int(path.read_text().strip())
        except (OSError, ValueError) as error:
            return None, f"cannot read {path}: {error}"
        readings[path.parent.name] = (energy, _rapl_max_range_uj(path.parent))
    return readings, None


def _rapl_max_range_uj(package: Path) -> int | None:
    """Return what this package's counter wraps at, or `None` when it does not publish it."""
    try:
        return int((package / "max_energy_range_uj").read_text().strip())
    except (OSError, ValueError):
        return None


def _rapl_delta_uj(
    start: Mapping[str, tuple[int, int | None]],
    end: Mapping[str, tuple[int, int | None]],
) -> tuple[int | None, str | None]:
    """Sum each package's own delta, correcting the wrap ``energy_uj`` may have made.

    A counter reading lower at the end than at the start wrapped at its
    ``max_energy_range_uj``, which is tens of minutes on a busy package rather than the hours a
    sweep can run for. Clamping that to zero published 0 J for a run that burned energy, and
    summing the packages before subtracting could hide one package's wrap inside another's
    rise. Corrected per package instead, and refused outright when a package that wrapped does
    not publish its range, because a wrap nothing can correct is not a measurement.

    ponytail: two samples cannot tell one wrap from two, so a package that wrapped more than
    once still undercounts by whole ranges. Accumulate in the poll loop instead if a run ever
    needs to be that long; at a 2 s interval no wrap is missed.
    """
    total = 0
    for name, (start_uj, max_range) in sorted(start.items()):
        if name not in end:
            return None, f"{name} stopped reporting energy_uj mid-run"
        delta = end[name][0] - start_uj
        if delta < 0:
            if max_range is None:
                return None, f"{name} wrapped and publishes no max_energy_range_uj"
            delta += max_range
        total += delta
    return total, None
