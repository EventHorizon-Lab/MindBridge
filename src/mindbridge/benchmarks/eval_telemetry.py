"""Bounded in-process aggregation for evaluation telemetry spans."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from types import TracebackType

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    CAPTURE_TIME_TO_SEARCHABLE,
    GEN_AI_TTFC,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
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
    VISION_BATCHES_FAILED,
    token_modality_attribute,
)
from mindbridge.benchmarks.eval_statistics import percentile

BENCHMARK_TASK = "mindbridge.benchmark.task"
BENCHMARK_SAMPLE = "mindbridge.benchmark.sample"
BENCHMARK_TASK_SPAN = "mindbridge.benchmark.run"
BENCHMARK_ANSWER_SPAN = "mindbridge.benchmark.answer"
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

ANSWER_SPAN = "mindbridge.ask"
# The whole retrieval leg of one `ask`, not the index lookup alone. Pointing this at
# `mindbridge.index.search` would exclude query embedding and content preparation and report a
# real number under the wrong name -- the same error as the prior round's "P50 74.6 min", which
# turned out to be queue depth. Nodes are keyed by span name, so this distribution never mixes
# with `search()`'s `mindbridge.search` operation span, and a run that only calls `search()`
# reports nothing here.
SEARCH_SPAN = "mindbridge.retrieve"
TRANSCRIPTION_SPAN = "mindbridge.model.transcription"
TRANSCRIPTION_MODULE = "transcription"
JUDGE_MODULE = "judge"

# ponytail: per-span durations are retained so p50/p95/p99 are exact rather than estimated.
# One int per span keeps a 100k-question run under a megabyte; past the cap the percentiles
# stop being exact and say so through ``latency_complete``.
_MAX_RETAINED_DURATIONS = 200_000


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
    ttft_seconds: float = 0.0
    ttfc_count: int = 0
    ttfc_seconds: float = 0.0
    durations_ns: list[int] = field(default_factory=list)
    first_start_ns: int | None = None
    last_end_ns: int | None = None

    def add(self, span: ReadableSpan) -> None:
        self.count += 1
        duration_ns = _duration_ns(span)
        self.total_ns += duration_ns
        if len(self.durations_ns) < _MAX_RETAINED_DURATIONS:
            self.durations_ns.append(duration_ns)
        if span.start_time is not None:
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
        ttft = _float_attribute(attributes, MODEL_TTFT)
        if ttft is not None:
            self.ttft_count += 1
            self.ttft_seconds += ttft
        ttfc = _float_attribute(attributes, GEN_AI_TTFC)
        if ttfc is not None:
            self.ttfc_count += 1
            self.ttfc_seconds += ttfc

    def wall_seconds(self) -> float | None:
        """Return the elapsed time from the first span start to the last span end."""
        if self.first_start_ns is None or self.last_end_ns is None:
            return None
        return max(0, self.last_end_ns - self.first_start_ns) / 1_000_000_000

    def latency_ms(self) -> dict[str, object]:
        """Return exact quantiles over the retained per-span durations."""
        return {
            "count": len(self.durations_ns),
            "complete": self.count == len(self.durations_ns),
            "p50": _ms(percentile(self.durations_ns, 0.50)),
            "p95": _ms(percentile(self.durations_ns, 0.95)),
            "p99": _ms(percentile(self.durations_ns, 0.99)),
        }

    def json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "count": self.count,
            "total_seconds": self.total_ns / 1_000_000_000,
            "average_ms": self.total_ns / self.count / 1_000_000,
            "latency_ms": self.latency_ms(),
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
class _Samples:
    """A bounded list of scalar observations, reported as exact count/average/percentiles."""

    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if len(self.values) < _MAX_RETAINED_DURATIONS:
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
    media_elided_hits: int = 0
    dropped_hits: int = 0
    vision_failed_batches: int = 0
    ingest_items: int = 0
    # Fast-plane and compiler measurements: populated only when the run actually exercises
    # `capture()`/`settle()` (the `--ingest capture` path) or the `compile` answering arm.
    time_to_searchable_ms: _Samples = field(default_factory=_Samples)
    compile_chars: _Samples = field(default_factory=_Samples)
    compile_items: _Samples = field(default_factory=_Samples)
    compile_media_items: _Samples = field(default_factory=_Samples)

    def add(self, span: ReadableSpan) -> None:
        attributes = span.attributes or {}
        if span.name == BENCHMARK_TASK_SPAN:
            self.run_ns += _duration_ns(span)
        elif span.name == BENCHMARK_INGEST_SPAN:
            self.ingest_items += _int_attribute(attributes, BENCHMARK_INGEST_ITEMS) or 0
        elif span.name == SETTLE_SPAN:
            self._add_time_to_searchable(attributes)
        elif span.name == BENCHMARK_COMPILE_SPAN:
            self._add_compile_bundle(attributes)
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
            self.media_elided_hits += _int_attribute(attributes, GROUNDING_MEDIA_ELIDED) or 0
            self.dropped_hits += _int_attribute(attributes, GROUNDING_HITS_DROPPED) or 0
            self.vision_failed_batches += _int_attribute(attributes, VISION_BATCHES_FAILED) or 0
            module = _string_attribute(attributes, MODEL_MODULE) or "unknown"
            self.tokens_by_module.setdefault(module, _Tokens()).add(attributes)

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
        """Sum the modules MindBridge itself spends, leaving the judge out.

        `token_usage.complete` is an AND over every module, so one judge request without usage
        nulled the whole task's total while the product's own cost -- embedding, generation,
        transcription, description -- was fully reported. The cost axis needs that number.
        """
        modules = {
            name: value for name, value in self.tokens_by_module.items() if name != JUDGE_MODULE
        }
        complete = all(
            value.complete and value.expected_request_count == value.reported_request_count
            for value in modules.values()
        )
        total = sum(value.total_tokens for value in modules.values())
        return {
            "modules": sorted(modules),
            "complete": complete,
            "total_tokens": total if complete else None,
            "average_tokens": total / question_count if complete and question_count else None,
        }

    def _ingest_json(self) -> dict[str, object]:
        """Report accepted input to durable, searchable memory plus sustained throughput.

        ``mindbridge.add``/``add_many`` commit SQLite, flush Zvec, and acknowledge the outbox
        before returning, so the wall clock of the traced call is the durable-and-searchable
        latency the design doc asks for rather than a call-accepted latency.
        """
        node = self.nodes.get(BENCHMARK_INGEST_SPAN)
        wall_seconds = None if node is None else node.wall_seconds()
        return {
            "measures": (
                "accepted input to durable and searchable memory: mindbridge.add_many wall "
                "clock, which returns only after the SQLite commit, the Zvec flush, and the "
                "search-index outbox acknowledgement"
            ),
            "span": BENCHMARK_INGEST_SPAN,
            "call_count": 0 if node is None else node.count,
            "item_count": self.ingest_items,
            "compute_seconds": 0.0 if node is None else node.total_ns / 1_000_000_000,
            "wall_seconds": wall_seconds,
            "call_latency_ms": None if node is None else node.latency_ms(),
            "item_latency_ms": (
                None
                if node is None or not self.ingest_items
                else node.total_ns / self.ingest_items / 1_000_000
            ),
            "items_per_second": (
                None
                if not wall_seconds or not self.ingest_items
                else self.ingest_items / wall_seconds
            ),
        }

    def _span_latency_json(self, span_name: str) -> dict[str, object]:
        node = self.nodes.get(span_name)
        wall_seconds = None if node is None else node.wall_seconds()
        count = 0 if node is None else node.count
        return {
            "span": span_name,
            "count": count,
            "wall_seconds": wall_seconds,
            "latency_ms": None if node is None else node.latency_ms(),
            "throughput_per_second": None
            if not wall_seconds or not count
            else count / wall_seconds,
        }

    def _answer_json(self) -> dict[str, object]:
        """Report end-to-end answer latency, and TTFT only when the backend streamed it."""
        result = self._span_latency_json(ANSWER_SPAN)
        generation = self.nodes.get("mindbridge.model.generation")
        streamed = generation is not None and generation.ttft_count > 0
        result["streaming"] = streamed
        result["time_to_first_token_ms"] = (
            {
                "count": generation.ttft_count,
                "average": generation.ttft_seconds / generation.ttft_count * 1_000,
            }
            if streamed and generation is not None
            else None
        )
        return result

    def _asr_json(self) -> dict[str, object] | None:
        """Report ASR audio-seconds per wall-second and its inference latency."""
        tokens = self.tokens_by_module.get(TRANSCRIPTION_MODULE)
        node = self.nodes.get(TRANSCRIPTION_SPAN)
        if tokens is None or node is None or not node.total_ns:
            return None
        compute_seconds = node.total_ns / 1_000_000_000
        return {
            "span": TRANSCRIPTION_SPAN,
            "call_count": node.count,
            "audio_seconds": tokens.audio_seconds,
            "compute_seconds": compute_seconds,
            "real_time_factor": tokens.audio_seconds / compute_seconds,
            "inference_latency_ms": node.latency_ms(),
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
            "ingest": self._ingest_json(),
            "search": self._span_latency_json(SEARCH_SPAN),
            "answer": self._answer_json(),
            "asr": self._asr_json(),
            "capture": self._span_latency_json(CAPTURE_SPAN),
            "time_to_searchable_ms": self._time_to_searchable_json(),
            "formation": self._span_latency_json(SETTLE_SPAN),
            "compile": self._compile_json(),
            "nodes": {name: value.json() for name, value in sorted(self.nodes.items())},
            "token_usage": {
                **self.tokens.json(question_count),
                "by_module": {
                    name: value.json(question_count)
                    for name, value in sorted(self.tokens_by_module.items())
                },
                "product": self._product_tokens_json(question_count),
            },
            "grounding": {
                "media_elided_hits": self.media_elided_hits,
                "dropped_hits": self.dropped_hits,
            },
            # Write-path loss, not answer-path loss, so it is its own block: a describer whose
            # reply could not be used leaves a media memory with no full-text document, which is
            # invisible in a token count because a failed batch reports no tokens.
            "vision": {"failed_batches": self.vision_failed_batches},
        }


class EvaluationTelemetry(SpanProcessor):
    """Aggregate evaluation spans online without retaining per-request traces."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._span_tasks: dict[int, str] = {}
        self._span_samples: dict[int, str] = {}
        self._samples: dict[str, _TaskTelemetry] = {}
        self._tasks: dict[str, _TaskTelemetry] = {}
        self._provider = TracerProvider()
        self._provider.add_span_processor(self)
        self.tracer: Tracer = self._provider.get_tracer(TRACER_NAME)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        attributes = span.attributes or {}
        task = _string_attribute(attributes, BENCHMARK_TASK)
        sample = _string_attribute(attributes, BENCHMARK_SAMPLE)
        parent_context_value = trace.get_current_span(parent_context).get_span_context()
        parent = 0 if parent_context_value is None else parent_context_value.span_id
        with self._lock:
            if task is None:
                task = self._span_tasks.get(parent)
            if sample is None:
                sample = self._span_samples.get(parent)
            span_context = span.get_span_context()
            if span_context is None:
                return
            if task is not None:
                self._span_tasks[span_context.span_id] = task
            if sample is not None:
                self._span_samples[span_context.span_id] = sample

    def on_end(self, span: ReadableSpan) -> None:
        span_context = span.get_span_context()
        span_id = 0 if span_context is None else span_context.span_id
        attributes = span.attributes or {}
        with self._lock:
            task = self._span_tasks.pop(span_id, None) or _string_attribute(
                attributes, BENCHMARK_TASK
            )
            sample = self._span_samples.pop(span_id, None) or _string_attribute(
                attributes, BENCHMARK_SAMPLE
            )
            if task is not None:
                self._tasks.setdefault(task, _TaskTelemetry()).add(span)
            if sample is not None:
                self._samples.setdefault(sample, _TaskTelemetry()).add(span)

    def result(self, task: str, *, question_count: int) -> Mapping[str, object]:
        with self._lock:
            values = self._tasks.get(task, _TaskTelemetry())
            return values.json(question_count)

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
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _string_attribute(attributes: Mapping[str, AttributeValue], name: str) -> str | None:
    value = attributes.get(name)
    return value if isinstance(value, str) else None


def _ms(nanoseconds: float | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / 1_000_000


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
class _GpuPeak:
    utilization_percent: float = 0.0
    memory_used_bytes: int = 0
    samples: int = 0
    # ponytail: rectangle-rule integral of periodic `power.draw` samples -- approximate, not a
    # calibrated energy meter, but the only per-GPU energy source `nvidia-smi` exposes.
    energy_joules: float = 0.0


class ResourceSampler:
    """Record CPU, resident memory, storage growth, and GPU peaks for one evaluation.

    CPU time and peak resident memory come from ``resource.getrusage``, so they need no
    sampling. Only GPU utilization is instantaneous, so a single background poll runs and
    only when ``nvidia-smi`` answers at all.
    """

    def __init__(self, *, storage_root: Path | None = None, interval_seconds: float = 2.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sampling interval must be positive")
        self._storage_root = storage_root
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._gpu: dict[int, _GpuPeak] = {}
        self._started_cpu_seconds = 0.0
        self._started_storage: dict[str, int] | None = None
        self._stopped_storage: dict[str, int] | None = None
        self._gpu_available = False
        self._gpu_power_reported = False
        self._rapl_start_uj: int | None = None
        self._rapl_end_uj: int | None = None
        self._rapl_reason: str | None = None

    def __enter__(self) -> ResourceSampler:
        self._started_cpu_seconds = _cpu_seconds()
        if self._storage_root is not None:
            self._started_storage = storage_bytes(self._storage_root)
        self._rapl_start_uj, self._rapl_reason = _rapl_energy_uj()
        self._gpu_available = bool(_nvidia_utilization())
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
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._interval_seconds + 5.0)
        if self._storage_root is not None:
            self._stopped_storage = storage_bytes(self._storage_root)
        if self._rapl_start_uj is not None:
            self._rapl_end_uj, end_reason = _rapl_energy_uj()
            self._rapl_reason = self._rapl_reason or end_reason

    def json(self, *, wall_seconds: float) -> dict[str, object]:
        """Return one JSON-ready resource record for the completed evaluation."""
        cpu_seconds = max(0.0, _cpu_seconds() - self._started_cpu_seconds)
        cores = os.cpu_count() or 1
        with self._lock:
            gpu = {
                str(index): {
                    "peak_utilization_percent": peak.utilization_percent,
                    "peak_memory_used_bytes": peak.memory_used_bytes,
                    "sample_count": peak.samples,
                }
                for index, peak in sorted(self._gpu.items())
            }
        return {
            "cpu": {
                "seconds": cpu_seconds,
                "logical_cores": cores,
                "utilization_percent": (
                    None if wall_seconds <= 0 else 100.0 * cpu_seconds / wall_seconds / cores
                ),
            },
            "memory": {"peak_resident_bytes": _peak_resident_bytes()},
            "storage": _storage_growth(self._started_storage, self._stopped_storage),
            "gpu": gpu if self._gpu_available else None,
            "energy": self._energy_json(),
        }

    def _energy_json(self) -> dict[str, object]:
        """Report package and per-GPU energy, or exactly why neither is readable.

        Never a fabricated number: a value is reported only when its source (Intel RAPL for the
        package, ``nvidia-smi --query-gpu=power.draw`` integrated over the poll interval for the
        GPU) actually answered.
        """
        cpu_joules = (
            None
            if self._rapl_start_uj is None or self._rapl_end_uj is None
            else max(0, self._rapl_end_uj - self._rapl_start_uj) / 1_000_000
        )
        with self._lock:
            gpu_joules = (
                {str(index): peak.energy_joules for index, peak in sorted(self._gpu.items())}
                if self._gpu_available and self._gpu_power_reported
                else None
            )
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
        while True:
            for index, utilization, memory_used, power_watts in _nvidia_utilization():
                with self._lock:
                    peak = self._gpu.setdefault(index, _GpuPeak())
                    peak.utilization_percent = max(peak.utilization_percent, utilization)
                    peak.memory_used_bytes = max(peak.memory_used_bytes, memory_used)
                    peak.samples += 1
                    if power_watts is not None:
                        self._gpu_power_reported = True
                        peak.energy_joules += power_watts * self._interval_seconds
            if self._stop.wait(self._interval_seconds):
                return


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


def _cpu_seconds() -> float:
    return sum(
        usage.ru_utime + usage.ru_stime
        for usage in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def _peak_resident_bytes() -> int:
    # getrusage reports ru_maxrss in kilobytes on Linux and in bytes on macOS.
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
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
            power = float(fields[3])
        except ValueError:
            power = None
        try:
            devices.append(
                (int(fields[0]), float(fields[1]), int(float(fields[2]) * 1_048_576), power)
            )
        except ValueError:
            continue
    return tuple(devices)


# Overridable so a test can point at a fake sysfs tree instead of the real machine.
_RAPL_ROOT = Path("/sys/class/powercap")


def _rapl_energy_uj(root: Path | None = None) -> tuple[int | None, str | None]:
    """Sum Intel RAPL package energy counters, or report exactly why none could be read.

    ``energy_uj`` wraps at each package's ``max_energy_range_uj``; one evaluation run is assumed
    short enough not to wrap, so a summed delta is reported as-is rather than detected and
    corrected. ``root`` resolves ``_RAPL_ROOT`` at call time (not as a bound default) so a test
    can point at a fake sysfs tree by monkeypatching the module attribute.
    """
    if root is None:
        root = _RAPL_ROOT
    try:
        packages = sorted(root.glob("intel-rapl:[0-9]*/energy_uj"))
    except OSError as error:
        return None, f"cannot list {root}: {error}"
    if not packages:
        return None, f"no intel-rapl packages under {root}"
    total = 0
    for path in packages:
        try:
            total += int(path.read_text().strip())
        except (OSError, ValueError) as error:
            return None, f"cannot read {path}: {error}"
    return total, None
