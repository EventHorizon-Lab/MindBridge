"""Narrow Prometheus snapshots for optional vLLM benchmark attribution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.request import Request, urlopen

HISTOGRAMS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_queue_time_seconds",
)
COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:mm_cache_queries_total",
    "vllm:mm_cache_hits_total",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)

_SELECTED = frozenset(
    (
        *COUNTERS,
        *GAUGES,
        *(f"{name}_sum" for name in HISTOGRAMS),
        *(f"{name}_count" for name in HISTOGRAMS),
    )
)
_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?:[^\"}]|\"(?:\\.|[^\"\\])*\")*\})?"
    r"\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    captured_at: str
    values: Mapping[str, float]
    series_counts: Mapping[str, int]
    error_type: str | None = None

    @property
    def available(self) -> bool:
        return self.error_type is None


def capture_metrics(url: str, *, timeout_seconds: float = 5.0) -> MetricsSnapshot:
    """Fetch and aggregate only the vLLM metrics used by the benchmark report."""
    captured_at = datetime.now(timezone.utc).isoformat()
    try:
        request = Request(
            url,
            headers={
                "Accept": "text/plain; version=0.0.4",
                "User-Agent": "mindbridge-benchmark-metrics/1",
            },
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise ValueError("metrics response is too large")
        values, counts = parse_prometheus(payload.decode("utf-8"))
        return MetricsSnapshot(captured_at, values, counts)
    except (OSError, UnicodeError, ValueError) as error:
        return MetricsSnapshot(captured_at, {}, {}, type(error).__name__)


def metrics_window(
    url: str,
    before: MetricsSnapshot,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Take the closing snapshot and report process-global deltas for the window."""
    after = capture_metrics(url, timeout_seconds=timeout_seconds)
    result: dict[str, object] = {
        "scope": "server_process_global",
        "exclusive_attribution": False,
        "shared_traffic_may_be_included": True,
        "metrics_url": url,
        "started_at": before.captured_at,
        "ended_at": after.captured_at,
    }
    if not before.available or not after.available:
        result.update(
            status="unavailable",
            start_error_type=before.error_type,
            end_error_type=after.error_type,
        )
        return result

    resets: list[str] = []
    counters: dict[str, object] = {}
    for name in COUNTERS:
        delta = _counter_delta(before, after, name)
        if delta is None and name in before.values and name in after.values:
            resets.append(name)
        if name in before.values or name in after.values:
            counters[name] = {
                "start": before.values.get(name),
                "end": after.values.get(name),
                "delta": delta,
            }

    histograms: dict[str, object] = {}
    for name in HISTOGRAMS:
        count_delta = _counter_delta(before, after, f"{name}_count")
        sum_delta = _counter_delta(before, after, f"{name}_sum")
        if (
            count_delta is None
            and f"{name}_count" in before.values
            and f"{name}_count" in after.values
        ):
            resets.append(f"{name}_count")
        if sum_delta is None and f"{name}_sum" in before.values and f"{name}_sum" in after.values:
            resets.append(f"{name}_sum")
        if count_delta is not None or sum_delta is not None:
            histograms[name] = {
                "count_delta": count_delta,
                "sum_delta": sum_delta,
                "mean": (
                    None
                    if count_delta is None or sum_delta is None or count_delta <= 0
                    else sum_delta / count_delta
                ),
                "unit": "seconds",
            }

    gauges = {
        name: {
            "start_sum": before.values.get(name),
            "end_sum": after.values.get(name),
            "start_series_count": before.series_counts.get(name),
            "end_series_count": after.series_counts.get(name),
            "start_average": _series_average(before, name),
            "end_average": _series_average(after, name),
        }
        for name in GAUGES
        if name in before.values or name in after.values
    }
    result.update(
        status="partial" if resets else "ok",
        request_count_delta=_counter_delta(before, after, "vllm:request_success_total"),
        counters=counters,
        histograms=histograms,
        gauges=gauges,
        counter_resets=sorted(set(resets)),
    )
    return result


def parse_prometheus(payload: str) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate a bounded whitelist of Prometheus text samples across label sets."""
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    for raw in payload.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None or match.group("name") not in _SELECTED:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        name = match.group("name")
        values[name] = values.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1
    return values, counts


def _counter_delta(before: MetricsSnapshot, after: MetricsSnapshot, name: str) -> float | None:
    start, end = before.values.get(name), after.values.get(name)
    if start is None or end is None or end < start:
        return None
    return end - start


def _series_average(snapshot: MetricsSnapshot, name: str) -> float | None:
    count = snapshot.series_counts.get(name, 0)
    return None if not count or name not in snapshot.values else snapshot.values[name] / count
