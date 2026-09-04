"""Checks for the five reported metric families and the mandatory result controls.

No benchmark corpus exists on this machine, so every check drives the metric calculation from
synthetic spans and synthetic samples with hand-computed expected values.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mindbridge import AssetRef, ContextBudget, MemoryType, Modality, SearchHit
from mindbridge._telemetry import (
    CAPTURE_TIME_TO_SEARCHABLE,
    MODEL_MODULE,
    MODEL_TTFT,
    SPAN_KIND,
    TOKEN_AUDIO_SECONDS,
)
from mindbridge.benchmarks import eval as eval_module
from mindbridge.benchmarks import eval_telemetry as eval_telemetry_module
from mindbridge.benchmarks.eval import NOISE_FLOOR, SampleResult
from mindbridge.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
    MemoryItem,
)
from mindbridge.benchmarks.eval_cache import EvidenceInterval
from mindbridge.benchmarks.eval_statistics import ScoredValue, percentile
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_COMPILE_CHARS,
    BENCHMARK_COMPILE_ITEMS,
    BENCHMARK_COMPILE_MEDIA_ITEMS,
    BENCHMARK_COMPILE_SPAN,
    BENCHMARK_INGEST_ITEMS,
    BENCHMARK_INGEST_SPAN,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    SEARCH_SPAN,
    EvaluationTelemetry,
    ResourceSampler,
    storage_bytes,
)
from mindbridge.context import compile_context


def _span(
    name: str,
    *,
    start_ms: float = 0.0,
    end_ms: float,
    kind: str = "stage",
    task: str = "fixture",
    attributes: Mapping[str, object] | None = None,
) -> ReadableSpan:
    """Build the minimum ``ReadableSpan`` surface the span processor consumes."""
    return cast(
        ReadableSpan,
        SimpleNamespace(
            name=name,
            attributes={BENCHMARK_TASK: task, SPAN_KIND: kind, **dict(attributes or {})},
            start_time=int(start_ms * 1_000_000),
            end_time=int(end_ms * 1_000_000),
            get_span_context=lambda: None,
        ),
    )


def _aggregate(spans: Sequence[ReadableSpan], *, question_count: int = 1) -> Mapping[str, object]:
    telemetry = EvaluationTelemetry()
    try:
        for span in spans:
            telemetry.on_end(span)
        return telemetry.result("fixture", question_count=question_count)
    finally:
        telemetry.close()


def _sample(
    question_id: str,
    *,
    sources: Sequence[str],
    gold: Sequence[str],
    candidate_count: int,
    score: float = 1.0,
    gold_key: str = "evidence_ids",
    unresolved: Sequence[str] = (),
) -> SampleResult:
    return SampleResult(
        task="fixture",
        benchmark="Fixture",
        dataset_sha256="d" * 64,
        evaluation_sha256="e" * 64,
        unit_id="unit",
        question_id=question_id,
        prediction="answer",
        parsed_choice=None,
        score=score,
        exact_match=None,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=tuple(sources),
        candidate_count=candidate_count,
        ingest_failure_count=0,
        error_code=None,
        metadata={gold_key: list(gold), "unresolved_evidence_ids": list(unresolved)},
        evidence=tuple(EvidenceInterval(f"m-{name}", name, None, None) for name in sources),
        ranked_source_ids=tuple(sources),
        metrics={"accuracy": score},
    )


def _arguments(**overrides: object) -> eval_module._Arguments:
    values: dict[str, object] = {
        "seed": 7,
        "bootstrap_samples": 32,
        "recall_limit": 20,
        "blind": False,
    }
    values.update(overrides)
    return cast(eval_module._Arguments, SimpleNamespace(**values))


# --- family 1: ingestion latency to durable, searchable memory, plus throughput -------------


def test_ingest_reports_durable_searchable_latency_and_sustained_throughput() -> None:
    result = _aggregate(
        (
            _span(
                BENCHMARK_INGEST_SPAN,
                start_ms=0,
                end_ms=100,
                attributes={BENCHMARK_INGEST_ITEMS: 3},
            ),
            _span(
                BENCHMARK_INGEST_SPAN,
                start_ms=200,
                end_ms=400,
                attributes={BENCHMARK_INGEST_ITEMS: 5},
            ),
        )
    )

    ingest = cast(Mapping[str, object], result["ingest"])
    assert ingest["span"] == BENCHMARK_INGEST_SPAN
    assert ingest["call_count"] == 2
    assert ingest["item_count"] == 8
    # 300 ms of durable write time inside a 400 ms window: the two must not be interchanged.
    assert cast(float, ingest["compute_seconds"]) == pytest.approx(0.3)
    assert cast(float, ingest["wall_seconds"]) == pytest.approx(0.4)
    assert cast(float, ingest["item_latency_ms"]) == pytest.approx(37.5)
    assert cast(float, ingest["items_per_second"]) == pytest.approx(20.0)
    latency = cast(Mapping[str, object], ingest["call_latency_ms"])
    assert cast(float, latency["p50"]) == pytest.approx(150.0)
    assert "durable" in str(ingest["measures"])
    assert "outbox" in str(ingest["measures"])


async def test_ingest_traces_one_span_per_accepted_batch_with_its_item_count() -> None:
    class _Memory:
        def __init__(self) -> None:
            self.batches: list[int] = []

        async def add_many(self, contents: Sequence[object], **_kwargs: object) -> tuple[Any, ...]:
            self.batches.append(len(contents))
            return tuple(contents)

    telemetry = EvaluationTelemetry()
    memory = _Memory()
    items = tuple(MemoryItem(f"source-{index}", (f"text {index}",)) for index in range(5))
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            failures = await eval_module._ingest(
                cast(Any, memory),
                items,
                batch_size=2,
                tracer=telemetry.tracer,
            )
        result = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    assert failures == 0
    assert memory.batches == [2, 2, 1]
    ingest = cast(Mapping[str, object], result["ingest"])
    assert ingest["call_count"] == 3
    assert ingest["item_count"] == 5
    assert cast(float, ingest["items_per_second"]) > 0.0


# --- family 2: search latency at p50/p95/p99 with throughput --------------------------------


def test_search_latency_reports_p50_p95_and_p99_for_the_named_span() -> None:
    result = _aggregate(
        tuple(_span(SEARCH_SPAN, start_ms=0, end_ms=duration) for duration in (10, 20, 30, 40))
    )

    search = cast(Mapping[str, object], result["search"])
    # The whole retrieval leg of one ask, not the narrower index lookup inside it.
    assert search["span"] == "mindbridge.retrieve"
    assert search["count"] == 4
    latency = cast(Mapping[str, object], search["latency_ms"])
    assert latency["complete"] is True
    assert cast(float, latency["p50"]) == pytest.approx(25.0)
    assert cast(float, latency["p95"]) == pytest.approx(38.5)
    assert cast(float, latency["p99"]) == pytest.approx(39.7)
    # Four searches inside a 40 ms observation window, not a queue depth.
    assert cast(float, search["throughput_per_second"]) == pytest.approx(100.0)


def test_answer_latency_names_the_quantity_it_measures() -> None:
    samples = tuple(
        _sample(str(index), sources=(), gold=(), candidate_count=0) for index in range(4)
    )
    samples = tuple(
        replace(sample, latency_ms=latency)
        for sample, latency in zip(samples, (10.0, 20.0, 30.0, 40.0), strict=True)
    )
    task = cast(Any, SimpleNamespace(spec=SimpleNamespace(name="fixture")))

    metrics = eval_module._metrics(task, samples, _arguments())
    latency = cast(Mapping[str, object], metrics["answer_latency_ms"])

    assert "memory.ask" in str(latency["measures"])
    assert "queue depth" in str(latency["measures"])
    assert cast(float, latency["p50"]) == pytest.approx(25.0)
    assert cast(float, latency["p99"]) == pytest.approx(39.7)


# --- family 3: ASR real-time factor and inference latency -----------------------------------


def test_asr_reports_audio_seconds_per_wall_second_and_inference_latency() -> None:
    result = _aggregate(
        (
            _span(
                "mindbridge.model.transcription",
                start_ms=0,
                end_ms=500,
                kind="model",
                attributes={MODEL_MODULE: "transcription", TOKEN_AUDIO_SECONDS: 5.0},
            ),
        )
    )

    asr = cast(Mapping[str, object], result["asr"])
    assert cast(float, asr["audio_seconds"]) == pytest.approx(5.0)
    assert cast(float, asr["compute_seconds"]) == pytest.approx(0.5)
    # Five seconds of audio transcribed in half a second of wall clock.
    assert cast(float, asr["real_time_factor"]) == pytest.approx(10.0)
    latency = cast(Mapping[str, object], asr["inference_latency_ms"])
    assert cast(float, latency["p99"]) == pytest.approx(500.0)


def test_asr_block_is_absent_when_no_transcription_ran() -> None:
    result = _aggregate((_span(SEARCH_SPAN, end_ms=5),))

    assert result["asr"] is None


# --- family 4: answer latency and time to first token when the backend streams ---------------


def test_time_to_first_token_is_reported_only_when_the_backend_streams() -> None:
    silent = _aggregate(
        (
            _span("mindbridge.ask", end_ms=200, kind="operation"),
            _span(
                "mindbridge.model.generation",
                end_ms=150,
                kind="model",
                attributes={MODEL_MODULE: "generation"},
            ),
        )
    )
    streaming = _aggregate(
        (
            _span("mindbridge.ask", end_ms=200, kind="operation"),
            _span(
                "mindbridge.model.generation",
                end_ms=150,
                kind="model",
                attributes={MODEL_MODULE: "generation", MODEL_TTFT: 0.025},
            ),
        )
    )

    quiet_answer = cast(Mapping[str, object], silent["answer"])
    assert quiet_answer["streaming"] is False
    assert quiet_answer["time_to_first_token_ms"] is None
    loud_answer = cast(Mapping[str, object], streaming["answer"])
    assert loud_answer["streaming"] is True
    first_token = cast(Mapping[str, float], loud_answer["time_to_first_token_ms"])
    assert first_token["average"] == pytest.approx(25.0)
    assert cast(
        float, cast(Mapping[str, object], loud_answer["latency_ms"])["p50"]
    ) == pytest.approx(200.0)


# --- family 5: CPU, memory, storage growth, and GPU -----------------------------------------


def test_resource_sampler_reports_cpu_memory_and_media_dominated_storage_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._nvidia_utilization",
        lambda: (),
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "zvec").mkdir()
    with ResourceSampler(storage_root=tmp_path) as sampler:
        (tmp_path / "assets" / "clip.bin").write_bytes(b"0" * 9_000)
        (tmp_path / "zvec" / "vectors.bin").write_bytes(b"0" * 1_000)
        sum(range(200_000))

    resources = sampler.json(wall_seconds=1.0)
    cpu = cast(Mapping[str, object], resources["cpu"])
    assert cast(float, cpu["seconds"]) >= 0.0
    assert cast(int, cpu["logical_cores"]) >= 1
    assert cast(int, cast(Mapping[str, object], resources["memory"])["peak_resident_bytes"]) > 0
    storage = cast(Mapping[str, object], resources["storage"])
    growth = cast(Mapping[str, int], storage["growth_bytes"])
    assert growth == {"media": 9_000, "rows": 0, "vectors": 1_000, "other": 0, "total": 10_000}
    assert cast(float, storage["media_share"]) == pytest.approx(0.9)
    assert resources["gpu"] is None


def test_storage_bytes_separates_media_rows_and_vectors(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "zvec" / "nested").mkdir(parents=True)
    (tmp_path / "assets" / "a.mp4").write_bytes(b"x" * 5)
    (tmp_path / "zvec" / "nested" / "segment").write_bytes(b"x" * 3)
    (tmp_path / "state.sqlite3").write_bytes(b"x" * 2)
    (tmp_path / "state.sqlite3-wal").write_bytes(b"x" * 4)
    (tmp_path / "notes.txt").write_bytes(b"x")

    assert storage_bytes(tmp_path) == {
        "media": 5,
        "rows": 6,
        "vectors": 3,
        "other": 1,
        "total": 15,
    }


def test_resource_sampler_integrates_gpu_power_into_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._nvidia_utilization",
        lambda: ((0, 50.0, 1024, 120.0),),
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._rapl_energy_uj",
        lambda root=None: (None, "no intel-rapl packages under /sys/class/powercap"),
    )
    with ResourceSampler(interval_seconds=0.05) as sampler:
        time.sleep(0.12)

    energy = cast(Mapping[str, object], sampler.json(wall_seconds=1.0)["energy"])
    gpu_joules = cast(Mapping[str, float], energy["gpu_joules"])
    # Rectangle-rule integral of at least two 120 W samples over ~0.05 s each.
    assert gpu_joules["0"] > 0.0
    assert energy["cpu_package_joules"] is None
    assert energy["available"] is True
    assert energy["reason"] is None


def test_resource_sampler_reports_unavailable_energy_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._nvidia_utilization",
        lambda: (),
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._rapl_energy_uj",
        lambda root=None: (None, "no intel-rapl packages under /sys/class/powercap"),
    )
    with ResourceSampler(interval_seconds=0.05) as sampler:
        pass

    energy = cast(Mapping[str, object], sampler.json(wall_seconds=1.0)["energy"])
    assert energy == {
        "cpu_package_joules": None,
        "gpu_joules": None,
        "available": False,
        "reason": (
            "no intel-rapl packages under /sys/class/powercap; no GPU visible to nvidia-smi"
        ),
    }


def _rapl_tree(tmp_path: Path, packages: Sequence[tuple[int, int | None]]) -> Path:
    """Write a fake powercap tree, one package per `(energy_uj, max_energy_range_uj)` pair."""
    root = tmp_path / "powercap"
    for index, (energy, max_range) in enumerate(packages):
        package = root / f"intel-rapl:{index}"
        package.mkdir(parents=True)
        (package / "energy_uj").write_text(str(energy))
        if max_range is not None:
            (package / "max_energy_range_uj").write_text(str(max_range))
    return root


def test_rapl_energy_reads_each_package_with_the_range_it_wraps_at(tmp_path: Path) -> None:
    root = _rapl_tree(tmp_path, ((1_000_000, 262_143_328_850), (2_500_000, None)))

    readings, reason = eval_telemetry_module._rapl_energy_uj(root)

    assert readings == {
        "intel-rapl:0": (1_000_000, 262_143_328_850),
        "intel-rapl:1": (2_500_000, None),
    }
    assert reason is None


def test_rapl_deltas_are_summed_per_package(tmp_path: Path) -> None:
    start = {"intel-rapl:0": (1_000_000, 100_000_000), "intel-rapl:1": (500_000, 100_000_000)}
    end = {"intel-rapl:0": (3_000_000, 100_000_000), "intel-rapl:1": (900_000, 100_000_000)}

    total, reason = eval_telemetry_module._rapl_delta_uj(start, end)

    assert total == 2_400_000
    assert reason is None


def test_a_wrapped_rapl_counter_is_corrected_against_its_own_range() -> None:
    """A run longer than a package's wrap period reported 0 J for real energy.

    `energy_uj` wraps at `max_energy_range_uj` -- tens of minutes on a busy package -- so a
    sweep that crossed one subtracted to a negative delta, which was clamped to zero. Worse,
    the packages were summed before subtracting, so one package's wrap could hide inside
    another's rise and publish a plausible, wrong number.
    """
    max_range = 100_000_000
    # The first package wrapped 2 000 000 uj past its range; the second rose normally.
    start = {"intel-rapl:0": (99_000_000, max_range), "intel-rapl:1": (1_000_000, max_range)}
    end = {"intel-rapl:0": (1_000_000, max_range), "intel-rapl:1": (4_000_000, max_range)}

    total, reason = eval_telemetry_module._rapl_delta_uj(start, end)

    assert total == 2_000_000 + 3_000_000
    assert reason is None


def test_a_wrap_with_no_published_range_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    start = {"intel-rapl:0": (99_000_000, None)}
    end = {"intel-rapl:0": (1_000_000, None)}

    total, reason = eval_telemetry_module._rapl_delta_uj(start, end)

    assert total is None
    assert reason is not None and "max_energy_range_uj" in reason


def test_a_sampler_whose_counter_wrapped_reports_the_corrected_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mindbridge.benchmarks.eval_telemetry._nvidia_utilization", lambda: ())
    readings = iter(
        (
            ({"intel-rapl:0": (99_000_000, 100_000_000)}, None),
            ({"intel-rapl:0": (1_000_000, 100_000_000)}, None),
        )
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_telemetry._rapl_energy_uj",
        lambda root=None: next(readings),
    )

    with ResourceSampler(interval_seconds=0.05) as sampler:
        pass

    energy = cast(Mapping[str, object], sampler.json(wall_seconds=1.0)["energy"])
    # 2 000 000 uj across the wrap, not the 0.0 the clamp used to publish.
    assert energy["cpu_package_joules"] == pytest.approx(2.0)


def test_rapl_energy_reports_why_it_could_not_read_anything(tmp_path: Path) -> None:
    total, reason = eval_telemetry_module._rapl_energy_uj(tmp_path / "missing")

    assert total is None
    assert reason is not None and "no intel-rapl packages" in reason


def test_rapl_energy_reports_the_unreadable_file(tmp_path: Path) -> None:
    root = tmp_path / "powercap"
    package = root / "intel-rapl:0"
    package.mkdir(parents=True)
    (package / "energy_uj").write_text("not-a-number")

    total, reason = eval_telemetry_module._rapl_energy_uj(root)

    assert total is None
    assert reason is not None and "energy_uj" in reason


# --- family 6: fast-plane and compiler telemetry (capture, settle, compile) -----------------


def _asset(asset_id: str) -> AssetRef:
    """A resolved image asset, which is what a record's assets must be."""
    return AssetRef(
        id=asset_id,
        modality=Modality.IMAGE,
        media_type="image/png",
        size_bytes=1,
        sha256="a" * 64,
        path=Path(f"{asset_id}.png"),
    )


def test_compile_telemetry_counts_every_grounded_media_part() -> None:
    """One omni memory with two assets is two media parts, the quantity the budget bounds.

    The count was of hits carrying any asset, so a bundle grounding a still and a clip from one
    observation reported the same single part as a bundle grounding only the still, and
    multi-asset artifacts undercounted the compiler's real media spend.
    """
    reference = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    hit = SearchHit(
        id="omni",
        content="a still and a clip from one observation",
        score=0.9,
        created_at=reference,
        memory_type=MemoryType.EPISODIC,
        modality=Modality.IMAGE,
        assets=(
            _asset("asset-still"),
            _asset("asset-second-still"),
        ),
    )
    bundle = compile_context(
        "what happened",
        (hit,),
        budget=ContextBudget(),
        reference_at=reference,
    )
    assert len(bundle.hits) == 1

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with eval_module._compile_span(provider.get_tracer("test"), bundle):
        pass

    attributes = exporter.get_finished_spans()[-1].attributes
    assert attributes is not None
    assert attributes[BENCHMARK_COMPILE_ITEMS] == 1
    assert attributes[BENCHMARK_COMPILE_MEDIA_ITEMS] == 2


def test_capture_settle_and_compile_spans_are_aggregated_separately() -> None:
    result = _aggregate(
        (
            _span("mindbridge.capture", end_ms=10, kind="operation"),
            _span("mindbridge.capture", end_ms=20, kind="operation"),
            _span(
                "mindbridge.settle",
                end_ms=40,
                kind="operation",
                attributes={CAPTURE_TIME_TO_SEARCHABLE: 55.0},
            ),
            _span("mindbridge.compile", end_ms=30, kind="operation"),
            _span(
                BENCHMARK_COMPILE_SPAN,
                end_ms=1,
                kind="stage",
                attributes={
                    BENCHMARK_COMPILE_CHARS: 900,
                    BENCHMARK_COMPILE_ITEMS: 3,
                    BENCHMARK_COMPILE_MEDIA_ITEMS: 1,
                },
            ),
        )
    )

    capture = cast(Mapping[str, object], result["capture"])
    assert capture["span"] == "mindbridge.capture"
    assert capture["count"] == 2
    assert cast(float, cast(Mapping[str, object], capture["latency_ms"])["p50"]) == pytest.approx(
        15.0
    )

    time_to_searchable = cast(Mapping[str, object], result["time_to_searchable_ms"])
    assert time_to_searchable["count"] == 1
    assert cast(float, time_to_searchable["p50"]) == pytest.approx(55.0)

    formation = cast(Mapping[str, object], result["formation"])
    assert formation["span"] == "mindbridge.settle"
    assert formation["count"] == 1
    assert cast(float, cast(Mapping[str, object], formation["latency_ms"])["p50"]) == pytest.approx(
        40.0
    )

    compile_block = cast(Mapping[str, object], result["compile"])
    assert compile_block["span"] == "mindbridge.compile"
    assert compile_block["count"] == 1
    bundle_chars = cast(Mapping[str, object], compile_block["bundle_chars"])
    assert bundle_chars["count"] == 1
    assert bundle_chars["p50"] == pytest.approx(900)
    bundle_items = cast(Mapping[str, object], compile_block["bundle_items"])
    assert bundle_items["p50"] == pytest.approx(3)
    bundle_media_items = cast(Mapping[str, object], compile_block["bundle_media_items"])
    assert bundle_media_items["p50"] == pytest.approx(1)


def test_capture_settle_and_compile_report_nothing_when_unexercised() -> None:
    result = _aggregate((_span(SEARCH_SPAN, end_ms=5),))

    assert cast(Mapping[str, object], result["capture"])["count"] == 0
    assert cast(Mapping[str, object], result["time_to_searchable_ms"])["count"] == 0
    assert cast(Mapping[str, object], result["formation"])["count"] == 0
    assert cast(Mapping[str, object], result["compile"])["count"] == 0


# --- controls: random ranker, blind answer, and R@20 next to R@1 -----------------------------


def test_random_ranker_expectation_is_reported_next_to_measured_recall() -> None:
    samples = (
        _sample("q1", sources=("s1", "s2"), gold=("s1",), candidate_count=10),
        _sample("q2", sources=("s3", "s1"), gold=("s1",), candidate_count=10),
    )

    retrieval = eval_module._retrieval_quality(
        samples, seed=7, bootstrap_samples=32, recall_limit=20
    )
    measured = cast(Mapping[str, Mapping[str, object]], retrieval["recall_at_k"])
    random_ranker = cast(Mapping[str, Mapping[str, object]], retrieval["random_ranker_recall_at_k"])

    assert retrieval["gold_evidence_key"] == "evidence_ids"
    assert retrieval["labelled_question_count"] == 2
    # q1 hits at rank 1, q2 only at rank 2.
    assert cast(float, measured["1"]["mean"]) == pytest.approx(0.5)
    assert cast(float, measured["5"]["mean"]) == pytest.approx(1.0)
    # A uniformly random ranker over ten candidates already reaches 1.0 at k = 10.
    assert cast(float, random_ranker["1"]["mean"]) == pytest.approx(0.1)
    assert cast(float, random_ranker["10"]["mean"]) == pytest.approx(1.0)
    assert cast(float, random_ranker["20"]["mean"]) == pytest.approx(1.0)
    assert cast(Mapping[str, object], retrieval["candidate_pool_size"])["mean"] == pytest.approx(10)


def test_retrieval_quality_says_so_when_the_adapter_carries_no_gold_evidence() -> None:
    samples = (_sample("q1", sources=("s1",), gold=(), candidate_count=4),)

    retrieval = eval_module._retrieval_quality(
        samples, seed=7, bootstrap_samples=32, recall_limit=20
    )

    assert retrieval["gold_evidence_key"] is None
    assert retrieval["recall_at_k"] == {}
    assert "no gold evidence source IDs" in str(retrieval["unavailable_reason"])


def test_gold_evidence_that_named_no_stored_memory_is_counted_not_absorbed() -> None:
    """A label vocabulary mismatch must not read as recall over what happened to join.

    An adapter that joins a separate label list onto its own source IDs reports the
    IDs that matched nothing. Here every one failed, so the task falls back to "no
    gold evidence" and stays uninterpretable, with the count saying why.
    """
    samples = (
        _sample("q1", sources=("s1",), gold=(), candidate_count=4, unresolved=("D9:9",)),
        _sample("q2", sources=("s1",), gold=(), candidate_count=4, unresolved=("D9:8", "D9:7")),
    )

    retrieval = eval_module._retrieval_quality(
        samples, seed=7, bootstrap_samples=32, recall_limit=20
    )
    controls = eval_module._controls("fixture", retrieval, None, is_blind_run=False)

    assert retrieval["gold_evidence_key"] is None
    assert retrieval["unresolved_gold_evidence_ids"] == 3
    assert controls["missing"] == ["random_ranker", "blind", "recall_at_20"]


def test_a_partly_joined_label_list_reports_both_the_measured_and_the_missed_ids() -> None:
    samples = (
        _sample("q1", sources=("s1",), gold=("s1",), candidate_count=4, unresolved=("D9:9",)),
        _sample("q2", sources=("s2",), gold=("s2",), candidate_count=4),
    )

    retrieval = eval_module._retrieval_quality(
        samples, seed=7, bootstrap_samples=32, recall_limit=20
    )
    measured = cast(Mapping[str, Mapping[str, object]], retrieval["recall_at_k"])

    assert retrieval["gold_evidence_key"] == "evidence_ids"
    assert retrieval["labelled_question_count"] == 2
    assert cast(float, measured["1"]["mean"]) == pytest.approx(1.0)
    # Recall is measured over the labels that joined; the rest is reported alongside
    # it rather than folded into the denominator or dropped in silence.
    assert retrieval["unresolved_gold_evidence_ids"] == 1


@pytest.mark.parametrize("cutoff", ("1", "20"))
def test_recall_at_1_and_recall_at_20_are_only_accepted_together(cutoff: str) -> None:
    retrieval = {
        "recall_at_k": {cutoff: {"mean": 0.4}},
        "random_ranker_recall_at_k": {cutoff: {"mean": 0.1}},
    }

    controls = eval_module._controls("fixture", retrieval, None, is_blind_run=False)

    assert controls["missing"] == ["blind", "recall_at_20"]
    assert controls["interpretable"] is False
    assert "recall_at_20" in str(controls["reason"])


def test_controls_are_complete_only_with_all_three_baselines() -> None:
    retrieval = {
        "recall_at_k": {"1": {"mean": 0.4}, "20": {"mean": 0.9}},
        "random_ranker_recall_at_k": {"1": {"mean": 0.1}, "20": {"mean": 1.0}},
    }

    without_blind = eval_module._controls("fixture", retrieval, None, is_blind_run=False)
    # A non-retrieving arm has nothing for the retrieval controls to check: they do not apply,
    # so its row stays interpretable when the blind control is present.
    non_retrieving = eval_module._controls(
        "fixture",
        {"recall_at_k": {}, "random_ranker_recall_at_k": {}},
        {"mean": 0.1},
        is_blind_run=False,
        retrieves=False,
    )
    assert non_retrieving["retrieval_controls_applicable"] is False
    assert non_retrieving["missing"] == []
    assert non_retrieving["interpretable"] is True
    with_blind = eval_module._controls("fixture", retrieval, {"mean": 0.383}, is_blind_run=False)
    blind_run = eval_module._controls("fixture", retrieval, None, is_blind_run=True)

    assert without_blind["missing"] == ["blind"]
    assert with_blind["missing"] == []
    assert with_blind["interpretable"] is True
    assert cast(Mapping[str, object], with_blind["blind"])["mean"] == pytest.approx(0.383)
    assert blind_run["interpretable"] is True
    assert blind_run["is_blind_run"] is True


def test_metrics_attach_the_controls_and_never_hide_a_missing_one() -> None:
    task = cast(Any, SimpleNamespace(spec=SimpleNamespace(name="fixture")))
    samples = (_sample("q1", sources=("s1",), gold=(), candidate_count=4),)

    metrics = eval_module._metrics(task, samples, _arguments())
    controls = cast(Mapping[str, object], metrics["controls"])

    assert controls["interpretable"] is False
    assert set(cast(Sequence[str], controls["missing"])) == {
        "random_ranker",
        "blind",
        "recall_at_20",
    }
    assert metrics["cross_harness_comparable"] is False
    assert "LoCoMo" in str(metrics["comparability_note"])


def test_table_prints_missing_controls_instead_of_only_a_score() -> None:
    results = {
        "tasks": [
            {
                "task": "fixture",
                "primary_metric": "accuracy",
                "score": {"mean": 0.9, "confidence_interval_95": None},
                "question_count": 2,
                "error_count": 0,
                "performance": {
                    "duration_seconds": {"total": 2.5, "average": 1.25},
                    "token_usage": {"total_tokens": 30, "average_tokens": 15.0},
                },
                # `_table` reports the writes that invalidated a score; a row
                # without this key cannot render.
                "ingest_failure_count": 0,
                "controls": {
                    "random_ranker": None,
                    "recall_at_1": None,
                    "recall_at_20": None,
                    "blind": None,
                    "is_blind_run": False,
                    "missing": ["random_ranker", "blind", "recall_at_20"],
                    "interpretable": False,
                    "reason": "fixture reports a score without random_ranker, blind, recall_at_20",
                },
            }
        ]
    }

    table = eval_module._table(results)

    header, row = table.splitlines()
    assert "controls" in header
    assert "R@20" in header
    assert "rand@20" in header
    # Four absent control cells render as MISSING, and the summary column names all three.
    assert row.split().count("MISSING") == 5
    assert "MISSING random_ranker,blind,recall_at_20" in row
    assert eval_module._uninterpretable_tasks(results) == (
        "fixture reports a score without random_ranker, blind, recall_at_20",
    )


def test_table_shows_the_control_values_when_they_are_present() -> None:
    results = {
        "tasks": [
            {
                "task": "fixture",
                "primary_metric": "accuracy",
                "score": {"mean": 0.9, "confidence_interval_95": None},
                "question_count": 2,
                "error_count": 0,
                "performance": {
                    "duration_seconds": {"total": 2.5, "average": 1.25},
                    "token_usage": {"total_tokens": 30, "average_tokens": 15.0},
                },
                # `_table` reports the writes that invalidated a score; a row
                # without this key cannot render.
                "ingest_failure_count": 0,
                "controls": {
                    "random_ranker": {"20": {"mean": 0.9941}},
                    "recall_at_1": {"mean": 0.5},
                    "recall_at_20": {"mean": 0.8},
                    "blind": {"mean": 0.383},
                    "is_blind_run": False,
                    "missing": [],
                    "interpretable": True,
                    "reason": None,
                },
            }
        ]
    }

    table = eval_module._table(results)

    assert "0.9941" in table
    assert "0.3830" in table
    assert "ok" in table
    assert eval_module._uninterpretable_tasks(results) == ()


def test_table_prints_the_product_token_cost_when_the_judge_left_the_total_null() -> None:
    results = {
        "tasks": [
            {
                "task": "fixture",
                "primary_metric": "accuracy",
                "score": {"mean": 0.9, "confidence_interval_95": None},
                "question_count": 2,
                "error_count": 0,
                "performance": {
                    "duration_seconds": {"total": 2.5, "average": 1.25},
                    "token_usage": {
                        "total_tokens": None,
                        "average_tokens": None,
                        "product": {"total_tokens": 80, "average_tokens": 40.0},
                    },
                },
                "ingest_failure_count": 0,
                "controls": {
                    "random_ranker": None,
                    "recall_at_1": None,
                    "recall_at_20": None,
                    "blind": None,
                    "is_blind_run": False,
                    "missing": [],
                    "interpretable": True,
                    "reason": None,
                },
            }
        ]
    }

    table = eval_module._table(results)

    # The marker says these are the product modules' tokens, not the run total.
    assert "80*" in table
    assert "40.0*" in table


# --- blind baseline provenance ---------------------------------------------------------------


def _blind_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": eval_module.EVAL_SCHEMA_VERSION,
        "run_id": "blind-run",
        "blind": True,
        "tasks": [
            {
                "task": "fixture",
                "primary_metric": "accuracy",
                "evaluation_sha256": "e" * 64,
                "question_count": 2,
                "score": {"mean": 0.383},
            }
        ],
    }
    document.update(overrides)
    return document


def _loaded_task(evaluation_sha256: str = "e" * 64) -> LoadedTask:
    return cast(
        LoadedTask,
        SimpleNamespace(
            spec=SimpleNamespace(name="fixture"),
            evaluation_sha256=evaluation_sha256,
        ),
    )


def test_blind_baseline_is_read_from_a_blind_run_of_the_same_inputs(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_blind_document()), encoding="utf-8")

    rows = eval_module._blind_baseline_rows(tmp_path, (_loaded_task(),))

    assert rows["fixture"]["mean"] == pytest.approx(0.383)
    assert rows["fixture"]["run_id"] == "blind-run"


def test_blind_baseline_rejects_a_run_that_had_memory(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_blind_document(blind=False)), encoding="utf-8")

    with pytest.raises(ValueError, match="--blind"):
        eval_module._blind_baseline_rows(path, (_loaded_task(),))


def test_blind_baseline_rejects_different_evaluation_inputs(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_blind_document()), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation inputs differ"):
        eval_module._blind_baseline_rows(path, (_loaded_task("f" * 64),))


# --- noise floor -----------------------------------------------------------------------------


def test_noise_floor_reports_the_smallest_resolvable_difference() -> None:
    task = cast(Any, SimpleNamespace(spec=SimpleNamespace(name="fixture")))
    samples = tuple(
        _sample(f"q{index}", sources=(), gold=(), candidate_count=0, score=float(index % 2))
        for index in range(8)
    )

    noise = cast(
        Mapping[str, object], eval_module._metrics(task, samples, _arguments())["noise_floor"]
    )

    assert noise["floor"] == pytest.approx(NOISE_FLOOR)
    assert cast(float, noise["per_question_standard_deviation"]) == pytest.approx(
        0.5345224838248488
    )
    assert cast(float, noise["minimum_meaningful_difference"]) >= NOISE_FLOOR
    assert "noise band" in str(noise["note"])


def test_percentile_interpolates_and_reports_nothing_for_no_observations() -> None:
    assert percentile((), 0.5) is None
    assert percentile((40.0, 10.0, 30.0, 20.0), 0.99) == pytest.approx(39.7)
    assert percentile((5.0,), 0.99) == pytest.approx(5.0)


# --- reproducibility fields ------------------------------------------------------------------


def test_metric_breakdown_families_all_exist_in_the_single_family_table() -> None:
    """No second, drifting copy of the task-family table may reappear in ``eval.py``."""
    from mindbridge.benchmarks.official_scorers import task_family
    from mindbridge.benchmarks.task_catalog import TASKS

    known = {task_family(name) for name in TASKS} - {None}
    task = cast(Any, SimpleNamespace(spec=SimpleNamespace(name="locomo-refined")))
    declared = eval_module._metric_breakdowns(task, (), _arguments())

    assert declared == {}
    assert "locomo-refined" in known
    for family in (
        "locomo-refined",
        "m3-bench",
        "video-mme",
        "video-mme-v2",
        "egolifeqa",
        "egotempo",
        "memlens",
        "mm-lifelong",
        "supermemory-vqa",
        "atm-bench",
        "mem-gallery",
        "openeqa",
    ):
        assert family in known, family


def test_hardware_and_runtime_revisions_are_recorded() -> None:
    hardware = eval_module._hardware()

    assert set(hardware) == {"machine", "processor", "logical_cores", "cuda_device_uuids"}
    assert cast(int, hardware["logical_cores"]) >= 1


def test_table_refuses_a_task_row_that_carries_no_controls() -> None:
    results = {
        "tasks": [
            {
                "task": "fixture",
                "primary_metric": "accuracy",
                "score": {"mean": 0.9, "confidence_interval_95": None},
                "question_count": 2,
                "error_count": 0,
                "performance": {
                    "duration_seconds": {"total": 1.0, "average": 0.5},
                    "token_usage": {"total_tokens": 1, "average_tokens": 0.5},
                },
            }
        ]
    }

    with pytest.raises(KeyError, match="controls"):
        eval_module._table(results)


def test_retrieved_sources_are_deduplicated_in_rank_order() -> None:
    sample = _sample("q", sources=("s1", "s1", "s2"), gold=("s2",), candidate_count=3)

    assert eval_module._retrieved_sources(sample) == ("s1", "s2")


def test_retrieval_recall_scores_the_ranked_list_not_the_cited_evidence() -> None:
    """Citations are the generator's choice; recall is the retriever's.

    A question whose gold was cited but never ranked scores zero, and one whose run recorded no
    ranked list is counted as unranked rather than scored.
    """
    from dataclasses import replace

    cited_not_ranked = replace(
        _sample("q1", sources=("s9",), gold=("s1",), candidate_count=10),
        evidence=(EvidenceInterval("m-s1", "s1", None, None),),
    )
    unranked = replace(
        _sample("q2", sources=("s1",), gold=("s1",), candidate_count=10),
        ranked_source_ids=(),
    )

    retrieval = eval_module._retrieval_quality(
        (cited_not_ranked, unranked), seed=7, bootstrap_samples=32, recall_limit=20
    )
    measured = cast(Mapping[str, Mapping[str, object]], retrieval["recall_at_k"])

    assert retrieval["labelled_question_count"] == 1
    assert retrieval["unranked_labelled_question_count"] == 1
    assert cast(float, measured["20"]["mean"]) == pytest.approx(0.0)


def test_candidate_pool_stops_at_the_question_cutoff() -> None:
    unit = EvalUnit(
        unit_id="unit",
        memories=(
            MemoryItem("a", ("first",), end_seconds=10.0),
            MemoryItem("b", ("second",), end_seconds=20.0),
            MemoryItem("c", ("third",), end_seconds=30.0),
        ),
        questions=(
            EvalQuestion("q", ("question",), references=("reference",), cutoff_seconds=20.0),
        ),
    )

    assert eval_module._candidate_count(unit, unit.questions[0]) == 2
    assert eval_module._candidate_count(unit, replace(unit.questions[0], cutoff_seconds=None)) == 3


def test_noise_floor_applies_the_measured_floor_to_a_tiny_standard_error() -> None:
    scored = (ScoredValue("a", "one", 1.0), ScoredValue("b", "two", 1.0))

    noise = eval_module._noise_floor(scored, {"cluster_standard_error": 0.001})

    assert cast(float, noise["minimum_meaningful_difference"]) == pytest.approx(NOISE_FLOOR)


def test_noise_floor_widens_past_the_measured_floor_for_a_noisy_run() -> None:
    scored = (ScoredValue("a", "one", 1.0), ScoredValue("b", "two", 0.0))

    noise = eval_module._noise_floor(scored, {"cluster_standard_error": 0.05})

    assert cast(float, noise["minimum_meaningful_difference"]) == pytest.approx(
        1.959963984540054 * 0.05 * math.sqrt(2)
    )


def test_search_latency_never_mixes_the_narrower_lookup_or_the_search_operation() -> None:
    """`mindbridge.retrieve` is the reported boundary; two other spans must not join it."""
    result = _aggregate(
        (
            _span(SEARCH_SPAN, start_ms=0, end_ms=100),
            # The lookup nested inside the retrieval leg, and the separate public operation.
            _span("mindbridge.index.search", start_ms=10, end_ms=20),
            _span("mindbridge.search", start_ms=200, end_ms=900, kind="operation"),
        )
    )

    search = cast(Mapping[str, object], result["search"])
    assert search["count"] == 1
    assert cast(float, cast(Mapping[str, object], search["latency_ms"])["p50"]) == pytest.approx(
        100.0
    )
    nodes = cast(Mapping[str, Mapping[str, object]], result["nodes"])
    assert set(nodes) == {SEARCH_SPAN, "mindbridge.index.search", "mindbridge.search"}


def test_a_search_only_run_reports_nothing_under_search() -> None:
    result = _aggregate((_span("mindbridge.search", end_ms=50, kind="operation"),))

    search = cast(Mapping[str, object], result["search"])
    assert search["count"] == 0
    assert search["latency_ms"] is None
    assert search["throughput_per_second"] is None
