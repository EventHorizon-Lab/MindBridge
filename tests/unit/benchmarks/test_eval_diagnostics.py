"""Boundary observations are not gold facts or causal failure labels."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from mindbridge import AnswerChunk, AnswerResult, AsyncMemory, SearchHit
from mindbridge._telemetry import _record_generation_inputs, _record_retrieval_results
from mindbridge.benchmarks.eval import (
    _answer_many,
    _answer_span,
    _cache_outcome,
    _cached_results,
    _with_grounding_loss,
)
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_arms import _Arm, _BaselineGenerator
from mindbridge.benchmarks.eval_cache import CachedAnswer, EvidenceInterval, ResponseCache
from mindbridge.benchmarks.eval_results import SampleResult, _AnswerOutcome
from mindbridge.benchmarks.eval_telemetry import EvaluationTelemetry
from mindbridge.benchmarks.task_catalog import TaskSpec


def _sample(**changes: object) -> SampleResult:
    sample = SampleResult(
        task="fixture",
        benchmark="Fixture",
        dataset_sha256="d" * 64,
        evaluation_sha256="e" * 64,
        unit_id="unit",
        question_id="q",
        prediction="answer",
        parsed_choice=None,
        score=0.0,
        exact_match=0.0,
        latency_ms=1.0,
        confidence=0.0,
        memory_ids=("m1",),
        ingest_failure_count=0,
        error_code=None,
        metadata={"evidence_groups": [["s1", "alternative"], ["s2"]]},
        ranked_source_ids=("s1", "s2"),
        ranked_source_ids_complete=True,
        evidence=(EvidenceInterval("m1", "s1", None, None),),
    )
    return replace(sample, **cast(dict[str, Any], changes))


def _diagnose(sample: SampleResult) -> dict[str, Any]:
    module = importlib.import_module("mindbridge.benchmarks.eval_diagnostics")
    result: dict[str, Any] = module.quality_diagnostics(sample)
    return result


def test_layers_use_observed_boundaries_instead_of_returned_hits() -> None:
    sample = _sample(
        input_source_ids=("s1", "s2"),
        generation_evidence=(
            EvidenceInterval("m1", "s1", None, None),
            EvidenceInterval("m2", "s2", None, None),
        ),
        grounding_request={
            "record_ids": ("m2",),
            "asset_ids": (),
            "successful": True,
            "media_fallback": True,
        },
        trace_id="a" * 32,
        answer_span_id="b" * 16,
    )
    diagnostic = _diagnose(sample)
    layers = diagnostic["layers"]
    assert tuple(layers) == ("input", "retrieval", "organization", "reader_input", "answer")
    assert diagnostic["trace_id"] == "a" * 32
    assert diagnostic["answer_span_id"] == "b" * 16
    assert layers["organization"]["source_coverage"]["matched_groups"] == [0, 1]
    assert layers["reader_input"]["source_coverage"]["matched_groups"] == [1]
    assert layers["reader_input"]["media_fallback"] is True
    assert diagnostic["fact_span_coverage"] is None
    assert diagnostic["causal_attribution"] is None
    assert sample.json()["quality_diagnostics"] == diagnostic


def test_missing_observations_and_labels_never_become_zero_recall() -> None:
    layers = _diagnose(_sample(metadata={}, ranked_source_ids_complete=False))["layers"]
    for name in ("input", "retrieval", "organization", "reader_input"):
        assert layers[name]["source_coverage"] is None
    assert layers["organization"]["observed"] is False

    empty = _diagnose(_sample(ranked_source_ids=(), generation_evidence=()))["layers"]
    assert empty["retrieval"]["source_coverage"]["recall"] == 0.0
    assert empty["organization"]["source_coverage"]["recall"] == 0.0


def test_unmapped_reader_ids_and_partial_excerpts_do_not_prove_coverage() -> None:
    diagnostic = _diagnose(
        _sample(
            generation_evidence=(EvidenceInterval("m1", "s1", None, None),),
            excerpt_evidence=(EvidenceInterval("excerpt", "s2", 1.0, 2.0),),
            grounding_request={"record_ids": ("unmapped",), "successful": True},
        )
    )
    assert diagnostic["layers"]["organization"]["source_coverage"]["recall"] == 0.5
    assert diagnostic["layers"]["organization"]["partial_source_ids"] == ["s2"]
    assert diagnostic["layers"]["reader_input"]["source_coverage"] is None
    assert diagnostic["layers"]["reader_input"]["unmapped_record_ids"] == ["unmapped"]


def test_returned_metadata_cannot_replace_observed_reader_source_provenance() -> None:
    diagnostic = _diagnose(
        _sample(
            generation_evidence=(EvidenceInterval("m1", "noise", None, None),),
            evidence=(EvidenceInterval("m1", "s1", None, None),),
            grounding_request={"record_ids": ("m1",), "successful": True},
        )
    )
    assert diagnostic["layers"]["reader_input"]["source_coverage"]["recall"] == 0.0
    unknown = _diagnose(_sample(ranked_source_ids=("s1", "")))["layers"]["retrieval"]
    assert unknown["observed"] is True
    assert unknown["source_coverage"] is None


def test_conditional_scores_count_unknown_observations_separately() -> None:
    module = importlib.import_module("mindbridge.benchmarks.eval_diagnostics")
    full = _sample(
        generation_evidence=(
            EvidenceInterval("m1", "s1", None, None),
            EvidenceInterval("m2", "s2", None, None),
        )
    )
    summary = module.summarize_diagnostics((full, _sample(), _sample(metadata={})))
    layer = summary["layers"]["organization"]
    assert layer["observed_count"] == 1
    assert layer["coverage_measured_count"] == 1
    assert layer["all_labelled_sources_present_count"] == 1
    assert layer["scored_with_all_labelled_sources_count"] == 1
    assert layer["mean_score_with_all_labelled_sources"] == 0.0
    assert summary["layers"]["answer"]["scored_count"] == 3


def test_failed_answer_keeps_its_trace_and_cached_sample_cannot_acquire_a_fresh_one() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with _answer_span(telemetry.tracer, "fixture", "fixture/unit/q") as (span, _):
            assert span is not None
            context = span.get_span_context()
            assert context is not None
        failed = _sample(error_code="provider_failed", prediction="")
        cached = _sample(cached=True, trace_id="a" * 32, answer_span_id="b" * 16)
        restored_failure, restored_cache = _with_grounding_loss((failed, cached), telemetry)
        assert restored_failure.trace_id == f"{context.trace_id:032x}"
        assert restored_failure.answer_span_id == f"{context.span_id:016x}"
        assert restored_cache == cached
    finally:
        telemetry.close()


@pytest.mark.asyncio
async def test_answer_pipeline_correlates_concurrent_samples_without_a_second_search() -> None:
    class Memory:
        async def ask_stream(
            self, question: object, **_kwargs: object
        ) -> AsyncIterator[AnswerChunk]:
            source_id = str(question)
            hit = SearchHit(
                id=source_id,
                content="fact",
                score=1.0,
                created_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
                metadata={"source_id": source_id},
            )
            _record_retrieval_results((hit,))
            _record_generation_inputs((hit,))
            yield AnswerChunk(text="answer")
            # Returning no hits must not erase the observed answerer input.
            yield AnswerChunk(result=AnswerResult("answer"))

        async def search(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("diagnosis must not issue a scoring search")

    telemetry = EvaluationTelemetry()
    try:
        outcomes = await _answer_many(
            cast(AsyncMemory, Memory()),
            tuple(
                EvalQuestion(f"q{index}", (f"source{index}",), references=("answer",))
                for index in range(3)
            ),
            request_concurrency=3,
            recall_limit=3,
            task_name="fixture",
            unit_id="unit",
            tracer=telemetry.tracer,
        )
        spans = set()
        for index, outcome in enumerate(outcomes):
            assert not isinstance(outcome, BaseException)
            assert outcome.evidence == ()
            assert outcome.generation_evidence is not None
            assert outcome.generation_evidence[0].source_id == f"source{index}"
            assert outcome.trace_id is not None and len(outcome.trace_id) == 32
            assert outcome.answer_span_id is not None
            spans.add(outcome.answer_span_id)
        assert len(spans) == 3
    finally:
        telemetry.close()


def test_cache_preserves_original_trace_and_organization_without_inventing_requests(
    tmp_path: Path,
) -> None:
    evidence = (EvidenceInterval("m1", "s1", None, None),)
    answer = CachedAnswer(
        "answer",
        0.0,
        ("m1",),
        evidence,
        generation_evidence=evidence,
        trace_id="a" * 32,
        answer_span_id="b" * 16,
    )
    cache = ResponseCache(tmp_path / "cache.db", "run", "fixture")
    try:
        cache.put("task", "unit", "q", answer)
        restored = cache.get("task", "unit", "q")
        assert restored == answer
    finally:
        cache.close()


def test_compile_partial_organization_survives_response_cache(tmp_path: Path) -> None:
    question = EvalQuestion("q", ("question",), references=("answer",))
    unit = EvalUnit("unit", (MemoryItem("s2", ("fact",)),), (question,))
    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "d" * 64,
        (unit,),
    )
    partial = (EvidenceInterval("m2", "s2", None, None),)
    outcome = _AnswerOutcome(
        "answer",
        1.0,
        0.0,
        (),
        (),
        generation_evidence=(),
        excerpt_source_ids=("m2",),
        excerpt_evidence=partial,
    )
    arm = _Arm("compile")
    cache = ResponseCache(tmp_path / "responses.db", "run", "fixture")
    try:
        _cache_outcome(cache, task, unit, question, outcome, 0, arm=arm)
        sample = _cached_results(cache, task, unit, arm=arm, predict_only=False, log_samples=False)[
            ("compile", "q")
        ]
        assert sample.excerpt_evidence == partial
        assert sample.excerpt_source_ids == ("m2",)
        assert _diagnose(sample)["layers"]["organization"]["partial_source_ids"] == ["s2"]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_failed_compile_generation_retains_the_organization_boundary() -> None:
    from mindbridge import ContextBudget, ContextBundle
    from mindbridge.context import compile_context

    class BrokenGenerator:
        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise ValueError("generation failed")

    hit = SearchHit(
        id="m",
        content="fact",
        score=1.0,
        created_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
        metadata={"source_id": "s1"},
    )

    class CompiledMemory:
        async def compile(self, *_args: object, **_kwargs: object) -> ContextBundle:
            return compile_context(
                "question",
                (hit,),
                budget=ContextBudget(),
                reference_at=hit.created_at,
            )

    outcome = (
        await _answer_many(
            cast(AsyncMemory, CompiledMemory()),
            (EvalQuestion("q", ("question",), references=("answer",)),),
            request_concurrency=1,
            recall_limit=3,
            arm=_Arm("compile", generator=cast(_BaselineGenerator, BrokenGenerator())),
            task_name="fixture",
            unit_id="unit",
        )
    )[0]
    assert not isinstance(outcome, BaseException)
    assert isinstance(outcome.error, ValueError)
    assert outcome.generation_evidence == (EvidenceInterval("m", "s1", None, None),)


def test_review_export_is_stratified_reproducible_and_has_no_synthetic_gold(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("mindbridge.benchmarks.eval_review")
    rows = [
        _sample(question_id=str(index), metadata={}, score=float(index % 2)).json()
        for index in range(20)
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "review.jsonl"
    args = ["--samples", str(samples_path), "--output", str(output), "--per-stratum", "2"]
    assert module.main(args) == 0
    first = output.read_bytes()
    assert module.main(args) == 0
    assert output.read_bytes() == first
    selected = [json.loads(line) for line in first.splitlines()]
    assert len(selected) == 4
    assert {row["score"] for row in selected} == {0.0, 1.0}
    for row in selected:
        annotation = row["annotation"]
        assert annotation["status"] == "pending"
        assert annotation["source_spans"] is None
        assert all(value is None for value in annotation["layers"].values())
        assert row["quality_diagnostics"]["fact_span_coverage"] is None

    output.write_text("human annotations", encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserve annotations"):
        module.main(args)


def test_review_publication_cannot_overwrite_a_concurrently_created_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("mindbridge.benchmarks.eval_review")
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps(_sample().json()) + "\n", encoding="utf-8")
    output = tmp_path / "review.jsonl"
    link = os.link

    def concurrent_annotation(source: str, target: Path) -> None:
        target.write_text("human annotations", encoding="utf-8")
        link(source, target)

    monkeypatch.setattr(os, "link", concurrent_annotation)
    with pytest.raises(FileExistsError):
        module.main(["--samples", str(source), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "human annotations"


def test_review_requires_current_sample_format(tmp_path: Path) -> None:
    module = importlib.import_module("mindbridge.benchmarks.eval_review")
    source = tmp_path / "samples.jsonl"
    row = _sample().json()
    del row["schema_version"]
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema version is unsupported"):
        module.main(["--samples", str(source), "--output", str(tmp_path / "review.jsonl")])
