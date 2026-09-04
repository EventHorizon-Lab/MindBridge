"""Core checks for deterministic task selection, causal ingest, and statistics."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from argparse import ArgumentTypeError
from collections.abc import Sequence
from dataclasses import fields, replace
from datetime import datetime, timezone
from inspect import getattr_static, signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import ValidationError

import mindbridge.benchmarks.eval as eval_module
from mindbridge import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    FaceAnalysis,
    FaceBackend,
    IndexUnavailableError,
    Memory,
    MemoryConfig,
    MemoryPlugins,
    MemoryType,
    MindBridgeConfig,
    Modality,
    ModelError,
    SearchHit,
)
from mindbridge.benchmarks.eval import (
    MemoryFactory,
    SampleResult,
    _answer_many,
    _BorrowedFaceBackend,
    _BorrowedSpeechBackend,
    _cache_namespace,
    _generation_kwargs,
    _ingest,
    _load_memory_config,
    _model_config,
    _ref_at_n,
    _run_identifier,
    _seed_values,
    _video_mme_v2_rating,
    main,
    run_loaded_task,
)
from mindbridge.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
    MediaResolver,
    MemoryItem,
    _choice_parts,
    _free_text_parts,
    _gallery_memory,
    _query_parts,
    load_task,
)
from mindbridge.benchmarks.eval_cache import ResponseCache
from mindbridge.benchmarks.eval_statistics import (
    ScoredValue,
    paired_comparison,
    parse_choice,
    summarize,
)
from mindbridge.benchmarks.eval_telemetry import EvaluationTelemetry
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.mem_gallery import MemGalleryRound, MemGallerySession
from mindbridge.benchmarks.model_config import (
    DEFAULT_HF_ENDPOINT,
    DownloadOverrides,
    DownloadSettings,
    HarnessOverrides,
    ModelConfig,
    default_benchmarks_root,
)
from mindbridge.benchmarks.official_scorers import scorer_protocol, task_family
from mindbridge.benchmarks.task_catalog import TASKS, TaskSpec, expand
from mindbridge.benchmarks.video_mme_v2 import score_group_answers
from mindbridge.configuration import OpenAIEmbeddingConfig
from mindbridge.models.base import EmbedTask, ModelInput, SpeechAnalysis, SpeechBackend


def _egomem_sample(example_id: int, answer: str | None = None) -> SampleResult:
    selected = answer or "ABCD"[(example_id - 1) % 4]
    return SampleResult(
        task="egomemreason",
        benchmark="EgoMemReason",
        dataset_sha256="1" * 64,
        evaluation_sha256="2" * 64,
        unit_id="A1_JAKE",
        question_id=f"q{example_id}",
        prediction=f"My answer is {selected}.",
        parsed_choice=selected,
        score=None,
        exact_match=None,
        latency_ms=1.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={"example_id": example_id, "choices": ("a", "b", "c", "d")},
        scorer_protocol=scorer_protocol("egomemreason"),
    )


def test_samples_report_structured_abstentions() -> None:
    abstained = replace(
        _egomem_sample(1),
        abstained=True,
        abstention_reason="insufficient_evidence",
    )

    assert abstained.json()["abstention_reason"] == "insufficient_evidence"
    assert eval_module._abstentions((abstained, _egomem_sample(2))) == {
        "count": 1,
        "rate": 0.5,
        "reasons": {"insufficient_evidence": 1},
    }


def test_query_parts_and_gallery_image_id_preserve_retrieval_evidence(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    session = MemGallerySession(
        session_id="D1",
        occurred_at=datetime(2024, 5, 2, tzinfo=timezone.utc),
        rounds=(
            MemGalleryRound(
                round_id="D1:4",
                user="I bought this.",
                assistant="It looks useful.",
                image_id="D1:IMG_002",
                image_path=image.name,
            ),
        ),
    )

    item = _gallery_memory(
        "Mira",
        session,
        session.rounds[0],
        MediaResolver("mem-gallery", tmp_path, None, None),
    )

    assert isinstance(item.content[0], str)
    assert "Image ID: D1:IMG_002" in item.content[0]
    assert item.content[1] == image
    assert _query_parts(
        "Instruction\n{question}\n{format_constraint}",
        "Where is it?",
        format_constraint="Answer briefly.",
    ) == ("Where is it?", "Instruction", "Answer briefly.")
    assert _free_text_parts("Where is it?")[0] == "Where is it?"
    assert _choice_parts("Where is it?", ("Desk", "Shelf")) == (
        "Where is it?",
        "Select the best answer using only the memories. Reply with one letter (A, B).\n"
        "A. Desk\nB. Shelf\nAnswer:",
    )


def test_catalog_covers_requested_benchmarks_and_aliases() -> None:
    assert {task.benchmark for task in TASKS.values()} == {
        "ATM-Bench",
        "BEAM",
        "CL-Bench",
        "EgoLifeQA",
        "EgoMemReason",
        "EgoTempo",
        "LoCoMo-Refined",
        "LongMemEval",
        "M3-Bench",
        "MM-Lifelong",
        "Mem-Gallery",
        "MemLens",
        "OpenEQA",
        "PersonaMem-v3",
        "SuperMemory-VQA",
        "Video-MME",
        "Video-MME-v2",
    }
    assert expand(("M3", "atm-main", "supermemory-subject-1")) == (
        "m3-bench-robot",
        "m3-bench-web",
        "supermemory-vqa",
        "atm-bench-main",
    )
    assert expand(("video-*",)) == ("video-mme", "video-mme-v2")
    assert expand(("open-eqa",)) == ("openeqa-hm3d", "openeqa-scannet")
    assert expand(("openeqa-scannet-v0",)) == ("openeqa-scannet",)
    assert all(
        task.media_source.revision is None or len(task.media_source.revision) == 40
        for task in TASKS.values()
        if task.media_source is not None
    )


def test_lmms_style_task_listing(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(("--tasks", "list", "--benchmarks-root", str(tmp_path))) == 0
    output = capsys.readouterr().out
    assert "groups:" in output
    assert "video-mme-v2" in output


def test_integrity_check_is_an_offline_json_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spec = TASKS["locomo-refined"]
    task = LoadedTask(
        spec,
        tmp_path / "dataset.json",
        "1" * 64,
        (EvalUnit("unit", (), (EvalQuestion("question", ("Prompt",), ("Answer",)),)),),
        {"dataset": "1" * 64, "memory": "2" * 64},
    )
    loaded_with: list[dict[str, object]] = []

    def load(_spec: TaskSpec, **kwargs: object) -> LoadedTask:
        assert _spec is spec
        loaded_with.append(kwargs)
        return task

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("integrity checks must remain offline and side-effect free")

    monkeypatch.setattr(eval_module, "load_task", load)
    for name in (
        "_model_config",
        "_require_output",
        "acquire_inputs",
        "prepare_task_media",
        "_execute",
    ):
        monkeypatch.setattr(eval_module, name, forbidden)
    output = tmp_path / "output"
    data = tmp_path / "data"

    assert (
        main(
            (
                "--tasks",
                spec.name,
                "--benchmarks-root",
                str(tmp_path),
                "--data-root",
                str(data),
                "--output-path",
                str(output),
                "--check-integrity",
            )
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert len(printed.splitlines()) == 1
    assert json.loads(printed) == {
        "status": "ok",
        "tasks": [
            {
                "task": spec.name,
                "dataset_sha256": task.dataset_sha256,
                "evaluation_sha256": task.evaluation_sha256,
                "unit_count": 1,
                "question_count": 1,
            }
        ],
    }
    assert loaded_with == [
        {
            "root": tmp_path.resolve(),
            "dataset_path": None,
            "media_root": None,
            "media_manifest": None,
            "manifest_directory": None,
            "limit": None,
            "offset": 0,
            "verify_digest": True,
        }
    ]
    assert not output.exists()
    assert not data.exists()


def test_result_table_includes_per_task_time_and_tokens() -> None:
    output = eval_module._table(
        {
            "tasks": [
                {
                    "task": "fixture",
                    "primary_metric": "accuracy",
                    "score": {"mean": 1.0, "confidence_interval_95": None},
                    "question_count": 2,
                    "error_count": 0,
                    "ingest_failure_count": 0,
                    "performance": {
                        "duration_seconds": {"total": 2.5, "average": 1.25},
                        "token_usage": {"total_tokens": 30, "average_tokens": 15.0},
                    },
                    "controls": {
                        "random_ranker": None,
                        "recall_at_1": None,
                        "recall_at_20": None,
                        "blind": None,
                        "is_blind_run": False,
                        "missing": ["random_ranker", "blind", "recall_at_20"],
                        "interpretable": False,
                        "reason": "fixture reports a score without its controls",
                    },
                }
            ]
        }
    )

    assert "total s" in output
    assert "avg ms" in output
    assert "30" in output
    assert "15.0" in output


def test_result_table_reports_the_writes_that_invalidated_a_score() -> None:
    # A run that loses every write still answers every question, so `error_count` stays zero and
    # the score reads INVALID beside it with nothing saying why. One real run lost 7 784 writes to
    # a transcription dependency that was not installed and printed exactly that.
    output = eval_module._table(
        {
            "tasks": [
                {
                    "task": "fixture",
                    "primary_metric": "accuracy",
                    "score": {"mean": None, "confidence_interval_95": None},
                    "question_count": 152,
                    "error_count": 0,
                    "ingest_failure_count": 7784,
                    # `_table` refuses to render a task row with no controls block, so a
                    # score can never be printed without the baselines that make it
                    # interpretable. This row is about the writes column, so the controls
                    # are all absent and correctly report themselves as missing.
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
                    "performance": {
                        "duration_seconds": {"total": 1.0, "average": 1.0},
                        "token_usage": {"total_tokens": 1, "average_tokens": 1.0},
                    },
                }
            ]
        }
    )

    assert "unwritten" in output
    assert "7784" in output


def test_cluster_statistics_and_pairing_are_seeded() -> None:
    values = (
        ScoredValue("a", "first", 1.0),
        ScoredValue("b", "first", 0.0),
        ScoredValue("c", "second", 1.0),
    )
    summary = summarize(values, seed=7, bootstrap_samples=200)

    assert summary == summarize(values, seed=7, bootstrap_samples=200)
    assert summary["mean"] == pytest.approx(2 / 3)
    assert summary["cluster_count"] == 2
    comparison = paired_comparison(
        values,
        tuple(ScoredValue(value.sample_id, value.cluster_id, 0.0) for value in values),
        seed=7,
        bootstrap_samples=200,
    )
    assert comparison["win_count"] == 2
    assert comparison["paired_sample_count"] == 3
    assert summarize(values[:2], seed=7, bootstrap_samples=20)["confidence_interval_95"] is None
    with pytest.raises(ValueError, match="identical scored samples"):
        paired_comparison(values, (), seed=7, bootstrap_samples=200)
    assert parse_choice("I don't know", ("one", "two", "three", "four")) is None
    assert parse_choice("I don't know", tuple(str(index) for index in range(10))) is None
    assert parse_choice("I", tuple(str(index) for index in range(10))) == "I"
    assert parse_choice("The answer is I.", tuple(str(index) for index in range(10))) == "I"
    assert parse_choice("B. Because it is visible.", ("one", "two", "three", "four")) == "B"
    assert parse_choice("**B**", ("one", "two", "three", "four")) == "B"
    assert parse_choice("B because it is visible.", ("one", "two", "three", "four")) == "B"
    assert parse_choice("Final: B", ("one", "two", "three", "four")) == "B"


def test_video_mme_v2_rating_uses_the_official_zero_to_one_hundred_scale() -> None:
    samples = tuple(
        SampleResult(
            task="video-mme-v2",
            benchmark="Video-MME-v2",
            dataset_sha256="1" * 64,
            evaluation_sha256="2" * 64,
            unit_id="001",
            question_id=str(position),
            prediction="A",
            parsed_choice="A",
            score=1.0 if position <= 2 else 0.0,
            exact_match=None,
            latency_ms=1.0,
            confidence=1.0,
            memory_ids=(),
            ingest_failure_count=0,
            error_code=None,
            metadata={"position": position, "group_type": "relevance"},
        )
        for position in range(1, 5)
    )

    rating = _video_mme_v2_rating(samples, seed=3, bootstrap_samples=20)

    assert rating["mean"] == 25.0
    assert score_group_answers(
        "logic", "[1, [2, 3], 4]", (True, False, True, False)
    ) == pytest.approx(100 / 3)


def test_video_mme_v2_comparison_uses_grouped_rating(tmp_path: Path) -> None:
    spec = TASKS["video-mme-v2"]
    task = LoadedTask(
        spec,
        tmp_path / "dataset.parquet",
        "1" * 64,
        (
            EvalUnit(
                "fixture",
                (),
                (
                    EvalQuestion(
                        "q1",
                        ("prompt",),
                        expected_choice="A",
                        score_kind="choice",
                    ),
                ),
            ),
        ),
    )

    def result(unit_id: str, position: int, score: float) -> SampleResult:
        return SampleResult(
            task=spec.name,
            benchmark=spec.benchmark,
            dataset_sha256=task.dataset_sha256,
            evaluation_sha256=task.evaluation_sha256,
            unit_id=unit_id,
            question_id=f"q{position}",
            prediction="A",
            parsed_choice="A" if score else "B",
            score=score,
            exact_match=None,
            latency_ms=1.0,
            confidence=1.0,
            memory_ids=(),
            ingest_failure_count=0,
            error_code=None,
            metadata={"position": position, "group_type": "relevance"},
            metrics={"question_accuracy": score},
            scorer_protocol=scorer_protocol(spec.name),
        )

    current = tuple(
        result(unit_id, position, float(position <= 2))
        for unit_id in ("g1", "g2")
        for position in range(1, 5)
    )
    baseline = tuple(
        result(unit_id, position, float(unit_id == "g1"))
        for unit_id in ("g1", "g2")
        for position in range(1, 5)
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "samples.jsonl").write_text(
        "".join(json.dumps(sample.json()) + "\n" for sample in baseline),
        encoding="utf-8",
    )
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(compare=baseline_dir, seed=7, bootstrap_samples=20),
    )

    comparison = eval_module._comparisons(arguments, (task,), current)[0]

    assert comparison["metric"] == "rating"
    assert comparison["mean"] == -25.0
    assert comparison["paired_sample_count"] == 2


def test_comparison_rejects_different_scorers_and_judges(tmp_path: Path) -> None:
    spec = TASKS["m3-bench-robot"]
    question = EvalQuestion("q1", ("prompt",), ("answer",), source_question="question")
    task = LoadedTask(
        spec,
        tmp_path / "dataset.json",
        "1" * 64,
        (EvalUnit("u1", (), (question,)),),
    )
    current = SampleResult(
        task=spec.name,
        benchmark=spec.benchmark,
        dataset_sha256=task.dataset_sha256,
        evaluation_sha256=task.evaluation_sha256,
        unit_id="u1",
        question_id="q1",
        prediction="answer",
        parsed_choice=None,
        score=1.0,
        exact_match=None,
        latency_ms=1.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={},
        metrics={"accuracy": 1.0},
        scorer_protocol=scorer_protocol(spec.name),
        judge_model="gpt-4o-2024-11-20",
    )
    cases = (
        (replace(current, scorer_protocol="old-protocol"), "scorer protocol"),
        (replace(current, judge_model="proxy-judge"), "judge model"),
    )
    for index, (baseline, message) in enumerate(cases):
        baseline_dir = tmp_path / f"baseline-{index}"
        baseline_dir.mkdir()
        (baseline_dir / "samples.jsonl").write_text(
            json.dumps(baseline.json()) + "\n",
            encoding="utf-8",
        )
        arguments = cast(
            eval_module._Arguments,
            SimpleNamespace(compare=baseline_dir, seed=7, bootstrap_samples=20),
        )

        with pytest.raises(ValueError, match=message):
            eval_module._comparisons(arguments, (task,), (current,))


def test_egomem_submission_is_upload_ready_only_when_complete(tmp_path: Path) -> None:
    samples = tuple(_egomem_sample(example_id) for example_id in range(1, 501))

    content, status = eval_module._egomem_submission(samples, requested=True, allow_partial=False)

    assert content is not None
    assert status is not None
    assert status["status"] == "ready"
    assert status["file"] == "egomemreason_submission.json"
    payload = json.loads(content)
    assert len(payload) == 500
    assert payload[0] == {"example_id": 1, "predicted_answer": "A"}
    assert payload[-1] == {"example_id": 500, "predicted_answer": "D"}
    assert [row["example_id"] for row in payload] == list(range(1, 501))
    assert all(set(row) == {"example_id", "predicted_answer"} for row in payload)

    partial_content, partial_status = eval_module._egomem_submission(
        samples[:10], requested=True, allow_partial=True
    )
    assert partial_content is None
    assert partial_status is not None and partial_status["status"] == "partial"

    invalid_content, invalid_status = eval_module._egomem_submission(
        (replace(samples[0], parsed_choice=None), *samples[1:]),
        requested=True,
        allow_partial=False,
    )
    assert invalid_content is None
    assert invalid_status is not None and invalid_status["status"] == "invalid"

    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(output_path=tmp_path, overwrite=True),
    )
    eval_module._write_artifacts(arguments, samples[:1], {}, content)
    submission_path = tmp_path / "egomemreason_submission.json"
    assert submission_path.read_bytes() == content
    assert {path.name for path in tmp_path.iterdir()} == {
        "egomemreason_submission.json",
        "results.jsonl",
        "samples.jsonl",
    }
    for path in (tmp_path / "results.jsonl", tmp_path / "samples.jsonl"):
        assert all(isinstance(json.loads(line), dict) for line in path.read_bytes().splitlines())
    eval_module._write_artifacts(arguments, samples[:1], {}, None)
    assert not submission_path.exists()


def test_mm_lifelong_ref_at_300_uses_official_quantized_iou() -> None:
    assert _ref_at_n(
        ((300.0, 600.0), (900.0, 1_200.0)),
        ((450.0, 750.0),),
        total_seconds=1_500.0,
        bucket_size=300.0,
    ) == pytest.approx(1 / 3)


def test_benchmark_speech_backend_skips_video_without_an_audio_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    silent = tmp_path / "silent.mp4"
    audible = tmp_path / "audible.mp4"
    silent.write_bytes(b"silent")
    audible.write_bytes(b"audible")
    calls: list[tuple[AssetRef, ...]] = []
    skipped_request_counts: list[int] = []

    class Backend:
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            calls.append(tuple(assets))
            return tuple(SpeechAnalysis((), ()) for _asset in assets)

    monkeypatch.setattr(eval_module, "_has_audio", lambda path: path == audible)
    monkeypatch.setattr(
        eval_module,
        "record_unmetered_model_usage",
        lambda *, request_count: skipped_request_counts.append(request_count),
    )
    backend = _BorrowedSpeechBackend(Backend())
    assets = (
        AssetRef("a" * 64, Modality.VIDEO, "video/mp4", 6, "a" * 64, path=silent),
        AssetRef("b" * 64, Modality.VIDEO, "video/mp4", 7, "b" * 64, path=audible),
    )

    assert backend.analyze(assets) == (SpeechAnalysis((), ()), SpeechAnalysis((), ()))
    assert calls == [(assets[1],)]
    assert backend.analyze((assets[0],)) == (SpeechAnalysis((), ()),)
    assert skipped_request_counts == [0]


def test_benchmark_speech_backend_satisfies_the_runtime_protocol() -> None:
    class Backend:
        transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
        transcription_model = "speech-model"
        transcription_space = "speech-space"

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            return tuple(SpeechAnalysis((), ()) for _asset in assets)

        def close(self) -> None:
            return None

    backend = _BorrowedSpeechBackend(Backend())

    assert isinstance(backend, SpeechBackend)
    assert backend.transcription_capabilities == frozenset({Modality.AUDIO, Modality.VIDEO})
    assert backend.transcription_model == "speech-model"
    assert backend.transcription_space == "speech-space"


def test_response_cache_namespace_changes_with_runner_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert eval_module.EVAL_SCHEMA_VERSION == 12
    assert eval_module.EVAL_RUNNER_VERSION == "mindbridge_eval_official_v12"
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            device=None,
            seed=7,
            gen_kwargs="{}",
            recall_limit=8,
            blind=False,
            ingest="add",
            compile_max_items=24,
            compile_max_chars=16000,
        ),
    )
    before = _cache_namespace(arguments, ModelConfig(), {"text": 1})

    monkeypatch.setattr(eval_module, "EVAL_RUNNER_VERSION", "next-runner-recipe")

    assert _cache_namespace(arguments, ModelConfig(), {"text": 1}) != before
    assert (
        _cache_namespace(arguments, ModelConfig(generation_min_video_seconds=2.0), {"text": 1})
        != before
    )


def test_backend_pool_warms_query_embedding_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    calls: list[tuple[tuple[ModelInput, ...], EmbedTask]] = []

    class Client:
        def __init__(self, **_kwargs: object) -> None: ...

        def close(self) -> None: ...

    class Models(Client):
        pass

    class Embedder:
        def __init__(self, **_kwargs: object) -> None: ...

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask
        ) -> tuple[tuple[float, ...], ...]:
            calls.append((tuple(inputs), task))
            return ((1.0,),)

        def close(self) -> None: ...

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(eval_module, "OpenAIModels", Models)
    monkeypatch.setattr(eval_module, "JinaOmniEmbedder", Embedder)

    pool = eval_module._BackendPool(
        ModelConfig(), device="cuda", batch_size=8, needs_speech=False, seed=7
    )
    pool.close()

    assert calls == [((ModelInput(text="MindBridge benchmark warmup"),), EmbedTask.QUERY)]


def test_backend_pool_forwards_every_memory_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hand-written forwarding list drops new policy or a new plugin silently, which is worse
    than crashing: the run measures the default composition while the artifact reports the
    configured one."""
    captured: dict[str, object] = {}

    class Recorder:
        def __init__(self, _data_dir: object, **values: object) -> None:
            captured.update(values)

    monkeypatch.setattr(eval_module, "AsyncMemory", Recorder)
    pool = object.__new__(eval_module._BackendPool)
    settings = MemoryConfig(evidence_budget_chars=4_242, minimum_relevance=0.11)
    pool._settings = settings
    pool._tracer = trace.get_tracer(__name__)
    for entry in fields(MemoryPlugins):
        setattr(pool, f"_{entry.name}", None)
    # Every declared plugin slot gets its own distinguishable sentinel, so a dropped or
    # mis-mapped backend cannot pass by looking like its neighbour or like "not configured".
    backends = {entry.name: f"backend:{entry.name}" for entry in fields(MemoryPlugins)}
    pool._settings = settings
    pool._tracer = cast(Tracer, None)
    for name, value in backends.items():
        setattr(pool, f"_{name}", value)

    pool.memory(Path("unused"))

    for entry in fields(MemoryConfig):
        assert captured[entry.name] == getattr(settings, entry.name), entry.name
    for name, value in backends.items():
        assert captured[name] == value, name
    # Deriving the forwarding from the dataclasses is only safe while every declared field names
    # a real constructor keyword; a field added without one would raise at call time instead.
    declared = {entry.name for entry in (*fields(MemoryConfig), *fields(MemoryPlugins))}
    for constructor in (Memory.__init__, AsyncMemory.__init__):
        unaccepted = declared - set(signature(constructor).parameters)
        assert not unaccepted, f"{constructor.__qualname__} has no keyword for {sorted(unaccepted)}"


def test_backend_pool_forwards_every_capability_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """`former` was declared on `MemoryPlugins`, built by the SDK adapter, and used by `Memory`,
    yet the harness never passed it, so no benchmark run has ever exercised derived-memory
    formation. Deriving the expected keywords from the dataclass makes the next omission red."""
    captured: dict[str, object] = {}

    class Recorder:
        def __init__(self, _data_dir: object, **values: object) -> None:
            captured.update(values)

    monkeypatch.setattr(eval_module, "AsyncMemory", Recorder)
    pool = object.__new__(eval_module._BackendPool)
    pool._settings = MemoryConfig()
    pool._tracer = trace.get_tracer(__name__)
    markers = {entry.name: object() for entry in fields(MemoryPlugins)}
    for name, marker in markers.items():
        setattr(pool, f"_{name}", marker)

    pool.memory(Path("unused"))

    for name, marker in markers.items():
        assert captured[name] is marker, name


def test_implementation_identity_tracks_editable_source_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "mindbridge"
    benchmark = package / "benchmarks" / "eval.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("RUNNER = 1\n", encoding="utf-8")
    implementation = package / "memory.py"
    implementation.write_text("RECIPE = 1\n", encoding="utf-8")
    monkeypatch.setattr(eval_module, "__file__", str(benchmark))

    identity = eval_module._implementation_identity()

    assert eval_module._implementation_identity() == identity
    implementation.write_text("RECIPE = 2\n", encoding="utf-8")
    assert eval_module._implementation_identity() != identity


def test_benchmark_device_lock_serializes_separate_processes(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from mindbridge.benchmarks.eval import _benchmark_device_lock

print("waiting", flush=True)
with _benchmark_device_lock("cuda", enabled=True, quiet=True, lock_root=Path(sys.argv[1])):
    print("acquired", flush=True)
"""
    with eval_module._benchmark_device_lock("cuda", enabled=True, quiet=True, lock_root=tmp_path):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline() == "waiting\n"
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.1)
    stdout, stderr = process.communicate(timeout=5)

    assert stdout == "acquired\n"
    assert stderr == ""


def test_benchmark_device_lock_uses_the_physical_cuda_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_module,
        "_nvidia_device_uuids",
        lambda: {0: "GPU-zero", 1: "GPU-one"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    masked = eval_module._physical_cuda_identity("cuda:0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    unmasked = eval_module._physical_cuda_identity("cuda:1")

    assert masked == unmasked == "gpu-one"


def test_benchmark_device_lock_skips_auto_when_cuda_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert eval_module._physical_cuda_identity("auto") is None


def test_benchmark_device_lock_can_be_disabled_for_manual_scheduling(tmp_path: Path) -> None:
    with (
        eval_module._benchmark_device_lock("cuda", enabled=True, quiet=True, lock_root=tmp_path),
        eval_module._benchmark_device_lock("cuda", enabled=False, quiet=True, lock_root=tmp_path),
    ):
        pass


def test_configured_devices_are_locked_in_physical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "jina-omni", "device": "cuda:0"},
            "speech": {"provider": "funasr", "device": "cuda:1"},
        }
    )
    identities = {"cuda:0": "gpu-z", "cuda:1": "gpu-a"}
    monkeypatch.setattr(
        eval_module,
        "_physical_cuda_identity",
        lambda device: identities[device],
    )

    assert eval_module._evaluation_devices(None, config) == ("cuda:1", "cuda:0")


def test_text_only_runs_do_not_lock_the_speech_device(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai", "model": "embed", "dimension": 4},
            "speech": {"provider": "funasr", "device": "cuda:1"},
        }
    )
    monkeypatch.setattr(eval_module, "_physical_cuda_identity", lambda device: device)

    assert eval_module._evaluation_devices(None, config, needs_speech=True) == ("cuda:1",)
    assert eval_module._evaluation_devices(None, config, needs_speech=False) == ()


def test_model_base_url_override_replaces_environment_operation_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDBRIDGE_GENERATION_BASE_URL", "https://old.example/v1")

    config = _model_config("mindbridge", "base_url=https://new.example/v1")

    assert config.generation_base_url == "https://new.example/v1"


def test_model_short_video_limit_is_explicit_and_validated() -> None:
    config = _model_config("mindbridge", "generation_min_video_seconds=2")

    assert config.generation_min_video_seconds == 2.0
    with pytest.raises(ValueError, match="generation_min_video_seconds"):
        _model_config("mindbridge", "generation_min_video_seconds=0")


def test_eval_config_reuses_the_declarative_memory_schema(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "data_dir": "ignored-by-benchmark",
                "embedding": {
                    "provider": "jina-omni",
                    "device": "cpu",
                    "batch_size": 4,
                },
                "generation": {
                    "provider": "openai",
                    "model": "configured-model",
                    "base_url": "https://configured.example/v1",
                    "timeout": 45.0,
                },
                "speech": {"provider": "funasr", "device": "cpu"},
                "settings": {"index_speech": True, "minimum_relevance": 0.2},
            }
        ),
        encoding="utf-8",
    )

    loaded, overrides = eval_module._load_memory_config(path)
    assert loaded is not None
    assert overrides == HarnessOverrides()
    model = _model_config(
        "mindbridge",
        "generation_model=override-model",
        memory_config=loaded,
    )
    model = replace(
        model,
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
    )
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            gen_kwargs="temperature=0,do_sample=false,seed=7,max_tokens=512",
            seed=7,
            device="cuda:1",
            recall_limit=20,
            model="mindbridge",
            blind=False,
            ingest="add",
            compile_max_items=24,
            compile_max_chars=16000,
        ),
    )
    effective = eval_module._evaluation_memory_config(loaded, model, arguments)

    assert effective is not None and effective.generation is not None
    assert effective.data_dir == Path("ignored-by-benchmark")
    assert getattr(effective.embedding, "device", None) == "cuda:1"
    assert effective.speech is not None
    assert getattr(effective.speech, "device", None) == "cuda:1"
    assert effective.generation.model == "override-model"
    assert effective.generation.base_url == "https://configured.example/v1"
    assert effective.generation.temperature == 0.0
    assert effective.generation.seed == 7
    assert effective.generation.max_tokens == 512
    assert effective.generation.modalities == frozenset({Modality.TEXT, Modality.IMAGE})
    assert effective.settings.minimum_relevance == 0.2
    assert "data_dir" not in eval_module._memory_config_payload(effective)
    assert _cache_namespace(arguments, model, {"task": 4}) != _cache_namespace(
        arguments,
        model,
        {"task": 4},
        memory_config=effective,
    )
    recorded = eval_module._model_result(arguments, model, effective)
    assert recorded["generation_model"] == "override-model"
    assert recorded["memory_config"] == eval_module._memory_config_payload(effective)


def test_configured_backend_pool_lends_plugins_and_closes_the_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []

    class Embedder:
        embedding_capabilities = frozenset({Modality.TEXT})
        embedding_model = "embedder"
        embedding_space = "embedder-space"
        embedding_dimension = 2

        def embed(self, *_args: object, **_kwargs: object) -> tuple[tuple[float, ...], ...]:
            return ((0.0, 1.0),)

        def close(self) -> None:
            raise AssertionError("a borrowed plugin must not close its owner")

    class Answerer:
        generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

        def answer(self, *_args: object, **_kwargs: object) -> AnswerResult:
            raise AssertionError("not called")

        def close(self) -> None:
            raise AssertionError("a borrowed plugin must not close its owner")

    plugins = MemoryPlugins(embedder=Embedder(), answerer=Answerer())
    settings = MemoryConfig(minimum_relevance=0.2)
    resolved = SimpleNamespace(
        plugins=plugins,
        settings=settings,
        close=lambda: closed.append(True),
    )
    configured: list[MindBridgeConfig] = []

    def resolve(config: MindBridgeConfig) -> object:
        configured.append(config)
        return resolved

    monkeypatch.setattr(eval_module, "resolve_memory_config", resolve)
    memory_config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai"},
        }
    )

    pool = eval_module._BackendPool(
        ModelConfig(generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})),
        device=None,
        batch_size=1,
        needs_speech=False,
        seed=0,
        memory_config=memory_config,
    )

    # Every configured setting survives, except that measurement turns answer-time reinforcement
    # off: reinforcing mid-run makes one question's retrieval depend on which earlier questions
    # answered, so a run would stop being reproducible from its seed.
    assert pool._settings == replace(settings, reinforce_on_answer=False)
    assert settings.reinforce_on_answer is True
    assert pool._settings.reinforce_on_answer is False
    assert configured[0].generation is not None
    assert configured[0].generation.modalities == frozenset({Modality.TEXT, Modality.IMAGE})
    assert pool._answerer is not None
    assert pool._answerer.generation_capabilities == frozenset({Modality.TEXT, Modality.IMAGE})
    asyncio.run(pool.memory(tmp_path).close())
    pool._embedder.close()
    assert closed == []
    pool.close()
    assert closed == [True]


def test_backend_pool_borrows_every_configured_plugin_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every declared plugin slot must survive the trip from configuration to the store.

    A slot the pool never captures is silently absent: the artifact reports the configured
    composition while the run measures the default one.
    """
    captured: dict[str, object] = {}
    names = tuple(entry.name for entry in fields(MemoryPlugins))
    plugins = SimpleNamespace(**{name: SimpleNamespace(marker=name) for name in names})
    resolved = SimpleNamespace(
        plugins=plugins,
        settings=MemoryConfig(),
        close=lambda: None,
    )

    class Recorder:
        def __init__(self, _data_dir: object, **values: object) -> None:
            captured.update(values)

    monkeypatch.setattr(eval_module, "resolve_memory_config", lambda _config: resolved)
    monkeypatch.setattr(eval_module, "AsyncMemory", Recorder)
    pool = eval_module._BackendPool(
        ModelConfig(),
        device=None,
        batch_size=1,
        needs_speech=False,
        seed=0,
        memory_config=MindBridgeConfig.model_validate({"embedding": {"provider": "openai"}}),
    )

    pool.memory(tmp_path)

    for name in names:
        borrowed = captured[name]
        assert borrowed is not None, name
        # Borrowed, not the owner itself: a per-store close must not close a shared backend.
        assert borrowed is not getattr(plugins, name), name
        assert getattr(borrowed, "marker", None) == name, name


def test_borrowed_face_backend_preserves_the_runtime_protocol() -> None:
    class Analyzer:
        face_capabilities = frozenset({Modality.IMAGE})
        face_model = "face"
        face_space = "face-space"
        face_analysis_space = "analysis-space"

        def analyze(self, _assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
            return (FaceAnalysis(()),)

        def close(self) -> None:
            raise AssertionError("a borrowed plugin must not close its owner")

    borrowed = _BorrowedFaceBackend(Analyzer())

    assert getattr_static(borrowed, "analyze") is not None
    assert isinstance(borrowed, FaceBackend)
    assert borrowed.analyze(()) == (FaceAnalysis(()),)
    borrowed.close()


@pytest.mark.parametrize("key", ("temperature", "seed", "max_tokens"))
def test_eval_config_rejects_benchmark_controls_in_extra_body(
    key: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "embedding": {"provider": "openai"},
                "generation": {"provider": "openai", "extra_body": {key: 1}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"benchmark controls: {key}"):
        eval_module._load_memory_config(path)


def test_generation_kwargs_normalize_bounded_non_thinking_inference() -> None:
    assert _generation_kwargs("max_tokens=512,enable_thinking=0", 7) == (
        "temperature=0,do_sample=false,seed=7,max_tokens=512,enable_thinking=false"
    )
    with pytest.raises(ValueError, match="positive integer"):
        _generation_kwargs("max_tokens=0", 7)


@pytest.mark.asyncio
async def test_memlens_question_date_is_a_reference_clock_not_query_text(tmp_path: Path) -> None:
    dataset = tmp_path / "memlens.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "preference",
                    "question": "What is my favorite color?",
                    "answer": "Blue.",
                    "question_date": "2025/01/15 (Wed) 10:00",
                    "haystack_dates": ["2025/01/14 (Tue) 09:00"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "My favorite color is blue."}]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    auxiliary = tmp_path / "memlens" / "agent_subset_195.json"
    auxiliary.parent.mkdir()
    auxiliary.write_text(
        json.dumps({"n_questions": 1, "question_ids": ["q1"]}),
        encoding="utf-8",
    )
    loaded = load_task(
        TASKS["memlens-32k"],
        root=tmp_path,
        dataset_path=dataset,
        verify_digest=False,
    )
    question = loaded.units[0].questions[0]
    observed: list[datetime | None] = []

    class Memory:
        async def ask(
            self,
            _question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            assert limit == 5
            observed.append(reference_at)
            return AnswerResult("Blue.")

    await _answer_many(
        cast(AsyncMemory, Memory()),
        (question,),
        request_concurrency=1,
        recall_limit=5,
    )

    assert "Question date:" not in str(question.content[0])
    assert observed == [datetime(2025, 1, 15, 10, tzinfo=timezone.utc)]


@pytest.mark.asyncio
async def test_answer_failure_preserves_the_product_retrieval_ranking(tmp_path: Path) -> None:
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    ranked = (
        SearchHit(
            id="memory-1",
            content="supporting memory",
            score=0.8,
            created_at=now,
            metadata={"source_id": "source-1"},
        ),
    )

    class Memory:
        async def ask(self, _question: object, **_kwargs: object) -> AnswerResult:
            raise RuntimeError("generation unavailable")

        async def search(self, _query: object, **_kwargs: object) -> tuple[SearchHit, ...]:
            return ranked

    question = EvalQuestion("q1", ("question",), references=("answer",))
    outcome = (
        await _answer_many(
            cast(AsyncMemory, Memory()),
            (question,),
            request_concurrency=1,
            recall_limit=5,
        )
    )[0]
    assert not isinstance(outcome, BaseException)
    unit = EvalUnit("unit", (), (question,))
    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (unit,),
    )
    sample = eval_module._sample(
        task,
        unit,
        question,
        outcome,
        ingest_failures=0,
        predict_only=True,
        log_samples=False,
    )

    assert sample.error_code == "RuntimeError"
    assert sample.memory_ids == ("memory-1",)
    assert sample.evidence[0].source_id == "source-1"


def test_a_provider_error_leaves_the_answer_unscored_but_keeps_retrieval(tmp_path: Path) -> None:
    """A 500 is a missing answer, not a wrong one.

    Scoring the empty prediction gave every provider failure an f1 of 0.0 and deflated the
    arms unevenly (one run: blind errored 3.4x more than the product arm). The retriever
    did rank before the generator failed, so its diagnostics stay.
    """
    question = EvalQuestion(
        "q1",
        ("question",),
        references=("answer",),
        metadata={"clue_ids": ("gold",)},
    )
    unit = EvalUnit("unit", (MemoryItem("gold", ("the answer",)),), (question,))
    # Mem-Gallery scores answers deterministically (f1 of "" is 0.0), so this family shows
    # the difference between "unscored" and "scored zero"; ATM's answer score is judge-only.
    task = LoadedTask(
        TaskSpec("mem-gallery", "Mem-Gallery", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (unit,),
    )
    outcome = eval_module._AnswerOutcome(
        "",
        12.0,
        0.0,
        (),
        (),
        error=ModelError("upstream 500", reason="model_failed", stage="generate"),
        ranked_source_ids=("gold", "other"),
        ranked_source_ids_complete=True,
    )

    sample = eval_module._sample(
        task, unit, question, outcome, ingest_failures=0, predict_only=False, log_samples=False
    )

    assert sample.error_code == "model_error"
    assert sample.score is None
    assert sample.metrics and all(name.startswith("retrieval_") for name in sample.metrics)
    assert "f1" not in sample.metrics


def test_run_identifier_cannot_escape_the_default_output_root() -> None:
    with pytest.raises(ArgumentTypeError):
        _run_identifier("../escape")
    assert _seed_values("7") == (7, 7, 7, 7)
    assert _seed_values("0,1,2") == (0, 1, 2, 1234)


def test_causal_task_requires_timestamped_prepared_media(tmp_path: Path) -> None:
    dataset = tmp_path / "robot.json"
    dataset.write_text(
        json.dumps(
            {
                "room": {
                    "video_path": "room.mp4",
                    "qa_list": [
                        {
                            "question": "Where?",
                            "answer": "There.",
                            "question_id": "q1",
                            "type": ["recall"],
                            "timestamp": "00:10",
                        }
                    ],
                },
                "z-room": {
                    "video_path": "z-room.mp4",
                    "qa_list": [
                        {
                            "question": "Ignored?",
                            "answer": "Yes.",
                            "question_id": "q2",
                            "type": ["recall"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "media"
    media.mkdir()
    (media / "room.mp4").touch()
    spec = TASKS["m3-bench-robot"]

    with pytest.raises(ValueError, match="must declare end_seconds"):
        load_task(
            spec,
            root=tmp_path,
            dataset_path=dataset,
            media_root=media,
            limit=1,
            verify_digest=False,
        )

    loaded = load_task(
        spec,
        root=tmp_path,
        dataset_path=dataset,
        media_manifest={
            "tasks": {
                spec.name: {
                    "units": {
                        "room": [
                            {
                                "text": "prepared segment",
                                "source_id": "room-0001",
                                "start_seconds": 0,
                                "end_seconds": 9,
                            }
                        ]
                    }
                }
            }
        },
        limit=1,
        verify_digest=False,
    )
    assert loaded.units[0].memories[0].end_seconds == 9
    assert loaded.units[0].questions[0].cutoff_seconds == 10
    assert "media_manifest" in loaded.input_sha256
    assert "memory" in loaded.input_sha256


def test_egotempo_limit_counts_questions_not_shared_video_units(tmp_path: Path) -> None:
    dataset = tmp_path / "egotempo.json"
    dataset.write_text(
        json.dumps(
            {
                "info": {"release date": "19.03.2025", "version": "1.0"},
                "annotations": [
                    {
                        "question_id": f"q{index}",
                        "clip_id": clip,
                        "question_type": "object",
                        "question": "What happened?",
                        "answer": "An event.",
                    }
                    for index, clip in enumerate(
                        ("video_0.0_30.0", "video_0.0_30.0", "other_0.0_30.0"), start=1
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "tasks": {
            "egotempo": {
                "units": {
                    "video_0.0_30.0": [{"text": "clip one"}],
                    "other_0.0_30.0": [{"text": "clip two"}],
                }
            }
        }
    }

    loaded = load_task(
        TASKS["egotempo"],
        root=tmp_path,
        dataset_path=dataset,
        media_manifest=manifest,
        limit=1,
        verify_digest=False,
    )

    assert len(loaded.units) == 1
    assert [question.question_id for question in loaded.units[0].questions] == ["q1"]


class _FakeMemory:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def add_many(
        self,
        contents: Sequence[object],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        self.events.extend(f"add:{content}" for content in contents)
        return ()

    async def add(self, content: object, **_kwargs: object) -> object:
        self.events.append(f"add:{content}")
        return object()

    async def ask(
        self,
        question: object,
        *,
        limit: int,
        reference_at: datetime | None = None,
    ) -> AnswerResult:
        del reference_at
        self.events.append(f"ask:{question}:{limit}")
        return AnswerResult("A")


class _BatchLimitedMemory(_FakeMemory):
    async def add_many(
        self,
        contents: Sequence[object],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        self.events.append(f"batch:{len(contents)}")
        if len(contents) > 2:
            raise RuntimeError("batch too large")
        return ()


class _ProvenanceMemory(_FakeMemory):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.kwargs: dict[str, object] = {}

    async def add_many(
        self,
        contents: Sequence[object],
        **kwargs: object,
    ) -> tuple[object, ...]:
        self.kwargs = kwargs
        return await super().add_many(contents)


@pytest.mark.asyncio
async def test_ingest_bisects_a_failed_batch_without_losing_items() -> None:
    events: list[str] = []
    memory = _BatchLimitedMemory(events)
    items = tuple(MemoryItem(str(index), (str(index),)) for index in range(4))

    assert await _ingest(cast(AsyncMemory, memory), items, batch_size=4) == 0
    assert events == ["batch:4", "batch:2", "batch:2"]


@pytest.mark.asyncio
async def test_ingest_records_the_failed_source_and_stable_error_detail() -> None:
    class Memory(_FakeMemory):
        async def add_many(
            self, contents: Sequence[object], **_kwargs: object
        ) -> tuple[object, ...]:
            del contents
            raise ModelError(
                "speech failed",
                reason="unsupported_backend",
                stage="analyze",
            ) from TypeError("protocol mismatch")

        async def add(self, content: object, **_kwargs: object) -> object:
            del content
            raise ModelError(
                "speech failed",
                reason="unsupported_backend",
                stage="analyze",
            ) from TypeError("protocol mismatch")

    failures: list[eval_module.FailureDetail] = []

    count = await _ingest(
        cast(AsyncMemory, Memory([])),
        (MemoryItem("clip-7", ("content",)),),
        batch_size=1,
        on_failure=failures.append,
    )

    assert count == 1
    assert failures == [
        eval_module.FailureDetail(
            source_id="clip-7",
            code="model_error",
            reason="unsupported_backend",
            stage="analyze",
            cause_type="TypeError",
        )
    ]


@pytest.mark.asyncio
async def test_ingest_announces_the_first_failure_once_with_its_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Memory(_FakeMemory):
        async def add_many(
            self, contents: Sequence[object], **_kwargs: object
        ) -> tuple[object, ...]:
            del contents
            raise ModelError(
                "embedding response was invalid: the model returned 2048 values but the "
                "configured dimension is 1536",
                reason="response_invalid",
                stage="embed",
            )

        async def add(self, content: object, **_kwargs: object) -> object:
            del content
            raise ModelError(
                "embedding response was invalid: the model returned 2048 values but the "
                "configured dimension is 1536",
                reason="response_invalid",
                stage="embed",
            )

    monkeypatch.setattr(eval_module, "_first_ingest_failure_announced", False)
    items = tuple(MemoryItem(f"turn-{index}", ("content",)) for index in range(3))

    count = await _ingest(cast(AsyncMemory, Memory([])), items, batch_size=3)

    captured = capsys.readouterr().err
    assert count == 3
    assert captured.count("first ingest failure") == 1
    assert "turn-0" in captured
    assert "model_error/response_invalid" in captured
    assert "returned 2048 values but the configured dimension is 1536" in captured


def test_progress_reporter_writes_milestones_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = eval_module._progress_reporter("running fixture", "samples")

    report(1, 20)
    report(2, 20)
    report(3, 20)
    report(20, 20)

    assert capsys.readouterr().err.splitlines() == [
        "mindbridge-bench eval: running fixture: 1/20 samples (5%)",
        "mindbridge-bench eval: running fixture: 2/20 samples (10%)",
        "mindbridge-bench eval: running fixture: 20/20 samples (100%)",
    ]
    eval_module._progress_reporter("running fixture", "samples", enabled=False)(1, 1)
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_judge_skips_samples_with_ingest_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = replace(_egomem_sample(1), ingest_failure_count=1)
    calls = 0

    def unexpected_plan(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(eval_module, "judge_plan", unexpected_plan)

    result = await eval_module._apply_judges(
        (),
        (sample,),
        arguments=cast(eval_module._Arguments, SimpleNamespace(quiet=True)),
        config=eval_module._JudgeConfig("judge", "https://judge.example/v1"),
    )

    assert result == (sample,)
    assert calls == 0


@pytest.mark.asyncio
async def test_runner_records_index_failure_without_recursive_ingest(tmp_path: Path) -> None:
    calls = 0

    class Memory(_FakeMemory):
        async def add_many(
            self,
            contents: Sequence[object],
            **_kwargs: object,
        ) -> tuple[object, ...]:
            nonlocal calls
            calls += 1
            raise IndexUnavailableError("index exhausted file descriptors")

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory([]))

        async def __aexit__(self, *_error: object) -> None:
            return None

    spec = TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40)
    task = LoadedTask(
        spec,
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                tuple(MemoryItem(str(number), (str(number),)) for number in range(4)),
                (EvalQuestion("q1", ("question",), references=("answer",)),),
            ),
        ),
    )

    telemetry = EvaluationTelemetry()
    try:
        samples = await run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
            memory_factory=cast(MemoryFactory, lambda _path: Context()),
            batch_size=4,
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=1,
            tracer=telemetry.tracer,
        )
        performance = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    assert calls == 1
    assert samples[0].error_code == "index_unavailable"
    assert cast(dict[str, object], performance["search_e2e"])["attempt_count"] == 0


@pytest.mark.asyncio
async def test_ingest_preserves_event_time_and_source_interval() -> None:
    occurred = datetime(2026, 8, 20, 9, tzinfo=timezone.utc)
    memory = _ProvenanceMemory([])
    item = MemoryItem("clip-1", ("event",), 10.0, 20.0, occurred)

    assert await _ingest(cast(AsyncMemory, memory), (item,), batch_size=1) == 0
    assert memory.kwargs == {
        "occurred_at": (occurred,),
        "occurred_end": (None,),
        "metadata": ({"source_id": "clip-1", "start_seconds": 10.0, "end_seconds": 20.0},),
        "memory_type": MemoryType.EPISODIC,
    }


class _FakeContext:
    def __init__(self, events: list[str]) -> None:
        self.memory = _FakeMemory(events)

    async def __aenter__(self) -> AsyncMemory:
        return cast(AsyncMemory, self.memory)

    async def __aexit__(self, *_error: object) -> None:
        return None


@pytest.mark.asyncio
async def test_runner_ingests_only_memory_available_at_each_cutoff(tmp_path: Path) -> None:
    spec = TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40)
    task = LoadedTask(
        spec,
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                (
                    MemoryItem("early", ("early",), end_seconds=5),
                    MemoryItem("later", ("later",), end_seconds=15),
                ),
                (
                    EvalQuestion(
                        "q1",
                        ("first",),
                        expected_choice="A",
                        score_kind="choice",
                        cutoff_seconds=10,
                    ),
                    EvalQuestion(
                        "q2",
                        ("second",),
                        expected_choice="A",
                        score_kind="choice",
                        cutoff_seconds=20,
                    ),
                ),
            ),
        ),
    )
    events: list[str] = []
    progress: list[tuple[int, int]] = []

    def factory(_path: Path) -> _FakeContext:
        return _FakeContext(events)

    samples = await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
        memory_factory=cast(MemoryFactory, factory),
        batch_size=8,
        unit_concurrency=1,
        request_concurrency=2,
        recall_limit=5,
        on_progress=lambda completed, total: progress.append((completed, total)),
    )

    assert events == [
        "add:[source_id: early]\nearly",
        "ask:first:5",
        "add:[source_id: later]\nlater",
        "ask:second:5",
    ]
    assert progress == [(1, 2), (2, 2)]
    assert [sample.score for sample in samples] == [1.0, 1.0]


@pytest.mark.asyncio
async def test_runner_rejects_gold_retrieval_scoring_at_a_causal_cutoff(
    tmp_path: Path,
) -> None:
    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                (),
                (
                    EvalQuestion(
                        "q1",
                        ("question",),
                        references=("answer",),
                        cutoff_seconds=10,
                        metadata={"evidence_ids": ("gold",)},
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="isolated store snapshot"):
        await run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
            memory_factory=cast(MemoryFactory, lambda _path: _FakeContext([])),
            batch_size=1,
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=1,
        )


@pytest.mark.asyncio
async def test_runner_allows_random_retrieval_at_a_causal_cutoff(tmp_path: Path) -> None:
    class Memory(_FakeMemory):
        async def search(
            self,
            query: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> tuple[SearchHit, ...]:
            del query, limit, reference_at
            return ()

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory([]))

        async def __aexit__(self, *_error: object) -> None:
            return None

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                (),
                (
                    EvalQuestion(
                        "q1",
                        ("question",),
                        references=("answer",),
                        cutoff_seconds=10,
                        metadata={"evidence_ids": ("gold",)},
                    ),
                ),
            ),
        ),
    )

    samples = await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
        memory_factory=cast(MemoryFactory, lambda _path: Context()),
        batch_size=1,
        unit_concurrency=1,
        request_concurrency=1,
        recall_limit=1,
        arms=(eval_module._Arm("random", seed=7),),
    )

    assert len(samples) == 1
    assert samples[0].arm == "random"


@pytest.mark.asyncio
async def test_runner_applies_request_concurrency_across_units(tmp_path: Path) -> None:
    active = 0
    peak = 0

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            nonlocal active, peak
            del question, limit, reference_at
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return AnswerResult("A")

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory([]))

        async def __aexit__(self, *_error: object) -> None:
            return None

    spec = TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40)
    task = LoadedTask(
        spec,
        tmp_path / "fixture.json",
        "1" * 64,
        tuple(
            EvalUnit(
                f"unit-{index}",
                (),
                (
                    EvalQuestion(
                        f"q{index}",
                        ("question",),
                        expected_choice="A",
                        score_kind="choice",
                    ),
                ),
            )
            for index in range(2)
        ),
    )

    await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
        memory_factory=cast(MemoryFactory, lambda _path: Context()),
        batch_size=1,
        unit_concurrency=2,
        request_concurrency=1,
        recall_limit=1,
    )

    assert peak == 1


@pytest.mark.asyncio
async def test_standalone_search_reopens_warm_stores_after_every_answer(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            events.append(f"answer-start:{question}")
            await asyncio.sleep(0)
            events.append(f"answer-end:{question}")
            return AnswerResult("A")

        async def search(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> tuple[SearchHit, ...]:
            del reference_at
            events.append(f"search:{limit}:{question}")
            return (
                SearchHit(
                    id="memory-gold",
                    content="gold",
                    score=1.0,
                    created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                    metadata={"source_id": "gold"},
                ),
            )

    class Context:
        def __init__(self, path: Path) -> None:
            self.path = path

        async def __aenter__(self) -> AsyncMemory:
            events.append(f"open:{self.path.name}")
            return cast(AsyncMemory, Memory(events))

        async def __aexit__(self, *_error: object) -> None:
            events.append(f"close:{self.path.name}")

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        tuple(
            EvalUnit(
                f"unit-{index}",
                (),
                (EvalQuestion(f"q{index}", (f"question-{index}",), references=("A",)),),
            )
            for index in range(2)
        ),
    )
    telemetry = EvaluationTelemetry()
    try:
        samples = await run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
            memory_factory=cast(MemoryFactory, lambda path: Context(path)),
            batch_size=1,
            unit_concurrency=2,
            request_concurrency=1,
            recall_limit=3,
            tracer=telemetry.tracer,
        )
        performance = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    answer_ends = [index for index, event in enumerate(events) if event.startswith("answer-end:")]
    searches = [index for index, event in enumerate(events) if event.startswith("search:")]
    assert len(samples) == 2
    assert len(answer_ends) == len(searches) == 2
    assert max(answer_ends) < min(searches)
    assert sum(event.startswith("open:") for event in events) == 4
    search_e2e = cast(dict[str, object], performance["search_e2e"])
    assert search_e2e["measurement"] == "post_answer_warm_store_replay"
    assert search_e2e["attempt_count"] == search_e2e["count"] == 2


@pytest.mark.asyncio
async def test_run_arms_defers_replay_until_every_task_answer_finishes(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            events.append(f"answer:{question}")
            return AnswerResult("A")

        async def search(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> tuple[SearchHit, ...]:
            del reference_at
            events.append(f"search:{limit}:{question}")
            return (
                SearchHit(
                    id="memory-gold",
                    content="gold",
                    score=1.0,
                    created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                    metadata={"source_id": "gold"},
                ),
            )

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory(events))

        async def __aexit__(self, *_error: object) -> None:
            return None

    def task(name: str) -> LoadedTask:
        return LoadedTask(
            TaskSpec(name, "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
            tmp_path / f"{name}.json",
            "1" * 64,
            (
                EvalUnit(
                    "unit",
                    (),
                    (
                        EvalQuestion(
                            "q1",
                            (f"question-{name}",),
                            references=("A",),
                            metadata={"evidence_ids": ("gold",)},
                        ),
                    ),
                ),
            ),
        )

    tasks = (task("task-a"), task("task-b"))
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            compile_max_items=24,
            compile_max_chars=16_000,
            quiet=True,
            data_root=tmp_path / "stores",
            run_id="run",
            unit_concurrency=1,
            request_concurrency=1,
            recall_limit=3,
            predict_only=False,
            log_samples=False,
            arms=(eval_module.DEFAULT_ARM,),
            full_context_chars=24_000,
            ingest="add",
        ),
    )
    telemetry = EvaluationTelemetry()
    try:
        samples = await eval_module._run_arms(
            tasks,
            arguments,
            arms=(eval_module.PRODUCT_ARM,),
            batch_sizes={task.spec.name: 1 for task in tasks},
            memory_factory=cast(MemoryFactory, lambda _path: Context()),
            response_cache=None,
            tracer=telemetry.tracer,
        )
        performance = {
            task.spec.name: telemetry.result(task.spec.name, question_count=1) for task in tasks
        }
    finally:
        telemetry.close()

    answers = [index for index, event in enumerate(events) if event.startswith("answer:")]
    warm_searches = [index for index, event in enumerate(events) if event.startswith("search:3:")]
    quality_searches = [
        index for index, event in enumerate(events) if event.startswith("search:100:")
    ]
    assert len(samples) == len(answers) == len(warm_searches) == len(quality_searches) == 2
    assert max(answers) < min(warm_searches)
    assert max(warm_searches) < min(quality_searches)
    assert all(sample.ranked_source_ids == ("gold",) for sample in samples)
    for result in performance.values():
        search_e2e = cast(dict[str, object], result["search_e2e"])
        assert search_e2e["attempt_count"] == search_e2e["count"] == 1


@pytest.mark.asyncio
async def test_standalone_search_reports_planned_error_when_store_reopen_fails(
    tmp_path: Path,
) -> None:
    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                (),
                (EvalQuestion("q1", ("question",), references=("A",)),),
            ),
        ),
    )
    sample = replace(
        _egomem_sample(1),
        task="fixture",
        unit_id="unit",
        question_id="q1",
        cached=False,
    )

    def failed_factory(_path: Path) -> object:
        raise RuntimeError("store cannot be reopened")

    telemetry = EvaluationTelemetry()
    try:
        await eval_module._measure_standalone_searches(
            task,
            slots=((sample,),),
            unit_paths=(tmp_path / "store",),
            stores_ready=(True,),
            memory_factory=cast(MemoryFactory, failed_factory),
            unit_concurrency=1,
            request_semaphore=asyncio.Semaphore(1),
            recall_limit=3,
            arms=(eval_module.PRODUCT_ARM,),
            tracer=telemetry.tracer,
        )
        performance = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    search_e2e = cast(dict[str, object], performance["search_e2e"])
    assert search_e2e["planned_count"] == search_e2e["attempt_count"] == 1
    assert search_e2e["success_count"] == 0
    assert search_e2e["error_count"] == 1
    assert search_e2e["complete"] is False
    assert search_e2e["latency_ms"] is None


@pytest.mark.asyncio
async def test_answer_many_latency_excludes_request_semaphore_wait() -> None:
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            if question == "slow":
                slow_started.set()
                await release_slow.wait()
            return AnswerResult("A")

    pending = asyncio.create_task(
        _answer_many(
            cast(AsyncMemory, Memory([])),
            (
                EvalQuestion("slow", ("slow",), references=("A",)),
                EvalQuestion("fast", ("fast",), references=("A",)),
            ),
            request_concurrency=1,
            recall_limit=1,
        )
    )
    await asyncio.wait_for(slow_started.wait(), timeout=1)
    await asyncio.sleep(0.05)
    release_slow.set()
    slow, fast = await pending

    assert not isinstance(slow, BaseException)
    assert not isinstance(fast, BaseException)
    assert fast.latency_ms < slow.latency_ms / 4


@pytest.mark.asyncio
async def test_answer_many_reports_each_completed_answer_immediately() -> None:
    completed: list[str] = []
    first_completed = asyncio.Event()
    release_slow = asyncio.Event()

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            if question == "slow":
                await release_slow.wait()
            return AnswerResult("A")

    def on_answer(question: EvalQuestion, _outcome: object) -> None:
        completed.append(question.question_id)
        if question.question_id == "fast":
            first_completed.set()

    pending = asyncio.create_task(
        _answer_many(
            cast(AsyncMemory, Memory([])),
            (
                EvalQuestion("fast", ("fast",), references=("A",)),
                EvalQuestion("slow", ("slow",), references=("A",)),
            ),
            request_concurrency=2,
            recall_limit=1,
            on_answer=on_answer,
        )
    )
    await asyncio.wait_for(first_completed.wait(), timeout=1)

    assert completed == ["fast"]

    release_slow.set()
    await pending


@pytest.mark.asyncio
async def test_answer_many_reports_failed_outcome_immediately() -> None:
    completed = 0
    failed = asyncio.Event()
    release_slow = asyncio.Event()

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            if question == "failed":
                raise RuntimeError("answer failed")
            await release_slow.wait()
            return AnswerResult("A")

        async def search(
            self,
            query: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> tuple[SearchHit, ...]:
            del query, limit, reference_at
            raise RuntimeError("fallback failed")

    def on_complete() -> None:
        nonlocal completed
        completed += 1
        failed.set()

    pending = asyncio.create_task(
        _answer_many(
            cast(AsyncMemory, Memory([])),
            (
                EvalQuestion("failed", ("failed",), references=("A",)),
                EvalQuestion("slow", ("slow",), references=("A",)),
            ),
            request_concurrency=2,
            recall_limit=1,
            on_complete=on_complete,
        )
    )
    await asyncio.wait_for(failed.wait(), timeout=1)

    assert completed == 1
    assert not pending.done()

    release_slow.set()
    outcomes = await pending
    assert isinstance(outcomes[0], eval_module._AnswerOutcome)
    assert isinstance(outcomes[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_runner_reports_cached_progress_before_pending_answer_finishes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_reported = asyncio.Event()
    release_slow = asyncio.Event()

    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del limit, reference_at
            await release_slow.wait()
            return AnswerResult("A")

    class Cache:
        def get(self, _task: str, _unit_id: str, question_id: str) -> object | None:
            if question_id != "fast":
                return None
            return SimpleNamespace(
                prediction="A",
                confidence=0.0,
                memory_ids=(),
                evidence=(),
                abstained=False,
                abstention_reason=None,
                ranked_source_ids=None,
            )

        def put(self, *_args: object) -> None:
            pass

    class Context:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, Memory([]))

        async def __aexit__(self, *_error: object) -> None:
            return None

    task = LoadedTask(
        TaskSpec("fixture", "Fixture", "fixture.json", "v1", "owner/repo", "0" * 40),
        tmp_path / "fixture.json",
        "1" * 64,
        (
            EvalUnit(
                "unit",
                (),
                (
                    EvalQuestion("fast", ("fast",), references=("A",)),
                    EvalQuestion("slow", ("slow",), references=("A",)),
                ),
            ),
        ),
    )
    report = eval_module._progress_reporter("running fixture", "samples")

    def on_progress(completed: int, total: int) -> None:
        report(completed, total)
        if completed == 1:
            first_reported.set()

    pending = asyncio.create_task(
        run_loaded_task(
            task,
            run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
            memory_factory=cast(MemoryFactory, lambda _path: Context()),
            batch_size=1,
            unit_concurrency=1,
            request_concurrency=2,
            recall_limit=1,
            response_cache=cast(ResponseCache, Cache()),
            on_progress=on_progress,
        )
    )
    try:
        await asyncio.wait_for(first_reported.wait(), timeout=1)
        assert not pending.done()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "running fixture: 1/2 samples (50%)" in captured.err
    finally:
        release_slow.set()

    await asyncio.wait_for(pending, timeout=1)
    assert "running fixture: 2/2 samples (100%)" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_answer_many_preserves_structured_abstention() -> None:
    class Memory(_FakeMemory):
        async def ask(
            self,
            question: object,
            *,
            limit: int,
            reference_at: datetime | None = None,
        ) -> AnswerResult:
            del question, limit, reference_at
            return AnswerResult(
                "unknown",
                abstained=True,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            )

    outcome = (
        await _answer_many(
            cast(AsyncMemory, Memory([])),
            (EvalQuestion("q", ("question",), references=("A",)),),
            request_concurrency=1,
            recall_limit=1,
        )
    )[0]

    assert isinstance(outcome, eval_module._AnswerOutcome)
    assert outcome.abstained is True
    assert outcome.abstention_reason == "insufficient_evidence"


def _openeqa_dataset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "episode_history": "hm3d-v0/000-hm3d-BFRyYbPCCPE",
                    "category": "object recognition",
                    "question": "What is above the TV?",
                    "answer": "Air conditioning unit",
                },
                {
                    "question_id": "q2",
                    "episode_history": "hm3d-v0/000-hm3d-BFRyYbPCCPE",
                    "category": "object localization",
                    "question": "Where is the mirror?",
                    "answer": "Next to the staircase",
                    "extra_answers": ["by the stairs"],
                },
                {
                    "question_id": "q3",
                    "episode_history": "hm3d-v0/001-hm3d-TPhiubUHKcP",
                    "category": "spatial understanding",
                    "question": "Is the door open?",
                    "answer": "open",
                },
                {
                    "question_id": "q4",
                    "episode_history": "scannet-v0/002-scannet-scene0709_00",
                    "category": "world knowledge",
                    "question": "What room is this?",
                    "answer": "an office",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _openeqa_manifest(task_name: str, *units: str) -> dict[str, object]:
    return {
        "tasks": {
            task_name: {
                "units": {
                    unit: [
                        {
                            "text": f"prepared segment of {unit}",
                            "source_id": f"{unit}-00000",
                            "start_seconds": 0,
                            "end_seconds": 30,
                        }
                    ]
                    for unit in units
                }
            }
        }
    }


def test_openeqa_limit_counts_episodes_and_keeps_all_their_questions(tmp_path: Path) -> None:
    dataset = _openeqa_dataset(tmp_path / "open-eqa-v0.json")
    spec = TASKS["openeqa-hm3d"]

    loaded = load_task(
        spec,
        root=tmp_path,
        dataset_path=dataset,
        media_manifest=_openeqa_manifest(spec.name, "000-hm3d-BFRyYbPCCPE"),
        limit=1,
        verify_digest=False,
    )

    # One episode is one physically isolated store fed by hundreds of frames,
    # so `--limit 1` bounds episodes; half an episode's questions would answer
    # against the same fully ingested scene anyway.
    assert [unit.unit_id for unit in loaded.units] == ["000-hm3d-BFRyYbPCCPE"]
    assert [question.question_id for question in loaded.units[0].questions] == ["q1", "q2"]

    question = loaded.units[0].questions[0]
    assert question.score_kind == "text"
    assert question.references == ("Air conditioning unit",)
    assert question.source_question == "What is above the TV?"
    # The official EM-EQA instruction reaches the model as its own query part,
    # leaving the question text unpolluted for retrieval.
    assert question.content[0] == "What is above the TV?"
    assert "You are an intelligent question answering agent." in str(question.content[1])
    assert question.metadata["category"] == "object recognition"
    assert question.metadata["episode_history"] == "hm3d-v0/000-hm3d-BFRyYbPCCPE"
    # Absent versus present decides the judge prompt, so it must survive the
    # loader as a missing key rather than an empty one.
    assert "extra_answers" not in question.metadata
    assert loaded.units[0].questions[1].metadata["extra_answers"] == ("by the stairs",)


def test_openeqa_tasks_answer_only_their_own_scene_split(tmp_path: Path) -> None:
    dataset = _openeqa_dataset(tmp_path / "open-eqa-v0.json")

    scannet = load_task(
        TASKS["openeqa-scannet"],
        root=tmp_path,
        dataset_path=dataset,
        media_manifest=_openeqa_manifest("openeqa-scannet", "002-scannet-scene0709_00"),
        verify_digest=False,
    )
    assert [unit.unit_id for unit in scannet.units] == ["002-scannet-scene0709_00"]

    hm3d = load_task(
        TASKS["openeqa-hm3d"],
        root=tmp_path,
        dataset_path=dataset,
        media_manifest=_openeqa_manifest(
            "openeqa-hm3d", "000-hm3d-BFRyYbPCCPE", "001-hm3d-TPhiubUHKcP"
        ),
        verify_digest=False,
    )
    assert [unit.unit_id for unit in hm3d.units] == [
        "000-hm3d-BFRyYbPCCPE",
        "001-hm3d-TPhiubUHKcP",
    ]

    # An episode history is a directory of frames, not a file the resolver can
    # match, so a run without prepared media must say so instead of indexing
    # every PNG under a 12-62 GB tree.
    media = tmp_path / "frames"
    media.mkdir()
    with pytest.raises(FileNotFoundError, match="has no media for unit"):
        load_task(
            TASKS["openeqa-hm3d"],
            root=tmp_path,
            dataset_path=dataset,
            media_root=media,
            verify_digest=False,
        )


def test_metric_breakdowns_cover_every_catalog_task(tmp_path: Path) -> None:
    """One family table, not two.

    `_metric_breakdowns` used to resolve families from a list local to
    `eval.py`, which had drifted from the scorer registry: it raised
    `unknown benchmark task` for `longmemeval-s`, `clbench`, `beam-*` and
    `personamem-v3`, so summarising a finished run of any of them crashed.
    """
    for name, spec in TASKS.items():
        task = LoadedTask(
            spec,
            tmp_path / "dataset.json",
            "0" * 64,
            (
                EvalUnit(
                    "unit",
                    (),
                    (
                        EvalQuestion(
                            "q1",
                            ("question",),
                            ("reference",),
                            metadata={"category": "object recognition"},
                        ),
                    ),
                ),
            ),
        )
        breakdowns = eval_module._metric_breakdowns(task, (), cast(Any, _breakdown_arguments()))
        assert isinstance(breakdowns, dict), name

    openeqa = LoadedTask(
        TASKS["openeqa-hm3d"],
        tmp_path / "dataset.json",
        "0" * 64,
        (EvalUnit("unit", (), (EvalQuestion("q1", ("q",), ("r",)),)),),
    )
    sample = replace(
        _egomem_sample(1),
        task="openeqa-hm3d",
        score=1.0,
        metadata={"category": "object recognition"},
    )
    grouped = eval_module._metric_breakdowns(openeqa, (sample,), cast(Any, _breakdown_arguments()))
    assert "category" in grouped
    assert "object recognition" in cast(dict[str, object], grouped["category"])


# The metadata fields each family groups its per-question scores by, written
# from the adapters in `eval_adapters.py` -- which is what a real run puts in
# `SampleResult.metadata` -- rather than read back off `_BREAKDOWN_FIELDS`, so
# that a family losing its breakdown turns these tests red instead of silently
# agreeing with the regression. That is how `longmemeval`, `clbench`, `beam`
# and `personamem-v3` went a release reporting no breakdown at all. An empty
# tuple means the family is expected to have no entry in the product table.
_EXPECTED_BREAKDOWN_FIELDS: dict[str, tuple[str, ...]] = {
    "locomo-refined": ("category",),
    "m3-bench": ("question_types",),
    "video-mme": ("duration", "domain", "task_type"),
    "video-mme-v2": ("group_type", "level", "second_head", "third_head"),
    "egolifeqa": ("day", "question_type"),
    # No breakdown: the public release ships no answer key, so every sample
    # scores `None` and there is nothing to group.
    "egomemreason": (),
    "egotempo": ("question_type",),
    "memlens": ("question_type", "question_subtype"),
    "mm-lifelong": ("question_type",),
    "supermemory-vqa": ("skill",),
    "atm-bench": ("qtype",),
    "mem-gallery": ("point",),
    "longmemeval": ("question_type",),
    "clbench": ("context_category", "sub_category"),
    "beam": ("category", "difficulty"),
    "personamem-v3": ("task_family", "task_type"),
    "openeqa": ("category",),
}


def _breakdown_task(family: str, tmp_path: Path) -> LoadedTask:
    spec = next((spec for name, spec in TASKS.items() if task_family(name) == family), None)
    assert spec is not None, f"no catalog task belongs to {family}"
    return LoadedTask(
        spec,
        tmp_path / "dataset.json",
        "0" * 64,
        (EvalUnit("unit", (), (EvalQuestion("q1", ("question",), ("reference",)),)),),
    )


def _breakdown_sample(task: str, metadata: dict[str, object]) -> SampleResult:
    return replace(_egomem_sample(1), task=task, score=1.0, metadata=metadata)


@pytest.mark.parametrize("family", sorted(_EXPECTED_BREAKDOWN_FIELDS))
def test_every_family_breaks_its_scores_down_by_its_own_metadata(
    family: str, tmp_path: Path
) -> None:
    """Every field the product declares is one an adapter really emits."""
    task = _breakdown_task(family, tmp_path)
    fields_expected = _EXPECTED_BREAKDOWN_FIELDS[family]
    sample = _breakdown_sample(task.spec.name, {name: f"{name}-value" for name in fields_expected})

    breakdowns = eval_module._metric_breakdowns(task, (sample,), cast(Any, _breakdown_arguments()))

    assert set(breakdowns) == set(fields_expected), family
    for name in fields_expected:
        cell = cast(dict[str, object], breakdowns[name])
        assert f"{name}-value" in cell, f"{family}.{name}"


def test_the_breakdown_table_is_exactly_the_fields_this_suite_pins() -> None:
    # The behavioural test above only feeds the fields it expects, so a field
    # name added to the product table that no adapter emits -- a rename, or a
    # `question_type`/`question_types` slip -- would group nothing and stay
    # invisible there. Compare the table itself so an addition has to be
    # reviewed too, and so the families expected to have no entry really have
    # none rather than an entry that happens to group nothing.
    grouped = {family: fields for family, fields in _EXPECTED_BREAKDOWN_FIELDS.items() if fields}

    assert grouped == eval_module._BREAKDOWN_FIELDS


def test_the_breakdown_table_covers_every_catalog_family() -> None:
    # A benchmark added to the catalog without a decision recorded here reports
    # no breakdown at all, which is invisible in a results document.
    assert {task_family(name) for name in TASKS} == set(_EXPECTED_BREAKDOWN_FIELDS)


def test_an_unlabelled_beam_difficulty_is_left_out_of_the_breakdown(tmp_path: Path) -> None:
    # BEAM publishes `difficulty` on some rows only. Grouping the rest under a
    # literal "None" bucket would read as a difficulty tier in the report.
    task = _breakdown_task("beam", tmp_path)
    sample = _breakdown_sample(task.spec.name, {"category": "temporal", "difficulty": None})

    breakdowns = eval_module._metric_breakdowns(task, (sample,), cast(Any, _breakdown_arguments()))

    assert set(breakdowns) == {"category"}


def _breakdown_arguments() -> SimpleNamespace:
    return SimpleNamespace(seed=1234, bootstrap_samples=8, recall_limit=20, blind=False)


def test_a_task_worded_refusal_counts_as_an_abstention() -> None:
    # `AnswerResult.abstained` recognises the product's own refusal sentence, which it cannot do
    # for a wording the task substituted. MEMLENS mandates "Insufficient information", so a real
    # dev run reported 0 abstentions where 16 of 59 answers were refusals.
    question = EvalQuestion(
        "q1",
        ("Where did she go?",),
        ("the market",),
        refusal="Insufficient information",
    )

    assert eval_module._declined("Insufficient information", question)
    assert eval_module._declined("Insufficient information.", question)
    assert not eval_module._declined("She went to the market.", question)
    # A task that mandates nothing must not have refusals invented for it.
    assert not eval_module._declined("Insufficient information", replace(question, refusal=None))


def test_memlens_questions_declare_the_refusal_their_own_prompt_mandates(tmp_path: Path) -> None:
    # Loading the real task, not the helper in isolation: a declaration the loader never attaches
    # returns the reported refusal rate to zero while the model keeps refusing, and a test of the
    # helper alone stays green through exactly that.
    from mindbridge.benchmarks.prompts import MEMLENS_QUERY_PROMPT

    dataset = tmp_path / "memlens.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "preference",
                    "question": "What is my favorite color?",
                    "answer": "Blue.",
                    "question_date": "2025/01/15 (Wed) 10:00",
                    "haystack_dates": ["2025/01/14 (Tue) 09:00"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "My favorite color is blue."}]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    auxiliary = tmp_path / "memlens" / "agent_subset_195.json"
    auxiliary.parent.mkdir()
    auxiliary.write_text(json.dumps({"n_questions": 1, "question_ids": ["q1"]}), encoding="utf-8")

    loaded = load_task(
        TASKS["memlens-32k"], root=tmp_path, dataset_path=dataset, verify_digest=False
    )
    question = loaded.units[0].questions[0]

    assert MEMLENS_QUERY_PROMPT.refusal is not None
    # Tied to the prompt text, so changing the mandated wording without the declaration fails.
    assert f'"{MEMLENS_QUERY_PROMPT.refusal}"' in MEMLENS_QUERY_PROMPT.text
    assert question.refusal == MEMLENS_QUERY_PROMPT.refusal
    assert eval_module._declined(MEMLENS_QUERY_PROMPT.refusal, question)


def test_mem_gallery_refusal_questions_declare_the_wording_their_constraint_mandates(
    tmp_path: Path,
) -> None:
    # Loading the real task, for the same reason the MemLens case does: `AR` is the only one of
    # the nine task types whose constraint overrides the refusal wording, so a declaration the
    # loader never attaches reports zero refusals for it while the model keeps refusing.
    from mindbridge.benchmarks.prompts import MEM_GALLERY_REFUSAL_PROMPT

    dialog = tmp_path / "dialog"
    dialog.mkdir()
    (dialog / "topic.json").write_text(
        json.dumps(
            {
                "character_profile": {
                    "name": "Mira",
                    "persona_summary": "A gardener.",
                    "conversation_style": "warm",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "D1",
                        "date": "2024-05-02",
                        "dialogues": [
                            {"round": "D1:1", "user": "I planted basil.", "assistant": "Nice."}
                        ],
                    }
                ],
                "human-annotated QAs": [
                    {
                        "point": "AR",
                        "question": "What did I plant in the greenhouse?",
                        "answer": "Not mentioned.",
                        "session_id": ["D1"],
                    },
                    {
                        "point": "FR",
                        "question": "What did I plant?",
                        "answer": "Basil.",
                        "session_id": ["D1"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_task(
        TASKS["mem-gallery"], root=tmp_path, dataset_path=dialog, verify_digest=False
    )
    by_point = {
        question.metadata["point"]: question for unit in loaded.units for question in unit.questions
    }

    assert MEM_GALLERY_REFUSAL_PROMPT.refusal is not None
    # Tied to the constraint text, so changing the mandated wording without the declaration fails.
    assert MEM_GALLERY_REFUSAL_PROMPT.refusal in MEM_GALLERY_REFUSAL_PROMPT.text
    assert by_point["AR"].refusal == MEM_GALLERY_REFUSAL_PROMPT.refusal
    assert eval_module._declined(MEM_GALLERY_REFUSAL_PROMPT.refusal, by_point["AR"])
    # The other eight types are asked without that constraint, so they must not claim it.
    assert by_point["FR"].refusal is None
    assert not eval_module._declined(MEM_GALLERY_REFUSAL_PROMPT.refusal, by_point["FR"])


def test_eval_config_reads_yaml_and_splits_the_harness_section(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        """
embedding:
  provider: openai
  model: local-embedder
  dimension: 64
generation:
  provider: openai
  model: configured-model
  api_key: config-generation-key
  min_video_seconds: 2.5
speech:
  provider: funasr
  device: cpu
benchmark:
  judge:
    model: config-judge
    base_url: https://judge.example/v1
    api_key: config-judge-key
    timeout_seconds: 90
  download:
    hf_home: /corpus/hf
    hf_endpoint: https://mirror.example
    youtube_sleep_seconds: 7
""",
        encoding="utf-8",
    )

    config, overrides = eval_module._load_memory_config(path)

    assert config is not None
    assert isinstance(config.embedding, OpenAIEmbeddingConfig)
    assert config.embedding.model == "local-embedder"
    assert config.generation is not None
    assert config.generation.model == "configured-model"
    # The credential lives with the endpoint it authenticates, in the product block.
    assert config.generation.api_key is not None
    assert config.generation.api_key.get_secret_value() == "config-generation-key"
    assert config.generation.min_video_seconds == 2.5
    # The harness section must not reach the product schema, which forbids unknown fields.
    assert overrides.judge.model == "config-judge"
    assert overrides.judge.timeout_seconds == 90.0
    assert overrides.download.hf_endpoint == "https://mirror.example"
    assert overrides.download.youtube_sleep_seconds == 7.0


def test_eval_config_still_reads_the_json_it_used_to_take(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "embedding": {"provider": "openai"},
                "generation": {"provider": "openai", "model": "json-model"},
            }
        ),
        encoding="utf-8",
    )

    config, overrides = eval_module._load_memory_config(path)

    assert config is not None
    assert config.generation is not None
    assert config.generation.model == "json-model"
    assert overrides == HarnessOverrides()


@pytest.mark.parametrize(
    "document",
    (
        "generation:\n  provider: openai\n",
        "embedding:\n  provider: openai\n",
        "benchmark:\n  judge:\n    model: only-a-judge\n",
        "{}\n",
        "",
    ),
)
def test_eval_config_defaults_absent_sections_instead_of_failing(
    document: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(document, encoding="utf-8")

    config, _ = eval_module._load_memory_config(path)

    assert config is not None
    assert config.generation is not None
    assert config.embedding.provider in {"openai", "jina-omni"}


def test_eval_config_rejects_an_unknown_harness_key(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("benchmark:\n  judge:\n    modle: typo\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        eval_module._load_memory_config(path)


def test_eval_config_reports_the_yaml_error_position(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("generation:\n  provider: openai\n :\n  - [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML at line"):
        eval_module._load_memory_config(path)


@pytest.mark.parametrize("spelling", ("all", "ALL", " all ", "-1"))
def test_eval_limit_accepts_the_spelling_the_help_advertises(spelling: str) -> None:
    assert eval_module._limit_value(spelling) == -1


@pytest.mark.parametrize("value", ("0", "-2", "nope", "nan"))
def test_eval_limit_still_rejects_unusable_values(value: str) -> None:
    with pytest.raises(ArgumentTypeError):
        eval_module._limit_value(value)


def test_default_benchmarks_root_reaches_the_main_checkout_from_a_worktree(
    tmp_path: Path,
) -> None:
    main = tmp_path / "checkout"
    (main / ".git" / "worktrees" / "feature").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / ".git").write_text(
        f"gitdir: {main / '.git' / 'worktrees' / 'feature'}\n", encoding="utf-8"
    )

    assert default_benchmarks_root(linked) == main / ".benchmarks"
    assert default_benchmarks_root(main) == main / ".benchmarks"


def test_default_benchmarks_root_falls_back_outside_a_repository(tmp_path: Path) -> None:
    assert default_benchmarks_root(tmp_path) == Path(".benchmarks")


def test_download_settings_prefer_the_flag_then_the_file_then_the_environment(
    tmp_path: Path,
) -> None:
    declared = DownloadOverrides(
        benchmarks_root=tmp_path / "from-file",
        hf_home=tmp_path / "file-hf",
        hf_endpoint="https://file.example",
        youtube_sleep_seconds=3,
    )
    environ = {
        "HF_HOME": str(tmp_path / "env-hf"),
        "HF_ENDPOINT": "https://env.example",
        "MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS": "99",
    }

    from_file = DownloadSettings.resolve(declared, environ=environ)
    assert from_file.benchmarks_root == tmp_path / "from-file"
    assert from_file.data_root == tmp_path / "from-file" / "data"
    assert from_file.hf_home == tmp_path / "file-hf"
    assert from_file.hf_endpoint == "https://file.example"
    assert from_file.youtube_sleep_seconds == 3.0

    from_flag = DownloadSettings.resolve(
        declared, benchmarks_root=tmp_path / "from-flag", environ=environ
    )
    assert from_flag.benchmarks_root == tmp_path / "from-flag"

    from_environment = DownloadSettings.resolve(environ=environ)
    assert from_environment.hf_home == tmp_path / "env-hf"
    assert from_environment.hf_endpoint == "https://env.example"
    assert from_environment.youtube_sleep_seconds == 99.0

    from_default = DownloadSettings.resolve(environ={})
    assert from_default.hf_home is None
    assert from_default.hf_endpoint == DEFAULT_HF_ENDPOINT
    assert from_default.youtube_sleep_seconds == 30.0


def test_download_settings_publish_the_addresses_acquisition_actually_reads(
    tmp_path: Path,
) -> None:
    published: dict[str, str] = {}

    DownloadSettings.resolve(
        DownloadOverrides(hf_home=tmp_path / "hf", hf_endpoint="https://mirror.example"),
        environ={},
    ).apply_environment(published)

    assert published["HF_HOME"] == str(tmp_path / "hf")
    assert published["HF_ENDPOINT"] == "https://mirror.example"
    assert float(published["MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS"]) == 30.0


def test_eval_generation_credential_prefers_the_file_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINDBRIDGE_GENERATION_API_KEY", "environment-key")
    declared = tmp_path / "declared.yaml"
    declared.write_text(
        "generation:\n  provider: openai\n  api_key: file-key\n",
        encoding="utf-8",
    )
    silent = tmp_path / "silent.yaml"
    silent.write_text("generation:\n  provider: openai\n", encoding="utf-8")

    config, overrides = _load_memory_config(declared)
    assert config is not None and config.generation is not None
    resolved = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    assert resolved.generation_api_key == "file-key"

    config, overrides = _load_memory_config(silent)
    inherited = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    assert inherited.generation_api_key == "environment-key"


def test_eval_video_floor_has_exactly_one_home(tmp_path: Path) -> None:
    """The floor is a generation-endpoint setting, so it lives in the product block only.

    It reaches `ModelConfig` from there, which is what puts it in the result artifact and in the
    cache namespace; naming it under `benchmark` must be rejected rather than quietly accepted in
    a second place whose effective value nobody can read off the file.
    """
    path = tmp_path / "eval.yaml"
    path.write_text(
        "generation:\n  provider: openai\n  min_video_seconds: 4\n",
        encoding="utf-8",
    )

    config, overrides = _load_memory_config(path)

    assert config is not None and config.generation is not None
    assert config.generation.min_video_seconds == 4.0
    resolved = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    assert resolved.generation_min_video_seconds == 4.0

    path.write_text(
        "benchmark:\n  generation:\n    min_video_seconds: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        _load_memory_config(path)


def test_eval_results_never_serialize_a_configured_credential(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        """
embedding:
  provider: openai
  api_key: embedding-secret
generation:
  provider: openai
  api_key: generation-secret
""",
        encoding="utf-8",
    )

    config, _ = _load_memory_config(path)
    assert config is not None

    payload = eval_module._memory_config_payload(config)
    serialized = json.dumps(payload)
    assert "embedding-secret" not in serialized
    assert "generation-secret" not in serialized
    assert repr(config.generation).find("generation-secret") == -1


def test_eval_judge_prefers_the_file_then_the_environment_then_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDBRIDGE_JUDGE_MODEL", "environment-judge")
    monkeypatch.setenv("MINDBRIDGE_JUDGE_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("MINDBRIDGE_JUDGE_API_KEY", "environment-judge-key")
    monkeypatch.setenv("MINDBRIDGE_JUDGE_TIMEOUT_SECONDS", "11")
    generation = ModelConfig(
        generation_api_key="generation-key",
        generation_base_url="https://generation.example/v1",
        generation_model="generation-model",
        timeout_seconds=33.0,
    )
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            judge_concurrency=4,
            judge_model_args="",
        ),
    )

    from_file = eval_module._judge_config(
        generation,
        arguments,
        overrides=HarnessOverrides.model_validate(
            {
                "judge": {
                    "model": "file-judge",
                    "base_url": "https://file.example/v1",
                    "api_key": "file-judge-key",
                    "timeout_seconds": 90,
                }
            }
        ),
    )
    assert from_file.model == "file-judge"
    assert from_file.base_url == "https://file.example/v1"
    assert from_file.api_key == "file-judge-key"
    assert from_file.timeout_seconds == 90.0

    from_environment = eval_module._judge_config(generation, arguments)
    assert from_environment.model == "environment-judge"
    assert from_environment.api_key == "environment-judge-key"
    assert from_environment.timeout_seconds == 11.0

    for name in (
        "MINDBRIDGE_JUDGE_MODEL",
        "MINDBRIDGE_JUDGE_BASE_URL",
        "MINDBRIDGE_JUDGE_API_KEY",
        "MINDBRIDGE_JUDGE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name)
    from_generation = eval_module._judge_config(generation, arguments)
    assert from_generation.model == "generation-model"
    assert from_generation.base_url == "https://generation.example/v1"
    assert from_generation.api_key == "generation-key"
    assert from_generation.timeout_seconds == 33.0


def test_eval_memory_config_survives_an_empty_gen_kwargs(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("generation:\n  provider: openai\n", encoding="utf-8")
    config, _ = eval_module._load_memory_config(path)
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(
            gen_kwargs="",
            seed=0,
            device=None,
        ),
    )

    resolved = eval_module._evaluation_memory_config(config, ModelConfig(), arguments)

    assert resolved is not None
    assert resolved.generation is not None
    assert resolved.generation.extra_body is None


def test_committed_example_configuration_still_loads() -> None:
    """The annotated example is documentation that executes, so it must stay loadable.

    Prose can drift silently; a configuration file cannot be allowed to. This fails the moment a
    schema change makes the published template invalid.
    """
    example = Path(__file__).parents[3] / "docs" / "examples" / "eval.example.yaml"
    assert example.exists()

    config, overrides = _load_memory_config(example)

    assert config is not None
    assert config.generation is not None
    assert config.embedding.provider == "openai"
    assert config.speech is not None
    # Formation is commented out in the template because enabling it costs an LLM call per write.
    assert config.formation is None
    assert overrides.judge.model is not None
    assert overrides.download.hf_endpoint == "https://huggingface.co"

    model = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    judge = eval_module._judge_config(
        model,
        cast(
            eval_module._Arguments,
            SimpleNamespace(judge_concurrency=4, judge_model_args=""),
        ),
        overrides=overrides,
    )
    assert judge.model == overrides.judge.model
    # The template must also drive a whole run on its own, with no flag but `--config`.
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(example)])
    arguments = eval_module._arguments(parser, parsed, overrides=overrides)
    assert arguments.tasks
    assert arguments.limit == -1
    assert arguments.recall_limit == 20
    # Turning thinking off is a generation setting, so the template carries it in `extra_body`
    # rather than needing `--gen-kwargs`.
    assert config.generation.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_committed_example_configuration_carries_no_usable_credential() -> None:
    """A committed template must not ship anything that could be a real key."""
    example = Path(__file__).parents[3] / "docs" / "examples" / "eval.example.yaml"

    config, overrides = _load_memory_config(example)

    assert config is not None and config.generation is not None
    assert isinstance(config.embedding, OpenAIEmbeddingConfig)
    secrets = [
        config.embedding.api_key,
        config.generation.api_key,
        overrides.judge.api_key,
    ]
    for secret in secrets:
        assert secret is not None
        value = secret if isinstance(secret, str) else secret.get_secret_value()
        # Placeholders only: no long opaque token, and nothing that looks like a vendor prefix.
        assert len(value) < 40
        assert not re.fullmatch(r"[A-Za-z0-9_-]{40,}", value)
        assert "replace" in value or "ignores" in value


def test_eval_model_config_keeps_env_model_when_file_omits_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A config file's `generation` block must not silently reset fields it never mentions.

    `model` and `modalities` have non-None schema defaults, unlike `base_url`/`timeout`, so a
    plain `x or fallback` merge can never detect "the file left this unset" -- it always sees the
    schema default and prefers it. This regressed the "config file beats environment, but only
    for what it actually declares" contract for exactly these two fields.
    """
    monkeypatch.setenv("MINDBRIDGE_GENERATION_MODEL", "environment-model")
    monkeypatch.setenv("MINDBRIDGE_GENERATION_MODALITIES", "text,image,video")
    path = tmp_path / "eval.yaml"
    path.write_text(
        "generation:\n  provider: openai\n  base_url: https://example.com/v1\n",
        encoding="utf-8",
    )

    config, overrides = _load_memory_config(path)
    resolved = _model_config("mindbridge", "", memory_config=config, overrides=overrides)

    assert resolved.generation_model == "environment-model"
    assert sorted(m.value for m in resolved.generation_capabilities) == [
        "image",
        "text",
        "video",
    ]


def test_eval_model_config_still_prefers_a_declared_model_and_modalities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MINDBRIDGE_GENERATION_MODEL", "environment-model")
    monkeypatch.setenv("MINDBRIDGE_GENERATION_MODALITIES", "text,image,video")
    path = tmp_path / "eval.yaml"
    path.write_text(
        "generation:\n  provider: openai\n  model: file-model\n  modalities: [text]\n",
        encoding="utf-8",
    )

    config, overrides = _load_memory_config(path)
    resolved = _model_config("mindbridge", "", memory_config=config, overrides=overrides)

    assert resolved.generation_model == "file-model"
    assert sorted(m.value for m in resolved.generation_capabilities) == ["text"]


def test_eval_resolved_generation_keeps_an_env_only_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The credential must survive into the config actually used to build the run's backend.

    `_evaluation_memory_config` rebuilds `generation` via `model_copy`, which only overwrites the
    keys named in its update dict -- an env-only credential merged into `ModelConfig` but never
    named there is silently dropped, and the real client falls back to whatever `OPENAI_API_KEY`
    happens to be (or to no credential at all).
    """
    monkeypatch.setenv("MINDBRIDGE_GENERATION_API_KEY", "environment-key")
    path = tmp_path / "eval.yaml"
    path.write_text("generation:\n  provider: openai\n", encoding="utf-8")

    config, overrides = _load_memory_config(path)
    model = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(gen_kwargs="", seed=0, device=None),
    )

    resolved = eval_module._evaluation_memory_config(config, model, arguments)

    assert resolved is not None and resolved.generation is not None
    assert resolved.generation.api_key is not None
    assert resolved.generation.api_key.get_secret_value() == "environment-key"


def test_eval_resolved_generation_still_prefers_a_declared_credential(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("generation:\n  provider: openai\n  api_key: file-key\n", encoding="utf-8")

    config, overrides = _load_memory_config(path)
    model = _model_config("mindbridge", "", memory_config=config, overrides=overrides)
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(gen_kwargs="", seed=0, device=None),
    )

    resolved = eval_module._evaluation_memory_config(config, model, arguments)

    assert resolved is not None and resolved.generation is not None
    assert resolved.generation.api_key is not None
    assert resolved.generation.api_key.get_secret_value() == "file-key"


@pytest.mark.parametrize("field", ("temperature", "seed"))
def test_eval_config_rejects_sampling_controls_the_harness_pins(field: str, tmp_path: Path) -> None:
    """Declaring a pinned control must fail loudly rather than be accepted and discarded.

    The harness always sends temperature 0 and the seed `--seed` names, so a file that sets
    either would otherwise describe a run that never happens.
    """
    path = tmp_path / "eval.yaml"
    path.write_text(
        f"generation:\n  provider: openai\n  {field}: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"cannot set {field}"):
        _load_memory_config(path)


def test_eval_run_section_supplies_every_tunable_the_flags_carry(tmp_path: Path) -> None:
    """A file alone must be able to describe a whole sweep, with no flags but `--config`."""
    path = tmp_path / "eval.yaml"
    path.write_text(
        """
benchmark:
  run:
    tasks: clbench
    limit: all
    offset: 3
    unit_concurrency: 6
    request_concurrency: 16
    judge_concurrency: 32
    recall_limit: 50
    seed: 7
    bootstrap_samples: 500
    repeat_index: 3
    batch_size: "8"
    max_batch_size: 16
    device: cuda:1
    device_lock: false
    log_samples: true
    predict_only: true
    allow_unverified_data: true
    download: false
    overwrite: true
    verbosity: WARNING
    regression_threshold: 0.25
""",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.tasks == ("clbench",)
    assert arguments.limit == -1
    assert arguments.offset == 3
    assert arguments.unit_concurrency == 6
    assert arguments.request_concurrency == 16
    assert arguments.judge_concurrency == 32
    assert arguments.recall_limit == 50
    assert arguments.seed == 7
    assert arguments.seeds == (7, 7, 7, 7)
    assert arguments.bootstrap_samples == 500
    assert arguments.repeat_index == 3
    assert arguments.batch_size == "8"
    assert arguments.max_batch_size == 16
    assert arguments.device == "cuda:1"
    assert arguments.device_lock is False
    assert arguments.log_samples is True
    assert arguments.predict_only is True
    assert arguments.allow_unverified_data is True
    assert arguments.download is False
    assert arguments.overwrite is True
    assert arguments.regression_threshold == 0.25
    # `--gen-kwargs` normalisation must pick up the configured seed, not the default.
    assert "seed=7" in arguments.gen_kwargs


def test_performance_budget_flags_override_named_config_values(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        """
benchmark:
  performance_budgets:
    answer_e2e_ttft_p95: 0.10
  run:
    tasks: clbench
    compare: baseline
    repeat_index: 2
""",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(
        [
            "--config",
            str(path),
            "--repeat-index",
            "4",
            "--performance-budget",
            "answer_e2e_ttft_p95=0.20",
            "--performance-budget",
            "retrieval_e2e_latency_p95=0.15",
        ]
    )

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.repeat_index == 4
    assert arguments.performance_budgets == {
        "answer_e2e_ttft_p95": 0.2,
        "retrieval_e2e_latency_p95": 0.15,
    }
    assert arguments.compare == (Path.cwd() / "baseline").resolve()


def test_performance_budget_requires_a_comparison_result() -> None:
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(
        ["--tasks", "clbench", "--performance-budget", "answer_e2e_latency_p95=0.1"]
    )

    with pytest.raises(SystemExit):
        eval_module._arguments(parser, parsed)


class _StopResolving(Exception):
    """Cut `_BackendPool.__init__` short once the document it would resolve has been captured."""


def test_backend_pool_hands_the_configured_video_floor_to_the_resolved_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor must reach the document the backends are built from, not just the artifact.

    `--config` always resolves a memory configuration, so the configured branch is the only one a
    file-driven run takes. A floor recorded on `ModelConfig` alone would be reported in
    `results.jsonl` while short videos still went to the endpoint whole.
    """
    configured: list[MindBridgeConfig] = []

    def resolve(config: MindBridgeConfig) -> object:
        configured.append(config)
        raise _StopResolving

    monkeypatch.setattr(eval_module, "resolve_memory_config", resolve)
    memory_config = MindBridgeConfig.model_validate(
        {"embedding": {"provider": "openai"}, "generation": {"provider": "openai"}}
    )

    with pytest.raises(_StopResolving):
        eval_module._BackendPool(
            ModelConfig(generation_min_video_seconds=2.5),
            device=None,
            batch_size=1,
            needs_speech=False,
            seed=0,
            memory_config=memory_config,
        )

    assert configured[0].generation is not None
    assert configured[0].generation.min_video_seconds == 2.5


def test_backend_pool_leaves_an_unset_video_floor_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent harness floor must not overwrite one the memory document declares itself."""
    configured: list[MindBridgeConfig] = []

    def resolve(config: MindBridgeConfig) -> object:
        configured.append(config)
        raise _StopResolving

    monkeypatch.setattr(eval_module, "resolve_memory_config", resolve)
    memory_config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "min_video_seconds": 4.0},
        }
    )

    with pytest.raises(_StopResolving):
        eval_module._BackendPool(
            ModelConfig(),
            device=None,
            batch_size=1,
            needs_speech=False,
            seed=0,
            memory_config=memory_config,
        )

    assert configured[0].generation is not None
    assert configured[0].generation.min_video_seconds == 4.0


def test_eval_run_section_selects_the_baseline_arms(tmp_path: Path) -> None:
    """A baseline sweep is what a file describes; the flag alone cannot be reused across runs."""
    path = tmp_path / "eval.yaml"
    path.write_text(
        "benchmark:\n  run:\n    tasks: clbench\n    arms: mindbridge,full-context\n"
        "    full_context_chars: 4096\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.arms == ("mindbridge", "full-context")
    assert arguments.full_context_chars == 4096


def test_eval_arm_flags_still_beat_the_run_section(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "benchmark:\n  run:\n    tasks: clbench\n    arms: full-context\n"
        "    full_context_chars: 4096\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(
        ["--config", str(path), "--arms", "random", "--full-context-chars", "512"]
    )

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.arms == ("random",)
    assert arguments.full_context_chars == 512


def test_eval_blind_rejects_a_configured_arm_selection(tmp_path: Path) -> None:
    """`--blind` labels the whole run as the control, so a file naming another arm is a conflict."""
    path = tmp_path / "eval.yaml"
    path.write_text(
        "benchmark:\n  run:\n    tasks: clbench\n    arms: mindbridge,full-context\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path), "--blind"])

    with pytest.raises(SystemExit):
        eval_module._arguments(parser, parsed, overrides=overrides)


def test_eval_flags_still_beat_the_run_section(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "benchmark:\n  run:\n    tasks: clbench\n    unit_concurrency: 6\n    seed: 7\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(
        ["--config", str(path), "--tasks", "beam-100k", "--unit-concurrency", "2", "--seed", "99"]
    )

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.tasks == ("beam-100k",)
    assert arguments.unit_concurrency == 2
    assert arguments.seeds == (99, 99, 99, 99)


def test_eval_defaults_survive_an_empty_run_section(tmp_path: Path) -> None:
    """Every constant moved out of argparse must still be the value an unset flag produces."""
    path = tmp_path / "eval.yaml"
    path.write_text("generation:\n  provider: openai\n", encoding="utf-8")
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path), "--tasks", "clbench"])

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.limit is None
    assert arguments.offset == 0
    assert arguments.batch_size == "auto"
    assert arguments.max_batch_size == 64
    assert arguments.unit_concurrency == 1
    assert arguments.request_concurrency == 4
    assert arguments.judge_concurrency == 8
    assert arguments.recall_limit == 20
    assert arguments.seeds == (0, 1234, 1234, 1234)
    assert arguments.bootstrap_samples == eval_module.DEFAULT_BOOTSTRAP_SAMPLES
    assert arguments.device is None
    assert arguments.device_lock is True
    assert arguments.download is True
    assert arguments.overwrite is False
    assert arguments.log_samples is False
    assert arguments.predict_only is False
    assert arguments.allow_unverified_data is False
    assert arguments.quiet is False
    assert arguments.dataset_overrides == {}
    assert arguments.media_overrides == {}


def test_eval_run_section_carries_per_task_path_overrides(tmp_path: Path) -> None:
    dataset = tmp_path / "clbench.json"
    dataset.write_text("{}", encoding="utf-8")
    path = tmp_path / "eval.yaml"
    path.write_text(
        f"benchmark:\n  run:\n    tasks: clbench\n    task_data:\n      clbench: {dataset}\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.dataset_overrides == {"clbench": dataset}


def test_eval_run_section_run_id_is_validated_like_the_flag(tmp_path: Path) -> None:
    """A configured run ID must pass the same charset check the flag enforces.

    The run ID becomes a path segment under `<benchmarks-root>/results/`, which is why the flag
    restricts it. A file that skipped the check could send every artifact outside the corpus tree.
    """
    path = tmp_path / "eval.yaml"
    path.write_text(
        'benchmark:\n  run:\n    tasks: clbench\n    run_id: "../../../escaped"\n',
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    with pytest.raises(SystemExit):
        eval_module._arguments(parser, parsed, overrides=overrides)


def test_eval_run_section_accepts_a_well_formed_run_id(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "benchmark:\n  run:\n    tasks: clbench\n    run_id: sweep-2026.09\n",
        encoding="utf-8",
    )
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    arguments = eval_module._arguments(parser, parsed, overrides=overrides)

    assert arguments.run_id == "sweep-2026.09"
    assert arguments.output_path.name == "sweep-2026.09"
    assert arguments.output_path.parent.name == "results"


@pytest.mark.parametrize(
    "body",
    (
        "benchmark:\n  run:\n    tasks: clbench\n    seed: [1, 2]\n",
        'benchmark:\n  run:\n    tasks: clbench\n    limit: "lots"\n',
    ),
)
def test_eval_run_section_reports_bad_values_as_usage_errors(body: str, tmp_path: Path) -> None:
    """`--seed` and `--limit` validate with `ArgumentTypeError`, which is not a `ValueError`.

    Without catching it, a configured value that fails those checks left `_arguments` as an
    unhandled exception instead of the usage message a bad flag produces.
    """
    path = tmp_path / "eval.yaml"
    path.write_text(body, encoding="utf-8")
    _, overrides = _load_memory_config(path)
    parser = eval_module._build_parser("eval")
    parsed = parser.parse_args(["--config", str(path)])

    with pytest.raises(SystemExit):
        eval_module._arguments(parser, parsed, overrides=overrides)
