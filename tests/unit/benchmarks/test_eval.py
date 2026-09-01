"""Core checks for deterministic task selection, causal ingest, and statistics."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from argparse import ArgumentTypeError
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import mindbridge.benchmarks.eval as eval_module
from mindbridge import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    IndexUnavailableError,
    MemoryType,
    Modality,
    ModelError,
    SearchHit,
)
from mindbridge.benchmarks.eval import (
    MemoryFactory,
    SampleResult,
    _answer_many,
    _BorrowedSpeechBackend,
    _cache_namespace,
    _generation_kwargs,
    _ingest,
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
from mindbridge.benchmarks.eval_statistics import (
    ScoredValue,
    paired_comparison,
    parse_choice,
    summarize,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.mem_gallery import MemGalleryRound, MemGallerySession
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.benchmarks.official_scorers import scorer_protocol
from mindbridge.benchmarks.task_catalog import TASKS, TaskSpec, expand
from mindbridge.benchmarks.video_mme_v2 import score_group_answers
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
        "EgoLifeQA",
        "EgoMemReason",
        "EgoTempo",
        "LoCoMo-Refined",
        "M3-Bench",
        "MM-Lifelong",
        "Mem-Gallery",
        "MemLens",
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

    assert json.loads(capsys.readouterr().out) == {
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
                    "performance": {
                        "duration_seconds": {"total": 2.5, "average": 1.25},
                        "token_usage": {"total_tokens": 30, "average_tokens": 15.0},
                    },
                }
            ]
        }
    )

    assert "total s" in output
    assert "avg ms" in output
    assert "30" in output
    assert "15.0" in output


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
    assert eval_module.EVAL_RUNNER_VERSION == "mindbridge_eval_official_v9"
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(device=None, seed=7, gen_kwargs="{}", recall_limit=8),
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
async def test_answer_failure_preserves_independent_retrieval_diagnostics(tmp_path: Path) -> None:
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)

    class Memory:
        async def ask(self, _question: object, **_kwargs: object) -> AnswerResult:
            raise RuntimeError("generation unavailable")

        async def search(self, _query: object, **_kwargs: object) -> tuple[SearchHit, ...]:
            return (
                SearchHit(
                    id="memory-1",
                    content="supporting memory",
                    score=0.8,
                    created_at=now,
                    metadata={"source_id": "source-1"},
                ),
            )

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

    samples = await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "fixture", "run"),
        memory_factory=cast(MemoryFactory, lambda _path: Context()),
        batch_size=4,
        unit_concurrency=1,
        request_concurrency=1,
        recall_limit=1,
    )

    assert calls == 1
    assert samples[0].error_code == "index_unavailable"


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
    )

    assert events == [
        "add:[source_id: early]\nearly",
        "ask:first:5",
        "add:[source_id: later]\nlater",
        "ask:second:5",
    ]
    assert [sample.score for sample in samples] == [1.0, 1.0]


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
