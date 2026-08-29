"""Core checks for deterministic task selection, causal ingest, and statistics."""

from __future__ import annotations

import json
from argparse import ArgumentTypeError
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import mindbridge.benchmarks.eval as eval_module
from mindbridge import AnswerResult, AssetRef, AsyncMemory, MemoryType, Modality
from mindbridge.benchmarks.eval import (
    MemoryFactory,
    SampleResult,
    _answer_many,
    _BorrowedSpeechBackend,
    _cache_namespace,
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
    MemoryItem,
    load_task,
)
from mindbridge.benchmarks.eval_statistics import (
    ScoredValue,
    paired_comparison,
    parse_choice,
    summarize,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.benchmarks.task_catalog import TASKS, TaskSpec, expand
from mindbridge.benchmarks.video_mme_v2 import score_group_answers
from mindbridge.models.base import SpeechAnalysis


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


def test_mm_lifelong_ref_at_300_uses_official_quantized_iou() -> None:
    assert _ref_at_n(
        ((300.0, 600.0), (900.0, 1_200.0)),
        ((450.0, 750.0),),
        total_seconds=1_500.0,
        bucket_size=300.0,
    ) == pytest.approx(50.0)


def test_benchmark_speech_backend_skips_video_without_an_audio_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    silent = tmp_path / "silent.mp4"
    audible = tmp_path / "audible.mp4"
    silent.write_bytes(b"silent")
    audible.write_bytes(b"audible")
    calls: list[tuple[AssetRef, ...]] = []

    class Backend:
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            calls.append(tuple(assets))
            return tuple(SpeechAnalysis((), ()) for _asset in assets)

    monkeypatch.setattr(eval_module, "_has_audio", lambda path: path == audible)
    backend = _BorrowedSpeechBackend(Backend())
    assets = (
        AssetRef("a" * 64, Modality.VIDEO, "video/mp4", 6, "a" * 64, path=silent),
        AssetRef("b" * 64, Modality.VIDEO, "video/mp4", 7, "b" * 64, path=audible),
    )

    assert backend.analyze(assets) == (SpeechAnalysis((), ()), SpeechAnalysis((), ()))
    assert calls == [(assets[1],)]


def test_response_cache_namespace_changes_with_result_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = cast(
        eval_module._Arguments,
        SimpleNamespace(device=None, seed=7, gen_kwargs="{}", recall_limit=8),
    )
    before = _cache_namespace(arguments, ModelConfig(), {"text": 1})

    monkeypatch.setattr(eval_module, "EVAL_SCHEMA_VERSION", eval_module.EVAL_SCHEMA_VERSION + 1)

    assert _cache_namespace(arguments, ModelConfig(), {"text": 1}) != before


def test_model_base_url_override_replaces_environment_operation_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDBRIDGE_GENERATION_BASE_URL", "https://old.example/v1")

    config = _model_config("mindbridge", "base_url=https://new.example/v1")

    assert config.generation_base_url == "https://new.example/v1"


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
