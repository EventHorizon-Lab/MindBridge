"""Run pinned MindBridge benchmarks with lmms-eval-style task selection."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast

from mindbridge import (
    DEFAULT_FUNASR_MODEL_ID,
    AssetRef,
    AsyncMemory,
    FunASRTranscriber,
    JinaOmniEmbedder,
    MemoryType,
    MindBridgeError,
    Modality,
    OpenAIModels,
    SearchHit,
)
from mindbridge.benchmarks.download import acquire_inputs
from mindbridge.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
    MemoryItem,
    load_media_manifest,
    load_task,
)
from mindbridge.benchmarks.eval_cache import CachedAnswer, EvidenceInterval, ResponseCache
from mindbridge.benchmarks.eval_statistics import (
    ScoredValue,
    exact_match,
    paired_comparison,
    parse_choice,
    summarize,
    token_f1,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.benchmarks.prepare_media import _has_audio, prepare_task_media
from mindbridge.benchmarks.task_catalog import (
    DEFAULT_BENCHMARKS_ROOT,
    TASKS,
    expand,
    listing,
)
from mindbridge.models.base import (
    EmbeddingBackend,
    GenerationBackend,
    SpeechAnalysis,
    SpeechBackend,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_DIMENSION,
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
)

EVAL_SCHEMA_VERSION = 3
EVAL_RUNNER_VERSION = "mindbridge_eval_local_v3"
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
_RESULTS_FILE = "results.json"
_SAMPLES_FILE = "samples.jsonl"
_MODALITY_BY_SUFFIX = {
    ".aac": Modality.AUDIO,
    ".flac": Modality.AUDIO,
    ".jpeg": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".m4a": Modality.AUDIO,
    ".mkv": Modality.VIDEO,
    ".mov": Modality.VIDEO,
    ".mp3": Modality.AUDIO,
    ".mp4": Modality.VIDEO,
    ".png": Modality.IMAGE,
    ".wav": Modality.AUDIO,
    ".webm": Modality.VIDEO,
}


@dataclass(frozen=True, slots=True)
class _Arguments:
    tasks: tuple[str, ...]
    benchmarks_root: Path
    data_root: Path
    output_path: Path
    run_id: str
    dataset_overrides: Mapping[str, Path]
    media_overrides: Mapping[str, Path]
    media_manifest: Path | None
    limit: int | float | None
    offset: int
    batch_size: str
    max_batch_size: int
    unit_concurrency: int
    request_concurrency: int
    recall_limit: int
    seed: int
    seeds: tuple[int, int, int, int]
    bootstrap_samples: int
    model: str
    model_args: str
    gen_kwargs: str
    num_fewshot: int
    use_cache: Path | None
    device: str | None
    compare: Path | None
    fail_on_regression: bool
    regression_threshold: float
    predict_only: bool
    log_samples: bool
    allow_unverified_data: bool
    download: bool
    overwrite: bool
    quiet: bool


@dataclass(frozen=True, slots=True)
class SampleResult:
    """One answered question plus diagnostics needed for replay and comparison."""

    task: str
    benchmark: str
    dataset_sha256: str
    evaluation_sha256: str
    unit_id: str
    question_id: str
    prediction: str
    parsed_choice: str | None
    score: float | None
    exact_match: float | None
    latency_ms: float
    confidence: float
    memory_ids: tuple[str, ...]
    ingest_failure_count: int
    error_code: str | None
    metadata: Mapping[str, object]
    cached: bool = False
    prompt: tuple[str, ...] | None = None
    references: tuple[str, ...] | None = None
    evidence: tuple[EvidenceInterval, ...] = ()
    ref_at_300: float | None = None

    @property
    def sample_id(self) -> str:
        return f"{self.task}/{self.unit_id}/{self.question_id}"

    def json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": EVAL_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "task": self.task,
            "benchmark": self.benchmark,
            "dataset_sha256": self.dataset_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "unit_id": self.unit_id,
            "question_id": self.question_id,
            "prediction": self.prediction,
            "parsed_choice": self.parsed_choice,
            "score": self.score,
            "exact_match": self.exact_match,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "memory_ids": self.memory_ids,
            "evidence": tuple(item.json() for item in self.evidence),
            "ref_at_300": self.ref_at_300,
            "ingest_failure_count": self.ingest_failure_count,
            "error_code": self.error_code,
            "cached": self.cached,
            "metadata": dict(self.metadata),
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.references is not None:
            payload["references"] = self.references
        return payload


class _MemoryContext(Protocol):
    async def __aenter__(self) -> AsyncMemory: ...

    async def __aexit__(self, *_error: object) -> None: ...


MemoryFactory = Callable[[Path], _MemoryContext]


@dataclass(frozen=True, slots=True)
class _AnswerOutcome:
    prediction: str
    latency_ms: float
    confidence: float
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    cached: bool = False


class _BorrowedBackend:
    """Forward a shared backend while making per-store ``close`` a no-op."""

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def close(self) -> None:
        return None


class _BorrowedSpeechBackend(_BorrowedBackend):
    """Skip visual-only videos before lending the shared speech backend."""

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        selected = tuple(
            asset.modality is not Modality.VIDEO or asset.path is None or _has_audio(asset.path)
            for asset in assets
        )
        audible = tuple(asset for asset, include in zip(assets, selected, strict=True) if include)
        generated = () if not audible else cast(SpeechBackend, self._backend).analyze(audible)
        if len(generated) != len(audible):
            raise RuntimeError("speech backend returned the wrong number of analyses")
        pending = iter(generated)
        return tuple(
            next(pending) if include else SpeechAnalysis(turns=(), speakers=())
            for include in selected
        )


class _BackendPool:
    """Load model weights once and lend them to every isolated store."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | None,
        batch_size: int,
        needs_speech: bool,
        seed: int,
    ) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("benchmark execution requires mindbridge[openai]") from None
        self.client = OpenAI(
            api_key=config.generation_api_key,
            base_url=config.generation_base_url,
            timeout=config.timeout_seconds,
        )
        self.models = OpenAIModels(
            generation_client=self.client,
            generation_model=config.generation_model,
            generation_capabilities=config.generation_capabilities,
            generation_seed=seed,
            generation_temperature=0.0,
        )
        self.embedder = JinaOmniEmbedder(device=device, batch_size=batch_size)
        self.transcriber = FunASRTranscriber(device=device or "auto") if needs_speech else None
        self._answerer = cast(GenerationBackend, _BorrowedBackend(self.models))
        self._embedder = cast(EmbeddingBackend, _BorrowedBackend(self.embedder))
        self._transcriber = (
            cast(SpeechBackend, _BorrowedSpeechBackend(self.transcriber))
            if self.transcriber is not None
            else None
        )

    def memory(self, data_dir: Path) -> AsyncMemory:
        return AsyncMemory(
            data_dir=data_dir,
            embedder=self._embedder,
            answerer=self._answerer,
            transcriber=self._transcriber,
            index_speech=self._transcriber is not None,
        )

    def close(self) -> None:
        resources = (self.transcriber, self.embedder, self.client)
        for resource in resources:
            if resource is not None:
                with suppress(Exception):
                    resource.close()


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Parse one reproducible evaluation sweep and write its artifacts."""
    parser = _build_parser(prog)
    parsed = parser.parse_args(argv)
    list_mode = _list_mode(parsed)
    if list_mode is not None:
        print(listing(parsed.benchmarks_root.expanduser().resolve(), list_mode))
        return 0
    arguments = _arguments(parser, parsed)
    base_config = _model_config(arguments.model, arguments.model_args)
    _require_output(arguments.output_path, overwrite=arguments.overwrite)
    manifest, manifest_directory = load_media_manifest(arguments.media_manifest)
    if arguments.download:
        for name in arguments.tasks:
            acquire_inputs(
                TASKS[name],
                arguments.benchmarks_root,
                include_dataset=name not in arguments.dataset_overrides,
            )
    generated_manifest: dict[str, object] = {}
    for name in arguments.tasks:
        prepared = prepare_task_media(
            TASKS[name],
            root=arguments.benchmarks_root,
            dataset_path=arguments.dataset_overrides.get(
                name, TASKS[name].dataset_path(arguments.benchmarks_root)
            ),
            media_root=arguments.media_overrides.get(name),
            manifest=manifest,
            limit=arguments.limit,
            offset=arguments.offset,
            download=arguments.download,
            announce=None if arguments.quiet else _announce,
        )
        if prepared is not None:
            generated_manifest[name] = prepared
    if generated_manifest:
        manifest = _merged_manifest(manifest, manifest_directory, generated_manifest)
        effective_manifest = arguments.output_path / "media-manifest.json"
        effective_manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _atomic_replace(((effective_manifest, _json_bytes(manifest, pretty=True)),))
        arguments = replace(arguments, media_manifest=effective_manifest)
        manifest_directory = effective_manifest.parent
    loaded = tuple(
        load_task(
            TASKS[name],
            root=arguments.benchmarks_root,
            dataset_path=arguments.dataset_overrides.get(name),
            media_root=arguments.media_overrides.get(name),
            media_manifest=manifest,
            manifest_directory=manifest_directory,
            limit=arguments.limit,
            offset=arguments.offset,
            verify_digest=not (
                arguments.allow_unverified_data and name in arguments.dataset_overrides
            ),
        )
        for name in arguments.tasks
    )
    config = _evaluation_config(base_config, loaded)
    batch_sizes = {task.spec.name: _batch_size(arguments, task) for task in loaded}
    samples, duration = _execute(loaded, arguments, config, batch_sizes)
    results = _results(arguments, config, loaded, samples, duration, batch_sizes)
    comparisons = _comparisons(arguments, loaded, samples)
    if comparisons:
        results["comparisons"] = comparisons
    _write_artifacts(arguments, samples, results)
    if not arguments.quiet:
        print(_table(results))
    has_errors = any(
        sample.error_code is not None or sample.ingest_failure_count for sample in samples
    )
    regressed = arguments.fail_on_regression and _regressed(
        comparisons, threshold=arguments.regression_threshold
    )
    return int(has_errors or regressed)


def _execute(
    loaded: Sequence[LoadedTask],
    arguments: _Arguments,
    config: ModelConfig,
    batch_sizes: Mapping[str, int],
) -> tuple[tuple[SampleResult, ...], float]:
    needs_speech = any(
        isinstance(atom, Path)
        and _MODALITY_BY_SUFFIX.get(atom.suffix.casefold()) in {Modality.AUDIO, Modality.VIDEO}
        for task in loaded
        for unit in task.units
        for item in unit.memories
        for atom in item.content
    )
    started = time.perf_counter()
    response_cache = (
        None
        if arguments.use_cache is None
        else ResponseCache(
            arguments.use_cache,
            arguments.run_id,
            _cache_namespace(arguments, config, batch_sizes),
        )
    )
    pool: _BackendPool | None = None
    memory_factory: MemoryFactory
    try:
        if response_cache is not None and _all_cached(response_cache, loaded):
            memory_factory = _cache_only_memory
        else:
            pool = _BackendPool(
                config,
                device=arguments.device,
                batch_size=max(batch_sizes.values()),
                needs_speech=needs_speech,
                seed=arguments.seed,
            )
            memory_factory = pool.memory
        samples = asyncio.run(
            _run_all(
                loaded,
                arguments,
                batch_sizes=batch_sizes,
                memory_factory=memory_factory,
                response_cache=response_cache,
            )
        )
    finally:
        if pool is not None:
            pool.close()
        if response_cache is not None:
            response_cache.close()
    return samples, time.perf_counter() - started


async def _run_all(
    tasks: Sequence[LoadedTask],
    arguments: _Arguments,
    *,
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
) -> tuple[SampleResult, ...]:
    results: list[SampleResult] = []
    for task in tasks:
        if not arguments.quiet:
            print(
                f"mindbridge-bench eval: running {task.spec.name} ({len(task.units)} units)",
                file=sys.stderr,
            )
        run = BenchmarkRun(
            arguments.data_root,
            task.spec.name,
            arguments.run_id,
        )
        results.extend(
            await run_loaded_task(
                task,
                run=run,
                memory_factory=memory_factory,
                batch_size=batch_sizes[task.spec.name],
                unit_concurrency=arguments.unit_concurrency,
                request_concurrency=arguments.request_concurrency,
                recall_limit=arguments.recall_limit,
                predict_only=arguments.predict_only,
                log_samples=arguments.log_samples,
                response_cache=response_cache,
            )
        )
    return tuple(results)


async def run_loaded_task(
    task: LoadedTask,
    *,
    run: BenchmarkRun,
    memory_factory: MemoryFactory,
    batch_size: int,
    unit_concurrency: int,
    request_concurrency: int,
    recall_limit: int,
    predict_only: bool = False,
    log_samples: bool = False,
    response_cache: ResponseCache | None = None,
) -> tuple[SampleResult, ...]:
    """Run normalized units with bounded workers while preserving release order."""
    if min(batch_size, unit_concurrency, request_concurrency, recall_limit) <= 0:
        raise ValueError("batch size, concurrency, and recall limit must be positive")
    if recall_limit > 100:
        raise ValueError("recall limit must not exceed 100")
    slots: list[tuple[SampleResult, ...] | None] = [None] * len(task.units)
    queue: asyncio.Queue[tuple[int, EvalUnit]] = asyncio.Queue()
    for index, unit in enumerate(task.units):
        queue.put_nowait((index, unit))

    async def worker() -> None:
        while not queue.empty():
            try:
                index, unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            data_dir = run.unit_dir(unit.unit_id)
            slots[index] = await _run_unit(
                task,
                unit,
                data_dir,
                memory_factory=memory_factory,
                batch_size=batch_size,
                request_concurrency=request_concurrency,
                recall_limit=recall_limit,
                predict_only=predict_only,
                log_samples=log_samples,
                response_cache=response_cache,
            )
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(unit_concurrency, len(task.units)))]
    await asyncio.gather(*workers)
    if any(group is None for group in slots):
        raise RuntimeError("evaluation worker exited before every unit completed")
    return tuple(sample for group in slots if group is not None for sample in group)


async def _run_unit(
    task: LoadedTask,
    unit: EvalUnit,
    data_dir: Path,
    *,
    memory_factory: MemoryFactory,
    batch_size: int,
    request_concurrency: int,
    recall_limit: int,
    predict_only: bool,
    log_samples: bool,
    response_cache: ResponseCache | None,
) -> tuple[SampleResult, ...]:
    results = _cached_results(
        response_cache,
        task,
        unit,
        predict_only=predict_only,
        log_samples=log_samples,
    )
    if len(results) == len(unit.questions):
        return tuple(results[question.question_id] for question in unit.questions)
    questions_by_cutoff, cutoffs = _pending_questions(unit, results)
    memories = tuple(
        sorted(
            unit.memories,
            key=lambda item: (
                math.inf if item.end_seconds is None else item.end_seconds,
                item.source_id,
            ),
        )
    )
    pending = 0
    ingest_failures = 0
    try:
        async with memory_factory(data_dir) as memory:
            for cutoff in cutoffs:
                boundary = math.inf if cutoff is None else cutoff
                end = pending
                while end < len(memories) and _memory_end(memories[end]) <= boundary:
                    end += 1
                ingest_failures += await _ingest(
                    memory, memories[pending:end], batch_size=batch_size
                )
                pending = end
                answered = await _answer_many(
                    memory,
                    questions_by_cutoff[cutoff],
                    request_concurrency=request_concurrency,
                    recall_limit=recall_limit,
                )
                for question, outcome in zip(questions_by_cutoff[cutoff], answered, strict=True):
                    _cache_outcome(
                        response_cache,
                        task,
                        unit,
                        question,
                        outcome,
                        ingest_failures,
                    )
                    results[question.question_id] = _sample(
                        task,
                        unit,
                        question,
                        outcome,
                        ingest_failures=ingest_failures,
                        predict_only=predict_only,
                        log_samples=log_samples,
                    )
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        if len(results) == len(unit.questions):
            raise
        for question in unit.questions:
            results.setdefault(
                question.question_id,
                _sample(
                    task,
                    unit,
                    question,
                    error,
                    ingest_failures=ingest_failures,
                    predict_only=predict_only,
                    log_samples=log_samples,
                ),
            )
    return tuple(results[question.question_id] for question in unit.questions)


def _cached_results(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    *,
    predict_only: bool,
    log_samples: bool,
) -> dict[str, SampleResult]:
    if cache is None:
        return {}
    results = {}
    for question in unit.questions:
        answer = cache.get(_cache_task(task), unit.unit_id, question.question_id)
        if answer is not None:
            outcome = _AnswerOutcome(
                answer.prediction,
                0.0,
                answer.confidence,
                answer.memory_ids,
                answer.evidence,
                cached=True,
            )
            results[question.question_id] = _sample(
                task,
                unit,
                question,
                outcome,
                ingest_failures=0,
                predict_only=predict_only,
                log_samples=log_samples,
            )
    return results


def _pending_questions(
    unit: EvalUnit, completed: Mapping[str, SampleResult]
) -> tuple[dict[float | None, list[EvalQuestion]], tuple[float | None, ...]]:
    groups: dict[float | None, list[EvalQuestion]] = {}
    for question in unit.questions:
        if question.question_id not in completed:
            groups.setdefault(question.cutoff_seconds, []).append(question)
    cutoffs: list[float | None] = [*sorted(value for value in groups if value is not None)]
    if None in groups:
        cutoffs.append(None)
    return groups, tuple(cutoffs)


def _cache_outcome(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    outcome: _AnswerOutcome | BaseException,
    ingest_failures: int,
) -> None:
    if (
        cache is None
        or ingest_failures
        or not isinstance(outcome, _AnswerOutcome)
        or not outcome.prediction.strip()
    ):
        return
    cache.put(
        _cache_task(task),
        unit.unit_id,
        question.question_id,
        CachedAnswer(
            outcome.prediction,
            outcome.confidence,
            outcome.memory_ids,
            outcome.evidence,
        ),
    )


async def _ingest(
    memory: AsyncMemory,
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
) -> int:
    async def add_chunk(chunk: Sequence[MemoryItem]) -> int:
        contents = tuple(_memory_content(item) for item in chunk)
        try:
            await memory.add_many(
                contents,
                occurred_at=tuple(item.occurred_at for item in chunk),
                occurred_end=tuple(item.occurred_end for item in chunk),
                metadata=tuple(_memory_metadata(item) for item in chunk),
                memory_type=MemoryType.EPISODIC,
            )
            return 0
        except Exception:
            if len(chunk) > 1:
                middle = len(chunk) // 2
                return await add_chunk(chunk[:middle]) + await add_chunk(chunk[middle:])
        try:
            await memory.add(
                contents[0],
                occurred_at=chunk[0].occurred_at,
                occurred_end=chunk[0].occurred_end,
                metadata=_memory_metadata(chunk[0]),
                memory_type=MemoryType.EPISODIC,
            )
        except Exception:
            return 1
        return 0

    return sum(
        [
            await add_chunk(items[offset : offset + batch_size])
            for offset in range(0, len(items), batch_size)
        ]
    )


async def _answer_many(
    memory: AsyncMemory,
    questions: Sequence[EvalQuestion],
    *,
    request_concurrency: int,
    recall_limit: int,
) -> tuple[_AnswerOutcome | BaseException, ...]:
    semaphore = asyncio.Semaphore(request_concurrency)

    async def answer(question: EvalQuestion) -> _AnswerOutcome:
        started = time.perf_counter()
        async with semaphore:
            result = await memory.ask(
                _content(question.content),
                limit=recall_limit,
                reference_at=question.reference_at,
            )
        return _AnswerOutcome(
            result.answer,
            (time.perf_counter() - started) * 1_000,
            max((hit.score for hit in result.hits), default=0.0),
            tuple(hit.id for hit in result.hits),
            tuple(_evidence(hit) for hit in result.hits),
        )

    return tuple(
        await asyncio.gather(*(answer(question) for question in questions), return_exceptions=True)
    )


def _sample(
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    outcome: _AnswerOutcome | BaseException,
    *,
    ingest_failures: int,
    predict_only: bool,
    log_samples: bool,
) -> SampleResult:
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    if isinstance(outcome, BaseException):
        prediction, latency_ms, confidence, memory_ids, evidence, cached = (
            "",
            0.0,
            0.0,
            (),
            (),
            False,
        )
        error_code = _error_code(outcome)
    else:
        prediction = outcome.prediction
        latency_ms = outcome.latency_ms
        confidence = outcome.confidence
        memory_ids = outcome.memory_ids
        evidence = outcome.evidence
        cached = outcome.cached
        error_code = None
    choices = tuple(
        str(value) for value in cast(Sequence[object], question.metadata.get("choices", ()))
    )
    parsed = _parsed_choice(task.spec.name, prediction, choices)
    score, matched = _score(question, prediction, parsed, predict_only=predict_only)
    return SampleResult(
        task=task.spec.name,
        benchmark=task.spec.benchmark,
        dataset_sha256=task.dataset_sha256,
        evaluation_sha256=task.evaluation_sha256,
        unit_id=unit.unit_id,
        question_id=question.question_id,
        prediction=prediction,
        parsed_choice=parsed,
        score=score,
        exact_match=matched,
        latency_ms=latency_ms,
        confidence=confidence,
        memory_ids=memory_ids,
        ingest_failure_count=ingest_failures,
        error_code=error_code,
        cached=cached,
        metadata=question.metadata,
        prompt=tuple(str(part) for part in question.content) if log_samples else None,
        references=question.references if log_samples else None,
        evidence=evidence,
        ref_at_300=_reference_grounding(task, unit, question, evidence),
    )


def _score(
    question: EvalQuestion,
    prediction: str,
    parsed_choice: str | None,
    *,
    predict_only: bool,
) -> tuple[float | None, float | None]:
    if predict_only or question.score_kind == "submission":
        return None, None
    if question.score_kind == "choice":
        return float(parsed_choice == question.expected_choice), None
    return token_f1(prediction, question.references), exact_match(prediction, question.references)


def _evidence(hit: SearchHit) -> EvidenceInterval:
    source_id = hit.metadata.get("source_id")
    start = hit.metadata.get("start_seconds")
    end = hit.metadata.get("end_seconds")
    return EvidenceInterval(
        hit.id,
        source_id if isinstance(source_id, str) else None,
        _optional_seconds(start),
        _optional_seconds(end),
    )


def _optional_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _reference_grounding(
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    evidence: Sequence[EvidenceInterval],
) -> float | None:
    if task.spec.benchmark != "MM-Lifelong":
        return None
    reference = question.metadata.get("reference_intervals")
    ends = tuple(item.end_seconds for item in unit.memories if item.end_seconds is not None)
    if not isinstance(reference, Sequence) or isinstance(reference, str | bytes) or not ends:
        return None
    try:
        expected = tuple((float(interval[0]), float(interval[1])) for interval in reference)
    except (IndexError, TypeError, ValueError):
        return None
    predicted = tuple(
        (item.start_seconds, item.end_seconds)
        for item in evidence
        if item.start_seconds is not None and item.end_seconds is not None
    )
    return _ref_at_n(predicted, expected, total_seconds=max(ends), bucket_size=300.0)


def _ref_at_n(
    predicted: Sequence[tuple[float, float]],
    expected: Sequence[tuple[float, float]],
    *,
    total_seconds: float,
    bucket_size: float,
) -> float:
    """Official MM-Lifelong quantized temporal IoU."""
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("total_seconds must be positive and finite")
    if not math.isfinite(bucket_size) or bucket_size <= 0:
        raise ValueError("bucket_size must be positive and finite")

    def buckets(intervals: Sequence[tuple[float, float]]) -> set[int]:
        values: set[int] = set()
        for start, end in intervals:
            start = max(0.0, start)
            end = min(total_seconds, end)
            if start >= end:
                continue
            first = int(start // bucket_size)
            last = int(end // bucket_size)
            values.update(range(first, last + 1))
        return values

    predicted_buckets = buckets(predicted)
    expected_buckets = buckets(expected)
    if not predicted_buckets and not expected_buckets:
        return 0.0
    return (
        100.0
        * len(predicted_buckets & expected_buckets)
        / len(predicted_buckets | expected_buckets)
    )


def _parsed_choice(task_name: str, prediction: str, choices: Sequence[str]) -> str | None:
    if task_name == "video-mme":
        from mindbridge.benchmarks.video_mme import parse_video_mme_option

        return parse_video_mme_option(prediction)
    if task_name == "video-mme-v2":
        from mindbridge.benchmarks.video_mme_v2 import parse_video_mme_v2_option

        return parse_video_mme_v2_option(prediction)
    return parse_choice(prediction, choices)


def _results(
    arguments: _Arguments,
    config: ModelConfig,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    duration_seconds: float,
    batch_sizes: Mapping[str, int],
) -> dict[str, object]:
    task_rows = []
    for task in tasks:
        selected = tuple(sample for sample in samples if sample.task == task.spec.name)
        task_rows.append(
            {
                "task": task.spec.name,
                "benchmark": task.spec.benchmark,
                "variant": task.spec.variant,
                "adapter_version": task.spec.adapter_version,
                "source_repository": task.spec.repository,
                "source_revision": task.spec.revision,
                "media_source": (
                    None
                    if task.spec.media_source is None
                    else {
                        "release": task.spec.media_source.release,
                        "repository": task.spec.media_source.repository,
                        "revision": task.spec.media_source.revision,
                        "patterns": task.spec.media_source.patterns,
                        "acquirer": task.spec.media_source.acquirer,
                    }
                ),
                "dataset_path": str(task.dataset_path),
                "dataset_sha256": task.dataset_sha256,
                "input_sha256": dict(task.input_sha256),
                "evaluation_sha256": task.evaluation_sha256,
                "batch_size": batch_sizes[task.spec.name],
                **_metrics(task, selected, arguments),
            }
        )
    media_roots = {}
    for name in arguments.tasks:
        path = arguments.media_overrides.get(name) or TASKS[name].media_root(
            arguments.benchmarks_root
        )
        media_roots[name] = None if path is None else str(path)
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "runner_version": EVAL_RUNNER_VERSION,
        "run_id": arguments.run_id,
        "status": (
            "completed_with_errors"
            if any(
                sample.error_code is not None or sample.ingest_failure_count for sample in samples
            )
            else "completed"
        ),
        "duration_seconds": duration_seconds,
        "seed": arguments.seed,
        "seeds": arguments.seeds,
        "bootstrap_samples": arguments.bootstrap_samples,
        "predict_only": arguments.predict_only,
        "num_fewshot": arguments.num_fewshot,
        "log_samples": arguments.log_samples,
        "response_cache": None if arguments.use_cache is None else str(arguments.use_cache),
        "cached_response_count": sum(sample.cached for sample in samples),
        "allow_unverified_data": arguments.allow_unverified_data,
        "limit": arguments.limit,
        "offset": arguments.offset,
        "data_root": str(arguments.data_root),
        "media_manifest_path": (
            None if arguments.media_manifest is None else str(arguments.media_manifest.resolve())
        ),
        "media_roots": media_roots,
        "unit_concurrency": arguments.unit_concurrency,
        "request_concurrency": arguments.request_concurrency,
        "recall_limit": arguments.recall_limit,
        "model": {
            "adapter": arguments.model,
            "embedding_model": DEFAULT_JINA_MODEL_ID,
            "embedding_revision": DEFAULT_JINA_REVISION,
            "embedding_dimension": DEFAULT_JINA_DIMENSION,
            "device": arguments.device or "auto",
            "generation_model": config.generation_model,
            "generation_base_url": config.generation_base_url,
            "generation_modalities": sorted(
                modality.value for modality in config.generation_capabilities
            ),
            "generation_seed": arguments.seed,
            "generation_temperature": 0.0,
            "generation_kwargs": arguments.gen_kwargs,
            "transcription_model": DEFAULT_FUNASR_MODEL_ID,
            "timeout_seconds": config.timeout_seconds,
        },
        "environment": {
            "mindbridge_version": _version("mindbridge"),
            "zvec_version": _version("zvec"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tasks": task_rows,
    }


def _metrics(
    task: LoadedTask,
    samples: Sequence[SampleResult],
    arguments: _Arguments,
) -> dict[str, object]:
    seed = _task_seed(arguments.seed, task.spec.name)
    scored = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
        for sample in samples
        if sample.score is not None
    )
    primary = summarize(
        scored,
        seed=seed,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    score_kinds = {question.score_kind for unit in task.units for question in unit.questions}
    metric_name = _metric_name(task, bool(scored), score_kinds)
    exact = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.exact_match)
        for sample in samples
        if sample.exact_match is not None
    )
    latencies = sorted(sample.latency_ms for sample in samples if sample.latency_ms > 0)
    values: dict[str, object] = {
        "primary_metric": metric_name,
        "official_metric": task.spec.name in {"egolifeqa", "supermemory-vqa"},
        "score": primary,
        "exact_match": (
            summarize(exact, seed=seed, bootstrap_samples=arguments.bootstrap_samples)
            if exact
            else None
        ),
        "question_count": len(samples),
        "error_count": sum(sample.error_code is not None for sample in samples),
        "ingest_failure_count": sum(
            max(sample.ingest_failure_count for sample in samples if sample.unit_id == unit_id)
            for unit_id in {sample.unit_id for sample in samples}
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }
    if task.spec.name == "video-mme" and scored:
        values["video_mme"] = _video_mme_metrics(
            samples,
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
            strict=primary,
        )
    if task.spec.name == "video-mme-v2" and scored:
        rating = _video_mme_v2_rating(
            samples,
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
        )
        values.update(
            {
                "primary_metric": "rating",
                "official_metric": True,
                "score": rating,
                "question_accuracy": primary,
            }
        )
    if task.spec.name == "supermemory-vqa" and scored:
        values["answerability"] = _answerability(samples)
    reference_scores = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.ref_at_300)
        for sample in samples
        if sample.ref_at_300 is not None
    )
    if reference_scores:
        values["ref_at_300"] = {
            "official_metric": True,
            **summarize(
                reference_scores,
                seed=seed,
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=(0.0, 100.0),
            ),
        }
    return values


def _metric_name(task: LoadedTask, has_scores: bool, score_kinds: Collection[str]) -> str:
    if not has_scores:
        return "submission"
    if task.spec.name == "video-mme":
        return "strict_accuracy"
    if task.spec.name == "supermemory-vqa":
        return "qa_accuracy"
    return "accuracy" if "choice" in score_kinds else "token_f1"


def _video_mme_metrics(
    samples: Sequence[SampleResult],
    *,
    seed: int,
    bootstrap_samples: int,
    strict: Mapping[str, object],
) -> dict[str, object]:
    overall = _video_mme_cell(
        samples,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        strict=strict,
    )
    return {
        **overall,
        "by_duration": {
            duration: _video_mme_cell(
                tuple(sample for sample in samples if sample.metadata.get("duration") == duration),
                seed=_task_seed(seed, duration),
                bootstrap_samples=bootstrap_samples,
            )
            for duration in ("short", "medium", "long")
            if any(sample.metadata.get("duration") == duration for sample in samples)
        },
    }


def _video_mme_cell(
    samples: Sequence[SampleResult],
    *,
    seed: int,
    bootstrap_samples: int,
    strict: Mapping[str, object] | None = None,
) -> dict[str, object]:
    scores = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
        for sample in samples
        if sample.score is not None
    )
    answered = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
        for sample in samples
        if sample.score is not None and sample.parsed_choice is not None
    )
    return {
        "accuracy": summarize(answered, seed=seed, bootstrap_samples=bootstrap_samples),
        "strict_accuracy": strict
        or summarize(scores, seed=seed, bootstrap_samples=bootstrap_samples),
        "question_count": len(samples),
        "answered_count": len(answered),
        "unanswered_count": len(samples) - len(answered),
    }


def _video_mme_v2_rating(
    samples: Sequence[SampleResult], *, seed: int, bootstrap_samples: int
) -> dict[str, object]:
    from mindbridge.benchmarks.video_mme_v2 import score_group_answers

    groups: dict[str, list[SampleResult]] = {}
    for sample in samples:
        groups.setdefault(sample.unit_id, []).append(sample)
    ratings = []
    for unit_id in sorted(groups):
        group = sorted(groups[unit_id], key=_position)
        correct = tuple(sample.score == 1.0 for sample in group)
        group_type = str(group[0].metadata["group_type"])
        rating = score_group_answers(
            group_type,
            str(group[0].metadata.get("group_structure", "")),
            correct,
        )
        ratings.append(ScoredValue(unit_id, unit_id, rating))
    return summarize(
        ratings,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        clamp=(0.0, 100.0),
    )


def _position(sample: SampleResult) -> int:
    value = sample.metadata.get("position")
    if not isinstance(value, int):
        raise ValueError("Video-MME-v2 sample position must be an integer")
    return value


def _answerability(samples: Sequence[SampleResult]) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    for sample in samples:
        actual = bool(sample.metadata["is_answerable"])
        predicted = (
            sample.parsed_choice is not None
            and sample.parsed_choice != sample.metadata["unanswerable_choice"]
        )
        true_positive += actual and predicted
        false_positive += not actual and predicted
        false_negative += actual and not predicted
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _comparisons(
    arguments: _Arguments,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
) -> list[dict[str, object]]:
    if arguments.compare is None:
        return []
    baseline = _baseline_samples(arguments.compare)
    rows = []
    for task in tasks:
        current = tuple(
            ScoredValue(sample.sample_id, sample.unit_id, sample.score)
            for sample in samples
            if sample.task == task.spec.name and sample.score is not None
        )
        previous_rows = tuple(row for row in baseline if row.get("task") == task.spec.name)
        digests = {row.get("evaluation_sha256") for row in previous_rows}
        if previous_rows and digests != {task.evaluation_sha256}:
            raise ValueError(f"baseline evaluation inputs differ for {task.spec.name}")
        previous = tuple(_baseline_value(row) for row in previous_rows if _has_score(row))
        if not current:
            continue
        if not previous:
            raise ValueError(f"baseline has no scored samples for {task.spec.name}")
        rows.append(
            {
                "task": task.spec.name,
                "metric": (
                    "question_accuracy"
                    if task.spec.name == "video-mme-v2"
                    else _comparison_metric(task)
                ),
                **paired_comparison(
                    current,
                    previous,
                    seed=_task_seed(arguments.seed, task.spec.name),
                    bootstrap_samples=arguments.bootstrap_samples,
                ),
            }
        )
    return rows


def _comparison_metric(task: LoadedTask) -> str:
    kinds = {question.score_kind for unit in task.units for question in unit.questions}
    return _metric_name(task, True, kinds)


def _has_score(row: Mapping[str, object]) -> bool:
    value = row.get("score")
    return not isinstance(value, bool) and isinstance(value, int | float)


def _baseline_value(row: Mapping[str, object]) -> ScoredValue:
    value = row["score"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("baseline score must be numeric")
    sample_id, unit_id = row.get("sample_id"), row.get("unit_id")
    if not isinstance(sample_id, str) or not isinstance(unit_id, str):
        raise ValueError("baseline sample and unit IDs must be strings")
    return ScoredValue(sample_id, unit_id, float(value))


def _baseline_samples(path: Path) -> tuple[dict[str, object], ...]:
    resolved = path.expanduser().resolve()
    sample_path = resolved / _SAMPLES_FILE if resolved.is_dir() else resolved
    if sample_path.name == _RESULTS_FILE:
        sample_path = sample_path.with_name(_SAMPLES_FILE)
    if not sample_path.is_file():
        raise FileNotFoundError(f"baseline samples do not exist: {sample_path}")
    rows = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("baseline sample rows must be JSON objects")
        if row.get("schema_version") != EVAL_SCHEMA_VERSION:
            raise ValueError("baseline sample schema version is unsupported")
        rows.append(row)
    return tuple(rows)


def _write_artifacts(
    arguments: _Arguments,
    samples: Sequence[SampleResult],
    results: Mapping[str, object],
) -> None:
    samples_bytes = "".join(
        json.dumps(
            sample.json(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for sample in samples
    ).encode("utf-8")
    document = dict(results)
    document["samples_sha256"] = hashlib.sha256(samples_bytes).hexdigest()
    results_bytes = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    arguments.output_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_replace(
        (
            (arguments.output_path / _SAMPLES_FILE, samples_bytes),
            (arguments.output_path / _RESULTS_FILE, results_bytes),
        )
    )


def _atomic_replace(files: Sequence[tuple[Path, bytes]]) -> None:
    temporary: list[tuple[Path, Path]] = []
    try:
        for target, content in files:
            with NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), target))
        for source, target in temporary:
            os.replace(source, target)
    finally:
        for source, _target in temporary:
            source.unlink(missing_ok=True)


def _merged_manifest(
    manifest: Mapping[str, object] | None,
    manifest_directory: Path | None,
    generated: Mapping[str, object],
) -> dict[str, object]:
    payload = cast(
        dict[str, object], _absolute_manifest_paths(dict(manifest or {}), manifest_directory)
    )
    tasks = payload.get("tasks")
    merged = dict(tasks) if isinstance(tasks, dict) else {}
    merged.update(generated)
    payload.update({"version": 1, "tasks": merged})
    return payload


def _absolute_manifest_paths(value: object, directory: Path | None) -> object:
    if isinstance(value, list):
        return [_absolute_manifest_paths(item, directory) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _absolute_manifest_paths(item, directory) for key, item in value.items()}
    path = result.get("path")
    if (
        isinstance(path, str)
        and directory is not None
        and not Path(path).expanduser().is_absolute()
    ):
        result["path"] = str((directory / Path(path).expanduser()).resolve())
    return result


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _announce(message: str) -> None:
    print(f"mindbridge-bench eval: {message}", file=sys.stderr)


def _table(results: Mapping[str, object]) -> str:
    tasks = cast(Sequence[Mapping[str, object]], results["tasks"])
    rows = []
    for task in tasks:
        score = cast(Mapping[str, object], task["score"])
        mean = score.get("mean")
        interval = score.get("confidence_interval_95")
        rows.append(
            (
                str(task["task"]),
                str(task["primary_metric"]),
                "—" if mean is None else f"{float(cast(float, mean)):.4f}",
                (
                    "—"
                    if not isinstance(interval, list)
                    else f"[{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
                ),
                str(task["question_count"]),
                str(task["error_count"]),
            )
        )
    headers = ("task", "metric", "value", "95% cluster CI", "n", "errors")
    widths = tuple(
        max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))
    )
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (headers, *rows)
    )


def _build_parser(prog: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="mindbridge", help="evaluation adapter")
    parser.add_argument(
        "--model-args",
        "--model_args",
        default="",
        help="comma-separated generation_model/base_url/timeout_seconds overrides",
    )
    parser.add_argument("--tasks", help="comma-separated task names or groups")
    parser.add_argument("--list-tasks", action="store_true", help="list task pins and readiness")
    parser.add_argument("--benchmarks-root", type=Path, default=DEFAULT_BENCHMARKS_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path(".benchmarks/data"))
    parser.add_argument("--output-path", "--output_path", type=Path)
    parser.add_argument("--run-id", type=_run_identifier)
    parser.add_argument(
        "--task-data",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="override one task's annotation path; repeatable",
    )
    parser.add_argument(
        "--media-root",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="override one task's media root; repeatable",
    )
    parser.add_argument("--media-manifest", type=Path, help="prepared clip/caption manifest")
    parser.add_argument(
        "--limit",
        type=_limit_value,
        help="-1/all, a 0-1 fraction, or an absolute example count",
    )
    parser.add_argument("--offset", type=_nonnegative_int, default=0)
    parser.add_argument("--num-fewshot", "--num_fewshot", type=_nonnegative_int, default=0)
    parser.add_argument("--gen-kwargs", "--gen_kwargs", default="")
    parser.add_argument("--batch-size", "--batch_size", "-b", default="auto")
    parser.add_argument("--max-batch-size", "--max_batch_size", type=_positive_int, default=64)
    parser.add_argument("--unit-concurrency", type=_positive_int, default=1)
    parser.add_argument("--request-concurrency", type=_positive_int, default=4)
    parser.add_argument("--recall-limit", type=_positive_int, default=20)
    parser.add_argument("--seed", type=_seed_values, default=(0, 1234, 1234, 1234))
    parser.add_argument(
        "--bootstrap-samples", type=_positive_int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--device", help="Jina/FunASR device: cpu, cuda, or cuda:N")
    parser.add_argument(
        "--use-cache",
        "--use_cache",
        "-c",
        type=Path,
        help="SQLite response-cache directory or .db file",
    )
    parser.add_argument("--compare", type=Path, help="prior result directory or samples.jsonl")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--regression-threshold", type=_nonnegative_float, default=0.0)
    parser.add_argument("--predict-only", "--predict_only", "-x", action="store_true")
    parser.add_argument("--log-samples", "--log_samples", action="store_true")
    parser.add_argument("--allow-unverified-data", action="store_true")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download missing pinned annotations and selected media",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--verbosity",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    parser.add_argument("--check-integrity", "--check_integrity", action="store_true")
    return parser


def _arguments(parser: argparse.ArgumentParser, parsed: argparse.Namespace) -> _Arguments:
    if parsed.model != "mindbridge":
        parser.error("--model must be mindbridge")
    if not parsed.tasks:
        parser.error("--tasks is required unless --list-tasks is used")
    if parsed.fail_on_regression and parsed.compare is None:
        parser.error("--fail-on-regression requires --compare")
    if parsed.recall_limit > 100:
        parser.error("--recall-limit must not exceed 100")
    if parsed.num_fewshot:
        parser.error("the supported memory benchmarks are zero-shot; --num_fewshot must be 0")
    requested = tuple(part.strip() for part in parsed.tasks.split(",") if part.strip())
    try:
        tasks = expand(requested)
        dataset_overrides = _assignments(parsed.task_data, tasks)
        media_overrides = _assignments(parsed.media_root, tasks)
        _parse_batch_size(parsed.batch_size)
        gen_kwargs = _generation_kwargs(parsed.gen_kwargs, parsed.seed[0])
    except ValueError as error:
        parser.error(str(error))
    run_id = parsed.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = parsed.output_path or parsed.benchmarks_root / "results" / run_id
    return _Arguments(
        tasks=tasks,
        benchmarks_root=parsed.benchmarks_root.expanduser().resolve(),
        data_root=parsed.data_root.expanduser().resolve(),
        output_path=output.expanduser().resolve(),
        run_id=run_id,
        dataset_overrides=dataset_overrides,
        media_overrides=media_overrides,
        media_manifest=(
            None if parsed.media_manifest is None else parsed.media_manifest.expanduser().resolve()
        ),
        limit=parsed.limit,
        offset=parsed.offset,
        batch_size=parsed.batch_size,
        max_batch_size=parsed.max_batch_size,
        unit_concurrency=parsed.unit_concurrency,
        request_concurrency=parsed.request_concurrency,
        recall_limit=parsed.recall_limit,
        seed=parsed.seed[0],
        seeds=parsed.seed,
        bootstrap_samples=parsed.bootstrap_samples,
        model=parsed.model,
        model_args=parsed.model_args,
        gen_kwargs=gen_kwargs,
        num_fewshot=parsed.num_fewshot,
        use_cache=(None if parsed.use_cache is None else parsed.use_cache.expanduser().resolve()),
        device=parsed.device,
        compare=parsed.compare,
        fail_on_regression=parsed.fail_on_regression,
        regression_threshold=parsed.regression_threshold,
        predict_only=parsed.predict_only,
        log_samples=parsed.log_samples,
        allow_unverified_data=parsed.allow_unverified_data,
        download=parsed.download,
        overwrite=parsed.overwrite,
        quiet=parsed.quiet or parsed.verbosity in {"ERROR", "CRITICAL"},
    )


def _model_config(model: str, arguments: str) -> ModelConfig:
    if model != "mindbridge":
        raise ValueError("model must be mindbridge")
    config = ModelConfig.from_environment()
    allowed = {
        "base_url",
        "generation_model",
        "timeout_seconds",
    }
    aliases = {"pretrained": "generation_model", "model": "generation_model"}
    for item in (part.strip() for part in arguments.split(",") if part.strip()):
        key, separator, value = item.partition("=")
        key = aliases.get(key.strip(), key.strip())
        if not separator or key not in allowed or not value.strip():
            raise ValueError(f"invalid --model-args item: {item}")
        parsed = value.strip()
        config = _replace_config(config, key, parsed)
    return config


def _evaluation_config(config: ModelConfig, tasks: Sequence[LoadedTask]) -> ModelConfig:
    required = {Modality.TEXT}
    required.update(
        modality
        for task in tasks
        for unit in task.units
        for memory in unit.memories
        for atom in memory.content
        if isinstance(atom, Path)
        and (modality := _MODALITY_BY_SUFFIX.get(atom.suffix.casefold())) is not None
    )
    required.update(
        modality
        for task in tasks
        for unit in task.units
        for question in unit.questions
        for atom in question.content
        if isinstance(atom, Path)
        and (modality := _MODALITY_BY_SUFFIX.get(atom.suffix.casefold())) is not None
    )
    return replace(
        config,
        generation_capabilities=frozenset((*config.generation_capabilities, *required)),
    )


def _replace_config(config: ModelConfig, key: str, value: str) -> ModelConfig:
    if key == "base_url":
        return replace(
            config,
            generation_base_url=value,
        )
    if key == "generation_model":
        return replace(config, generation_model=value)
    if key == "timeout_seconds":
        return replace(config, timeout_seconds=float(value))
    raise AssertionError(f"unhandled model setting: {key}")


def _batch_size(arguments: _Arguments, task: LoadedTask) -> int:
    explicit = _parse_batch_size(arguments.batch_size)
    if explicit is not None:
        return min(explicit, arguments.max_batch_size)
    has_media = any(
        isinstance(atom, Path)
        for unit in task.units
        for memory in unit.memories
        for atom in memory.content
    )
    automatic_cap = (
        int(arguments.batch_size.partition(":")[2])
        if arguments.batch_size.startswith("auto:")
        else arguments.max_batch_size
    )
    return min(8 if has_media else 64, arguments.max_batch_size, automatic_cap)


def _parse_batch_size(value: str) -> int | None:
    if value == "auto":
        return None
    if value.startswith("auto:"):
        suffix = value.partition(":")[2]
        if suffix.isdigit() and int(suffix) > 0:
            return None
        raise ValueError("--batch-size must be auto, auto:N, or a positive integer")
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError("--batch-size must be auto, auto:N, or a positive integer") from None
    if parsed <= 0:
        raise ValueError("--batch-size must be auto, auto:N, or a positive integer")
    return parsed


def _assignments(values: Sequence[str], selected: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or not task.strip() or not path.strip():
            raise ValueError("path overrides must use TASK=PATH")
        normalized = expand((task.strip(),))
        if len(normalized) != 1:
            raise ValueError(f"path override must name one concrete task: {task}")
        name = normalized[0]
        if name not in selected:
            raise ValueError(f"path override names an unselected task: {name}")
        if name in result:
            raise ValueError(f"path override repeats task: {name}")
        result[name] = Path(path).expanduser().resolve()
    return result


def _require_output(path: Path, *, overwrite: bool) -> None:
    for name in (_RESULTS_FILE, _SAMPLES_FILE):
        target = path / name
        if target.exists() and not overwrite:
            raise FileExistsError(f"evaluation artifact already exists: {target}")


def _content(parts: tuple[str | Path, ...]) -> str | Path | tuple[str | Path, ...]:
    return parts[0] if len(parts) == 1 else parts


def _memory_content(item: MemoryItem) -> str | tuple[str | Path, ...]:
    marker = f"[source_id: {item.source_id}]"
    first, *rest = item.content
    parts: tuple[str | Path, ...] = (
        (f"{marker}\n{first}", *rest) if isinstance(first, str) else (marker, first, *rest)
    )
    return cast(str, parts[0]) if len(parts) == 1 else parts


def _memory_metadata(item: MemoryItem) -> dict[str, object]:
    values: dict[str, object] = {
        "source_id": item.source_id,
        "start_seconds": item.start_seconds,
    }
    if item.end_seconds is not None:
        values["end_seconds"] = item.end_seconds
    return values


def _memory_end(item: MemoryItem) -> float:
    return math.inf if item.end_seconds is None else item.end_seconds


def _error_code(error: BaseException) -> str:
    return error.code if isinstance(error, MindBridgeError) else type(error).__name__


def _task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cache_namespace(
    arguments: _Arguments, config: ModelConfig, batch_sizes: Mapping[str, int]
) -> str:
    payload = {
        "runner": EVAL_RUNNER_VERSION,
        "schema": EVAL_SCHEMA_VERSION,
        "model": config.generation_model,
        "base_url": config.generation_base_url,
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "embedding_revision": DEFAULT_JINA_REVISION,
        "transcription_model": DEFAULT_FUNASR_MODEL_ID,
        "device": arguments.device or "auto",
        "seed": arguments.seed,
        "gen_kwargs": arguments.gen_kwargs,
        "recall_limit": arguments.recall_limit,
        "batch_sizes": dict(sorted(batch_sizes.items())),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _cache_task(task: LoadedTask) -> str:
    return f"{task.spec.name}:{task.spec.adapter_version}:{task.evaluation_sha256}"


def _all_cached(cache: ResponseCache, tasks: Sequence[LoadedTask]) -> bool:
    return all(
        cache.get(_cache_task(task), unit.unit_id, question.question_id) is not None
        for task in tasks
        for unit in task.units
        for question in unit.questions
    )


def _cache_only_memory(_path: Path) -> _MemoryContext:
    raise RuntimeError("response cache was incomplete after the cache-only preflight")


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "source"


def _regressed(comparisons: Sequence[Mapping[str, object]], *, threshold: float) -> bool:
    for comparison in comparisons:
        interval = comparison.get("confidence_interval_95")
        if isinstance(interval, list) and len(interval) == 2 and float(interval[1]) < -threshold:
            return True
    return False


def _list_mode(parsed: argparse.Namespace) -> str | None:
    if parsed.list_tasks:
        return "all"
    return {
        "list": "all",
        "list_groups": "groups",
        "list_subtasks": "tasks",
        "list_tags": "tags",
    }.get(parsed.tasks)


def _limit_value(value: str) -> int | float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be numeric") from error
    if not math.isfinite(parsed) or parsed == 0 or parsed < -1:
        raise argparse.ArgumentTypeError("limit must be -1, a positive fraction, or a count")
    if parsed < 0 and parsed != -1:
        raise argparse.ArgumentTypeError("limit must be -1, a positive fraction, or a count")
    return int(parsed) if parsed == -1 or parsed.is_integer() else parsed


def _seed_values(value: str) -> tuple[int, int, int, int]:
    fields = tuple(part.strip() for part in value.split(","))
    if len(fields) == 1:
        fields *= 4
    elif len(fields) == 3:
        fields = (*fields, "1234")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "seed must be one, three, or four comma-separated integers"
        )
    try:
        seeds = tuple(int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "MindBridge eval requires numeric deterministic seeds"
        ) from error
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        raise argparse.ArgumentTypeError("seed values must be between 0 and 2^63 - 1")
    return cast(tuple[int, int, int, int], seeds)


def _generation_kwargs(value: str, seed: int) -> str:
    supplied: dict[str, str] = {}
    for item in (part.strip() for part in value.split(",") if part.strip()):
        key, separator, raw = item.partition("=")
        if not separator or not key.strip() or not raw.strip() or key.strip() in supplied:
            raise ValueError(f"invalid --gen_kwargs item: {item}")
        supplied[key.strip()] = raw.strip()
    unsupported = set(supplied) - {"temperature", "do_sample", "seed"}
    if unsupported:
        raise ValueError(
            "MindBridge eval supports deterministic --gen_kwargs only: temperature, do_sample, seed"
        )
    if "temperature" in supplied and float(supplied["temperature"]) != 0:
        raise ValueError("reproducible evaluation requires temperature=0")
    if "do_sample" in supplied and supplied["do_sample"].casefold() not in {"false", "0"}:
        raise ValueError("reproducible evaluation requires do_sample=false")
    if "seed" in supplied and int(supplied["seed"]) != seed:
        raise ValueError("--gen_kwargs seed must match the first --seed value")
    return f"temperature=0,do_sample=false,seed={seed}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 2**63:
        raise argparse.ArgumentTypeError("value must be between 0 and 2^63 - 1")
    return parsed


def _run_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise argparse.ArgumentTypeError(
            "run ID must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
