"""Run pinned MindBridge benchmarks with lmms-eval-style task selection."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Any, Protocol, cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from mindbridge import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_RECIPE,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    FaceAnalysis,
    FunASRTranscriber,
    IndexUnavailableError,
    JinaOmniEmbedder,
    MemoryConfig,
    MemoryType,
    MindBridgeConfig,
    MindBridgeError,
    Modality,
    OpenAIModels,
    SearchHit,
    resolve_memory_config,
)
from mindbridge._telemetry import (
    MODEL_MODULE,
    SPAN_KIND,
    mark_model_requests,
    record_unmetered_model_usage,
    traced_span,
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
    paired_comparison,
    parse_choice,
    summarize,
)
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_JUDGE_SPAN,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    EvaluationTelemetry,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import ModelConfig
from mindbridge.benchmarks.official_scorers import (
    SCORER_VERSION,
    JudgeMessage,
    JudgePlan,
    combine_judge_scores,
    finalize_scores,
    judge_model_is_official,
    judge_plan,
    local_scores,
    metric_is_official,
    official_judge_model,
    parse_judge_response,
    sample_primary_metric,
    scorer_protocol,
    task_primary_metric,
)
from mindbridge.benchmarks.prepare_media import _has_audio, prepare_task_media
from mindbridge.benchmarks.task_catalog import (
    DEFAULT_BENCHMARKS_ROOT,
    TASKS,
    expand,
    listing,
)
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError, DataDirectoryLock
from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    FaceBackend,
    GenerationBackend,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
    TranscriptionBackend,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_DIMENSION,
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
)
from mindbridge.models.openai_sdk import _model_usage, _record_usage_batch

EVAL_SCHEMA_VERSION = 8
EVAL_RUNNER_VERSION = "mindbridge_eval_official_v9"
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
_RESULTS_FILE = "results.json"
_SAMPLES_FILE = "samples.jsonl"
_EGOMEM_SUBMISSION_FILE = "egomemreason_submission.json"
_EGOMEM_SUBMISSION_COUNT = 500
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
    memory_config: Path | None
    judge_model_args: str
    judge_concurrency: int
    gen_kwargs: str
    num_fewshot: int
    use_cache: Path | None
    device: str | None
    device_lock: bool
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
class _JudgeConfig:
    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 3_600.0
    concurrency: int = 4

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("judge model and base URL must not be blank")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("judge timeout must be positive")
        if self.concurrency <= 0:
            raise ValueError("judge concurrency must be positive")


@dataclass(frozen=True, slots=True)
class FailureDetail:
    """Safe, stable benchmark failure diagnostics."""

    source_id: str | None
    code: str
    reason: str | None
    stage: str | None
    cause_type: str | None

    def json(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "code": self.code,
            "reason": self.reason,
            "stage": self.stage,
            "cause_type": self.cause_type,
        }


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
    abstained: bool = False
    abstention_reason: str | None = None
    ingest_failures: tuple[FailureDetail, ...] = ()
    error_reason: str | None = None
    error_stage: str | None = None
    error_cause_type: str | None = None
    cached: bool = False
    prompt: tuple[str, ...] | None = None
    references: tuple[str, ...] | None = None
    evidence: tuple[EvidenceInterval, ...] = ()
    ref_at_300: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    scorer_protocol: str | None = None
    scorer_details: Mapping[str, str] = field(default_factory=dict)
    scorer_error: str | None = None
    judge_model: str | None = None
    judge_response: str | None = None
    judge_cached: bool = False

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
            "metrics": dict(self.metrics),
            "scorer_protocol": self.scorer_protocol,
            "scorer_details": dict(self.scorer_details),
            "scorer_error": self.scorer_error,
            "judge_model": self.judge_model,
            "judge_cached": self.judge_cached,
            "ingest_failure_count": self.ingest_failure_count,
            "ingest_failures": tuple(item.json() for item in self.ingest_failures),
            "error_code": self.error_code,
            "error_reason": self.error_reason,
            "error_stage": self.error_stage,
            "error_cause_type": self.error_cause_type,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "cached": self.cached,
            "metadata": dict(self.metadata),
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.references is not None:
            payload["references"] = self.references
        if self.judge_response is not None:
            payload["judge_response"] = self.judge_response
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
    abstained: bool = False
    abstention_reason: str | None = None
    cached: bool = False
    error: BaseException | None = None


class _BorrowedBackend:
    """Forward a shared backend while making per-store ``close`` a no-op."""

    def __init__(self, backend: object) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> Iterator[str]:
        return cast(Iterator[str], cast(Any, self._backend).stream_answer(question, hits))

    def close(self) -> None:
        return None


class _BorrowedGenerationBackend(_BorrowedBackend):
    """Lend one answerer without transferring ownership to a per-unit memory."""

    @property
    def generation_capabilities(self) -> frozenset[Modality]:
        return cast(GenerationBackend, self._backend).generation_capabilities

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
        return cast(GenerationBackend, self._backend).answer(question, hits)


class _BorrowedFaceBackend(_BorrowedBackend):
    """Lend one face analyzer while preserving the runtime-checkable protocol."""

    @property
    def face_capabilities(self) -> frozenset[Modality]:
        return cast(FaceBackend, self._backend).face_capabilities

    @property
    def face_model(self) -> str:
        return cast(FaceBackend, self._backend).face_model

    @property
    def face_space(self) -> str:
        return cast(FaceBackend, self._backend).face_space

    @property
    def face_analysis_space(self) -> str:
        return cast(FaceBackend, self._backend).face_analysis_space

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return cast(FaceBackend, self._backend).analyze(assets)


class _BorrowedSpeechBackend(_BorrowedBackend):
    """Skip visual-only videos before lending the shared speech backend."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]:
        return cast(SpeechBackend, self._backend).transcription_capabilities

    @property
    def transcription_model(self) -> str:
        return cast(SpeechBackend, self._backend).transcription_model

    @property
    def transcription_space(self) -> str:
        return cast(SpeechBackend, self._backend).transcription_space

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        selected = tuple(
            asset.modality is not Modality.VIDEO or asset.path is None or _has_audio(asset.path)
            for asset in assets
        )
        audible = tuple(asset for asset, include in zip(assets, selected, strict=True) if include)
        if audible:
            generated = cast(SpeechBackend, self._backend).analyze(audible)
        else:
            record_unmetered_model_usage(request_count=0)
            generated = ()
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
        gen_kwargs: str = "",
        memory_config: MindBridgeConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.config = config
        self._resolved_config = None
        self._tracer = trace.get_tracer("mindbridge.benchmarks.eval") if tracer is None else tracer
        if memory_config is not None:
            if memory_config.generation is not None:
                memory_config = memory_config.model_copy(
                    update={
                        "generation": memory_config.generation.model_copy(
                            update={"modalities": config.generation_capabilities}
                        )
                    }
                )
            resolved = resolve_memory_config(memory_config)
            self._resolved_config = resolved
            plugins = resolved.plugins
            self._embedder = cast(EmbeddingBackend, _BorrowedBackend(plugins.embedder))
            self._answerer = (
                None
                if plugins.answerer is None
                else cast(GenerationBackend, _BorrowedGenerationBackend(plugins.answerer))
            )
            self._transcriber = (
                None
                if plugins.transcriber is None
                else cast(
                    SpeechBackend | TranscriptionBackend,
                    (
                        _BorrowedSpeechBackend(plugins.transcriber)
                        if isinstance(plugins.transcriber, SpeechBackend)
                        else _BorrowedBackend(plugins.transcriber)
                    ),
                )
            )
            self._face_analyzer = (
                None
                if plugins.face_analyzer is None
                else cast(FaceBackend, _BorrowedFaceBackend(plugins.face_analyzer))
            )
            self._settings = resolved.settings
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("benchmark execution requires mindbridge[openai]") from None
        self.client = OpenAI(
            api_key=config.generation_api_key,
            base_url=config.generation_base_url,
            timeout=config.timeout_seconds,
        )
        generation_options = dict(
            item.split("=", 1) for item in gen_kwargs.split(",") if "=" in item
        )
        enable_thinking = generation_options.get("enable_thinking")
        self.models = OpenAIModels(
            generation_client=self.client,
            generation_model=config.generation_model,
            generation_capabilities=config.generation_capabilities,
            generation_seed=seed,
            generation_temperature=0.0,
            generation_max_tokens=(
                None
                if "max_tokens" not in generation_options
                else int(generation_options["max_tokens"])
            ),
            generation_min_video_seconds=config.generation_min_video_seconds,
            generation_extra_body=(
                None
                if enable_thinking is None
                else {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking == "true",
                    }
                }
            ),
        )
        self.embedder = JinaOmniEmbedder(device=device, batch_size=batch_size)
        self.embedder.embed((ModelInput(text="MindBridge benchmark warmup"),), task=EmbedTask.QUERY)
        self.transcriber = (
            FunASRTranscriber(
                recipe=replace(
                    DEFAULT_FUNASR_RECIPE,
                    speaker_model=None,
                    speaker_revision=None,
                ),
                device=device or "auto",
            )
            if needs_speech
            else None
        )
        self._answerer = cast(
            GenerationBackend,
            _BorrowedGenerationBackend(self.models),
        )
        self._embedder = cast(EmbeddingBackend, _BorrowedBackend(self.embedder))
        self._transcriber = (
            cast(SpeechBackend, _BorrowedSpeechBackend(self.transcriber))
            if self.transcriber is not None
            else None
        )
        self._face_analyzer = None
        self._settings = MemoryConfig(index_speech=self._transcriber is not None)

    def memory(self, data_dir: Path) -> AsyncMemory:
        # Forwarded from the dataclass rather than field by field. The hand-written list silently
        # dropped every setting added after it was written, which does not fail anything: the
        # evaluation simply measures the default policy while reporting the configured one.
        # `MemoryPlugins` cannot be used here because the shared-backend proxies are structural
        # and its runtime protocol check reads attributes statically.
        policy = {entry.name: getattr(self._settings, entry.name) for entry in fields(MemoryConfig)}
        return AsyncMemory(
            data_dir,
            embedder=self._embedder,
            answerer=self._answerer,
            transcriber=self._transcriber,
            face_analyzer=self._face_analyzer,
            tracer=self._tracer,
            **policy,
        )

    def close(self) -> None:
        if self._resolved_config is not None:
            self._resolved_config.close()
            return
        resources = (self.transcriber, self.embedder, self.client)
        for resource in resources:
            if resource is not None:
                with suppress(Exception):
                    resource.close()


def main(  # noqa: C901 - offline gates and evaluation share one CLI entry point
    argv: Sequence[str] | None = None, *, prog: str | None = None
) -> int:
    """Parse one reproducible evaluation sweep and write its artifacts."""
    parser = _build_parser(prog)
    parsed = parser.parse_args(argv)
    list_mode = _list_mode(parsed)
    if list_mode is not None:
        print(listing(parsed.benchmarks_root.expanduser().resolve(), list_mode))
        return 0
    arguments = _arguments(parser, parsed)
    if parsed.check_integrity:
        manifest, manifest_directory = load_media_manifest(arguments.media_manifest)
        loaded = _load_tasks(arguments, manifest, manifest_directory)
        print(
            _json_bytes(
                {
                    "status": "ok",
                    "tasks": [
                        {
                            "task": task.spec.name,
                            "dataset_sha256": task.dataset_sha256,
                            "evaluation_sha256": task.evaluation_sha256,
                            "unit_count": len(task.units),
                            "question_count": sum(len(unit.questions) for unit in task.units),
                        }
                        for task in loaded
                    ],
                },
                pretty=True,
            ).decode(),
            end="",
        )
        return 0
    try:
        memory_config = _load_memory_config(arguments.memory_config)
        base_config = _model_config(
            arguments.model,
            arguments.model_args,
            memory_config=memory_config,
        )
        judge_config = _judge_config(base_config, arguments)
    except ValueError as error:
        parser.error(str(error))
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
    loaded = _load_tasks(arguments, manifest, manifest_directory)
    config = _evaluation_config(base_config, loaded)
    memory_config = _evaluation_memory_config(memory_config, config, arguments)
    batch_sizes = {task.spec.name: _batch_size(arguments, task) for task in loaded}
    samples, duration, performance = _execute(
        loaded,
        arguments,
        config,
        judge_config,
        batch_sizes,
        memory_config=memory_config,
    )
    submission_bytes, submission_status = _egomem_submission(
        samples,
        requested="egomemreason" in arguments.tasks,
        allow_partial=arguments.offset > 0 or arguments.limit not in (None, -1),
    )
    results = _results(
        arguments,
        config,
        judge_config,
        loaded,
        samples,
        duration,
        batch_sizes,
        submission_status,
        performance,
        memory_config=memory_config,
    )
    comparisons = _comparisons(arguments, loaded, samples)
    if comparisons:
        results["comparisons"] = comparisons
    _write_artifacts(arguments, samples, results, submission_bytes)
    _announce_submission(arguments, submission_status)
    if not arguments.quiet:
        print(_table(results))
    has_errors = any(
        sample.error_code is not None or sample.ingest_failure_count for sample in samples
    )
    regressed = arguments.fail_on_regression and _regressed(
        comparisons, threshold=arguments.regression_threshold
    )
    submission_invalid = submission_status is not None and submission_status["status"] == "invalid"
    return int(has_errors or regressed or submission_invalid)


def _load_tasks(
    arguments: _Arguments,
    manifest: Mapping[str, object] | None,
    manifest_directory: Path | None,
) -> tuple[LoadedTask, ...]:
    return tuple(
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


def _announce_submission(arguments: _Arguments, status: Mapping[str, object] | None) -> None:
    if status is None or arguments.quiet:
        return
    if status["status"] == "ready":
        _announce(f"wrote {arguments.output_path / _EGOMEM_SUBMISSION_FILE}")
        return
    _announce(f"EgoMemReason submission {status['status']}: {status['reason']}")


@contextmanager
def _benchmark_device_lock(
    device: str | None,
    *,
    enabled: bool,
    quiet: bool,
    lock_root: Path | None = None,
) -> Iterator[None]:
    normalized = (device or "auto").strip().lower()
    if not enabled or normalized == "cpu":
        yield
        return
    index = _cuda_logical_index(normalized)
    identity = _physical_cuda_identity(normalized)
    if identity is None:
        yield
        return
    if lock_root is None:
        owner = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:12]
        lock_root = Path(os.environ.get("XDG_RUNTIME_DIR", gettempdir())) / (
            f"mindbridge-benchmark-{owner}"
        )
    device_key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    # ponytail: one local model process per CUDA device; use a VRAM budget only when
    # concurrent model pools demonstrate a safe, repeatable throughput gain.
    announced = False
    while True:
        try:
            lock = DataDirectoryLock(lock_root / f"cuda-{device_key}")
            break
        except DataDirectoryInUseError:
            if not quiet and not announced:
                _announce(f"waiting for local CUDA device {index}")
                announced = True
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.close()


def _cuda_logical_index(device: str) -> int:
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    return int(match.group(1)) if match is not None and match.group(1) is not None else 0


def _physical_cuda_identity(device: str) -> str | None:
    logical_index = _cuda_logical_index(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(logical_index)
    else:
        exposed = tuple(value.strip() for value in visible.split(",") if value.strip())
        if logical_index >= len(exposed) or exposed[logical_index] == "-1":
            return None
        selected = exposed[logical_index]
    uuids = _nvidia_device_uuids()
    if selected.isdecimal():
        physical_index = int(selected)
        return uuids.get(physical_index, f"index:{physical_index}").casefold()
    normalized = selected.casefold()
    return next(
        (
            uuid.casefold()
            for uuid in uuids.values()
            if uuid.casefold().startswith(normalized) or normalized.startswith(uuid.casefold())
        ),
        normalized,
    )


def _nvidia_device_uuids() -> dict[int, str]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode:
        return {}
    devices = {}
    for line in result.stdout.splitlines():
        index, separator, uuid = line.partition(",")
        if separator and index.strip().isdecimal() and uuid.strip():
            devices[int(index.strip())] = uuid.strip()
    return devices


def _release_device_memory(device: str | None) -> None:
    if (device or "auto").strip().lower() == "cpu":
        return
    gc.collect()
    with suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _execute(
    loaded: Sequence[LoadedTask],
    arguments: _Arguments,
    config: ModelConfig,
    judge_config: _JudgeConfig,
    batch_sizes: Mapping[str, int],
    *,
    memory_config: MindBridgeConfig | None,
) -> tuple[tuple[SampleResult, ...], float, Mapping[str, Mapping[str, object]]]:
    needs_speech = any(
        isinstance(atom, Path)
        and _MODALITY_BY_SUFFIX.get(atom.suffix.casefold()) in {Modality.AUDIO, Modality.VIDEO}
        for task in loaded
        for unit in task.units
        for item in unit.memories
        for atom in item.content
    )
    started = time.perf_counter()
    telemetry = EvaluationTelemetry()
    try:
        response_cache = (
            None
            if arguments.use_cache is None
            else ResponseCache(
                arguments.use_cache,
                arguments.run_id,
                _cache_namespace(
                    arguments,
                    config,
                    batch_sizes,
                    memory_config=memory_config,
                ),
            )
        )
        pool: _BackendPool | None = None
        memory_factory: MemoryFactory
        all_cached = response_cache is not None and _all_cached(response_cache, loaded)
        devices = _evaluation_devices(arguments.device, memory_config)
        with ExitStack() as device_locks:
            for device in devices:
                device_locks.enter_context(
                    _benchmark_device_lock(
                        device,
                        enabled=arguments.device_lock and not all_cached,
                        quiet=arguments.quiet,
                    )
                )
            try:
                if all_cached:
                    memory_factory = _cache_only_memory
                else:
                    pool = _BackendPool(
                        config,
                        device=arguments.device,
                        batch_size=max(batch_sizes.values()),
                        needs_speech=needs_speech,
                        seed=arguments.seed,
                        gen_kwargs=arguments.gen_kwargs,
                        memory_config=memory_config,
                        tracer=telemetry.tracer,
                    )
                    memory_factory = pool.memory
                samples = asyncio.run(
                    _run_all(
                        loaded,
                        arguments,
                        batch_sizes=batch_sizes,
                        memory_factory=memory_factory,
                        response_cache=response_cache,
                        tracer=telemetry.tracer,
                    )
                )
            finally:
                if pool is not None:
                    closing = pool
                    pool = None
                    memory_factory = _cache_only_memory
                    closing.close()
                    del closing
                    for device in devices:
                        _release_device_memory(device)
                if response_cache is not None:
                    response_cache.close()
        if not arguments.predict_only:
            samples = asyncio.run(
                _apply_judges(
                    loaded,
                    samples,
                    arguments=arguments,
                    config=judge_config,
                    tracer=telemetry.tracer,
                )
            )
        performance = {
            task.spec.name: telemetry.result(
                task.spec.name,
                question_count=sum(len(unit.questions) for unit in task.units),
            )
            for task in loaded
        }
        return samples, time.perf_counter() - started, performance
    finally:
        telemetry.close()


async def _run_all(
    tasks: Sequence[LoadedTask],
    arguments: _Arguments,
    *,
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
    tracer: Tracer,
) -> tuple[SampleResult, ...]:
    results: list[SampleResult] = []
    for task in tasks:
        if not arguments.quiet:
            print(
                f"mindbridge-bench eval: running {task.spec.name} ({len(task.units)} units)",
                file=sys.stderr,
            )
        with traced_span(
            tracer,
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: task.spec.name, SPAN_KIND: "benchmark"},
        ):
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
    request_semaphore = asyncio.Semaphore(request_concurrency)
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
                request_semaphore=request_semaphore,
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
    request_semaphore: asyncio.Semaphore,
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
    ingest_failure_details: list[FailureDetail] = []
    try:
        async with memory_factory(data_dir) as memory:
            for cutoff in cutoffs:
                boundary = math.inf if cutoff is None else cutoff
                end = pending
                while end < len(memories) and _memory_end(memories[end]) <= boundary:
                    end += 1
                ingest_failures += await _ingest(
                    memory,
                    memories[pending:end],
                    batch_size=batch_size,
                    on_failure=ingest_failure_details.append,
                )
                pending = end

                def cache_completed(
                    question: EvalQuestion,
                    outcome: _AnswerOutcome,
                    failure_count: int = ingest_failures,
                ) -> None:
                    _cache_outcome(
                        response_cache,
                        task,
                        unit,
                        question,
                        outcome,
                        failure_count,
                    )

                answered = await _answer_many(
                    memory,
                    questions_by_cutoff[cutoff],
                    request_concurrency=request_concurrency,
                    request_semaphore=request_semaphore,
                    recall_limit=recall_limit,
                    on_answer=cache_completed,
                )
                for question, outcome in zip(questions_by_cutoff[cutoff], answered, strict=True):
                    results[question.question_id] = _sample(
                        task,
                        unit,
                        question,
                        outcome,
                        ingest_failures=ingest_failures,
                        ingest_failure_details=tuple(ingest_failure_details),
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
                    ingest_failure_details=tuple(ingest_failure_details),
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
                abstained=answer.abstained,
                abstention_reason=answer.abstention_reason,
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
            outcome.abstained,
            outcome.abstention_reason,
        ),
    )


async def _ingest(
    memory: AsyncMemory,
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
    on_failure: Callable[[FailureDetail], None] | None = None,
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
        except IndexUnavailableError:
            raise
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
        except IndexUnavailableError:
            raise
        except Exception as error:
            if on_failure is not None:
                on_failure(_failure_detail(error, source_id=chunk[0].source_id))
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
    request_semaphore: asyncio.Semaphore | None = None,
    recall_limit: int,
    on_answer: Callable[[EvalQuestion, _AnswerOutcome], None] | None = None,
) -> tuple[_AnswerOutcome | BaseException, ...]:
    semaphore = request_semaphore or asyncio.Semaphore(request_concurrency)

    async def answer(question: EvalQuestion) -> _AnswerOutcome:
        async with semaphore:
            started = time.perf_counter()
            content = _content(question.content)
            try:
                result = await memory.ask(
                    content,
                    limit=recall_limit,
                    reference_at=question.reference_at,
                )
            except Exception as error:
                hits = await memory.search(
                    content,
                    limit=recall_limit,
                    reference_at=question.reference_at,
                )
                outcome = _AnswerOutcome(
                    "",
                    (time.perf_counter() - started) * 1_000,
                    max((hit.score for hit in hits), default=0.0),
                    tuple(hit.id for hit in hits),
                    tuple(_evidence(hit) for hit in hits),
                    error=error,
                )
            else:
                outcome = _AnswerOutcome(
                    result.answer,
                    (time.perf_counter() - started) * 1_000,
                    max((hit.score for hit in result.hits), default=0.0),
                    tuple(hit.id for hit in result.hits),
                    tuple(_evidence(hit) for hit in result.hits),
                    abstained=result.abstained,
                    abstention_reason=(
                        None if result.abstention_reason is None else result.abstention_reason.value
                    ),
                )
        if on_answer is not None:
            on_answer(question, outcome)
        return outcome

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
    ingest_failure_details: tuple[FailureDetail, ...] = (),
    predict_only: bool,
    log_samples: bool,
) -> SampleResult:
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    error_code: str | None
    error_detail: FailureDetail | None
    if isinstance(outcome, BaseException):
        prediction, latency_ms, confidence, memory_ids, evidence, cached = (
            "",
            0.0,
            0.0,
            (),
            (),
            False,
        )
        error_detail = _failure_detail(outcome)
        error_code = error_detail.code
        abstained = False
        abstention_reason = None
    else:
        prediction = outcome.prediction
        latency_ms = outcome.latency_ms
        confidence = outcome.confidence
        memory_ids = outcome.memory_ids
        evidence = outcome.evidence
        cached = outcome.cached
        error_detail = None if outcome.error is None else _failure_detail(outcome.error)
        error_code = None if error_detail is None else error_detail.code
        abstained = outcome.abstained
        abstention_reason = outcome.abstention_reason
    choices = tuple(
        str(value) for value in cast(Sequence[object], question.metadata.get("choices", ()))
    )
    parsed = _parsed_choice(task.spec.name, prediction, choices)
    metrics, score, matched = _score(
        task.spec.name,
        question,
        prediction,
        parsed,
        evidence,
        predict_only=predict_only,
    )
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
        ingest_failures=ingest_failure_details,
        error_code=error_code,
        error_reason=None if error_detail is None else error_detail.reason,
        error_stage=None if error_detail is None else error_detail.stage,
        error_cause_type=None if error_detail is None else error_detail.cause_type,
        abstained=abstained,
        abstention_reason=abstention_reason,
        cached=cached,
        metadata=question.metadata,
        prompt=tuple(str(part) for part in question.content) if log_samples else None,
        references=question.references if log_samples else None,
        evidence=evidence,
        ref_at_300=_reference_grounding(task, unit, question, evidence),
        metrics=metrics,
        scorer_protocol=scorer_protocol(task.spec.name),
    )


def _score(
    task_name: str,
    question: EvalQuestion,
    prediction: str,
    parsed_choice: str | None,
    evidence: Sequence[EvidenceInterval],
    *,
    predict_only: bool,
) -> tuple[Mapping[str, float], float | None, float | None]:
    if predict_only:
        return {}, None, None
    metrics = local_scores(
        task_name,
        score_kind=question.score_kind,
        prediction=prediction,
        parsed_choice=parsed_choice,
        expected_choice=question.expected_choice,
        references=question.references,
        question=question.source_question,
        metadata=question.metadata,
        evidence_source_ids=tuple(item.source_id or "" for item in evidence),
    )
    score = metrics.get(sample_primary_metric(task_name), metrics.get("token_f1"))
    return metrics, score, metrics.get("exact_match")


async def _apply_judges(
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    *,
    arguments: _Arguments,
    config: _JudgeConfig,
    tracer: Tracer | None = None,
) -> tuple[SampleResult, ...]:
    selected_tracer = trace.get_tracer("mindbridge.benchmarks.eval") if tracer is None else tracer
    questions = {
        (task.spec.name, unit.unit_id, question.question_id): question
        for task in tasks
        for unit in task.units
        for question in unit.questions
    }
    planned: dict[str, JudgePlan] = {}
    for sample in samples:
        if sample.error_code is not None or sample.ingest_failure_count:
            continue
        question = questions[(sample.task, sample.unit_id, sample.question_id)]
        plan = judge_plan(
            sample.task,
            question=question.source_question,
            references=question.references,
            prediction=sample.prediction,
            metadata=question.metadata,
        )
        if plan is not None:
            planned[sample.sample_id] = plan
    if not planned:
        return tuple(samples)
    if not arguments.quiet:
        print(
            f"mindbridge-bench eval: judging {len(planned)} answers with {config.model}",
            file=sys.stderr,
        )
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("official LLM scorers require mindbridge[openai]") from None
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    cache = (
        None
        if arguments.use_cache is None
        else ResponseCache(
            arguments.use_cache,
            arguments.run_id,
            _judge_cache_namespace(config),
        )
    )
    semaphore = asyncio.Semaphore(config.concurrency)
    try:
        judged = await asyncio.gather(
            *(
                _judge_sample(
                    sample,
                    planned.get(sample.sample_id),
                    client=client,
                    cache=cache,
                    semaphore=semaphore,
                    config=config,
                    log_samples=arguments.log_samples,
                    tracer=selected_tracer,
                )
                for sample in samples
            )
        )
    finally:
        await client.close()
        if cache is not None:
            cache.close()
    return tuple(judged)


async def _judge_sample(
    sample: SampleResult,
    plan: JudgePlan | None,
    *,
    client: AsyncOpenAI,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    log_samples: bool,
    tracer: Tracer,
) -> SampleResult:
    if plan is None:
        return sample
    try:
        outcomes = tuple(
            [
                await _traced_judge_call(
                    client,
                    messages,
                    sample=sample,
                    plan=plan,
                    call_index=index,
                    cache=cache,
                    semaphore=semaphore,
                    config=config,
                    tracer=tracer,
                )
                for index, messages in enumerate(plan.calls)
            ]
        )
        scores = combine_judge_scores(plan, tuple(item[0] for item in outcomes))
        metrics = finalize_scores(sample.task, {**sample.metrics, **scores})
        responses = tuple(item[1] for item in outcomes)
        return replace(
            sample,
            score=metrics.get(sample_primary_metric(sample.task)),
            exact_match=metrics.get("exact_match"),
            metrics=metrics,
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
            judge_response=(json.dumps(responses, ensure_ascii=False) if log_samples else None),
            judge_cached=all(item[2] for item in outcomes),
        )
    except Exception as error:
        message = " ".join(str(error).split())[:500]
        return replace(
            sample,
            score=sample.metrics.get(sample_primary_metric(sample.task)),
            error_code=sample.error_code or "JudgeError",
            scorer_error=f"{type(error).__name__}: {message}",
            scorer_details={**sample.scorer_details, **plan.details},
            judge_model=config.model,
        )


async def _traced_judge_call(
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
    tracer: Tracer,
) -> tuple[Mapping[str, float], str, bool]:
    with traced_span(
        tracer,
        BENCHMARK_JUDGE_SPAN,
        attributes={
            BENCHMARK_TASK: sample.task,
            SPAN_KIND: "model",
            MODEL_MODULE: "judge",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": config.model,
        },
    ):
        return await _judge_call(
            client,
            messages,
            sample=sample,
            plan=plan,
            call_index=call_index,
            cache=cache,
            semaphore=semaphore,
            config=config,
        )


async def _judge_call(  # noqa: C901 - retry and provider fallback belong in one request path
    client: AsyncOpenAI,
    messages: Sequence[JudgeMessage],
    *,
    sample: SampleResult,
    plan: JudgePlan,
    call_index: int,
    cache: ResponseCache | None,
    semaphore: asyncio.Semaphore,
    config: _JudgeConfig,
) -> tuple[Mapping[str, float], str, bool]:
    key = hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "model": config.model,
                "protocol": plan.protocol,
                "call": call_index,
                "messages": tuple((message.role, message.content) for message in messages),
            }
        )
    ).hexdigest()
    cache_task = f"judge:{plan.protocol}:{config.model}"
    if cache is not None:
        cached = cache.get(cache_task, sample.unit_id, key)
        if cached is not None:
            return parse_judge_response(plan, cached.prediction), cached.prediction, True
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": message.role, "content": message.content} for message in messages],
        "temperature": 0.0,
    }
    if plan.max_tokens is not None:
        request["max_tokens"] = plan.max_tokens
    extra_body = dict(plan.extra_body or {})
    model_key = re.sub(r"[^a-z0-9]+", "", config.model.casefold())
    if "qwen3" in model_key and not (
        "enable_thinking" in extra_body and judge_model_is_official(sample.task, config.model)
    ):
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    if extra_body:
        request["extra_body"] = extra_body
    use_responses_api = plan.parser == "atm" and judge_model_is_official(sample.task, config.model)
    last_error: Exception | None = None
    usages = []
    attempted = 0
    try:
        for attempt in range(3):
            try:
                async with semaphore:
                    attempted = attempt + 1
                    mark_model_requests(attempted)
                    if use_responses_api:
                        response = await client.responses.create(
                            model=config.model,
                            input="\n".join(message.content for message in messages),
                            max_output_tokens=plan.max_tokens,
                            reasoning={"effort": "minimal"},
                        )
                        text = str(response.output_text or "").strip()
                    else:
                        response = await client.chat.completions.create(**request)
                        text = str(response.choices[0].message.content or "").strip()
                usages.append(
                    _model_usage(
                        response,
                        input_modalities=frozenset({Modality.TEXT}),
                        output_modalities=frozenset({Modality.TEXT}),
                    )
                )
                scores = parse_judge_response(plan, text)
                if cache is not None:
                    cache.put(cache_task, sample.unit_id, key, CachedAnswer(text, 0.0, ()))
                return scores, text, False
            except Exception as error:
                last_error = error
                if "extra_body" in request and _unsupported_extra_body(error):
                    request.pop("extra_body")
                    continue
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        if last_error is None:
            raise RuntimeError("judge call failed without an exception")
        raise last_error
    finally:
        _record_usage_batch(usages, request_count=attempted)


def _unsupported_extra_body(error: Exception) -> bool:
    message = str(error).casefold()
    field = "enable_thinking" in message or "chat_template_kwargs" in message
    marker = any(
        value in message
        for value in (
            "unknown",
            "unsupported",
            "unexpected",
            "unrecognized",
            "not allowed",
            "invalid parameter",
        )
    )
    return field and marker


def _judge_cache_namespace(config: _JudgeConfig) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "version": SCORER_VERSION,
                "model": config.model,
                "base_url": config.base_url,
                "temperature": 0.0,
            }
        )
    ).hexdigest()


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
            last = int((end - 1e-9) // bucket_size)
            values.update(range(first, last + 1))
        return values

    predicted_buckets = buckets(predicted)
    expected_buckets = buckets(expected)
    if not predicted_buckets and not expected_buckets:
        return 0.0
    return len(predicted_buckets & expected_buckets) / len(predicted_buckets | expected_buckets)


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
    judge_config: _JudgeConfig,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    duration_seconds: float,
    batch_sizes: Mapping[str, int],
    submission_status: Mapping[str, object] | None,
    performance: Mapping[str, Mapping[str, object]],
    *,
    memory_config: MindBridgeConfig | None = None,
) -> dict[str, object]:
    task_rows = []
    for task in tasks:
        selected = tuple(sample for sample in samples if sample.task == task.spec.name)
        metrics = _metrics(task, selected, arguments)
        if task.spec.name == "egomemreason" and submission_status is not None:
            metrics["submission"] = dict(submission_status)
            if submission_status["status"] == "invalid":
                metrics["score_valid"] = False
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
                "performance": dict(performance.get(task.spec.name, {})),
                **metrics,
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
            or (submission_status is not None and submission_status["status"] == "invalid")
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
        "cached_judge_count": sum(sample.judge_cached for sample in samples),
        "abstentions": _abstentions(samples),
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
        "model": _model_result(arguments, config, memory_config),
        "judge": {
            "model": judge_config.model,
            "base_url": judge_config.base_url,
            "sampling": "benchmark_protocol",
            "concurrency": judge_config.concurrency,
            "timeout_seconds": judge_config.timeout_seconds,
        },
        "environment": {
            "mindbridge_version": _version("mindbridge"),
            "zvec_version": _version("zvec"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "tasks": task_rows,
    }


def _model_result(
    arguments: _Arguments,
    config: ModelConfig,
    memory_config: MindBridgeConfig | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "adapter": arguments.model,
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "embedding_revision": DEFAULT_JINA_REVISION,
        "embedding_dimension": DEFAULT_JINA_DIMENSION,
        "embedding_warmup": {"count": 1, "task": EmbedTask.QUERY.value},
        "device": arguments.device or "auto",
        "generation_model": config.generation_model,
        "generation_base_url": config.generation_base_url,
        "generation_modalities": sorted(
            modality.value for modality in config.generation_capabilities
        ),
        "generation_seed": arguments.seed,
        "generation_temperature": 0.0,
        "generation_kwargs": arguments.gen_kwargs,
        "generation_min_video_seconds": config.generation_min_video_seconds,
        "transcription_model": DEFAULT_FUNASR_MODEL_ID,
        "timeout_seconds": config.timeout_seconds,
    }
    if memory_config is None:
        return result
    embedding = memory_config.embedding.model_dump(mode="json")
    speech = None if memory_config.speech is None else memory_config.speech.model_dump(mode="json")
    provider = str(embedding["provider"])
    result.update(
        embedding_model=(DEFAULT_JINA_MODEL_ID if provider == "jina-omni" else embedding["model"]),
        embedding_revision=(
            DEFAULT_JINA_REVISION if provider == "jina-omni" else embedding.get("revision")
        ),
        embedding_dimension=embedding.get("dimension"),
        device=_configured_device_label(arguments.device, memory_config),
        transcription_model=(
            None
            if speech is None
            else DEFAULT_FUNASR_MODEL_ID
            if speech["provider"] == "funasr"
            else speech["model"]
        ),
        memory_config=_memory_config_payload(memory_config),
    )
    return result


def _configured_device_label(explicit: str | None, config: MindBridgeConfig) -> str:
    if explicit is not None:
        return explicit
    devices = []
    if config.embedding.provider in {"jina-omni", "sentence-transformers"}:
        devices.append(config.embedding.device or "auto")
    if config.speech is not None and config.speech.provider == "funasr":
        devices.append(config.speech.device)
    return ",".join(dict.fromkeys(devices)) or "remote"


def _memory_config_payload(config: MindBridgeConfig) -> dict[str, object]:
    return cast(
        dict[str, object],
        config.model_dump(mode="json", exclude={"data_dir"}),
    )


def _metrics(
    task: LoadedTask,
    samples: Sequence[SampleResult],
    arguments: _Arguments,
) -> dict[str, object]:
    seed = _task_seed(arguments.seed, task.spec.name)
    primary_name = task_primary_metric(task.spec.name)
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
    metric_rows: dict[str, dict[str, object]] = {}
    metric_names = sorted({name for sample in samples for name in sample.metrics})
    judge_models = sorted(
        {sample.judge_model for sample in samples if sample.judge_model is not None}
    )
    judge_model = judge_models[0] if len(judge_models) == 1 else ""
    for metric_name in metric_names:
        metric_values = tuple(
            ScoredValue(sample.sample_id, sample.unit_id, sample.metrics[metric_name])
            for sample in samples
            if metric_name in sample.metrics
        )
        uses_judge = any(
            sample.judge_model is not None and metric_name in sample.metrics for sample in samples
        )
        clamp = (0.0, 5.0) if metric_name == "judge_score_0_5" else (0.0, 1.0)
        metric_rows[metric_name] = {
            "official_metric": metric_is_official(
                task.spec.name,
                metric_name,
                judge_model,
                uses_judge=uses_judge,
            ),
            **summarize(
                metric_values,
                seed=_task_seed(seed, metric_name),
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=clamp,
            ),
        }
    latencies = sorted(sample.latency_ms for sample in samples if sample.latency_ms > 0)
    error_count = sum(sample.error_code is not None for sample in samples)
    ingest_failure_count = sum(
        max(sample.ingest_failure_count for sample in samples if sample.unit_id == unit_id)
        for unit_id in {sample.unit_id for sample in samples}
    )
    result: dict[str, object] = {
        "primary_metric": primary_name,
        "official_metric": bool(
            primary_name in metric_rows and metric_rows[primary_name]["official_metric"]
        ),
        "scorer_protocol": scorer_protocol(task.spec.name),
        "official_judge_model": official_judge_model(task.spec.name),
        "judge_model": judge_models[0] if len(judge_models) == 1 else judge_models or None,
        "judge_model_official": (
            None
            if not judge_models or official_judge_model(task.spec.name) is None
            else len(judge_models) == 1 and judge_model_is_official(task.spec.name, judge_models[0])
        ),
        "score": primary,
        "score_valid": error_count == 0 and ingest_failure_count == 0,
        "metrics": metric_rows,
        "exact_match": metric_rows.get("exact_match"),
        "question_count": len(samples),
        "error_count": error_count,
        "ingest_failure_count": ingest_failure_count,
        "abstentions": _abstentions(samples),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "breakdowns": _metric_breakdowns(task, samples, arguments),
    }
    if task.spec.name == "video-mme" and scored:
        result["video_mme"] = _video_mme_metrics(
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
        result.update(
            {
                "primary_metric": "rating",
                "official_metric": True,
                "score": rating,
                "question_accuracy": metric_rows.get("question_accuracy", primary),
            }
        )
        metric_rows["rating"] = {"official_metric": True, **rating}
    if task.spec.name == "supermemory-vqa" and scored:
        result["answerability"] = {"official_metric": True, **_answerability(samples)}
        result["unavailable_metrics"] = {
            "qa_mrr": "answer-option scores are not exposed by the MindBridge answer backend"
        }
    if task.spec.name == "egomemreason":
        result["unavailable_metrics"] = {
            "accuracy": "the public release has no answer key; official server scoring is required"
        }
    reference_scores = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.ref_at_300)
        for sample in samples
        if sample.ref_at_300 is not None
    )
    if reference_scores:
        ref_summary = {
            "official_metric": True,
            **summarize(
                reference_scores,
                seed=seed,
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=(0.0, 1.0),
            ),
        }
        result["ref_at_300"] = ref_summary
        metric_rows["ref_at_300"] = ref_summary
    return result


def _abstentions(samples: Sequence[SampleResult]) -> dict[str, object]:
    count = sum(sample.abstained for sample in samples)
    reasons = {
        reason: sum(sample.abstention_reason == reason for sample in samples)
        for reason in sorted(
            {sample.abstention_reason for sample in samples if sample.abstention_reason is not None}
        )
    }
    return {
        "count": count,
        "rate": 0.0 if not samples else count / len(samples),
        "reasons": reasons,
    }


def _metric_breakdowns(
    task: LoadedTask,
    samples: Sequence[SampleResult],
    arguments: _Arguments,
) -> dict[str, object]:
    fields = {
        "locomo-refined": ("category",),
        "m3-bench": ("question_types",),
        "video-mme": ("duration", "domain", "task_type"),
        "video-mme-v2": ("group_type", "level", "second_head", "third_head"),
        "egolifeqa": ("day", "question_type"),
        "egotempo": ("question_type",),
        "memlens": ("question_type", "question_subtype"),
        "mm-lifelong": ("question_type",),
        "supermemory-vqa": ("skill",),
        "atm-bench": ("qtype",),
        "mem-gallery": ("point",),
    }.get(_task_family(task.spec.name), ())
    result: dict[str, object] = {}
    for field_name in fields:
        grouped: dict[str, list[ScoredValue]] = {}
        for sample in samples:
            if sample.score is None:
                continue
            raw = sample.metadata.get(field_name)
            labels = (
                tuple(str(value) for value in raw)
                if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
                else (str(raw),)
            )
            for label in labels:
                if label and label != "None":
                    grouped.setdefault(label, []).append(
                        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
                    )
        if grouped:
            result[field_name] = {
                label: summarize(
                    tuple(rows),
                    seed=_task_seed(arguments.seed, f"{task.spec.name}:{field_name}:{label}"),
                    bootstrap_samples=arguments.bootstrap_samples,
                )
                for label, rows in sorted(grouped.items())
            }
    return result


def _task_family(task: str) -> str:
    for family in (
        "supermemory-vqa",
        "video-mme-v2",
        "locomo-refined",
        "mm-lifelong",
        "mem-gallery",
        "egomemreason",
        "egolifeqa",
        "egotempo",
        "memlens",
        "atm-bench",
        "m3-bench",
        "video-mme",
    ):
        if task == family or task.startswith(f"{family}-"):
            return family
    raise ValueError(f"unknown benchmark task: {task}")


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
        "accuracy": strict or summarize(scores, seed=seed, bootstrap_samples=bootstrap_samples),
        "answered_accuracy": summarize(answered, seed=seed, bootstrap_samples=bootstrap_samples),
        "question_count": len(samples),
        "answered_count": len(answered),
        "unanswered_count": len(samples) - len(answered),
    }


def _video_mme_v2_rating(
    samples: Sequence[SampleResult], *, seed: int, bootstrap_samples: int
) -> dict[str, object]:
    return summarize(
        _video_mme_v2_group_values(
            tuple(
                {
                    "unit_id": sample.unit_id,
                    "score": sample.score,
                    "metadata": sample.metadata,
                }
                for sample in samples
            )
        ),
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        clamp=(0.0, 100.0),
    )


def _video_mme_v2_group_values(
    rows: Sequence[Mapping[str, object]],
) -> tuple[ScoredValue, ...]:
    from mindbridge.benchmarks.video_mme_v2 import score_group_answers

    groups: dict[str, list[tuple[int, float, Mapping[str, object]]]] = {}
    for row in rows:
        unit_id, score, metadata = row.get("unit_id"), row.get("score"), row.get("metadata")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("Video-MME-v2 sample unit ID must be a non-empty string")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("Video-MME-v2 sample score must be numeric")
        if not isinstance(metadata, Mapping):
            raise ValueError("Video-MME-v2 sample metadata must be an object")
        position = metadata.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("Video-MME-v2 sample position must be an integer")
        groups.setdefault(unit_id, []).append((position, float(score), metadata))

    ratings = []
    for unit_id in sorted(groups):
        group = sorted(groups[unit_id], key=lambda item: item[0])
        if tuple(item[0] for item in group) != (1, 2, 3, 4):
            raise ValueError(f"Video-MME-v2 group {unit_id} must hold positions 1 to 4")
        metadata = group[0][2]
        rating = score_group_answers(
            str(metadata["group_type"]),
            str(metadata.get("group_structure", "")),
            tuple(item[1] == 1.0 for item in group),
        )
        ratings.append(ScoredValue(unit_id, unit_id, rating))
    return tuple(ratings)


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
        task_samples = tuple(sample for sample in samples if sample.task == task.spec.name)
        previous_rows = tuple(row for row in baseline if row.get("task") == task.spec.name)
        digests = {row.get("evaluation_sha256") for row in previous_rows}
        if previous_rows and digests != {task.evaluation_sha256}:
            raise ValueError(f"baseline evaluation inputs differ for {task.spec.name}")
        current_protocols = {sample.scorer_protocol for sample in task_samples}
        previous_protocols = {row.get("scorer_protocol") for row in previous_rows}
        if previous_rows and previous_protocols != current_protocols:
            raise ValueError(f"baseline scorer protocol differs for {task.spec.name}")
        current_judges = {
            sample.judge_model for sample in task_samples if sample.judge_model is not None
        }
        previous_judges = {
            model for row in previous_rows if isinstance((model := row.get("judge_model")), str)
        }
        if current_judges and previous_judges and current_judges != previous_judges:
            raise ValueError(f"baseline judge model differs for {task.spec.name}")
        current_scored = tuple(sample for sample in task_samples if sample.score is not None)
        previous_scored = tuple(row for row in previous_rows if _has_score(row))
        if task.spec.name == "video-mme-v2":
            current_ids = tuple(sample.sample_id for sample in current_scored)
            previous_ids = tuple(
                sample_id
                for row in previous_scored
                if isinstance((sample_id := row.get("sample_id")), str)
            )
            if (
                len(previous_ids) != len(previous_scored)
                or len(set(previous_ids)) != len(previous_ids)
                or set(current_ids) != set(previous_ids)
            ):
                raise ValueError("candidate and baseline must contain identical scored samples")
            current = _video_mme_v2_group_values(
                tuple(
                    {
                        "unit_id": sample.unit_id,
                        "score": sample.score,
                        "metadata": sample.metadata,
                    }
                    for sample in current_scored
                )
            )
            previous = _video_mme_v2_group_values(previous_scored)
        else:
            current = tuple(
                ScoredValue(sample.sample_id, sample.unit_id, cast(float, sample.score))
                for sample in current_scored
            )
            previous = tuple(_baseline_value(row) for row in previous_scored)
        if not current:
            continue
        if not previous:
            raise ValueError(f"baseline has no scored samples for {task.spec.name}")
        rows.append(
            {
                "task": task.spec.name,
                "metric": _comparison_metric(task),
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
    return task_primary_metric(task.spec.name)


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


def _egomem_submission(
    samples: Sequence[SampleResult], *, requested: bool, allow_partial: bool
) -> tuple[bytes | None, dict[str, object] | None]:
    if not requested:
        return None, None
    selected = tuple(sample for sample in samples if sample.task == "egomemreason")
    predictions: list[tuple[int, str]] = []
    invalid_count = 0
    for sample in selected:
        example_id = sample.metadata.get("example_id")
        choices = sample.metadata.get("choices")
        choice_count = (
            len(choices)
            if isinstance(choices, Sequence) and not isinstance(choices, str | bytes)
            else 0
        )
        answer = sample.parsed_choice
        if (
            isinstance(example_id, bool)
            or not isinstance(example_id, int)
            or sample.error_code is not None
            or sample.ingest_failure_count
            or not 4 <= choice_count <= 10
            or not isinstance(answer, str)
            or answer not in "ABCDEFGHIJ"[:choice_count]
        ):
            invalid_count += 1
            continue
        predictions.append((example_id, answer))

    example_ids = tuple(example_id for example_id, _answer in predictions)
    duplicate_count = len(example_ids) - len(set(example_ids))
    status: dict[str, object] = {
        "file": None,
        "sample_count": len(selected),
        "required_sample_count": _EGOMEM_SUBMISSION_COUNT,
    }
    if invalid_count or duplicate_count:
        status.update(
            {
                "status": "invalid",
                "reason": (
                    f"{invalid_count} invalid prediction(s), "
                    f"{duplicate_count} duplicate example_id(s)"
                ),
            }
        )
        return None, status

    expected_ids = set(range(1, _EGOMEM_SUBMISSION_COUNT + 1))
    actual_ids = set(example_ids)
    if actual_ids != expected_ids:
        partial = allow_partial and actual_ids < expected_ids
        status.update(
            {
                "status": "partial" if partial else "invalid",
                "reason": (
                    f"found {len(actual_ids)} of {_EGOMEM_SUBMISSION_COUNT} required example IDs"
                ),
            }
        )
        return None, status

    content = _json_bytes(
        [
            {"example_id": example_id, "predicted_answer": answer}
            for example_id, answer in sorted(predictions)
        ],
        pretty=True,
    )
    status.update(
        {
            "status": "ready",
            "file": _EGOMEM_SUBMISSION_FILE,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    return content, status


def _write_artifacts(
    arguments: _Arguments,
    samples: Sequence[SampleResult],
    results: Mapping[str, object],
    submission: bytes | None,
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
    submission_path = arguments.output_path / _EGOMEM_SUBMISSION_FILE
    if submission is None and arguments.overwrite:
        submission_path.unlink(missing_ok=True)
    files = [
        (arguments.output_path / _SAMPLES_FILE, samples_bytes),
        (arguments.output_path / _RESULTS_FILE, results_bytes),
    ]
    if submission is not None:
        files.append((submission_path, submission))
    _atomic_replace(files)


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
        performance = cast(Mapping[str, object], task["performance"])
        duration = cast(Mapping[str, object], performance["duration_seconds"])
        usage = cast(Mapping[str, object], performance["token_usage"])
        mean = score.get("mean")
        interval = score.get("confidence_interval_95")
        total_duration = duration.get("total")
        average_duration = duration.get("average")
        total_tokens = usage.get("total_tokens")
        average_tokens = usage.get("average_tokens")
        valid = task.get("score_valid") is not False
        rows.append(
            (
                str(task["task"]),
                str(task["primary_metric"]),
                (
                    "INVALID"
                    if not valid
                    else "—"
                    if mean is None
                    else f"{float(cast(float, mean)):.4f}"
                ),
                (
                    "—"
                    if not isinstance(interval, list)
                    else f"[{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
                ),
                str(task["question_count"]),
                str(task["error_count"]),
                (
                    "—"
                    if isinstance(total_duration, bool)
                    or not isinstance(total_duration, int | float)
                    else f"{float(total_duration):.3f}"
                ),
                (
                    "—"
                    if isinstance(average_duration, bool)
                    or not isinstance(average_duration, int | float)
                    else f"{float(average_duration) * 1_000:.1f}"
                ),
                (
                    "—"
                    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int)
                    else str(total_tokens)
                ),
                (
                    "—"
                    if isinstance(average_tokens, bool)
                    or not isinstance(average_tokens, int | float)
                    else f"{float(average_tokens):.1f}"
                ),
            )
        )
    headers = (
        "task",
        "metric",
        "value",
        "95% cluster CI",
        "n",
        "errors",
        "total s",
        "avg ms",
        "tokens",
        "tokens/q",
    )
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
        help=(
            "comma-separated generation_model/base_url/timeout_seconds/"
            "generation_min_video_seconds overrides"
        ),
    )
    parser.add_argument(
        "--config",
        "--memory-config",
        dest="memory_config",
        type=Path,
        help="JSON MindBridgeConfig; data_dir is replaced by isolated benchmark directories",
    )
    parser.add_argument(
        "--judge-model-args",
        "--judge_model_args",
        default="",
        help="comma-separated model/base_url/api_key/timeout_seconds overrides",
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
    parser.add_argument(
        "--gen-kwargs",
        "--gen_kwargs",
        default="",
        help="deterministic generation settings; supports max_tokens and enable_thinking",
    )
    parser.add_argument("--batch-size", "--batch_size", "-b", default="auto")
    parser.add_argument("--max-batch-size", "--max_batch_size", type=_positive_int, default=64)
    parser.add_argument("--unit-concurrency", type=_positive_int, default=1)
    parser.add_argument("--request-concurrency", type=_positive_int, default=4)
    parser.add_argument("--judge-concurrency", type=_positive_int, default=8)
    parser.add_argument("--recall-limit", type=_positive_int, default=20)
    parser.add_argument("--seed", type=_seed_values, default=(0, 1234, 1234, 1234))
    parser.add_argument(
        "--bootstrap-samples", type=_positive_int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--device", help="local embedding/FunASR device: cpu, cuda, or cuda:N")
    parser.add_argument(
        "--device-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="serialize local model pools that share one CUDA device",
    )
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
        memory_config=(
            None if parsed.memory_config is None else parsed.memory_config.expanduser().resolve()
        ),
        judge_model_args=parsed.judge_model_args,
        judge_concurrency=parsed.judge_concurrency,
        gen_kwargs=gen_kwargs,
        num_fewshot=parsed.num_fewshot,
        use_cache=(None if parsed.use_cache is None else parsed.use_cache.expanduser().resolve()),
        device=parsed.device,
        device_lock=parsed.device_lock,
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


def _load_memory_config(path: Path | None) -> MindBridgeConfig | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read benchmark config {path}: {error}") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"benchmark config {path} is invalid JSON at line {error.lineno}, column {error.colno}"
        ) from None
    config = MindBridgeConfig.model_validate(value)
    if config.generation is None:
        raise ValueError("benchmark --config requires config.generation")
    conflicts = sorted(
        {"max_tokens", "seed", "temperature"}.intersection(config.generation.extra_body or {})
    )
    if conflicts:
        raise ValueError(
            "benchmark config generation.extra_body cannot set benchmark controls: "
            + ", ".join(conflicts)
        )
    return config


def _model_config(
    model: str,
    arguments: str,
    *,
    memory_config: MindBridgeConfig | None = None,
) -> ModelConfig:
    if model != "mindbridge":
        raise ValueError("model must be mindbridge")
    config = ModelConfig.from_environment()
    if memory_config is not None and memory_config.generation is not None:
        generation = memory_config.generation
        config = replace(
            config,
            generation_base_url=generation.base_url or config.generation_base_url,
            generation_model=generation.model,
            generation_capabilities=generation.modalities,
            timeout_seconds=generation.timeout or config.timeout_seconds,
        )
    allowed = {
        "base_url",
        "generation_model",
        "generation_min_video_seconds",
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


def _judge_config(config: ModelConfig, arguments: _Arguments) -> _JudgeConfig:
    judge = _JudgeConfig(
        model=os.getenv("MINDBRIDGE_JUDGE_MODEL", config.generation_model),
        base_url=os.getenv("MINDBRIDGE_JUDGE_BASE_URL", config.generation_base_url),
        api_key=os.getenv("MINDBRIDGE_JUDGE_API_KEY") or config.generation_api_key,
        timeout_seconds=float(
            os.getenv("MINDBRIDGE_JUDGE_TIMEOUT_SECONDS", str(config.timeout_seconds))
        ),
        concurrency=arguments.judge_concurrency,
    )
    allowed = {"model", "base_url", "api_key", "timeout_seconds"}
    aliases = {"pretrained": "model"}
    for item in (part.strip() for part in arguments.judge_model_args.split(",") if part.strip()):
        key, separator, value = item.partition("=")
        key = aliases.get(key.strip(), key.strip())
        if not separator or key not in allowed or not value.strip():
            raise ValueError(f"invalid --judge-model-args item: {item}")
        parsed = value.strip()
        if key == "model":
            judge = replace(judge, model=parsed)
        elif key == "base_url":
            judge = replace(judge, base_url=parsed)
        elif key == "api_key":
            judge = replace(judge, api_key=parsed)
        else:
            judge = replace(judge, timeout_seconds=float(parsed))
    return judge


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


def _evaluation_memory_config(
    config: MindBridgeConfig | None,
    model: ModelConfig,
    arguments: _Arguments,
) -> MindBridgeConfig | None:
    if config is None:
        return None
    generation = config.generation
    if generation is None:
        return config
    options = dict(item.split("=", 1) for item in arguments.gen_kwargs.split(","))
    extra_body = None if generation.extra_body is None else dict(generation.extra_body)
    if "enable_thinking" in options:
        extra_body = {} if extra_body is None else extra_body
        current = extra_body.get("chat_template_kwargs")
        template = dict(current) if isinstance(current, Mapping) else {}
        template["enable_thinking"] = options["enable_thinking"] == "true"
        extra_body["chat_template_kwargs"] = template
    generation = generation.model_copy(
        update={
            "base_url": model.generation_base_url,
            "model": model.generation_model,
            "timeout": model.timeout_seconds,
            "temperature": 0.0,
            "seed": arguments.seed,
            "modalities": model.generation_capabilities,
            "max_tokens": (
                generation.max_tokens if "max_tokens" not in options else int(options["max_tokens"])
            ),
            "extra_body": extra_body,
        }
    )
    embedding = config.embedding
    speech = config.speech
    if arguments.device is not None:
        if embedding.provider in {"jina-omni", "sentence-transformers"}:
            embedding = embedding.model_copy(update={"device": arguments.device})
        if speech is not None and speech.provider == "funasr":
            speech = speech.model_copy(update={"device": arguments.device})
    return config.model_copy(
        update={
            "embedding": embedding,
            "generation": generation,
            "speech": speech,
        }
    )


def _evaluation_devices(
    explicit: str | None,
    config: MindBridgeConfig | None,
) -> tuple[str | None, ...]:
    if config is None:
        return (explicit,)
    configured = []
    if config.embedding.provider in {"jina-omni", "sentence-transformers"}:
        configured.append(explicit or config.embedding.device or "auto")
    if config.speech is not None and config.speech.provider == "funasr":
        configured.append(explicit or config.speech.device)
    devices: dict[str, str | None] = {}
    for device in sorted(set(configured)):
        normalized = device.strip().lower()
        identity = None if normalized == "cpu" else _physical_cuda_identity(normalized)
        devices.setdefault(identity or normalized, None if normalized == "auto" else device)
    return tuple(devices[identity] for identity in sorted(devices))


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
    if key == "generation_min_video_seconds":
        return replace(config, generation_min_video_seconds=float(value))
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
    for name in (_RESULTS_FILE, _SAMPLES_FILE, _EGOMEM_SUBMISSION_FILE):
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


def _failure_detail(error: BaseException, *, source_id: str | None = None) -> FailureDetail:
    cause = error.__cause__ or error.__context__
    seen = {id(error)}
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        next_cause = cause.__cause__ or cause.__context__
        if next_cause is None:
            break
        cause = next_cause
    return FailureDetail(
        source_id=source_id,
        code=_error_code(error),
        reason=error.reason if isinstance(error, MindBridgeError) else None,
        stage=error.stage if isinstance(error, MindBridgeError) else None,
        cause_type=None if cause is None else type(cause).__name__,
    )


def _task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cache_namespace(
    arguments: _Arguments,
    config: ModelConfig,
    batch_sizes: Mapping[str, int],
    *,
    memory_config: MindBridgeConfig | None = None,
) -> str:
    payload = {
        "runner": EVAL_RUNNER_VERSION,
        "schema": EVAL_SCHEMA_VERSION,
        "implementation": _implementation_identity(),
        "model": config.generation_model,
        "base_url": config.generation_base_url,
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "embedding_revision": DEFAULT_JINA_REVISION,
        "transcription_model": DEFAULT_FUNASR_MODEL_ID,
        "device": arguments.device or "auto",
        "seed": arguments.seed,
        "gen_kwargs": arguments.gen_kwargs,
        "generation_min_video_seconds": config.generation_min_video_seconds,
        "recall_limit": arguments.recall_limit,
        "batch_sizes": dict(sorted(batch_sizes.items())),
    }
    if memory_config is not None:
        payload["memory_config"] = _memory_config_payload(memory_config)
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _implementation_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*.py"))
    ]
    return hashlib.sha256(_json_bytes(sources)).hexdigest()


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


def _generation_kwargs(  # noqa: C901 - ordered trust-boundary validation is linear
    value: str, seed: int
) -> str:
    supplied: dict[str, str] = {}
    for item in (part.strip() for part in value.split(",") if part.strip()):
        key, separator, raw = item.partition("=")
        if not separator or not key.strip() or not raw.strip() or key.strip() in supplied:
            raise ValueError(f"invalid --gen_kwargs item: {item}")
        supplied[key.strip()] = raw.strip()
    unsupported = set(supplied) - {
        "temperature",
        "do_sample",
        "seed",
        "max_tokens",
        "enable_thinking",
    }
    if unsupported:
        raise ValueError(
            "MindBridge eval supports deterministic --gen_kwargs only: temperature, do_sample, "
            "seed, max_tokens, enable_thinking"
        )
    if "temperature" in supplied and float(supplied["temperature"]) != 0:
        raise ValueError("reproducible evaluation requires temperature=0")
    if "do_sample" in supplied and supplied["do_sample"].casefold() not in {"false", "0"}:
        raise ValueError("reproducible evaluation requires do_sample=false")
    if "seed" in supplied and int(supplied["seed"]) != seed:
        raise ValueError("--gen_kwargs seed must match the first --seed value")
    if "max_tokens" in supplied:
        try:
            max_tokens = int(supplied["max_tokens"])
        except ValueError:
            raise ValueError("--gen_kwargs max_tokens must be a positive integer") from None
        if max_tokens <= 0:
            raise ValueError("--gen_kwargs max_tokens must be a positive integer")
    if "enable_thinking" in supplied and supplied["enable_thinking"].casefold() not in {
        "true",
        "false",
        "1",
        "0",
    }:
        raise ValueError("--gen_kwargs enable_thinking must be true or false")
    normalized = ["temperature=0", "do_sample=false", f"seed={seed}"]
    if "max_tokens" in supplied:
        normalized.append(f"max_tokens={max_tokens}")
    if "enable_thinking" in supplied:
        enabled = supplied["enable_thinking"].casefold() in {"true", "1"}
        normalized.append(f"enable_thinking={'true' if enabled else 'false'}")
    return ",".join(normalized)


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
