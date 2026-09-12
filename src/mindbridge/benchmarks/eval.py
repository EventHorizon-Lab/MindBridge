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
import random
import re
import time
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, fields, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import TYPE_CHECKING, Protocol, cast

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode, Tracer

if TYPE_CHECKING:
    pass

from mindbridge import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_RECIPE,
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    AsyncMemory,
    ContextBudget,
    ContextBundle,
    ContextExcerpt,
    FaceAnalysis,
    FunASRTranscriber,
    IndexUnavailableError,
    JinaOmniEmbedder,
    Memory,
    MemoryConfig,
    MemoryType,
    MindBridgeConfig,
    Modality,
    OpenAIModels,
    SearchHit,
    resolve_memory_config,
)
from mindbridge._telemetry import (
    OPERATION_TTFT,
    SPAN_KIND,
    _observe_retrieval_results,
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
from mindbridge.benchmarks.eval_arms import (
    _COMPILE_SURFACE_SYSTEM_PROMPT,
    BLIND_PROMPT_VERSION,
    COMPILE_SURFACE_PROMPT_VERSION,
    FULL_CONTEXT_PROMPT_VERSION,
    PRODUCT_ARM,
    _Arm,
    _BaselineGenerator,
    _media_asset_ref,
)
from mindbridge.benchmarks.eval_artifacts import (
    _atomic_replace,
    _config_artifact,
    _json_bytes,
    _jsonl_bytes,
    _merged_manifest,
    _write_artifacts,
)
from mindbridge.benchmarks.eval_cache import (
    CachedAnswer,
    DescriptionCache,
    EvidenceInterval,
    ResponseCache,
)
from mindbridge.benchmarks.eval_config import (
    _MEDIA_MANIFEST_FILE,
    _MODALITY_BY_SUFFIX,
    _PARTIAL_SAMPLES_FILE,
    BASELINE_ARMS,
    DEFAULT_ARM,
    DEFAULT_COMPILE_BUDGET,
    DEFAULT_FULL_CONTEXT_CHARS,
    DEFAULT_INGEST_MODE,
    INGEST_MODES,
    RETRIEVAL_CANDIDATE_LIMIT,
    _Arguments,
    _arguments,
    _batch_size,
    _build_parser,
    _description_cache_path,
    _evaluation_config,
    _evaluation_memory_config,
    _judge_config,
    _JudgeConfig,
    _load_memory_config,
    _memory_config_payload,
    _model_config,
    _picked,
    _require_output,
)
from mindbridge.benchmarks.eval_console import (
    _announce,
    _configure_logging,
    _deferred_progress,
    _ignore_progress,
    _progress,
    _table,
    _uninterpretable_tasks,
)
from mindbridge.benchmarks.eval_environment import (
    acceleration_runtime_metadata,
    hardware_metadata,
    nvidia_smi_rows,
    source_metadata,
    unavailable_server_resources,
)
from mindbridge.benchmarks.eval_judge import _apply_judges
from mindbridge.benchmarks.eval_metrics import (
    _abstentions,
    _answer_retrieval_candidate_limit,
    _blind_baseline_rows,
    _comparisons,
    _evidence_budget_chars,
    _in_run_blind_rows,
    _metrics,
)
from mindbridge.benchmarks.eval_regression import (
    load_result,
    performance_comparisons,
)
from mindbridge.benchmarks.eval_results import (
    EVAL_SCHEMA_VERSION,
    FailureDetail,
    SampleResult,
    _AnswerOutcome,
    _failure_detail,
    _raise_if_systemic_embedding,
    _restored_failure,
    _Retried,
    _retry_transient,
    _systemic_embedding_stage_failure,
    _SystemicEmbeddingFailure,
)
from mindbridge.benchmarks.eval_server_metrics import (
    MetricsSnapshot,
    capture_metrics,
    metrics_window,
)
from mindbridge.benchmarks.eval_statistics import parse_choice
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_ANSWER_SPAN,
    BENCHMARK_ARM,
    BENCHMARK_ARM_SPAN,
    BENCHMARK_COMPILE_CHARS,
    BENCHMARK_COMPILE_ITEMS,
    BENCHMARK_COMPILE_MEDIA_ITEMS,
    BENCHMARK_COMPILE_SPAN,
    BENCHMARK_DIAGNOSTIC_SPAN,
    BENCHMARK_INGEST_ITEMS,
    BENCHMARK_INGEST_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    DIAGNOSTIC_PURPOSE,
    PRODUCT_PURPOSE,
    SHARED_BENCHMARK_ARM,
    EvaluationTelemetry,
    ResourceSampler,
)
from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.benchmarks.model_config import (
    DownloadSettings,
    ModelConfig,
    ServerMetricsOverrides,
)
from mindbridge.benchmarks.official_scorers import (
    finalize_scores,
    local_scores,
    retrieval_gold_ids,
    sample_primary_metric,
    scorer_protocol,
)
from mindbridge.benchmarks.prepare_media import _has_audio, prepare_task_media
from mindbridge.benchmarks.prompts import task_answer_policy, task_answer_surface
from mindbridge.benchmarks.task_catalog import (
    TASKS,
    listing,
)
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError, DataDirectoryLock
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    EmbedTask,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_DIMENSION,
    DEFAULT_JINA_MODEL_ID,
    DEFAULT_JINA_REVISION,
)

EVAL_RUNNER_VERSION = "mindbridge_eval_official_v16"
_BENCHMARK_SEARCH_REPLAY_SETUP_SPAN = "mindbridge.benchmark.search_replay_setup"


class _MemoryContext(Protocol):
    async def __aenter__(self) -> AsyncMemory: ...

    async def __aexit__(self, *_error: object) -> None: ...


MemoryFactory = Callable[[Path], _MemoryContext]
_SearchReplay = Callable[[], Awaitable[None]]
# Called once per task, the moment that task has answered every question, and returns the samples
# to keep. It is what lets a run persist and report a finished task without waiting for the rest.
_TaskCompletion = Callable[
    ["LoadedTask", tuple["SampleResult", ...]], Awaitable[tuple["SampleResult", ...]]
]


# The optional capabilities `Memory` probes with `isinstance` against a `runtime_checkable`
# protocol, which reads attributes with `inspect.getattr_static`: a method reached only through
# `__getattr__` is invisible to it, and one declared on the proxy claims a capability the pooled
# backend may not have. `plan_recall` was hidden that way, so every harness question silently
# answered from the fallback point plan; `stream_answer` was declared unconditionally, so a
# pooled backend without it failed the call instead of taking the buffered path. Binding the
# ones the pool really has onto the proxy instance declares exactly its own capabilities. Every
# required protocol member stays an explicit declaration below, because a property cannot be
# forwarded this way -- reading it here would snapshot its value.
_OPTIONAL_CAPABILITY_METHODS = ("plan_recall", "stream_answer")


class _BorrowedBackend:
    """Forward a shared backend while making per-store ``close`` a no-op."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        for name in _OPTIONAL_CAPABILITY_METHODS:
            # A subclass that wraps the call itself owns the name; anything else binds the
            # pooled backend's own method, which also keeps its real signature.
            if hasattr(type(self), name):
                continue
            method = getattr(backend, name, None)
            if callable(method):
                setattr(self, name, method)

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def close(self) -> None:
        return None


class _CachedVisionDescriber:
    """Describe only the visuals not yet described in this benchmark invocation.

    The endpoint returns a different caption for the same image on every call, so without this
    two ingests of one corpus build different full-text documents and a paired arm cannot be
    compared with itself. Keys use the asset's own SHA-256, while the cache path is namespaced by
    run so a repeat still executes the same measured model workload.
    """

    def __init__(self, backend: VisionDescriptionBackend, cache: DescriptionCache) -> None:
        self._backend = backend
        self._cache = cache

    @property
    def vision_capabilities(self) -> frozenset[Modality]:
        return self._backend.vision_capabilities

    @property
    def vision_model(self) -> str:
        return self._backend.vision_model

    @property
    def vision_space(self) -> str:
        return self._backend.vision_space

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        keys = tuple(_description_digest(value) for value in batch)
        known = tuple(None if key is None else self._cache.get(key) for key in keys)
        pending = tuple(value for value, found in zip(batch, known, strict=True) if found is None)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("mindbridge.model.batch_size", len(pending))
            span.set_attribute(
                "mindbridge.input.modalities",
                tuple(
                    sorted({modality.value for value in pending for modality in value.modalities})
                ),
            )
        if not pending:
            mark_model_requests(0, token_usage_expected=0)
        # One call for the whole miss set, so a partially cached batch still costs one request.
        fresh = iter(() if not pending else self._backend.describe(pending))
        described: list[str] = []
        written: list[tuple[str, str]] = []
        for key, found in zip(keys, known, strict=True):
            if found is not None:
                described.append(found)
                continue
            value = next(fresh)
            if key is not None:
                written.append((key, value))
            described.append(value)
        # One commit for the batch: the cache is `synchronous=FULL`, so a per-caption write cost
        # an fsync each.
        self._cache.put_many(written)
        return tuple(described)

    def close(self) -> None:
        return None


def _description_digest(value: ModelInput) -> str | None:
    """Identify one description input by the content of the assets it carries.

    An input with no resolved digest is describable but not cacheable; it is passed through
    rather than silently sharing a key with everything else in its shape.
    """
    digests = tuple(asset.sha256 for asset in value.assets if asset.sha256)
    if not digests or len(digests) != len(value.assets):
        return None
    return ":".join((*digests, value.text))


class _BorrowedGenerationBackend(_BorrowedBackend):
    """Lend one answerer without transferring ownership to a per-unit memory."""

    @property
    def generation_capabilities(self) -> frozenset[Modality]:
        return cast(GenerationBackend, self._backend).generation_capabilities

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return cast(GenerationBackend, self._backend).answer(
            question, hits, answer_policy=answer_policy, exhaustive=exhaustive
        )


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
    """Skip visual-only videos before lending the shared speech backend, and analyse ahead.

    The store ingests a chunk as speech, then embedding, then write, so the GPU idles while the
    embedding request is in flight and the network idles while the speech model runs -- measured
    at about 5 s and 3.7 s of a 10 s batch. `prefetch()` analyses the next chunk's clips on one
    background thread while the store embeds and writes the current one; the store's own
    `analyze()` for those clips then finds the result here. Entries are keyed by content digest,
    which is also the store's asset id, so the source clip and the copy the store analyses are
    the same bytes. Only the analysis runs early: the store still writes chunk by chunk, so
    corpus order and every stored row are what they were.
    """

    def __init__(self, backend: object) -> None:
        super().__init__(backend)
        self._ahead: dict[str, Future[dict[str, SpeechAnalysis]]] = {}
        self._ahead_lock = Lock()
        # One worker: the speech model serialises on its own lock, and a second thread would
        # only queue behind it.
        self._ahead_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-ahead")

    @property
    def transcription_capabilities(self) -> frozenset[Modality]:
        return cast(SpeechBackend, self._backend).transcription_capabilities

    def prefetch(self, items: Sequence[MemoryItem]) -> None:
        """Analyse these items' audible media in the background, once per distinct clip."""
        refs = []
        for item in items:
            for atom in item.content:
                if not isinstance(atom, Path):
                    continue
                modality = _MODALITY_BY_SUFFIX.get(atom.suffix.casefold())
                if modality not in self.transcription_capabilities:
                    continue
                if modality is Modality.VIDEO and not _has_audio(atom):
                    continue
                refs.append(_media_asset_ref(atom, modality))
        with self._ahead_lock:
            fresh = tuple({ref.id: ref for ref in refs if ref.id not in self._ahead}.values())
            if not fresh:
                return
            future = self._ahead_worker.submit(self._analyze_by_id, fresh)
            for ref in fresh:
                self._ahead[ref.id] = future

    def _analyze_by_id(self, assets: Sequence[AssetRef]) -> dict[str, SpeechAnalysis]:
        analyses = cast(SpeechBackend, self._backend).analyze(assets)
        if len(analyses) != len(assets):
            raise RuntimeError("speech backend returned the wrong number of analyses")
        return {asset.id: analysis for asset, analysis in zip(assets, analyses, strict=True)}

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
        with self._ahead_lock:
            futures = {
                asset.id: self._ahead.pop(asset.id) for asset in audible if asset.id in self._ahead
            }
        ahead: dict[str, SpeechAnalysis] = {}
        for asset_id, future in futures.items():
            # A prefetch that failed is simply analysed again here, where the error belongs to
            # the write that needed it.
            with suppress(Exception):
                ahead[asset_id] = future.result()[asset_id]
        missing = tuple(asset for asset in audible if asset.id not in ahead)
        if missing:
            generated = cast(SpeechBackend, self._backend).analyze(missing)
        else:
            record_unmetered_model_usage(request_count=0)
            generated = ()
        if len(generated) != len(missing):
            raise RuntimeError("speech backend returned the wrong number of analyses")
        fresh = dict(zip((asset.id for asset in missing), generated, strict=True))
        return tuple(
            (ahead.get(asset.id) or fresh[asset.id])
            if include
            else SpeechAnalysis(turns=(), speakers=())
            for asset, include in zip(assets, selected, strict=True)
        )

    def shutdown_prefetch(self) -> None:
        """Finish running look-ahead, cancel queued work, and reject later submissions."""
        self._ahead_worker.shutdown(wait=True, cancel_futures=True)
        with self._ahead_lock:
            self._ahead.clear()


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
        description_cache: Path | None = None,
    ) -> None:
        self.config = config
        self._resolved_config = None
        self._description_cache: DescriptionCache | None = None
        self._tracer = trace.get_tracer("mindbridge.benchmarks.eval") if tracer is None else tracer
        self.embedding_warmup_count = 0
        if memory_config is not None:
            if memory_config.generation is not None:
                # The harness owns both of these for a run: `--model-args` and the environment
                # name the modalities, and the video floor belongs to the corpus rather than to
                # the endpoint. Injecting them here is what makes the configured path honour
                # them -- the resolved backends are built from this document, so a floor left
                # only on `ModelConfig` would be reported in the result artifact while short
                # videos still went to the endpoint whole.
                update: dict[str, object] = {"modalities": config.generation_capabilities}
                if config.generation_min_video_seconds is not None:
                    update["min_video_seconds"] = config.generation_min_video_seconds
                memory_config = memory_config.model_copy(
                    update={"generation": memory_config.generation.model_copy(update=update)}
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
            self._former = (
                None
                if plugins.former is None
                else cast(FormationBackend, _BorrowedBackend(plugins.former))
            )
            self._consolidator = (
                None
                if plugins.consolidator is None
                else cast(ConsolidationBackend, _BorrowedBackend(plugins.consolidator))
            )
            if plugins.vision_describer is None:
                self._vision_describer = None
            else:
                borrowed = cast(
                    VisionDescriptionBackend, _BorrowedBackend(plugins.vision_describer)
                )
                # Opened only when `vision:` is configured, so a run without the slot touches no
                # cache file at all and behaves exactly as it does today.
                if description_cache is not None:
                    self._description_cache = DescriptionCache(
                        description_cache, plugins.vision_describer.vision_space
                    )
                    borrowed = cast(
                        VisionDescriptionBackend,
                        _CachedVisionDescriber(borrowed, self._description_cache),
                    )
                self._vision_describer = borrowed
            # Answer-time reinforcement is a product behaviour, not a measured one: it makes a
            # question's retrieval depend on which earlier questions ran, and under concurrency on
            # the order their updates committed, so a run stops being reproducible from its seed.
            self._settings = replace(resolved.settings, reinforce_on_answer=False)
            configured_embed = getattr(self._embedder, "embed", None)
            if callable(configured_embed):
                configured_embed(
                    (ModelInput(text="MindBridge benchmark warmup"),),
                    task=EmbedTask.QUERY,
                )
                self.embedding_warmup_count = 1
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
        self.embedding_warmup_count = 1
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
        self._former = None
        self._consolidator = None
        self._vision_describer = None
        self._settings = MemoryConfig(
            index_speech=self._transcriber is not None,
            # Answer-time reinforcement is a product behaviour, not a measured one: it makes a
            # question's retrieval depend on which earlier questions ran, and under concurrency on
            # the order their updates committed, so a run stops being reproducible from its seed.
            reinforce_on_answer=False,
        )

    def prefetch_speech(self, items: Sequence[MemoryItem]) -> None:
        """Analyse the next chunk's speech while a store embeds and writes the current one."""
        if isinstance(self._transcriber, _BorrowedSpeechBackend):
            self._transcriber.prefetch(items)

    def memory(self, data_dir: Path) -> AsyncMemory:
        # Forwarded from the dataclass rather than field by field. The hand-written list silently
        # dropped every setting added after it was written, which does not fail anything: the
        # evaluation simply measures the default policy while reporting the configured one.
        # `MemoryPlugins` cannot be used here because the shared-backend proxies are structural
        # and its runtime protocol check reads attributes statically. The capability keywords below
        # stay explicit for the same reason, so a test derives the expected set from
        # `fields(MemoryPlugins)` instead.
        policy = {entry.name: getattr(self._settings, entry.name) for entry in fields(MemoryConfig)}
        return AsyncMemory(
            Memory(
                data_dir,
                embedder=self._embedder,
                answerer=self._answerer,
                transcriber=self._transcriber,
                face_analyzer=self._face_analyzer,
                former=self._former,
                consolidator=self._consolidator,
                vision_describer=self._vision_describer,
                tracer=self._tracer,
                **policy,
            )
        )

    def close(self) -> None:
        if isinstance(self._transcriber, _BorrowedSpeechBackend):
            self._transcriber.shutdown_prefetch()
        if self._description_cache is not None:
            with suppress(Exception):
                self._description_cache.close()
            self._description_cache = None
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
    # The configuration file can name the corpus root, so it is read before the listing and
    # before argument resolution derives the default output directory from that root.
    config_path = (
        None if parsed.memory_config is None else parsed.memory_config.expanduser().resolve()
    )
    try:
        memory_config, overrides = _load_memory_config(config_path)
    except ValueError as error:
        parser.error(str(error))
    _configure_logging(_picked(parsed.verbosity, overrides.run.verbosity, "INFO"))
    if parsed.deliberate and (memory_config is None or memory_config.consolidation is None):
        # Refused rather than silently ignored: without a consolidation backend every round
        # would raise, and a run that quietly skipped the loop would report a score for it.
        parser.error("--deliberate requires a --config declaring a `consolidation` section")
    download = DownloadSettings.resolve(
        overrides.download,
        benchmarks_root=parsed.benchmarks_root,
        data_root=parsed.data_root,
    )
    download.apply_environment()
    list_mode = _list_mode(parsed)
    if list_mode is not None:
        print(listing(download.benchmarks_root, list_mode))
        return 0
    arguments = _arguments(parser, parsed, download, overrides=overrides)
    if parsed.check_integrity:
        manifest, manifest_directory = load_media_manifest(arguments.media_manifest)
        loaded = _load_tasks(arguments, manifest, manifest_directory)
        print(
            _jsonl_bytes(
                (
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
                )
            ).decode(),
            end="",
        )
        return 0
    try:
        base_config = _model_config(
            arguments.model,
            arguments.model_args,
            memory_config=memory_config,
            overrides=overrides,
        )
        judge_config = _judge_config(base_config, arguments, overrides=overrides)
    except ValueError as error:
        parser.error(str(error))
    _require_output(arguments.output_path, overwrite=arguments.overwrite, resume=arguments.resume)
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
        with _deferred_progress(
            f"preparing {name} media", "source", enabled=not arguments.quiet
        ) as progress:
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
                on_progress=None if arguments.quiet else progress,
            )
        if prepared is not None:
            generated_manifest[name] = prepared
    if generated_manifest:
        manifest = _merged_manifest(manifest, manifest_directory, generated_manifest)
        effective_manifest = arguments.output_path / _MEDIA_MANIFEST_FILE
        effective_manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _atomic_replace(((effective_manifest, _jsonl_bytes((manifest,))),))
        arguments = replace(arguments, media_manifest=effective_manifest)
        manifest_directory = effective_manifest.parent
    loaded = _load_tasks(arguments, manifest, manifest_directory)
    config = _evaluation_config(base_config, loaded)
    memory_config = _evaluation_memory_config(memory_config, config, arguments)
    batch_sizes = {task.spec.name: _batch_size(arguments, task) for task in loaded}
    samples, duration, performance, resources, embedding_warmup_count = _execute(
        loaded,
        arguments,
        config,
        judge_config,
        batch_sizes,
        memory_config=memory_config,
        server_metrics=overrides.server_metrics,
    )
    results = _results(
        arguments,
        config,
        judge_config,
        loaded,
        samples,
        duration,
        batch_sizes,
        performance,
        memory_config=memory_config,
        resources=resources,
        embedding_warmup_count=embedding_warmup_count,
    )
    comparisons = _comparisons(arguments, loaded, samples)
    if comparisons:
        results["comparisons"] = comparisons
    performance_rows = (
        []
        if not arguments.performance_budgets
        else performance_comparisons(
            results,
            load_result(cast(Path, arguments.compare)),
            arguments.performance_budgets,
        )
    )
    if performance_rows:
        results["performance_comparisons"] = performance_rows
    _write_artifacts(
        arguments,
        samples,
        results,
        config_bytes=_config_artifact(
            arguments,
            config,
            judge_config,
            memory_config,
            download,
            overrides.server_metrics,
        ),
    )
    for reason in _uninterpretable_tasks(results):
        _announce(f"UNINTERPRETABLE: {reason}")
    if not arguments.quiet:
        print(_table(results))
    has_errors = _execution_has_errors(samples, results)
    regressed = arguments.fail_on_regression and _regressed(
        comparisons, threshold=arguments.regression_threshold
    )
    performance_regressed = arguments.fail_on_regression and any(
        row["regressed"] is True for row in performance_rows
    )
    return int(has_errors or regressed or performance_regressed)


def _execution_has_errors(samples: Sequence[SampleResult], results: Mapping[str, object]) -> bool:
    sample_errors = any(
        sample.error_code is not None
        or sample.ingest_failure_count
        or sample.retrieval_diagnostic_error is not None
        for sample in samples
    )
    rows = cast(Sequence[Mapping[str, object]], results["tasks"])
    return sample_errors or _incomplete_search_replay(rows)


def _load_tasks(
    arguments: _Arguments,
    manifest: Mapping[str, object] | None,
    manifest_directory: Path | None,
) -> tuple[LoadedTask, ...]:
    return tuple(
        _with_fallback_reference(
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
            ),
            getattr(arguments, "fallback_reference_at", None),
        )
        for name in arguments.tasks
    )


def _with_fallback_reference(
    task: LoadedTask,
    fallback: datetime | None,
) -> LoadedTask:
    """Fill only missing question clocks after dataset and corpus clocks are resolved."""
    if fallback is None:
        return task
    count = sum(question.reference_at is None for unit in task.units for question in unit.questions)
    if not count:
        return task
    return replace(
        task,
        units=tuple(
            replace(
                unit,
                questions=tuple(
                    question
                    if question.reference_at is not None
                    else replace(question, reference_at=fallback)
                    for question in unit.questions
                ),
            )
            for unit in task.units
        ),
        fallback_reference_at=fallback,
        fallback_reference_question_count=count,
    )


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
    return {
        int(row[0]): row[1]
        for row in nvidia_smi_rows("index,uuid")
        if len(row) >= 2 and row[0].isdecimal() and row[1]
    }


def _release_device_memory(device: str | None) -> None:
    if (device or "auto").strip().lower() == "cpu":
        return
    gc.collect()
    with suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _server_metrics_endpoints(settings: ServerMetricsOverrides) -> dict[str, str]:
    return {
        name: url
        for name, url in (
            ("generation", settings.generation_url),
            ("embedding", settings.embedding_url),
        )
        if url is not None
    }


def _server_metric_results(
    settings: ServerMetricsOverrides,
    starts: Mapping[str, MetricsSnapshot],
    *,
    config: ModelConfig,
    memory_config: MindBridgeConfig | None,
    cache_only: bool,
    excluded: Mapping[str, Sequence[tuple[MetricsSnapshot, MetricsSnapshot]]] | None = None,
) -> dict[str, object]:
    endpoints = _server_metrics_endpoints(settings)
    result: dict[str, object] = {}
    for name, url in endpoints.items():
        if cache_only:
            result[name] = {
                "scope": "server_process_global",
                "exclusive_attribution": False,
                "metrics_url": url,
                "status": "skipped",
                "reason": "the response cache issued no product model requests",
            }
        else:
            result[name] = {
                **metrics_window(
                    url,
                    starts[name],
                    timeout_seconds=settings.timeout_seconds,
                    excluded=() if excluded is None else excluded.get(name, ()),
                ),
                "phase": "product_execution_including_post_answer_search_replay",
                "judge_traffic_excluded": True,
            }
    if "generation" not in result:
        result["generation"] = unavailable_server_resources(base_url=config.generation_base_url)
    if "embedding" not in result and memory_config is not None:
        embedding = memory_config.embedding
        if embedding.provider == "openai" and embedding.base_url is not None:
            result["embedding"] = unavailable_server_resources(base_url=embedding.base_url)
    return result


def _execute(
    loaded: Sequence[LoadedTask],
    arguments: _Arguments,
    config: ModelConfig,
    judge_config: _JudgeConfig,
    batch_sizes: Mapping[str, int],
    *,
    memory_config: MindBridgeConfig | None,
    server_metrics: ServerMetricsOverrides | None = None,
) -> tuple[
    tuple[SampleResult, ...],
    float,
    Mapping[str, Mapping[str, Mapping[str, object]]],
    Mapping[str, object],
    int,
]:
    global _first_ingest_failure_announced, _deliberation_applied
    _first_ingest_failure_announced = False
    _deliberation_applied = 0
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
    sampler = ResourceSampler(
        storage_roots=tuple(
            BenchmarkRun.path_for(arguments.data_root, task.spec.name, arguments.run_id)
            for task in loaded
        )
    )
    metrics_settings = ServerMetricsOverrides() if server_metrics is None else server_metrics
    metric_endpoints = _server_metrics_endpoints(metrics_settings)
    metric_exclusions: dict[str, list[tuple[MetricsSnapshot, MetricsSnapshot]]] = {
        name: [] for name in metric_endpoints
    }
    embedding_warmup_count = 0
    streamed: set[str] = set()

    @contextmanager
    def exclude_task_tail_measurement() -> Iterator[None]:
        # Immediate judging and interim reporting are not product execution. Keep them between
        # tasks for prompt feedback, but split both client and server accounting around the whole
        # tail so a shared endpoint does not acquire judge traffic and client measurements do not
        # acquire bootstrap aggregation or console-I/O time.
        with sampler.exclude():
            before = {
                name: capture_metrics(
                    url,
                    timeout_seconds=metrics_settings.timeout_seconds,
                )
                for name, url in metric_endpoints.items()
            }
            try:
                yield
            finally:
                for name, url in metric_endpoints.items():
                    metric_exclusions[name].append(
                        (
                            before[name],
                            capture_metrics(
                                url,
                                timeout_seconds=metrics_settings.timeout_seconds,
                            ),
                        )
                    )

    try:
        task_completed = _task_completion(
            arguments,
            judge_config=judge_config,
            batch_sizes=batch_sizes,
            telemetry=telemetry,
            memory_config=memory_config,
            streamed=streamed,
            exclude_task_tail_measurement=exclude_task_tail_measurement,
        )
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
        arm_specs = tuple(_Arm(name) for name in arguments.arms)
        all_cached = response_cache is not None and _all_cached(response_cache, loaded, arm_specs)
        devices = _evaluation_devices(arguments.device, memory_config, needs_speech=needs_speech)
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
                prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None
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
                        description_cache=_description_cache_path(arguments, memory_config),
                    )
                    memory_factory = pool.memory
                    prefetch_speech = pool.prefetch_speech
                    embedding_warmup_count = pool.embedding_warmup_count
                metric_starts = (
                    {}
                    if all_cached
                    else {
                        name: capture_metrics(url, timeout_seconds=metrics_settings.timeout_seconds)
                        for name, url in _server_metrics_endpoints(metrics_settings).items()
                    }
                )
                with sampler:
                    samples = asyncio.run(
                        _run_all(
                            loaded,
                            arguments,
                            batch_sizes=batch_sizes,
                            memory_factory=memory_factory,
                            response_cache=response_cache,
                            tracer=telemetry.tracer,
                            config=config,
                            memory_config=memory_config,
                            on_task_complete=task_completed,
                            prefetch_speech=prefetch_speech,
                        )
                    )
                model_servers = _server_metric_results(
                    metrics_settings,
                    metric_starts,
                    config=config,
                    memory_config=memory_config,
                    cache_only=all_cached,
                    excluded=metric_exclusions,
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
                    already_judged=streamed,
                )
            )
        samples = _with_grounding_loss(samples, telemetry)
        performance = _task_performance(telemetry, loaded, samples, arguments.arms)
        duration = time.perf_counter() - started
        resources = sampler.json(wall_seconds=duration)
        resources["model_servers"] = model_servers
        return samples, duration, performance, resources, embedding_warmup_count
    finally:
        telemetry.close()


def _task_completion(
    arguments: _Arguments,
    *,
    judge_config: _JudgeConfig,
    batch_sizes: Mapping[str, int],
    telemetry: EvaluationTelemetry,
    memory_config: MindBridgeConfig | None,
    streamed: set[str],
    exclude_task_tail_measurement: Callable[[], AbstractContextManager[None]] | None = None,
) -> _TaskCompletion:
    """Build the per-task tail: persist, then normally score and print immediately.

    Answering every task before anything is written or scored made a multi-task run silent until
    the last task finished, and made an outage during the last task discard every earlier task's
    answers. The crash copy costs nothing and is always written. Immediate judging is the default;
    `--no-stream-results` retains the former run-global scoring pass when explicitly requested.
    """
    partial_path = arguments.output_path / _PARTIAL_SAMPLES_FILE
    partial_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)

    async def completed(
        task: LoadedTask, task_samples: tuple[SampleResult, ...]
    ) -> tuple[SampleResult, ...]:
        # Persist before anything that can fail. Judging raises on a missing extra or an
        # unreachable judge, and it must not be able to take the answers down with it: those cost
        # a full ingest to produce, the scores cost one more model call. Writing here also makes
        # the copy identical whether or not the run streams -- predictions, not scores.
        # Same 0o600 as the real artifacts, and fsynced: a copy that survives only a clean exit
        # would not survive the failures it exists for.
        descriptor = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "wb") as partial:
            partial.write(_jsonl_bytes(sample.json() for sample in task_samples))
            partial.flush()
            os.fsync(partial.fileno())
        if arguments.stream_results:
            measurement_exclusion = (
                exclude_task_tail_measurement
                if exclude_task_tail_measurement is not None
                and (not arguments.predict_only or not arguments.quiet)
                else nullcontext
            )
            with measurement_exclusion():
                task_samples = _with_grounding_loss(task_samples, telemetry)
                if not arguments.predict_only:
                    task_samples = await _apply_judges(
                        (task,),
                        task_samples,
                        arguments=arguments,
                        config=judge_config,
                        tracer=telemetry.tracer,
                    )
                streamed.update(sample.sample_id for sample in task_samples)
                if not arguments.quiet:
                    _announce(f"interim results for {task.spec.name} (later tasks still running)")
                    print(
                        _table(
                            {
                                "tasks": _task_rows(
                                    arguments,
                                    (task,),
                                    task_samples,
                                    batch_sizes,
                                    _task_performance(
                                        telemetry, (task,), task_samples, arguments.arms
                                    ),
                                    memory_config=memory_config,
                                )
                            }
                        )
                    )
        return task_samples

    return completed


def _task_performance(
    telemetry: EvaluationTelemetry,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    arms: Sequence[str],
) -> dict[str, dict[str, Mapping[str, object]]]:
    """Read each (task, arm)'s aggregated spans.

    `EvaluationTelemetry.result` is a non-destructive read, so a task that has just finished can
    be summarised mid-run and summarised again in the final document. A mid-run read is missing
    the run-global standalone search replay, which by design runs only once every task has
    answered; the final read is the complete one.
    """
    return {
        task.spec.name: {
            arm: telemetry.result(
                task.spec.name,
                arm=arm,
                question_count=sum(
                    sample.task == task.spec.name and sample.arm == arm and not sample.cached
                    for sample in samples
                ),
            )
            for arm in arms
        }
        for task in tasks
    }


def _with_grounding_loss(
    samples: Sequence[SampleResult],
    telemetry: EvaluationTelemetry,
) -> tuple[SampleResult, ...]:
    """Attach each answer's inline-budget loss, so it reads apart from retrieval loss.

    The recall plan's shape rides along: it is the same per-sample join, and a per-question
    shape is what tells a reader which question classes the planner actually reshaped.
    """
    return tuple(
        sample
        if (grounding := telemetry.sample_grounding(sample.sample_id)) is None
        else replace(
            sample,
            dropped_hits=grounding.dropped_hits,
            recall_shape=grounding.recall_shape,
        )
        for sample in samples
    )


async def _run_all(
    tasks: Sequence[LoadedTask],
    arguments: _Arguments,
    *,
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
    tracer: Tracer,
    config: ModelConfig | None = None,
    memory_config: MindBridgeConfig | None = None,
    on_task_complete: _TaskCompletion | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> tuple[SampleResult, ...]:
    generated_arms = _generator_arms(arguments.arms, tuple(task.spec.name for task in tasks))
    generator = (
        None
        if config is None or not generated_arms
        else _BaselineGenerator(
            config,
            seed=arguments.seed,
            gen_kwargs=arguments.gen_kwargs,
            generation=memory_config,
            tracer=tracer,
        )
    )
    arms = tuple(
        _Arm(
            name,
            generator=generator if name in generated_arms else None,
            seed=arguments.seed,
            allow_partial_sources=(
                arguments.compile_allow_partial_sources if name == "compile" else False
            ),
        )
        for name in arguments.arms
    )
    try:
        return await _run_arms(
            tasks,
            arguments,
            arms=arms,
            batch_sizes=batch_sizes,
            memory_factory=memory_factory,
            response_cache=response_cache,
            tracer=tracer,
            on_task_complete=on_task_complete,
            memory_config=memory_config,
            prefetch_speech=prefetch_speech,
        )
    finally:
        if generator is not None:
            await generator.close()


def _answers_through_ask(arm: _Arm | str, task_name: str) -> bool:
    """Report whether this arm reaches `Memory.ask` on this task.

    Only that path requests an answer policy and exposes the in-answer ranked list, so every
    check keyed on either -- the policy recorded on a row, the ranked-list diagnostic, the cache
    guard -- asks this one question rather than `arm.name == DEFAULT_ARM` alone.
    """
    name = arm if isinstance(arm, str) else arm.name
    return name == DEFAULT_ARM and task_answer_surface(task_name) == "ask"


def _generator_arms(arms: Sequence[str], task_names: Sequence[str]) -> frozenset[str]:
    """Name the arms that answer through the harness generator.

    The generating baselines always do. The product arm does only when a selected task answers
    through `Memory.compile`, whose bundle is handed to the same generator standing in for the
    host assistant; on every other task it answers through `Memory.ask` and needs none.
    """
    names = {name for name in arms if name in BASELINE_ARMS and _Arm(name).generates}
    if DEFAULT_ARM in arms and not all(
        _answers_through_ask(DEFAULT_ARM, name) for name in task_names
    ):
        names.add(DEFAULT_ARM)
    return frozenset(names)


async def _run_arms(
    tasks: Sequence[LoadedTask],
    arguments: _Arguments,
    *,
    arms: Sequence[_Arm],
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
    tracer: Tracer,
    on_task_complete: _TaskCompletion | None = None,
    memory_config: MindBridgeConfig | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> tuple[SampleResult, ...]:
    compile_budget = ContextBudget(
        max_items=arguments.compile_max_items,
        max_chars=arguments.compile_max_chars,
    )
    results: list[SampleResult] = []
    deferred_searches: list[_SearchReplay] = []
    for task in tasks:
        sample_count = sum(len(unit.questions) for unit in task.units) * len(arms)
        if not arguments.quiet:
            _announce(f"running {task.spec.name} ({len(task.units)} units, {sample_count} samples)")
        with (
            _progress(
                f"running {task.spec.name}",
                "sample",
                total=sample_count,
                enabled=not arguments.quiet,
            ) as progress,
            # A unit writes its whole corpus before it answers one question, so the sample bar
            # can sit at zero for an hour on a media task. The ingest bar is what proves that
            # hour is progress. Its total is fixed for the task and known only to the runner,
            # which is why the bar is deferred until the runner reports it; a second total would
            # be rejected rather than resize the bar.
            _deferred_progress(
                f"ingesting {task.spec.name}", "item", enabled=not arguments.quiet
            ) as ingest_progress,
            traced_span(
                tracer,
                BENCHMARK_TASK_SPAN,
                attributes={
                    BENCHMARK_TASK: task.spec.name,
                    BENCHMARK_ARM: SHARED_BENCHMARK_ARM,
                    BENCHMARK_PURPOSE: "orchestration",
                    SPAN_KIND: "benchmark",
                },
            ),
        ):
            run = BenchmarkRun(
                arguments.data_root,
                task.spec.name,
                arguments.run_id,
                resume=arguments.resume,
            )
            task_samples = await run_loaded_task(
                task,
                run=run,
                memory_factory=memory_factory,
                prefetch_speech=prefetch_speech,
                batch_size=batch_sizes[task.spec.name],
                unit_concurrency=arguments.unit_concurrency,
                request_concurrency=arguments.request_concurrency,
                recall_limit=arguments.recall_limit,
                predict_only=arguments.predict_only,
                log_samples=arguments.log_samples,
                response_cache=response_cache,
                arms=arms,
                full_context_chars=arguments.full_context_chars,
                compile_budget=compile_budget,
                answer_policy=arguments.answer_policy,
                ingest_mode=arguments.ingest,
                ingest_digest=_ingest_digest(task, arguments, memory_config),
                deliberate=arguments.deliberate,
                tracer=tracer,
                on_progress=progress,
                on_ingest_progress=ingest_progress,
                on_activity=progress.set_activity,
                on_search_replay_ready=deferred_searches.append,
            )
        if on_task_complete is not None:
            task_samples = await on_task_complete(task, task_samples)
        results.extend(task_samples)
    # This is deliberately a run-global second pass. Replaying one task while a later task is
    # still answering changes shared model-service load and contaminates both latency families.
    for search_replay in deferred_searches:
        await search_replay()
    return tuple(results)


async def run_loaded_task(  # noqa: C901 - bounded workers also own one isolated replay barrier
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
    arms: Sequence[_Arm] = (PRODUCT_ARM,),
    full_context_chars: int = DEFAULT_FULL_CONTEXT_CHARS,
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET,
    answer_policy: AnswerPolicy | None = None,
    ingest_mode: str = DEFAULT_INGEST_MODE,
    ingest_digest: str | None = None,
    deliberate: bool = False,
    tracer: Tracer | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_ingest_progress: Callable[[int, int], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
    on_search_replay_ready: Callable[[_SearchReplay], None] | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> tuple[SampleResult, ...]:
    """Run normalized units with bounded workers while preserving release order."""
    if min(batch_size, unit_concurrency, request_concurrency, recall_limit) <= 0:
        raise ValueError("batch size, concurrency, and recall limit must be positive")
    if recall_limit > 100:
        raise ValueError("recall limit must not exceed 100")
    slots: list[tuple[SampleResult, ...] | None] = [None] * len(task.units)
    unit_paths: list[Path | None] = [None] * len(task.units)
    stores_ready = [False] * len(task.units)
    queue: asyncio.Queue[tuple[int, EvalUnit]] = asyncio.Queue()
    request_semaphore = asyncio.Semaphore(request_concurrency)
    completed = 0
    total = sum(len(unit.questions) for unit in task.units) * len(arms)
    notify_progress = on_progress or _ignore_progress
    notify_ingest = on_ingest_progress or _ignore_progress
    notify_activity = on_activity or _ignore_activity
    unit_activities: list[str | None] = [None] * len(task.units)
    # One bar spans the task rather than one per unit: concurrent workers would otherwise fight
    # over the terminal, and what a reader wants is how much of the corpus is written, not which
    # worker wrote it. An arm that reads no memory ingests nothing and draws no bar.
    reads_memory = any(arm.reads_memory for arm in arms)
    planned_ingest = tuple(
        _planned_ingest_count(unit) if reads_memory else 0 for unit in task.units
    )
    ingest_total = sum(planned_ingest)
    ingested_total = 0
    notify_ingest(0, ingest_total)
    for index, unit in enumerate(task.units):
        queue.put_nowait((index, unit))

    async def worker() -> None:
        while not queue.empty():
            try:
                index, unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            data_dir = run.unit_dir(unit.unit_id)
            unit_paths[index] = data_dir
            checkpoint = (
                None
                if ingest_digest is None
                else _IngestCheckpoint(run, unit.unit_id, ingest_digest)
            )
            reported = 0
            reported_ingest = 0

            def ingest_advanced(count: int, planned: int = planned_ingest[index]) -> None:
                # A checkpoint resumes mid-corpus and a cached unit skips the corpus outright,
                # so the count arrives as an absolute position and may not arrive at all. The
                # clamp is not load-bearing -- `_IngestCheckpoint.start` already drops a note that
                # runs past the first pending cutoff -- it only keeps a display value honest.
                nonlocal ingested_total, reported_ingest
                bounded = min(count, planned)
                if bounded <= reported_ingest:
                    return
                ingested_total += bounded - reported_ingest
                reported_ingest = bounded
                notify_ingest(ingested_total, ingest_total)

            def activity(
                phase: str, unit_index: int = index, unit_total: int = len(task.units)
            ) -> None:
                unit_activities[unit_index] = phase
                notify_activity(_activity_summary(unit_activities, unit_total))

            def sample_completed() -> None:
                nonlocal completed, reported
                completed += 1
                reported += 1
                notify_progress(completed, total)

            def store_ready(unit_index: int = index) -> None:
                stores_ready[unit_index] = True

            activity("preparing")
            samples = await _run_unit(
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
                arms=arms,
                full_context_chars=full_context_chars,
                compile_budget=compile_budget,
                answer_policy=answer_policy,
                ingest_mode=ingest_mode,
                checkpoint=checkpoint,
                deliberate=deliberate,
                tracer=tracer,
                on_sample_completed=sample_completed,
                on_ingested=ingest_advanced,
                on_activity=activity,
                on_store_ready=store_ready,
                prefetch_speech=prefetch_speech,
            )
            slots[index] = samples
            for _ in range(len(samples) - reported):
                sample_completed()
            # A unit that answered from the cache, or whose cutoffs stop short of its tail, wrote
            # fewer items than planned. Settling its share here is what lets the bar finish.
            ingest_advanced(planned_ingest[index])
            unit_activities[index] = None
            notify_activity(_activity_summary(unit_activities, len(task.units)))
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(unit_concurrency, len(task.units)))]
    await asyncio.gather(*workers)
    if any(group is None for group in slots):
        raise RuntimeError("evaluation worker exited before every unit completed")

    async def replay_searches() -> None:
        await _measure_standalone_searches(
            task,
            slots=slots,
            unit_paths=unit_paths,
            stores_ready=stores_ready,
            memory_factory=memory_factory,
            unit_concurrency=unit_concurrency,
            request_semaphore=request_semaphore,
            recall_limit=recall_limit,
            arms=arms,
            tracer=tracer,
        )

    if on_search_replay_ready is None:
        await replay_searches()
    else:
        on_search_replay_ready(replay_searches)
    return tuple(sample for group in slots if group is not None for sample in group)


async def _measure_standalone_searches(  # noqa: C901 - setup failures need per-plan accounting
    task: LoadedTask,
    *,
    slots: Sequence[tuple[SampleResult, ...] | None],
    unit_paths: Sequence[Path | None],
    stores_ready: Sequence[bool],
    memory_factory: MemoryFactory,
    unit_concurrency: int,
    request_semaphore: asyncio.Semaphore,
    recall_limit: int,
    arms: Sequence[_Arm],
    tracer: Tracer | None,
) -> None:
    """Replay public searches only after every formal answer has finished.

    Store opening is setup, not caller latency. Each caller span starts immediately before the
    shared request semaphore so the measured latency includes benchmark admission, while the
    nested ``mindbridge.search`` span remains the public SDK boundary. The diagnostic purpose
    keeps replay embedding work out of product answer nodes and token totals.
    """
    if tracer is None or not any(arm.name == DEFAULT_ARM for arm in arms):
        return
    selected_tracer = tracer
    store_semaphore = asyncio.Semaphore(unit_concurrency)

    def span_attributes(sample: SampleResult) -> dict[str, str]:
        return {
            BENCHMARK_TASK: task.spec.name,
            BENCHMARK_SAMPLE: sample.sample_id,
            BENCHMARK_ARM: DEFAULT_ARM,
            BENCHMARK_PURPOSE: DIAGNOSTIC_PURPOSE,
            SPAN_KIND: "benchmark",
        }

    async def measure_one(
        memory: AsyncMemory,
        question: EvalQuestion,
        sample: SampleResult,
    ) -> None:
        try:
            with traced_span(
                selected_tracer,
                BENCHMARK_DIAGNOSTIC_SPAN,
                attributes=span_attributes(sample),
            ):
                async with request_semaphore:
                    await memory.search(
                        _content(question.content),
                        limit=recall_limit,
                        reference_at=question.reference_at,
                    )
        except Exception:
            # The span records an error attempt. Search replay is diagnostic and must not replace
            # or invalidate the already-completed product answer.
            return

    async def measure_unit(index: int, unit: EvalUnit) -> None:
        path = unit_paths[index]
        samples = slots[index]
        if path is None or samples is None or not stores_ready[index]:
            return
        product_samples = tuple(sample for sample in samples if sample.arm == DEFAULT_ARM)
        # A partial ingest is not a valid warm store. Cache-only units likewise have no fresh
        # product question to replay and never reach the memory factory here.
        if any(sample.ingest_failure_count for sample in product_samples):
            return
        pending = {sample.question_id: sample for sample in product_samples if not sample.cached}
        questions = tuple(
            (question, pending[question.question_id])
            for question in unit.questions
            if question.question_id in pending
        )
        if not questions:
            return
        async with store_semaphore:
            opened = False
            try:
                with traced_span(
                    selected_tracer,
                    _BENCHMARK_SEARCH_REPLAY_SETUP_SPAN,
                    attributes={
                        BENCHMARK_TASK: task.spec.name,
                        BENCHMARK_ARM: DEFAULT_ARM,
                        BENCHMARK_PURPOSE: DIAGNOSTIC_PURPOSE,
                        SPAN_KIND: "benchmark",
                    },
                ):
                    async with memory_factory(path) as memory:
                        opened = True
                        await asyncio.gather(
                            *(
                                measure_one(memory, question, sample)
                                for question, sample in questions
                            )
                        )
            except Exception:
                if not opened:
                    # Store-open time is setup rather than caller latency. Still emit one
                    # zero-work error attempt per planned question so missing measurements cannot
                    # masquerade as a complete successful distribution.
                    for _question, sample in questions:
                        with traced_span(
                            selected_tracer,
                            BENCHMARK_DIAGNOSTIC_SPAN,
                            attributes=span_attributes(sample),
                        ) as span:
                            span.set_status(StatusCode.ERROR)
                return

    await asyncio.gather(*(measure_unit(index, unit) for index, unit in enumerate(task.units)))


@dataclass(frozen=True, slots=True)
class _IngestCheckpoint:
    """Record how much of one unit's causal prefix is durably in its store.

    A store is authoritative for what it holds but says nothing about which benchmark items were
    meant to land there, so a run that is killed mid-corpus leaves nothing that says where to
    start again. The note is written after each committed chunk and never before: one that trails
    the store costs a duplicated chunk, one that leads it silently drops evidence.
    """

    run: BenchmarkRun
    unit_id: str
    digest: str

    async def start(
        self,
        memories: Sequence[MemoryItem],
        cutoffs: Sequence[float | None],
        *,
        memory_factory: MemoryFactory,
        data_dir: Path,
    ) -> tuple[int, list[FailureDetail]]:
        """Return the prefix a resumed run may keep, rebuilding the store when it may not."""
        ingested, failures = self._read()
        # A store ingested past the first pending cutoff would answer that cutoff's questions with
        # memories they must not have seen yet. That leak is silent and scores well, so the store
        # is rebuilt rather than reused.
        if ingested and cutoffs and ingested > _prefix_end(memories, cutoffs[0]):
            ingested, failures = 0, []
        # A note is evidence only about a store that is still there. One left over from a deleted
        # or already-rebuilt directory would skip a prefix that nothing ever wrote.
        occupied = _holds_anything(data_dir)
        if ingested and not occupied:
            ingested, failures = 0, []
        if self.run.resume and not ingested:
            if occupied:
                # A store keeps its ownership lock inside its own directory, so deleting one
                # another run still owns would replace that run's `DataDirectoryInUseError` with
                # two live writers. Opening the store first is how this run learns through the
                # public path that the directory it is about to destroy is nobody else's.
                async with memory_factory(data_dir):
                    pass
                self.run.reset_unit(self.unit_id)
            # The rebuilt store and its note move together. A run killed between the two would
            # otherwise leave the old count describing a directory that no longer holds anything.
            self.record(0, ())
        return ingested, failures

    def record(self, ingested: int, failures: Sequence[FailureDetail]) -> None:
        path = self.run.checkpoint_path(self.unit_id)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "digest": self.digest,
                "ingested": ingested,
                "failures": [detail.json() for detail in failures],
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def _read(self) -> tuple[int, list[FailureDetail]]:
        if not self.run.resume:
            return 0, []
        try:
            document = self.run.checkpoint_path(self.unit_id).read_text(encoding="utf-8")
            payload = json.loads(document)
        except (OSError, ValueError):
            return 0, []
        if not isinstance(payload, Mapping) or payload.get("digest") != self.digest:
            return 0, []
        ingested = payload.get("ingested")
        failures = payload.get("failures")
        if (
            not isinstance(ingested, int)
            or isinstance(ingested, bool)
            or ingested < 0
            or not isinstance(failures, list)
        ):
            return 0, []
        try:
            details = [_restored_failure(detail) for detail in failures]
        except (KeyError, TypeError):
            return 0, []
        return ingested, details


def _holds_anything(directory: Path) -> bool:
    """Say whether a unit directory still holds the store a checkpoint claims to describe."""
    return directory.is_dir() and any(directory.iterdir())


def _ingest_observer(
    checkpoint: _IngestCheckpoint | None,
    ingested: int,
    failures: Sequence[FailureDetail],
    notify: Callable[[int], None],
) -> Callable[[int], None]:
    """Note each committed chunk: a killed run resumes from it, and a live bar advances on it."""

    def observe(done: int) -> None:
        written = ingested + done
        if checkpoint is not None:
            checkpoint.record(written, failures)
        notify(written)

    return observe


async def _run_unit(  # noqa: C901 - causal ingest and store-readiness share one lifecycle
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
    arms: Sequence[_Arm] = (PRODUCT_ARM,),
    full_context_chars: int = DEFAULT_FULL_CONTEXT_CHARS,
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET,
    answer_policy: AnswerPolicy | None = None,
    ingest_mode: str = DEFAULT_INGEST_MODE,
    checkpoint: _IngestCheckpoint | None = None,
    deliberate: bool = False,
    tracer: Tracer | None = None,
    on_sample_completed: Callable[[], None] | None = None,
    on_ingested: Callable[[int], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
    on_store_ready: Callable[[], None] | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> tuple[SampleResult, ...]:
    ordered = tuple((arm, question) for arm in arms for question in unit.questions)
    notify_store_ready = on_store_ready or _ignore_store_ready
    notify_ingested = on_ingested or _ignore_ingested
    notify_activity = on_activity or _ignore_activity
    results: dict[tuple[str, str], SampleResult] = {}
    for arm in arms:
        results.update(
            _cached_results(
                response_cache,
                task,
                unit,
                arm=arm,
                predict_only=predict_only,
                log_samples=log_samples,
                answer_policy=answer_policy,
            )
        )
    _report_completions(on_sample_completed, len(results))
    if len(results) == len(ordered):
        return tuple(results[(arm.name, question.question_id)] for arm, question in ordered)
    pending_questions = {
        arm.name: _pending_questions(
            unit,
            {name for (arm_name, name) in results if arm_name == arm.name},
        )
        for arm in arms
    }
    cutoffs = _ordered_cutoffs(pending_questions.values())
    memories = tuple(
        sorted(
            unit.memories,
            # Python's sort is stable, so equal causal boundaries retain the adapter's
            # release order.  Source IDs identify evidence; they are not a chronology and
            # using them as a tie-breaker silently reordered dialogue turns and media parts.
            key=_memory_end,
        )
    )
    stuffs_context = any(arm.name == "full-context" and pending_questions[arm.name] for arm in arms)
    reading_arms = tuple(
        arm.name for arm in arms if arm.reads_memory and pending_questions[arm.name]
    )
    reads_memory = bool(reading_arms)
    # Ingestion is product setup whenever the product arm is present. This must not change when a
    # caller merely reorders the same arms; a random-only run still owns its required setup.
    memory_arm = (
        DEFAULT_ARM
        if DEFAULT_ARM in reading_arms
        else reading_arms[0]
        if len(reading_arms) == 1
        else SHARED_BENCHMARK_ARM
    )
    ingest_failure_details: list[FailureDetail] = []
    ingested = 0
    ingest_failures = 0
    # `pending` is the causal cursor every arm reads from; `ingested` is how much of it a previous
    # run already wrote. They differ only while a resumed store runs ahead of the current cutoff.
    pending = 0
    try:
        # An arm that reads no memory never writes one either, so it neither trusts nor rebuilds
        # a store another run left behind. Resolving the checkpoint inside this block reports a
        # directory owned by another run as the store failure it is.
        if checkpoint is not None and reads_memory:
            ingested, ingest_failure_details = await checkpoint.start(
                memories, cutoffs, memory_factory=memory_factory, data_dir=data_dir
            )
            ingest_failures = len(ingest_failure_details)
            # A resumed store starts the bar where the killed run left it, not at zero.
            notify_ingested(ingested)
        async with memory_factory(data_dir) as memory:
            for cutoff in cutoffs:
                end = _prefix_end(memories, cutoff, pending)
                if reads_memory and end > ingested:
                    notify_activity("ingesting")
                    ingest_failures += await _ingest(
                        memory,
                        memories[ingested:end],
                        batch_size=batch_size,
                        on_failure=ingest_failure_details.append,
                        tracer=tracer,
                        mode=ingest_mode,
                        arm=memory_arm,
                        on_chunk=_ingest_observer(
                            checkpoint, ingested, ingest_failure_details, notify_ingested
                        ),
                        prefetch=prefetch_speech,
                    )
                    ingested = end
                    if deliberate:
                        notify_activity("deliberating")
                        await _deliberate_after_ingest(memory)
                pending = end
                context = (
                    _full_context(memories[:pending], full_context_chars) if stuffs_context else ""
                )
                notify_activity("answering")
                cutoff_results = await _answer_arms(
                    memory,
                    task,
                    unit,
                    arms=arms,
                    questions_by_arm={
                        arm.name: pending_questions[arm.name].get(cutoff, []) for arm in arms
                    },
                    context=context,
                    ingest_failures=ingest_failures,
                    ingest_failure_details=tuple(ingest_failure_details),
                    request_concurrency=request_concurrency,
                    request_semaphore=request_semaphore,
                    recall_limit=recall_limit,
                    predict_only=predict_only,
                    log_samples=log_samples,
                    response_cache=response_cache,
                    compile_budget=compile_budget,
                    answer_policy=answer_policy,
                    tracer=tracer,
                    on_sample_completed=on_sample_completed,
                )
                results.update(cutoff_results)
                if any(
                    sample.error_code == "_SystemicEmbeddingFailure"
                    for sample in cutoff_results.values()
                ):
                    raise _SystemicEmbeddingFailure(
                        "systemic query embedding failure; remaining questions aborted"
                    )
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        if len(results) == len(ordered) and not isinstance(error, _SystemicEmbeddingFailure):
            raise
        for arm, question in ordered:
            identity = (arm.name, question.question_id)
            if identity not in results:
                results[identity] = _sample(
                    task,
                    unit,
                    question,
                    error,
                    ingest_failures=ingest_failures,
                    ingest_failure_details=tuple(ingest_failure_details),
                    predict_only=predict_only,
                    log_samples=log_samples,
                    arm=arm,
                    answer_policy=answer_policy,
                )
    else:
        notify_store_ready()
    return tuple(results[(arm.name, question.question_id)] for arm, question in ordered)


async def _answer_arms(
    memory: AsyncMemory,
    task: LoadedTask,
    unit: EvalUnit,
    *,
    arms: Sequence[_Arm],
    questions_by_arm: Mapping[str, Sequence[EvalQuestion]],
    context: str,
    ingest_failures: int,
    ingest_failure_details: tuple[FailureDetail, ...],
    request_concurrency: int,
    request_semaphore: asyncio.Semaphore,
    recall_limit: int,
    predict_only: bool,
    log_samples: bool,
    response_cache: ResponseCache | None,
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET,
    answer_policy: AnswerPolicy | None = None,
    tracer: Tracer | None = None,
    on_sample_completed: Callable[[], None] | None = None,
) -> dict[tuple[str, str], SampleResult]:
    """Answer one cutoff's pending questions once per arm, against one ingested store."""
    results: dict[tuple[str, str], SampleResult] = {}
    for arm in arms:
        questions = questions_by_arm.get(arm.name, ())
        if not questions:
            continue

        answered = await _answer_many(
            memory,
            questions,
            request_concurrency=request_concurrency,
            request_semaphore=request_semaphore,
            recall_limit=recall_limit,
            arm=arm,
            task_name=task.spec.name,
            unit_id=unit.unit_id,
            context=context,
            compile_budget=compile_budget,
            answer_policy=answer_policy,
            tracer=tracer,
            on_complete=on_sample_completed,
        )
        for question, outcome in zip(questions, answered, strict=True):
            _cache_outcome(
                response_cache,
                task,
                unit,
                question,
                outcome,
                ingest_failures,
                arm=arm,
            )
            results[(arm.name, question.question_id)] = _sample(
                task,
                unit,
                question,
                outcome,
                ingest_failures=ingest_failures,
                ingest_failure_details=ingest_failure_details,
                predict_only=predict_only,
                log_samples=log_samples,
                arm=arm,
                answer_policy=answer_policy,
            )
    return results


def _cached_results(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    *,
    arm: _Arm = PRODUCT_ARM,
    predict_only: bool,
    log_samples: bool,
    answer_policy: AnswerPolicy | None = None,
) -> dict[tuple[str, str], SampleResult]:
    if cache is None:
        return {}
    results = {}
    for question in unit.questions:
        answer = cache.get(_cache_task(task, arm), unit.unit_id, question.question_id)
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
                ranked_source_ids=answer.ranked_source_ids or (),
                ranked_source_ids_complete=answer.ranked_source_ids is not None,
            )
            results[(arm.name, question.question_id)] = _sample(
                task,
                unit,
                question,
                outcome,
                ingest_failures=0,
                predict_only=predict_only,
                log_samples=log_samples,
                arm=arm,
                answer_policy=answer_policy,
            )
    return results


def _pending_questions(
    unit: EvalUnit, completed: Sequence[str] | set[str]
) -> dict[float | None, list[EvalQuestion]]:
    groups: dict[float | None, list[EvalQuestion]] = {}
    for question in unit.questions:
        if question.question_id not in completed:
            groups.setdefault(question.cutoff_seconds, []).append(question)
    return groups


def _ordered_cutoffs(
    groups: Iterable[Mapping[float | None, Sequence[EvalQuestion]]],
) -> tuple[float | None, ...]:
    """Ingest in causal order, with the uncut questions last, across every arm's pending set."""
    values = {cutoff for group in groups for cutoff in group}
    cutoffs: list[float | None] = [*sorted(value for value in values if value is not None)]
    if None in values:
        cutoffs.append(None)
    return tuple(cutoffs)


def _cache_outcome(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    outcome: _AnswerOutcome | BaseException,
    ingest_failures: int,
    *,
    arm: _Arm = PRODUCT_ARM,
) -> None:
    if (
        cache is None
        or ingest_failures
        or not isinstance(outcome, _AnswerOutcome)
        or not outcome.prediction.strip()
        or outcome.retrieval_diagnostic_error is not None
        or (
            _answers_through_ask(arm, task.spec.name)
            and retrieval_gold_ids(task.spec.name, question.metadata)
            and not outcome.ranked_source_ids_complete
        )
    ):
        return
    cache.put(
        _cache_task(task, arm),
        unit.unit_id,
        question.question_id,
        CachedAnswer(
            outcome.prediction,
            outcome.confidence,
            outcome.memory_ids,
            outcome.evidence,
            outcome.abstained,
            outcome.abstention_reason,
            outcome.ranked_source_ids if outcome.ranked_source_ids_complete else None,
        ),
    )


@contextmanager
def _durable_write(
    tracer: Tracer | None,
    count: int,
    *,
    arm: str = DEFAULT_ARM,
) -> Iterator[None]:
    """Time one accepted batch through to durable, searchable memory.

    ``add``/``add_many`` return only after SQLite commits, Zvec flushes, and the search-index
    outbox is acknowledged, so this span measures accepted input to durable and searchable
    memory rather than the time until the call was accepted.
    """
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_INGEST_SPAN,
        attributes={
            SPAN_KIND: "stage",
            BENCHMARK_ARM: arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            BENCHMARK_INGEST_ITEMS: count,
        },
    ):
        yield


def _declined(answer: str, question: EvalQuestion) -> bool:
    """Report a refusal a task worded itself, which the product cannot recognise."""
    return question.refusal is not None and answer.strip().rstrip(".") == (
        question.refusal.strip().rstrip(".")
    )


async def _capture_chunk(
    memory: AsyncMemory,
    chunk: Sequence[MemoryItem],
    *,
    on_failure: Callable[[FailureDetail], None] | None,
    tracer: Tracer | None,
    arm: str = DEFAULT_ARM,
) -> int:
    """Ingest one chunk through `capture()` + `settle()`, so both produce real
    `mindbridge.capture` and `mindbridge.settle` spans -- the only way `--ingest capture` makes
    capture acknowledgement and time-to-searchable measurable by a real run instead of
    unmeasurable.
    """
    failures = 0
    with _durable_write(tracer, len(chunk), arm=arm):
        for item in chunk:
            try:
                await memory.capture(
                    _memory_content(item),
                    occurred_at=item.occurred_at,
                    occurred_end=item.occurred_end,
                    metadata=_memory_metadata(item),
                    memory_type=MemoryType.EPISODIC,
                )
            except IndexUnavailableError:
                raise
            except Exception as error:
                failures += 1
                if on_failure is not None:
                    on_failure(_failure_detail(error, source_id=item.source_id))
        # Drain the store's whole capture queue so the span still measures durable-and-searchable
        # wall time, as `_ingest_json` documents for the `add` path. Each `AsyncMemory` here is
        # one evaluation unit's isolated store and cutoffs settle sequentially, so this only ever
        # drains what this unit itself captured.
        while True:
            try:
                settled = await memory.settle(limit=100)
            except IndexUnavailableError:
                raise
            except Exception as error:
                failures += 1
                if on_failure is not None:
                    on_failure(_failure_detail(error))
                continue
            if not settled:
                break
    return failures


async def _deliberate_after_ingest(memory: AsyncMemory) -> None:
    """Run the slow loop over what was just ingested, before the cutoff's questions are asked.

    Between ingest and questions rather than after the whole unit: that is where a real
    deployment's loop would have run, and a consolidation applied after the questions could not
    change an answer it was meant to improve.
    """
    global _deliberation_applied
    report = await memory.deliberate()
    _deliberation_applied += report.applied


async def _ingest(  # noqa: C901 - bisection, systemic-outage abort, and transient retry share one chunk path
    memory: AsyncMemory,
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
    on_failure: Callable[[FailureDetail], None] | None = None,
    on_chunk: Callable[[int], None] | None = None,
    tracer: Tracer | None = None,
    mode: str = DEFAULT_INGEST_MODE,
    arm: str = DEFAULT_ARM,
    prefetch: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> int:
    # An arm that reads no memory never reaches here: `_run_unit` skips ingestion for it, so the
    # blind control cannot accidentally score a store it was supposed to run without.
    if mode not in INGEST_MODES:
        raise ValueError(f"unknown ingest mode {mode!r}; choose from {', '.join(INGEST_MODES)}")

    async def add_chunk(chunk: Sequence[MemoryItem]) -> int:
        contents = tuple(_memory_content(item) for item in chunk)

        async def add_all() -> None:
            with _durable_write(tracer, len(chunk), arm=arm):
                await memory.add_many(
                    contents,
                    occurred_at=tuple(item.occurred_at for item in chunk),
                    occurred_end=tuple(item.occurred_end for item in chunk),
                    metadata=tuple(_memory_metadata(item) for item in chunk),
                    memory_type=MemoryType.EPISODIC,
                )

        async def add_first() -> None:
            with _durable_write(tracer, 1, arm=arm):
                await memory.add(
                    contents[0],
                    occurred_at=chunk[0].occurred_at,
                    occurred_end=chunk[0].occurred_end,
                    metadata=_memory_metadata(chunk[0]),
                    memory_type=MemoryType.EPISODIC,
                )

        try:
            await _retry_transient(add_all)
            return 0
        except IndexUnavailableError:
            raise
        except Exception as error:
            _raise_if_systemic_embedding(
                error, "systemic embedding provider failure; item isolation aborted"
            )
            if len(chunk) > 1:
                middle = len(chunk) // 2
                return await add_chunk(chunk[:middle]) + await add_chunk(chunk[middle:])
        try:
            await _retry_transient(add_first)
        except IndexUnavailableError:
            raise
        except Exception as error:
            _raise_if_systemic_embedding(
                error, "systemic embedding provider failure; single-item fallback aborted"
            )
            _announce_first_ingest_failure(error, chunk[0].source_id)
            if on_failure is not None:
                on_failure(_failure_detail(error, source_id=chunk[0].source_id))
            return 1
        return 0

    async def capture_chunk(chunk: Sequence[MemoryItem]) -> int:
        return await _capture_chunk(
            memory,
            chunk,
            on_failure=on_failure,
            tracer=tracer,
            arm=arm,
        )

    chunk_ingest = capture_chunk if mode == "capture" else add_chunk
    return await _ingest_chunks(
        chunk_ingest, items, batch_size=batch_size, on_chunk=on_chunk, prefetch=prefetch
    )


async def _ingest_chunks(
    ingest_chunk: Callable[[Sequence[MemoryItem]], Awaitable[int]],
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
    on_chunk: Callable[[int], None] | None,
    prefetch: Callable[[Sequence[MemoryItem]], None] | None = None,
) -> int:
    """Ingest one chunk at a time, noting each boundary a killed run could restart from.

    Chunks still commit in order; `prefetch` only lets the speech model start on the next chunk
    while the store embeds and writes this one.
    """
    failures = 0
    for offset in range(0, len(items), batch_size):
        if prefetch is not None and offset + batch_size < len(items):
            prefetch(items[offset + batch_size : offset + 2 * batch_size])
        # Every item of a chunk is written or counted as failed by the time it returns, so a
        # chunk boundary is the only place a resumable note is exactly true.
        failures += await ingest_chunk(items[offset : offset + batch_size])
        if on_chunk is not None:
            on_chunk(min(offset + batch_size, len(items)))
    return failures


def _candidate_count(unit: EvalUnit, question: EvalQuestion) -> int:
    """Count distinct sources a ranker could return for one question at its cutoff."""
    boundary = math.inf if question.cutoff_seconds is None else question.cutoff_seconds
    return len({item.source_id for item in unit.memories if _memory_end(item) <= boundary})


def _require_active_embedding_service(systemic_stop: asyncio.Event) -> None:
    if systemic_stop.is_set():
        raise _SystemicEmbeddingFailure("systemic query embedding failure; queued question aborted")


async def _guarded_answer(
    memory: AsyncMemory,
    question: EvalQuestion,
    *,
    semaphore: asyncio.Semaphore,
    systemic_stop: asyncio.Event,
    arm: _Arm,
    task_name: str,
    unit_id: str,
    recall_limit: int,
    context: str,
    compile_budget: ContextBudget,
    answer_policy: AnswerPolicy | None,
    tracer: Tracer | None,
    on_answer: Callable[[EvalQuestion, _AnswerOutcome], None] | None,
    on_complete: Callable[[], None] | None,
) -> _AnswerOutcome:
    identity = f"{task_name}/{unit_id}/{question.question_id}"
    sample_id = identity if arm.name == DEFAULT_ARM else f"{arm.name}:{identity}"
    try:
        _require_active_embedding_service(systemic_stop)
        if not arm.generates:
            async with semaphore:
                _require_active_embedding_service(systemic_stop)
                outcome = await _arm_answer(
                    memory,
                    question,
                    arm=arm,
                    task_name=task_name,
                    recall_limit=recall_limit,
                    context=context,
                    compile_budget=compile_budget,
                    answer_policy=answer_policy,
                    tracer=tracer,
                    sample_id=sample_id,
                    started=time.perf_counter(),
                    answer_span=None,
                )
        else:
            # This caller span starts before request admission. Its latency and TTFT therefore
            # include benchmark queueing; nested SDK/model spans expose service time.
            with _answer_span(tracer, task_name, sample_id, arm.name) as (answer_span, started):
                async with semaphore:
                    _require_active_embedding_service(systemic_stop)
                    outcome = await _arm_answer(
                        memory,
                        question,
                        arm=arm,
                        task_name=task_name,
                        recall_limit=recall_limit,
                        context=context,
                        compile_budget=compile_budget,
                        answer_policy=answer_policy,
                        tracer=tracer,
                        sample_id=sample_id,
                        started=started,
                        answer_span=answer_span,
                    )
                    if outcome.error is not None and answer_span is not None:
                        answer_span.set_status(StatusCode.ERROR)
        if arm.generates and on_answer is not None:
            on_answer(question, outcome)
        return outcome
    except _SystemicEmbeddingFailure:
        systemic_stop.set()
        raise
    finally:
        if on_complete is not None:
            on_complete()


async def _answer_many(
    memory: AsyncMemory,
    questions: Sequence[EvalQuestion],
    *,
    request_concurrency: int,
    request_semaphore: asyncio.Semaphore | None = None,
    recall_limit: int,
    on_answer: Callable[[EvalQuestion, _AnswerOutcome], None] | None = None,
    on_complete: Callable[[], None] | None = None,
    arm: _Arm = PRODUCT_ARM,
    task_name: str = "",
    unit_id: str = "",
    context: str = "",
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET,
    answer_policy: AnswerPolicy | None = None,
    tracer: Tracer | None = None,
) -> tuple[_AnswerOutcome | BaseException, ...]:
    semaphore = request_semaphore or asyncio.Semaphore(request_concurrency)
    systemic_stop = asyncio.Event()

    with _arm_run_span(tracer, task_name, arm.name):
        answered = tuple(
            await asyncio.gather(
                *(
                    _guarded_answer(
                        memory,
                        question,
                        semaphore=semaphore,
                        systemic_stop=systemic_stop,
                        arm=arm,
                        task_name=task_name,
                        unit_id=unit_id,
                        recall_limit=recall_limit,
                        context=context,
                        compile_budget=compile_budget,
                        answer_policy=answer_policy,
                        tracer=tracer,
                        on_answer=on_answer,
                        on_complete=on_complete,
                    )
                    for question in questions
                ),
                return_exceptions=True,
            )
        )

    diagnosed = tuple(
        replace(
            outcome,
            retrieval_diagnostic_error=RuntimeError(
                "the ranked retrieval list from the product answer was not observed"
            ),
        )
        if isinstance(outcome, _AnswerOutcome)
        # `Memory.compile` exposes no ranked list, so a product answer on a compile task has
        # none to observe; only the `ask` and `random` paths owe one.
        and (arm.name == "random" or _answers_through_ask(arm, task_name))
        and retrieval_gold_ids(task_name, question.metadata)
        and not outcome.ranked_source_ids_complete
        else outcome
        for question, outcome in zip(questions, answered, strict=True)
    )
    if not arm.generates and on_answer is not None:
        for question, outcome in zip(questions, diagnosed, strict=True):
            if isinstance(outcome, _AnswerOutcome):
                on_answer(question, outcome)
    return diagnosed


@contextmanager
def _compile_span(tracer: Tracer | None, bundle: ContextBundle) -> Iterator[None]:
    """Tag one compiled bundle's size so `eval_telemetry` can report it next to its latency.

    `compile()` itself carries no size attribute -- it is a benchmark-only measurement, so the
    harness tags it on a harness-owned span rather than asking product code to carry it.
    """
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_COMPILE_SPAN,
        attributes={
            SPAN_KIND: "stage",
            BENCHMARK_COMPILE_CHARS: bundle.chars,
            BENCHMARK_COMPILE_ITEMS: len(bundle.hits) + len(bundle.excerpts),
            # Grounded parts, not memories carrying them, because that is the quantity
            # `ContextBudget.max_media_items` bounds: an omni memory with a still and a clip is
            # two parts against the budget and has to be two here, or a multi-asset bundle
            # reports as thrifty as a single-asset one.
            BENCHMARK_COMPILE_MEDIA_ITEMS: sum(len(hit.assets) for hit in bundle.hits),
        },
    ):
        yield


@contextmanager
def _arm_run_span(tracer: Tracer | None, task_name: str, arm: str) -> Iterator[None]:
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_ARM_SPAN,
        attributes={
            BENCHMARK_TASK: task_name,
            BENCHMARK_ARM: arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            SPAN_KIND: "benchmark",
        },
    ):
        yield


@contextmanager
def _answer_span(
    tracer: Tracer | None,
    task_name: str,
    sample_id: str,
    arm: str | None = None,
) -> Iterator[tuple[Span | None, float]]:
    """Scope one answer's model spans so grounding loss can be attributed to its sample."""
    if tracer is None:
        yield None, time.perf_counter()
        return
    selected_arm = (
        sample_id.split(":", 1)[0]
        if arm is None and ":" in sample_id
        else DEFAULT_ARM
        if arm is None
        else arm
    )
    with traced_span(
        tracer,
        BENCHMARK_ANSWER_SPAN,
        attributes={
            BENCHMARK_TASK: task_name,
            BENCHMARK_SAMPLE: sample_id,
            BENCHMARK_ARM: selected_arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            SPAN_KIND: "benchmark",
        },
    ) as span:
        yield span, time.perf_counter()


async def _arm_answer(  # noqa: C901 - baseline and streamed product paths share one clock
    memory: AsyncMemory,
    question: EvalQuestion,
    *,
    arm: _Arm,
    task_name: str,
    recall_limit: int,
    context: str,
    sample_id: str,
    started: float,
    answer_span: Span | None,
    compile_budget: ContextBudget = DEFAULT_COMPILE_BUDGET,
    answer_policy: AnswerPolicy | None = None,
    tracer: Tracer | None = None,
) -> _AnswerOutcome:
    latency_started = time.perf_counter()
    content = _content(question.content)
    ranked: tuple[SearchHit, ...] = ()
    ranked_complete = False

    async def attempt(operation: Callable[[], Awaitable[_Retried]]) -> _Retried:
        # Only the attempt that answered is timed: a retry's backoff is outage time, not latency.
        async def timed() -> _Retried:
            nonlocal latency_started
            latency_started = time.perf_counter()
            return await operation()

        return await _retry_transient(timed)

    def observe_retrieval(value: object) -> None:
        nonlocal ranked, ranked_complete
        if isinstance(value, tuple) and all(isinstance(hit, SearchHit) for hit in value):
            ranked = value
            ranked_complete = True

    try:
        if arm.name == "random":
            ranked = await attempt(
                lambda: memory.search(
                    content,
                    limit=(
                        RETRIEVAL_CANDIDATE_LIMIT
                        if retrieval_gold_ids(task_name, question.metadata)
                        else recall_limit
                    ),
                    reference_at=question.reference_at,
                )
            )
            order = list(ranked)
            random.Random(f"{arm.seed}:{sample_id}").shuffle(order)
            return _AnswerOutcome(
                "",
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                tuple(hit.id for hit in order),
                tuple(_evidence(hit) for hit in order),
                ranked_source_ids=_source_ids(order),
                ranked_source_ids_complete=True,
            )
        # The product arm answers through the surface the task calls for: `Memory.ask` for a
        # question with an answer in memory, `Memory.compile` plus the host generator for a task
        # that wants an assistant's response. `task_answer_surface` owns the mapping and its
        # measurements.
        compile_surface = arm.name == DEFAULT_ARM and not _answers_through_ask(arm, task_name)
        if arm.name == "compile" or compile_surface:
            bundle = await attempt(
                lambda: memory.compile(
                    content,
                    budget=compile_budget,
                    reference_at=question.reference_at,
                    allow_partial_sources=arm.allow_partial_sources,
                )
            )
            with _compile_span(tracer, bundle):
                rendered = bundle.render()
            compile_generator = arm.generator
            if compile_generator is None:
                raise RuntimeError(
                    f"answering through Memory.compile requires a generator ({arm.name} arm, "
                    f"task {task_name})"
                )
            prediction = await attempt(
                lambda: compile_generator.answer(
                    _question_text(question),
                    rendered,
                    question_assets=tuple(
                        atom for atom in question.content if isinstance(atom, Path)
                    ),
                    evidence_hits=bundle.hits,
                    system_prompt=_COMPILE_SURFACE_SYSTEM_PROMPT if compile_surface else None,
                )
            )
            # The product arm counts a task-worded refusal on this surface as it does on `ask`;
            # the `compile` baseline keeps the baseline rule below and counts none.
            declined_here = compile_surface and _declined(prediction, question)
            return _AnswerOutcome(
                prediction,
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                tuple(hit.id for hit in bundle.hits),
                tuple(_evidence(hit) for hit in bundle.hits),
                abstained=declined_here,
                abstention_reason=(
                    AbstentionReason.INSUFFICIENT_EVIDENCE.value if declined_here else None
                ),
                ranked_source_ids=_source_ids(ranked),
                compiled_chars=bundle.chars,
                compiled_items=len(bundle.hits) + len(bundle.excerpts),
                excerpt_source_ids=tuple(excerpt.source_memory_id for excerpt in bundle.excerpts),
                excerpt_evidence=tuple(_excerpt_evidence(excerpt) for excerpt in bundle.excerpts),
            )
        generator = arm.generator
        if generator is not None and arm.name != DEFAULT_ARM:
            prediction = await attempt(
                lambda: generator.answer(
                    _question_text(question),
                    context if arm.name == "full-context" else None,
                    question_assets=tuple(
                        atom for atom in question.content if isinstance(atom, Path)
                    ),
                )
            )
            return _AnswerOutcome(
                prediction,
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                (),
                (),
            )
        # Protocol alignment, not a scorer change: one task's official evaluation gives no credit
        # for "unknown", so the request asks for a committed answer there. `task_answer_policy`
        # owns the mapping -- `{m3-bench-robot}` -- and its rationale.
        requested_policy = task_answer_policy(task_name, answer_policy)
        ask_stream = getattr(memory, "ask_stream", None)

        async def answered() -> AnswerResult:
            if ask_stream is None:
                answer: AnswerResult = await memory.ask(
                    content,
                    limit=recall_limit,
                    reference_at=question.reference_at,
                    answer_policy=requested_policy,
                )
                return answer
            first_token_seen = False
            streamed: AnswerResult | None = None
            async for chunk in ask_stream(
                content,
                limit=recall_limit,
                reference_at=question.reference_at,
                answer_policy=requested_policy,
            ):
                if chunk.text.strip() and not first_token_seen:
                    first_token_seen = True
                    if answer_span is not None:
                        answer_span.set_attribute(
                            OPERATION_TTFT,
                            (time.perf_counter() - started) * 1_000,
                        )
                if chunk.result is not None:
                    streamed = chunk.result
            if streamed is None:
                raise RuntimeError("answer stream ended without a terminal result")
            return streamed

        with _observe_retrieval_results(observe_retrieval):
            result = await attempt(answered)
    except Exception as error:
        if _systemic_embedding_stage_failure(error):
            raise _SystemicEmbeddingFailure(
                "systemic query embedding provider failure; answer matrix aborted"
            ) from error
        return _AnswerOutcome(
            "",
            (time.perf_counter() - latency_started) * 1_000,
            max((hit.score for hit in ranked), default=0.0),
            tuple(hit.id for hit in ranked),
            tuple(_evidence(hit) for hit in ranked),
            error=error,
            ranked_source_ids=_source_ids(ranked),
            ranked_source_ids_complete=ranked_complete,
        )
    # `_declined` stays on this path: the product cannot recognise a refusal a task worded
    # itself, so the harness counts it. It is deliberately not extended to the baseline arms
    # here, which would be new behaviour rather than a merge of the two intents.
    declined = _declined(result.answer, question)
    return _AnswerOutcome(
        result.answer,
        (time.perf_counter() - latency_started) * 1_000,
        max((hit.score for hit in result.hits), default=0.0),
        tuple(hit.id for hit in result.hits),
        tuple(_evidence(hit) for hit in result.hits),
        abstained=result.abstained or declined,
        abstention_reason=(
            result.abstention_reason.value
            if result.abstention_reason is not None
            else (AbstentionReason.INSUFFICIENT_EVIDENCE.value if declined else None)
        ),
        ranked_source_ids=_source_ids(ranked),
        ranked_source_ids_complete=ranked_complete,
    )


def _source_ids(hits: Sequence[SearchHit]) -> tuple[str, ...]:
    return tuple(_evidence(hit).source_id or "" for hit in hits)


def _question_text(question: EvalQuestion) -> str:
    return "\n".join(str(part) for part in question.content if not isinstance(part, Path))


def _full_context(items: Sequence[MemoryItem], budget_chars: int) -> str:
    """Stuff the corpus a retrieval arm would have searched, oldest first, under one budget.

    Media atoms have no text to stuff, so a media corpus reduces this arm to its prompt, and a
    corpus whose items all exceed the budget reduces it to the same thing. Both are lower bounds,
    recorded as such in `_arm_provenance`; neither is distinguishable from `blind` in the score.
    """
    parts: list[str] = []
    used = 0
    for item in items:
        text = "\n".join(str(atom) for atom in item.content if not isinstance(atom, Path))
        if not text:
            continue
        if used + len(text) > budget_chars:
            # Skipped, not `break`: breaking on the first item that does not fit discarded every
            # later memory, and when that item was the first one the arm answered from an empty
            # context under the full-context prompt -- a blind arm wearing the wrong label.
            continue
        used += len(text)
        parts.append(text)
    return "\n\n".join(parts)


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
    arm: _Arm = PRODUCT_ARM,
    answer_policy: AnswerPolicy | None = None,
) -> SampleResult:
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    excerpt_source_ids: tuple[str, ...]
    excerpt_evidence: tuple[EvidenceInterval, ...]
    error_code: str | None
    error_detail: FailureDetail | None
    retrieval_diagnostic_error: FailureDetail | None = None
    ranked_source_ids: tuple[str, ...] = ()
    ranked_source_ids_complete = False
    if isinstance(outcome, BaseException):
        prediction, latency_ms, confidence, memory_ids, evidence, cached = (
            "",
            0.0,
            0.0,
            (),
            (),
            False,
        )
        excerpt_source_ids, excerpt_evidence = (), ()
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
        excerpt_source_ids = outcome.excerpt_source_ids
        excerpt_evidence = outcome.excerpt_evidence
        cached = outcome.cached
        ranked_source_ids = outcome.ranked_source_ids
        ranked_source_ids_complete = outcome.ranked_source_ids_complete
        error_detail = None if outcome.error is None else _failure_detail(outcome.error)
        retrieval_diagnostic_error = (
            None
            if outcome.retrieval_diagnostic_error is None
            else _failure_detail(outcome.retrieval_diagnostic_error)
        )
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
        ranked_source_ids,
        predict_only=predict_only,
        arm=arm,
        # A question whose run recorded no complete ranked list -- a replay from a cache
        # written before the list was stored -- carries no retrieval score. Reporting the empty
        # list as recall zero would invent one, and the task's `retrieval` block excludes exactly
        # the same samples, so both surfaces report over one denominator.
        retrieval_available=(
            arm.retrieves and ranked_source_ids_complete and retrieval_diagnostic_error is None
        ),
        # A provider failure produced no answer; `_score` counts it as a wrong one. Transient
        # failures are waited out before they get here, so what remains is the system's loss.
        answer_failed=error_code is not None,
    )
    if (
        not isinstance(outcome, BaseException)
        and outcome.compiled_chars is not None
        and outcome.compiled_items is not None
    ):
        # "Useful evidence per token": the compile arm's own bundle size, reported per question
        # next to its other metrics so it never needs a second pass to reconstruct.
        metrics = {
            **metrics,
            "compile_bundle_chars": float(outcome.compiled_chars),
            "compile_bundle_items": float(outcome.compiled_items),
            "compile_bundle_excerpts": float(len(outcome.excerpt_source_ids)),
        }
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
        candidate_count=_candidate_count(unit, question),
        ingest_failure_count=ingest_failures,
        ingest_failures=ingest_failure_details,
        error_code=error_code,
        error_reason=None if error_detail is None else error_detail.reason,
        error_stage=None if error_detail is None else error_detail.stage,
        error_cause_type=None if error_detail is None else error_detail.cause_type,
        error_message=None if error_detail is None else error_detail.message,
        retrieval_diagnostic_error=retrieval_diagnostic_error,
        abstained=abstained,
        abstention_reason=abstention_reason,
        cached=cached,
        metadata=question.metadata,
        prompt=tuple(str(part) for part in question.content) if log_samples else None,
        references=question.references if log_samples else None,
        evidence=evidence,
        excerpt_source_ids=excerpt_source_ids,
        excerpt_evidence=excerpt_evidence,
        ref_at_300=(
            _reference_grounding(task, unit, question, evidence) if arm.generates else None
        ),
        metrics=metrics,
        scorer_protocol=scorer_protocol(task.spec.name),
        arm=arm.name,
        # Only the product arm answering through `ask` requests a policy: a baseline row, and a
        # product row on a task that answers through `Memory.compile`, carry none.
        answer_policy=(
            task_answer_policy(task.spec.name, answer_policy)
            if _answers_through_ask(arm, task.spec.name)
            else None
        ),
        retrieval_candidates=len(ranked_source_ids),
        ranked_source_ids=tuple(ranked_source_ids),
        ranked_source_ids_complete=ranked_source_ids_complete,
    )


def _score(
    task_name: str,
    question: EvalQuestion,
    prediction: str,
    parsed_choice: str | None,
    ranked_source_ids: Sequence[str],
    *,
    predict_only: bool,
    arm: _Arm = PRODUCT_ARM,
    retrieval_available: bool = True,
    answer_failed: bool = False,
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
        evidence_source_ids=tuple(ranked_source_ids),
    )
    metrics = _arm_metrics(metrics, arm, retrieval_available=retrieval_available)
    if answer_failed:
        # A question the system did not answer is a question it got wrong: it scores zero under
        # the task's primary metric, and the separately measured retrieval diagnostic stays.
        # Judge-scored families carry no local answer metric, so the zero is written explicitly
        # rather than read off the empty prediction, and the derived `joint_*` metrics follow.
        # The earlier rule left it unscored so one failure did not deflate the arms unevenly,
        # at the price of a single residual failure invalidating a task's whole score; the
        # failure is still counted and its provider message kept for diagnosis.
        failed = finalize_scores(
            task_name,
            {
                **metrics,
                sample_primary_metric(task_name): 0.0,
                **({"exact_match": 0.0} if "exact_match" in metrics else {}),
            },
        )
        return failed, 0.0, failed.get("exact_match")
    score = metrics.get(sample_primary_metric(task_name), metrics.get("token_f1"))
    return metrics, score, metrics.get("exact_match")


def _arm_metrics(
    metrics: Mapping[str, float], arm: _Arm, *, retrieval_available: bool
) -> dict[str, float]:
    """Keep only the metrics an arm can honestly carry.

    An arm that never retrieved has no retrieval score -- reporting zero would read as a
    retrieval failure -- and an arm that never generated has no answer score.
    """
    diagnostic = ("retrieval_", "joint_")
    return {
        name: value
        for name, value in metrics.items()
        if (retrieval_available or not name.startswith(diagnostic))
        and (arm.generates or name.startswith(diagnostic))
    }


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


def _excerpt_evidence(excerpt: ContextExcerpt) -> EvidenceInterval:
    """Describe a partial source without promoting it to full grounded evidence."""
    context = excerpt.context
    return EvidenceInterval(
        excerpt.source_memory_id,
        None if context is None else context.source_id,
        None,
        None,
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
    if task_name == "video-mme-v2":
        from mindbridge.benchmarks.video_mme_v2 import parse_video_mme_v2_option

        return parse_video_mme_v2_option(prediction)
    return parse_choice(prediction, choices)


def _incomplete_search_replay(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        if row.get("arm") != DEFAULT_ARM:
            continue
        performance = row.get("performance")
        search = performance.get("search_e2e") if isinstance(performance, Mapping) else None
        if isinstance(search, Mapping) and search.get("complete") is False:
            return True
    return False


def _task_rows(
    arguments: _Arguments,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    batch_sizes: Mapping[str, int],
    performance: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    memory_config: MindBridgeConfig | None,
) -> list[dict[str, object]]:
    """Score one row per (task, arm).

    Separate from `_results` so a task that has finished answering can be scored and printed on
    its own, with exactly the arithmetic the final document uses, while later tasks still run.
    """
    # A blind arm run here is the same control as an external `--blind` document, measured on the
    # same inputs, so it satisfies the blind control too. An explicitly supplied document still
    # wins: the caller named it.
    blind_rows = {
        **_in_run_blind_rows(arguments, tasks, samples),
        **_blind_baseline_rows(arguments.blind_baseline, tasks),
    }
    product_candidate_limit = _answer_retrieval_candidate_limit(
        arguments.recall_limit,
        memory_config,
    )
    task_rows = []
    for task, arm in ((task, arm) for task in tasks for arm in arguments.arms):
        selected = tuple(
            sample for sample in samples if sample.task == task.spec.name and sample.arm == arm
        )
        metrics = _metrics(
            task,
            selected,
            arguments,
            blind_rows.get(task.spec.name),
            arm=arm,
            retrieval_candidate_limit=(
                product_candidate_limit
                if arm == DEFAULT_ARM
                else RETRIEVAL_CANDIDATE_LIMIT
                if (
                    arm == "random"
                    and any(
                        retrieval_gold_ids(task.spec.name, sample.metadata) for sample in selected
                    )
                )
                else arguments.recall_limit
            ),
        )
        task_rows.append(
            {
                "arm": arm,
                "task": task.spec.name,
                # The request policy, so a run that asked for a committed answer is not
                # byte-indistinguishable from every earlier run of the same task. Only the
                # product arm answering through `ask` requests one, so the baseline arms and a
                # task answered through `Memory.compile` carry no policy.
                "answer_policy": (
                    task_answer_policy(task.spec.name, arguments.answer_policy)
                    if _answers_through_ask(arm, task.spec.name)
                    else None
                ),
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
                "fallback_reference_at": (
                    None
                    if task.fallback_reference_at is None
                    else task.fallback_reference_at.isoformat()
                ),
                "fallback_reference_question_count": task.fallback_reference_question_count,
                "batch_size": batch_sizes[task.spec.name],
                "input_modalities": _task_modalities(task),
                "performance": dict(performance.get(task.spec.name, {}).get(arm, {})),
                **metrics,
            }
        )

    return task_rows


def _results(
    arguments: _Arguments,
    config: ModelConfig,
    judge_config: _JudgeConfig,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    duration_seconds: float,
    batch_sizes: Mapping[str, int],
    performance: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    memory_config: MindBridgeConfig | None = None,
    resources: Mapping[str, object] | None = None,
    embedding_warmup_count: int | None = None,
) -> dict[str, object]:
    fallback_reference_at = getattr(arguments, "fallback_reference_at", None)
    task_rows = _task_rows(
        arguments,
        tasks,
        samples,
        batch_sizes,
        performance,
        memory_config=memory_config,
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
                sample.error_code is not None
                or sample.ingest_failure_count
                or sample.retrieval_diagnostic_error is not None
                for sample in samples
            )
            or _incomplete_search_replay(task_rows)
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
        "resume": arguments.resume,
        "cached_response_count": sum(sample.cached for sample in samples),
        "cached_judge_count": sum(sample.judge_cached for sample in samples),
        "abstentions": _abstentions(samples),
        "blind": arguments.blind,
        "blind_baseline": (
            None if arguments.blind_baseline is None else str(arguments.blind_baseline)
        ),
        "controls_complete": all(
            cast(Mapping[str, object], row["controls"])["interpretable"] for row in task_rows
        ),
        "allow_unverified_data": arguments.allow_unverified_data,
        "limit": arguments.limit,
        "offset": arguments.offset,
        "fallback_reference_at": (
            None if fallback_reference_at is None else fallback_reference_at.isoformat()
        ),
        "fallback_reference_question_count": sum(
            task.fallback_reference_question_count for task in tasks
        ),
        "data_root": str(arguments.data_root),
        "media_manifest_path": (
            None if arguments.media_manifest is None else str(arguments.media_manifest.resolve())
        ),
        "media_roots": media_roots,
        "unit_concurrency": arguments.unit_concurrency,
        "request_concurrency": arguments.request_concurrency,
        "recall_limit": arguments.recall_limit,
        "answer_policy_override": arguments.answer_policy,
        # P5/P6: without this the configured consolidator was constructed, registered for close,
        # and never called, and nothing in the report said so.
        "deliberation": {
            "enabled": arguments.deliberate,
            "operations_applied": _deliberation_applied,
        },
        "arms": _arm_provenance(arguments, memory_config),
        "measurement_protocol": _measurement_protocol(
            arguments,
            samples,
            embedding_warmup_count=embedding_warmup_count,
        ),
        "model": _model_result(
            arguments,
            config,
            memory_config,
            embedding_warmup_count=embedding_warmup_count,
        ),
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
            "runtime_versions": {
                name: _version(name)
                for name in ("torch", "transformers", "opentelemetry-sdk", "openai", "funasr")
            },
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hardware": hardware_metadata(),
            "acceleration_runtime": acceleration_runtime_metadata(),
            "source": source_metadata(),
        },
        "resources": None if resources is None else dict(resources),
        "tasks": task_rows,
    }


def _measurement_protocol(
    arguments: _Arguments,
    samples: Sequence[SampleResult],
    *,
    embedding_warmup_count: int | None,
) -> dict[str, object]:
    cached = sum(sample.cached for sample in samples)
    state = (
        "fresh_store"
        if arguments.use_cache is None
        else "response_cache_only"
        if cached == len(samples)
        else "mixed_fresh_store_and_response_cache"
        if cached
        else "fresh_store_with_response_cache_enabled"
    )
    return {
        "state": state,
        "store": (
            "one reused-or-created physical data directory per benchmark unit"
            if arguments.resume
            else "one newly-created physical data directory per benchmark unit"
        ),
        "repeat_index": getattr(arguments, "repeat_index", 0),
        "repeat_execution": "independent_eval_invocation",
        "measured_response_count": len(samples) - cached,
        "cached_responses_excluded_from_performance_denominators": True,
        "embedding_warmup": {
            "count": embedding_warmup_count,
            "task": EmbedTask.QUERY.value,
            "included_in_product_measurement": False,
        },
        "generation_warmup": {"count": 0},
        "vision_description_cache": "shared_within_run_only; cold_again_for_each_repeat",
        "remote_server_state": "uncontrolled",
        "answer_e2e_includes_request_admission": True,
        "post_answer_search_replay_in_client_resource_window": True,
        "judge_in_client_resource_window": False,
        "judge_measurement_exclusion": (
            "interleaved sub-windows removed from client and model-server counters"
            if getattr(arguments, "stream_results", False) and not arguments.predict_only
            else "judging runs after the product measurement window"
        ),
    }


def _arm_provenance(
    arguments: _Arguments,
    memory_config: MindBridgeConfig | None = None,
) -> dict[str, object]:
    """Describe every arm precisely enough that a reader can attribute each number to one."""
    answer_candidate_limit = _answer_retrieval_candidate_limit(
        arguments.recall_limit,
        memory_config,
    )
    evidence_budget_chars = _evidence_budget_chars(memory_config)
    definitions: dict[str, object] = {
        DEFAULT_ARM: {
            "answers_from": (
                "retrieved memories through Memory.ask, or the generator over Memory.compile's "
                "rendered bundle on the tasks answer_surface maps to compile"
            ),
            "answer_surface": {task: task_answer_surface(task) for task in sorted(arguments.tasks)},
            "compile_surface_prompt": COMPILE_SURFACE_PROMPT_VERSION,
            "retrieval": (
                "Memory.ask in-answer ranked list; no second scoring search. None on a compile "
                "task: Memory.compile exposes no ranked list"
            ),
            "retrieval_candidate_limit": answer_candidate_limit,
            "retrieval_candidate_limit_basis": (
                "full rerank pool because evidence_budget_chars is configured"
                if evidence_budget_chars is not None
                else "min(100, recall_limit * 3)"
            ),
            "answer_retrieval_candidate_limit": answer_candidate_limit,
            "answer_retrieval_candidate_limit_basis": (
                "full rerank pool because evidence_budget_chars is configured"
                if evidence_budget_chars is not None
                else "min(100, recall_limit * 3)"
            ),
            "official_metrics": True,
        },
        "blind": {
            "answers_from": "the generator's prior, with no evidence",
            "retrieval": None,
            "prompt": BLIND_PROMPT_VERSION,
            "media_in_question": "dropped: the baseline prompt is text only",
            "official_metrics": False,
        },
        "full-context": {
            "answers_from": "the corpus stuffed into the prompt, oldest first",
            "retrieval": None,
            "prompt": FULL_CONTEXT_PROMPT_VERSION,
            "budget_chars": arguments.full_context_chars,
            "media_in_corpus": "dropped: only text atoms are stuffed",
            "official_metrics": False,
        },
        "random": {
            "answers_from": "nothing; retrieval metrics only",
            "retrieval": (
                f"uniform shuffle of the top {RETRIEVAL_CANDIDATE_LIMIT} ranked candidates for "
                "gold-labelled questions; recall_limit otherwise"
            ),
            "retrieval_candidate_limit": {
                "gold_labelled_questions": RETRIEVAL_CANDIDATE_LIMIT,
                "questions_without_gold_labels": arguments.recall_limit,
            },
            "seed": arguments.seed,
            "official_metrics": False,
        },
        "compile": {
            "answers_from": "Memory.compile's rendered bundle",
            "retrieval": "Memory.compile",
            "prompt": FULL_CONTEXT_PROMPT_VERSION,
            "budget_max_items": arguments.compile_max_items,
            "budget_max_chars": arguments.compile_max_chars,
            "allow_partial_sources": getattr(arguments, "compile_allow_partial_sources", False),
            "official_metrics": False,
        },
    }
    return {
        "selected": list(arguments.arms),
        "retrieval_candidate_limit": answer_candidate_limit,
        "retrieval_candidate_limit_arm": DEFAULT_ARM,
        "retrieval_candidate_source": (
            "Memory.ask in-answer ranked list; none on a task answered through Memory.compile"
        ),
        "search_e2e_limit": arguments.recall_limit,
        "search_e2e_source": "post-answer public Memory.search replay",
        "evidence_budget_chars": evidence_budget_chars,
        "ingest": arguments.ingest,
        "definitions": {name: definitions[name] for name in arguments.arms},
    }


def _model_result(
    arguments: _Arguments,
    config: ModelConfig,
    memory_config: MindBridgeConfig | None,
    *,
    embedding_warmup_count: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "adapter": arguments.model,
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "embedding_revision": DEFAULT_JINA_REVISION,
        "embedding_dimension": DEFAULT_JINA_DIMENSION,
        "embedding_warmup": {
            "count": embedding_warmup_count,
            "task": EmbedTask.QUERY.value,
            "included_in_product_measurement": False,
        },
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
        "memory_config": None,
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


def _task_modalities(task: LoadedTask) -> list[str]:
    """List the atomic input modalities this task actually routes through the SDK."""
    modalities = (
        {Modality.TEXT.value}
        if any(
            isinstance(atom, str)
            for unit in task.units
            for item in unit.memories
            for atom in item.content
        )
        else set()
    )
    for unit in task.units:
        for item in unit.memories:
            for atom in item.content:
                if isinstance(atom, Path):
                    modality = _MODALITY_BY_SUFFIX.get(atom.suffix.casefold())
                    if modality is not None:
                        modalities.add(modality.value)
    return sorted(modalities)


def _configured_device_label(explicit: str | None, config: MindBridgeConfig) -> str:
    if explicit is not None:
        return explicit
    devices = []
    if config.embedding.provider in {"jina-omni", "sentence-transformers"}:
        devices.append(config.embedding.device or "auto")
    if config.speech is not None and config.speech.provider == "funasr":
        devices.append(config.speech.device)
    return ",".join(dict.fromkeys(devices)) or "remote"


def _report_completions(callback: Callable[[], None] | None, count: int) -> None:
    if callback is not None:
        for _ in range(count):
            callback()


def _ignore_store_ready() -> None:
    pass


def _ignore_activity(_activity: str) -> None:
    pass


def _ignore_ingested(_written: int) -> None:
    pass


def _activity_summary(activities: Sequence[str | None], total: int) -> str:
    active = tuple((index, phase) for index, phase in enumerate(activities) if phase is not None)
    if not active:
        return "finishing"
    if len(active) == 1:
        index, phase = active[0]
        return f"{phase} unit {index + 1}/{total}"
    counts: dict[str, int] = {}
    for _index, phase in active:
        counts[phase] = counts.get(phase, 0) + 1
    phases = ", ".join(f"{phase} {count}" for phase, count in counts.items())
    return f"{len(active)} active units: {phases}"


_first_ingest_failure_announced = False
# Run-global like `_first_ingest_failure_announced` above and for the same reason: the whole
# evaluation runs inside one `asyncio.run`, so this is single-threaded, and threading a counter
# back through four call layers would change five signatures to report one integer.
_deliberation_applied = 0


def _announce_first_ingest_failure(error: BaseException, source_id: str) -> None:
    """Say once, immediately, that writes are failing.

    Ingest failures are bisected, counted and reported in the final table's `unwritten` column,
    which is right for a corpus with a few unreadable items and wrong for a misconfiguration that
    fails every write: a run with an embedding dimension the model does not produce looked alive
    for fourteen minutes while every store on the machine stayed empty. One line at the first
    failure carries the error text the summary cannot.
    """
    global _first_ingest_failure_announced
    if _first_ingest_failure_announced:
        return
    _first_ingest_failure_announced = True
    detail = _failure_detail(error, source_id=source_id)
    reason = "" if detail.reason is None else f"/{detail.reason}"
    _announce(
        f"first ingest failure at source {source_id} ({detail.code}{reason}): {error}"
        " -- further failures are counted in the unwritten column"
    )


def _evaluation_devices(
    explicit: str | None,
    config: MindBridgeConfig | None,
    *,
    needs_speech: bool = True,
) -> tuple[str | None, ...]:
    if config is None:
        return (explicit,)
    configured = []
    if config.embedding.provider in {"jina-omni", "sentence-transformers"}:
        configured.append(explicit or config.embedding.device or "auto")
    # The speech backend is only constructed when a unit carries audio or video, so a text-only
    # run must not hold the GPU lock for a model it never loads: that lock serialized every
    # text task behind one video run on a shared card.
    if needs_speech and config.speech is not None and config.speech.provider == "funasr":
        configured.append(explicit or config.speech.device)
    devices: dict[str, str | None] = {}
    for device in sorted(set(configured)):
        normalized = device.strip().lower()
        identity = None if normalized == "cpu" else _physical_cuda_identity(normalized)
        devices.setdefault(identity or normalized, None if normalized == "auto" else device)
    return tuple(devices[identity] for identity in sorted(devices))


def _content(parts: tuple[str | Path, ...]) -> str | Path | tuple[str | Path, ...]:
    return parts[0] if len(parts) == 1 else parts


def _memory_content(item: MemoryItem) -> str | tuple[str | Path, ...]:
    first, *rest = item.content
    if not isinstance(first, str):
        # A media item's source id is already metadata, which is where `_evidence` reads it. As
        # content it was a third retrieval key and made the aggregate key differ from the clip's
        # own key even for a clip with no speech, so every clip went to the embedder twice; a
        # clip with a transcript still does, because its aggregate key carries the transcript.
        return item.content
    labelled = f"[source_id: {item.source_id}]\n{first}"
    return labelled if not rest else (labelled, *rest)


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


def _planned_ingest_count(unit: EvalUnit) -> int:
    """Count the memories a unit will write, so an ingest bar has a total before it starts.

    The causal cursor stops at the latest cutoff any question asks for, so a unit whose questions
    all carry one never writes its tail. Counting from every question rather than from the pending
    set holds the total still when a resumed run answers only some of them.
    """
    boundary = -math.inf
    for question in unit.questions:
        cutoff = math.inf if question.cutoff_seconds is None else question.cutoff_seconds
        boundary = max(boundary, cutoff)
    return sum(1 for item in unit.memories if _memory_end(item) <= boundary)


def _prefix_end(memories: Sequence[MemoryItem], cutoff: float | None, start: int = 0) -> int:
    """Return how many ordered memories a question at ``cutoff`` is allowed to have seen."""
    boundary = math.inf if cutoff is None else cutoff
    end = start
    while end < len(memories) and _memory_end(memories[end]) <= boundary:
        end += 1
    return end


def _ingest_digest(
    task: LoadedTask,
    arguments: _Arguments,
    memory_config: MindBridgeConfig | None,
) -> str:
    """Identify everything that decides what one unit's store ends up holding.

    A resumed run trusts an existing store only when this matches, because a store written by
    another embedder, ingest mode, or dataset revision is not the store this run would write.
    The generator is deliberately absent: it answers questions and never writes.
    """
    payload: dict[str, object] = {
        "runner": EVAL_RUNNER_VERSION,
        "implementation": _implementation_identity(),
        "task": _cache_task(task),
        "embedding_model": DEFAULT_JINA_MODEL_ID,
        "embedding_revision": DEFAULT_JINA_REVISION,
        "transcription_model": DEFAULT_FUNASR_MODEL_ID,
        "device": arguments.device or "auto",
        "ingest": arguments.ingest,
        # `--deliberate` applies consolidation operations to the store between chunks, so a store
        # built with it holds different memories than one built without it.
        "deliberate": arguments.deliberate,
        # Every unoverridden dataset and media path is resolved under this root, and two roots
        # can hold differently prepared copies of the same pinned corpus.
        "benchmarks_root": str(arguments.benchmarks_root),
        "media_manifest": (
            None if arguments.media_manifest is None else str(arguments.media_manifest)
        ),
        "media_root": str(arguments.media_overrides.get(task.spec.name, "")),
    }
    if memory_config is not None:
        payload["memory_config"] = _memory_config_payload(
            memory_config, exclude=_ANSWER_ONLY_SECTIONS
        )
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


# `generation` builds the answerer and nothing else on this path: it answers questions and never
# writes one. Naming it here would rebuild every store in the run when only the answer model
# changed. A configured consolidator may reuse that answerer, but consolidation runs only when a
# caller asks for it and no evaluation does.
_ANSWER_ONLY_SECTIONS = frozenset({"generation"})


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
        # The override and the resolved per-task policy, because either one changes what the
        # request asked for. Without them a `best_effort` arm replayed a cached `strict` refusal
        # and labelled it `best_effort`, and widening `BEST_EFFORT_TASKS` replayed a task's older
        # strict answers under its new default.
        "answer_policy_override": arguments.answer_policy,
        "answer_policy": {
            task: task_answer_policy(task, arguments.answer_policy)
            for task in sorted(arguments.tasks)
        },
        # The surface for the same reason: widening `COMPILE_SURFACE_TASKS` must not replay a
        # task's cached `ask` answers as answers from the compiled bundle.
        "answer_surface": {task: task_answer_surface(task) for task in sorted(arguments.tasks)},
        "compile_surface_prompt": COMPILE_SURFACE_PROMPT_VERSION,
        "blind": arguments.blind,
        "batch_sizes": dict(sorted(batch_sizes.items())),
        "ingest": arguments.ingest,
        "deliberate": arguments.deliberate,
        "compile_max_items": arguments.compile_max_items,
        "compile_max_chars": arguments.compile_max_chars,
        "compile_allow_partial_sources": getattr(arguments, "compile_allow_partial_sources", False),
    }
    fallback_reference_at = getattr(arguments, "fallback_reference_at", None)
    if fallback_reference_at is not None:
        payload["fallback_reference_at"] = fallback_reference_at.isoformat()
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


def _cache_task(task: LoadedTask, arm: _Arm = PRODUCT_ARM) -> str:
    identity = f"{task.spec.name}:{task.spec.adapter_version}:{task.evaluation_sha256}"
    return identity if arm.name == DEFAULT_ARM else f"{arm.name}:{identity}"


def _all_cached(
    cache: ResponseCache,
    tasks: Sequence[LoadedTask],
    arms: Sequence[_Arm] = (PRODUCT_ARM,),
) -> bool:
    return all(
        cache.get(_cache_task(task, arm), unit.unit_id, question.question_id) is not None
        for task in tasks
        for arm in arms
        for unit in task.units
        for question in unit.questions
    )


def _cache_only_memory(_path: Path) -> _MemoryContext:
    raise RuntimeError("response cache was incomplete after the cache-only preflight")


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


if __name__ == "__main__":
    raise SystemExit(main())
