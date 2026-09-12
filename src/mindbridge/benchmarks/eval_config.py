"""Command-line parsing and configuration resolution for mindbridge-bench eval."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast, get_args, overload

import yaml
from pydantic import SecretStr

if TYPE_CHECKING:
    pass
from mindbridge import (
    AnswerPolicy,
    ContextBudget,
    MindBridgeConfig,
    Modality,
)
from mindbridge.benchmarks.eval_adapters import LoadedTask
from mindbridge.benchmarks.eval_regression import PERFORMANCE_BUDGET_NAMES
from mindbridge.benchmarks.model_config import (
    DEFAULT_TIMEOUT_SECONDS,
    DownloadSettings,
    HarnessOverrides,
    ModelConfig,
)
from mindbridge.benchmarks.task_catalog import expand
from mindbridge.configuration import _absolute_http_url

DEFAULT_ARM = "mindbridge"


BASELINE_ARMS = ("blind", "full-context", "random", "compile")


ARMS = (DEFAULT_ARM, *BASELINE_ARMS)


DEFAULT_FULL_CONTEXT_CHARS = 24_000


DEFAULT_INGEST_MODE = "add"


INGEST_MODES = (DEFAULT_INGEST_MODE, "capture")


# A frozen dataclass instance is a safe default argument value; naming it once avoids repeating
# the call (and the linter's objection to a call in a default) at every threading site below.
DEFAULT_COMPILE_BUDGET = ContextBudget()


# The largest ranked window requested by an answer or the random retrieval control.
RETRIEVAL_CANDIDATE_LIMIT = 100


# The `compile` arm reuses this prompt verbatim rather than defining its own: its context is
# `ContextBundle.render()` instead of the raw stuffed corpus, but it is still context handed to
# the same generator the same way, so it is honestly the same prompt, not a new one to version.
DEFAULT_BOOTSTRAP_SAMPLES = 2_000


_RESULTS_FILE = "results.jsonl"


_SAMPLES_FILE = "samples.jsonl"


_CONFIG_FILE = "config.yaml"


# Written task by task while the run is still going, and removed once the real artifacts land. A
# multi-task run used to hold every sample in memory until the last task finished, so an upstream
# outage during task five threw away four tasks of answers. This file is never an evaluation
# artifact -- it carries no results document and no digest -- it is the crash copy.
_PARTIAL_SAMPLES_FILE = "samples.partial.jsonl"


_MEDIA_MANIFEST_FILE = "media-manifest.jsonl"


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
    # None means every task keeps the policy its own protocol calls for; a value overrides the
    # whole table, which is how a run measures the policy itself rather than one task's protocol.
    answer_policy: AnswerPolicy | None
    seed: int
    seeds: tuple[int, int, int, int]
    bootstrap_samples: int
    repeat_index: int
    model: str
    arms: tuple[str, ...]
    full_context_chars: int
    ingest: str
    deliberate: bool
    compile_max_items: int
    compile_max_chars: int
    compile_allow_partial_sources: bool
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
    performance_budgets: Mapping[str, float]
    blind: bool
    blind_baseline: Path | None
    fail_on_regression: bool
    regression_threshold: float
    predict_only: bool
    log_samples: bool
    stream_results: bool
    allow_unverified_data: bool
    download: bool
    overwrite: bool
    resume: bool
    quiet: bool
    fallback_reference_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _JudgeConfig:
    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    concurrency: int = 4

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("judge model must not be blank")
        if not _absolute_http_url(self.base_url):
            raise ValueError("judge base URL must be an absolute http(s) URL")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("judge timeout must be positive")
        if self.concurrency <= 0:
            raise ValueError("judge concurrency must be positive")


def _memory_config_payload(
    config: MindBridgeConfig,
    *,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, object]:
    return cast(
        dict[str, object],
        config.model_dump(mode="json", exclude={"data_dir", *exclude}),
    )


def _build_parser(prog: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="mindbridge", help="evaluation adapter")
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "comma-separated evaluation arms: mindbridge (the product), blind (no evidence), "
            "full-context (corpus stuffed into the prompt), random (shuffled candidates, "
            "retrieval metrics only), compile (Memory.compile's rendered bundle). Baseline arms "
            "share one ingest and are never official."
        ),
    )
    parser.add_argument(
        "--full-context-chars",
        type=_positive_int,
        default=None,
        help="character budget the full-context arm stuffs into one prompt",
    )
    parser.add_argument(
        "--compile-max-items",
        type=_positive_int,
        default=None,
        help="ContextBudget.max_items for the compile arm",
    )
    parser.add_argument(
        "--compile-max-chars",
        type=_positive_int,
        default=None,
        help="ContextBudget.max_chars for the compile arm",
    )
    parser.add_argument(
        "--compile-allow-partial-sources",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="opt the compile arm into verified partial raw-text sources",
    )
    parser.add_argument(
        "--deliberate",
        action="store_true",
        help=(
            "run Memory.deliberate() after each cutoff's ingest and before its questions, so "
            "the slow loop's effect on QA scores is measurable; requires a config declaring "
            "`consolidation`"
        ),
    )
    parser.add_argument(
        "--fallback-reference-at",
        type=_fallback_reference_at,
        help=(
            "timezone-aware ISO 8601 clock used only when a selected question and its corpus "
            "provide no reference time"
        ),
    )
    parser.add_argument(
        "--ingest",
        choices=INGEST_MODES,
        default=None,
        help=(
            "how memories reach the store: add (Memory.add_many/add, the strong default) or "
            "capture (Memory.capture then Memory.settle, so capture acknowledgement and "
            "time-to-searchable are measured against a real run)"
        ),
    )
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
        help=(
            "YAML MindBridgeConfig plus an optional benchmark section; absent sections take "
            "defaults and data_dir is replaced by isolated benchmark directories"
        ),
    )
    parser.add_argument(
        "--judge-model-args",
        "--judge_model_args",
        default="",
        help="comma-separated model/base_url/api_key/timeout_seconds overrides",
    )
    parser.add_argument("--tasks", help="comma-separated task names or groups")
    parser.add_argument("--list-tasks", action="store_true", help="list task pins and readiness")
    # Defaults are resolved after the configuration file is read, so an unset flag must stay
    # distinguishable from one the caller typed.
    parser.add_argument("--benchmarks-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-path", "--output_path", type=Path)
    parser.add_argument("--run-id", type=_run_identifier)
    parser.add_argument(
        "--task-data",
        action="append",
        default=None,
        metavar="TASK=PATH",
        help="override one task's annotation path; repeatable",
    )
    parser.add_argument(
        "--media-root",
        action="append",
        default=None,
        metavar="TASK=PATH",
        help="override one task's media root; repeatable",
    )
    parser.add_argument("--media-manifest", type=Path, help="prepared clip/caption manifest")
    parser.add_argument(
        "--limit",
        type=_limit_value,
        help="all (or -1), a 0-1 fraction, or an absolute example count",
    )
    parser.add_argument("--offset", type=_nonnegative_int, default=None)
    parser.add_argument("--num-fewshot", "--num_fewshot", type=_nonnegative_int, default=None)
    parser.add_argument(
        "--gen-kwargs",
        "--gen_kwargs",
        default="",
        help="deterministic generation settings; supports max_tokens and enable_thinking",
    )
    parser.add_argument("--batch-size", "--batch_size", "-b", default=None)
    parser.add_argument("--max-batch-size", "--max_batch_size", type=_positive_int, default=None)
    parser.add_argument("--unit-concurrency", type=_positive_int, default=None)
    parser.add_argument("--request-concurrency", type=_positive_int, default=None)
    parser.add_argument("--judge-concurrency", type=_positive_int, default=None)
    parser.add_argument("--recall-limit", type=_positive_int, default=None)
    parser.add_argument(
        "--answer-policy",
        choices=get_args(AnswerPolicy),
        default=None,
        help=(
            "override every task's answer policy; unset keeps each task's official protocol"
            " (mindbridge.benchmarks.prompts.BEST_EFFORT_TASKS)"
        ),
    )
    parser.add_argument("--seed", type=_seed_values, default=None)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--repeat-index",
        type=_nonnegative_int,
        default=None,
        help="zero-based index of this independent baseline invocation",
    )
    parser.add_argument("--device", help="local embedding/FunASR device: cpu, cuda, or cuda:N")
    parser.add_argument(
        "--device-lock",
        action=argparse.BooleanOptionalAction,
        default=None,
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
    parser.add_argument(
        "--performance-budget",
        action="append",
        default=None,
        metavar="METRIC=FRACTION",
        help=(
            "maximum relative performance regression; repeatable; metrics: "
            + ", ".join(PERFORMANCE_BUDGET_NAMES)
        ),
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "run the no-memory control: answer every question through the public path with "
            "nothing ingested, so the score measures the generator instead of the memory"
        ),
    )
    parser.add_argument(
        "--blind-baseline",
        "--blind_baseline",
        type=Path,
        help="results.jsonl from a --blind run of the same evaluation inputs",
    )
    parser.add_argument("--fail-on-regression", action="store_true", default=None)
    parser.add_argument("--regression-threshold", type=_nonnegative_float, default=None)
    parser.add_argument("--predict-only", "--predict_only", "-x", action="store_true", default=None)
    parser.add_argument("--log-samples", "--log_samples", action="store_true", default=None)
    parser.add_argument(
        "--stream-results",
        "--stream_results",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "judge and print each task's table as soon as that task finishes answering (default); "
            "use --no-stream-results to defer all judging until every task has answered"
        ),
    )
    parser.add_argument("--allow-unverified-data", action="store_true", default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="download missing pinned annotations and selected media",
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse the stores and ingest checkpoints of an interrupted run with the same "
            "--run-id instead of ingesting every unit again"
        ),
    )
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument(
        "--verbosity",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=None,
    )
    parser.add_argument("--check-integrity", "--check_integrity", action="store_true")
    return parser


def _selected_arms(
    parser: argparse.ArgumentParser, parsed: argparse.Namespace, declared: str
) -> tuple[str, ...]:
    """Resolve which arms this run answers, coupling `--blind` to the arm that has no evidence."""
    arms = tuple(dict.fromkeys(part.strip() for part in declared.split(",") if part.strip()))
    if not arms:
        parser.error("--arms must name at least one arm")
    unknown = tuple(name for name in arms if name not in ARMS)
    if unknown:
        parser.error(f"unknown arm(s): {', '.join(unknown)}; choose from {', '.join(ARMS)}")
    if not parsed.blind:
        return arms
    # `--blind` labels the whole run as the control, in the results document and in the
    # response-cache namespace, so it has to select the arm that actually answers without
    # evidence. Leaving the product arm running under that label is how a memory-backed run gets
    # published as its own baseline.
    if declared != DEFAULT_ARM and arms != ("blind",):
        parser.error("--blind runs the blind arm alone; drop --arms or pass --arms blind")
    return ("blind",)


def _arguments(
    parser: argparse.ArgumentParser,
    parsed: argparse.Namespace,
    download: DownloadSettings | None = None,
    overrides: HarnessOverrides | None = None,
) -> _Arguments:
    if parsed.model != "mindbridge":
        parser.error("--model must be mindbridge")
    harness = HarnessOverrides() if overrides is None else overrides
    run = harness.run
    arms = _selected_arms(parser, parsed, _picked(parsed.arms, run.arms, DEFAULT_ARM))
    tasks_value = parsed.tasks or run.tasks
    if not tasks_value:
        parser.error("--tasks is required unless --list-tasks is used")
    # A flag beats the file; the file beats the built-in default. Every constant below is the
    # default the flag used to carry, moved here so that "unset" stays distinguishable.
    offset = _picked(parsed.offset, run.offset, 0)
    num_fewshot = _picked(parsed.num_fewshot, run.num_fewshot, 0)
    batch_size = _picked(parsed.batch_size, run.batch_size, "auto")
    max_batch_size = _picked(parsed.max_batch_size, run.max_batch_size, 64)
    unit_concurrency = _picked(parsed.unit_concurrency, run.unit_concurrency, 1)
    request_concurrency = _picked(parsed.request_concurrency, run.request_concurrency, 4)
    judge_concurrency = _picked(parsed.judge_concurrency, run.judge_concurrency, 8)
    recall_limit = _picked(parsed.recall_limit, run.recall_limit, 20)
    answer_policy = _picked(parsed.answer_policy, run.answer_policy, None)
    full_context_chars = _picked(
        parsed.full_context_chars, run.full_context_chars, DEFAULT_FULL_CONTEXT_CHARS
    )
    compile_max_items = _picked(
        parsed.compile_max_items, run.compile_max_items, DEFAULT_COMPILE_BUDGET.max_items
    )
    compile_max_chars = _picked(
        parsed.compile_max_chars, run.compile_max_chars, DEFAULT_COMPILE_BUDGET.max_chars
    )
    compile_allow_partial_sources = _picked(
        parsed.compile_allow_partial_sources,
        run.compile_allow_partial_sources,
        False,
    )
    ingest = _picked(parsed.ingest, run.ingest, DEFAULT_INGEST_MODE)
    bootstrap_samples = _picked(
        parsed.bootstrap_samples, run.bootstrap_samples, DEFAULT_BOOTSTRAP_SAMPLES
    )
    repeat_index = _picked(parsed.repeat_index, run.repeat_index, 0)
    device = _picked(parsed.device, run.device, None)
    device_lock = _picked(parsed.device_lock, run.device_lock, True)
    use_cache = _picked(parsed.use_cache, run.use_cache, None)
    compare = _picked(parsed.compare, run.compare, None)
    fail_on_regression = _picked(parsed.fail_on_regression, run.fail_on_regression, False)
    regression_threshold = _picked(parsed.regression_threshold, run.regression_threshold, 0.0)
    predict_only = _picked(parsed.predict_only, run.predict_only, False)
    log_samples = _picked(parsed.log_samples, run.log_samples, False)
    stream_results = _picked(getattr(parsed, "stream_results", None), run.stream_results, True)
    allow_unverified = _picked(parsed.allow_unverified_data, run.allow_unverified_data, False)
    download_inputs = _picked(parsed.download, run.download, True)
    overwrite = _picked(parsed.overwrite, run.overwrite, False)
    quiet = _picked(parsed.quiet, run.quiet, False)
    verbosity = _picked(parsed.verbosity, run.verbosity, "INFO")
    media_manifest = _picked(parsed.media_manifest, run.media_manifest, None)
    if fail_on_regression and compare is None:
        parser.error("--fail-on-regression requires --compare")
    if recall_limit > 100:
        parser.error("--recall-limit must not exceed 100")
    if num_fewshot:
        parser.error("the supported memory benchmarks are zero-shot; --num_fewshot must be 0")
    requested = tuple(part.strip() for part in tasks_value.split(",") if part.strip())
    # `--seed` and `--limit` are argparse `type=` callables, which signal a bad value with
    # `ArgumentTypeError`; that is not a `ValueError`, so a configured value has to be caught here
    # explicitly or it escapes as an unhandled exception instead of a usage message.
    try:
        seeds = _resolved_seeds(parsed.seed, run.seed)
        limit = _resolved_limit(parsed.limit, run.limit)
        declared_run_id = None if run.run_id is None else _run_identifier(run.run_id)
        tasks = expand(requested)
        dataset_overrides = _assignments(_overridden_paths(parsed.task_data, run.task_data), tasks)
        media_overrides = _assignments(_overridden_paths(parsed.media_root, run.media_root), tasks)
        _parse_batch_size(batch_size)
        gen_kwargs = _generation_kwargs(parsed.gen_kwargs, seeds[0])
        performance_budgets = _resolved_performance_budgets(
            parsed.performance_budget,
            harness.performance_budgets,
        )
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    settings = (
        DownloadSettings.resolve(benchmarks_root=parsed.benchmarks_root, data_root=parsed.data_root)
        if download is None
        else download
    )
    run_id = (
        parsed.run_id or declared_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    # A generated run identifier names a directory no earlier run wrote to, so resuming one would
    # quietly ingest everything again. `--resume` stays command-line-only for the reason `--blind`
    # does: it describes one invocation's recovery, not the sweep a configuration file declares.
    if parsed.resume and not (parsed.run_id or declared_run_id):
        parser.error("--resume needs the --run-id of the run it continues")
    output = parsed.output_path or run.output_path or settings.benchmarks_root / "results" / run_id
    if performance_budgets and compare is None:
        parser.error("--performance-budget requires --compare")
    return _Arguments(
        tasks=tasks,
        benchmarks_root=settings.benchmarks_root.expanduser().resolve(),
        data_root=settings.data_root.expanduser().resolve(),
        output_path=output.expanduser().resolve(),
        run_id=run_id,
        dataset_overrides=dataset_overrides,
        media_overrides=media_overrides,
        media_manifest=(None if media_manifest is None else media_manifest.expanduser().resolve()),
        limit=limit,
        offset=offset,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        unit_concurrency=unit_concurrency,
        request_concurrency=request_concurrency,
        recall_limit=recall_limit,
        answer_policy=answer_policy,
        seed=seeds[0],
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        repeat_index=repeat_index,
        model=parsed.model,
        arms=arms,
        full_context_chars=full_context_chars,
        ingest=ingest,
        deliberate=parsed.deliberate,
        compile_max_items=compile_max_items,
        compile_max_chars=compile_max_chars,
        compile_allow_partial_sources=compile_allow_partial_sources,
        model_args=parsed.model_args,
        memory_config=(
            None if parsed.memory_config is None else parsed.memory_config.expanduser().resolve()
        ),
        judge_model_args=parsed.judge_model_args,
        judge_concurrency=judge_concurrency,
        gen_kwargs=gen_kwargs,
        num_fewshot=num_fewshot,
        use_cache=(None if use_cache is None else use_cache.expanduser().resolve()),
        device=device,
        device_lock=device_lock,
        compare=None if compare is None else compare.expanduser().resolve(),
        performance_budgets=performance_budgets,
        blind=parsed.blind,
        blind_baseline=(
            None if parsed.blind_baseline is None else parsed.blind_baseline.expanduser().resolve()
        ),
        fail_on_regression=fail_on_regression,
        regression_threshold=regression_threshold,
        predict_only=predict_only,
        log_samples=log_samples,
        stream_results=stream_results,
        allow_unverified_data=allow_unverified,
        download=download_inputs,
        overwrite=overwrite,
        resume=bool(parsed.resume),
        quiet=quiet or verbosity in {"ERROR", "CRITICAL"},
        fallback_reference_at=parsed.fallback_reference_at,
    )


# What a configuration file that names no provider gets. These mirror the backends the harness
# builds when `--config` is omitted entirely, so adding a file to set one unrelated knob does not
# silently change which models run. Naming a section overrides the default for that section only.
DEFAULT_CONFIG_SECTIONS: Mapping[str, Mapping[str, object]] = {
    "embedding": {"provider": "jina-omni"},
    "generation": {"provider": "openai"},
}


def _read_config_document(path: Path) -> Mapping[str, object]:
    """Parse one harness configuration file, reporting where invalid YAML went wrong."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read benchmark config {path}: {error}") from None
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = "" if mark is None else f" at line {mark.line + 1}, column {mark.column + 1}"
        raise ValueError(f"benchmark config {path} is invalid YAML{location}") from None
    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise ValueError(f"benchmark config {path} must be a mapping")
    return document


_Picked = TypeVar("_Picked")


@overload
def _picked(flag: _Picked | None, declared: _Picked | None, default: None) -> _Picked | None: ...


@overload
def _picked(flag: _Picked | None, declared: _Picked | None, default: _Picked) -> _Picked: ...


def _picked(
    flag: _Picked | None, declared: _Picked | None, default: _Picked | None
) -> _Picked | None:
    """Return the first supplied value: command line, then configuration file, then default."""
    if flag is not None:
        return flag
    if declared is not None:
        return declared
    return default


def _resolved_seeds(
    flag: tuple[int, int, int, int] | None, declared: int | Sequence[int] | None
) -> tuple[int, int, int, int]:
    """Expand a configured seed the way `--seed` expands one: 1 or 3 values fill out to 4."""
    if flag is not None:
        return flag
    if declared is None:
        return (0, 1234, 1234, 1234)
    values = (declared,) if isinstance(declared, int) else tuple(declared)
    return _seed_values(",".join(str(value) for value in values))


def _resolved_limit(
    flag: int | float | None, declared: int | float | str | None
) -> int | float | None:
    """Accept the configured limit in the spellings the flag accepts, `all` included."""
    if flag is not None:
        return flag
    if declared is None:
        return None
    return _limit_value(declared if isinstance(declared, str) else repr(declared))


def _overridden_paths(
    flag: Sequence[str] | None, declared: Mapping[str, Path] | None
) -> tuple[str, ...]:
    """Render both spellings of a per-task path override into the one `TASK=PATH` form."""
    if flag:
        return tuple(flag)
    if not declared:
        return ()
    return tuple(f"{task}={path}" for task, path in declared.items())


def _resolved_performance_budgets(
    flags: Sequence[str] | None,
    declared: Mapping[str, float],
) -> dict[str, float]:
    budgets = dict(declared)
    seen: set[str] = set()
    for item in flags or ():
        name, separator, raw = item.partition("=")
        name = name.strip()
        if not separator or name not in PERFORMANCE_BUDGET_NAMES or not raw.strip():
            raise ValueError(
                "performance budgets must use METRIC=FRACTION; metric must be one of: "
                + ", ".join(PERFORMANCE_BUDGET_NAMES)
            )
        if name in seen:
            raise ValueError(f"performance budget repeats metric: {name}")
        seen.add(name)
        try:
            budgets[name] = _nonnegative_float(raw.strip())
        except (ValueError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"invalid performance budget for {name}: {raw.strip()}") from error
    return budgets


def _load_memory_config(
    path: Path | None,
) -> tuple[MindBridgeConfig | None, HarnessOverrides]:
    """Load one harness configuration file into its product and harness halves.

    The file is YAML, which parses the JSON files this flag used to take without a second code
    path. The `benchmark:` mapping is the harness's own: credentials, judging, and corpus
    acquisition have no field in `MindBridgeConfig` and must not gain one, because that schema is
    a public contract and its credentials are deliberately kept off disk.
    """
    if path is None:
        return None, HarnessOverrides()
    values = dict(_read_config_document(path))
    section = values.pop("benchmark", None)
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"benchmark config {path} benchmark section must be a mapping")
    overrides = HarnessOverrides.model_validate(section)
    for name, default in DEFAULT_CONFIG_SECTIONS.items():
        if values.get(name) is None:
            values[name] = dict(default)
    config = MindBridgeConfig.model_validate(values)
    generation = config.generation
    if generation is None:
        raise ValueError("benchmark config generation cannot be null")
    conflicts = sorted(
        {"max_tokens", "seed", "temperature"}.intersection(generation.extra_body or {})
    )
    if conflicts:
        raise ValueError(
            "benchmark config generation.extra_body cannot set benchmark controls: "
            + ", ".join(conflicts)
        )
    # A reproducible sweep pins sampling: the harness always sends temperature 0 and the seed
    # `--seed` names. Declaring either here used to be accepted and then silently discarded, so
    # the file said one thing and the run did another; say so instead.
    pinned = sorted(name for name in ("temperature", "seed") if name in generation.model_fields_set)
    if pinned:
        raise ValueError(
            "benchmark config generation cannot set "
            + ", ".join(pinned)
            + ": reproducible evaluation pins temperature to 0 and takes the seed from --seed "
            "or benchmark.run.seed"
        )
    return config, overrides


def _model_config(
    model: str,
    arguments: str,
    *,
    memory_config: MindBridgeConfig | None = None,
    overrides: HarnessOverrides | None = None,
) -> ModelConfig:
    if model != "mindbridge":
        raise ValueError("model must be mindbridge")
    config = ModelConfig.from_environment()
    if memory_config is not None and memory_config.generation is not None:
        generation = memory_config.generation
        # `model` and `modalities` carry non-`None` schema defaults, so an absent value cannot be
        # told apart from an explicit one by truthiness the way `base_url`/`timeout` can. Checking
        # `model_fields_set` is what lets an env-only MINDBRIDGE_GENERATION_MODEL or
        # MINDBRIDGE_GENERATION_MODALITIES survive a file that does not mention the field at all.
        declared = generation.model_fields_set
        config = replace(
            config,
            generation_base_url=generation.base_url or config.generation_base_url,
            generation_api_key=(
                config.generation_api_key
                if generation.api_key is None
                else generation.api_key.get_secret_value()
            ),
            generation_model=(generation.model if "model" in declared else config.generation_model),
            generation_capabilities=(
                generation.modalities
                if "modalities" in declared
                else config.generation_capabilities
            ),
            timeout_seconds=generation.timeout or config.timeout_seconds,
            generation_min_video_seconds=(
                config.generation_min_video_seconds
                if generation.min_video_seconds is None
                else generation.min_video_seconds
            ),
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


def _judge_config(
    config: ModelConfig,
    arguments: _Arguments,
    *,
    overrides: HarnessOverrides | None = None,
) -> _JudgeConfig:
    # The configuration file wins over the environment; either wins over falling back to the
    # generation endpoint, which is what an unset judge has always meant.
    declared = HarnessOverrides().judge if overrides is None else overrides.judge
    judge = _JudgeConfig(
        model=(declared.model or os.getenv("MINDBRIDGE_JUDGE_MODEL") or config.generation_model),
        base_url=(
            declared.base_url
            or os.getenv("MINDBRIDGE_JUDGE_BASE_URL")
            or config.generation_base_url
        ),
        api_key=(
            declared.api_key or os.getenv("MINDBRIDGE_JUDGE_API_KEY") or config.generation_api_key
        ),
        timeout_seconds=(
            declared.timeout_seconds
            if declared.timeout_seconds is not None
            else float(os.getenv("MINDBRIDGE_JUDGE_TIMEOUT_SECONDS", str(config.timeout_seconds)))
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
    options = dict(item.split("=", 1) for item in arguments.gen_kwargs.split(",") if "=" in item)
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
            # `model` is the already-resolved credential (file, then MINDBRIDGE_GENERATION_API_KEY,
            # then the SDK's own lookup): without repeating it here, an env-only credential is
            # dropped the moment the file declares no `api_key`, since `model_copy(update=...)`
            # only touches keys named in this dict and this field would otherwise stay `None`.
            "api_key": (
                None if model.generation_api_key is None else SecretStr(model.generation_api_key)
            ),
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


def _description_cache_path(
    arguments: _Arguments,
    memory_config: MindBridgeConfig | None,
) -> Path | None:
    """Share descriptions inside one run without warming later performance repeats."""
    if memory_config is None or memory_config.vision is None:
        return None
    return arguments.data_root / "cache" / arguments.run_id / "descriptions.db"


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


def _require_output(path: Path, *, overwrite: bool, resume: bool = False) -> None:
    # The crash copy is guarded like the artifacts it stands in for. A run that died leaves one
    # behind and no `results.jsonl`, so without this a same-`--run-id` retry would pass the guard
    # and delete the only record of the tasks that did finish, before answering anything.
    # `--resume` is the exception, and only for the crash copy: it names the interrupted run it
    # continues, so the leftover is that run's own and refusing it would block the one command
    # written to recover from it. The finished artifacts still need `--overwrite`, because a run
    # that wrote them is not one to resume.
    guarded = (_RESULTS_FILE, _SAMPLES_FILE)
    for name in guarded if resume else (*guarded, _PARTIAL_SAMPLES_FILE, _CONFIG_FILE):
        target = path / name
        if target.exists() and not overwrite:
            raise FileExistsError(f"evaluation artifact already exists: {target}")


def _fallback_reference_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--fallback-reference-at must be a timezone-aware ISO 8601 datetime"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            "--fallback-reference-at must be a timezone-aware ISO 8601 datetime"
        )
    return parsed.astimezone(timezone.utc)


def _limit_value(value: str) -> int | float:
    # `all` is the word the help text advertises and the word a reader reaches for; accepting only
    # -1 made the documented spelling an error.
    if value.strip().casefold() in {"all", "-1"}:
        return -1
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "limit must be `all`, -1, a positive fraction, or a count"
        ) from error
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
