"""Provider settings used only by the benchmark harness."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mindbridge.benchmarks.eval_regression import PERFORMANCE_BUDGET_NAMES
from mindbridge.configuration import _absolute_http_url
from mindbridge.types import Modality

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GENERATION_MODEL = "gpt-5-mini"
# An hour bounds nothing a benchmark run cares about: a request the server never answers
# holds its task for the whole hour while the remaining workers idle, and the run reports
# the stall as elapsed time rather than as a failure. The slowest mean model call measured
# across this suite is video grounding at ~36s, so five minutes leaves ample headroom for a
# request that is genuinely slow while still cutting a hung one loose.
DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Reproducible OpenAI SDK inputs for benchmark generation."""

    generation_api_key: str | None = field(default=None, repr=False)
    generation_base_url: str = DEFAULT_OPENAI_BASE_URL
    generation_model: str = DEFAULT_GENERATION_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT})
    generation_min_video_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _absolute_http_url(self.generation_base_url):
            raise ValueError("generation_base_url must be an absolute http(s) URL")
        if not self.generation_model.strip():
            raise ValueError("generation_model must not be blank")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.generation_min_video_seconds is not None and (
            not math.isfinite(self.generation_min_video_seconds)
            or self.generation_min_video_seconds <= 0
        ):
            raise ValueError("generation_min_video_seconds must be positive")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ModelConfig:
        source = os.environ if environ is None else environ
        return cls(
            generation_api_key=(
                source.get("MINDBRIDGE_GENERATION_API_KEY") or source.get("OPENAI_API_KEY")
            ),
            generation_base_url=(
                source.get("MINDBRIDGE_GENERATION_BASE_URL")
                or source.get("OPENAI_BASE_URL")
                or DEFAULT_OPENAI_BASE_URL
            ),
            generation_model=source.get("MINDBRIDGE_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            timeout_seconds=_float(
                source.get("MINDBRIDGE_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS
            ),
            generation_capabilities=_modalities(source.get("MINDBRIDGE_GENERATION_MODALITIES")),
        )


def _float(value: str | None, default: float) -> float:
    try:
        return default if value is None else float(value)
    except ValueError:
        raise ValueError("MINDBRIDGE_TIMEOUT_SECONDS must be a number") from None


def _modalities(value: str | None) -> frozenset[Modality]:
    if value is None:
        return frozenset({Modality.TEXT})
    try:
        parsed = {Modality(item.strip().lower()) for item in value.split(",") if item.strip()}
    except ValueError:
        raise ValueError("MINDBRIDGE_GENERATION_MODALITIES is invalid") from None
    if Modality.OMNI in parsed:
        parsed.remove(Modality.OMNI)
        parsed.update({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
    return frozenset(parsed)


# Where the pinned corpora live when no `--benchmarks-root` is given. Resolved from the
# repository rather than the process directory: the corpus is hundreds of gigabytes and is
# shared, so a run started inside a Git worktree must reach the checkout's one copy instead of
# silently reporting every task as missing against a sibling directory that was never populated.
BENCHMARKS_DIRECTORY_NAME = ".benchmarks"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_YOUTUBE_SLEEP_SECONDS = 30.0


class _HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class JudgeOverrides(_HarnessModel):
    """Judge endpoint settings. Judging exists only in the harness, never in the product."""

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: Annotated[float, Field(strict=True, gt=0)] | None = None


class DownloadOverrides(_HarnessModel):
    """Corpus and Hugging Face acquisition addresses."""

    benchmarks_root: Path | None = None
    data_root: Path | None = None
    hf_home: Path | None = None
    hf_endpoint: str | None = None
    youtube_sleep_seconds: Annotated[float, Field(strict=True, ge=0)] | None = None


class ServerMetricsOverrides(_HarnessModel):
    """Optional process-global vLLM metric endpoints for the measured product phase."""

    generation_url: str | None = None
    embedding_url: str | None = None
    timeout_seconds: Annotated[float, Field(strict=True, gt=0)] = 5.0

    @field_validator("generation_url", "embedding_url")
    @classmethod
    def _absolute_metrics_url(cls, value: str | None) -> str | None:
        if value is not None and not _absolute_http_url(value):
            raise ValueError("server metrics URL must be an absolute HTTP or HTTPS URL")
        return value


class RunOverrides(_HarnessModel):
    """Run tunables the command line also exposes, so one file can describe a whole sweep.

    Every field is optional and mirrors the flag of the same name. A flag still wins, because a
    flag is typed for one run while a file is reused across many. Only three kinds of input stay
    command-line-only: the `--model` literal, the modes that do something other than evaluate
    (`--list-tasks`, `--check-integrity`), and the `--model-args`/`--gen-kwargs`/
    `--judge-model-args` strings, whose settings each already have a typed home elsewhere in this
    file.
    """

    tasks: str | None = None
    # Which arms answer, and the budget the full-context arm stuffs. A baseline sweep is exactly
    # the thing a file describes and a flag does not, so both belong here; `--blind` stays
    # command-line-only because it labels one run as the control rather than describing a sweep.
    arms: str | None = None
    full_context_chars: Annotated[int, Field(strict=True, gt=0)] | None = None
    # The compile arm's `ContextBudget` and the ingest mode `capture()`+`settle()` exercises
    # through the public SDK -- baseline-sweep knobs, the same reason `full_context_chars` lives
    # here.
    compile_max_items: Annotated[int, Field(strict=True, gt=0)] | None = None
    compile_max_chars: Annotated[int, Field(strict=True, gt=0)] | None = None
    ingest: Literal["add", "capture"] | None = None
    output_path: Path | None = None
    run_id: str | None = None
    task_data: Mapping[str, Path] | None = None
    media_root: Mapping[str, Path] | None = None
    media_manifest: Path | None = None
    # `all`, -1, a 0-1 fraction, or a count -- normalised by the same rule as the flag.
    limit: int | float | str | None = None
    offset: Annotated[int, Field(strict=True, ge=0)] | None = None
    num_fewshot: Annotated[int, Field(strict=True, ge=0)] | None = None
    batch_size: str | None = None
    max_batch_size: Annotated[int, Field(strict=True, gt=0)] | None = None
    unit_concurrency: Annotated[int, Field(strict=True, gt=0)] | None = None
    request_concurrency: Annotated[int, Field(strict=True, gt=0)] | None = None
    judge_concurrency: Annotated[int, Field(strict=True, gt=0)] | None = None
    recall_limit: Annotated[int, Field(strict=True, gt=0, le=100)] | None = None
    # One integer, or three, or four -- the shorter forms expand exactly as the flag expands them.
    seed: int | Sequence[int] | None = None
    bootstrap_samples: Annotated[int, Field(strict=True, gt=0)] | None = None
    repeat_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    device: str | None = None
    device_lock: bool | None = None
    use_cache: Path | None = None
    compare: Path | None = None
    fail_on_regression: bool | None = None
    regression_threshold: Annotated[float, Field(strict=True, ge=0)] | None = None
    predict_only: bool | None = None
    log_samples: bool | None = None
    # Reporting cadence, not a sweep knob: it only decides when each task's table is printed.
    stream_results: bool | None = None
    allow_unverified_data: bool | None = None
    download: bool | None = None
    overwrite: bool | None = None
    quiet: bool | None = None
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None


class HarnessOverrides(_HarnessModel):
    """The `benchmark:` section of a harness configuration file.

    Every field is optional and every field wins over the matching environment variable. The
    section is absent from `MindBridgeConfig` on purpose: judging and corpus acquisition are
    harness concerns, and putting them in the product schema would make them part of the public
    contract. Nothing that describes a model endpoint appears here -- not even the short-video
    floor, which is `generation.min_video_seconds` in the product block -- because a setting with
    two homes is a setting whose effective value nobody can read off the file.
    """

    judge: JudgeOverrides = Field(default_factory=JudgeOverrides)
    download: DownloadOverrides = Field(default_factory=DownloadOverrides)
    server_metrics: ServerMetricsOverrides = Field(default_factory=ServerMetricsOverrides)
    performance_budgets: Mapping[str, Annotated[float, Field(strict=True, ge=0)]] = Field(
        default_factory=dict
    )
    run: RunOverrides = Field(default_factory=RunOverrides)

    @field_validator("performance_budgets")
    @classmethod
    def _known_performance_budgets(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        unknown = sorted(set(value) - set(PERFORMANCE_BUDGET_NAMES))
        if unknown:
            raise ValueError(f"unknown performance budget(s): {', '.join(unknown)}")
        return value


def default_benchmarks_root(start: Path | None = None) -> Path:
    """Return the checkout's benchmark corpus directory, or a relative path if there is none.

    A linked worktree stores `.git` as a file naming the common directory inside the main
    checkout; its parent is the directory holding the single populated corpus. Falling back to a
    relative path keeps the function usable outside a repository at all.
    """
    current = (Path.cwd() if start is None else start).resolve()
    for directory in (current, *current.parents):
        marker = directory / ".git"
        if marker.is_dir():
            return directory / BENCHMARKS_DIRECTORY_NAME
        if marker.is_file():
            return _main_worktree(marker, directory) / BENCHMARKS_DIRECTORY_NAME
    return Path(BENCHMARKS_DIRECTORY_NAME)


def _main_worktree(marker: Path, fallback: Path) -> Path:
    try:
        pointer = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    prefix = "gitdir:"
    if not pointer.startswith(prefix):
        return fallback
    git_dir = Path(pointer[len(prefix) :].strip())
    if not git_dir.is_absolute():
        git_dir = (fallback / git_dir).resolve()
    # .../<main>/.git/worktrees/<name> -> <main>
    for ancestor in git_dir.parents:
        if ancestor.name == ".git":
            return ancestor.parent
    return fallback


@dataclass(frozen=True, slots=True)
class DownloadSettings:
    """Resolved corpus and Hugging Face acquisition addresses."""

    benchmarks_root: Path
    data_root: Path
    hf_home: Path | None
    hf_endpoint: str
    youtube_sleep_seconds: float

    @classmethod
    def resolve(
        cls,
        overrides: DownloadOverrides | None = None,
        *,
        benchmarks_root: Path | None = None,
        data_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> DownloadSettings:
        """Resolve acquisition addresses with the configuration file taking precedence.

        Precedence is command line, then configuration file, then environment, then default. The
        command line stays above the file because a flag is typed for one run while a file is
        reused across many.
        """
        source = os.environ if environ is None else environ
        declared = DownloadOverrides() if overrides is None else overrides
        root = benchmarks_root or declared.benchmarks_root or default_benchmarks_root()
        home = declared.hf_home or _optional_path(source.get("HF_HOME"))
        return cls(
            benchmarks_root=root,
            data_root=data_root or declared.data_root or root / "data",
            hf_home=home,
            hf_endpoint=(declared.hf_endpoint or source.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT),
            youtube_sleep_seconds=(
                declared.youtube_sleep_seconds
                if declared.youtube_sleep_seconds is not None
                else _float(
                    source.get("MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS"),
                    DEFAULT_YOUTUBE_SLEEP_SECONDS,
                )
            ),
        )

    def apply_environment(self, environ: dict[str, str] | None = None) -> None:
        """Publish the resolved addresses as process environment.

        `huggingface_hub` reads `HF_HOME` and `HF_ENDPOINT` when it is first imported, and that
        import is lazy inside the download path, so writing them here is what actually redirects
        acquisition. Publishing rather than threading a parameter also keeps one resolved answer
        for every reader, including the media preparer's throttle.
        """
        target = os.environ if environ is None else environ
        if self.hf_home is not None:
            target["HF_HOME"] = str(self.hf_home)
        target["HF_ENDPOINT"] = self.hf_endpoint
        target["MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS"] = repr(self.youtube_sleep_seconds)


def _optional_path(value: str | None) -> Path | None:
    text = "" if value is None else value.strip()
    return Path(text).expanduser() if text else None
