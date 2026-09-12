"""Console output for eval runs: logging setup, progress bars, announcements, and the summary table."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import Event, Thread
from typing import TYPE_CHECKING, Protocol, cast

from tqdm import tqdm

if TYPE_CHECKING:
    pass
from mindbridge.benchmarks.eval_config import DEFAULT_ARM

# The log namespace this harness is allowed to be verbose about: everything under the
# installed package, derived so a rename cannot leave a stale literal behind.
_ROOT_PACKAGE = __name__.split(".", 1)[0]


class _TqdmHandler(logging.Handler):
    """Emit every record through `tqdm.write`, so a live progress bar survives a log line.

    The alternative, wrapping the run in `tqdm.contrib.logging.logging_redirect_tqdm`, swaps this
    handler out for one of tqdm's own for the duration of the bar, and carrying the filters across
    is a detail tqdm only started honouring in 4.69.1. Owning the handler keeps `_VerbosityFilter`
    attached to the thing that actually emits, at every version, bar or no bar. `sys.stderr` is
    read per record rather than captured, so redirecting it after configuration still works.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # logging must never raise into the run it is reporting on
            self.handleError(record)


class _VerbosityFilter(logging.Filter):
    """Admit MindBridge's records at the requested level and everyone else's at ``third_party``.

    A root level alone does not hold: `modelscope`, `numba`, `jieba` and `torch.__trace` each set
    their own logger level when imported, and a logger with an explicit level never consults the
    root's, so their INFO and DEBUG records reach the root handler whatever it was configured
    with. Deciding per record on the handler needs no list of names to keep up to date, and no
    dependency that reaches the root handler can get around it.

    A dependency that installs its own handler and sets `propagate = False` never reaches this
    filter at all; `_dependency_log_levels` is where those are turned down by name.
    """

    def __init__(self, level: int, third_party: int) -> None:
        super().__init__()
        self._level = level
        self._third_party = third_party

    def filter(self, record: logging.LogRecord) -> bool:
        own = record.name.split(".", 1)[0] == _ROOT_PACKAGE
        return record.levelno >= (self._level if own else self._third_party)


def _dependency_log_levels(third_party: int) -> None:
    """Turn down the dependencies that log through a handler of their own, not the root's.

    `modelscope.utils.logger` attaches its own stderr handler to the `modelscope` logger and sets
    `propagate = False`, so `_VerbosityFilter` never sees those records: every FunASR run --
    `import funasr` imports modelscope -- printed its INFO whatever `--verbosity` said, through a
    plain `StreamHandler` that also smears the live progress bar. It reads its level from the
    environment once, at import, so this has to be set before anything imports it, and an
    explicit setting from the caller wins.

    ponytail: one name, because one installed dependency does this today. The next one gets a
    line here; there is no generic hook short of patching `logging.Logger.addHandler`.
    """
    os.environ.setdefault("MODELSCOPE_LOG_LEVEL", str(third_party))


def _configure_logging(verbosity: str) -> None:
    """Claim the root handler before an imported dependency installs a noisier one.

    `import funasr` runs `logging.basicConfig(level=INFO)` at module scope, which switches every
    library in the process to INFO: each successful model call then prints its own
    `HTTP Request: ... 200 OK` line and buries the warnings worth reading. Configuring first makes
    that call a no-op. `--verbosity` then applies to MindBridge only; a dependency has to reach
    WARNING to be heard, because a benchmark run is not the place to read anyone else's INFO.
    `DEBUG` is the exception that opens the whole process, transport chatter included.
    """
    level: int = logging.getLevelName(verbosity)
    third_party = level if verbosity == "DEBUG" else max(level, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[_TqdmHandler()],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_VerbosityFilter(level, third_party))
    _dependency_log_levels(third_party)


def _announce(message: str) -> None:
    # `tqdm.write` is how a bar and a message share one stream: it erases the bar, writes the
    # line, and redraws. With no bar live it is a `print` with a flush, so every caller is
    # bar-safe without knowing whether one is running.
    tqdm.write(f"mindbridge-bench eval: {message}", file=sys.stderr)


def _ignore_progress(_completed: int, _total: int) -> None:
    pass


class _ProgressReporter(Protocol):
    def __call__(self, completed: int, total: int) -> None: ...

    def set_activity(self, activity: str) -> None: ...


@dataclass(slots=True)
class _CallbackProgressReporter:
    callback: Callable[[int, int], None]

    def __call__(self, completed: int, total: int) -> None:
        self.callback(completed, total)

    def set_activity(self, activity: str) -> None:
        del activity


# How long a non-interactive run may stay silent between progress lines.
_PROGRESS_LOG_SECONDS = 60.0


# A sample can spend minutes rebuilding and ingesting before its answer exists. Redraw the live
# meter while its count is unchanged so elapsed time and the current phase still prove liveness.
_PROGRESS_REFRESH_SECONDS = 1.0


# tqdm's own meter with the bar glyphs removed. A redrawn bar in a log file is one unreadable
# line of carriage returns, but the counts and the ETA are the ones a terminal would have shown,
# so both modes report identical numbers.
_PROGRESS_LOG_FORMAT = (
    "{desc}: {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]"
)


@contextmanager
def _progress(
    stage: str, noun: str, *, total: int, enabled: bool = True
) -> Iterator[_ProgressReporter]:
    """Report progress as a live bar on a terminal and as throttled lines anywhere else."""
    if not enabled or total <= 0:
        yield _CallbackProgressReporter(_ignore_progress)
        return
    if sys.stderr.isatty():
        with tqdm(total=total, desc=stage, unit=noun, file=sys.stderr, leave=False) as bar:
            stopped = Event()

            def refresh() -> None:
                while not stopped.wait(_PROGRESS_REFRESH_SECONDS):
                    bar.refresh()

            @dataclass(slots=True)
            class LiveProgressReporter:
                def __call__(self, completed: int, total: int) -> None:
                    del total
                    bar.update(completed - bar.n)

                def set_activity(self, activity: str) -> None:
                    bar.set_postfix_str(activity, refresh=True)

            refresher = Thread(target=refresh, name="mindbridge-bench-progress", daemon=True)
            refresher.start()
            try:
                yield LiveProgressReporter()
            finally:
                stopped.set()
                refresher.join()
        return
    started = time.monotonic()
    last = 0.0

    def report(completed: int, _total: int) -> None:
        # Throttle on elapsed time, not on a fraction of the work: a tenth of a forty-minute task
        # and a tenth of a twenty-second one are not the same amount of silence. The first and
        # the last completion always report. Preparation may also report zero before starting, so
        # a run that stalls on its first source says so at once and the log ends on the final count.
        nonlocal last
        now = time.monotonic()
        if completed not in {0, 1, total} and now - last < _PROGRESS_LOG_SECONDS:
            return
        last = now
        _announce(
            tqdm.format_meter(
                completed,
                total,
                now - started,
                prefix=stage,
                unit=noun,
                bar_format=_PROGRESS_LOG_FORMAT,
            )
        )

    yield _CallbackProgressReporter(report)


@contextmanager
def _deferred_progress(
    stage: str, noun: str, *, enabled: bool = True
) -> Iterator[Callable[[int, int], None]]:
    """Start a progress reporter once a producer discovers its total work count."""
    with ExitStack() as stack:
        expected_total: int | None = None
        report: Callable[[int, int], None] = _ignore_progress

        def advance(completed: int, total: int) -> None:
            nonlocal expected_total, report
            if expected_total is None:
                expected_total = total
                report = stack.enter_context(_progress(stage, noun, total=total, enabled=enabled))
            elif total != expected_total:
                raise ValueError(f"{stage} progress total changed from {expected_total} to {total}")
            report(completed, total)

        yield advance


def _table(results: Mapping[str, object]) -> str:
    tasks = cast(Sequence[Mapping[str, object]], results["tasks"])
    rows = []
    for task in tasks:
        score = cast(Mapping[str, object], task["score"])
        performance = cast(Mapping[str, object], task["performance"])
        duration = cast(Mapping[str, object], performance["duration_seconds"])
        usage = cast(Mapping[str, object], performance["token_usage"])
        controls = cast(Mapping[str, object], task["controls"])
        mean = score.get("mean")
        interval = score.get("confidence_interval_95")
        total_duration = duration.get("total")
        average_duration = duration.get("average")
        total_tokens = usage.get("total_tokens")
        average_tokens = usage.get("average_tokens")
        # When the judge's usage is incomplete the task total is honestly null, but the product's
        # own spend is usually complete; print that with a marker rather than a dash.
        token_marker = ""
        product = usage.get("product")
        if (
            total_tokens is None
            and isinstance(product, Mapping)
            and product.get("total_tokens") is not None
        ):
            total_tokens = product.get("total_tokens")
            average_tokens = product.get("average_tokens")
            token_marker = "*"
        valid = task.get("score_valid") is not False
        rows.append(
            (
                (
                    str(task["task"])
                    if task.get("arm", DEFAULT_ARM) == DEFAULT_ARM
                    else f"{task['task']} [{task['arm']}]"
                ),
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
                str(task["ingest_failure_count"]),
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
                    else f"{total_tokens}{token_marker}"
                ),
                (
                    "—"
                    if isinstance(average_tokens, bool)
                    or not isinstance(average_tokens, int | float)
                    else f"{float(average_tokens):.1f}{token_marker}"
                ),
                _control_cell(controls.get("recall_at_1")),
                _control_cell(controls.get("recall_at_20")),
                _control_cell(
                    cast(Mapping[str, object], controls["random_ranker"]).get("20")
                    if isinstance(controls.get("random_ranker"), Mapping)
                    else None
                ),
                ("SELF" if controls.get("is_blind_run") else _control_cell(controls.get("blind"))),
                (
                    "ok"
                    if controls.get("interpretable")
                    else "MISSING " + ",".join(cast(Sequence[str], controls.get("missing", ())))
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
        # A score reads INVALID whenever writes failed, and until this column existed the table
        # said only "errors 0" beside it: answering had in fact succeeded, on an empty store. One
        # run lost 7 784 writes to a missing transcription dependency and showed nothing.
        "unwritten",
        "total s",
        "avg ms",
        "tokens",
        "tokens/q",
        "R@1",
        "R@20",
        "rand@20",
        "blind",
        "controls",
    )
    widths = tuple(
        max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))
    )
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (headers, *rows)
    )


def _control_cell(value: object) -> str:
    """Render one control value, and never let an absent control render as a number."""
    if not isinstance(value, Mapping):
        return "MISSING"
    mean = value.get("mean")
    if isinstance(mean, bool) or not isinstance(mean, int | float):
        return "MISSING"
    return f"{float(mean):.4f}"


def _uninterpretable_tasks(results: Mapping[str, object]) -> tuple[str, ...]:
    """Return the tasks whose mandatory controls are absent."""
    tasks = cast(Sequence[Mapping[str, object]], results["tasks"])
    return tuple(
        str(cast(Mapping[str, object], task["controls"])["reason"])
        for task in tasks
        if not cast(Mapping[str, object], task["controls"])["interpretable"]
    )
