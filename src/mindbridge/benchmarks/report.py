"""Render one sweep's results as a table, out of the artifacts that sweep already wrote.

Nothing here measures anything, which is the whole reason it can print numbers at all. Every
figure in the table was written into a run manifest by a runner or judge, or into a score sidecar
by `mindbridge-bench score` after an external scorer ran. The `source` column preserves that
provenance; a `bypass` value is identified as an unevaluated sentinel below the table. A task with
no recorded metric is printed as `not scored` until its external result is attached.

The table is built from serialized JSON rather than from the summary object the sweep holds in
memory, so the sweep's own final table and `--report` on the same directory afterwards are the
same code reading the same bytes. They cannot drift into disagreeing about a run.

Models here read with `extra="ignore"` and only the fields the table needs. A summary written by
a newer version still renders, and a manifest's benchmark-specific counts cost nothing to skip.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mindbridge.benchmarks.artifacts import sidecar_manifest_path
from mindbridge.benchmarks.official_score import score_sidecar_path
from mindbridge.benchmarks.scoring import BYPASS_VALUE

METRIC_ROW_LIMIT = 12
"""Most metric rows one task contributes before the rest are summarised as a count.

Video-MME-v2 pins four taxonomies under each of two scoring protocols, which is upwards of forty
numbers for a single task -- a terminal summary that long is a file, not a summary. Rows are
ordered shallowest first, so what a cap drops is always a breakdown cell and never the headline
figure the breakdown decomposes.
"""

HIGHER_IS_BETTER_SYMBOLS = {True: "↑", False: "↓"}
"""lmms-eval's own mapping, for the unnamed column between Metric and Value."""

_UNSCORED = "not scored"
_ABSENT = "—"
"""An em dash, for a cell with nothing to put in it rather than a zero."""


class _Outcome(BaseModel):
    """One task's row in a sweep summary, narrowed to what a table needs."""

    model_config = ConfigDict(extra="ignore")

    name: str
    benchmark: str
    status: str
    duration_seconds: float
    output_path: str


class _Summary(BaseModel):
    """One sweep's summary, narrowed to what a table needs."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    tasks: tuple[_Outcome, ...]


@dataclass(frozen=True, slots=True)
class _Metric:
    """One number, its dotted path inside the artifact, and which artifact claimed it."""

    label: str
    value: float
    source: str
    depth: int
    direction: str = ""
    """`↑`, `↓`, or blank -- lmms-eval's `HIGHER_IS_BETTER_SYMBOLS`, for the column beside Metric.

    Blank rather than a guess where the manifest declares nothing: `bypass` points nowhere, and a
    breakdown cell inherits no direction of its own from the headline it decomposes.
    """


def render(summary_json: str | bytes, *, directory: Path) -> str:
    """Render the results table for one sweep summary and the artifacts beside it."""
    summary = _Summary.model_validate_json(summary_json)
    rows = [cell for task in summary.tasks for cell in _task_rows(task, directory=directory)]
    table = _table(
        ("Task", "Benchmark", "Status", "Wall", "Metric", "", "Value", "Source"),
        rows,
        right=(6,),
    )
    return "\n".join((table, "", *_footer(summary, directory=directory)))


def render_directory(directory: Path) -> str:
    """Render the table for a results directory written by an earlier sweep."""
    summary_path = directory / "suite-summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no sweep summary here: {summary_path}")
    return render(summary_path.read_bytes(), directory=directory)


def _task_rows(task: _Outcome, *, directory: Path) -> tuple[tuple[str, ...], ...]:
    """Render one task as its identity row plus a continuation row per extra metric."""
    metrics = _task_metrics(task, directory=directory)
    shown = metrics[:METRIC_ROW_LIMIT]
    identity = (task.name, task.benchmark, task.status, _duration(task.duration_seconds))
    if not shown:
        empty = _UNSCORED if task.status == "completed" else _ABSENT
        return ((*identity, _ABSENT, "", _ABSENT, empty),)
    rows: list[tuple[str, ...]] = [(*identity, *_metric_cells(shown[0]))]
    rows += [("", "", "", "", *_metric_cells(metric)) for metric in shown[1:]]
    dropped = len(metrics) - len(shown)
    if dropped:
        rows.append(("", "", "", "", f"+{dropped} more, in the artifacts", "", _ABSENT, _ABSENT))
    return tuple(rows)


def _metric_cells(metric: _Metric) -> tuple[str, str, str, str]:
    """Render one metric's cells, labelled by its own path inside the artifact.

    The dotted path rather than the leaf name alone, so `by_duration.long.accuracy` cannot be
    mistaken for the overall figure and can be found again in the manifest by searching for it.
    """
    return (metric.label, metric.direction, _value(metric.value), metric.source)


def _task_metrics(task: _Outcome, *, directory: Path) -> tuple[_Metric, ...]:
    """Collect every number recorded for one task, shallowest first within each source."""
    predictions = _resolve(task, directory=directory)
    if predictions is None:
        return ()
    run = _sorted(_manifest_metrics(sidecar_manifest_path(predictions)))
    official = _sorted(_score_metrics(score_sidecar_path(predictions)))
    return _deduplicated((*run, *official))


def _deduplicated(metrics: Sequence[_Metric]) -> tuple[_Metric, ...]:
    """Keep one row per label, the first one, which is the declared headline.

    A self-scoring benchmark records its headline twice: once flat under `scoring.metrics`, where
    it carries the direction it points in, and once inside the typed breakdown its runner pins.
    Reading both is what makes the breakdown visible; printing `accuracy` twice would make the
    table look like the run disagreed with itself.
    """
    seen: set[str] = set()
    kept: list[_Metric] = []
    for metric in metrics:
        if metric.label not in seen:
            seen.add(metric.label)
            kept.append(metric)
    return tuple(kept)


def _sorted(metrics: Iterable[_Metric]) -> tuple[_Metric, ...]:
    """Order a source's metrics: shallowest first, and what a run scored before how much it ran.

    What survives `METRIC_ROW_LIMIT` is decided here, and so is what the eye lands on. A headline
    figure sits at depth zero and its taxonomy cells below it, so ordering by depth makes the cap
    drop only breakdowns; putting counts last within a depth keeps `accuracy` off the sixth row
    of its own task, behind the four denominators it was computed from.
    """
    return tuple(sorted(metrics, key=lambda metric: (metric.depth, _is_count(metric.label))))


def _is_count(label: str) -> bool:
    """Whether a metric says how much of a release ran rather than how well it did."""
    return label.rpartition(".")[2].endswith("_count")


def _resolve(task: _Outcome, *, directory: Path) -> Path | None:
    """Find this task's predictions, whether or not the results directory has been moved.

    The summary records the path the sweep wrote to, which is what a rerun in place needs. A
    directory copied off the machine that produced it -- how a scored run usually reaches the
    person reading this table -- has every one of those paths pointing at nothing, so the task's
    own subdirectory is tried as well before the row is called absent.
    """
    for candidate in (Path(task.output_path), directory / task.name / Path(task.output_path).name):
        if candidate.exists():
            return candidate
    return None


def _manifest_metrics(path: Path) -> Iterator[_Metric]:
    """Read what the run itself recorded, and who in the run recorded it.

    Two places, in this order. `scoring.metrics` is the declared `metric_list` -- the headline
    figure plus the direction it points in -- and `metrics` is the typed breakdown a self-scoring
    runner pins beside it. The declared one comes first so its direction survives deduplication.

    `scoring.mode` names the source, so a judge's number is never labelled as an exact match: the
    judge model was MindBridge's choice and a reader has to be able to see that from the table.
    It names the source of the *declared* block only. The runner's typed breakdown is always the
    runner's own arithmetic, so it is labelled `runner` whatever mode scored the headline --
    otherwise a judged benchmark's runner-computed breakdown reads as the judge's, and a
    `--predict-only` run printed a real measured accuracy in a row sourced `bypass` underneath a
    footer saying the run had not been evaluated.
    """
    document = _load(path)
    if not document:
        return
    scoring = document.get("scoring")
    scoring = scoring if isinstance(scoring, dict) else {}
    mode = str(scoring.get("mode") or "runner")
    declared = scoring.get("higher_is_better")
    directions = declared if isinstance(declared, dict) else {}
    for block, source in ((scoring.get("metrics"), mode), (document.get("metrics"), "runner")):
        if isinstance(block, dict):
            yield from _flatten(block, source=source, directions=directions)


def _score_metrics(path: Path) -> Iterator[_Metric]:
    """Read the metrics an official scorer produced, once `bench score` has attached them."""
    document = _load(path)
    metrics = document.get("metrics") if document else None
    if isinstance(metrics, dict):
        yield from _flatten(metrics, source="official", directions={})


def _load(path: Path) -> dict[str, object] | None:
    """Read one JSON artifact, treating an absent or unreadable one as simply having no numbers.

    A malformed artifact must not take the table down with it: the run it describes may be the
    one that failed, and the other tasks' numbers are exactly what the reader came for.
    """
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _flatten(
    value: object,
    *,
    source: str,
    directions: Mapping[str, object],
    prefix: str = "",
    depth: int = 0,
) -> Iterator[_Metric]:
    """Walk a metrics object, yielding one row per number under its dotted path.

    Generic rather than a reader per benchmark: the four self-scoring runners pin four different
    shapes -- a flat model, a list of per-category models, a mapping keyed by duration band, two
    scoring protocols each carrying four taxonomies -- and a walk handles a fifth shape added
    tomorrow without this module learning about it.

    Counts are kept only near the top. `question_count` says how much of a release a run covered
    and belongs in the table; the same field repeated inside every taxonomy cell is the denominator
    of a number already shown, and printing forty of them buries the figures worth reading.
    """
    if isinstance(value, bool):  # A bool is an int in Python, and no benchmark scores one.
        return
    if isinstance(value, (int, float)):
        indent = max(depth - 1, 0)
        if not (_is_count(prefix) and indent >= 2):
            yield _Metric(
                label=prefix,
                value=float(value),
                source=source,
                depth=indent,
                direction=_direction(prefix, directions),
            )
        return
    entries: Iterable[tuple[str, object]] = ()
    if isinstance(value, dict):
        entries = ((str(key), item) for key, item in value.items())
    elif isinstance(value, list):
        entries = ((_name(item, index), item) for index, item in enumerate(value))
    for key, item in entries:
        yield from _flatten(
            item,
            source=source,
            directions=directions,
            prefix=_join(prefix, key),
            depth=depth + 1,
        )


def _direction(label: str, directions: Mapping[str, object]) -> str:
    """Which way this metric points, when the run declared it, and blank when it did not.

    Counts are never given an arrow even where a declaration would supply one: `question_count`
    going up says the run covered more of the release, not that it did better.
    """
    declared = directions.get(label)
    if declared is None or _is_count(label):
        return ""
    return HIGHER_IS_BETTER_SYMBOLS.get(bool(declared), "")


def _name(item: object, index: int) -> str:
    """Name one entry of a list of scored groups, preferring the label it carries.

    EgoLifeQA's per-category metrics are a list whose entries name their own question type, so
    the row reads `categories.EventRecall.accuracy` rather than `categories.0.accuracy`. A list
    whose entries carry no label falls back to the position, which is still where the reader
    would go looking in the manifest.
    """
    if isinstance(item, dict):
        labels = [value for value in item.values() if isinstance(value, str) and value.strip()]
        if labels:
            return labels[0]
    return str(index)


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _footer(summary: _Summary, *, directory: Path) -> tuple[str, ...]:
    """Say what the sweep did overall, and name what would give an unscored task a number."""
    completed = sum(task.status == "completed" for task in summary.tasks)
    failed = sum(task.status == "failed" for task in summary.tasks)
    total = sum(task.duration_seconds for task in summary.tasks)
    headline = (
        f"{summary.run_id}: {len(summary.tasks)} tasks · {completed} completed · "
        f"{failed} failed · {_duration(total)} total"
    )
    unscored = tuple(
        task
        for task in summary.tasks
        if task.status == "completed" and not _task_metrics(task, directory=directory)
    )
    lines = [headline, *_caveats(summary, directory=directory)]
    if not unscored:
        return tuple(lines)
    names = ", ".join(task.name for task in unscored)
    example = _resolve(unscored[0], directory=directory) or Path(unscored[0].output_path)
    lines.append(
        f"{_UNSCORED}: {names} — these are scored outside MindBridge; attach the result with"
    )
    lines.append(
        f"  mindbridge-bench score --predictions {example} --manifest "
        f"{sidecar_manifest_path(example)} ..."
    )
    return tuple(lines)


def _caveats(summary: _Summary, *, directory: Path) -> tuple[str, ...]:
    """Say which numbers above are floors or sentinels rather than measurements.

    Both cases are invisible in the value itself, which is the cost of copying lmms-eval here. A
    judge that could not be read scores the answer 0.0, so a run with an unreachable judge reports
    a low score rather than an error; `bypass` reports 999, which is not a score at all. Neither is
    recoverable from the column, so it is said in words underneath.
    """
    lines: list[str] = []
    for task in summary.tasks:
        scoring = _scoring(task, directory=directory)
        failures = scoring.get("judge_failure_count")
        if isinstance(failures, int) and failures > 0:
            lines.append(
                f"{task.name}: {failures} answers scored 0.0 because the judge could not be "
                f"read — a floor, not a measurement (judge {scoring.get('judge_model')})"
            )
        if scoring.get("mode") == "bypass":
            lines.append(
                f"{task.name}: --predict-only, so {int(BYPASS_VALUE)} is lmms-eval's bypass "
                "sentinel and this run has not been evaluated"
            )
    return tuple(lines)


def _scoring(task: _Outcome, *, directory: Path) -> dict[str, object]:
    """Read one task's `scoring` block, or an empty one where the manifest has none."""
    predictions = _resolve(task, directory=directory)
    document = _load(sidecar_manifest_path(predictions)) if predictions else None
    scoring = document.get("scoring") if document else None
    return scoring if isinstance(scoring, dict) else {}


def _duration(seconds: float) -> str:
    """Render a wall clock the way a benchmark run is actually read: minutes, or hours."""
    whole = int(seconds)
    hours, remainder = divmod(whole, 3_600)
    minutes, second = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{second:02d}" if hours else f"{minutes:02d}:{second:02d}"


def _value(value: float) -> str:
    """Print a count as a count and a rate as a rate, without inventing precision for either.

    `int()` is guarded because `json.loads` accepts bare `NaN` and `Infinity`, which a
    hand-edited or third-party score sidecar carries for a 0/0 category. Unguarded, one such
    cell raised `ValueError`/`OverflowError` out of `render` and took every other task's numbers
    with it -- the failure `_load` is deliberately defensive about, one step further on.
    """
    if not math.isfinite(value):
        return str(value)
    return str(int(value)) if value == int(value) and abs(value) >= 1 else f"{value:.4f}"


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right: Sequence[int] = (),
) -> str:
    """Lay out fixed-width columns, without a dependency for what one comprehension does.

    No colour and no box drawing beyond a single rule: this goes to stdout, where it is as likely
    to be piped into a file or a diff as read in a terminal.
    """
    widths = [
        max(len(str(row[column])) for row in (headers, *rows)) for column in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        rendered = [
            str(cell).rjust(width) if column in right else str(cell).ljust(width)
            for column, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(rendered).rstrip()

    rule = "─" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join((line(headers), rule, *(line(row) for row in rows)))
