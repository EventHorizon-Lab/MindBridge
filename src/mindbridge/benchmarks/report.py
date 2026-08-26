"""Render one sweep's results as a table, out of the artifacts that sweep already wrote.

Nothing here measures anything, which is the whole reason it can print numbers at all. Every
figure in the table was written by a runner into its own manifest -- the four benchmarks whose
official protocol is exact-match, so their runner can score themselves -- or by
`mindbridge-bench score` into a sidecar after an external scorer ran. The `source` column says
which of the two a row came from, so a number read off this table can be traced to the artifact
that claims it. A task with neither is printed as `not scored`, which is the state most
benchmarks here are in until their official scorer has been run against the predictions.

The table is built from serialized JSON rather than from the summary object the sweep holds in
memory, so the sweep's own final table and `--report` on the same directory afterwards are the
same code reading the same bytes. They cannot drift into disagreeing about a run.

Models here read with `extra="ignore"` and only the fields the table needs. A summary written by
a newer version still renders, and a manifest's benchmark-specific counts cost nothing to skip.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mindbridge.benchmarks.artifacts import sidecar_manifest_path
from mindbridge.benchmarks.official_score import score_sidecar_path

METRIC_ROW_LIMIT = 12
"""Most metric rows one task contributes before the rest are summarised as a count.

Video-MME-v2 pins four taxonomies under each of two scoring protocols, which is upwards of forty
numbers for a single task -- a terminal summary that long is a file, not a summary. Rows are
ordered shallowest first, so what a cap drops is always a breakdown cell and never the headline
figure the breakdown decomposes.
"""

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


def render(summary_json: str | bytes, *, directory: Path) -> str:
    """Render the results table for one sweep summary and the artifacts beside it."""
    summary = _Summary.model_validate_json(summary_json)
    rows = [cell for task in summary.tasks for cell in _task_rows(task, directory=directory)]
    table = _table(
        ("Task", "Benchmark", "Status", "Wall", "Metric", "Value", "Source"),
        rows,
        right=(5,),
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
        return ((*identity, _ABSENT, _ABSENT, empty),)
    rows: list[tuple[str, ...]] = [(*identity, *_metric_cells(shown[0]))]
    rows += [("", "", "", "", *_metric_cells(metric)) for metric in shown[1:]]
    dropped = len(metrics) - len(shown)
    if dropped:
        rows.append(("", "", "", "", f"+{dropped} more, in the artifacts", _ABSENT, _ABSENT))
    return tuple(rows)


def _metric_cells(metric: _Metric) -> tuple[str, str, str]:
    """Render one metric's three cells, labelled by its own path inside the artifact.

    The dotted path rather than the leaf name alone, so `by_duration.long.accuracy` cannot be
    mistaken for the overall figure and can be found again in the manifest by searching for it.
    """
    return (metric.label, _value(metric.value), metric.source)


def _task_metrics(task: _Outcome, *, directory: Path) -> tuple[_Metric, ...]:
    """Collect every number recorded for one task, shallowest first within each source."""
    predictions = _resolve(task, directory=directory)
    if predictions is None:
        return ()
    runner = _sorted(_manifest_metrics(sidecar_manifest_path(predictions)))
    official = _sorted(_score_metrics(score_sidecar_path(predictions)))
    return (*runner, *official)


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
    """Read the metrics a runner scored itself, if this benchmark's runner scores itself."""
    document = _load(path)
    metrics = document.get("metrics") if document else None
    if isinstance(metrics, dict):
        yield from _flatten(metrics, source="runner")


def _score_metrics(path: Path) -> Iterator[_Metric]:
    """Read the metrics an official scorer produced, once `bench score` has attached them."""
    document = _load(path)
    metrics = document.get("metrics") if document else None
    if isinstance(metrics, dict):
        yield from _flatten(metrics, source="official")


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
            yield _Metric(label=prefix, value=float(value), source=source, depth=indent)
        return
    entries: Iterable[tuple[str, object]] = ()
    if isinstance(value, dict):
        entries = ((str(key), item) for key, item in value.items())
    elif isinstance(value, list):
        entries = ((_name(item, index), item) for index, item in enumerate(value))
    for key, item in entries:
        yield from _flatten(item, source=source, prefix=_join(prefix, key), depth=depth + 1)


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
    if not unscored:
        return (headline,)
    names = ", ".join(task.name for task in unscored)
    example = _resolve(unscored[0], directory=directory) or Path(unscored[0].output_path)
    return (
        headline,
        f"{_UNSCORED}: {names} — these are scored outside MindBridge; attach the result with",
        f"  mindbridge-bench score --predictions {example} --manifest "
        f"{sidecar_manifest_path(example)} ...",
    )


def _duration(seconds: float) -> str:
    """Render a wall clock the way a benchmark run is actually read: minutes, or hours."""
    whole = int(seconds)
    hours, remainder = divmod(whole, 3_600)
    minutes, second = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{second:02d}" if hours else f"{minutes:02d}:{second:02d}"


def _value(value: float) -> str:
    """Print a count as a count and a rate as a rate, without inventing precision for either."""
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
