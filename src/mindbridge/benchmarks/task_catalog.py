"""The tasks `mindbridge-bench eval --tasks` can name, and where each release is expected.

Every entry here is a benchmark plus the choices its runner requires and cannot default:
ATM-Bench's `--split`, MEMLENS's `--context-window`, M3-Bench's `--subset`, Video-MME's
`--transcript-source`. That is the rule the table follows — one task per required choice — which
is why MEMLENS appears four times and LoCoMo-Refined once. Optional filters are not enumerated:
`--limit` covers a smoke run, and a run scoped to Video-MME's `long` band or three Mem-Gallery
topics is a task of your own, spelled in a `--suite` file or handed straight to the runner.

Paths are written against `--benchmarks-root`, which defaults to the `.benchmarks/` layout
`docs/benchmarking.md` tells you to download into. Prepared-media manifests are the exception
worth knowing: MindBridge does not produce them, so what the table holds is the file name that
document uses. If yours is named something else, `--list-tasks` shows the task as missing and
prints the path it wanted. The three MM-Lifelong manifests extend the one documented name
(`mm-lifelong-month-val-prepared.json`) to its other splits.

A task in this table is not a claim that the run is citable. It fixes the release and the
protocol; whether the split was complete is recorded per run by the manifest the runner writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mindbridge.benchmarks.prepare import PREPARERS
from mindbridge.benchmarks.releases import missing_inputs, release_for

DEFAULT_BENCHMARKS_ROOT = Path(".benchmarks")
"""Where `docs/benchmarking.md` downloads every official release to."""

ROOT = "{root}"
"""What stands in for `--benchmarks-root` inside an argument, and marks it as a path.

Substitution is a plain string replace rather than `str.format`, so an argument that happens to
contain a brace is not a format string this has to escape.
"""


@dataclass(frozen=True, slots=True)
class CatalogTask:
    """One named task: which runner, the arguments it needs, and what it writes."""

    benchmark: str
    arguments: tuple[str, ...]
    output_suffix: str = ".jsonl"
    """`.json` for the runners emitting one array rather than one object per line."""

    def resolved(self, *, root: Path) -> tuple[str, ...]:
        """Substitute the benchmarks root into every path this task names."""
        return tuple(argument.replace(ROOT, str(root)) for argument in self.arguments)

    def inputs(self, *, root: Path) -> tuple[Path, ...]:
        """The files this task reads, which is exactly the arguments carrying a root."""
        return tuple(
            Path(argument.replace(ROOT, str(root)))
            for argument in self.arguments
            if ROOT in argument
        )


def _memlens(window: str) -> CatalogTask:
    return CatalogTask(
        "memlens",
        (
            "--dataset",
            f"{ROOT}/memlens/dataset_{window}.json",
            "--agent-subset-index",
            f"{ROOT}/memlens/agent_subset_195.json",
            "--context-window",
            window,
            "--text-only",
        ),
        output_suffix=".json",
    )


def _atm(split: str, *, media_source: str) -> CatalogTask:
    archive = (
        ("--media-source", "sgm")
        if media_source == "sgm"
        else ("--media-source", "raw", "--prepared-media", f"{ROOT}/atm-prepared-media.json")
    )
    release = "atm-bench.json" if split == "main" else f"atm-bench-{split}.json"
    return CatalogTask(
        "atm",
        (
            "--dataset",
            f"{ROOT}/atm-bench/data/atm-bench/{release}",
            "--split",
            split,
            *archive,
            "--emails",
            f"{ROOT}/atm-bench/data/raw_memory/email/emails.json",
            "--sgm-image",
            f"{ROOT}/atm-bench/data/processed_memory/image_batch_results.json",
            "--sgm-video",
            f"{ROOT}/atm-bench/data/processed_memory/video_batch_results.json",
        ),
        output_suffix=".json",
    )


def _m3(subset: str) -> CatalogTask:
    return CatalogTask(
        "m3",
        (
            "--dataset",
            f"{ROOT}/m3-agent/data/annotations/{subset}.json",
            "--subset",
            subset,
        ),
    )


def _mm_lifelong(split: str, release: str) -> CatalogTask:
    return CatalogTask(
        "mm-lifelong",
        (
            "--dataset",
            f"{ROOT}/mm-lifelong/{release}.json",
            "--prepared-media",
            f"{ROOT}/mm-lifelong-{split.replace('_', '-')}-prepared.json",
            "--split",
            split,
        ),
    )


TASKS: dict[str, CatalogTask] = {
    "locomo-refined": CatalogTask(
        "locomo-refined",
        ("--dataset", f"{ROOT}/locomo-refined/data/raw/locomo_refined.json"),
    ),
    "m3-robot": _m3("robot"),
    "m3-web": _m3("web"),
    "egolife": CatalogTask(
        "egolife",
        (
            "--dataset",
            f"{ROOT}/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json",
            "--prepared-media",
            f"{ROOT}/egolife-prepared-a1.json",
        ),
        output_suffix=".json",
    ),
    "egomem": CatalogTask(
        "egomem",
        (
            "--dataset",
            f"{ROOT}/egomem-reason/annotations_public.jsonl",
            "--prepared-media",
            f"{ROOT}/egomem-prepared.json",
        ),
        output_suffix=".json",
    ),
    "egotempo": CatalogTask(
        "egotempo",
        (
            "--dataset",
            f"{ROOT}/egotempo/egotempo_openQA.json",
            "--prepared-media",
            f"{ROOT}/egotempo-prepared.json",
        ),
        output_suffix=".json",
    ),
    "memlens-32k": _memlens("32k"),
    "memlens-64k": _memlens("64k"),
    "memlens-128k": _memlens("128k"),
    "memlens-256k": _memlens("256k"),
    "mm-lifelong-day-test": _mm_lifelong("day_test", "day/test"),
    "mm-lifelong-week-test": _mm_lifelong("week_test", "week/test"),
    "mm-lifelong-month-train": _mm_lifelong("month_train", "month/train"),
    "mm-lifelong-month-val": _mm_lifelong("month_val", "month/val"),
    "atm-main": _atm("main", media_source="raw"),
    "atm-hard": _atm("hard", media_source="raw"),
    "atm-main-sgm": _atm("main", media_source="sgm"),
    "atm-hard-sgm": _atm("hard", media_source="sgm"),
    "mem-gallery": CatalogTask(
        "mem-gallery",
        (
            "--dataset",
            f"{ROOT}/mem-gallery/data/dialog",
        ),
        output_suffix=".json",
    ),
    "supermemory-subject-1": CatalogTask(
        "supermemory",
        (
            "--dataset",
            f"{ROOT}/supermemory-vqa/data/json/all_qa.json",
            "--prepared-media",
            f"{ROOT}/supermemory-prepared-person-1.json",
            "--subject",
            "1",
        ),
        output_suffix=".json",
    ),
    "video-mme": CatalogTask(
        "video-mme",
        (
            "--dataset",
            f"{ROOT}/video-mme/videomme/test-00000-of-00001.parquet",
            "--prepared-media",
            f"{ROOT}/video-mme-prepared.json",
            "--transcript-source",
            "none",
        ),
        output_suffix=".json",
    ),
    "video-mme-v2": CatalogTask(
        "video-mme-v2",
        (
            "--dataset",
            f"{ROOT}/video-mme-v2/test.parquet",
            "--prepared-media",
            f"{ROOT}/video-mme-v2-prepared.json",
            "--transcript-source",
            "none",
        ),
        output_suffix=".json",
    ),
}

GROUPS: dict[str, tuple[str, ...]] = {
    "released-text": (
        "locomo-refined",
        "memlens-32k",
        "atm-main-sgm",
        "atm-hard-sgm",
    ),
    "all": tuple(TASKS),
}
"""Names that expand to several tasks.

`released-text` is the set that replays official text alone: no prepared media, so it runs
against a deployment with no Worker plugins and its numbers are a memory-layer claim rather than
a multimodal one. `all` exists for `--tasks all --limit 1`, which is a smoke run of the harness
and not an evaluation.
"""


def expand(names: Sequence[str]) -> tuple[str, ...]:
    """Resolve task and group names to task names, in the order the catalog lists them.

    Catalog order rather than the order asked for: a sweep of the same names has to run the same
    way twice, and two groups naming one task must not run it twice.
    """
    unknown = tuple(name for name in names if name not in TASKS and name not in GROUPS)
    if unknown:
        raise ValueError(
            f"unknown task: {', '.join(unknown)}; `--list-tasks` prints every name this accepts"
        )
    wanted = {task for name in names for task in GROUPS.get(name, (name,))}
    return tuple(name for name in TASKS if name in wanted)


def task_payloads(names: Sequence[str], *, root: Path) -> tuple[dict[str, object], ...]:
    """Build the suite tasks the named catalog entries stand for."""
    return tuple(
        {
            "name": name,
            "benchmark": TASKS[name].benchmark,
            "arguments": TASKS[name].resolved(root=root),
            "output_name": f"predictions{TASKS[name].output_suffix}",
        }
        for name in expand(names)
    )


def task_inputs(names: Sequence[str], *, root: Path) -> dict[str, tuple[Path, ...]]:
    """The files each named task reads, keyed by task name."""
    return {name: TASKS[name].inputs(root=root) for name in expand(names)}


def listing(*, root: Path) -> str:
    """Render every name `--tasks` accepts, and what obtaining it would still take.

    Four states, because they call for four different things: `ready` runs now, `download` runs
    after a fetch the sweep performs itself, `prepare` after it also stages the media, and a
    named path is a manifest with no producer yet, which has to be made out-of-band.
    """
    lines = [f"groups (--tasks expands these), inputs resolved against {root}:"]
    lines += [f"  {name:<24}{', '.join(members)}" for name, members in GROUPS.items()]
    lines += ["", "tasks:"]
    for name, task in TASKS.items():
        lines.append(f"  {name:<24}{task.benchmark:<16}{_state(task, root=root)}")
    return "\n".join(lines)


def _state(task: CatalogTask, *, root: Path) -> str:
    """Say what stands between this task and a run."""
    absent = missing_inputs(task.inputs(root=root), root=root)
    if task.benchmark in PREPARERS:
        # Its manifest is written per run into the sweep's own output directory, so there is no
        # file here to find or to have gone stale: what it needs is the staging, every time.
        return "prepare" if not absent else "download, prepare"
    if not absent:
        return "ready"
    unobtainable = tuple(path for path in absent if release_for(path, root=root) is None)
    if unobtainable:
        return f"needs {', '.join(str(path) for path in unobtainable)}"
    return "download"
