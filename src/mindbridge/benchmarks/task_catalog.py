"""Pinned benchmark tasks accepted by ``mindbridge-bench eval --tasks``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from mindbridge.benchmarks.atm_bench import ATM_BENCH_ADAPTER_VERSION

DEFAULT_BENCHMARKS_ROOT = Path(".benchmarks")


@dataclass(frozen=True, slots=True)
class MediaSource:
    """Pinned source media, or the external tool that supplies it."""

    release: str
    repository: str | None = None
    revision: str | None = None
    patterns: tuple[str, ...] = ()
    acquirer: str | None = None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One reproducible dataset/protocol choice."""

    name: str
    benchmark: str
    dataset: str
    adapter_version: str
    repository: str
    revision: str
    digest: str | None = None
    variant: str | None = None
    auxiliary: tuple[str, ...] = ()
    media: str | None = None
    media_source: MediaSource | None = None

    def dataset_path(self, root: Path) -> Path:
        return root / self.dataset

    def input_paths(self, root: Path) -> tuple[Path, ...]:
        return (self.dataset_path(root), *(root / path for path in self.auxiliary))

    def media_root(self, root: Path) -> Path | None:
        return None if self.media is None else root / self.media


_LOCOMO = ("mem-eval-suite/LoCoMo_refined", "887091190789e8d6760e70b9edd696539923dc4f")
_M3 = ("ByteDance-Seed/m3-agent", "0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c")
_VIDEO_MME = ("lmms-eval/Video-MME", "ead1408f75b618502df9a1d8e0950166bf0a2a0b")
_VIDEO_MME_V2 = (
    "MME-Benchmarks/Video-MME-v2",
    "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
)
_EGOLIFE = ("lmms-lab/EgoLife", "143fb319be7aa5ae210c936bf4f0f3a86092afb0")
_EGOMEM = ("Ted412/EgoMemReason", "7e581505b9dce0e85193a27ae689ff899d0bc507")
_EGOTEMPO = (
    "google-research-datasets/egotempo",
    "7022ba77b4d89f51cf34e499767995ccd5c90c7a",
)
_MEMLENS = ("xiyuRenBill/MEMLENS", "afa101a1907cc37db40b50d649547964387b96b7")
_LIFELONG = ("MM-Lifelong/MM-Lifelong", "248aa82039a574e63a2e524746a7cd8f32330443")
_SUPERMEMORY = (
    "OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
    "1d228e0f10049a8a84c458dded2aa25b1e21ce8f",
)
_ATM = ("Jingbiao/ATM-Bench", "78e826dc07e97466b2f54443831ef9a83ab8b27c")
_GALLERY = ("Ethan-Bei/Mem-Gallery", "af912daba984e896e253016b7c7e334ef92c2a6f")

_M3_ROBOT_MEDIA = MediaSource(
    "m3-bench",
    "ByteDance-Seed/M3-Bench",
    "2672152eee36b25ccb38fdbc3b72135347adbb63",
    ("videos/robot/*",),
)
_M3_WEB_MEDIA = MediaSource("m3-bench", acquirer="youtube")
_VIDEO_MME_MEDIA = MediaSource("video-mme", *_VIDEO_MME, ("videos_chunked_*.zip",))
_VIDEO_MME_V2_MEDIA = MediaSource("video-mme-v2", *_VIDEO_MME_V2, ("videos/*.zip",))
_EGOLIFE_MEDIA = MediaSource("egolife", *_EGOLIFE, ("A?_*/DAY*/*.mp4",))
_EGOTEMPO_MEDIA = MediaSource("egotempo", acquirer="ego4d")
_LIFELONG_MEDIA = MediaSource("mm-lifelong", *_LIFELONG, ("videos/*",))
_SUPERMEMORY_MEDIA = MediaSource("supermemory-vqa", *_SUPERMEMORY, ("data/video/*",))
_ATM_MEDIA = MediaSource(
    "atm-bench",
    *_ATM,
    ("data/raw_memory/image/*", "data/raw_memory/video/*"),
)
_GALLERY_MEDIA = MediaSource("mem-gallery", *_GALLERY, ("data/image/*",))


def _task(
    name: str,
    benchmark: str,
    dataset: str,
    adapter_version: str,
    source: tuple[str, str],
    *,
    digest: str | None = None,
    variant: str | None = None,
    auxiliary: tuple[str, ...] = (),
    media: str | None = None,
    media_source: MediaSource | None = None,
) -> TaskSpec:
    return TaskSpec(
        name,
        benchmark,
        dataset,
        adapter_version,
        source[0],
        source[1],
        digest,
        variant,
        auxiliary,
        media,
        media_source,
    )


TASKS: dict[str, TaskSpec] = {
    task.name: task
    for task in (
        _task(
            "locomo-refined",
            "LoCoMo-Refined",
            "locomo-refined/data/raw/locomo_refined.json",
            "locomo_refined_v1",
            _LOCOMO,
            digest="1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17",
        ),
        _task(
            "m3-bench-robot",
            "M3-Bench",
            "m3-agent/data/annotations/robot.json",
            "m3_bench_official_v2",
            _M3,
            digest="f43031bf0216a2ef2e7909f20ecd534e0098da17b63cdf94325a07f1bea372c1",
            variant="robot",
            media="m3-bench/videos/robot",
            media_source=_M3_ROBOT_MEDIA,
        ),
        _task(
            "m3-bench-web",
            "M3-Bench",
            "m3-agent/data/annotations/web.json",
            "m3_bench_official_v2",
            _M3,
            digest="9af953751203471e00c360dbf4b6c072d5b0782ae89e26a60ea366f4d8f14e02",
            variant="web",
            media="m3-bench/videos/web",
            media_source=_M3_WEB_MEDIA,
        ),
        _task(
            "video-mme",
            "Video-MME",
            "video-mme/videomme/test-00000-of-00001.parquet",
            "video_mme_official_v1",
            _VIDEO_MME,
            digest="7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986",
            media="video-mme/data",
            media_source=_VIDEO_MME_MEDIA,
        ),
        _task(
            "video-mme-v2",
            "Video-MME-v2",
            "video-mme-v2/test.parquet",
            "video_mme_v2_official_v1",
            _VIDEO_MME_V2,
            digest="8dc7f8c8830aa49dd08a82592f8276899472a145155dde3bea5dd6914a65a9b4",
            media="video-mme-v2/videos",
            media_source=_VIDEO_MME_V2_MEDIA,
        ),
        _task(
            "egolifeqa",
            "EgoLifeQA",
            "egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json",
            "egolife_qa_official_v3",
            _EGOLIFE,
            digest="688ae079f458132f13150711b7e099fbe4fdedd97f19f5a33bdebb7bfde74a52",
            variant="A1_JAKE",
            media="egolife",
            media_source=_EGOLIFE_MEDIA,
        ),
        _task(
            "egomemreason",
            "EgoMemReason",
            "egomem-reason/annotations_public.jsonl",
            "egomem_reason_official_v2",
            _EGOMEM,
            digest="8ec70ea94396df5fd405dd1fa890e1c70cd8ccdeb7d85ba73689083cd92c2a3b",
            media="egolife",
            media_source=_EGOLIFE_MEDIA,
        ),
        _task(
            "egotempo",
            "EgoTempo",
            "egotempo/egotempo_openQA.json",
            "egotempo_official_v1",
            _EGOTEMPO,
            digest="adc9e7d5b1075a46e2648d4e34260b41d26b8c6e339159b3746a3fe7fdf94eaf",
            media="egotempo/videos",
            media_source=_EGOTEMPO_MEDIA,
        ),
        *(
            _task(
                f"memlens-{window}",
                "MemLens",
                f"memlens/dataset_{window}.json",
                "memlens_official_v1",
                _MEMLENS,
                digest=(
                    "4c75acc3e0a7e1cc71bc962bff3d2a0f1f35e268bc28254ca85f1d9867b03eb9"
                    if window == "32k"
                    else None
                ),
                variant=window,
                auxiliary=("memlens/agent_subset_195.json",),
            )
            for window in ("32k", "64k", "128k", "256k")
        ),
        _task(
            "mm-lifelong-day-test",
            "MM-Lifelong",
            "mm-lifelong/day/test.json",
            "mm_lifelong_official_v1",
            _LIFELONG,
            digest="a2ac08e1c90997acea234a088ac4a46ef9ddf9f0c13354a7d077a076b1004090",
            variant="day_test",
            media="mm-lifelong/videos/day",
            media_source=_LIFELONG_MEDIA,
        ),
        _task(
            "mm-lifelong-week-test",
            "MM-Lifelong",
            "mm-lifelong/week/test.json",
            "mm_lifelong_official_v1",
            _LIFELONG,
            digest="de11ac68f2b632fc02280e9ba5a2ea04747f40b34a10a5e99a7357398bb49cfe",
            variant="week_test",
            media="mm-lifelong/videos/week",
            media_source=_LIFELONG_MEDIA,
        ),
        _task(
            "mm-lifelong-month-train",
            "MM-Lifelong",
            "mm-lifelong/month/train.json",
            "mm_lifelong_official_v1",
            _LIFELONG,
            digest="1f6a252d37e07a51241517595feb6b4176d685614e1ed07a2b1fb84ea6685190",
            variant="month_train",
            media="mm-lifelong/videos/month",
            media_source=_LIFELONG_MEDIA,
        ),
        _task(
            "mm-lifelong-month-val",
            "MM-Lifelong",
            "mm-lifelong/month/val.json",
            "mm_lifelong_official_v1",
            _LIFELONG,
            digest="e288b75531d22a78572383fc1d7b8c78064825ee55f44246dee3e0b1968bbd37",
            variant="month_val",
            media="mm-lifelong/videos/month",
            media_source=_LIFELONG_MEDIA,
        ),
        _task(
            "supermemory-vqa",
            "SuperMemory-VQA",
            "supermemory-vqa/data/json/all_qa.json",
            "supermemory_vqa_official_v3",
            _SUPERMEMORY,
            digest="fbb3f234d8ce79e5cfbe31482d735038035ee13f59a6468a3f3a078c2aa15663",
            variant="subject-1",
            auxiliary=("supermemory-vqa/data/transcripts",),
            media="supermemory-vqa/data/video",
            media_source=_SUPERMEMORY_MEDIA,
        ),
        *(
            _task(
                f"atm-bench-{split}",
                "ATM-Bench",
                f"atm-bench/data/atm-bench/atm-bench{'-hard' if split == 'hard' else ''}.json",
                ATM_BENCH_ADAPTER_VERSION,
                _ATM,
                digest=(
                    "acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a"
                    if split == "hard"
                    else "ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762"
                ),
                variant=split,
                auxiliary=("atm-bench/data/raw_memory/email/emails.json",),
                media="atm-bench/data/raw_memory",
                media_source=_ATM_MEDIA,
            )
            for split in ("main", "hard")
        ),
        *(
            _task(
                f"atm-bench-{split}-sgm",
                "ATM-Bench",
                f"atm-bench/data/atm-bench/atm-bench{'-hard' if split == 'hard' else ''}.json",
                ATM_BENCH_ADAPTER_VERSION,
                _ATM,
                digest=(
                    "acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a"
                    if split == "hard"
                    else "ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762"
                ),
                variant=f"{split}_sgm",
                auxiliary=(
                    "atm-bench/data/raw_memory/email/emails.json",
                    "atm-bench/data/processed_memory/image_batch_results.json",
                    "atm-bench/data/processed_memory/video_batch_results.json",
                ),
            )
            for split in ("main", "hard")
        ),
        _task(
            "mem-gallery",
            "Mem-Gallery",
            "mem-gallery/data/dialog",
            "mem_gallery_official_v2",
            _GALLERY,
            digest="fcd47af2b493cd9a7856cb77a622291b5aa9c6dad12b1f3553d8f569e2c5f6b8",
            media="mem-gallery/data/image",
            media_source=_GALLERY_MEDIA,
        ),
    )
}

GROUPS: dict[str, tuple[str, ...]] = {
    "m3-bench": ("m3-bench-robot", "m3-bench-web"),
    "memlens": tuple(name for name in TASKS if name.startswith("memlens-")),
    "mm-lifelong": tuple(name for name in TASKS if name.startswith("mm-lifelong-")),
    "atm-bench": tuple(name for name in TASKS if name.startswith("atm-bench-")),
    "all": tuple(TASKS),
}

ALIASES = {
    "egolife": "egolifeqa",
    "ego-life-qa": "egolifeqa",
    "egomem": "egomemreason",
    "ego-mem-reason": "egomemreason",
    "m3": "m3-bench",
    "m3-robot": "m3-bench-robot",
    "m3-web": "m3-bench-web",
    "supermemory": "supermemory-vqa",
    "supermemory-subject-1": "supermemory-vqa",
    "atm": "atm-bench",
    "atm-main": "atm-bench-main",
    "atm-hard": "atm-bench-hard",
    "atm-main-sgm": "atm-bench-main-sgm",
    "atm-hard-sgm": "atm-bench-hard-sgm",
}


def expand(names: Sequence[str]) -> tuple[str, ...]:
    """Expand task groups deterministically in catalog order."""
    normalized = tuple(ALIASES.get(_normalize(name), _normalize(name)) for name in names)
    available = (*TASKS, *GROUPS)
    normalized = tuple(
        match
        for name in normalized
        for match in (
            tuple(candidate for candidate in available if fnmatchcase(candidate, name))
            if any(marker in name for marker in "*?[")
            else (name,)
        )
    )
    unknown = tuple(name for name in normalized if name not in TASKS and name not in GROUPS)
    if unknown:
        raise ValueError(f"unknown task: {', '.join(unknown)}; --list-tasks prints valid names")
    wanted = {task for name in normalized for task in GROUPS.get(name, (name,))}
    selected = tuple(name for name in TASKS if name in wanted)
    if not selected:
        raise ValueError("task selection is empty; --tasks list prints valid names")
    return selected


def listing(root: Path, section: str = "all") -> str:
    """Render task groups, pins, and local input readiness without side effects."""
    lines: list[str] = []
    if section in {"all", "groups"}:
        lines.append("groups:")
        lines.extend(f"  {name:<18}{','.join(tasks)}" for name, tasks in GROUPS.items())
    if section in {"all", "tasks"}:
        if lines:
            lines.append("")
        lines.append("tasks:")
        for name, task in TASKS.items():
            inputs = (
                *task.input_paths(root),
                *((task.media_root(root),) if task.media_root(root) is not None else ()),
            )
            missing = tuple(path for path in inputs if not path.exists())
            state = (
                "ready" if not missing else "missing " + ", ".join(str(path) for path in missing)
            )
            pin = f"{task.repository}@{task.revision}"
            lines.append(f"  {name:<28}{task.benchmark:<20}{pin}  {state}")
    if section == "tags":
        lines.extend(("tags:", "  (none)"))
    return "\n".join(lines)


def _normalize(name: str) -> str:
    return name.strip().casefold().replace("_", "-")
