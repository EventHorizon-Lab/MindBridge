"""Where each official release comes from, so naming a task is enough to obtain it.

`docs/benchmarking.md` used to be the only place these coordinates existed, as a wall of
`hf download` and `git clone` commands to run before a benchmark could start. That is what this
table replaces: `--tasks` resolves a task to the files it reads, those files resolve to entries
here, and anything absent is fetched before the sweep begins.

Three properties are deliberate.

**Only the files a task actually reads.** ATM-Bench's Hub repository is 3.2 GB and Mem-Gallery's
is 530 MB, but the questions, emails, and schema-guided text a run consumes are five JSON files
and one directory -- about 40 MB between them, against 302 GB of full releases. Fetching by
declared input rather than by repository is what makes that the difference. MEMLENS is the one
release where the annotation is itself large: its four context windows are 98 MB, 191 MB,
369 MB and 732 MB, so `--tasks all` fetches about 1.4 GB rather than 40 MB.

**Pinned to a commit, every one of them.** A branch name would make the same task name mean
different bytes on different days, which is the drift that makes two scores incomparable, so
`test_releases.py` asserts that every revision here is a 40-character commit. That is what
fixes the corpus; the digests below are the second line.

**Verified where a digest exists.** Most annotations here have a `source_sha256` in
`benchmarks/manifests/dataset-adapters-smoke.json`, recorded when `mindbridge-bench datasets`
last ran; a fetch producing other bytes stops the run and the file is deleted rather than left
to be trusted next time. The rest are covered by the pin alone -- the smoke manifest keys them
by something other than a file name, listed under `RECORDED_DIGESTS` -- and
`mindbridge-bench datasets` is what checks a corpus already on disk.

**Media too, now that something reads it.** This table used to stop at annotations, because a
run consumed prepared-media manifests naming objects already in storage and nothing in the tree
could produce one. The `prepare_*` producers changed that: they cut clips from the release's own
source files, so the source files have to be on disk, and `ensure_media` is what puts them there.
It is deliberately unbounded -- `--tasks video-mme` on an empty corpus downloads 94 GiB of
archives and extracts 95 GB out of them without asking -- because a size gate on an operation
whose whole purpose is "naming a task is enough" only moves the manual step rather than removing
it. Budget roughly twice a release's media size while archives are involved: the volumes are kept
after extraction, since deleting them is what would make the next run download them again.

Where each benchmark's media lands, relative to `--benchmarks-root`. These are the paths the
producers build source files from, so they are the contract rather than an illustration:

- `video-mme` -- `video-mme/data/<youtube_id>.mp4`, 900 files. That is the annotation's
  `source_video_id` (the parquet's `videoID`), **not** its `video_id`, which is an ordinal.
- `video-mme-v2` -- `video-mme-v2/videos/<nnn>.mp4`, 800 files, `001` through `800` zero-padded
  to three digits, which is that release's `video_id` verbatim.
- `egolife` -- `egolife/<A#_NAME>/DAY<n>/DAY<n>_<A#_NAME>_<HHMMSSFF>.mp4` for the six identities
  `A1_JAKE`, `A2_ALICE`, `A3_TASHA`, `A4_LUCIA`, `A5_KATRINA`, `A6_SHURE`. Note the file name
  leads with the day and the *directory* leads with the identity; the release's caption sidecars
  under `EgoLifeCap/` use the opposite order, so a name copied from one does not open the other.
- `egomem-reason` -- no media of its own. Its questions carry an `identity` and a `query_time`
  naming EgoLife streams, so its producer asks for `egolife` and the two benchmarks share one
  copy rather than each holding 477 GiB.
- `mm-lifelong` -- `mm-lifelong/videos/day/0.mp4`, `mm-lifelong/videos/month/<n>.mp4` for 1..23,
  and `mm-lifelong/videos/week/day<n>/DAY<n>_A1_JAKE_<HHMMSSFF>.mp4`. The week scale is EgoLife
  A1_JAKE, re-published here byte for byte, and the release's `video_list.txt` is where its
  videos came from rather than how to get them -- all three scales are on the Hub at the pinned
  revision, so nothing here needs `yt-dlp`.
- `supermemory-vqa` -- `supermemory-vqa/data/video/Person_<n>/<session>.mp4`, 82 files. The
  release's much larger `data/mps` and `data/raw` are Aria sensor streams no runner opens.
- `atm-bench` -- `atm-bench/data/raw_memory/image/<timestamp>.jpg` and
  `atm-bench/data/raw_memory/video/<timestamp>.mp4`.
- `mem-gallery` -- `mem-gallery/data/image/<topic>/<image_key>.jpg`, named relative to the
  dialogue file that references them.
- `m3-robot` -- `m3-bench/videos/robot/<video_id>.mp4`, 100 files. It is the one media set that
  does not live in its benchmark's own release: the annotations are a Git repository and the
  videos are a Hub dataset, which is why `Media` names a release rather than assuming one.

Two media sets cannot be fetched at all, and `UNOBTAINABLE` says so by name rather than letting a
producer discover it as an absent file. `egotempo` is Ego4D, behind a signed access agreement no
unattended download can accept; `m3-web` is 920 YouTube URLs carried in the annotation. Both raise
naming the destination the operator has to fill in themselves.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mindbridge.file_integrity import sha256_file

_RAW_URL = "https://raw.githubusercontent.com"
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class Release:
    """One official release: where it lives, and at which revision."""

    repository: str
    revision: str
    hub: bool = True
    """Hub datasets resolve through `huggingface_hub`; the rest are Git repositories."""


RELEASES: dict[str, Release] = {
    # Keyed by the directory each release occupies under `--benchmarks-root`, which is the first
    # component of every path the catalog names. That is the whole join: a task's declared inputs
    # say which release supplies them, with nothing restated here.
    "locomo-refined": Release(
        "mem-eval-suite/LoCoMo_refined",
        "887091190789e8d6760e70b9edd696539923dc4f",
        hub=False,
    ),
    "m3-agent": Release(
        "ByteDance-Seed/m3-agent",
        "0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c",
        hub=False,
    ),
    "m3-bench": Release("ByteDance-Seed/M3-Bench", "2672152eee36b25ccb38fdbc3b72135347adbb63"),
    # M3-Bench is the only release here no task declares a file from: its annotations ship in the
    # `m3-agent` Git repository above and only its videos live on the Hub. It is a release all the
    # same, and putting its pin anywhere but this table is how the two would drift apart.
    "egotempo": Release(
        "google-research-datasets/egotempo",
        "7022ba77b4d89f51cf34e499767995ccd5c90c7a",
        hub=False,
    ),
    "video-mme": Release("lmms-eval/Video-MME", "ead1408f75b618502df9a1d8e0950166bf0a2a0b"),
    "video-mme-v2": Release(
        "MME-Benchmarks/Video-MME-v2",
        "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
    ),
    "egolife": Release("lmms-lab/EgoLife", "143fb319be7aa5ae210c936bf4f0f3a86092afb0"),
    "supermemory-vqa": Release(
        "OSU-AIoT-MLSys-Lab/SuperMemory-VQA", "1d228e0f10049a8a84c458dded2aa25b1e21ce8f"
    ),
    "egomem-reason": Release("Ted412/EgoMemReason", "7e581505b9dce0e85193a27ae689ff899d0bc507"),
    "memlens": Release("xiyuRenBill/MEMLENS", "afa101a1907cc37db40b50d649547964387b96b7"),
    "mm-lifelong": Release("MM-Lifelong/MM-Lifelong", "248aa82039a574e63a2e524746a7cd8f32330443"),
    "atm-bench": Release("Jingbiao/ATM-Bench", "78e826dc07e97466b2f54443831ef9a83ab8b27c"),
    "mem-gallery": Release("Ethan-Bei/Mem-Gallery", "af912daba984e896e253016b7c7e334ef92c2a6f"),
}


@dataclass(frozen=True, slots=True)
class Media:
    """One benchmark's source media: the release holding it, and which of its files those are."""

    release: str
    patterns: tuple[str, ...]
    """Matched with `fnmatch` against repository paths, so `*` crosses `/` as the Hub client does.

    A pattern ending in `.zip` names an archive to unpack rather than a file to keep, which is
    what tells `ensure_media` the media is packaged rather than loose.
    """


MEDIA: dict[str, Media] = {
    # Keyed by media set rather than by release: `m3-robot` and `m3-web` are two halves of one
    # benchmark that are acquired in completely different ways, and only one of them is here.
    "video-mme": Media("video-mme", ("videos_chunked_*.zip",)),
    "video-mme-v2": Media("video-mme-v2", ("videos/*.zip",)),
    "egolife": Media("egolife", ("A?_*/DAY*/*.mp4",)),
    "mm-lifelong": Media("mm-lifelong", ("videos/*",)),
    "supermemory-vqa": Media("supermemory-vqa", ("data/video/*",)),
    "atm-bench": Media("atm-bench", ("data/raw_memory/image/*", "data/raw_memory/video/*")),
    "mem-gallery": Media("mem-gallery", ("data/image/*",)),
    "m3-robot": Media("m3-bench", ("videos/robot/*",)),
}
"""Where a producer's source files come from, for the media sets something can fetch."""

UNOBTAINABLE: dict[str, str] = {
    "egotempo": (
        "its videos are Ego4D, which is released under a signed access agreement no unattended "
        "download can accept; request access at https://ego4d-data.org, fetch each question's "
        "source_video_id with the ego4d CLI, and cut clip_start_seconds..clip_end_seconds out of "
        "it into <benchmarks-root>/egotempo/videos/<clip_id>.mp4"
    ),
    "m3-web": (
        "its 920 videos are web sources the release distributes as the video_url of each entry "
        "in m3-agent/data/annotations/web.json rather than as files; download them yourself into "
        "<benchmarks-root>/m3-bench/videos/web/<video_id>.mp4"
    ),
}
"""Media that has to be acquired by hand, and the sentence saying how.

Named rather than omitted. A media set absent from both tables is a typo in a producer and says
so; one absent from `MEDIA` alone would be indistinguishable from a fetch that quietly found
nothing, which is the failure this whole module is arranged to make impossible.
"""

RECORDED_DIGESTS: dict[str, str] = {
    # Copied from `benchmarks/manifests/dataset-adapters-smoke.json`, keyed by the file name that
    # manifest records. It lives outside the wheel, so an installed MindBridge cannot read it;
    # `tests/unit/benchmarks/test_releases.py` fails if the two ever disagree.
    "locomo_refined.json": "1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17",
    "robot.json": "f43031bf0216a2ef2e7909f20ecd534e0098da17b63cdf94325a07f1bea372c1",
    "web.json": "9af953751203471e00c360dbf4b6c072d5b0782ae89e26a60ea366f4d8f14e02",
    "test-00000-of-00001.parquet": (
        "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986"
    ),
    "test.parquet": "8dc7f8c8830aa49dd08a82592f8276899472a145155dde3bea5dd6914a65a9b4",
    "EgoLifeQA_A1_JAKE.json": "688ae079f458132f13150711b7e099fbe4fdedd97f19f5a33bdebb7bfde74a52",
    "egotempo_openQA.json": "adc9e7d5b1075a46e2648d4e34260b41d26b8c6e339159b3746a3fe7fdf94eaf",
    "annotations_public.jsonl": "8ec70ea94396df5fd405dd1fa890e1c70cd8ccdeb7d85ba73689083cd92c2a3b",
    "dataset_32k.json": "4c75acc3e0a7e1cc71bc962bff3d2a0f1f35e268bc28254ca85f1d9867b03eb9",
    "all_qa.json": "fbb3f234d8ce79e5cfbe31482d735038035ee13f59a6468a3f3a078c2aa15663",
    "atm-bench.json": "ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762",
    "atm-bench-hard.json": "acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a",
}
"""What each annotation's bytes have to hash to.

MM-Lifelong's four splits and Mem-Gallery's dialogue directory are recorded in that manifest under
keys that are not file names -- `month_val:val.json`, `dialog/*.json` -- so they are absent here
rather than matched by a rule that would silently pair the wrong digest with a file. The test that
compares the two tables asserts exactly which entries are expected to be missing.
"""


def release_for(path: Path, *, root: Path) -> tuple[str, str] | None:
    """Name the release one input belongs to, and the path within it, or None if it has neither.

    A path directly under the root is an operator artifact -- a prepared-media manifest or the
    deployment file -- which no release supplies and nothing here can fetch.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 2 or parts[0] not in RELEASES:
        return None
    return parts[0], "/".join(parts[1:])


def missing_inputs(inputs: Iterable[Path]) -> tuple[Path, ...]:
    """The declared inputs that are not on disk, in the order they were declared."""
    return tuple(path for path in inputs if not path.exists())


def fetch(inputs: Sequence[Path], *, root: Path, announce: object = None) -> tuple[Path, ...]:
    """Download every absent input a release can supply, and return the ones nothing can.

    Returns rather than raises for what it cannot obtain: a prepared-media manifest is missing
    work, not a broken invocation, and the caller reports it against the task that wanted it.
    """
    unobtainable: list[Path] = []
    wanted: dict[str, list[tuple[str, Path]]] = {}
    for path in missing_inputs(inputs):
        located = release_for(path, root=root)
        if located is None:
            unobtainable.append(path)
            continue
        name, within = located
        wanted.setdefault(name, []).append((within, path))
    for name, members in wanted.items():
        _fetch_release(name, members, root=root, announce=announce)
    return tuple(unobtainable)


def ensure_media(
    release: str,
    *,
    root: Path,
    only: Sequence[str] = (),
    announce: Callable[[str], None] | None = None,
) -> Path:
    """Download and extract this media set if absent, and return the directory holding it.

    The return is the release's own directory under the corpus root -- `<root>/video-mme`, and
    `<root>/m3-bench` for `m3-robot` -- not the innermost directory the files sit in. Producers
    join the rest themselves from the layout the module docstring fixes, because that layout is
    per-benchmark and a caller that has to know `data/<youtube_id>.mp4` anyway gains nothing from
    being handed `data` instead.

    Idempotent, and idempotent cheaply enough to call on every run. The Hub client compares each
    file's recorded ETag against what is already in `<destination>/.cache` and transfers nothing
    when they agree, and extraction skips an entry whose target already exists at the archive's
    own length. Length rather than a marker file, because interruption is the case that matters:
    a partly-written entry is shorter than its header says and gets redone, while a `.incomplete`
    download is simply lost -- the Hub client picks a new name for each attempt, so an interrupted
    4 GiB volume is 4 GiB to fetch again no matter what this function does.

    `only` narrows the fetch to particular repository paths, which is what keeps a `--limit 2` run
    off the 477 GiB of EgoLife it will not read. It is refused for a media set that ships as
    archives: a video's bytes live inside one opaque multi-gigabyte volume with no published index
    of which, so the honest answer to "just this video" there is all of them.

    Whether a specific unit arrived is the caller's to check. This says the media set was fetched;
    a producer that then cannot find one video knows which video and which unit wanted it, and
    says so far better than anything here could. Call it when the file a producer wants is absent
    rather than before every producer: a set in `UNOBTAINABLE` raises on the way in, so calling it
    unconditionally would refuse a corpus the operator had already filled in by hand.
    """
    unobtainable = UNOBTAINABLE.get(release)
    if unobtainable is not None:
        raise FileNotFoundError(
            f"{release} media cannot be downloaded automatically: {unobtainable}"
        )
    media = MEDIA.get(release)
    if media is None:
        known = ", ".join(sorted([*MEDIA, *UNOBTAINABLE]))
        raise KeyError(f"no media is registered for {release!r}; the table holds {known}")
    archived = any(pattern.endswith(".zip") for pattern in media.patterns)
    if only and archived:
        raise ValueError(
            f"{release} media ships as {len(media.patterns)} archive patterns and no index says "
            "which archive holds which file, so it cannot be narrowed to "
            f"{', '.join(only)}; call ensure_media without `only`"
        )
    source = RELEASES[media.release]
    patterns = tuple(only) or media.patterns
    destination = root / media.release
    if announce is not None:
        announce(
            f"fetching {release} media from {source.repository}@{source.revision[:12]}: "
            f"{', '.join(patterns)}"
        )
    _download_from_hub(source, patterns, destination=destination)
    for pattern in patterns:
        if pattern.endswith(".zip"):
            for archive in sorted(destination.glob(pattern)):
                _extract(archive, announce=announce)
    return destination


def _extract(archive: Path, *, announce: Callable[[str], None] | None) -> None:
    """Unpack one archive beside itself, leaving entries that are already there alone.

    Beside itself rather than into a fixed directory because both releases that need this put the
    structure they want in one place or the other: Video-MME's volumes sit at the repository root
    and name their entries `data/<id>.mp4`, Video-MME-v2's sit in `videos/` and name theirs
    `<id>.mp4`, and unpacking each into its own parent is what turns both into the layout the
    module docstring documents.
    """
    root = archive.parent.resolve()
    with zipfile.ZipFile(archive) as volume:
        entries = [entry for entry in volume.infolist() if not entry.is_dir()]
        pending = []
        for entry in entries:
            target = archive.parent / entry.filename
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"{archive} holds {entry.filename}, which escapes {root}")
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            pending.append((entry, target))
        if not pending:
            return
        if announce is not None:
            announce(f"extracting {len(pending)} of {len(entries)} files from {archive.name}")
        for entry, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with volume.open(entry) as reading, target.open("wb") as writing:
                shutil.copyfileobj(reading, writing, _CHUNK_BYTES)


def _fetch_release(
    name: str,
    members: Sequence[tuple[str, Path]],
    *,
    root: Path,
    announce: object,
) -> None:
    """Obtain one release's absent files, then hold them to their recorded digest."""
    release = RELEASES[name]
    patterns = tuple(sorted({_pattern(within) for within, _ in members}))
    if callable(announce):
        kind = "hub" if release.hub else "git"
        announce(
            f"fetching {name} from {release.repository}@{release.revision[:12]} ({kind}): "
            f"{', '.join(patterns)}"
        )
    if release.hub:
        _download_from_hub(release, patterns, destination=root / name)
    else:
        for within, _ in members:
            _download_from_git(release, within, destination=root / name / within)
    for _, path in members:
        _require_recorded_digest(path)


def _pattern(within: str) -> str:
    """Turn one declared input into a download pattern.

    A path with no suffix is a directory of the release -- Mem-Gallery's `data/dialog` is the only
    one today -- and every file these adapters read carries an extension, which the release test
    asserts rather than trusts.
    """
    return within if Path(within).suffix else f"{within}/*"


def _download_from_hub(release: Release, patterns: Sequence[str], *, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - exercised by the benchmarks extra probe
        raise ImportError(
            "downloading an official release needs huggingface-hub; "
            "install it with `uv sync --extra benchmarks`"
        ) from error
    snapshot_download(
        repo_id=release.repository,
        repo_type="dataset",
        revision=release.revision,
        allow_patterns=list(patterns),
        local_dir=str(destination),
    )


def _download_from_git(release: Release, within: str, *, destination: Path) -> None:
    """Stream one file out of a Git release at its pinned commit.

    One file over HTTP rather than a clone: these three repositories carry code and media the
    adapters never read, and GitHub rate-limits `git clone` an order of magnitude below a plain
    file read. The commit is in the URL, so this pins exactly as a checkout would.
    """
    import httpx

    url = f"{_RAW_URL}/{release.repository}/{release.revision}/{within}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                handle.write(chunk)
    partial.replace(destination)


def _require_recorded_digest(path: Path) -> None:
    """Refuse a download whose bytes are not the ones this repository recorded.

    The rejected file is deleted. Leaving it at its final path meant a drifted release was
    verified exactly once: the next sweep's `missing_inputs` saw the file present, excluded it
    from `members`, never reached this check, and measured the drifted corpus with no signal.
    Removing it makes the next run fetch again, which is the only outcome that can either
    succeed or fail the same way twice.

    A file with no recorded digest is accepted. That is safe only because every release is
    pinned to a commit -- `test_releases.py` asserts it -- so its bytes are fixed even where no
    digest names them; `RECORDED_DIGESTS` covers the annotations the smoke manifest keys by
    file name, and the module docstring says which are deliberately absent.
    """
    recorded = RECORDED_DIGESTS.get(path.name)
    if recorded is None or not path.exists():
        return
    actual = sha256_file(path)
    if actual != recorded:
        path.unlink()
        raise ValueError(
            f"{path} hashed to {actual}, not the {recorded} recorded in "
            "benchmarks/manifests/dataset-adapters-smoke.json, and has been deleted; upstream "
            "changed the release, so re-run `mindbridge-bench datasets` and record the new "
            "digest before measuring against it"
        )
