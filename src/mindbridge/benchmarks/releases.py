"""Where each official release comes from, so naming a task is enough to obtain it.

`docs/benchmarking.md` used to be the only place these coordinates existed, as a wall of
`hf download` and `git clone` commands to run before a benchmark could start. That is what this
table replaces: `--tasks` resolves a task to the files it reads, those files resolve to entries
here, and anything absent is fetched before the sweep begins.

Three properties are deliberate.

**Only the files a task actually reads.** ATM-Bench's Hub repository is 3.2 GB and Mem-Gallery's
is 530 MB, but the questions, emails, and schema-guided text a run consumes are five JSON files
and one directory. Fetching by declared input rather than by repository is the difference between
40 MB and 302 GB.

**Pinned, and verified against a digest this repository already committed.** Every annotation
here has a `source_sha256` in `benchmarks/manifests/dataset-adapters-smoke.json`, recorded when
`mindbridge-bench datasets` last ran. A fetch that produces different bytes is upstream drift,
and the run stops rather than quietly measuring a different corpus.

**No media.** The releases hold videos and audio that a run cannot use as files anyway: the
runners read prepared-media manifests naming objects already in storage, and MindBridge contains
no clipper or uploader by design. So a media benchmark still needs that manifest produced
out-of-band; what this saves it is the annotation download, not the preparation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_HUB_URL = "https://huggingface.co"
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
    "egotempo": Release(
        "google-research-datasets/egotempo",
        "7022ba77b4d89f51cf34e499767995ccd5c90c7a",
        hub=False,
    ),
    "video-mme": Release("lmms-eval/Video-MME", "main"),
    "video-mme-v2": Release(
        "MME-Benchmarks/Video-MME-v2",
        "6e4bebb03202e1ddbf3d37703e560e51c5aa2d64",
    ),
    "egolife": Release("lmms-lab/EgoLife", "main"),
    "supermemory-vqa": Release("OSU-AIoT-MLSys-Lab/SuperMemory-VQA", "main"),
    "egomem-reason": Release("Ted412/EgoMemReason", "main"),
    "memlens": Release("xiyuRenBill/MEMLENS", "main"),
    "mm-lifelong": Release("MM-Lifelong/MM-Lifelong", "main"),
    "atm-bench": Release("Jingbiao/ATM-Bench", "78e826dc07e97466b2f54443831ef9a83ab8b27c"),
    "mem-gallery": Release("Ethan-Bei/Mem-Gallery", "af912daba984e896e253016b7c7e334ef92c2a6f"),
}

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


def missing_inputs(inputs: Iterable[Path], *, root: Path) -> tuple[Path, ...]:
    """The declared inputs that are not on disk, in the order they were declared."""
    return tuple(path for path in inputs if not path.exists())


def fetch(inputs: Sequence[Path], *, root: Path, announce: object = None) -> tuple[Path, ...]:
    """Download every absent input a release can supply, and return the ones nothing can.

    Returns rather than raises for what it cannot obtain: a prepared-media manifest is missing
    work, not a broken invocation, and the caller reports it against the task that wanted it.
    """
    unobtainable: list[Path] = []
    wanted: dict[str, list[tuple[str, Path]]] = {}
    for path in missing_inputs(inputs, root=root):
        located = release_for(path, root=root)
        if located is None:
            unobtainable.append(path)
            continue
        name, within = located
        wanted.setdefault(name, []).append((within, path))
    for name, members in wanted.items():
        _fetch_release(name, members, root=root, announce=announce)
    return tuple(unobtainable)


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
    """Refuse a download whose bytes are not the ones this repository recorded."""
    recorded = RECORDED_DIGESTS.get(path.name)
    if recorded is None or not path.exists():
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != recorded:
        raise ValueError(
            f"{path} hashes to {actual}, not the {recorded} recorded in "
            "benchmarks/manifests/dataset-adapters-smoke.json; upstream changed the release, so "
            "re-run `mindbridge-bench datasets` and record the new digest before measuring "
            "against it"
        )
