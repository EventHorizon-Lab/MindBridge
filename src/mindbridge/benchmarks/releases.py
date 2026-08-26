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

**No media.** The releases hold videos and audio that a run cannot use as files anyway: the
runners read prepared-media manifests naming objects already in storage, and MindBridge contains
no clipper or uploader by design. So a media benchmark still needs that manifest produced
out-of-band; what this saves it is the annotation download, not the preparation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    # The AML text corpora. Three of the six, because the other two cannot be obtained by the
    # rule at the top of this docstring -- only the files a task reads -- and registering them
    # anyway would make the sweep fetch, then fail, mid-run:
    #
    #   beam            mohammadtavakoli78/BEAM@3e12035532eb85768f1a7cd779832b650c4b2ef9
    #                   200 files under chats/{tier}/{conv}/, discovered by glob. Git releases
    #                   here are streamed one declared file at a time from raw.githubusercontent,
    #                   which cannot expand a pattern, and the repository is 695 MB against the
    #                   ~30 MB a run reads.
    #   personamem-v2   bowen-upenn/PersonaMem-v2@0622e56d1cc6f1bc990a5100a6ec4022a60e66a6
    #                   `data/` is 3.9 GB across five history variants and the loader reads one
    #                   of them, but the path it is handed is the root, so the declared input
    #                   cannot be narrowed without changing the loader's signature.
    #
    # Both stay operator-fetched, pinned in `docs/benchmarking.md`; `--list-tasks` reports them
    # as `needs <path>`, which is the state that exists for exactly this.
    "clbench": Release("tencent/CL-bench", "b28a5832a09b0d96c0cf4c22e90d7c60ede25b80"),
    "longmemeval": Release("xiaowu0162/longmemeval", "2ec2a557f339b6c0369619b1ed5793734cc87533"),
    "personamem-v1": Release(
        "bowen-upenn/PersonaMem-v1", "fd7c30f071d5c2ee2a211506783be222d7b6002e"
    ),
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


def _fetch_release(
    name: str,
    members: Sequence[tuple[str, Path]],
    *,
    root: Path,
    announce: object,
) -> None:
    """Obtain one release's absent files, then hold them to their recorded digest."""
    release = RELEASES[name]
    patterns = tuple(sorted({pattern for within, _ in members for pattern in _patterns(within)}))
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


def _patterns(within: str) -> tuple[str, ...]:
    """Turn one declared input into the download patterns that can supply it.

    A path with a suffix is a file and names itself. A path without one is ambiguous, and both
    readings exist in the catalog: Mem-Gallery's `data/dialog` is a directory, LongMemEval's
    `longmemeval_s` is a 266 MB extension-less file. So a suffix-less input asks for both, and
    whichever the release actually holds is what matches -- `allow_patterns` is a filter over the
    repository's own listing, so the reading that is wrong selects nothing rather than failing.

    Guessing "directory" for everything without a suffix, which is what this did while
    Mem-Gallery was the only such input, made `longmemeval_s` resolve to `longmemeval_s/*`: it
    matched no file in the repository, `snapshot_download` succeeded having written nothing, and
    the task then failed on an absent dataset the sweep had just reported as fetched.
    """
    return (within,) if Path(within).suffix else (within, f"{within}/*")


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
