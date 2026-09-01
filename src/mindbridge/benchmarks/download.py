"""Fetch pinned benchmark annotations and published media into the catalog layout."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from collections.abc import Callable, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from mindbridge.benchmarks.task_catalog import TaskSpec

_GITHUB_REPOSITORIES = frozenset(
    {
        "mem-eval-suite/LoCoMo_refined",
        "ByteDance-Seed/m3-agent",
        "google-research-datasets/egotempo",
        "mohammadtavakoli78/BEAM",
        "facebookresearch/open-eqa",
    }
)


def acquire_inputs(spec: TaskSpec, root: Path, *, include_dataset: bool = True) -> None:
    """Download absent catalog inputs at the task's immutable source revision."""
    inputs = (
        spec.input_paths(root) if include_dataset else tuple(root / path for path in spec.auxiliary)
    )
    # A directory input is always re-resolved: it can exist while still missing
    # files, and the fetch is a no-op once every file is present.
    wanted = tuple(path for path in inputs if not path.exists() or path.is_dir())
    if not wanted:
        return
    release_root = root / Path(spec.dataset).parts[0]
    dataset = spec.dataset_path(root)
    if spec.repository in _GITHUB_REPOSITORIES:
        for path in wanted:
            name = path.relative_to(release_root).as_posix()
            if Path(name).suffix:
                _github_file(spec, name, path)
            else:
                _github_directory(spec, _patterns(spec, path, dataset, name), release_root)
    else:
        patterns = tuple(
            pattern
            for path in wanted
            for pattern in _patterns(spec, path, dataset, path.relative_to(release_root).as_posix())
        )
        _snapshot(spec.repository, spec.revision, patterns, release_root)
    absent = tuple(path for path in wanted if not path.exists())
    if absent:
        raise FileNotFoundError(f"download did not produce: {', '.join(map(str, absent))}")


def _patterns(spec: TaskSpec, path: Path, dataset: Path, name: str) -> tuple[str, ...]:
    """Return the release-relative globs that fetch one catalog input.

    A task may pin exactly which files under a directory input it reads. A
    suffix-less input is offered under both spellings because a suffix is not
    evidence of anything: `longmemeval_s` is a 278 MB JSON file with no
    extension, and asking only for `longmemeval_s/*` matched nothing while the
    download still reported success.
    """
    if spec.dataset_patterns and path == dataset:
        return spec.dataset_patterns
    return (name,) if Path(name).suffix else (name, f"{name}/*")


def acquire_media(
    spec: TaskSpec,
    root: Path,
    *,
    patterns: Sequence[str] = (),
    download: bool = True,
    allow_missing: bool = False,
    announce: Callable[[str], None] | None = None,
) -> Path:
    """Download and safely extract one task's pinned Hub media."""
    source = spec.media_source
    if source is None:
        raise ValueError(f"{spec.name} has no downloadable media")
    if source.acquirer is not None:
        raise ValueError(f"{spec.name} media requires its {source.acquirer} acquirer")
    if source.repository is None or source.revision is None:
        raise ValueError(f"{spec.name} media source is incomplete")
    selected = tuple(dict.fromkeys(patterns or source.patterns))
    if not selected:
        raise ValueError(f"{spec.name} media source has no paths")
    destination = root / source.release
    if download:
        if announce is not None:
            description = ", ".join(selected) if len(selected) <= 4 else f"{len(selected)} paths"
            announce(
                f"downloading {spec.name} media from "
                f"{source.repository}@{source.revision[:12]}: {description}"
            )
        _snapshot(source.repository, source.revision, selected, destination)
    archives = tuple(
        archive
        for pattern in selected
        if pattern.endswith(".zip")
        for archive in sorted(destination.glob(pattern))
    )
    for archive in archives:
        _extract_zip(archive, announce=announce)
    missing = tuple(
        pattern
        for pattern in selected
        if not tuple(destination.glob(pattern)) and (download or not pattern.endswith(".zip"))
    )
    if missing and not allow_missing:
        action = "download did not produce" if download else "offline media is missing"
        raise FileNotFoundError(f"{spec.name} {action}: {', '.join(missing)} under {destination}")
    return destination


def _snapshot(repository: str, revision: str, patterns: Sequence[str], destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "benchmark downloads require `uv sync --extra benchmarks --extra local`"
        ) from error
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(patterns),
        local_dir=destination,
    )


def _extract_zip(
    archive: Path,
    *,
    announce: Callable[[str], None] | None,
) -> None:
    """Resume extraction beside an archive without accepting links or path traversal."""
    root = archive.parent.resolve()
    with zipfile.ZipFile(archive) as volume:
        entries = [entry for entry in volume.infolist() if not entry.is_dir()]
        targets: set[Path] = set()
        pending: list[tuple[zipfile.ZipInfo, Path]] = []
        for entry in entries:
            target = (archive.parent / entry.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(
                    f"{archive} contains a path outside its directory: {entry.filename}"
                )
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"{archive} contains a symbolic link: {entry.filename}")
            if target in targets:
                raise ValueError(f"{archive} contains a duplicate path: {entry.filename}")
            targets.add(target)
            if target.is_file() and target.stat().st_size == entry.file_size:
                continue
            pending.append((entry, target))
        if announce is not None and pending:
            announce(f"extracting {len(pending)} files from {archive}")
        for entry, target in pending:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with (
                    volume.open(entry) as reading,
                    NamedTemporaryFile(
                        mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
                    ) as writing,
                ):
                    temporary = Path(writing.name)
                    shutil.copyfileobj(reading, writing, 1 << 20)
                    writing.flush()
                    os.fsync(writing.fileno())
                os.replace(temporary, target)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def _github_directory(spec: TaskSpec, patterns: Sequence[str], release_root: Path) -> None:
    """Fetch every pinned-revision blob matching `patterns` under a directory.

    GitHub serves no recursive raw endpoint, so the revision's tree is listed
    once and each matching blob is fetched individually. Restricting to the
    task's own patterns matters: BEAM ships a `.pickle` next to every JSON its
    loader reads, and its 10M tier is a gigabyte of chat transcripts.
    """
    url = f"https://api.github.com/repos/{spec.repository}/git/trees/{spec.revision}"
    response = httpx.get(url, params={"recursive": "1"}, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("truncated"):
        raise RuntimeError(f"{spec.repository}@{spec.revision[:12]} tree listing was truncated")
    names = tuple(
        str(entry["path"])
        for entry in payload.get("tree", ())
        if entry.get("type") == "blob"
        and any(fnmatchcase(str(entry["path"]), pattern) for pattern in patterns)
    )
    if not names:
        raise FileNotFoundError(
            f"{spec.name} matched no files under {spec.repository}@{spec.revision[:12]}: "
            f"{', '.join(patterns)}"
        )
    for name in names:
        destination = release_root / name
        if destination.is_file():
            continue
        _github_file(spec, name, destination)


def _github_file(spec: TaskSpec, name: str, destination: Path) -> None:
    if not Path(name).suffix:
        raise FileNotFoundError(f"cannot fetch a GitHub directory input: {name}")
    url = f"https://raw.githubusercontent.com/{spec.repository}/{spec.revision}/{name}"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
            response.raise_for_status()
            with NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_bytes(1 << 20):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
