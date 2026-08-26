"""Checks for the official-release table the sweep downloads from.

The download itself is not exercised here — it reaches the network. What is checked is
everything that decides *what* would be downloaded and whether it would be accepted: the join
from a declared input to the release that supplies it, the digest table's agreement with the
committed manifest, and the rule that turns an input into a download pattern.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mindbridge.benchmarks.releases import (
    RECORDED_DIGESTS,
    RELEASES,
    _pattern,
    _require_recorded_digest,
    fetch,
    missing_inputs,
    release_for,
)
from mindbridge.benchmarks.task_catalog import TASKS, task_inputs

SMOKE_MANIFEST = (
    Path(__file__).parents[3] / "benchmarks" / "manifests" / "dataset-adapters-smoke.json"
)

UNKEYED_BY_FILE_NAME = frozenset(
    {
        # MM-Lifelong's four splits are all named for their split rather than their file, because
        # three of them would otherwise collide on `test.json`; Mem-Gallery's digest covers a
        # directory. Pairing either with a file by name would attach the wrong digest.
        "day_test:test.json",
        "week_test:test.json",
        "month_train:train.json",
        "month_val:val.json",
        "dialog/*.json",
    }
)


def test_every_recorded_digest_matches_the_committed_dataset_smoke() -> None:
    """The manifest is outside the wheel, so the table is a copy — and copies drift."""
    recorded = {
        dataset["source_file"]: dataset["source_sha256"]
        for dataset in json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))["datasets"]
    }

    keyed_by_file_name = {
        name: digest for name, digest in recorded.items() if name not in UNKEYED_BY_FILE_NAME
    }

    assert keyed_by_file_name == RECORDED_DIGESTS
    # And the exclusions are exactly the entries whose key is not a file name, so a new
    # dataset cannot join that set by accident.
    assert set(recorded) - set(RECORDED_DIGESTS) == UNKEYED_BY_FILE_NAME


def test_every_release_a_task_reads_from_is_one_the_table_can_obtain() -> None:
    """A catalog entry pointing at a release with no source is a task that can never run."""
    root = Path("/corpus")
    for name, task in TASKS.items():
        for path in task.inputs(root=root):
            located = release_for(path, root=root)
            if located is None:
                # The only inputs without a release are the manifests an operator produces, and
                # those sit directly under the root rather than inside a release directory.
                assert path.parent == root, f"{name} reads {path} from no known release"
                continue
            assert located[0] in RELEASES


def test_every_declared_input_is_a_file_or_the_one_directory_that_is_not() -> None:
    """`_pattern` tells the two apart by extension, so an extensionless file would break it."""
    extensionless = {
        str(path)
        for task in TASKS.values()
        for path in task.inputs(root=Path("/corpus"))
        if not path.suffix
    }

    assert extensionless == {"/corpus/mem-gallery/data/dialog"}
    assert _pattern("data/dialog") == "data/dialog/*"
    assert _pattern("data/atm-bench/atm-bench.json") == "data/atm-bench/atm-bench.json"


def test_an_operator_produced_manifest_is_reported_rather_than_downloaded(
    tmp_path: Path,
) -> None:
    """Nothing can fetch a prepared-media manifest, and pretending otherwise would hang a run."""
    operator_artifact = tmp_path / "m3-prepared-robot.json"

    unobtainable = fetch((operator_artifact,), root=tmp_path)

    assert unobtainable == (operator_artifact,)


def test_a_present_input_is_never_refetched(tmp_path: Path) -> None:
    """The corpus is hundreds of gigabytes; a second sweep must not re-download any of it."""
    present = tmp_path / "locomo-refined" / "data" / "raw" / "locomo_refined.json"
    present.parent.mkdir(parents=True)
    present.write_text("[]", encoding="utf-8")

    assert missing_inputs((present,)) == ()
    # No announcement means nothing was fetched; a network call here would raise instead.
    assert fetch((present,), root=tmp_path, announce=_never) == ()


def test_a_release_whose_bytes_changed_upstream_stops_the_run(tmp_path: Path) -> None:
    """An auto-download with no digest check is how a corpus silently becomes a different one."""
    path = tmp_path / "locomo_refined.json"
    path.write_text("not the official release", encoding="utf-8")

    with pytest.raises(ValueError, match="re-run `mindbridge-bench datasets`"):
        _require_recorded_digest(path)


def test_a_file_with_no_recorded_digest_is_left_alone(tmp_path: Path) -> None:
    """MM-Lifelong's splits are recorded under keys that are not file names, so they have none."""
    path = tmp_path / "val.json"
    path.write_text("whatever the release says", encoding="utf-8")

    _require_recorded_digest(path)


def test_every_release_is_pinned_to_a_commit() -> None:
    """A branch name makes one task name mean different bytes on different days.

    That is the drift two scores cannot survive, and it is invisible: the fetch succeeds, the
    run succeeds, and only the number moves. A digest catches it for the annotations the smoke
    manifest keys by file name; for the rest the pin is the only thing that does, so the pin is
    the property asserted here rather than left to review.
    """
    unpinned = {
        name: release.revision
        for name, release in RELEASES.items()
        if not re.fullmatch(r"[0-9a-f]{40}", release.revision)
    }

    assert unpinned == {}


def test_a_download_that_fails_its_digest_is_deleted(tmp_path: Path) -> None:
    """A rejected file left at its final path is verified once and trusted forever after.

    `missing_inputs` skips what exists, so the next sweep never reaches the check that just
    failed and measures the drifted corpus in silence. Deleting it is what makes the failure
    reproducible.
    """
    path = tmp_path / "locomo_refined.json"
    path.write_text("not what upstream published", encoding="utf-8")

    with pytest.raises(ValueError, match="has been deleted"):
        _require_recorded_digest(path)

    assert not path.exists()


def test_task_inputs_name_the_files_each_task_reads() -> None:
    inputs = task_inputs(["locomo-refined"], root=Path("/corpus"))

    assert inputs == {
        "locomo-refined": (Path("/corpus/locomo-refined/data/raw/locomo_refined.json"),)
    }


def _never(message: str) -> None:
    raise AssertionError(f"nothing should have been fetched, but got: {message}")
