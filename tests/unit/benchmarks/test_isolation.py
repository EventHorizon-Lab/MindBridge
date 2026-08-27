"""Physical benchmark isolation checks."""

from __future__ import annotations

import multiprocessing
from contextlib import suppress
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from mindbridge.benchmarks.isolation import BenchmarkRun
from mindbridge.infrastructure.local import (
    DataDirectoryInUseError,
    LocalStore,
    StoredMemory,
)

_MEMORY_ID = "memory-shared"
_CONTENT = "The wrench is in drawer two"
_NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def _write_unit(data_dir: str, unit: int, control: Connection) -> None:
    try:
        with LocalStore(data_dir) as store:
            store.write_memory(
                StoredMemory(
                    memory_id=_MEMORY_ID,
                    content=_CONTENT,
                    metadata_json=f'{{"unit":{unit}}}',
                    created_at=_NOW,
                    updated_at=_NOW,
                )
            )
            stored = store.read_memory(_MEMORY_ID)
            if stored is None:
                raise RuntimeError("child could not read its memory")
            control.send(stored.metadata_json)
            control.recv()
    finally:
        control.close()


def test_layout_is_stable_safe_and_collision_free(tmp_path: Path) -> None:
    run = BenchmarkRun(tmp_path, "../M3/video", "../run:01")
    other_benchmark = BenchmarkRun(tmp_path, ".._M3/video", "../run:01")
    other_run = BenchmarkRun(tmp_path, "../M3/video", ".._run:01")

    assert run.path.relative_to(tmp_path) == run.relative_layout
    assert (
        len({run.relative_layout, other_benchmark.relative_layout, other_run.relative_layout}) == 3
    )
    assert len(run.relative_layout.parts) == 2
    assert all(
        part not in {"", ".", ".."} and "/" not in part for part in run.relative_layout.parts
    )

    unit_ids = ("a/b", "a_b", "../a", "A", "a", "é", "e\u0301")
    paths = tuple(run.unit_dir(unit_id) for unit_id in unit_ids)
    assert len(set(paths)) == len(unit_ids)
    assert all(path.parent == run.path for path in paths)

    with pytest.raises(FileExistsError, match="already exists"):
        run.unit_dir(unit_ids[0])
    with pytest.raises(FileExistsError, match="not empty"):
        BenchmarkRun(tmp_path, "../M3/video", "../run:01")

    resumed = BenchmarkRun(tmp_path, "../M3/video", "../run:01", resume=True)
    assert resumed.relative_layout == run.relative_layout
    assert resumed.unit_dir(unit_ids[0]) == paths[0]


@pytest.mark.parametrize("value", ["", " ", " leading", "trailing "])
def test_layout_rejects_ambiguous_identifiers(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="non-empty and trimmed"):
        BenchmarkRun(tmp_path, value, "run-01")
    with pytest.raises(ValueError, match="non-empty and trimmed"):
        BenchmarkRun(tmp_path, "benchmark", "run-01").unit_dir(value)


def test_parallel_units_have_independent_local_stores(tmp_path: Path) -> None:
    run = BenchmarkRun(tmp_path, "parallel", "run-01")
    unit_dirs = [run.unit_dir(f"unit-{index}") for index in range(8)]
    context = multiprocessing.get_context("spawn")
    parents: list[Connection] = []
    processes: list[BaseProcess] = []

    for index, data_dir in enumerate(unit_dirs):
        parent, child = context.Pipe()
        process = context.Process(target=_write_unit, args=(str(data_dir), index, child))
        process.start()
        child.close()
        parents.append(parent)
        processes.append(process)

    try:
        observed = []
        for parent in parents:
            if not parent.poll(15):
                pytest.fail("benchmark unit did not report its write")
            observed.append(parent.recv())
        assert observed == [f'{{"unit":{index}}}' for index in range(8)]

        with pytest.raises(DataDirectoryInUseError, match="already in use"):
            LocalStore(unit_dirs[0])
    finally:
        for parent in parents:
            with suppress(BrokenPipeError, EOFError):
                parent.send("close")
            parent.close()
        for child_process in processes:
            child_process.join(15)
            if child_process.is_alive():
                child_process.terminate()
                child_process.join(15)

    assert all(process.exitcode == 0 for process in processes)
    for index, data_dir in enumerate(unit_dirs):
        with LocalStore(data_dir) as store:
            stored = store.read_memory(_MEMORY_ID)
            assert stored is not None
            assert stored.content == _CONTENT
            assert stored.metadata_json == f'{{"unit":{index}}}'
