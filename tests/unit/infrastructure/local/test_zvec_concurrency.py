"""Real-native characterization of Zvec reads during durable segment changes."""

from __future__ import annotations

import faulthandler
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event

import pytest

import mindbridge.infrastructure.local.zvec_index as zvec_index_module
from mindbridge.infrastructure.local.store import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import ZvecIndex


def _document(number: int) -> IndexDocument:
    anchor = number == -1
    identifier = "anchor" if anchor else f"moving_{number}"
    return IndexDocument(
        embedding=StoredEmbedding(
            embedding_id=identifier,
            memory_id=f"memory_{identifier}",
            values=(1.0, 0.0) if anchor else (0.0, 1.0),
            model_id="concurrency-test",
            space_id="concurrency-space",
            task="document",
            created_at=datetime(2026, 10, 26, tzinfo=timezone.utc),
        ),
        content="quasar 東京" if anchor else f"ordinary changing document {number}",
        metadata_json="{}",
        memory_type="episodic" if anchor else "semantic",
        place_id="observatory" if anchor else None,
        identity_ids=("astronomer",) if anchor else (),
    )


def _write_batches(
    index: ZvecIndex,
    start: Barrier,
    writing: Event,
    finished: Event,
    read_during_batch: tuple[Event, Event],
    counts: list[int],
) -> None:
    start.wait()
    writing.set()
    try:
        for batch in range(16):
            for event in read_during_batch:
                event.clear()
            documents = tuple(_document(batch * 64 + offset) for offset in range(64))
            index.upsert(documents)
            index.delete([document.embedding.embedding_id for document in documents[:32]])
            index.flush()
            assert all(event.wait(timeout=10) for event in read_during_batch)
            print(f"flushed batch {batch + 1}; reader counts {counts}", flush=True)
    finally:
        writing.clear()
        finished.set()


def _exercise_native_concurrency(path: Path) -> None:
    start = Barrier(3, timeout=10)
    finished = Event()
    writing = Event()
    read_during_batch = (Event(), Event())
    counts = [0, 0]

    with ZvecIndex(path, dimension=2) as index:
        index.upsert((_document(-1),))
        index.flush()
        index.optimize()

        def dense_reader() -> None:
            start.wait()
            while not finished.is_set():
                hits = index.search(
                    (1.0, 0.0),
                    limit=1,
                    space_id="concurrency-space",
                    task="document",
                    memory_type="episodic",
                    place_id="observatory",
                    identity_id="astronomer",
                )
                assert len(hits) == 1 and hits[0].id == "anchor"
                assert hits[0].relevance == pytest.approx(1.0)
                if writing.is_set():
                    counts[0] += 1
                    read_during_batch[0].set()

        def lexical_reader() -> None:
            start.wait()
            while not finished.is_set():
                for text in ("quasar", "東京"):
                    hits = index.lexical_search(
                        text,
                        limit=1,
                        space_id="concurrency-space",
                        task="document",
                        memory_type="episodic",
                        place_id="observatory",
                        identity_id="astronomer",
                    )
                    assert len(hits) == 1 and hits[0].id == "anchor"
                    assert hits[0].lexical_match
                if writing.is_set():
                    counts[1] += 1
                    read_during_batch[1].set()

        with ThreadPoolExecutor(max_workers=3) as executor:
            readers = (executor.submit(dense_reader), executor.submit(lexical_reader))
            written = executor.submit(
                _write_batches, index, start, writing, finished, read_during_batch, counts
            )
            try:
                for future in as_completed((*readers, written), timeout=40):
                    future.result()
            finally:
                finished.set()

        assert all(count >= 16 for count in counts), counts
        # The workload must exercise durable native segment changes before they are merged.
        assert len(tuple(path.rglob("*.ipc"))) >= 4
        assert index.doc_count == 513
        index.optimize()

    with ZvecIndex(path, dimension=2) as reopened:
        assert reopened.doc_count == 513
        assert reopened.search((1.0, 0.0), limit=1)[0].id == "anchor"
        for text in ("quasar", "東京"):
            assert [hit.id for hit in reopened.lexical_search(text)] == ["anchor"]
        survivors = {hit.id for hit in reopened.lexical_search("ordinary", limit=1024)}
        assert survivors == {
            f"moving_{batch * 64 + offset}" for batch in range(16) for offset in range(32, 64)
        }


def _exercise_flush_pressure_compaction(path: Path) -> None:
    """Flush must drop its shared lease before replacing the handle under FD pressure."""
    pressure_checked = Event()

    def pressure() -> bool:
        pressure_checked.set()
        return True

    with ZvecIndex(path, dimension=2) as index:
        index.upsert((_document(-1),))
        index.flush()
        original = zvec_index_module._file_descriptor_pressure
        zvec_index_module._file_descriptor_pressure = pressure
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                with index._gate.read():
                    flushed = executor.submit(index.flush)
                    assert pressure_checked.wait(timeout=5)
                    assert not flushed.done()
                    written = executor.submit(index.upsert, (_document(0),))
                flushed.result(timeout=10)
                written.result(timeout=10)
        finally:
            zvec_index_module._file_descriptor_pressure = original
        assert index.doc_count == 2
        index.flush()
    with ZvecIndex(path, dimension=2) as reopened:
        assert reopened.doc_count == 2
        assert reopened.search((1.0, 0.0), limit=1)[0].id == "anchor"
        assert reopened.lexical_search("東京", limit=1)[0].id == "anchor"


def test_native_readers_survive_segment_changes_and_fts_reopen(tmp_path: Path) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.8 native wheel is not installed")
    # Native failures must leave the rest of pytest alive and include the child diagnostics.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path / "index")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    status = signal.Signals(-result.returncode).name if result.returncode < 0 else result.returncode
    assert result.returncode == 0, (
        f"native concurrency subprocess exited with {status}\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
    faulthandler.enable()
    _exercise_native_concurrency(Path(sys.argv[1]))
    _exercise_flush_pressure_compaction(Path(sys.argv[1]).with_name("pressure-index"))
