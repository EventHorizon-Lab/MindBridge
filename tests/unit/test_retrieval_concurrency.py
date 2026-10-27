"""Retrieval snapshots must protect media without serializing scoring and ranking."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event
from typing import cast

import pytest

import mindbridge.infrastructure.local.store.index as index_module
import mindbridge.kernel.retrieval as retrieval_module
from mindbridge import (
    Blob,
    IndexQuantization,
    Memory,
    Modality,
    RetrievalMode,
    SearchHit,
    StorageError,
)
from mindbridge.kernel.embedding import DOCUMENT_TASK
from mindbridge.kernel.ranking import parent_index_signals
from mindbridge.models.base import EmbedTask, ModelInput


class _Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = "retrieval-concurrency-fixture-v1"
    embedding_space = "retrieval-concurrency-fixture-v1:64"
    embedding_dimension = 64

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (
                (1.0, 0.0)
                if task is EmbedTask.QUERY
                else (0.9, math.sqrt(0.19))
                if "target" in value.text
                else (0.8, 0.6)
            )
            + (0.0,) * 62
            for value in inputs
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "stage",
    ("max_cosine_part_scores", "max_cosine_scores", "lexical_relevance", "ranking_signals"),
)
@pytest.mark.parametrize("fail", (False, True))
def test_delete_and_search_progress_during_ranking_while_media_remains_leased(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, fail: bool
) -> None:
    # One ID per SQL batch makes a deletion between scoring batches observable unless the
    # scoring stream holds a snapshot. Media keys give each parent multiple embedding rows.
    monkeypatch.setattr(index_module, "SQLITE_PARAMETER_BATCH", 1)
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        index_quantization=(
            IndexQuantization.NONE if stage == "max_cosine_scores" else IndexQuantization.INT8
        ),
        retrieval_mode=RetrievalMode.HYBRID,
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add(("shared target", Blob(b"target", "image/png")))
        decoy = memory.add(("shared neighbor", Blob(b"neighbor", "image/png")))
        target_path, decoy_path = target.assets[0].path, decoy.assets[0].path
        assert target_path is not None and decoy_path is not None
        if stage == "max_cosine_scores":
            # Make both parents lexical-only candidates so unquantized completion is required.
            monkeypatch.setattr(memory._index, "search", lambda *_args, **_kwargs: ())
        expected = memory.search_with_trace("shared", limit=2)
        entered, release = Event(), Event()
        original = cast(Callable[..., object], getattr(retrieval_module, stage))

        def paused(*args: object, **kwargs: object) -> object:
            if not entered.is_set():
                entered.set()
                assert release.wait(10)
                if fail:
                    raise ValueError("injected ranking failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(retrieval_module, stage, paused)
        with ThreadPoolExecutor(max_workers=3) as pool:
            searching = pool.submit(memory.search_with_trace, "shared", limit=2)
            try:
                assert entered.wait(5)
                assert pool.submit(memory.delete, target.id).result(timeout=3)
                assert target_path.is_file()
                later = pool.submit(memory.search, "shared", limit=2).result(timeout=3)
                assert [hit.id for hit in later] == [decoy.id]
            finally:
                release.set()
            if fail:
                with pytest.raises(
                    StorageError if stage.startswith("max_cosine") else ValueError,
                    match=r"score search candidates|injected ranking failure",
                ):
                    searching.result(timeout=5)
            else:
                result = searching.result(timeout=5)
                assert result.hits == expected.hits
                assert result.trace == expected.trace
        assert not target_path.exists()
        assert memory.delete(decoy.id)
        assert not decoy_path.exists()
        # A leaked scoring reader would prevent a truncate checkpoint even after operations end.
        with closing(sqlite3.connect(tmp_path / "state.sqlite3", timeout=0.1)) as connection:
            assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0


def test_embedding_stream_keeps_one_snapshot_across_sql_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(index_module, "SQLITE_PARAMETER_BATCH", 1)
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        first = memory.add("target")
        second = memory.add("neighbor")
        with closing(
            memory._store.index.iter_memory_embedding_parts(
                (first.id, second.id), space_id=memory._backends.space_id, task=DOCUMENT_TASK
            )
        ) as parts:
            assert next(parts)[0] == first.id
            assert memory.delete(second.id)
            assert [memory_id for memory_id, _embedding_id, _vector in parts] == [second.id]


def test_delete_before_authoritative_hydration_does_not_return_a_snapshot_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Memory(tmp_path, embedder=_Embedder(), ambiguity_margin=0) as memory:
        target = memory.add("shared target")
        original = cast(Callable[..., object], parent_index_signals)

        def deleted(*args: object, **kwargs: object) -> object:
            signals = original(*args, **kwargs)
            assert memory.delete(target.id)
            return signals

        monkeypatch.setattr(retrieval_module, "parent_index_signals", deleted)
        assert memory.search("shared") == ()


@pytest.mark.parametrize("missing_target", (False, True))
def test_media_validation_is_limited_to_selected_candidates(
    tmp_path: Path, missing_target: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=_Embedder(),
        index_quantization=IndexQuantization.INT8,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.85,
        ambiguity_margin=0,
    ) as memory:
        target = memory.add(("target", Blob(b"target", "image/png")))
        decoy = memory.add(("neighbor", Blob(b"neighbor", "image/png")))
        missing_path = (target if missing_target else decoy).assets[0].path
        assert missing_path is not None
        missing_path.unlink()
        if missing_target:
            with pytest.raises(StorageError, match="lease local media"):
                memory.search("query")
        else:
            assert [hit.id for hit in memory.search("query")] == [target.id]


def test_only_selected_media_remains_leased_during_hit_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Memory(tmp_path, embedder=_Embedder(), minimum_relevance=0, ambiguity_margin=0) as memory:
        target = memory.add(("shared target", Blob(b"target", "image/png")))
        decoy = memory.add(("shared neighbor", Blob(b"neighbor", "image/png")))
        target_path, decoy_path = target.assets[0].path, decoy.assets[0].path
        assert target_path is not None and decoy_path is not None
        entered, release = Event(), Event()
        original = memory._hydrator.search_hit

        def paused(*args: object, **kwargs: object) -> SearchHit:
            entered.set()
            assert release.wait(10)
            return cast(Callable[..., SearchHit], original)(*args, **kwargs)

        monkeypatch.setattr(memory._hydrator, "search_hit", paused)
        with ThreadPoolExecutor(max_workers=2) as pool:
            searching = pool.submit(memory.search, "shared", limit=1)
            try:
                assert entered.wait(5)
                assert memory.delete(decoy.id)
                assert not decoy_path.exists()
                assert memory.delete(target.id)
                assert target_path.is_file()
            finally:
                release.set()
            assert [hit.id for hit in searching.result(timeout=5)] == [target.id]
        assert not target_path.exists()
