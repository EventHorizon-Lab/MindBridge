"""Read-cache admission, scope isolation, transaction boundaries, and bounded retention."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Event

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.store import LocalStoreClosedError
from mindbridge.infrastructure.local.store._cache import ReadCache

_NOW = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)


def _memory(memory_id: str, content: str) -> StoredMemory:
    return StoredMemory(
        memory_id=memory_id,
        content=content,
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _store_one(store: LocalStore, suffix: str, content: str) -> tuple[str, str]:
    memory_id, embedding_id = f"memory-{suffix}", f"embedding-{suffix}"
    store.records.write_memory(
        _memory(memory_id, content),
        (
            StoredEmbedding(
                embedding_id=embedding_id,
                memory_id=memory_id,
                values=(0.6, 0.8),
                model_id="cache-fixture",
                space_id="cache-fixture:2",
                task="retrieval.document",
                created_at=_NOW,
                normalized=True,
            ),
        ),
    )
    return memory_id, embedding_id


def _must_not_iterate() -> Iterator[tuple[str, int]]:
    raise AssertionError("a disabled cache evaluated values it cannot retain")
    yield "unreachable", 0


def test_frequent_commits_do_not_evaluate_cache_fills() -> None:
    cache: ReadCache[int] = ReadCache(2)
    for generation in range(128):
        assert cache.get(("first",), generation=generation) is None
        cache.put(_must_not_iterate(), generation=generation)


def test_oversized_working_set_stops_filling_until_a_new_generation() -> None:
    cache: ReadCache[int] = ReadCache(2)
    for _ in range(128):
        assert cache.get(("first",), generation=0) is None
    cache.put((("first", 1), ("second", 2), ("third", 3)), generation=0)
    assert cache.get(("first",), generation=0) is None
    cache.put(_must_not_iterate(), generation=0)

    for _ in range(128):
        assert cache.get(("first",), generation=1) is None
    cache.put((("first", 4),), generation=1)
    assert cache.get(("first",), generation=1) == (4,)


def test_old_snapshot_cannot_publish_into_a_new_generation() -> None:
    cache: ReadCache[int] = ReadCache(2)
    for _ in range(128):
        cache.get(("first",), generation=0)
    cache.put((("first", 1),), generation=0)
    assert cache.get(("first", "missing"), generation=0) is None
    for _ in range(128):
        cache.get(("first",), generation=1)
    cache.put((("first", 2),), generation=1)
    cache.put(_must_not_iterate(), generation=0)
    assert cache.get(("first", "first"), generation=1) == (2, 2)


@pytest.mark.parametrize(
    "changed",
    (
        replace(_memory("cached", "original"), content="changed"),
        replace(_memory("cached", "original"), metadata_json='{"state":"changed"}'),
        replace(_memory("cached", "original"), memory_type="episodic"),
        replace(_memory("cached", "original"), occurred_at=_NOW),
        replace(_memory("cached", "original"), last_accessed_at=_NOW, access_count=1),
        replace(_memory("cached", "original"), forgotten_at=_NOW),
    ),
)
def test_text_hydration_observes_changed_columns_and_deletion(
    tmp_path: Path, changed: StoredMemory
) -> None:
    with LocalStore(tmp_path) as store:
        original = _memory("cached", "original")
        store.records.write_memory(original)
        assert store.records.read_memory(original.memory_id) == original
        assert store.records.read_memories((original.memory_id,)) == (original,)
        for _ in range(128):
            assert store.records.read_memories((original.memory_id,), active_only=True) == (
                original,
            )
        store.records.write_memory(changed)
        if changed.access_count:
            store.records.reinforce_memories((original.memory_id,), accessed_at=_NOW)
        if changed.forgotten_at is not None:
            store.records.set_forgotten((original.memory_id,), forgotten_at=_NOW)
        assert store.records.read_memory(original.memory_id) == changed
        assert store.records.read_memories((original.memory_id,)) == (changed,)
        assert store.records.read_memories((original.memory_id,), active_only=True) == (
            () if changed.forgotten_at is not None else (changed,)
        )
        assert store.records.delete_memory(original.memory_id)
        assert store.records.read_memory(original.memory_id) is None
        assert store.records.read_memories((original.memory_id,)) == ()


def test_text_hydration_does_not_reuse_another_directorys_row(tmp_path: Path) -> None:
    with LocalStore(tmp_path / "first") as first, LocalStore(tmp_path / "second") as second:
        left, right = _memory("shared-id", "first"), _memory("shared-id", "second")
        first.records.write_memory(left)
        second.records.write_memory(right)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(first.records.read_memory, left.memory_id)
            b = pool.submit(second.records.read_memory, right.memory_id)
            assert a.result(timeout=3) == left
            assert b.result(timeout=3) == right


def test_unchanged_text_slates_reuse_reads_but_scopes_and_deletion_still_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LocalStore(tmp_path) as store:
        first, second = _memory("first", "first"), _memory("second", "second")
        store.records.write_memories((first, second))
        for _ in range(128):
            assert store.records.read_memories(
                (first.memory_id, second.memory_id), active_only=True
            ) == (first, second)

        def refuse() -> None:
            raise AssertionError("an unchanged complete slate queried SQLite again")

        with monkeypatch.context() as patch:
            patch.setattr(store._connections, "read_transaction", refuse)
            assert store.records.read_memories(
                (second.memory_id, first.memory_id, second.memory_id), active_only=True
            ) == (second, first, second)
        assert (
            store.records.read_memories(
                (first.memory_id,), known_at=_NOW - timedelta(days=1), active_only=True
            )
            == ()
        )
        assert (
            store.records.read_memories((first.memory_id,), place_id="absent", active_only=True)
            == ()
        )
        assert store.records.delete_memory(first.memory_id)
        assert store.records.read_memories(
            (first.memory_id, second.memory_id), active_only=True
        ) == (second,)


def test_cached_index_candidates_drop_deleted_ids_and_observe_changed_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LocalStore(tmp_path) as store:
        memory_id, embedding_id = _store_one(store, "cached-candidate", "candidate")
        expected = store.index.read_index_candidates((embedding_id,))
        assert len(expected) == 1
        for _ in range(128):
            assert store.index.read_index_candidates((embedding_id,)) == expected

        def refuse() -> None:
            raise AssertionError("unchanged index candidates queried SQLite again")

        with monkeypatch.context() as patch:
            patch.setattr(store._connections, "connection", refuse)
            assert store.index.read_index_candidates((embedding_id, embedding_id)) == expected * 2
        store.records.write_memory(replace(_memory(memory_id, "candidate"), occurred_at=_NOW))
        assert store.index.read_index_candidates((embedding_id,))[0].occurred_at == _NOW
        assert store.records.delete_memory(memory_id)
        assert store.index.read_index_candidates((embedding_id,)) == ()


@pytest.mark.parametrize("pause_before_read", (False, True))
def test_cached_slate_miss_preserves_one_snapshot_and_cannot_publish_across_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_before_read: bool
) -> None:
    entered, release = Event(), Event()
    with LocalStore(tmp_path) as store:
        old = (_memory("first", "old first"), _memory("second", "old second"))
        new = tuple(replace(memory, content=f"new {memory.memory_id}") for memory in old)
        ids = tuple(memory.memory_id for memory in old)
        store.records.write_memories(old)
        # The first row is cached, but the second is not. A miss must still read both rows.
        for _ in range(128):
            assert store.records.read_memories(ids[:1], active_only=True) == old[:1]
        original = store._connections.read_transaction

        @contextmanager
        def paused() -> Iterator[sqlite3.Connection]:
            pause_here = not entered.is_set()
            with original() as connection:
                if pause_here and pause_before_read:
                    entered.set()
                    assert release.wait(10)
                yield connection
                if pause_here and not pause_before_read:
                    entered.set()
                    assert release.wait(10)

        monkeypatch.setattr(store._connections, "read_transaction", paused)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(store.records.read_memories, ids, active_only=True)
            try:
                assert entered.wait(5)
                store.records.write_memories(new)
                for _ in range(128):
                    assert store.records.read_memories(ids, active_only=True) == new
            finally:
                release.set()
            assert pending.result(timeout=5) == (new if pause_before_read else old)
        assert store.records.read_memories(ids, active_only=True) == new


def test_closed_store_refuses_warm_read_caches(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        memory_id, embedding_id = _store_one(store, "closed-cache", "candidate")
        for _ in range(128):
            assert store.records.read_memories((memory_id,), active_only=True)
            assert store.index.read_index_candidates((embedding_id,))
    with pytest.raises(LocalStoreClosedError):
        store.records.read_memories((memory_id,), active_only=True)
    with pytest.raises(LocalStoreClosedError):
        store.index.read_index_candidates((embedding_id,))


@pytest.mark.parametrize("commit_error", (False, True))
def test_caches_cannot_mix_rows_while_sqlite_commit_is_still_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit_error: bool
) -> None:
    pause, committed, release = Event(), Event(), Event()

    class PausedCommit(sqlite3.Connection):
        def commit(self) -> None:
            super().commit()
            if pause.is_set():
                committed.set()
                assert release.wait(10)
                if commit_error:
                    raise sqlite3.OperationalError("injected commit response failure")

    monkeypatch.setattr(sqlite3, "connect", partial(sqlite3.connect, factory=PausedCommit))
    with LocalStore(tmp_path) as store:
        first, first_embedding = _store_one(store, "commit-first", "old first")
        second, second_embedding = _store_one(store, "commit-second", "old second")
        for _ in range(128):
            assert store.records.read_memories((first,), active_only=True)
            assert store.index.read_index_candidates((first_embedding,))
        new = tuple(
            replace(_memory(memory_id, f"new {memory_id}"), occurred_at=_NOW)
            for memory_id in (first, second)
        )
        pause.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(store.records.write_memories, new)
            try:
                assert committed.wait(5)
                assert store._connections.read_generation is None
                # Reading the previously uncached second row must not combine its new value
                # with a cached old first row while SQLite's COMMIT has not returned to Python.
                assert store.records.read_memories((second,), active_only=True) == new[1:]
                assert store.records.read_memories((first, second), active_only=True) == new
                assert store.index.read_index_candidates((second_embedding,))[0].occurred_at == _NOW
                assert all(
                    candidate.occurred_at == _NOW
                    for candidate in store.index.read_index_candidates(
                        (first_embedding, second_embedding)
                    )
                )
            finally:
                release.set()
            if commit_error:
                with pytest.raises(
                    sqlite3.OperationalError, match="injected commit response failure"
                ):
                    pending.result(timeout=5)
            else:
                pending.result(timeout=5)
        assert store._connections.read_generation is not None
        assert store.records.read_memories((first, second), active_only=True) == new


def test_rollback_releases_read_cache_generation_and_keeps_committed_values(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        original = _memory("rollback-cache", "committed")
        store.records.write_memory(original)
        for _ in range(128):
            assert store.records.read_memories((original.memory_id,), active_only=True) == (
                original,
            )
        with (
            pytest.raises(RuntimeError, match="rollback requested"),
            store._connections.transaction() as connection,
        ):
            connection.execute("UPDATE memory_records SET content = 'uncommitted'")
            assert store._connections.read_generation is None
            raise RuntimeError("rollback requested")
        assert store._connections.read_generation is not None
        assert store.records.read_memories((original.memory_id,), active_only=True) == (original,)
