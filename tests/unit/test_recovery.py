"""Startup recovery must not admit incompatible or partly migrated embedding spaces."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest

from mindbridge import Memory, ModelError, StorageError
from mindbridge.infrastructure.local.store import LocalStore
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import IndexQuantization, Modality


class _Embedder:
    embedding_model = "recovery-test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self, space: str = "original", *, fail_on_batch: int | None = None) -> None:
        self.embedding_space = space
        self._legacy_embedding_spaces = frozenset({"original"})
        self.fail_on_batch = fail_on_batch
        self.batches = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.batches += 1
        if self.batches == self.fail_on_batch:
            raise RuntimeError("interrupted migration")
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


def _metadata(data_dir: Path) -> list[tuple[str, str, str]]:
    with closing(sqlite3.connect(data_dir / "state.sqlite3")) as connection:
        return connection.execute(
            "SELECT key, value, updated_at FROM store_metadata ORDER BY key"
        ).fetchall()


def test_rejected_open_does_not_write_missing_or_matching_metadata(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        memory.add("preserved memory")
    with LocalStore(tmp_path) as store:
        store.delete_metadata("embedding.model_id")
        store.set_metadata("transcription.space_id", "incompatible")
    before = _metadata(tmp_path)
    sentinel = tmp_path / "zvec" / "preserved"
    sentinel.write_text("derived index", encoding="utf-8")

    with pytest.raises(StorageError, match=r"transcription\.space_id"):
        Memory(tmp_path, embedder=_Embedder())

    assert _metadata(tmp_path) == before
    assert sentinel.read_text(encoding="utf-8") == "derived index"


def test_matching_open_leaves_metadata_timestamps_unchanged(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_Embedder()):
        pass
    before = _metadata(tmp_path)
    with Memory(tmp_path, embedder=_Embedder()):
        pass
    assert _metadata(tmp_path) == before


def test_metadata_update_rolls_back_marker_changes_and_deletion_together(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        store.set_metadata("embedding.space_id", "original")
        store.set_metadata("index.recipe", "original")
        store.set_metadata("embedding.migration_target", "pending")
        before = _metadata(tmp_path)
        with closing(sqlite3.connect(store.database_path)) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_index_marker BEFORE UPDATE ON store_metadata
                WHEN NEW.key = 'index.recipe'
                BEGIN SELECT RAISE(ABORT, 'failed marker commit'); END
                """
            )
            connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="failed marker commit"):
            store.update_metadata(
                {
                    "embedding.space_id": "upgraded",
                    "embedding.migration_target": None,
                    "index.recipe": "upgraded",
                }
            )
        assert _metadata(tmp_path) == before


@pytest.mark.parametrize("different_target", ("space", "index_recipe"))
def test_partial_embedding_migration_rejects_other_targets_and_resumes(
    tmp_path: Path, different_target: str
) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        memory.add_many(tuple(f"preserved memory {index}" for index in range(40)))

    with pytest.raises(ModelError, match="embed memory input"):
        Memory(tmp_path, embedder=_Embedder("upgraded", fail_on_batch=2))

    # The failure follows a successful page commit, rather than the first model call.
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        spaces = dict(
            connection.execute("SELECT space_id, COUNT(*) FROM embeddings GROUP BY space_id")
        )
    assert spaces == {"original": 8, "upgraded": 32}

    incompatible = _Embedder("original" if different_target == "space" else "upgraded")
    quantization = IndexQuantization.NONE if different_target == "space" else IndexQuantization.INT8
    before = _metadata(tmp_path)
    with pytest.raises(StorageError, match="unfinished embedding migration"):
        Memory(tmp_path, embedder=incompatible, index_quantization=quantization)
    assert incompatible.batches == 0
    assert _metadata(tmp_path) == before

    resumed = _Embedder("upgraded")
    with Memory(tmp_path, embedder=resumed) as memory:
        assert len(memory.list(limit=100).items) == 40
        assert memory.search("preserved")
    with LocalStore(tmp_path) as store:
        assert store.get_metadata("embedding.space_id") == "upgraded"
        assert store.get_metadata("embedding.migration_target") is None
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        assert connection.execute("SELECT DISTINCT space_id FROM embeddings").fetchall() == [
            ("upgraded",)
        ]
