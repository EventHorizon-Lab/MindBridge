"""Startup validates current store identities before changing durable or derived state."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest

from mindbridge import Memory, StorageError
from mindbridge.infrastructure.local.store import LocalStore
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import Modality


class _Embedder:
    embedding_model = "recovery-test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self, space: str = "original") -> None:
        self.embedding_space = space
        self.batches = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.batches += 1
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
        store.update_metadata(
            {"embedding.model_id": None, "transcription.space_id": "incompatible"}
        )
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
        store.update_metadata(
            {
                "embedding.space_id": "original",
                "index.recipe": "original",
                "retired.marker": "pending",
            }
        )
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
                    "retired.marker": None,
                    "index.recipe": "upgraded",
                }
            )
        assert _metadata(tmp_path) == before


@pytest.mark.parametrize(
    "recipe",
    (
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:"
        "context-keys-v8:quantization-none",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:"
        "context-keys-v9:quantization-none",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-stemmed-plus-bigram:grouped-range:"
        "context-keys-v10:quantization-none",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-bigram-fused:grouped-range:"
        "context-keys-v12:quantization-none",
    ),
)
def test_old_index_recipes_are_rejected_without_rewriting_the_store(
    tmp_path: Path, recipe: str
) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        record = memory.add("preserved memory")
    with LocalStore(tmp_path) as store:
        store.update_metadata({"index.recipe": recipe})
        documents = store.index.read_memory_index_documents((record.id,))
    before = _metadata(tmp_path)
    sentinel = tmp_path / "zvec" / "preserved"
    sentinel.write_text("derived index", encoding="utf-8")
    embedder = _Embedder()

    with pytest.raises(StorageError, match=r"index\.recipe"):
        Memory(tmp_path, embedder=embedder)

    assert embedder.batches == 0
    assert _metadata(tmp_path) == before
    assert sentinel.read_text(encoding="utf-8") == "derived index"
    with LocalStore(tmp_path) as store:
        assert store.index.read_memory_index_documents((record.id,)) == documents


@pytest.mark.parametrize(
    "key",
    (
        "embedding.model_id",
        "embedding.space_id",
        "embedding.dimension",
        "transcription.space_id",
        "index.recipe",
        "evidence.projection_recipe",
    ),
)
def test_nonempty_store_requires_every_current_metadata_marker(tmp_path: Path, key: str) -> None:
    with Memory(tmp_path, embedder=_Embedder()) as memory:
        memory.add("preserved memory")
    with LocalStore(tmp_path) as store:
        store.update_metadata({key: None})
    before = _metadata(tmp_path)
    embedder = _Embedder()

    with pytest.raises(StorageError, match=r"metadata.*missing"):
        Memory(tmp_path, embedder=embedder)

    assert embedder.batches == 0
    assert _metadata(tmp_path) == before
