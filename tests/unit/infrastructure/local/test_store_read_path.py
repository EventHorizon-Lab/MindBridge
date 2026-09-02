"""What the search read path is allowed to cost, and what it must still guarantee.

Hydrating a search candidate used to rebuild and re-validate its FP32 vector even though no
retrieval code reads it: `memory.py` touches only `embedding_id`, `memory_id` and `object_part` on
a hydrated document, and the vector is consumed solely by `ZvecIndex.upsert`. The two O(dimension)
Python loops in `StoredEmbedding.__post_init__` therefore ran once per candidate per search and
paid for nobody. They now run where a vector enters SQLite instead.

These tests pin both halves of that trade: hydration does not re-check stored vector content, and
the write boundary still refuses a vector it would be wrong to store. They also pin the two
storage invariants the change is closest to breaking -- stale index IDs are dropped by SQLite
hydration, and a missing index rebuilds from stored vectors without asking the embedder again.
"""

from __future__ import annotations

import math
import shutil
import sqlite3
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge import Memory
from mindbridge.exceptions import ModelError
from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import Modality

_NOW = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)


def _memory(memory_id: str, content: str) -> StoredMemory:
    return StoredMemory(
        memory_id=memory_id,
        content=content,
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _embedding(
    embedding_id: str,
    memory_id: str,
    *,
    values: tuple[float, ...] = (0.6, 0.8),
    normalized: bool = True,
) -> StoredEmbedding:
    return StoredEmbedding(
        embedding_id=embedding_id,
        memory_id=memory_id,
        values=values,
        model_id="read-path-probe",
        space_id="read-path-probe:2",
        task="retrieval.document",
        created_at=_NOW,
        normalized=normalized,
    )


def _store_one(store: LocalStore, suffix: str, content: str) -> tuple[str, str]:
    memory_id = f"memory-{suffix}"
    embedding_id = f"embedding-{suffix}"
    store.write_memory(_memory(memory_id, content))
    store.write_embedding(_embedding(embedding_id, memory_id))
    return memory_id, embedding_id


def test_hydration_does_not_re_validate_a_stored_vector(tmp_path: Path) -> None:
    """A candidate hydrates on identity and text alone, whatever the stored vector holds.

    The vector is overwritten out of band with a non-finite, non-unit payload of the same byte
    length, which is the one corruption the `length(vector) = dimension * 4` CHECK cannot see. A
    search must not spend an O(dimension) loop discovering it, because a search never reads it.
    """
    directory = tmp_path / "data"
    directory.mkdir()
    with LocalStore(directory) as store:
        _memory_id, embedding_id = _store_one(store, "corrupt", "the kitchen at dusk")
        database_path = store.database_path

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE embeddings SET vector = ? WHERE embedding_id = ?",
            (struct.pack("<2f", float("nan"), 9.0), embedding_id),
        )
        connection.commit()
    connection.close()

    with LocalStore(directory) as store:
        documents = store.read_index_documents((embedding_id,))
        assert len(documents) == 1
        assert documents[0].content == "the kitchen at dusk"
        assert documents[0].embedding.embedding_id == embedding_id


def test_writing_an_embedding_still_refuses_a_vector_it_must_not_store(tmp_path: Path) -> None:
    """The check hydration no longer repeats still guards the one place a vector arrives."""
    directory = tmp_path / "data"
    directory.mkdir()
    with LocalStore(directory) as store:
        store.write_memory(_memory("memory-guard", "the garden at noon"))
        non_finite = _embedding(
            "embedding-non-finite",
            "memory-guard",
            values=(float("nan"), 0.0),
            normalized=False,
        )
        with pytest.raises(ValueError, match="finite"):
            store.write_embedding(non_finite)

        empty_dimension = _embedding("embedding-empty", "memory-guard", values=(), normalized=False)
        with pytest.raises(ValueError, match="non-empty"):
            store.write_embedding(empty_dimension)

        not_unit_length = _embedding(
            "embedding-not-unit",
            "memory-guard",
            values=(0.6, 0.6),
            normalized=True,
        )
        with pytest.raises(ValueError, match="unit length"):
            store.write_embedding(not_unit_length)

        # Nothing partial was left behind, so the refusal is a refusal and not a rollback bug.
        assert (
            store.read_index_documents(
                ("embedding-non-finite", "embedding-empty", "embedding-not-unit")
            )
            == ()
        )


def test_hydration_drops_index_ids_sqlite_no_longer_knows(tmp_path: Path) -> None:
    """Zvec lags SQLite by design, so hydration is what removes an ID SQLite has dropped."""
    directory = tmp_path / "data"
    directory.mkdir()
    with LocalStore(directory) as store:
        live_memory, live = _store_one(store, "live", "the kitchen at dusk")
        stale_memory, stale = _store_one(store, "stale", "the garden at noon")

        assert [
            document.embedding.embedding_id
            for document in store.read_index_documents((live, stale))
        ] == [live, stale]

        assert store.delete_memory(stale_memory) is True

        # Order is still the caller's order, and the vanished ID is simply absent rather than
        # raising or shifting a neighbour into its place.
        hydrated = store.read_index_documents((stale, live, "embedding-never-existed"))
        assert [document.embedding.embedding_id for document in hydrated] == [live]
        assert store.read_index_document(stale) is None
        assert store.read_memory_index_documents((stale_memory, live_memory)) == (
            store.read_index_document(live),
        )


class _CountingEmbedder:
    """A deterministic embedder that can refuse to embed documents but still answer queries."""

    embedding_model = "read-path-probe"
    embedding_space = "read-path-probe:2:test"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def __init__(self) -> None:
        self.document_calls = 0
        self.refuse_documents = False

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        if task is EmbedTask.DOCUMENT:
            self.document_calls += 1
            if self.refuse_documents:
                raise ModelError(
                    "the embedder was asked to re-embed stored content",
                    reason="server_error",
                )
        return tuple(_unit_vector(value.text or "") for value in batch)

    def close(self) -> None:
        return None


def _unit_vector(text: str) -> tuple[float, ...]:
    angle = float(sum(text.encode("utf-8")) % 360) * math.pi / 180.0
    return (math.cos(angle), math.sin(angle))


def test_a_missing_index_rebuilds_from_stored_vectors_without_re_embedding(
    tmp_path: Path,
) -> None:
    """Zvec is derived: deleting it must cost a hydration, never a model call.

    The embedder raises on any document embed once ingest is over, so the reopen below can only
    succeed by reading the FP32 vectors SQLite kept -- and the search that follows proves the
    rebuilt index holds real vectors rather than an empty collection.
    """
    embedder = _CountingEmbedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        kitchen = memory.add("the kitchen at dusk")
        memory.add("the garden at noon")
    ingest_calls = embedder.document_calls
    assert ingest_calls > 0

    shutil.rmtree(tmp_path / "zvec")
    embedder.refuse_documents = True

    with Memory(tmp_path, embedder=embedder) as memory:
        results = memory.search("the kitchen at dusk", limit=2)
        assert kitchen.id in {result.id for result in results}

    assert embedder.document_calls == ingest_calls


def test_reindex_rebuilds_from_stored_vectors_without_re_embedding(tmp_path: Path) -> None:
    """The explicit rebuild entry point reads the same stored vectors as the implicit one."""
    embedder = _CountingEmbedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        kitchen = memory.add("the kitchen at dusk")
        ingest_calls = embedder.document_calls
        embedder.refuse_documents = True

        assert memory.reindex() == 1
        results = memory.search("the kitchen at dusk", limit=2)
        assert kitchen.id in {result.id for result in results}
        assert embedder.document_calls == ingest_calls
