"""Public-SDK checks for temporary image retrieval provenance."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Blob,
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    SearchHit,
    StorageError,
)


class VisualBackend:
    embedding_model = "visual-test"
    embedding_space = "visual-test:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    generation_model = "visual-reader-test"
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self) -> None:
        self.embedding_calls = 0
        self.questions: list[ModelInput] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.embedding_calls += 1
        return tuple(
            (1.0, 0.0)
            if value.assets and (value.assets[0].name or "").startswith(("near", "query"))
            else (0.8, 0.6)
            for value in inputs
        )

    def answer(
        self,
        question: ModelInput | str,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        assert isinstance(question, ModelInput)
        self.questions.append(question)
        return AnswerResult(answer="Read the supplied sources.", hits=tuple(hits))

    def close(self) -> None:
        return None


def test_ask_preserves_image_only_provenance_without_an_extra_model_call(tmp_path: Path) -> None:
    backend = VisualBackend()
    with Memory(
        tmp_path,
        embedder=backend,
        answerer=backend,
        recall_planning=False,
        evidence_budget_chars=None,
    ) as memory:
        near = [
            memory.add(
                (
                    f"Original taught name {i}",
                    Blob(f"near{i}".encode(), "image/png", f"near{i}.png"),
                )
            )
            for i in range(3)
        ]
        far = memory.add(("Another taught object", Blob(b"far", "image/png", "far.png")))
        before = backend.embedding_calls
        memory.ask(("Which taught object?", Blob(b"query", "image/png", "query.png")), limit=4)
        matches = backend.questions[-1].image_retrieval_matches
        assert backend.embedding_calls == before + 1
        assert {match.memory_id for match in matches} == {record.id for record in near}
        assert far.id not in {match.memory_id for match in matches}
        assert [match.rank for match in matches] == [1, 2, 3]
        assert all(match.cosine_similarity == pytest.approx(1.0) for match in matches)
        assert all(match.query_asset_id == backend.questions[-1].assets[0].id for match in matches)
        assert all(
            "image_retrieval_matches" not in memory.get(record.id).metadata for record in near
        )
        result = memory.ask(
            ("Which taught object?", Blob(b"query", "image/png", "query.png")), limit=2
        )
        assert len(backend.questions[-1].image_retrieval_matches) == 2
        assert all(
            match.memory_id in {hit.id for hit in result.hits}
            for match in backend.questions[-1].image_retrieval_matches
        )


def test_stale_index_ids_and_missing_index_do_not_create_matches_or_reembed(tmp_path: Path) -> None:
    backend = VisualBackend()
    with Memory(tmp_path, embedder=backend, answerer=backend, recall_planning=False) as memory:
        live = memory.add(("Current source", Blob(b"near-live", "image/png", "near-live.png")))
        stale = memory.add(("Removed source", Blob(b"near-stale", "image/png", "near-stale.png")))
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("DELETE FROM memory_records WHERE memory_id=?", (stale.id,))
        memory.ask(Blob(b"query", "image/png", "query.png"), limit=3)
        assert {match.memory_id for match in backend.questions[-1].image_retrieval_matches} == {
            live.id
        }
    shutil.rmtree(tmp_path / "zvec")
    before = backend.embedding_calls
    with Memory(tmp_path, embedder=backend, answerer=backend, recall_planning=False) as memory:
        assert backend.embedding_calls == before
        memory.ask(Blob(b"query", "image/png", "query.png"), limit=3)
        assert backend.embedding_calls == before + 1
        assert {match.memory_id for match in backend.questions[-1].image_retrieval_matches} == {
            live.id
        }


def test_incomplete_stored_key_layout_does_not_guess_an_asset_mapping(tmp_path: Path) -> None:
    backend = VisualBackend()
    with Memory(tmp_path, embedder=backend, answerer=backend, recall_planning=False) as memory:
        source = memory.add(("Source label", Blob(b"near", "image/png", "near.png")))
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as db, db:
            db.execute("DELETE FROM embeddings WHERE memory_id=? AND object_part=0", (source.id,))
        memory.ask(Blob(b"query", "image/png", "query.png"), limit=3)
        assert backend.questions[-1].image_retrieval_matches == ()


def test_search_and_text_ask_do_not_read_visual_document_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = VisualBackend()
    with Memory(tmp_path, embedder=backend, answerer=backend, recall_planning=False) as memory:
        memory.add(("Source", Blob(b"near", "image/png", "near.png")))

        def forbidden(_ids: Sequence[str]) -> None:
            raise AssertionError("The ordinary ranking path read visual document vectors")

        monkeypatch.setattr(memory._store.index, "read_memory_index_documents", forbidden)
        assert memory.search(Blob(b"query", "image/png", "query.png"), limit=3)
        memory.ask("A text question", limit=3)
        assert backend.questions[-1].image_retrieval_matches == ()


def test_image_matches_keep_physical_directory_isolation(tmp_path: Path) -> None:
    first, second = VisualBackend(), VisualBackend()
    with (
        Memory(tmp_path / "first", embedder=first, answerer=first, recall_planning=False) as a,
        Memory(tmp_path / "second", embedder=second, answerer=second, recall_planning=False) as b,
    ):
        left = a.add(("First source", Blob(b"near-left", "image/png", "near-left.png")))
        right = b.add(("Second source", Blob(b"near-right", "image/png", "near-right.png")))
        with pytest.raises(StorageError):
            Memory(tmp_path / "first", embedder=VisualBackend())
        a.ask(Blob(b"query", "image/png", "query.png"), limit=3)
        b.ask(Blob(b"query", "image/png", "query.png"), limit=3)
        assert {match.memory_id for match in first.questions[-1].image_retrieval_matches} == {
            left.id
        }
        assert {match.memory_id for match in second.questions[-1].image_retrieval_matches} == {
            right.id
        }
