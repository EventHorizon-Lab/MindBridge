from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random

import pytest

from mindbridge import Memory, RetrievalScope
from mindbridge.exceptions import StorageError
from mindbridge.infrastructure.local.scoring import max_cosine_scores
from mindbridge.infrastructure.local.store import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import IndexHit, ZvecIndex
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import MemoryType, Modality, RetrievalMode


class _DirectionalEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "score-completion-test"
    embedding_space = "score-completion-test:2"
    embedding_dimension = 2

    def __init__(self, *, all_documents_align: bool = False) -> None:
        self.all_documents_align = all_documents_align
        self.calls: list[tuple[EmbedTask, tuple[str, ...]]] = []

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        self.calls.append((task, tuple(value.text for value in batch)))
        if task is EmbedTask.QUERY:
            return tuple((1.0, 0.0) for _value in batch)
        return tuple(
            (1.0, 0.0) if self.all_documents_align or value.text == "aligned part" else (0.0, 1.0)
            for value in batch
        )

    def close(self) -> None:
        return None


class _CloseTrackingIterator:
    def __init__(self, values: Iterator[tuple[str, tuple[float, ...]]]) -> None:
        self._values = values
        self.closed = False

    def __iter__(self) -> _CloseTrackingIterator:
        return self

    def __next__(self) -> tuple[str, tuple[float, ...]]:
        return next(self._values)

    def close(self) -> None:
        self.closed = True
        close = getattr(self._values, "close", None)
        if close is not None:
            close()


def _lexical_hit(index_id: str, relevance: float = 1.0) -> IndexHit:
    return IndexHit(
        id=index_id,
        relevance=relevance,
        confidence=0.0,
        lexical_match=True,
    )


def test_completed_scores_match_the_real_zvec_cosine_scale(tmp_path: Path) -> None:
    dimension = 8
    generator = Random(20260908)
    query_vectors = [
        tuple(generator.uniform(-7.0, 7.0) for _index in range(dimension)),
        (11.0, -3.0, 5.0, 2.0, -9.0, 4.0, 1.0, -6.0),
    ]
    raw_documents = [
        tuple(generator.uniform(-13.0, 13.0) for _index in range(dimension))
        for _document_index in range(6)
    ]
    raw_documents.extend((query_vectors[1], tuple(-value for value in query_vectors[1])))
    # SQLite persists FP32, so use the values the completion path would unpack rather than the
    # Python inputs that existed before the durable write.
    documents = [
        struct.unpack(f"<{dimension}f", struct.pack(f"<{dimension}f", *vector))
        for vector in raw_documents
    ]
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)
    indexed = tuple(
        IndexDocument(
            embedding=StoredEmbedding(
                embedding_id=f"document-{index}",
                memory_id=f"memory-{index}",
                values=tuple(vector),
                model_id="scale-test",
                space_id="scale-test:8",
                task=EmbedTask.DOCUMENT.value,
                created_at=now,
            ),
            content=f"document {index}",
            metadata_json="{}",
        )
        for index, vector in enumerate(documents)
    )

    maximum_error = 0.0
    with ZvecIndex(tmp_path / "scale-index", dimension=dimension) as index:
        index.upsert(indexed)
        index.flush()
        for query in query_vectors:
            native: dict[str, tuple[float, float]] = {}
            for hit in index.search(
                query,
                limit=len(indexed),
                space_id="scale-test:8",
                task=EmbedTask.DOCUMENT.value,
                exact=True,
            ):
                assert hit.confidence is not None
                native[hit.id] = (hit.relevance, hit.confidence)
            completed = max_cosine_scores(
                (query,),
                (
                    (document.embedding.embedding_id, document.embedding.values)
                    for document in indexed
                ),
            )
            assert native.keys() == completed.keys()
            for document_id in native:
                error = max(
                    abs(native[document_id][component] - completed[document_id][component])
                    for component in (0, 1)
                )
                maximum_error = max(maximum_error, error)
                assert native[document_id] == pytest.approx(
                    completed[document_id], rel=2e-6, abs=2e-6
                )

    assert maximum_error < 2e-6


def test_completion_uses_every_parent_part_without_an_additional_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = _DirectionalEmbedder()
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0, ambiguity_margin=0) as memory:
        record = memory.add(("rare aggregate preface", "aligned part"))
        documents = memory._store.read_memory_index_documents((record.id,))
        assert len(documents) >= 2
        aggregate_id = documents[0].embedding.embedding_id
        assert documents[0].embedding.values == pytest.approx((0.0, 1.0))
        assert any(document.embedding.values == pytest.approx((1.0, 0.0)) for document in documents)

        monkeypatch.setattr(memory._index, "search", lambda *_args, **_kwargs: ())
        monkeypatch.setattr(
            memory._index,
            "lexical_search",
            lambda *_args, **_kwargs: (_lexical_hit(aggregate_id),),
        )
        calls_before = len(embedder.calls)

        result = memory.search_with_trace("rare")

    assert [hit.id for hit in result.hits] == [record.id]
    candidate = next(item for item in result.trace.candidates if item.memory_id == record.id)
    assert candidate.dense_relevance == pytest.approx(1.0)
    assert candidate.dense_confidence == pytest.approx(1.0)
    # Candidate provenance remains the lexical index document. The completed score can come from
    # another persisted part, so `index_ids` must not be interpreted as dense-score provenance.
    assert candidate.index_ids == (aggregate_id,)
    assert embedder.calls[calls_before:] == [(EmbedTask.QUERY, ("rare",))]


def test_a_valid_zero_dense_hit_is_not_treated_as_a_missing_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = _DirectionalEmbedder(all_documents_align=True)
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0, ambiguity_margin=0) as memory:
        measured = memory.add("rare measured")
        missing = memory.add("rare missing")
        monkeypatch.setattr(
            memory._index,
            "search",
            lambda *_args, **_kwargs: (IndexHit(id=measured.id, relevance=0.0, confidence=0.5),),
        )
        monkeypatch.setattr(
            memory._index,
            "lexical_search",
            lambda *_args, **_kwargs: (
                _lexical_hit(measured.id),
                _lexical_hit(missing.id, 0.9),
            ),
        )
        requested: list[tuple[str, ...]] = []
        original = memory._store.iter_memory_embedding_vectors

        def read_vectors(
            memory_ids: Sequence[str],
            *,
            space_id: str,
            task: str,
        ) -> Iterator[tuple[str, tuple[float, ...]]]:
            requested.append(tuple(memory_ids))
            yield from original(memory_ids, space_id=space_id, task=task)

        monkeypatch.setattr(memory._store, "iter_memory_embedding_vectors", read_vectors)

        result = memory.search_with_trace("rare")

    by_id = {
        candidate.memory_id: candidate
        for candidate in result.trace.candidates
        if candidate.memory_id is not None
    }
    assert requested == [(missing.id,)]
    assert by_id[measured.id].dense_relevance == pytest.approx(0.0)
    assert by_id[measured.id].dense_confidence == pytest.approx(0.5)
    assert by_id[missing.id].dense_relevance == pytest.approx(1.0)
    assert by_id[missing.id].dense_confidence == pytest.approx(1.0)


def test_completion_reads_only_records_that_survive_type_time_and_known_at_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = _DirectionalEmbedder(all_documents_align=True)
    event = datetime(2026, 1, 2, tzinfo=timezone.utc)
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0, ambiguity_margin=0) as memory:
        included = memory.add("rare included", occurred_at=event)
        wrong_type = memory.add(
            "rare episodic",
            occurred_at=event,
            memory_type=MemoryType.EPISODIC,
        )
        outside_time = memory.add("rare outside", occurred_at=event + timedelta(days=10))
        ids = (included.id, wrong_type.id, outside_time.id)
        monkeypatch.setattr(memory._index, "search", lambda *_args, **_kwargs: ())
        monkeypatch.setattr(
            memory._index,
            "lexical_search",
            lambda *_args, **_kwargs: tuple(_lexical_hit(memory_id) for memory_id in ids),
        )
        requested: list[tuple[str, ...]] = []
        original = memory._store.iter_memory_embedding_vectors

        def read_vectors(
            memory_ids: Sequence[str],
            *,
            space_id: str,
            task: str,
        ) -> Iterator[tuple[str, tuple[float, ...]]]:
            requested.append(tuple(memory_ids))
            yield from original(memory_ids, space_id=space_id, task=task)

        monkeypatch.setattr(memory._store, "iter_memory_embedding_vectors", read_vectors)

        hits = memory.search(
            "rare",
            memory_type=MemoryType.SEMANTIC,
            occurred_from=event - timedelta(hours=1),
            occurred_until=event + timedelta(hours=1),
        )
        assert [hit.id for hit in hits] == [included.id]
        assert requested == [(included.id,)]

        requested.clear()
        assert (
            memory.search(
                "rare",
                scope=RetrievalScope(known_at=datetime(2000, 1, 1, tzinfo=timezone.utc)),
            )
            == ()
        )
        assert requested == []


def test_corrupt_persisted_vector_uses_the_storage_error_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = _DirectionalEmbedder(all_documents_align=True)
    with Memory(tmp_path, embedder=embedder, minimum_relevance=0, ambiguity_margin=0) as memory:
        record = memory.add("rare corrupt")
        monkeypatch.setattr(memory._index, "search", lambda *_args, **_kwargs: ())
        monkeypatch.setattr(
            memory._index,
            "lexical_search",
            lambda *_args, **_kwargs: (_lexical_hit(record.id),),
        )
        # Flush and acknowledge the add's batched outbox row first, so the trigger row the
        # corruption below enqueues is the only one left to drop.
        memory.optimize()
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            connection.execute(
                "UPDATE embeddings SET vector = ? WHERE embedding_id = ?",
                (struct.pack("<2f", float("nan"), 0.0), record.id),
            )
            # Keep the already-built derived index unchanged so this exercises authoritative
            # score completion rather than the earlier outbox-replay corruption boundary.
            connection.execute("DELETE FROM search_index_queue")

        original = memory._store.iter_memory_embedding_vectors
        iterator_spies: list[_CloseTrackingIterator] = []

        def tracked_vectors(
            memory_ids: Sequence[str],
            *,
            space_id: str,
            task: str,
        ) -> _CloseTrackingIterator:
            spy = _CloseTrackingIterator(original(memory_ids, space_id=space_id, task=task))
            iterator_spies.append(spy)
            return spy

        monkeypatch.setattr(memory._store, "iter_memory_embedding_vectors", tracked_vectors)

        with pytest.raises(StorageError) as raised:
            memory.search("rare")

    assert raised.value.reason == "io_failed"
    assert len(iterator_spies) == 1
    assert iterator_spies[0].closed is True


def test_dense_mode_keeps_the_index_score_and_skips_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    embedder = _DirectionalEmbedder(all_documents_align=True)
    with Memory(
        tmp_path,
        embedder=embedder,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        record = memory.add("rare dense")
        monkeypatch.setattr(
            memory._index,
            "search",
            lambda *_args, **_kwargs: (IndexHit(id=record.id, relevance=0.25, confidence=0.625),),
        )
        monkeypatch.setattr(
            memory._store,
            "iter_memory_embedding_vectors",
            lambda *_args, **_kwargs: pytest.fail("dense mode must not complete scores"),
        )

        result = memory.search_with_trace("rare")

    candidate = next(item for item in result.trace.candidates if item.memory_id == record.id)
    assert candidate.dense_relevance == pytest.approx(0.25)
    assert candidate.dense_confidence == pytest.approx(0.625)
