from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import ClassVar

import pytest

import mindbridge.memory as memory_module
from mindbridge.exceptions import (
    IndexUnavailableError,
    MemoryNotFoundError,
    ModelError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)
from mindbridge.infrastructure.local.assets import AssetStore
from mindbridge.infrastructure.local.store import (
    IndexDocument,
    LocalStore,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
)
from mindbridge.infrastructure.local.zvec_index import IndexHit
from mindbridge.memory import AsyncMemory, Memory
from mindbridge.models.base import (
    EmbedTask,
    ModelInput,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)
from mindbridge.types import AnswerResult, AssetRef, Blob, MemoryType, Modality, SearchHit

ALL_INPUT_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


@dataclass(frozen=True, slots=True)
class _Capabilities:
    embedding: frozenset[Modality]
    generation: frozenset[Modality]
    transcription: frozenset[Modality]


class _FakeModels:
    def __init__(
        self,
        *,
        model: str = "fake-embedding",
        transcription_space: str = "fake-asr:test",
        capabilities: _Capabilities | None = None,
    ) -> None:
        self.embed_batches: list[tuple[str, ...]] = []
        self.embed_inputs: list[tuple[ModelInput, ...]] = []
        self.embed_tasks: list[EmbedTask] = []
        self.answer_calls: list[tuple[ModelInput, tuple[SearchHit, ...]]] = []
        self.transcribe_calls: list[tuple[AssetRef, ...]] = []
        self.embedding_model = model
        self.embedding_space = f"{model}:2:test"
        self.transcription_model = "fake-transcription"
        self.transcription_space = transcription_space
        self.embedding_dimension = 2
        selected = capabilities or _Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=ALL_INPUT_MODALITIES,
            transcription=frozenset({Modality.AUDIO}),
        )
        self.embedding_capabilities = selected.embedding
        self.generation_capabilities = selected.generation
        self.transcription_capabilities = selected.transcription
        self.closed = False
        self.close_calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        self.embed_inputs.append(batch)
        self.embed_batches.append(tuple(value.text for value in batch))
        self.embed_tasks.append(task)
        return tuple(
            (1.0, 0.0) if "red" in value.text.casefold() else (0.0, 1.0) for value in batch
        )

    def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
        grounded = tuple(hits)
        self.answer_calls.append((question, grounded))
        answer = f"Grounded in: {grounded[0].content}" if grounded else "I do not know."
        return AnswerResult(answer=answer, hits=grounded)

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        batch = tuple(assets)
        self.transcribe_calls.append(batch)
        return tuple("spoken red wrench" for _asset in batch)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _FakeEmbedder:
    def __init__(
        self,
        *,
        capabilities: frozenset[Modality] = ALL_INPUT_MODALITIES,
        model_id: str = "fake-separate-embedding",
    ) -> None:
        self.embedding_capabilities = capabilities
        self.embedding_model = model_id
        self.embedding_space = f"{model_id}:2:test"
        self.embedding_dimension = 2
        self.embed_inputs: list[tuple[ModelInput, ...]] = []
        self.embed_tasks: list[EmbedTask] = []
        self.closed = False
        self.close_calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        batch = tuple(inputs)
        self.embed_inputs.append(batch)
        self.embed_tasks.append(task)
        return tuple(
            (1.0, 0.0) if "red" in value.text.casefold() else (0.0, 1.0) for value in batch
        )

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _FakeSpeech:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "fake-funasr"
    transcription_space = "fake-funasr:speech:test"

    def __init__(self) -> None:
        self.calls: list[tuple[AssetRef, ...]] = []
        self.closed = False

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        batch = tuple(assets)
        self.calls.append(batch)
        return tuple(
            SpeechAnalysis(
                turns=(SpeechTurn(0, 900, "spoken red wrench", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            )
            for _asset in batch
        )

    def close(self) -> None:
        self.closed = True


class _FakeIndex:
    documents_by_path: ClassVar[dict[str, dict[str, IndexDocument]]] = {}
    instances: ClassVar[list[_FakeIndex]] = []

    def __init__(self, path: str | Path, dimension: int) -> None:
        self.path = Path(path).resolve()
        self.dimension = dimension
        key = str(self.path)
        if not self.path.exists():
            self.documents_by_path[key] = {}
            self.path.mkdir(parents=True)
        self.documents = self.documents_by_path.setdefault(key, {})
        self.fail_next_flush = False
        self.hits_override: tuple[IndexHit, ...] | None = None
        self.upsert_calls: list[tuple[str, ...]] = []
        self.delete_calls: list[tuple[str, ...]] = []
        self.dense_search_calls = 0
        self.hybrid_search_calls = 0
        self.optimize_calls = 0
        self.rebuild_calls = 0
        self.rebuild_batch_sizes: list[int] = []
        self.fail_next_rebuild = False
        self.closed = False
        self.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.documents_by_path.clear()
        cls.instances.clear()

    def upsert(self, documents: Sequence[IndexDocument]) -> None:
        self.upsert_calls.append(tuple(document.embedding.embedding_id for document in documents))
        for document in documents:
            self.documents[document.embedding.embedding_id] = document

    def delete(self, ids: Sequence[str]) -> None:
        self.delete_calls.append(tuple(ids))
        for document_id in ids:
            self.documents.pop(document_id, None)

    def search(
        self,
        values: Sequence[float],
        *,
        limit: int = 10,
        space_id: str | None = None,
        task: str | None = None,
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]:
        del values, ef, exact
        self.dense_search_calls += 1
        if self.hits_override is not None:
            return self.hits_override[:limit]
        return tuple(
            IndexHit(id=document_id, relevance=0.5)
            for document_id, document in self.documents.items()
            if (space_id is None or document.embedding.space_id == space_id)
            and (task is None or document.embedding.task == task)
            and self._matches_time_and_type(document, memory_type, occurred_from, occurred_until)
        )[:limit]

    def hybrid_search(
        self,
        text: str,
        values: Sequence[float],
        *,
        limit: int = 10,
        candidate_limit: int | None = None,
        space_id: str | None = None,
        task: str | None = None,
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]:
        del values, candidate_limit, ef, exact
        self.hybrid_search_calls += 1
        if self.hits_override is not None:
            return self.hits_override[:limit]
        query_terms = set(text.casefold().split())
        hits = []
        for document_id, document in self.documents.items():
            embedding = document.embedding
            if space_id is not None and embedding.space_id != space_id:
                continue
            if task is not None and embedding.task != task:
                continue
            if not self._matches_time_and_type(
                document, memory_type, occurred_from, occurred_until
            ):
                continue
            content_terms = set(document.content.casefold().split())
            relevance = 1.0 if query_terms & content_terms else 0.5
            hits.append(IndexHit(id=document_id, relevance=relevance))
        return tuple(sorted(hits, key=lambda hit: (-hit.relevance, hit.id))[:limit])

    @staticmethod
    def _matches_time_and_type(
        document: IndexDocument | None,
        memory_type: str | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
    ) -> bool:
        if memory_type is None and occurred_from is None and occurred_until is None:
            return True
        if document is None or (memory_type is not None and document.memory_type != memory_type):
            return False
        if occurred_from is None and occurred_until is None:
            return True
        occurred_at = document.occurred_at
        return (
            occurred_at is not None
            and (occurred_from is None or occurred_at >= occurred_from)
            and (occurred_until is None or occurred_at < occurred_until)
        )

    def flush(self) -> None:
        if self.fail_next_flush:
            self.fail_next_flush = False
            raise RuntimeError("simulated flush failure")

    def optimize(self, *, concurrency: int = 0) -> None:
        del concurrency
        self.optimize_calls += 1

    def rebuild(
        self,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int = 1_024,
        optimize_concurrency: int = 0,
    ) -> int:
        del optimize_concurrency
        self.rebuild_calls += 1
        self.rebuild_batch_sizes.append(batch_size)
        self.documents.clear()
        count = 0
        for document in documents:
            self.documents[document.embedding.embedding_id] = document
            count += 1
            if self.fail_next_rebuild:
                self.fail_next_rebuild = False
                raise RuntimeError("simulated interrupted rebuild")
        self.flush()
        return count

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeIndex.reset()
    monkeypatch.setattr(memory_module, "ZvecIndex", _FakeIndex)


def _memory(
    data_dir: Path,
    models: _FakeModels | None = None,
    *,
    embedder: _FakeEmbedder | None = None,
    transcriber: _FakeSpeech | None = None,
    decay_half_life_days: float | None = None,
) -> Memory:
    models = models or _FakeModels()
    return Memory(
        data_dir,
        embedder=models if embedder is None else embedder,
        answerer=models,
        transcriber=models if transcriber is None else transcriber,
        decay_half_life_days=decay_half_life_days,
    )


def test_crud_search_ask_and_stable_duplicate(tmp_path: Path) -> None:
    models = _FakeModels()
    occurred_at = datetime(2026, 8, 27, 9, 30, tzinfo=timezone(timedelta(hours=8)))

    with _memory(tmp_path, models) as memory:
        first = memory.add(
            "  red screwdriver in drawer two  ",
            occurred_at=occurred_at,
            metadata={"room": "workshop", "priority": 2},
        )
        duplicate = memory.add(
            "red screwdriver in drawer two",
            occurred_at=occurred_at.astimezone(timezone.utc),
            metadata={"priority": 2, "room": "workshop"},
        )
        assert duplicate == first
        assert len(first.id) == 64
        assert len(models.embed_batches) == 1
        assert memory.get(first.id) == first

        hits = memory.search("red screwdriver")
        assert [hit.id for hit in hits] == [first.id]
        answer = memory.ask("where is the red screwdriver?")
        assert first.content in answer.answer
        assert answer.hits[0].id == first.id

        assert memory.delete(first.id) is True
        assert memory.delete(first.id) is False
        with pytest.raises(MemoryNotFoundError):
            memory.get(first.id)

    assert models.closed is True
    assert models.close_calls == 1


def test_memory_types_are_stable_and_filterable(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        semantic = memory.add("shared instruction")
        assert memory.add("shared instruction", memory_type=MemoryType.SEMANTIC) == semantic
        episodic = memory.add("shared instruction", memory_type=MemoryType.EPISODIC)
        procedural = memory.add("shared instruction", memory_type=MemoryType.PROCEDURAL)
        _FakeIndex.instances[-1].hits_override = tuple(
            IndexHit(id=record.id, relevance=1.0) for record in (semantic, procedural, episodic)
        )

        assert len({semantic.id, episodic.id, procedural.id}) == 3
        assert semantic.memory_type is MemoryType.SEMANTIC
        assert memory.search("shared", memory_type=MemoryType.EPISODIC)[0].id == episodic.id
        assert memory.search("shared", memory_type=MemoryType.PROCEDURAL)[0].id == procedural.id
        with pytest.raises(ValidationError, match="MemoryType"):
            memory.add("invalid type", memory_type="episodic")  # type: ignore[arg-type]


def test_relative_time_prefers_event_time_and_routes_the_reference(tmp_path: Path) -> None:
    reference = datetime(2026, 8, 27, 0, 30, tzinfo=timezone(timedelta(hours=14)))
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        previous = memory.add(
            "项目评审发生了",
            occurred_at=datetime(2026, 8, 20, 9, tzinfo=timezone.utc),
            memory_type=MemoryType.EPISODIC,
        )
        memory.add(
            "项目评审发生了",
            occurred_at=datetime(2026, 8, 26, 9, tzinfo=timezone.utc),
            memory_type=MemoryType.EPISODIC,
        )

        answer = memory.ask(
            "上周发生了什么?",
            limit=1,
            memory_type=MemoryType.EPISODIC,
            reference_at=reference,
        )

    assert answer.hits[0].id == previous.id
    assert "Reference time for relative dates: 2026-08-27T00:30:00.000000+14:00" in (
        models.answer_calls[-1][0].text
    )


def test_decay_reranks_softly_and_reinforces_only_returned_hits(tmp_path: Path) -> None:
    reference = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    with _memory(
        tmp_path,
        _FakeModels(),
        decay_half_life_days=7,
    ) as memory:
        memory.add(
            "shared memory old",
            occurred_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
        fresh = memory.add(
            "shared memory fresh",
            occurred_at=datetime(2026, 8, 27, 11, tzinfo=timezone.utc),
        )
        hit = memory.search("shared memory", limit=1, reference_at=reference)[0]
        assert hit.id == fresh.id
        assert hit.score == 1.0

    with LocalStore(tmp_path) as store:
        stored = store.read_memory(fresh.id)
        assert stored is not None
        assert stored.access_count == 1
        assert stored.last_accessed_at is not None


def test_ask_without_answerer_fails_before_retrieval_or_reinforcement(tmp_path: Path) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        decay_half_life_days=7,
    ) as memory:
        record = memory.add("red target")
        index = _FakeIndex.instances[-1]

        with pytest.raises(ModelError, match="answer backend is not configured"):
            memory.ask("red target")

        stored = memory._store.read_memory(record.id)
        assert stored is not None
        assert stored.access_count == 0
        assert models.embed_tasks == [EmbedTask.DOCUMENT]
        assert index.hybrid_search_calls == 0


def test_explicit_embedder_owns_embedding_and_models_own_generation(tmp_path: Path) -> None:
    models = _FakeModels()
    embedder = _FakeEmbedder(capabilities=frozenset({Modality.TEXT}))

    with _memory(tmp_path, models, embedder=embedder) as memory:
        record = memory.add("red separate backend")
        answer = memory.ask("where is red?")
        document = _FakeIndex.instances[-1].documents[record.id]

        assert answer.hits[0].id == record.id
        assert models.embed_batches == []
        assert models.answer_calls
        assert embedder.embed_tasks == [EmbedTask.DOCUMENT, EmbedTask.QUERY]
        assert document.embedding.model_id == embedder.embedding_model
        assert document.embedding.space_id == embedder.embedding_space

        with pytest.raises(ModelError, match="image"):
            memory.add(Blob(b"unsupported image", "image/png", "frame.png"))

    assert embedder.close_calls == 1
    assert models.close_calls == 1


def test_memory_requires_an_explicit_embedder(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="embedder"):
        Memory(tmp_path)  # type: ignore[call-arg]


def test_explicit_speech_is_lazy_and_recognizes_a_speaker_across_recordings(
    tmp_path: Path,
) -> None:
    embedder = _FakeEmbedder()
    transcriber = _FakeSpeech()

    with Memory(tmp_path, embedder=embedder, transcriber=transcriber) as memory:
        first = memory.add(Blob(b"first recording", "audio/wav", "first.wav"))
        second = memory.add(Blob(b"second recording", "audio/wav", "second.wav"))
        assert transcriber.calls == []

        first_speech = memory.speech(first.id)
        second_speech = memory.speech(second.id)
        speaker_id = first_speech[0].speaker_id
        assert speaker_id is not None
        assert second_speech[0].speaker_id == speaker_id
        assert first_speech[0].identity_score is None
        assert second_speech[0].identity_score == pytest.approx(1.0)
        memory.register_speaker(speaker_id, "Alice")
        assert memory.speech(first.id)[0].speaker_name == "Alice"
        memory.register_speaker(speaker_id, "Alicia")
        assert memory.speech(second.id)[0].speaker_name == "Alicia"
        with pytest.raises(SpeakerNotFoundError):
            memory.register_speaker("speaker_missing", "Nobody")
        with pytest.raises(ValidationError):
            memory.register_speaker(speaker_id, "bad\nname")
        assert len(transcriber.calls) == 2

    assert transcriber.closed is True


def test_add_many_deduplicates_one_model_and_store_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    original = LocalStore.write_memories
    store_batches: list[tuple[str, ...]] = []

    def counted_write(
        store: LocalStore,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, ...]:
        batch = tuple(memories)
        store_batches.append(tuple(memory.memory_id for memory in batch))
        result: tuple[bool, ...] = original(store, batch, embeddings)
        return result

    monkeypatch.setattr(LocalStore, "write_memories", counted_write)
    with _memory(tmp_path, models) as memory:
        records = memory.add_many(("alpha", "alpha", "red beta"))
        assert [record.id for record in records] == [records[0].id, records[0].id, records[2].id]
        assert models.embed_batches == [("alpha", "red beta")]
        assert len(store_batches) == 1
        assert len(store_batches[0]) == 2

        repeated = memory.add_many(("alpha", "red beta"))
        assert [record.id for record in repeated] == [records[0].id, records[2].id]
        assert models.embed_batches == [("alpha", "red beta")]
        assert len(store_batches) == 1


def test_add_many_hydrates_the_index_outbox_in_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_connection = LocalStore._connection
    connection_count = 0

    @contextmanager
    def counted_connection(store: LocalStore) -> Iterator[sqlite3.Connection]:
        nonlocal connection_count
        connection_count += 1
        with original_connection(store) as connection:
            yield connection

    monkeypatch.setattr(LocalStore, "_connection", counted_connection)
    with _memory(tmp_path, _FakeModels()) as memory:
        before = connection_count
        memory.add_many(tuple(f"memory {index}" for index in range(100)))

    assert connection_count - before <= 8


def test_outbox_bounds_index_batches(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        memory.add_many(tuple(f"memory {index}" for index in range(600)))

    assert [len(batch) for batch in _FakeIndex.instances[-1].upsert_calls] == [256, 256, 88]


def test_concurrent_adds_share_one_durable_index_flush(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = LocalStore.write_memories
    committed = Barrier(2)

    def synchronized_write(
        store: LocalStore,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, ...]:
        result = original(store, memories, embeddings)
        committed.wait(timeout=3)
        return result

    monkeypatch.setattr(LocalStore, "write_memories", synchronized_write)
    with (
        _memory(tmp_path, _FakeModels()) as memory,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        records = tuple(pool.map(memory.add, ("first concurrent", "second concurrent")))

    assert len(records) == 2
    assert len(_FakeIndex.instances[-1].upsert_calls) == 1
    assert set(_FakeIndex.instances[-1].upsert_calls[0]) == {record.id for record in records}


def test_reindex_replays_an_add_committed_after_its_sqlite_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_taken = Event()
    add_committed = Event()
    original_rebuild = _FakeIndex.rebuild
    original_write = LocalStore.write_memories

    def paused_rebuild(
        index: _FakeIndex,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int = 1_024,
        optimize_concurrency: int = 0,
    ) -> int:
        snapshot = tuple(documents)
        snapshot_taken.set()
        assert add_committed.wait(timeout=3)
        return original_rebuild(
            index,
            snapshot,
            batch_size=batch_size,
            optimize_concurrency=optimize_concurrency,
        )

    def tracked_write(
        store: LocalStore,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, ...]:
        result = original_write(store, memories, embeddings)
        add_committed.set()
        return result

    with _memory(tmp_path, _FakeModels()) as memory:
        existing = memory.add("existing")
        monkeypatch.setattr(_FakeIndex, "rebuild", paused_rebuild)
        monkeypatch.setattr(LocalStore, "write_memories", tracked_write)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reindexed = pool.submit(memory.reindex)
            assert snapshot_taken.wait(timeout=3)
            added = pool.submit(memory.add, "committed during rebuild")
            assert reindexed.result(timeout=3) == 1
            new = added.result(timeout=3)

        assert set(_FakeIndex.instances[-1].documents) == {existing.id, new.id}


def test_keyset_pages_reindex_optimize_and_missing_index_recovery(tmp_path: Path) -> None:
    models = _FakeModels()
    memory = _memory(tmp_path, models)
    records = memory.add_many(("one", "two", "three", "four", "five"))

    seen: list[str] = []
    cursor = None
    while True:
        page = memory.list(limit=2, cursor=cursor)
        seen.extend(record.id for record in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == len(records)
    assert memory.reindex() == len(records)
    memory.optimize()
    assert _FakeIndex.instances[-1].rebuild_calls == 1
    assert _FakeIndex.instances[-1].rebuild_batch_sizes == [256]
    assert _FakeIndex.instances[-1].optimize_calls == 1
    memory.close()

    embed_calls = len(models.embed_batches)
    shutil.rmtree(tmp_path / "zvec")
    reopened_models = _FakeModels()
    with _memory(tmp_path, reopened_models) as reopened:
        assert reopened.search("one")
        assert reopened_models.embed_batches == [("one",)]
        assert len(models.embed_batches) == embed_calls


def test_missing_index_checkpoint_precedes_collection_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        records = memory.add_many(("one", "two"))
    shutil.rmtree(tmp_path / "zvec")

    def fail_after_create(path: str | Path, dimension: int) -> _FakeIndex:
        _FakeIndex(path, dimension)
        raise RuntimeError("simulated crash after collection creation")

    monkeypatch.setattr(memory_module, "ZvecIndex", fail_after_create)
    with pytest.raises(IndexUnavailableError, match="open"):
        _memory(tmp_path, _FakeModels())

    monkeypatch.setattr(memory_module, "ZvecIndex", _FakeIndex)
    with _memory(tmp_path, _FakeModels()):
        assert set(_FakeIndex.instances[-1].documents) == {record.id for record in records}


def test_legacy_index_recipe_is_rebuilt_from_sqlite(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add("preserved episodic memory", memory_type=MemoryType.EPISODIC)
    with LocalStore(tmp_path) as store:
        store.set_metadata(
            "index.recipe",
            "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2",
        )

    with _memory(tmp_path, _FakeModels()):
        document = _FakeIndex.instances[-1].documents[record.id]
        assert document.memory_type == "episodic"


def test_interrupted_reindex_is_completed_from_durable_sqlite(tmp_path: Path) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        records = memory.add_many(("one", "two", "three"))
        _FakeIndex.instances[-1].fail_next_rebuild = True
        with pytest.raises(IndexUnavailableError, match="rebuild"):
            memory.reindex()

    with _memory(tmp_path, _FakeModels()):
        assert set(_FakeIndex.instances[-1].documents) == {record.id for record in records}


def test_delete_recreate_coalesces_outbox_and_stale_hits_are_filtered(tmp_path: Path) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        record = memory.add("red notebook")
        index = _FakeIndex.instances[-1]
        index.hits_override = (
            IndexHit(id="stale-index-only-id", relevance=1.0),
            IndexHit(id=record.id, relevance=0.8),
        )
        assert [hit.id for hit in memory.search("red notebook", limit=2)] == [record.id]
        index.hits_override = None

        index.fail_next_flush = True
        with pytest.raises(IndexUnavailableError):
            memory.delete(record.id)
        index.upsert_calls.clear()
        index.delete_calls.clear()

        recreated = memory.add("red notebook")
        assert recreated.id == record.id
        assert index.delete_calls == []
        assert index.upsert_calls == [(record.id,)]
        assert [hit.id for hit in memory.search("red notebook")] == [record.id]


def test_data_directories_are_isolated_and_metadata_changes_fail_fast(tmp_path: Path) -> None:
    first_models = _FakeModels()
    second_models = _FakeModels()
    with (
        _memory(tmp_path / "first", first_models) as first,
        _memory(tmp_path / "second", second_models) as second,
    ):
        with pytest.raises(StorageError, match="already in use"):
            _memory(tmp_path / "first", _FakeModels())
        record = first.add("red item only in first")
        assert second.search("red item") == ()
        assert second.list().items == ()
        assert first.get(record.id).id == record.id

    with pytest.raises(StorageError, match="metadata mismatch"):
        _memory(tmp_path / "first", _FakeModels(model="different-model"))
    with pytest.raises(StorageError, match=r"transcription\.space_id"):
        _memory(tmp_path / "second", _FakeModels(transcription_space="different-asr"))


def test_invalid_zvec_embedding_space_cannot_poison_store_metadata(tmp_path: Path) -> None:
    invalid = _FakeModels()
    invalid.embedding_space = "invalid'space"

    with pytest.raises(ValidationError, match="unsupported by Zvec"):
        _memory(tmp_path, invalid)

    with _memory(tmp_path, _FakeModels()) as memory:
        assert memory.add("valid after rejected contract").content


def test_path_media_uses_native_dense_search_and_cas_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"stored image")
    models = _FakeModels()

    with _memory(tmp_path / "memory", models) as memory:
        record = memory.add(source)
        assert record.content == ""
        assert record.modality is Modality.IMAGE
        assert record.assets[0].media_type == "image/png"
        assert record.assets[0].path is not None
        assert record.assets[0].path.read_bytes() == b"stored image"
        assert models.embed_inputs[0][0].modalities == {Modality.IMAGE}

        hits = memory.search(Blob(b"query image", "image/png", "query.png"))
        assert hits[0].id == record.id
        assert _FakeIndex.instances[-1].dense_search_calls == 1
        asset_files = tuple(
            path for path in (tmp_path / "memory" / "assets").rglob("*") if path.is_file()
        )
        assert asset_files == (record.assets[0].path,)

        assert memory.delete(record.id) is True
        assert not record.assets[0].path.exists()


def test_persisted_media_reads_do_not_run_gc_ownership_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    original = LocalStore.read_unreferenced_assets

    def counted(
        store: LocalStore,
        asset_ids: Sequence[str],
    ) -> tuple[StoredAsset, ...]:
        nonlocal calls
        calls += 1
        return original(store, asset_ids)

    monkeypatch.setattr(LocalStore, "read_unreferenced_assets", counted)
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add(("red image", Blob(b"stored", "image/png", "stored.png")))
        calls = 0

        memory.get(record.id)
        memory.list()
        memory.search("red image")

        assert calls == 0


def test_duplicate_asset_names_return_authoritative_cas_metadata(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add(
            (
                Blob(b"same image", "image/png", "first.png"),
                Blob(b"same image", "image/png", "second.png"),
            )
        )

        assert [asset.name for asset in record.assets] == ["first.png", "first.png"]
        assert memory.get(record.id) == record


def test_add_many_transcribes_one_shared_asset_once(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=frozenset({Modality.TEXT}),
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )
    audio = Blob(b"same audio", "audio/wav", "first.wav")

    with _memory(tmp_path, models) as memory:
        records = memory.add_many((("red first", audio), ("red second", audio)))

    assert len(records) == 2
    assert len(models.transcribe_calls) == 1
    assert len(models.transcribe_calls[0]) == 1


def test_add_many_batches_distinct_audio_transcriptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=frozenset({Modality.TEXT}),
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )

    def unexpected_transcript_transaction(
        _store: LocalStore,
        _values: Sequence[tuple[str, str]],
    ) -> int:
        pytest.fail("add_many wrote transcripts outside its memory transaction")

    monkeypatch.setattr(LocalStore, "set_asset_transcripts", unexpected_transcript_transaction)
    with _memory(tmp_path, models) as memory:
        records = memory.add_many(
            (
                Blob(b"first audio", "audio/wav", "first.wav"),
                Blob(b"second audio", "audio/wav", "second.wav"),
            )
        )
        stored = memory._store.read_assets(tuple(record.assets[0].id for record in records))

    assert len(models.transcribe_calls) == 1
    assert len(models.transcribe_calls[0]) == 2
    assert [asset.transcript for asset in stored] == ["spoken red wrench"] * 2


def test_ask_reuses_query_transcript_for_generation(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=frozenset({Modality.TEXT}),
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        memory.add("red target")
        models.transcribe_calls.clear()
        memory.ask(("find it", Blob(b"query audio", "audio/wav", "query.wav")))

    assert len(models.transcribe_calls) == 1
    assert models.answer_calls[-1][0].modalities == {Modality.TEXT}


def test_ask_batches_and_persists_hit_transcripts_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )
    writes: list[tuple[tuple[str, str], ...]] = []
    set_asset_transcripts = LocalStore.set_asset_transcripts

    def record_transcript_transaction(
        store: LocalStore,
        values: Sequence[tuple[str, str]],
    ) -> int:
        batch = tuple(values)
        writes.append(batch)
        return set_asset_transcripts(store, batch)

    monkeypatch.setattr(LocalStore, "set_asset_transcripts", record_transcript_transaction)

    with _memory(tmp_path, models) as memory:
        memory.add_many(
            (
                ("red first", Blob(b"first hit audio", "audio/wav", "first.wav")),
                ("red second", Blob(b"second hit audio", "audio/wav", "second.wav")),
            )
        )
        memory.ask("red")
        memory.ask("red")

    assert len(models.transcribe_calls) == 1
    assert len(models.transcribe_calls[0]) == 2
    assert len(writes) == 1
    assert len(writes[0]) == 2


def test_no_hit_ask_routes_media_and_cannot_accept_fabricated_hits(tmp_path: Path) -> None:
    class FabricatingModels(_FakeModels):
        def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
            super().answer(question, hits)
            fabricated = SearchHit(
                id="fabricated",
                content="not retrieved",
                score=1.0,
                created_at=datetime.now(timezone.utc),
            )
            return AnswerResult(answer="unknown", hits=(fabricated,))

    models = FabricatingModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )
    with _memory(tmp_path, models) as memory:
        result = memory.ask(Blob(b"question audio", "audio/wav", "question.wav"))

    assert result.hits == ()
    assert models.answer_calls[-1][0].modalities == {Modality.TEXT}


def test_startup_removes_a_cas_file_without_sqlite_metadata(tmp_path: Path) -> None:
    asset_store = AssetStore(tmp_path)
    orphan = asset_store.materialize_bytes(
        b"crash window",
        modality="image",
        mime_type="image/png",
        name="orphan.png",
    )
    assert asset_store.resolve(orphan).exists()

    with _memory(tmp_path, _FakeModels()):
        assert not (tmp_path / orphan.relative_path).exists()


def test_delete_gc_recovers_from_index_and_file_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add(Blob(b"first image", "image/png", "first.png"))
        path = record.assets[0].path
        assert path is not None
        _FakeIndex.instances[-1].fail_next_flush = True
        with pytest.raises(IndexUnavailableError):
            memory.delete(record.id)
        assert not path.exists()
        assert memory.delete(record.id) is False

        second = memory.add(Blob(b"second image", "image/png", "second.png"))
        second_path = second.assets[0].path
        assert second_path is not None
        original_unlink = Path.unlink

        def fail_asset_unlink(value: Path, missing_ok: bool = False) -> None:
            if value == second_path:
                raise OSError("simulated unlink failure")
            original_unlink(value, missing_ok=missing_ok)

        with monkeypatch.context() as changed:
            changed.setattr(Path, "unlink", fail_asset_unlink)
            with pytest.raises(StorageError, match="orphaned media"):
                memory.delete(second.id)
        assert second_path.exists()
        assert memory.delete(second.id) is False
        assert not second_path.exists()


def test_audio_falls_back_to_asr_while_visual_input_stays_native(tmp_path: Path) -> None:
    capabilities = _Capabilities(
        embedding=frozenset({Modality.TEXT, Modality.VIDEO}),
        generation=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        transcription=frozenset({Modality.AUDIO}),
    )
    models = _FakeModels(capabilities=capabilities)

    with _memory(tmp_path, models) as memory:
        record = memory.add(
            (
                "red repair session",
                Blob(b"spoken instructions", "audio/wav", "instructions.wav"),
                Blob(b"video frames", "video/mp4", "session.mp4"),
            )
        )
        embedded = models.embed_inputs[0][0]
        assert record.modality is Modality.OMNI
        assert {asset.modality for asset in record.assets} == {
            Modality.AUDIO,
            Modality.VIDEO,
        }
        assert embedded.modalities == {Modality.TEXT, Modality.VIDEO}
        assert "spoken red wrench" in embedded.text
        assert len(models.transcribe_calls) == 1

        answer = memory.ask("What happened in the red repair session?")
        _question, routed_hits = models.answer_calls[-1]
        assert routed_hits[0].modality is Modality.VIDEO
        assert {asset.modality for asset in routed_hits[0].assets} == {Modality.VIDEO}
        assert "spoken red wrench" in routed_hits[0].content
        assert answer.hits[0].modality is Modality.OMNI
        assert len(models.transcribe_calls) == 1


def test_vlm_generation_transcribes_audio_once_and_keeps_video(tmp_path: Path) -> None:
    capabilities = _Capabilities(
        embedding=ALL_INPUT_MODALITIES,
        generation=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        transcription=frozenset({Modality.AUDIO}),
    )
    models = _FakeModels(capabilities=capabilities)

    with _memory(tmp_path, models) as memory:
        record = memory.add(
            (
                "red workbench",
                Blob(b"meeting audio", "audio/wav", "meeting.wav"),
                Blob(b"bench video", "video/mp4", "bench.mp4"),
            )
        )
        assert models.transcribe_calls == []

        question = (
            "What is on the red workbench?",
            AssetRef(record.assets[0].id),
            AssetRef(record.assets[1].id),
        )
        result = memory.ask(question)
        routed_question, routed_hits = models.answer_calls[-1]
        assert routed_question.modalities == {Modality.TEXT, Modality.VIDEO}
        assert "spoken red wrench" in routed_question.text
        assert routed_hits[0].modality is Modality.VIDEO
        assert result.hits[0].modality is Modality.OMNI
        assert len(models.transcribe_calls) == 1

        memory.ask(question)
        assert len(models.transcribe_calls) == 1


def test_vlm_generation_recognizes_complete_speaker_identity_in_parallel(
    tmp_path: Path,
) -> None:
    query_started = Event()
    identity_started = Event()

    class ParallelModels(_FakeModels):
        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.QUERY:
                query_started.set()
                assert identity_started.wait(timeout=1)
            return super().embed(inputs, task)

    class IdentitySpeech(_FakeSpeech):
        def __init__(self) -> None:
            super().__init__()
            self.expect_parallel = False

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            if self.expect_parallel:
                identity_started.set()
                assert query_started.wait(timeout=1)
            return super().analyze(assets)

        def transcribe(self, _assets: Sequence[AssetRef]) -> tuple[str, ...]:
            raise AssertionError("answer generation must retain complete speaker identity")

    capabilities = _Capabilities(
        embedding=ALL_INPUT_MODALITIES,
        generation=frozenset({Modality.TEXT, Modality.VIDEO}),
        transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
    )
    models = ParallelModels(capabilities=capabilities)
    speech = IdentitySpeech()

    with _memory(tmp_path, models, transcriber=speech) as memory:
        record = memory.add(
            (
                "red recording",
                Blob(b"stored audio", "audio/wav", "stored.wav"),
                Blob(b"stored video", "video/mp4", "stored.mp4"),
            )
        )
        enrolled = memory.speech(record.id)
        speaker_id = enrolled[0].speaker_id
        assert speaker_id is not None
        assert {segment.speaker_id for segment in enrolled} == {speaker_id}
        memory.register_speaker(speaker_id, "Alice")

        speech.expect_parallel = True
        result = memory.ask(
            ("red: who is speaking?", Blob(b"query audio", "audio/wav", "query.wav"))
        )

        routed_question, routed_hits = models.answer_calls[-1]
        question_evidence = [
            json.loads(line)
            for line in routed_question.text.splitlines()
            if line.startswith('{"asset_id":')
        ]
        hit_evidence = [
            json.loads(line)
            for line in routed_hits[0].content.splitlines()
            if line.startswith('{"asset_id":')
        ]
        assert len(question_evidence) == 1
        assert question_evidence[0]["segments"] == [
            {
                "start_ms": 0,
                "end_ms": 900,
                "text": "spoken red wrench",
                "speaker_id": speaker_id,
                "speaker_name": "Alice",
                "identity_score": 1.0,
            }
        ]
        assert len(hit_evidence) == 2
        assert all(
            evidence["segments"][0]["speaker_id"] == speaker_id
            and evidence["segments"][0]["speaker_name"] == "Alice"
            for evidence in hit_evidence
        )
        assert routed_question.modalities == {Modality.TEXT}
        assert routed_hits[0].modality is Modality.VIDEO
        assert {asset.modality for asset in routed_hits[0].assets} == {Modality.VIDEO}
        assert "[speech identities:" not in result.hits[0].content
        assert memory._store.read_asset(question_evidence[0]["asset_id"]) is None
        assert len(speech.calls) == 2


def test_invalid_long_transcript_is_not_persisted(tmp_path: Path) -> None:
    class LongTranscriptModels(_FakeModels):
        def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
            batch = tuple(assets)
            self.transcribe_calls.append(batch)
            return tuple("x" * 70_000 for _asset in batch)

    models = LongTranscriptModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )
    with _memory(tmp_path, models) as memory:
        memory.add(("red audio", Blob(b"audio", "audio/wav", "audio.wav")))

        with pytest.raises(ModelError, match="transcription exceeded"):
            memory.ask("red")
        with pytest.raises(ModelError, match="transcription exceeded"):
            memory.ask("red")

    assert len(models.transcribe_calls) == 2


def test_unsupported_visual_embedding_fails_without_leaking_media(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=frozenset({Modality.TEXT}),
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        with pytest.raises(ModelError, match="image"):
            memory.add(Blob(b"unsupported image", "image/png", "frame.png"))
        assert memory.list().items == ()
        assert not any(path.is_file() for path in (tmp_path / "assets").rglob("*"))


def test_generation_never_silently_drops_visual_evidence(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=frozenset({Modality.TEXT}),
            transcription=frozenset({Modality.AUDIO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        memory.add(("red diagram", Blob(b"diagram", "image/png", "diagram.png")))
        with pytest.raises(ModelError, match="image"):
            memory.ask("Show the red diagram")


def test_memory_rejects_oversized_and_recursive_input_before_model_work(tmp_path: Path) -> None:
    models = _FakeModels()
    recursive: dict[str, object] = {}
    recursive["recursive"] = recursive

    with _memory(tmp_path, models) as memory:
        with pytest.raises(ValidationError, match="65536"):
            memory.add("x" * 65_537)
        with pytest.raises(ValidationError, match="262144"):
            memory.add("valid", metadata={"blob": "x" * 262_144})
        with pytest.raises(ValidationError, match="JSON-compatible"):
            memory.add("valid", metadata=recursive)

    assert models.embed_batches == []


def test_cas_write_failure_maps_to_storage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated full disk")

    monkeypatch.setattr("mindbridge.infrastructure.local.assets.os.replace", fail_replace)
    with (
        _memory(tmp_path, _FakeModels()) as memory,
        pytest.raises(StorageError, match="materialize media"),
    ):
        memory.add(Blob(b"image", "image/png", "image.png"))

    assert not any(path.is_file() for path in (tmp_path / ".asset-staging").iterdir())


def test_memory_instances_cannot_be_reused_after_fork(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path, _FakeModels())
    owner_pid = os.getpid()
    with monkeypatch.context() as changed:
        changed.setattr(os, "getpid", lambda: owner_pid + 1)
        with pytest.raises(StorageError, match="after fork"):
            memory.list()
        with pytest.raises(StorageError, match="after fork"):
            memory.close()
    memory.close()


def test_close_waits_for_an_active_search(tmp_path: Path) -> None:
    class BlockingEmbedder(_FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()
            self.closed_during_embed = False

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            self.started.set()
            assert self.release.wait(5)
            self.closed_during_embed = self.closed
            return super().embed(inputs, task)

    models = _FakeModels()
    embedder = BlockingEmbedder()
    memory = _memory(tmp_path, models, embedder=embedder)
    errors: list[BaseException] = []
    close_done = Event()

    def search() -> None:
        try:
            memory.search("active query")
        except BaseException as error:
            errors.append(error)

    def close() -> None:
        try:
            memory.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_done.set()

    search_thread = Thread(target=search)
    close_thread = Thread(target=close)
    search_thread.start()
    assert embedder.started.wait(2)
    close_thread.start()
    assert not close_done.wait(0.05)
    embedder.release.set()
    search_thread.join(5)
    close_thread.join(5)

    assert not search_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert embedder.closed_during_embed is False
    assert embedder.close_calls == 1
    assert models.closed is True
    assert models.close_calls == 1


def test_model_work_runs_concurrently_while_index_access_stays_serialized(tmp_path: Path) -> None:
    class ConcurrentModels(_FakeModels):
        def __init__(self) -> None:
            super().__init__()
            self.query_barrier = Barrier(2)
            self.block_queries = False

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if self.block_queries and task is EmbedTask.QUERY:
                self.query_barrier.wait(timeout=3)
            return super().embed(inputs, task)

    models = ConcurrentModels()
    memory = _memory(tmp_path, models)
    memory.add("red concurrent memory")
    models.block_queries = True
    errors: list[BaseException] = []

    def search() -> None:
        try:
            memory.search("red concurrent")
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=search) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    memory.close()

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_temporary_media_gc_is_not_starved_by_an_unrelated_long_request(tmp_path: Path) -> None:
    class LongQueryModels(_FakeModels):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.QUERY and inputs[0].text == "long query":
                self.started.set()
                assert self.release.wait(5)
            return super().embed(inputs, task)

    models = LongQueryModels()
    memory = _memory(tmp_path, models)
    memory.add("red target")
    errors: list[BaseException] = []

    def long_search() -> None:
        try:
            memory.search("long query")
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=long_search)
    thread.start()
    assert models.started.wait(2)
    for index in range(8):
        memory.search(Blob(f"query-{index}".encode(), "image/png", f"query-{index}.png"))

    assert not any(path.is_file() for path in (tmp_path / "assets").rglob("*"))
    models.release.set()
    thread.join(5)
    memory.close()

    assert not thread.is_alive()
    assert errors == []


@pytest.mark.asyncio
async def test_async_memory_matches_sync_surface(tmp_path: Path) -> None:
    models = _FakeModels()
    embedder = _FakeEmbedder()
    async with AsyncMemory(
        tmp_path,
        embedder=embedder,
        answerer=models,
        transcriber=models,
    ) as memory:
        records = await memory.add_many(("red async memory", "another memory"))
        assert await memory.get(records[0].id) == records[0]
        assert (await memory.search("red async"))[0].id == records[0].id
        assert (await memory.ask("what is red?")).hits
        assert (await memory.list(limit=1)).items
        assert await memory.reindex() == 2
        await memory.optimize()
        assert await memory.delete(records[0].id) is True

    assert models.embed_batches == []
    assert embedder.close_calls == 1
    assert models.close_calls == 1
