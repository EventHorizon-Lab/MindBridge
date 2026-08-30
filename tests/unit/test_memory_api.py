from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from collections.abc import Generator, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event, Thread
from typing import ClassVar

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import mindbridge.memory as memory_module
from mindbridge._telemetry import (
    MODEL_REQUEST_COUNT,
    MODEL_TTFT,
    TOKEN_COMPLETE,
    TOKEN_TOTAL,
    mark_model_requests,
    record_model_usage,
    record_unmetered_model_usage,
)
from mindbridge.exceptions import (
    IdentityNotFoundError,
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
    FaceAnalysis,
    FaceEmbedding,
    ModelInput,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)
from mindbridge.types import (
    AnswerResult,
    AssetRef,
    Blob,
    IndexQuantization,
    MemoryType,
    Modality,
    SearchHit,
)

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
        self._legacy_embedding_spaces: frozenset[str] = frozenset()
        self.transcription_model = "fake-transcription"
        self.transcription_space = transcription_space
        self.embedding_dimension = 2
        self.fail_embedding = False
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
        if self.fail_embedding:
            raise RuntimeError("simulated embedding failure")
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


class _FakeFace:
    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "fake-sface"
    face_space = "fake-sface:2:test"
    face_analysis_space = "fake-yunet-sface:test"

    def __init__(self) -> None:
        self.calls: list[tuple[AssetRef, ...]] = []
        self.closed = False

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        batch = tuple(assets)
        self.calls.append(batch)
        return tuple(
            FaceAnalysis(
                (
                    FaceEmbedding(
                        "face-0",
                        (0.0, 1.0),
                        (0.1, 0.1, 0.4, 0.5),
                        100 if asset.modality is Modality.VIDEO else None,
                    ),
                )
            )
            for asset in batch
        )

    def close(self) -> None:
        self.closed = True


class _FakeIndex:
    documents_by_path: ClassVar[dict[str, dict[str, IndexDocument]]] = {}
    instances: ClassVar[list[_FakeIndex]] = []

    def __init__(
        self,
        path: str | Path,
        dimension: int,
        *,
        quantization: IndexQuantization = IndexQuantization.NONE,
    ) -> None:
        self.path = Path(path).resolve()
        self.dimension = dimension
        self.quantization = quantization
        key = str(self.path)
        if not self.path.exists():
            self.documents_by_path[key] = {}
            self.path.mkdir(parents=True)
        self.documents = self.documents_by_path.setdefault(key, {})
        self.fail_next_flush = False
        self.hits_override: tuple[IndexHit, ...] | None = None
        self.dense_hits_override: tuple[IndexHit, ...] | None = None
        self.lexical_hits_override: tuple[IndexHit, ...] | None = None
        self.upsert_calls: list[tuple[str, ...]] = []
        self.delete_calls: list[tuple[str, ...]] = []
        self.dense_search_calls = 0
        self.lexical_search_calls = 0
        self.lexical_queries: list[str] = []
        self.optimize_calls = 0
        self.optimize_if_needed_calls = 0
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
        if self.dense_hits_override is not None:
            return self.dense_hits_override[:limit]
        if self.hits_override is not None:
            return self.hits_override[:limit]
        return tuple(
            IndexHit(id=document_id, relevance=0.75, confidence=0.75)
            for document_id, document in self.documents.items()
            if (space_id is None or document.embedding.space_id == space_id)
            and (task is None or document.embedding.task == task)
            and self._matches_time_and_type(document, memory_type, occurred_from, occurred_until)
        )[:limit]

    def lexical_search(
        self,
        text: str,
        *,
        limit: int = 10,
        space_id: str | None = None,
        task: str | None = None,
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> tuple[IndexHit, ...]:
        self.lexical_search_calls += 1
        self.lexical_queries.append(text)
        if self.lexical_hits_override is not None:
            return self.lexical_hits_override[:limit]
        if self.hits_override is not None:
            return tuple(hit for hit in self.hits_override if hit.lexical_match)[:limit]
        query_terms = set(re.findall(r"\w+", text.casefold()))
        matched = []
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
            if query_terms & set(re.findall(r"\w+", document.content.casefold())):
                matched.append(document_id)
        return tuple(
            IndexHit(
                id=document_id,
                relevance=61 / (60 + rank),
                confidence=0.0,
                lexical_match=True,
            )
            for rank, document_id in enumerate(sorted(matched)[:limit], start=1)
        )

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
        if occurred_at is None:
            return False
        occurred_end = document.occurred_end or occurred_at + timedelta(microseconds=1)
        return (occurred_from is None or occurred_end > occurred_from) and (
            occurred_until is None or occurred_at < occurred_until
        )

    def flush(self) -> None:
        if self.fail_next_flush:
            self.fail_next_flush = False
            raise RuntimeError("simulated flush failure")

    def optimize(self, *, concurrency: int = 0) -> None:
        del concurrency
        self.optimize_calls += 1

    def optimize_if_needed(self, *, minimum_unindexed: int = 100_000) -> bool:
        del minimum_unindexed
        self.optimize_if_needed_calls += 1
        return False

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


def test_memory_traces_end_to_end_stages_and_streaming_ttft(tmp_path: Path) -> None:
    class StreamingModels(_FakeModels):
        generation_model = "fake-generation"

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            vectors = super().embed(inputs, task)
            record_unmetered_model_usage()
            return vectors

        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
        ) -> Iterator[str]:
            del question, hits
            record_model_usage(input_tokens=5, output_tokens=3, total_tokens=8)
            yield "grounded "
            yield "answer"

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    models = StreamingModels()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        transcriber=models,
        tracer=provider.get_tracer("test"),
    ) as memory:
        memory.add("red trace evidence")
        assert memory.ask("what is red?").answer == "grounded answer"
    provider.shutdown()

    spans = exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert {
        "mindbridge.add",
        "mindbridge.ask",
        "mindbridge.content.prepare",
        "mindbridge.model.embedding",
        "mindbridge.index.search",
        "mindbridge.storage.write",
        "mindbridge.retrieval.rank",
        "mindbridge.model.generation",
    } <= names
    generation = next(span for span in spans if span.name == "mindbridge.model.generation")
    ask = next(span for span in spans if span.name == "mindbridge.ask")
    assert generation.attributes is not None
    ttft = generation.attributes[MODEL_TTFT]
    assert isinstance(ttft, int | float) and ttft >= 0
    assert generation.parent is not None
    assert generation.parent.span_id == ask.context.span_id
    assert ask.attributes is not None
    assert ask.attributes[TOKEN_COMPLETE] is True
    assert ask.attributes[TOKEN_TOTAL] == 8


def test_trace_errors_never_export_exception_details(tmp_path: Path) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    models = _FakeModels()
    with Memory(tmp_path, embedder=models, tracer=provider.get_tracer("test")) as memory:
        with pytest.raises(ValidationError):
            memory.add(tmp_path / "private-missing-image.png")
        models.fail_embedding = True
        with pytest.raises(ModelError):
            memory.add("private model input")
    provider.shutdown()

    failed = tuple(
        span
        for span in exporter.get_finished_spans()
        if span.status.status_code is StatusCode.ERROR
    )
    assert {span.name for span in failed} >= {
        "mindbridge.add",
        "mindbridge.content.prepare",
        "mindbridge.model.embedding",
    }
    assert all(not span.events and span.status.description is None for span in failed)


@pytest.mark.parametrize("chunks", [(), (" ",)])
def test_empty_stream_is_invalid_model_output(tmp_path: Path, chunks: tuple[str, ...]) -> None:
    class EmptyStreamModels(_FakeModels):
        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
        ) -> Iterator[str]:
            del question, hits
            yield from chunks

    models = EmptyStreamModels()
    with _memory(tmp_path, models) as memory:
        memory.add("red evidence")
        with pytest.raises(ModelError, match="invalid answer"):
            memory.ask("what is red?")


def test_stream_ttft_requires_an_actual_model_request(tmp_path: Path) -> None:
    class SkippingStreamModels(_FakeModels):
        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
        ) -> Iterator[str]:
            del question, hits
            mark_model_requests(0, token_usage_expected=0)
            yield "I do not know."

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    models = SkippingStreamModels()
    with Memory(
        tmp_path, embedder=models, answerer=models, tracer=provider.get_tracer("test")
    ) as memory:
        memory.add("red evidence")
        assert memory.ask("what is red?").answer == "I do not know."
    provider.shutdown()

    generation = next(
        span for span in exporter.get_finished_spans() if span.name == "mindbridge.model.generation"
    )
    assert generation.attributes is not None
    assert generation.attributes[MODEL_REQUEST_COUNT] == 0
    assert MODEL_TTFT not in generation.attributes


def test_streaming_answer_reports_only_the_hits_the_stream_used(tmp_path: Path) -> None:
    class SelectingStreamModels(_FakeModels):
        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
        ) -> Generator[str, None, tuple[SearchHit, ...]]:
            del question
            yield "grounded"
            return (hits[0],)

    models = SelectingStreamModels()
    with _memory(tmp_path, models) as memory:
        memory.add_many(("red first", "red second"))
        result = memory.ask("red", limit=2)

    assert result.answer == "grounded"
    assert len(result.hits) == 1


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


def test_named_month_and_calendar_year_prefer_event_time(tmp_path: Path) -> None:
    reference = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
    relative = memory_module._parse_temporal_range("2024 days ago", reference)
    assert relative is not None
    assert relative[0].date() == reference.date() - timedelta(days=2024)

    with _memory(tmp_path, _FakeModels()) as memory:
        december = memory.add(
            "shared conference memory",
            occurred_at=datetime(2023, 12, 15, tzinfo=timezone.utc),
        )
        april = memory.add(
            "shared conference memory",
            occurred_at=datetime(2024, 4, 15, tzinfo=timezone.utc),
        )
        memory.add(
            "shared conference memory",
            occurred_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        )

        assert memory.search("What happened in December 2023?", limit=1)[0].id == december.id
        assert memory.search("2024年4月发生了什么?", limit=1)[0].id == april.id
        assert memory.search("What happened at BMVC 2024?", limit=3)[0].id == april.id


def test_natural_today_anchor_sets_relative_time_unless_reference_is_explicit(
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        anchored = memory.add(
            "weekly project review happened",
            occurred_at=datetime(2024, 4, 25, tzinfo=timezone.utc),
        )
        explicit = memory.add(
            "weekly project review happened",
            occurred_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
        )
        question = "Today is May 2, 2024. What happened last week?"

        answer = memory.ask(question, limit=1)
        overridden = memory.search(
            question,
            limit=1,
            reference_at=datetime(2026, 5, 2, 12, tzinfo=timezone.utc),
        )

        with pytest.raises(ValidationError, match="today reference date is invalid"):
            memory.search("Today is February 30, 2024. What happened yesterday?")

    assert answer.hits[0].id == anchored.id
    assert overridden[0].id == explicit.id
    assert "Reference time for relative dates: 2024-05-02T00:00:00.000000+00:00" in (
        models.answer_calls[-1][0].text
    )


def test_decay_reranks_softly_and_requires_explicit_reinforcement(tmp_path: Path) -> None:
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
        assert 0.0 < hit.score < 1.0
        stored = memory._store.read_memory(fresh.id)
        assert stored is not None
        assert stored.access_count == 0
        assert memory.reinforce((fresh.id, fresh.id, "missing")) == 1

    with LocalStore(tmp_path) as store:
        stored = store.read_memory(fresh.id)
        assert stored is not None
        assert stored.access_count == 1
        assert stored.last_accessed_at is not None
        assert (
            memory_module._ranked_relevance(
                stored,
                0.5,
                reference_at=stored.last_accessed_at,
                temporal_range=None,
                decay_half_life=None,
            )
            > 0.5
        )
        assert (
            memory_module._ranked_relevance(
                stored,
                0.5,
                reference_at=reference,
                temporal_range=None,
                decay_half_life=None,
            )
            == 0.5
        )


@pytest.mark.parametrize("days", [True, 0, -1, float("nan"), float("inf"), "7"])
def test_decay_half_life_rejects_invalid_values(tmp_path: Path, days: object) -> None:
    with pytest.raises(ValidationError, match="positive finite number"):
        Memory(tmp_path, embedder=_FakeModels(), decay_half_life_days=days)  # type: ignore[arg-type]


def test_temporal_proximity_is_a_soft_score_not_a_hard_sort_key(tmp_path: Path) -> None:
    reference = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    with _memory(tmp_path, _FakeModels()) as memory:
        exact = memory.add(
            "shared project review exact",
            occurred_at=datetime(2026, 8, 20, 12, tzinfo=timezone.utc),
        )
        adjacent = memory.add(
            "shared project review adjacent",
            occurred_at=datetime(2026, 8, 21, 1, tzinfo=timezone.utc),
        )
        _FakeIndex.instances[-1].hits_override = (
            IndexHit(id=exact.id, relevance=0.55),
            IndexHit(id=adjacent.id, relevance=1.0),
        )

        hits = memory.search(
            "What happened on 2026-08-20?",
            limit=2,
            reference_at=reference,
        )
        assert _FakeIndex.instances[-1].dense_search_calls == 2
        assert _FakeIndex.instances[-1].lexical_search_calls == 1

    assert [hit.id for hit in hits] == [adjacent.id, exact.id]


def test_text_reranking_preserves_negation_and_bounded_scores(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        negated = memory.add(
            "Mira did not put the obsidian key in the red drawer; it is in the green drawer."
        )
        contradiction = memory.add("Mira put the obsidian key in the red drawer.")
        index = _FakeIndex.instances[-1]
        index.dense_hits_override = (
            IndexHit(
                id=contradiction.id,
                relevance=0.79,
                confidence=0.9,
            ),
            IndexHit(
                id=negated.id,
                relevance=0.788,
                confidence=0.9,
            ),
        )
        index.lexical_hits_override = (
            IndexHit(id=negated.id, relevance=1.0, confidence=0.0, lexical_match=True),
            IndexHit(
                id=contradiction.id,
                relevance=61 / 62,
                confidence=0.0,
                lexical_match=True,
            ),
        )

        hits = memory.search(
            "Which memory says Mira did not put the obsidian key in the red drawer?",
            limit=2,
        )

    assert [hit.id for hit in hits] == [negated.id, contradiction.id]
    assert all(0.0 < hit.score < 1.0 for hit in hits)


def test_text_reranking_does_not_override_strong_semantic_evidence(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        semantic = memory.add("A scarlet package reached the destination.")
        question_echo = memory.add("Did the red parcel arrive?")
        _FakeIndex.instances[-1].hits_override = (
            IndexHit(id=question_echo.id, relevance=0.1, confidence=0.9, lexical_match=True),
            IndexHit(id=semantic.id, relevance=0.9, confidence=0.95),
        )

        hits = memory.search("Did the red parcel arrive?", limit=2)

    assert [hit.id for hit in hits] == [semantic.id, question_echo.id]


def test_exact_lexical_evidence_beats_a_weak_dense_neighbor(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        exact = memory.add("obsidian key green drawer")
        neighbor = memory.add("unrelated scene")
        index = _FakeIndex.instances[-1]
        index.dense_hits_override = (
            IndexHit(id=neighbor.id, relevance=0.72, confidence=0.86),
            IndexHit(id=exact.id, relevance=0.1, confidence=0.55),
        )
        index.lexical_hits_override = (
            IndexHit(id=exact.id, relevance=1.0, confidence=0.0, lexical_match=True),
        )

        hits = memory.search("obsidian key green drawer", limit=2)

    assert [hit.id for hit in hits] == [exact.id, neighbor.id]


def test_event_span_overlapping_query_day_is_temporally_exact(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        spanning = memory.add(
            "overnight project review",
            occurred_at=datetime(2026, 8, 19, 23, tzinfo=timezone.utc),
            occurred_end=datetime(2026, 8, 20, 1, tzinfo=timezone.utc),
        )
        memory.add(
            "later project review",
            occurred_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        )

        hit = memory.search("What happened on 2026-08-20?", limit=1)[0]

    assert hit.id == spanning.id
    assert hit.occurred_end == datetime(2026, 8, 20, 1, tzinfo=timezone.utc)


def test_temporal_candidate_merge_keeps_the_best_vector_score() -> None:
    merged = memory_module._merge_index_hits(
        (IndexHit(id="shared", relevance=0.2),),
        (IndexHit(id="shared", relevance=0.9),),
    )

    assert merged == (IndexHit(id="shared", relevance=0.9),)


def test_ambiguity_gate_only_rejects_single_result_retrieval(tmp_path: Path) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        first = memory.add("first scene")
        second = memory.add("second scene")
        index = _FakeIndex.instances[-1]

        index.hits_override = (IndexHit(id=first.id, relevance=0.54),)
        assert memory.search("unrelated question") == ()

        index.hits_override = (
            IndexHit(id=first.id, relevance=0.9, confidence=0.8),
            IndexHit(id=second.id, relevance=0.89, confidence=0.795),
        )
        assert [hit.id for hit in memory.search("which repeated scene?", limit=2)] == [
            first.id,
            second.id,
        ]
        assert memory.search("which repeated scene?", limit=1) == ()
        assert [hit.id for hit in memory.ask("which repeated scene?", limit=2).hits] == [
            first.id,
            second.id,
        ]
        assert memory.ask("which repeated scene?", limit=1).hits == ()
        assert models.answer_calls[-1][1] == ()

        index.hits_override = (
            IndexHit(
                id=first.id,
                relevance=0.9,
                confidence=0.8,
                lexical_match=True,
            ),
            IndexHit(id=second.id, relevance=0.89, confidence=0.795),
        )
        assert memory.search("first scene")[0].id == first.id

        index.dense_hits_override = ()
        index.lexical_hits_override = (
            IndexHit(id=first.id, relevance=1.0, confidence=0.0, lexical_match=True),
            IndexHit(id=second.id, relevance=61 / 62, confidence=0.0, lexical_match=True),
        )
        assert [hit.id for hit in memory.search("scene")] == [first.id, second.id]

        index.lexical_hits_override = (
            IndexHit(id=first.id, relevance=1.0, confidence=0.0, lexical_match=True),
        )
        assert memory.search("what") == ()


def test_ask_without_answerer_fails_before_retrieval_or_reinforcement(tmp_path: Path) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        decay_half_life_days=7,
    ) as memory:
        record = memory.add("red target")
        index = _FakeIndex.instances[-1]

        with pytest.raises(ModelError, match="answer backend is not configured") as unconfigured:
            memory.ask("red target")
        assert unconfigured.value.reason == "backend_not_configured"
        assert unconfigured.value.retryable is False

        stored = memory._store.read_memory(record.id)
        assert stored is not None
        assert stored.access_count == 0
        assert models.embed_tasks == [EmbedTask.DOCUMENT]
        assert index.dense_search_calls == 0
        assert index.lexical_search_calls == 0


def test_unsupported_schema_is_permanent_where_a_busy_directory_is_not(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add("remember this")

    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.execute("PRAGMA user_version = 999")
        connection.commit()

    with pytest.raises(StorageError) as failure:
        _memory(tmp_path)

    assert failure.value.reason == "schema_unsupported"
    assert failure.value.retryable is False
    assert failure.value.stage == "open"


def test_a_failing_batch_item_names_its_position(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory, pytest.raises(ValidationError) as failure:
        memory.add_many(("first", "second", "   "))

    assert failure.value.subject == "contents[2]"
    assert failure.value.stage == "content.prepare"


def test_storage_failures_name_the_stage_that_failed(tmp_path: Path) -> None:
    with _memory(tmp_path) as memory:
        memory.add("remember this")
        with (
            _failing_store(memory, "list_memories"),
            pytest.raises(StorageError) as failure,
        ):
            memory.list()

    assert failure.value.reason == "io_failed"
    assert failure.value.retryable is False


@contextmanager
def _failing_store(memory: Memory, method: str) -> Iterator[None]:
    original = getattr(memory._store, method)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("disk I/O error")

    setattr(memory._store, method, fail)
    try:
        yield
    finally:
        setattr(memory._store, method, original)


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


def test_face_and_voice_recognition_share_one_durable_identity(tmp_path: Path) -> None:
    speech = _FakeSpeech()
    faces = _FakeFace()

    with Memory(
        tmp_path,
        embedder=_FakeEmbedder(),
        transcriber=speech,
        face_analyzer=faces,
    ) as memory:
        first = memory.add(Blob(b"first video", "video/mp4", "first.mp4"))
        second = memory.add(Blob(b"second video", "video/mp4", "second.mp4"))

        first_face = memory.faces(first.id)[0]
        first_speaker = memory.speech(first.id)[0]
        second_face = memory.faces(second.id)[0]
        second_speaker = memory.speech(second.id)[0]

        assert first_face.identity_id == first_speaker.speaker_id
        assert second_face.identity_id == first_face.identity_id
        assert second_speaker.speaker_id == first_face.identity_id
        memory.register_identity(first_face.identity_id, "Alice")
        assert memory.faces(first.id)[0].identity_name == "Alice"
        assert memory.speech(second.id)[0].speaker_name == "Alice"
        with pytest.raises(IdentityNotFoundError):
            memory.register_identity("identity_missing", "Nobody")

    assert faces.closed is True
    assert speech.closed is True


def test_face_voice_merge_atomically_refreshes_speech_index_and_keeps_alias(
    tmp_path: Path,
) -> None:
    class FailingRefreshEmbedder(_FakeEmbedder):
        fail_documents = False

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if self.fail_documents and task is EmbedTask.DOCUMENT:
                raise RuntimeError("embedding failed")
            return super().embed(inputs, task)

    embedder = FailingRefreshEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        transcriber=_FakeSpeech(),
        face_analyzer=_FakeFace(),
        index_speech=True,
    ) as memory:
        image = memory.add(Blob(b"known face", "image/png", "face.png"))
        face_id = memory.faces(image.id)[0].identity_id
        video = memory.add(Blob(b"new voice video", "video/mp4", "voice.mp4"))
        voice_id = memory.speech(video.id)[0].speaker_id
        assert voice_id is not None and voice_id != face_id
        assert voice_id in memory.get(video.id).content

        embedder.fail_documents = True
        with pytest.raises(ModelError, match="embed"):
            memory.faces(video.id)
        assert memory._store.resolve_identity_id(voice_id) == voice_id
        assert memory.speech(video.id)[0].speaker_id == voice_id
        assert voice_id in memory.get(video.id).content

        embedder.fail_documents = False
        assert memory.faces(video.id)[0].identity_id == face_id
        assert memory._store.resolve_identity_id(voice_id) == face_id
        assert memory.speech(video.id)[0].speaker_id == face_id
        refreshed = memory.get(video.id)
        assert face_id in refreshed.content
        assert voice_id not in refreshed.content
        documents = tuple(
            document
            for document in _FakeIndex.instances[-1].documents.values()
            if document.embedding.memory_id == video.id
        )
        assert documents and all(voice_id not in document.content for document in documents)

        memory.register_speaker(voice_id, "Alice")
        assert memory.speech(video.id)[0].speaker_name == "Alice"
        assert '"speaker_name":"Alice"' in memory.get(video.id).content


def test_faces_requires_an_explicit_face_backend(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_FakeEmbedder()) as memory:
        record = memory.add(Blob(b"image", "image/png", "face.png"))
        with pytest.raises(ModelError, match="face backend"):
            memory.faces(record.id)


def test_face_only_identity_uses_generic_registration(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_FakeEmbedder(), face_analyzer=_FakeFace()) as memory:
        record = memory.add(Blob(b"image", "image/png", "face.png"))
        identity_id = memory.faces(record.id)[0].identity_id

        with pytest.raises(SpeakerNotFoundError):
            memory.register_speaker(identity_id, "Alice")
        memory.register_identity(identity_id, "Alice")
        assert memory.faces(record.id)[0].identity_name == "Alice"


def test_face_embedding_recipe_change_fails_fast(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_FakeEmbedder(), face_analyzer=_FakeFace()):
        pass

    incompatible = _FakeFace()
    incompatible.face_space = "fake-sface:2:different"
    with pytest.raises(StorageError, match=r"face\.space_id"):
        Memory(tmp_path, embedder=_FakeEmbedder(), face_analyzer=incompatible)


def test_answer_generation_receives_linked_face_and_voice_identity_evidence(
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        transcriber=_FakeSpeech(),
        face_analyzer=_FakeFace(),
    ) as memory:
        record = memory.add(Blob(b"red person video", "video/mp4", "person.mp4"))
        identity_id = memory.faces(record.id)[0].identity_id
        memory.register_identity(identity_id, "Alice")

        memory.ask("where is red?")

    evidence = models.answer_calls[-1][1][0].content
    assert "[face identities:" in evidence
    assert "[speech identities:" in evidence
    assert '"identity_name":"Alice"' in evidence
    assert '"speaker_name":"Alice"' in evidence


def test_opt_in_speech_indexing_makes_registered_names_retrievable(
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    speech = _FakeSpeech()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        transcriber=speech,
        index_speech=True,
    ) as memory:
        first = memory.add(Blob(b"first speech", "audio/wav", "first.wav"))
        first_segments = memory.speech(first.id)
        speaker_id = first_segments[0].speaker_id
        assert speaker_id is not None
        memory.register_speaker(speaker_id, "Alice")
        refreshed_first = memory.get(first.id)
        assert '"speaker_name":"Alice"' in refreshed_first.content

        second = memory.add(Blob(b"second speech", "audio/wav", "second.wav"))
        memory.register_speaker(speaker_id, "Alicia")
        index = _FakeIndex.instances[-1]

        assert len(speech.calls) == 2
        assert '"speaker_name":"Alicia"' in memory.get(first.id).content
        assert '"speaker_name":"Alicia"' in memory.get(second.id).content
        assert '"speaker_name":"Alice"' in second.content
        assert {document.embedding.object_part for document in index.documents.values()} == {
            0,
            1,
            2,
        }


def test_generation_deduplicates_indexed_speech_evidence(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )
    )
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        transcriber=_FakeSpeech(),
        index_speech=True,
    ) as memory:
        record = memory.add(Blob(b"red speech", "audio/wav", "speech.wav"))

        result = memory.ask("what was said?")

    routed = models.answer_calls[-1][1][0].content
    assert routed.count("[speech identities:") == 1
    assert routed.count("spoken red wrench") == 1
    assert "[transcript:" not in routed
    assert result.hits[0].content == record.content


def test_speech_indexing_batches_add_many_recognition(tmp_path: Path) -> None:
    speech = _FakeSpeech()
    with Memory(
        tmp_path,
        embedder=_FakeModels(),
        transcriber=speech,
        index_speech=True,
    ) as memory:
        memory.add_many(
            (
                Blob(b"first speech", "audio/wav", "first.wav"),
                Blob(b"second speech", "audio/wav", "second.wav"),
            )
        )

    assert len(speech.calls) == 1
    assert len(speech.calls[0]) == 2


def test_failed_speech_index_add_rolls_back_identity_state(tmp_path: Path) -> None:
    models = _FakeModels()
    models.fail_embedding = True

    with Memory(
        tmp_path,
        embedder=models,
        transcriber=_FakeSpeech(),
        index_speech=True,
    ) as memory:
        with pytest.raises(ModelError, match="embed memory input"):
            memory.add(Blob(b"failed speech", "audio/wav", "failed.wav"))

        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            counts = tuple(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "memory_records",
                    "media_assets",
                    "speech_analyses",
                    "identities",
                    "identity_exemplars",
                )
            )

    assert counts == (0, 0, 0, 0, 0)


def test_failed_speech_index_add_restores_matched_identity(tmp_path: Path) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        transcriber=_FakeSpeech(),
        index_speech=True,
    ) as memory:
        memory.add(Blob(b"first speech", "audio/wav", "first.wav"))
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            before = (
                connection.execute(
                    "SELECT identity_id, name, created_at, updated_at FROM identities"
                ).fetchall(),
                connection.execute(
                    """
                    SELECT identity_id, modality, position, model_id, space_id,
                           dimension, vector, created_at
                    FROM identity_exemplars
                    ORDER BY identity_id, modality, position
                    """
                ).fetchall(),
            )
        models.fail_embedding = True

        with pytest.raises(ModelError, match="embed memory input"):
            memory.add(Blob(b"second speech", "audio/wav", "second.wav"))

        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            after = (
                connection.execute(
                    "SELECT identity_id, name, created_at, updated_at FROM identities"
                ).fetchall(),
                connection.execute(
                    """
                    SELECT identity_id, modality, position, model_id, space_id,
                           dimension, vector, created_at
                    FROM identity_exemplars
                    ORDER BY identity_id, modality, position
                    """
                ).fetchall(),
            )
            analysis_count = connection.execute("SELECT COUNT(*) FROM speech_analyses").fetchone()[
                0
            ]

    assert after == before
    assert analysis_count == 1


def test_speaker_rename_refreshes_indexed_memory_after_reopen(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=_FakeModels(),
        transcriber=_FakeSpeech(),
        index_speech=True,
    ) as memory:
        record = memory.add(Blob(b"speech", "audio/wav", "speech.wav"))
        speaker_id = memory.speech(record.id)[0].speaker_id
        assert speaker_id is not None
        memory.register_speaker(speaker_id, "Alice")

    with Memory(
        tmp_path,
        embedder=_FakeModels(),
        transcriber=_FakeSpeech(),
    ) as reopened:
        reopened.register_speaker(speaker_id, "Alicia")
        refreshed = reopened.get(record.id)

        assert '"speaker_name":"Alicia"' in refreshed.content
        assert '"speaker_name":"Alice"' not in refreshed.content
        assert [hit.id for hit in reopened.search("Alicia")] == [record.id]


def test_speaker_registration_rolls_back_when_refresh_embedding_fails(tmp_path: Path) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        transcriber=_FakeSpeech(),
        index_speech=True,
    ) as memory:
        record = memory.add(Blob(b"speech", "audio/wav", "speech.wav"))
        speaker_id = memory.speech(record.id)[0].speaker_id
        assert speaker_id is not None
        models.fail_embedding = True

        with pytest.raises(ModelError, match="embed memory input"):
            memory.register_speaker(speaker_id, "Alice")

        assert memory.speech(record.id)[0].speaker_name is None
        assert '"speaker_name":null' in memory.get(record.id).content


def test_composite_memory_uses_max_over_aggregate_and_atomic_vectors(tmp_path: Path) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        record = memory.add(("red label", Blob(b"image", "image/png", "frame.png")))
        index = _FakeIndex.instances[-1]
        child_ids = tuple(
            document_id for document_id in index.documents if document_id != record.id
        )
        assert len(child_ids) == 2
        assert all(len(child_id) == 64 and child_id.isalnum() for child_id in child_ids)
        assert len(set(child_ids)) == 2
        assert [value.modalities for value in models.embed_inputs[0]] == [
            {Modality.TEXT, Modality.IMAGE},
            {Modality.TEXT},
            {Modality.IMAGE},
        ]
        index.hits_override = (
            IndexHit(id=record.id, relevance=0.2),
            IndexHit(id=child_ids[-1], relevance=0.9),
        )

        hits = memory.search("find the frame")
        assert memory.reindex() == 1

    assert len(hits) == 1
    assert hits[0].id == record.id
    assert hits[0].score == pytest.approx(0.9)


def test_parent_fusion_preserves_strong_atomic_dense_evidence(tmp_path: Path) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        semantic = memory.add(
            ("A scarlet package reached the destination.", Blob(b"image", "image/png"))
        )
        question_echo = memory.add("Did the red parcel arrive?")
        index = _FakeIndex.instances[-1]
        atomic_id = next(
            document_id
            for document_id, document in index.documents.items()
            if document.embedding.memory_id == semantic.id and document_id != semantic.id
        )
        index.dense_hits_override = (
            IndexHit(id=atomic_id, relevance=0.9, confidence=0.95),
            IndexHit(id=question_echo.id, relevance=0.1, confidence=0.9),
        )
        index.lexical_hits_override = (
            IndexHit(
                id=question_echo.id,
                relevance=1.0,
                confidence=0.0,
                lexical_match=True,
            ),
        )

        hits = memory.search("Did the red parcel arrive?", limit=2)

    assert [hit.id for hit in hits] == [semantic.id, question_echo.id]


def test_composite_query_keeps_aggregate_and_focused_retrieval_keys(tmp_path: Path) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        memory.add(("red frame", Blob(b"stored", "image/png", "stored.png")))

        memory.search(
            (
                "find the red frame",
                "Answer with the matching evidence only.",
                Blob(b"query", "image/png", "query.png"),
            )
        )
        composite = models.embed_inputs[-1]
        lexical_query = _FakeIndex.instances[-1].lexical_queries[-1]
        memory.search("find the red frame")
        single = models.embed_inputs[-1]

    assert [value.modalities for value in composite] == [
        {Modality.TEXT, Modality.IMAGE},
        {Modality.TEXT, Modality.IMAGE},
        {Modality.TEXT},
        {Modality.IMAGE},
    ]
    assert "Answer with the matching evidence only." in composite[0].text
    assert [value.text for value in composite[1:]] == [
        "find the red frame",
        "find the red frame",
        "",
    ]
    assert lexical_query == "find the red frame"
    assert len(single) == 1
    assert models.embed_tasks[-2:] == [EmbedTask.QUERY, EmbedTask.QUERY]


def test_composite_query_retains_atomic_media_recall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    with _memory(tmp_path, models) as memory:
        record = memory.add(Blob(b"stored", "image/png", "stored.png"))
        index = _FakeIndex.instances[-1]

        def media_only_search(
            current: _FakeIndex,
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
            del space_id, task, memory_type, occurred_from, occurred_until, ef, exact
            current.dense_search_calls += 1
            return (
                (IndexHit(id=record.id, relevance=0.9, confidence=0.9),)
                if tuple(values) == (0.0, 1.0)
                else ()
            )[:limit]

        monkeypatch.setattr(_FakeIndex, "search", media_only_search)
        hits = memory.search(
            (
                "red question",
                "Answer with the matching evidence only.",
                Blob(b"query", "image/png", "query.png"),
            ),
            limit=1,
        )

    assert [hit.id for hit in hits] == [record.id]
    assert index.dense_search_calls == 2


def test_search_does_not_serialize_callers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = _FakeIndex.search
    barrier = Barrier(2)

    def synchronized_search(
        index: _FakeIndex,
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
        barrier.wait(timeout=3)
        return original(
            index,
            values,
            limit=limit,
            space_id=space_id,
            task=task,
            memory_type=memory_type,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            ef=ef,
            exact=exact,
        )

    with _memory(tmp_path, _FakeModels()) as memory:
        memory.add("red reference")
        monkeypatch.setattr(_FakeIndex, "search", synchronized_search)
        index = _FakeIndex.instances[-1]

        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent = tuple(executor.map(memory.search, ("what", "where")))

        assert all(concurrent)
        assert index.dense_search_calls == 2


def test_equal_scores_use_memory_id_as_a_stable_tiebreaker(tmp_path: Path) -> None:
    models = _FakeModels()
    with Memory(tmp_path, embedder=models, answerer=models, ambiguity_margin=0) as memory:
        records = memory.add_many(("first evidence", "second evidence"))
        _FakeIndex.instances[-1].hits_override = tuple(
            IndexHit(id=record.id, relevance=0.9, confidence=0.9) for record in reversed(records)
        )

        hits = memory.search("unrelated", limit=2)

    assert [hit.id for hit in hits] == sorted(record.id for record in records)


def test_search_expands_candidates_until_it_has_distinct_parent_memories(
    tmp_path: Path,
) -> None:
    models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        transcriber=models,
        ambiguity_margin=0,
    ) as memory:
        crowded = memory.add(tuple(f"crowded part {index}" for index in range(110)))
        other = memory.add("other parent")
        index = _FakeIndex.instances[-1]
        crowded_hits = tuple(
            IndexHit(id=document_id, relevance=0.9, confidence=0.9)
            for document_id, document in index.documents.items()
            if document.embedding.memory_id == crowded.id
        )
        assert len(crowded_hits) > 100
        other_id = next(
            document_id
            for document_id, document in index.documents.items()
            if document.embedding.memory_id == other.id
        )
        index.hits_override = (
            *crowded_hits,
            IndexHit(id=other_id, relevance=0.8, confidence=0.8),
        )

        hits = memory.search("find parents", limit=2)

    assert {hit.id for hit in hits} == {crowded.id, other.id}
    assert index.dense_search_calls == 2
    assert index.lexical_search_calls == 2


def test_long_text_adds_bounded_contextual_retrieval_keys(tmp_path: Path) -> None:
    models = _FakeModels()
    content = "source: diary\n" + "before " * 700 + "red needle " + "after " * 700

    with _memory(tmp_path, models) as memory:
        memory.add(content)

    embedded = models.embed_inputs[0]
    assert embedded[0].text == content.strip()
    assert any(
        "red needle" in value.text and len(value.text) < len(content) for value in embedded[1:]
    )
    assert len(embedded) <= 129


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


def test_add_many_preserves_per_record_event_time_and_metadata(tmp_path: Path) -> None:
    occurred = datetime(2026, 8, 20, 9, tzinfo=timezone.utc)
    occurred_end = occurred + timedelta(minutes=5)
    with _memory(tmp_path, _FakeModels()) as memory:
        records = memory.add_many(
            ("first", "second"),
            occurred_at=(occurred, None),
            occurred_end=(occurred_end, None),
            metadata=({"source_id": "one"}, {"source_id": "two"}),
        )

        assert records[0].occurred_at == occurred
        assert records[0].occurred_end == occurred_end
        assert records[0].metadata == {"source_id": "one"}
        assert records[1].occurred_at is None
        assert records[1].metadata == {"source_id": "two"}
        with pytest.raises(ValidationError, match="one value per content"):
            memory.add_many(("third",), occurred_at=(occurred, None))


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

    def fail_after_create(
        path: str | Path,
        dimension: int,
        *,
        quantization: IndexQuantization = IndexQuantization.NONE,
    ) -> _FakeIndex:
        _FakeIndex(path, dimension, quantization=quantization)
        raise RuntimeError("simulated crash after collection creation")

    monkeypatch.setattr(memory_module, "ZvecIndex", fail_after_create)
    with pytest.raises(IndexUnavailableError, match="open"):
        _memory(tmp_path, _FakeModels())

    monkeypatch.setattr(memory_module, "ZvecIndex", _FakeIndex)
    with _memory(tmp_path, _FakeModels()):
        assert set(_FakeIndex.instances[-1].documents) == {record.id for record in records}


@pytest.mark.parametrize(
    "recipe",
    (
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:single-vector-v3",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:multi-vector-v4",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:multi-vector-v5",
    ),
)
def test_legacy_index_recipe_is_rebuilt_from_sqlite(tmp_path: Path, recipe: str) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add("preserved episodic memory", memory_type=MemoryType.EPISODIC)
    with LocalStore(tmp_path) as store:
        store.set_metadata("index.recipe", recipe)

    reopened_models = _FakeModels()
    with _memory(tmp_path, reopened_models):
        document = _FakeIndex.instances[-1].documents[record.id]
        assert document.memory_type == "episodic"
        assert reopened_models.embed_inputs


def test_known_embedding_space_upgrade_reembeds_and_commits_marker(tmp_path: Path) -> None:
    legacy_space = "fake-embedding:2:legacy"
    original = _FakeModels()
    original.embedding_space = legacy_space
    with _memory(tmp_path, original) as memory:
        record = memory.add("preserved episodic memory", memory_type=MemoryType.EPISODIC)
    with LocalStore(tmp_path) as store:
        store.set_metadata(
            "index.recipe",
            "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:grouped-range:context-keys-v7",
        )

    upgraded = _FakeModels()
    upgraded._legacy_embedding_spaces = frozenset({legacy_space})
    with _memory(tmp_path, upgraded):
        document = _FakeIndex.instances[-1].documents[record.id]
        assert document.embedding.space_id == upgraded.embedding_space

    assert upgraded.embed_inputs
    with LocalStore(tmp_path) as store:
        assert store.get_metadata("embedding.space_id") == upgraded.embedding_space


def test_failed_embedding_space_upgrade_keeps_legacy_marker(tmp_path: Path) -> None:
    legacy_space = "fake-embedding:2:legacy"
    original = _FakeModels()
    original.embedding_space = legacy_space
    with _memory(tmp_path, original) as memory:
        memory.add("preserved memory")

    failing = _FakeModels()
    failing._legacy_embedding_spaces = frozenset({legacy_space})
    failing.fail_embedding = True
    with pytest.raises(ModelError, match="embed memory input"):
        _memory(tmp_path, failing)

    with LocalStore(tmp_path) as store:
        assert store.get_metadata("embedding.space_id") == legacy_space


@pytest.mark.parametrize(
    "legacy",
    (
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:context-keys-v6",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:grouped-range:context-keys-v7",
    ),
)
def test_schema_recipe_rebuilds_only_the_disposable_index(
    tmp_path: Path,
    legacy: str,
) -> None:
    with _memory(tmp_path, _FakeModels()) as memory:
        record = memory.add("preserved memory")
    with LocalStore(tmp_path) as store:
        store.set_metadata("index.recipe", legacy)

    reopened_models = _FakeModels()
    with _memory(tmp_path, reopened_models):
        assert record.id in _FakeIndex.instances[-1].documents

    assert reopened_models.embed_inputs == []


def test_index_quantization_change_rebuilds_without_reembedding(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_FakeModels()) as memory:
        record = memory.add("preserved memory")

    reopened_models = _FakeModels()
    with Memory(
        tmp_path,
        embedder=reopened_models,
        index_quantization=IndexQuantization.INT8,
    ):
        index = _FakeIndex.instances[-1]
        assert index.quantization is IndexQuantization.INT8
        assert record.id in index.documents

    assert reopened_models.embed_inputs == []
    with LocalStore(tmp_path) as store:
        recipe = store.get_metadata("index.recipe")
        assert recipe is not None
        assert recipe.endswith("quantization-int8")


def test_index_quantization_requires_the_public_enum(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="index_quantization"):
        Memory(
            tmp_path,
            embedder=_FakeModels(),
            index_quantization="int8",  # type: ignore[arg-type]
        )


def test_invalid_rabitq_dimension_does_not_mutate_the_current_index_recipe(
    tmp_path: Path,
) -> None:
    with Memory(tmp_path, embedder=_FakeModels()) as memory:
        memory.add("preserved memory")
    sentinel = tmp_path / "zvec" / "preserved"
    sentinel.write_text("still here", encoding="utf-8")
    with LocalStore(tmp_path) as store:
        recipe = store.get_metadata("index.recipe")

    with pytest.raises(ValidationError, match="RABITQ requires"):
        Memory(
            tmp_path,
            embedder=_FakeModels(),
            index_quantization=IndexQuantization.RABITQ,
        )

    assert sentinel.read_text(encoding="utf-8") == "still here"
    with LocalStore(tmp_path) as store:
        assert store.get_metadata("index.recipe") == recipe


def test_failed_embedding_recipe_migration_keeps_legacy_marker(tmp_path: Path) -> None:
    legacy = (
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:multi-vector-v5"
    )
    with _memory(tmp_path, _FakeModels()) as memory:
        memory.add("preserved memory")
    with LocalStore(tmp_path) as store:
        store.set_metadata("index.recipe", legacy)

    failing = _FakeModels()
    failing.fail_embedding = True
    with pytest.raises(ModelError, match="embed memory input"):
        _memory(tmp_path, failing)

    with LocalStore(tmp_path) as store:
        assert store.get_metadata("index.recipe") == legacy


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
        with pytest.raises(StorageError, match="already in use") as busy:
            _memory(tmp_path / "first", _FakeModels())
        # Transient and permanent open failures no longer collapse into one message, and the
        # directory travels in `subject` rather than in the message every transport forwards.
        assert busy.value.reason == "data_dir_in_use"
        assert busy.value.retryable is True
        assert busy.value.stage == "open"
        assert busy.value.subject == str(tmp_path / "first")
        assert str(tmp_path / "first") not in str(busy.value)
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


def test_omni_add_batches_declared_transcripts_and_ask_reuses_them(
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
        assert len(models.transcribe_calls) == 1
        assert len(models.transcribe_calls[0]) == 2
        models.transcribe_calls.clear()

        memory.ask("red")
        memory.ask("red")

    assert models.transcribe_calls == []
    # add persists a declared transcript inside its own memory write, so answering never
    # re-transcribes a stored hit or takes the separate transcript transaction.
    assert writes == []


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


def test_ask_returns_only_retrieved_hits_the_answerer_used(tmp_path: Path) -> None:
    class SelectingModels(_FakeModels):
        def answer(self, question: ModelInput, hits: Sequence[SearchHit]) -> AnswerResult:
            super().answer(question, hits)
            fabricated = SearchHit(
                id="fabricated",
                content="not retrieved",
                score=1.0,
                created_at=datetime.now(timezone.utc),
            )
            return AnswerResult(answer="grounded", hits=(hits[0], fabricated))

    models = SelectingModels()
    with _memory(tmp_path, models) as memory:
        memory.add_many(("red first", "red second"))
        result = memory.ask("red", limit=2)

    assert result.answer == "grounded"
    assert result.hits == (models.answer_calls[-1][1][0],)


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


def test_bare_media_memory_indexes_a_declared_transcript(tmp_path: Path) -> None:
    models = _FakeModels()

    with _memory(tmp_path, models) as memory:
        record = memory.add(Blob(b"kettle recording", "audio/wav", "kettle.wav"))
        index = _FakeIndex.instances[-1]
        documents = memory._store.read_memory_index_documents((record.id,))

    assert "spoken red wrench" in record.content
    # Zvec attaches the BM25 document to the aggregate part alone, so an empty record content is
    # an empty lexical document for the whole memory.
    assert "spoken red wrench" in index.documents[record.id].content
    assert [document.embedding.object_part for document in documents] == [0, 1, 2]
    assert {document.embedding.task for document in documents} == {"retrieval.document"}
    assert {document.embedding.space_id for document in documents} == {models.embedding_space}


def test_declared_video_speech_becomes_indexed_text(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=ALL_INPUT_MODALITIES,
            transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        record = memory.add(Blob(b"kitchen recording", "video/mp4", "kitchen.mp4"))

    assert record.modality is Modality.VIDEO
    assert "spoken red wrench" in record.content
    assert [asset.modality for asset in models.transcribe_calls[0]] == [Modality.VIDEO]


def test_the_transcript_marker_names_no_modality(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=ALL_INPUT_MODALITIES,
            generation=ALL_INPUT_MODALITIES,
            transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        record = memory.add(Blob(b"kitchen recording", "video/mp4", "kitchen.mp4"))
        indexed = _FakeIndex.instances[-1].documents[record.id].content

    # One lexical match on its own reaches `_LEXICAL_MATCH_CONFIDENCE`, which is above the default
    # weak-evidence floor, so every word of the marker is a term that makes this memory visible to
    # any query containing it. A marker naming a modality both mislabels a video transcript as
    # audio and hands each media memory a free match on an ordinary English word.
    assert memory_module._LEXICAL_MATCH_CONFIDENCE > memory_module._DEFAULT_MINIMUM_RELEVANCE
    assert "spoken red wrench" in indexed
    assert not {"audio", "video"} & set(re.findall(r"\w+", indexed.casefold()))


def test_transcript_fallback_remains_when_speech_has_no_identity_block(tmp_path: Path) -> None:
    asset = replace(
        AssetStore(tmp_path).materialize_bytes(
            b"speech",
            modality="audio",
            mime_type="audio/wav",
            name="speech.wav",
        ),
        transcript="spoken red wrench",
    )

    derived = memory_module._derived_text("", (asset,))

    assert derived == f"[transcript:{asset.asset_id}]\nspoken red wrench"


def test_audio_fallback_also_transcribes_declared_video_speech(tmp_path: Path) -> None:
    models = _FakeModels(
        capabilities=_Capabilities(
            embedding=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            generation=ALL_INPUT_MODALITIES,
            transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )
    )

    with _memory(tmp_path, models) as memory:
        record = memory.add(
            (
                "red repair session",
                Blob(b"spoken instructions", "audio/wav", "instructions.wav"),
                Blob(b"session frames", "video/mp4", "session.mp4"),
            )
        )
        stored = memory._store.read_assets(tuple(asset.id for asset in record.assets))

    assert len(models.transcribe_calls) == 1
    assert len(models.transcribe_calls[0]) == 2
    assert [asset.transcript for asset in stored] == ["spoken red wrench"] * 2
    assert record.content.count("[transcript:") == 2
    # The indexed content is also the BM25 document, so the marker must not hand every video
    # memory a free lexical match on a word as ordinary as "audio" or "video".
    assert not {"audio", "video"} & set(re.findall(r"\w+", record.content))


def test_text_only_add_never_reaches_the_transcriber(tmp_path: Path) -> None:
    models = _FakeModels()

    with _memory(tmp_path, models) as memory:
        record = memory.add("red wrench on the bench")

    assert record.content == "red wrench on the bench"
    assert models.transcribe_calls == []
    assert models.embed_batches == [("red wrench on the bench",)]


def test_speech_backend_analysis_stays_behind_the_index_speech_opt_in(tmp_path: Path) -> None:
    speech = _FakeSpeech()

    with _memory(tmp_path, transcriber=speech) as memory:
        record = memory.add(Blob(b"kitchen recording", "video/mp4", "kitchen.mp4"))

    assert speech.calls == []
    assert record.content == ""


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
        assert len(models.transcribe_calls) == 1

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
    audio = Blob(b"audio", "audio/wav", "audio.wav")
    with _memory(tmp_path, models) as memory:
        with pytest.raises(ModelError, match="transcription exceeded"):
            memory.add(("red audio", audio))
        with pytest.raises(ModelError, match="transcription exceeded"):
            memory.add(("red audio", audio))

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
        assert await memory.reinforce((records[0].id,)) == 1
        assert await memory.reindex() == 2
        await memory.optimize()
        assert await memory.delete(records[0].id) is True

    assert models.embed_batches == []
    assert embedder.close_calls == 1
    assert models.close_calls == 1
