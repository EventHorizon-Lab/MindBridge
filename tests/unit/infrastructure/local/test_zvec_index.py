"""Focused checks for the disposable local Zvec index."""

from __future__ import annotations

import errno
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Barrier, Event
from typing import Any, NoReturn, cast

import pytest

import mindbridge.infrastructure.local.zvec_index as zvec_index_module
from mindbridge.infrastructure.local.store import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import (
    IndexHit,
    ZvecIndex,
    ZvecUnavailableError,
    ZvecWriteError,
    _CollectionGate,
)
from mindbridge.types import IndexQuantization

_NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def test_missing_zvec_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_module(_name: str) -> NoReturn:
        raise ModuleNotFoundError("No module named 'zvec'")

    monkeypatch.setattr(
        "mindbridge.infrastructure.local.zvec_index.import_module",
        missing_module,
    )

    with pytest.raises(ZvecUnavailableError, match=r"Zvec 0.7.*64-bit zvec wheel"):
        ZvecIndex(tmp_path / "index", dimension=2)


def test_rabitq_rejects_unsupported_dimensions_before_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 64 and 4095"):
        ZvecIndex(
            tmp_path / "index",
            dimension=2,
            quantization=IndexQuantization.RABITQ,
        )


def test_every_batch_status_is_checked() -> None:
    index = object.__new__(ZvecIndex)
    index.dimension = 2
    index._zvec = _FakeZvec
    index._collection = _FailingCollection()

    with pytest.raises(ZvecWriteError, match="embedding_bad: INTERNAL_ERROR: disk full"):
        index.upsert([_document("embedding_bad", "content", (1.0, 0.0))])


def test_only_aggregate_vector_carries_full_text() -> None:
    index = object.__new__(ZvecIndex)
    index.dimension = 2
    index._zvec = _FakeZvec
    collection = _CapturingCollection()
    index._collection = collection

    index.upsert(
        [
            _document("aggregate", "shared text", (1.0, 0.0)),
            _document("child", "shared text", (0.0, 1.0), object_part=1),
        ]
    )

    assert [
        cast(dict[str, object], document["fields"])["content"] for document in collection.documents
    ] == [
        "shared text",
        "",
    ]
    assert [
        cast(dict[str, object], document["fields"])["memory_id"]
        for document in collection.documents
    ] == ["memory_aggregate", "memory_child"]


def test_delete_accepts_only_not_found_failures() -> None:
    index = object.__new__(ZvecIndex)
    index._zvec = _FakeZvec
    index._collection = _DeletingCollection()

    index.delete(["already_deleted"])


def test_collection_gate_allows_readers_and_excludes_replacement() -> None:
    gate = _CollectionGate()
    readers = Barrier(2)

    def read() -> None:
        with gate.read():
            readers.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(lambda _index: read(), range(2)))

    waiting = Event()
    entered = Event()

    def write() -> None:
        waiting.set()
        with gate.write():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with gate.read():
            future = executor.submit(write)
            assert waiting.wait(timeout=2)
            assert not entered.wait(timeout=0.05)
        future.result(timeout=2)
    assert entered.is_set()


def test_lexical_search_preserves_relative_bm25_score_and_all_filters() -> None:
    collection = _LexicalCollection(
        [
            _QueryDocument("first", 0.8),
            _QueryDocument("second", 0.4),
        ]
    )
    index = object.__new__(ZvecIndex)
    index._zvec = _QueryZvec
    index._collection = collection
    index._gate = _CollectionGate()

    hits = index.lexical_search(
        "project review",
        limit=2,
        space_id="workspace",
        task="document",
        memory_type="episodic",
        occurred_from=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        occurred_until=datetime(1970, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
    )

    assert tuple(hit.id for hit in hits) == ("first", "second")
    assert tuple(hit.relevance for hit in hits) == (1.0, 0.5)
    assert all(hit.confidence == 0.0 for hit in hits)
    assert all(hit.lexical_match for hit in hits)
    assert collection.options == {
        "queries": {
            "field_name": "content",
            "fts": {"match_string": "project review"},
        },
        "topk": 2,
        "filter": (
            "space_id = 'workspace' AND task = 'document' AND "
            "memory_type = 'episodic' AND occurred_end > 1000000 AND "
            "occurred_at < 2000000"
        ),
        "include_vector": False,
        "output_fields": [],
    }


def test_group_by_falls_back_until_it_has_distinct_parent_memories() -> None:
    parent = _QueryDocument("parent_best", 0.0, memory_id="parent")
    collection = _GroupedCollection(
        groups=(_Group("parent", (parent,)),),
        documents=(
            parent,
            _QueryDocument("parent_child", 0.01, memory_id="parent"),
            _QueryDocument("neighbor", 0.5, memory_id="neighbor"),
        ),
    )
    index = object.__new__(ZvecIndex)
    index.dimension = 2
    index.ef_search = 300
    index.quantization = IndexQuantization.NONE
    index._zvec = _DenseZvec
    index._collection = collection

    docs = index._dense_query(
        (1.0, 0.0),
        limit=2,
        space_id=None,
        task=None,
        memory_type=None,
        occurred_from=None,
        occurred_until=None,
        ef=None,
        exact=False,
    )

    assert [cast(Any, doc).id for doc in docs] == ["parent_best", "neighbor"]
    assert collection.group_counts == [4]
    assert collection.topks == [50]
    assert collection.output_fields == [["memory_id"]]


def test_create_search_flush_close_and_reopen(tmp_path: Path) -> None:
    _require_zvec()
    path = tmp_path / "index"
    documents = [
        _document("embedding_one", "machine learning systems", (1.0, 0.0)),
        _document("embedding_two", "cooking recipes", (0.0, 1.0)),
        _document(
            "embedding_other_space",
            "machine intelligence",
            (0.8, 0.2),
            space_id="other-space",
            task="query",
        ),
    ]

    index = ZvecIndex(path, dimension=2)
    index.upsert(documents)
    assert index.doc_count == 3

    dense = index.search(
        (1.0, 0.0),
        limit=3,
        space_id="default-space",
        task="document",
        exact=True,
    )
    assert isinstance(dense[0], IndexHit)
    assert tuple(hit.id for hit in dense) == ("embedding_one", "embedding_two")
    assert dense[0].relevance == pytest.approx(1.0)
    assert dense[1].relevance == pytest.approx(0.5)

    lexical = index.lexical_search(
        "machine learning",
        limit=2,
        space_id="default-space",
        task="document",
    )
    assert lexical[0].id == "embedding_one"
    assert lexical[0].relevance == pytest.approx(1.0)

    index.optimize()
    assert index.index_completeness == pytest.approx(1.0)
    index.flush()
    index.close()
    index.close()
    with pytest.raises(RuntimeError, match="closed"):
        index.search((1.0, 0.0))

    with ZvecIndex(path, dimension=2) as reopened:
        assert reopened.search((1.0, 0.0), limit=1, exact=True)[0].id == "embedding_one"


def test_dense_index_accepts_media_only_document_without_text(tmp_path: Path) -> None:
    _require_zvec()
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        index.upsert([_document("embedding_media", "", (1.0, 0.0))])
        index.flush()

        assert index.search((1.0, 0.0), exact=True)[0].id == "embedding_media"


def test_dense_search_returns_one_embedding_per_parent_memory(tmp_path: Path) -> None:
    _require_zvec()
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        index.upsert(
            [
                _document("parent_best", "parent", (1.0, 0.0), memory_id="parent"),
                _document(
                    "parent_child",
                    "parent",
                    (0.99, 0.01),
                    memory_id="parent",
                    object_part=1,
                ),
                _document("neighbor", "neighbor", (0.0, 1.0), memory_id="neighbor"),
            ]
        )

        hits = index.search((1.0, 0.0), limit=2, exact=True)

    assert tuple(hit.id for hit in hits) == ("parent_best", "neighbor")


@pytest.mark.parametrize("quantization", tuple(IndexQuantization))
def test_quantized_vector_indexes_query_and_reopen(
    tmp_path: Path,
    quantization: IndexQuantization,
) -> None:
    _require_zvec()
    dimension = 64
    path = tmp_path / quantization.value
    documents = [
        _document(
            f"embedding_{row}",
            "target" if row == 0 else f"document {row}",
            tuple(1.0 if column == row % dimension else 0.0 for column in range(dimension)),
        )
        for row in range(64)
    ]
    expected_ids = {document.embedding.embedding_id for document in documents}
    with ZvecIndex(path, dimension=dimension, quantization=quantization) as index:
        index.upsert(documents)
        index.optimize()
        index.flush()
        hits = index.search(documents[0].embedding.values, limit=len(documents), exact=True)
        assert {hit.id for hit in hits} == expected_ids

    with ZvecIndex(path, dimension=dimension, quantization=quantization) as reopened:
        hits = reopened.search(documents[0].embedding.values, limit=len(documents), exact=True)
        assert {hit.id for hit in hits} == expected_ids


def test_full_text_search_stems_folds_accents_and_routes_chinese(tmp_path: Path) -> None:
    _require_zvec()
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        index.upsert(
            [
                _document("english", "Runners visited cafés", (1.0, 0.0)),
                _document("chinese", "今天在上海进行项目复盘", (0.0, 1.0)),
            ]
        )

        assert index.lexical_search("run cafe", limit=2)[0].id == "english"
        assert index.lexical_search("项目复盘", limit=2)[0].id == "chinese"


def test_schema_optimizes_time_ranges_and_automatic_maintenance(tmp_path: Path) -> None:
    _require_zvec()
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        fields = {field.name: field for field in cast(Any, index._schema).fields}
        assert fields["occurred_at"].index_param.enable_range_optimization
        assert fields["occurred_end"].index_param.enable_range_optimization

        index.upsert([_document("embedding", "content", (1.0, 0.0))])
        assert index.optimize_if_needed(minimum_unindexed=1) is True
        assert index.optimize_if_needed(minimum_unindexed=1) is False


def test_flush_maintenance_compacts_persisted_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_zvec()
    monkeypatch.setattr(zvec_index_module, "_AUTO_COMPACT_FLUSHES", 4)
    path = tmp_path / "index"
    with ZvecIndex(path, dimension=2) as index:
        for number in range(4):
            index.upsert(
                [
                    _document(
                        f"embedding_{number}",
                        "unique compacted target" if number == 3 else f"ordinary {number}",
                        (0.0, 1.0) if number == 3 else (1.0, 0.0),
                    )
                ]
            )
            index.flush()
            maintained = index.optimize_if_needed()

        assert maintained is True
        assert index.doc_count == 4
        assert len(tuple((path / "idmap.0").glob("*.sst"))) == 1

    with ZvecIndex(path, dimension=2) as reopened:
        assert reopened.doc_count == 4
        assert reopened.search((0.0, 1.0), limit=1, exact=True)[0].id == "embedding_3"
        assert reopened.lexical_search("unique compacted target", limit=1)[0].id == "embedding_3"


@pytest.mark.parametrize("failure_stage", ("validate", "stats"))
def test_compaction_restores_the_previous_collection_when_reopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _require_zvec()
    path = tmp_path / "index"
    with ZvecIndex(path, dimension=2) as index:
        index.upsert([_document("preserved", "preserved content", (0.0, 1.0))])
        index.flush()

        with monkeypatch.context() as injected:
            expected = f"replacement {failure_stage} failed"
            if failure_stage == "validate":

                def reject_replacement(_schema: object) -> NoReturn:
                    raise RuntimeError("replacement validation failed")

                injected.setattr(index, "_validate_schema", reject_replacement)
                expected = "replacement validation failed"
            else:

                def reject_replacement_stats(_stats: object) -> NoReturn:
                    raise RuntimeError("replacement stats failed")

                injected.setattr(
                    zvec_index_module,
                    "_indexed_document_count",
                    reject_replacement_stats,
                )
            with pytest.raises(RuntimeError, match=expected):
                index._compact()

        assert index.doc_count == 1
        assert index.search((0.0, 1.0), exact=True)[0].id == "preserved"

    with ZvecIndex(path, dimension=2) as reopened:
        assert reopened.lexical_search("preserved content", limit=1)[0].id == "preserved"


def test_write_refuses_to_consume_the_fd_safety_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = object.__new__(ZvecIndex)
    index.dimension = 2
    index._zvec = _FakeZvec
    collection = _CapturingCollection()
    index._collection = collection
    monkeypatch.setattr(zvec_index_module, "_file_descriptor_usage", lambda: (900, 1_024))

    with pytest.raises(OSError, match="before file descriptor exhaustion") as failure:
        index.upsert([_document("embedding", "content", (1.0, 0.0))])

    assert failure.value.errno == errno.EMFILE
    assert collection.documents == []


def test_type_and_event_time_filters_apply_to_every_search_mode(tmp_path: Path) -> None:
    _require_zvec()
    start = datetime(2026, 8, 17, tzinfo=timezone.utc)
    until = datetime(2026, 8, 24, tzinfo=timezone.utc)
    documents = [
        _document(
            "episodic_match",
            "project review",
            (1.0, 0.0),
            memory_type="episodic",
            occurred_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        ),
        _document(
            "episodic_late",
            "project review",
            (1.0, 0.0),
            memory_type="episodic",
            occurred_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
        ),
        _document(
            "episodic_overlap",
            "project review",
            (1.0, 0.0),
            memory_type="episodic",
            occurred_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            occurred_end=datetime(2026, 8, 18, tzinfo=timezone.utc),
        ),
        _document(
            "semantic_match",
            "project review",
            (1.0, 0.0),
            occurred_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        ),
    ]
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        index.upsert(documents)
        dense = index.search(
            (1.0, 0.0),
            memory_type="episodic",
            occurred_from=start,
            occurred_until=until,
            exact=True,
        )
        lexical = index.lexical_search(
            "project review",
            memory_type="episodic",
            occurred_from=start,
            occurred_until=until,
        )

    assert {hit.id for hit in dense} == {"episodic_match", "episodic_overlap"}
    assert {hit.id for hit in lexical} == {"episodic_match", "episodic_overlap"}


def test_open_rejects_an_incompatible_schema(tmp_path: Path) -> None:
    _require_zvec()
    path = tmp_path / "index"
    ZvecIndex(path, dimension=2).close()

    with pytest.raises(ValueError, match=r"schema mismatch.*rebuild"):
        ZvecIndex(path, dimension=3)


def test_rebuild_replaces_all_documents_and_delete_is_idempotent(tmp_path: Path) -> None:
    _require_zvec()
    with ZvecIndex(tmp_path / "index", dimension=2) as index:
        index.upsert([_document("embedding_old", "old text", (1.0, 0.0))])

        assert (
            index.rebuild(
                [_document("embedding_new", "new text", (0.0, 1.0))],
                batch_size=1,
            )
            == 1
        )
        assert index.doc_count == 1
        assert index._flushes_since_optimization == 1
        assert index._flushes_since_compaction == 1
        rebuilt = index.search((0.0, 1.0), exact=True)
        assert tuple(hit.id for hit in rebuilt) == ("embedding_new",)
        assert rebuilt[0].relevance == pytest.approx(1.0)

        index.delete(["missing", "embedding_new"])
        index.flush()
        assert index.doc_count == 0


def _document(
    embedding_id: str,
    content: str,
    values: tuple[float, ...],
    *,
    space_id: str = "default-space",
    task: str = "document",
    memory_type: str = "semantic",
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    object_part: int = 0,
    memory_id: str | None = None,
) -> IndexDocument:
    return IndexDocument(
        embedding=StoredEmbedding(
            embedding_id=embedding_id,
            memory_id=memory_id or f"memory_{embedding_id}",
            values=values,
            model_id="test-model",
            space_id=space_id,
            task=task,
            object_part=object_part,
            normalized=False,
            created_at=_NOW,
        ),
        content=content,
        metadata_json="{}",
        memory_type=memory_type,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
    )


def _require_zvec() -> None:
    pytest.importorskip("zvec", reason="Zvec 0.7 native wheel is not installed")


class _StatusCode(Enum):
    OK = 0
    NOT_FOUND = 1
    INTERNAL_ERROR = 2


class _Status:
    def __init__(self, code: _StatusCode, message: str = "") -> None:
        self._code = code
        self._message = message

    def ok(self) -> bool:
        return self._code is _StatusCode.OK

    def code(self) -> _StatusCode:
        return self._code

    def message(self) -> str:
        return self._message


class _FakeZvec:
    StatusCode = _StatusCode

    @staticmethod
    def Doc(**values: object) -> dict[str, object]:
        return values


class _FailingCollection:
    @staticmethod
    def upsert(_documents: Sequence[object]) -> list[_Status]:
        return [_Status(_StatusCode.INTERNAL_ERROR, "disk full")]


class _CapturingCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []

    def upsert(self, documents: Sequence[object]) -> list[_Status]:
        self.documents = cast(list[dict[str, object]], list(documents))
        return [_Status(_StatusCode.OK) for _document in documents]


class _DeletingCollection:
    @staticmethod
    def delete(_ids: Sequence[str]) -> list[_Status]:
        return [_Status(_StatusCode.NOT_FOUND)]


class _QueryDocument:
    def __init__(
        self,
        document_id: str,
        score: float,
        *,
        memory_id: str | None = None,
    ) -> None:
        self.id = document_id
        self.score = score
        self._memory_id = memory_id

    def field(self, name: str) -> str | None:
        assert name == "memory_id"
        return self._memory_id


class _Group:
    def __init__(self, value: str, docs: tuple[_QueryDocument, ...]) -> None:
        self.group_by_value = value
        self.docs = docs


class _GroupedCollection:
    def __init__(
        self,
        *,
        groups: tuple[_Group, ...],
        documents: tuple[_QueryDocument, ...],
    ) -> None:
        self.groups = groups
        self.documents = documents
        self.group_counts: list[int] = []
        self.topks: list[int] = []
        self.output_fields: list[list[str]] = []

    def group_by_query(self, **options: object) -> tuple[_Group, ...]:
        self.group_counts.append(cast(int, options["group_count"]))
        return self.groups

    def query(self, **options: object) -> list[_QueryDocument]:
        self.topks.append(cast(int, options["topk"]))
        self.output_fields.append(cast(list[str], options["output_fields"]))
        return list(self.documents)


class _LexicalCollection:
    def __init__(self, documents: list[_QueryDocument]) -> None:
        self.documents = documents
        self.options: dict[str, object] = {}

    def query(self, **options: object) -> list[_QueryDocument]:
        self.options = options
        return self.documents


class _QueryZvec:
    Query = dict
    Fts = dict


class _DenseZvec:
    Query = dict
    HnswQueryParam = dict
