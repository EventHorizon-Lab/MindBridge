"""Focused checks for the disposable local Zvec index."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import NoReturn, cast

import pytest

from mindbridge.infrastructure.local.store import IndexDocument, StoredEmbedding
from mindbridge.infrastructure.local.zvec_index import (
    IndexHit,
    ZvecIndex,
    ZvecUnavailableError,
    ZvecWriteError,
)

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


def test_every_batch_status_is_checked() -> None:
    index = object.__new__(ZvecIndex)
    index.dimension = 2
    index._zvec = _FakeZvec
    index._collection = _FailingCollection()

    with pytest.raises(ZvecWriteError, match="embedding_bad: INTERNAL_ERROR: disk full"):
        index.upsert([_document("embedding_bad", "content", (1.0, 0.0))])


def test_delete_accepts_only_not_found_failures() -> None:
    index = object.__new__(ZvecIndex)
    index._zvec = _FakeZvec
    index._collection = _DeletingCollection()

    index.delete(["already_deleted"])


def test_hybrid_search_uses_a_broader_default_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_limits: list[int] = []
    collection = _QueryCollection()
    index = object.__new__(ZvecIndex)
    index._zvec = _HybridZvec
    index._collection = collection
    index.rrf_rank_constant = 60

    def dense_query(
        _values: Sequence[float],
        *,
        limit: int,
        **_options: object,
    ) -> list[object]:
        dense_limits.append(limit)
        return []

    monkeypatch.setattr(index, "_dense_query", dense_query)

    assert index.hybrid_search("shared result", (1.0, 0.0), limit=10) == ()
    assert dense_limits == [50]
    assert collection.topks == [50]


def test_create_search_hybrid_flush_close_and_reopen(tmp_path: Path) -> None:
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

    hybrid = index.hybrid_search(
        "machine learning",
        (1.0, 0.0),
        limit=2,
        space_id="default-space",
        task="document",
        exact=True,
    )
    assert hybrid[0].id == "embedding_one"
    assert hybrid[0].relevance == pytest.approx(1.0)
    assert all(0.0 <= hit.relevance <= 1.0 for hit in hybrid)

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


def test_type_and_event_time_filters_apply_to_dense_and_hybrid_search(tmp_path: Path) -> None:
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
        hybrid = index.hybrid_search(
            "project review",
            (1.0, 0.0),
            memory_type="episodic",
            occurred_from=start,
            occurred_until=until,
            exact=True,
        )

    assert tuple(hit.id for hit in dense) == ("episodic_match",)
    assert tuple(hit.id for hit in hybrid) == ("episodic_match",)


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
) -> IndexDocument:
    return IndexDocument(
        embedding=StoredEmbedding(
            embedding_id=embedding_id,
            memory_id=f"memory_{embedding_id}",
            values=values,
            model_id="test-model",
            space_id=space_id,
            task=task,
            normalized=False,
            created_at=_NOW,
        ),
        content=content,
        metadata_json="{}",
        memory_type=memory_type,
        occurred_at=occurred_at,
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


class _DeletingCollection:
    @staticmethod
    def delete(_ids: Sequence[str]) -> list[_Status]:
        return [_Status(_StatusCode.NOT_FOUND)]


class _QueryCollection:
    def __init__(self) -> None:
        self.topks: list[int] = []

    def query(self, **options: object) -> list[object]:
        self.topks.append(cast(int, options["topk"]))
        return []


class _HybridZvec:
    Query = dict
    Fts = dict

    class RrfReRanker:
        def __init__(self, *, rank_constant: int) -> None:
            assert rank_constant == 60

        @staticmethod
        def rerank(_rankings: object, *, topn: int) -> list[object]:
            assert topn == 10
            return []
