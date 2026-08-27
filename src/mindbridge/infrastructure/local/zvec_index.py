"""Rebuildable Zvec search index for the local runtime."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, cast

from mindbridge.infrastructure.local.store import IndexDocument

_COLLECTION_NAME = "mindbridge_memory_index"
_CONTENT_FIELD = "content"
_SPACE_FIELD = "space_id"
_TASK_FIELD = "task"
_VECTOR_FIELD = "embedding"
_SCALAR_FIELDS = frozenset({_CONTENT_FIELD, _SPACE_FIELD, _TASK_FIELD})
_HNSW_M = 50
_HNSW_EF_CONSTRUCTION = 500
_DEFAULT_EF_SEARCH = 300
_DEFAULT_RRF_RANK_CONSTANT = 60
_DEFAULT_REBUILD_BATCH_SIZE = 1_024
_DEFAULT_HYBRID_CANDIDATES = 50


class ZvecUnavailableError(RuntimeError):
    """Raised when the optional native Zvec runtime cannot be imported."""


class ZvecWriteError(RuntimeError):
    """Raised when any operation in a Zvec write batch fails."""


@dataclass(frozen=True, slots=True)
class IndexHit:
    """One index-only result; SQLite must hydrate and authorize the ID."""

    id: str
    relevance: float
    """Higher is always better and the value is normalized to ``[0, 1]``."""


class ZvecIndex:
    """Own one FP32 cosine/FTS collection at a local filesystem path."""

    def __init__(
        self,
        path: str | Path,
        dimension: int,
        *,
        ef_search: int = _DEFAULT_EF_SEARCH,
        rrf_rank_constant: int = _DEFAULT_RRF_RANK_CONSTANT,
    ) -> None:
        if isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if isinstance(ef_search, bool) or ef_search <= 0:
            raise ValueError("ef_search must be a positive integer")
        if isinstance(rrf_rank_constant, bool) or rrf_rank_constant <= 0:
            raise ValueError("rrf_rank_constant must be a positive integer")

        self.path = Path(path).expanduser().resolve()
        self.dimension = dimension
        self.ef_search = ef_search
        self.rrf_rank_constant = rrf_rank_constant
        self._zvec: Any = _load_zvec()
        self._schema = self._build_schema()
        option = self._zvec.CollectionOption(read_only=False, enable_mmap=True)
        if self.path.exists():
            self._collection: object | None = self._zvec.open(str(self.path), option=option)
        else:
            self._collection = self._zvec.create_and_open(
                str(self.path),
                schema=self._schema,
                option=option,
            )
        try:
            self._validate_schema(cast(Any, self._collection).schema)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> ZvecIndex:
        self._require_collection()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    @property
    def doc_count(self) -> int:
        """Return the number of documents currently visible to Zvec."""
        collection = cast(Any, self._require_collection())
        return int(collection.stats.doc_count)

    @property
    def index_completeness(self) -> float:
        """Return the indexed fraction of the dense-vector field."""
        collection = cast(Any, self._require_collection())
        values = collection.stats.index_completeness
        return float(values[_VECTOR_FIELD])

    def upsert(self, documents: Sequence[IndexDocument]) -> None:
        """Idempotently apply a batch, checking every per-document status."""
        if not documents:
            return
        docs = []
        ids = []
        for document in documents:
            embedding = document.embedding
            self._validate_vector(embedding.values)
            _filter_literal(embedding.space_id, _SPACE_FIELD)
            _filter_literal(embedding.task, _TASK_FIELD)
            ids.append(embedding.embedding_id)
            docs.append(
                self._zvec.Doc(
                    id=embedding.embedding_id,
                    fields={
                        _CONTENT_FIELD: document.content,
                        _SPACE_FIELD: embedding.space_id,
                        _TASK_FIELD: embedding.task,
                    },
                    vectors={_VECTOR_FIELD: list(embedding.values)},
                )
            )
        collection = cast(Any, self._require_collection())
        statuses = cast(Sequence[object], collection.upsert(docs))
        self._check_statuses("upsert", ids, statuses)

    def delete(self, ids: Sequence[str]) -> None:
        """Idempotently delete IDs, accepting Zvec's NOT_FOUND status."""
        if not ids:
            return
        for document_id in ids:
            if not document_id or document_id != document_id.strip():
                raise ValueError("index document IDs must be non-empty and trimmed")
        collection = cast(Any, self._require_collection())
        statuses = cast(Sequence[object], collection.delete(list(ids)))
        self._check_statuses("delete", ids, statuses, allow_not_found=True)

    def search(
        self,
        values: Sequence[float],
        *,
        limit: int = 10,
        space_id: str | None = None,
        task: str | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]:
        """Return dense cosine matches with higher-is-better relevance."""
        docs = self._dense_query(
            values,
            limit=limit,
            space_id=space_id,
            task=task,
            ef=ef,
            exact=exact,
        )
        return tuple(
            IndexHit(
                id=_required_id(doc),
                relevance=max(0.0, min(1.0, 1.0 - _required_score(doc) / 2.0)),
            )
            for doc in docs
        )

    def hybrid_search(
        self,
        text: str,
        values: Sequence[float],
        *,
        limit: int = 10,
        candidate_limit: int | None = None,
        space_id: str | None = None,
        task: str | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]:
        """Fuse dense and BM25 ranks; return relevance normalized to ``[0, 1]``."""
        if not text.strip():
            raise ValueError("hybrid query text must not be empty")
        _require_positive(limit, "limit")
        candidates = (
            max(_DEFAULT_HYBRID_CANDIDATES, limit * 5)
            if candidate_limit is None
            else candidate_limit
        )
        _require_positive(candidates, "candidate_limit")
        if candidates < limit:
            raise ValueError("candidate_limit must not be smaller than limit")

        filter_expression = _filter_expression(space_id=space_id, task=task)
        dense = self._dense_query(
            values,
            limit=candidates,
            space_id=space_id,
            task=task,
            ef=ef,
            exact=exact,
        )
        collection = cast(Any, self._require_collection())
        lexical = collection.query(
            queries=self._zvec.Query(
                field_name=_CONTENT_FIELD,
                fts=self._zvec.Fts(match_string=text),
            ),
            topk=candidates,
            filter=filter_expression,
            include_vector=False,
            output_fields=[],
        )
        fused = self._zvec.RrfReRanker(rank_constant=self.rrf_rank_constant).rerank(
            [dense, lexical], topn=limit
        )
        maximum = 2.0 / (self.rrf_rank_constant + 1.0)
        return tuple(
            IndexHit(
                id=_required_id(doc),
                relevance=max(0.0, min(1.0, _required_score(doc) / maximum)),
            )
            for doc in fused
        )

    def flush(self) -> None:
        """Make prior Zvec writes durable before SQLite acknowledges its outbox."""
        collection = cast(Any, self._require_collection())
        collection.flush()

    def optimize(self, *, concurrency: int = 0) -> None:
        """Merge pending vectors into HNSW without blocking normal queries."""
        if isinstance(concurrency, bool) or concurrency < 0:
            raise ValueError("concurrency must be a non-negative integer")
        collection = cast(Any, self._require_collection())
        collection.optimize(self._zvec.OptimizeOption(concurrency=concurrency))

    def rebuild(
        self,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int = _DEFAULT_REBUILD_BATCH_SIZE,
        optimize_concurrency: int = 0,
    ) -> int:
        """Replace this disposable collection from authoritative SQLite documents."""
        _require_positive(batch_size, "batch_size")
        if isinstance(optimize_concurrency, bool) or optimize_concurrency < 0:
            raise ValueError("optimize_concurrency must be a non-negative integer")

        collection = cast(Any, self._require_collection())
        collection.destroy()
        self._collection = None
        self._collection = self._zvec.create_and_open(
            str(self.path),
            schema=self._schema,
            option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
        )

        count = 0
        batch: list[IndexDocument] = []
        for document in documents:
            batch.append(document)
            if len(batch) == batch_size:
                self.upsert(batch)
                count += len(batch)
                batch.clear()
        if batch:
            self.upsert(batch)
            count += len(batch)
        if count:
            self.optimize(concurrency=optimize_concurrency)
        self.flush()
        return count

    def close(self) -> None:
        """Close and release Zvec's native file lock; repeated calls are safe."""
        if self._collection is None:
            return
        collection = cast(Any, self._collection)
        self._collection = None
        collection.close()

    def _dense_query(
        self,
        values: Sequence[float],
        *,
        limit: int,
        space_id: str | None,
        task: str | None,
        ef: int | None,
        exact: bool,
    ) -> list[object]:
        self._validate_vector(values)
        _require_positive(limit, "limit")
        selected_ef = self.ef_search if ef is None else ef
        _require_positive(selected_ef, "ef")
        return cast(
            list[object],
            cast(Any, self._require_collection()).query(
                queries=self._zvec.Query(
                    field_name=_VECTOR_FIELD,
                    vector=list(values),
                    param=self._zvec.HnswQueryParam(
                        ef=max(selected_ef, limit),
                        is_linear=exact,
                    ),
                ),
                topk=limit,
                filter=_filter_expression(space_id=space_id, task=task),
                include_vector=False,
                output_fields=[],
            ),
        )

    def _build_schema(self) -> object:
        schema: object = self._zvec.CollectionSchema(
            name=_COLLECTION_NAME,
            fields=[
                self._zvec.FieldSchema(
                    name=_CONTENT_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                    index_param=self._zvec.FtsIndexParam(
                        tokenizer_name="standard",
                        filters=["lowercase"],
                    ),
                ),
                self._zvec.FieldSchema(
                    name=_SPACE_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                    index_param=self._zvec.InvertIndexParam(),
                ),
                self._zvec.FieldSchema(
                    name=_TASK_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                    index_param=self._zvec.InvertIndexParam(),
                ),
            ],
            vectors=[
                self._zvec.VectorSchema(
                    name=_VECTOR_FIELD,
                    data_type=self._zvec.DataType.VECTOR_FP32,
                    dimension=self.dimension,
                    index_param=self._zvec.HnswIndexParam(
                        metric_type=self._zvec.MetricType.COSINE,
                        m=_HNSW_M,
                        ef_construction=_HNSW_EF_CONSTRUCTION,
                    ),
                )
            ],
        )
        return schema

    def _validate_schema(self, schema: object) -> None:
        native_schema = cast(Any, schema)
        if native_schema.name != _COLLECTION_NAME:
            _schema_mismatch(f"collection name is {native_schema.name!r}")
        fields = {field.name: field for field in native_schema.fields}
        if fields.keys() != _SCALAR_FIELDS:
            _schema_mismatch("scalar fields differ")
        content = fields[_CONTENT_FIELD]
        if (
            content.data_type != self._zvec.DataType.STRING
            or content.nullable
            or content.index_param is None
            or content.index_param.type != self._zvec.IndexType.FTS
            or content.index_param.tokenizer_name != "standard"
            or tuple(content.index_param.filters) != ("lowercase",)
        ):
            _schema_mismatch("content FTS field differs")
        for name in (_SPACE_FIELD, _TASK_FIELD):
            field = fields[name]
            if (
                field.data_type != self._zvec.DataType.STRING
                or field.nullable
                or field.index_param is None
                or field.index_param.type != self._zvec.IndexType.INVERT
            ):
                _schema_mismatch(f"{name} filter field differs")

        vectors = {vector.name: vector for vector in native_schema.vectors}
        if vectors.keys() != {_VECTOR_FIELD}:
            _schema_mismatch("vector fields differ")
        vector = vectors[_VECTOR_FIELD]
        index = vector.index_param
        if (
            vector.data_type != self._zvec.DataType.VECTOR_FP32
            or vector.dimension != self.dimension
            or index.type != self._zvec.IndexType.HNSW
            or index.metric_type != self._zvec.MetricType.COSINE
            or index.m != _HNSW_M
            or index.ef_construction != _HNSW_EF_CONSTRUCTION
            or index.quantize_type != self._zvec.QuantizeType.UNDEFINED
        ):
            _schema_mismatch("FP32 cosine HNSW field differs")

    def _validate_vector(self, values: Sequence[float]) -> None:
        if len(values) != self.dimension:
            raise ValueError(f"vector must contain exactly {self.dimension} values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite")

    def _check_statuses(
        self,
        action: str,
        ids: Sequence[str],
        statuses: Sequence[object],
        *,
        allow_not_found: bool = False,
    ) -> None:
        if len(statuses) != len(ids):
            raise ZvecWriteError(
                f"Zvec {action} returned {len(statuses)} statuses for {len(ids)} documents"
            )
        failures = []
        for document_id, status in zip(ids, statuses, strict=True):
            native_status = cast(Any, status)
            if native_status.ok() or (
                allow_not_found and native_status.code() == self._zvec.StatusCode.NOT_FOUND
            ):
                continue
            failures.append(
                f"{document_id}: {native_status.code().name}: {native_status.message()}"
            )
        if failures:
            raise ZvecWriteError(f"Zvec {action} failed: {'; '.join(failures)}")

    def _require_collection(self) -> object:
        if self._collection is None:
            raise RuntimeError("Zvec index is closed")
        return self._collection


def _load_zvec() -> ModuleType:
    try:
        return import_module("zvec")
    except ImportError as error:
        raise ZvecUnavailableError(
            "Zvec 0.7 is required for local search; install a supported 64-bit zvec wheel"
        ) from error


def _filter_expression(*, space_id: str | None, task: str | None) -> str | None:
    clauses = []
    if space_id is not None:
        clauses.append(f"{_SPACE_FIELD} = {_filter_literal(space_id, _SPACE_FIELD)}")
    if task is not None:
        clauses.append(f"{_TASK_FIELD} = {_filter_literal(task, _TASK_FIELD)}")
    return " AND ".join(clauses) or None


def _filter_literal(value: str, name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    if "'" in value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains characters unsupported by Zvec filters")
    return f"'{value}'"


def _required_score(doc: object) -> float:
    native_doc = cast(Any, doc)
    if native_doc.score is None:
        raise RuntimeError(f"Zvec query result {native_doc.id!r} has no score")
    score = float(native_doc.score)
    if not math.isfinite(score):
        raise RuntimeError(f"Zvec query result {native_doc.id!r} has a non-finite score")
    return score


def _required_id(doc: object) -> str:
    document_id = cast(Any, doc).id
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError("Zvec query result has no document ID")
    return document_id


def _require_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _schema_mismatch(detail: str) -> NoReturn:
    raise ValueError(f"Zvec schema mismatch: {detail}; rebuild the disposable index")
