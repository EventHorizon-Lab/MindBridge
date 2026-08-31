"""Rebuildable Zvec search index for the local runtime."""

from __future__ import annotations

import errno
import math
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Condition
from types import ModuleType
from typing import Any, NoReturn, cast

from mindbridge.infrastructure.local.store import IndexDocument
from mindbridge.types import IndexQuantization

_COLLECTION_NAME = "mindbridge_memory_index"
_CONTENT_FIELD = "content"
_CONTENT_CJK_FIELD = "content_cjk"
_MEMORY_ID_FIELD = "memory_id"
_MEMORY_TYPE_FIELD = "memory_type"
_OCCURRED_AT_FIELD = "occurred_at"
_OCCURRED_END_FIELD = "occurred_end"
_SPACE_FIELD = "space_id"
_TASK_FIELD = "task"
_VECTOR_FIELD = "embedding"
_SCALAR_FIELDS = frozenset(
    {
        _CONTENT_FIELD,
        _CONTENT_CJK_FIELD,
        _MEMORY_ID_FIELD,
        _MEMORY_TYPE_FIELD,
        _OCCURRED_AT_FIELD,
        _OCCURRED_END_FIELD,
        _SPACE_FIELD,
        _TASK_FIELD,
    }
)
_MISSING_OCCURRED_AT = -(2**63)
_HNSW_M = 50
_HNSW_EF_CONSTRUCTION = 500
_RABITQ_TOTAL_BITS = 7
_RABITQ_NUM_CLUSTERS = 16
_DEFAULT_EF_SEARCH = 300
_LEXICAL_RANK_CONSTANT = 60
_DEFAULT_REBUILD_BATCH_SIZE = 1_024
_AUTO_OPTIMIZE_UNINDEXED_DOCUMENTS = 100_000
_AUTO_OPTIMIZE_FLUSHES = 64
_AUTO_COMPACT_FLUSHES = 256
_FILE_DESCRIPTOR_RESERVE = 128
_GROUP_OVERSAMPLE = 2
_GROUP_FALLBACK_MINIMUM = 50
_MAX_EMBEDDINGS_PER_MEMORY = 129
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff]")


class ZvecUnavailableError(RuntimeError):
    """Raised when the optional native Zvec runtime cannot be imported."""


class ZvecWriteError(RuntimeError):
    """Raised when any operation in a Zvec write batch fails."""


def validate_index_configuration(
    dimension: int,
    quantization: IndexQuantization,
) -> None:
    """Validate options before either SQLite metadata or Zvec can be mutated."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(quantization, IndexQuantization):
        raise ValueError("quantization must be an IndexQuantization value")
    if quantization is IndexQuantization.RABITQ and not 64 <= dimension <= 4_095:
        raise ValueError("RABITQ requires an embedding dimension between 64 and 4095")


class _CollectionGate:
    """Let queries overlap while collection replacement remains exclusive."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class IndexHit:
    """One index-only result; SQLite must hydrate and authorize the ID."""

    id: str
    relevance: float
    """Higher is always better and the value is normalized to ``[0, 1]``."""
    confidence: float | None = None
    """Calibratable dense evidence strength, independent of rank fusion."""
    lexical_match: bool = False

    def __post_init__(self) -> None:
        if not self.id or self.id != self.id.strip():
            raise ValueError("index hit ID must be non-empty and trimmed")
        for value, name in ((self.relevance, "relevance"), (self.confidence, "confidence")):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"index hit {name} must be between zero and one")
        if not isinstance(self.lexical_match, bool):
            raise ValueError("index hit lexical_match must be a boolean")
        if self.confidence is None:
            object.__setattr__(self, "confidence", self.relevance)


class ZvecIndex:
    """Own one cosine/FTS collection derived from authoritative FP32 vectors."""

    def __init__(
        self,
        path: str | Path,
        dimension: int,
        *,
        ef_search: int = _DEFAULT_EF_SEARCH,
        quantization: IndexQuantization = IndexQuantization.NONE,
    ) -> None:
        validate_index_configuration(dimension, quantization)
        if isinstance(ef_search, bool) or ef_search <= 0:
            raise ValueError("ef_search must be a positive integer")

        self.path = Path(path).expanduser().resolve()
        self.dimension = dimension
        self.ef_search = ef_search
        self.quantization = quantization
        self._gate = _CollectionGate()
        self._zvec: Any = _load_zvec()
        self._schema = self._build_schema()
        _require_file_descriptor_headroom(_persisted_index_file_count(self.path))
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
            self._optimization_watermark = _indexed_document_count(
                cast(Any, self._collection).stats
            )
            persisted_segments = _persisted_segment_count(self.path)
            self._flushes_since_optimization = persisted_segments
            self._flushes_since_compaction = persisted_segments
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
        _require_file_descriptor_headroom()
        docs = []
        ids = []
        for document in documents:
            embedding = document.embedding
            self._validate_vector(embedding.values)
            _filter_literal(embedding.space_id, _SPACE_FIELD)
            _filter_literal(embedding.task, _TASK_FIELD)
            _filter_literal(document.memory_type, _MEMORY_TYPE_FIELD)
            ids.append(embedding.embedding_id)
            docs.append(
                self._zvec.Doc(
                    id=embedding.embedding_id,
                    fields={
                        # Only the aggregate vector participates in BM25. Repeating one memory's
                        # text for every part would let multi-vector records crowd out neighbors.
                        _CONTENT_FIELD: document.content if embedding.object_part == 0 else "",
                        _CONTENT_CJK_FIELD: (
                            document.content
                            if embedding.object_part == 0
                            and _CJK_CHARACTER.search(document.content) is not None
                            else ""
                        ),
                        _MEMORY_ID_FIELD: embedding.memory_id,
                        _MEMORY_TYPE_FIELD: document.memory_type,
                        _OCCURRED_AT_FIELD: _timestamp(document.occurred_at),
                        _OCCURRED_END_FIELD: _end_timestamp(
                            document.occurred_at,
                            document.occurred_end,
                        ),
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
        _require_file_descriptor_headroom()
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
        memory_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        ef: int | None = None,
        exact: bool = False,
    ) -> tuple[IndexHit, ...]:
        """Return nonnegative cosine relevance and separately rescaled confidence."""
        with self._gate.read():
            docs = self._dense_query(
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
        hits = tuple((_required_id(doc), _required_score(doc)) for doc in docs)
        return tuple(
            IndexHit(
                id=document_id,
                relevance=max(0.0, min(1.0, 1.0 - distance)),
                confidence=max(0.0, min(1.0, 1.0 - distance / 2.0)),
            )
            for document_id, distance in hits
        )

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
        """Return BM25 matches with scores normalized against the best candidate."""
        if not text.strip():
            raise ValueError("lexical query text must not be empty")
        _require_positive(limit, "limit")
        with self._gate.read():
            docs = self._lexical_query(
                text,
                limit=limit,
                space_id=space_id,
                task=task,
                memory_type=memory_type,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
            )
        scores = tuple(max(0.0, _required_score(doc)) for doc in docs)
        maximum = max(scores, default=0.0)
        return tuple(
            IndexHit(
                id=_required_id(doc),
                relevance=(
                    score / maximum
                    if maximum > 0.0
                    else (_LEXICAL_RANK_CONSTANT + 1) / (_LEXICAL_RANK_CONSTANT + rank)
                ),
                confidence=0.0,
                lexical_match=True,
            )
            for rank, (doc, score) in enumerate(zip(docs, scores, strict=True), start=1)
        )

    def flush(self) -> None:
        """Make prior Zvec writes durable before SQLite acknowledges its outbox."""
        _require_file_descriptor_headroom()
        collection = cast(Any, self._require_collection())
        collection.flush()
        self._flushes_since_optimization += 1
        self._flushes_since_compaction += 1

    def optimize(self, *, concurrency: int = 0) -> None:
        """Merge pending vectors into HNSW without blocking normal queries."""
        if isinstance(concurrency, bool) or concurrency < 0:
            raise ValueError("concurrency must be a non-negative integer")
        collection = cast(Any, self._require_collection())
        collection.optimize(self._zvec.OptimizeOption(concurrency=concurrency))
        self._optimization_watermark = self.doc_count
        self._flushes_since_optimization = 0

    def optimize_if_needed(
        self,
        *,
        minimum_unindexed: int = _AUTO_OPTIMIZE_UNINDEXED_DOCUMENTS,
    ) -> bool:
        """Run maintenance after a meaningful flat-buffer or durable-segment buildup."""
        _require_positive(minimum_unindexed, "minimum_unindexed")
        # Zvec optimize compacts search segments but leaves one idmap SST per durable flush.
        if self._flushes_since_compaction >= _AUTO_COMPACT_FLUSHES:
            self._compact()
            return True
        collection = cast(Any, self._require_collection())
        stats = collection.stats
        document_count = int(stats.doc_count)
        self._optimization_watermark = min(self._optimization_watermark, document_count)
        indexed = _indexed_document_count(stats)
        if (
            self._flushes_since_optimization < _AUTO_OPTIMIZE_FLUSHES
            and document_count - max(indexed, self._optimization_watermark) < minimum_unindexed
        ):
            return False
        self.optimize()
        return True

    def _compact(self) -> None:
        """Copy visible documents into one flushed collection, then atomically replace it."""
        _require_file_descriptor_headroom()
        with TemporaryDirectory(
            prefix=f".{self.path.name}.compact-",
            dir=self.path.parent,
            ignore_cleanup_errors=True,
        ) as workspace:
            temporary = Path(workspace) / "new"
            backup = Path(workspace) / "old"
            target: Any | None = None
            try:
                target = self._zvec.create_and_open(
                    str(temporary),
                    schema=self._schema,
                    option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                with self._gate.read():
                    source = cast(Any, self._require_collection())
                    batch = []
                    with source.iter_docs() as documents:
                        for document in documents:
                            batch.append(document)
                            if len(batch) == _DEFAULT_REBUILD_BATCH_SIZE:
                                self._upsert_native(target, batch, action="compact")
                                batch.clear()
                    if batch:
                        self._upsert_native(target, batch, action="compact")
                if int(target.stats.doc_count):
                    target.optimize(self._zvec.OptimizeOption(concurrency=0))
                target.flush()
                target.close()
                target = None
                self._replace_compacted_collection(source, temporary, backup)
            finally:
                if target is not None:
                    with suppress(Exception):
                        target.close()

    def _replace_compacted_collection(
        self,
        source: object,
        temporary: Path,
        backup: Path,
    ) -> None:
        with self._gate.write():
            if self._collection is not source:
                raise RuntimeError("Zvec collection changed during compaction")
            cast(Any, source).close()
            self._collection = None
            # If the process stops between renames, the missing canonical path makes the
            # authoritative SQLite store rebuild this disposable index on the next open.
            try:
                self.path.replace(backup)
            except BaseException:
                self._collection = self._zvec.open(
                    str(self.path),
                    option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                raise
            try:
                temporary.replace(self.path)
            except BaseException:
                backup.replace(self.path)
                self._collection = self._zvec.open(
                    str(self.path),
                    option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                raise
            replacement: Any | None = None
            try:
                replacement = self._zvec.open(
                    str(self.path),
                    option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                self._validate_schema(replacement.schema)
                replacement_watermark = _indexed_document_count(replacement.stats)
            except BaseException:
                if replacement is not None:
                    with suppress(Exception):
                        replacement.close()
                self.path.replace(temporary)
                backup.replace(self.path)
                self._collection = self._zvec.open(
                    str(self.path),
                    option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                raise
            self._collection = replacement
            self._optimization_watermark = replacement_watermark
            self._flushes_since_optimization = 0
            self._flushes_since_compaction = 0

    def _upsert_native(
        self, collection: object, documents: Sequence[object], *, action: str
    ) -> None:
        ids = [str(cast(Any, document).id) for document in documents]
        statuses = cast(Sequence[object], cast(Any, collection).upsert(list(documents)))
        self._check_statuses(action, ids, statuses)

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

        with self._gate.write():
            return self._rebuild(
                documents,
                batch_size=batch_size,
                optimize_concurrency=optimize_concurrency,
            )

    def _rebuild(
        self,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int,
        optimize_concurrency: int,
    ) -> int:
        collection = cast(Any, self._require_collection())
        collection.destroy()
        self._collection = None
        self._collection = self._zvec.create_and_open(
            str(self.path),
            schema=self._schema,
            option=self._zvec.CollectionOption(read_only=False, enable_mmap=True),
        )
        self._optimization_watermark = 0
        self._flushes_since_optimization = 0
        self._flushes_since_compaction = 0

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
        with self._gate.write():
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
        memory_type: str | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        ef: int | None,
        exact: bool,
    ) -> list[object]:
        self._validate_vector(values)
        _require_positive(limit, "limit")
        selected_ef = self.ef_search if ef is None else ef
        _require_positive(selected_ef, "ef")
        collection = cast(Any, self._require_collection())
        query = self._zvec.Query(
            field_name=_VECTOR_FIELD,
            vector=list(values),
            param=self._query_param(ef=max(selected_ef, limit), exact=exact),
        )
        filter_expression = _filter_expression(
            space_id=space_id,
            task=task,
            memory_type=memory_type,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
        )
        groups = collection.group_by_query(
            query=query,
            group_by_field_name=_MEMORY_ID_FIELD,
            group_count=limit * _GROUP_OVERSAMPLE,
            topk_per_group=1,
            filter=filter_expression,
            include_vector=False,
            output_fields=[],
        )
        by_memory = {str(group.group_by_value): group.docs[0] for group in groups if group.docs}
        if len(by_memory) >= limit:
            return _best_documents(by_memory.values(), limit)

        # Zvec documents group-by as best effort. Fill missing groups from progressively wider
        # ordinary results, bounded by MindBridge's maximum vectors per parent memory.
        topk = max(_GROUP_FALLBACK_MINIMUM, limit * 4)
        ceiling = max(topk, limit * _MAX_EMBEDDINGS_PER_MEMORY)
        while len(by_memory) < limit:
            docs = cast(
                list[object],
                collection.query(
                    queries=query,
                    topk=topk,
                    filter=filter_expression,
                    include_vector=False,
                    output_fields=[_MEMORY_ID_FIELD],
                ),
            )
            for doc in docs:
                memory_id = _required_field(doc, _MEMORY_ID_FIELD)
                current = by_memory.get(memory_id)
                if current is None or _required_score(doc) < _required_score(current):
                    by_memory[memory_id] = doc
            if len(docs) < topk or topk >= ceiling:
                break
            topk = min(topk * 2, ceiling)
        return _best_documents(by_memory.values(), limit)

    def _lexical_query(
        self,
        text: str,
        *,
        limit: int,
        space_id: str | None,
        task: str | None,
        memory_type: str | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
    ) -> list[object]:
        return cast(
            list[object],
            cast(Any, self._require_collection()).query(
                queries=self._zvec.Query(
                    field_name=(
                        _CONTENT_CJK_FIELD
                        if _CJK_CHARACTER.search(text) is not None
                        else _CONTENT_FIELD
                    ),
                    fts=self._zvec.Fts(match_string=text),
                ),
                topk=limit,
                filter=_filter_expression(
                    space_id=space_id,
                    task=task,
                    memory_type=memory_type,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                ),
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
                        filters=["lowercase", "ascii_folding", "stemmer"],
                    ),
                ),
                self._zvec.FieldSchema(
                    name=_CONTENT_CJK_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                    index_param=self._zvec.FtsIndexParam(
                        tokenizer_name="jieba",
                        filters=["lowercase"],
                    ),
                ),
                self._zvec.FieldSchema(
                    name=_MEMORY_ID_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                ),
                self._zvec.FieldSchema(
                    name=_MEMORY_TYPE_FIELD,
                    data_type=self._zvec.DataType.STRING,
                    nullable=False,
                    index_param=self._zvec.InvertIndexParam(),
                ),
                self._zvec.FieldSchema(
                    name=_OCCURRED_AT_FIELD,
                    data_type=self._zvec.DataType.INT64,
                    nullable=False,
                    index_param=self._zvec.InvertIndexParam(enable_range_optimization=True),
                ),
                self._zvec.FieldSchema(
                    name=_OCCURRED_END_FIELD,
                    data_type=self._zvec.DataType.INT64,
                    nullable=False,
                    index_param=self._zvec.InvertIndexParam(enable_range_optimization=True),
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
                    index_param=self._vector_index_param(),
                )
            ],
        )
        return schema

    def _validate_schema(self, schema: object) -> None:  # noqa: C901 - one persisted schema
        native_schema = cast(Any, schema)
        if native_schema.name != _COLLECTION_NAME:
            _schema_mismatch(f"collection name is {native_schema.name!r}")
        fields = {field.name: field for field in native_schema.fields}
        if fields.keys() != _SCALAR_FIELDS:
            _schema_mismatch("scalar fields differ")
        for name, tokenizer, filters in (
            (_CONTENT_FIELD, "standard", ("lowercase", "ascii_folding", "stemmer")),
            (_CONTENT_CJK_FIELD, "jieba", ("lowercase",)),
        ):
            content = fields[name]
            if (
                content.data_type != self._zvec.DataType.STRING
                or content.nullable
                or content.index_param is None
                or content.index_param.type != self._zvec.IndexType.FTS
                or content.index_param.tokenizer_name != tokenizer
                or tuple(content.index_param.filters) != filters
            ):
                _schema_mismatch(f"{name} FTS field differs")
        for name in (_MEMORY_ID_FIELD, _MEMORY_TYPE_FIELD, _SPACE_FIELD, _TASK_FIELD):
            field = fields[name]
            if (
                field.data_type != self._zvec.DataType.STRING
                or field.nullable
                or (name == _MEMORY_ID_FIELD and field.index_param is not None)
                or (
                    name != _MEMORY_ID_FIELD
                    and (
                        field.index_param is None
                        or field.index_param.type != self._zvec.IndexType.INVERT
                    )
                )
            ):
                _schema_mismatch(f"{name} scalar field differs")
        for name in (_OCCURRED_AT_FIELD, _OCCURRED_END_FIELD):
            field = fields[name]
            if (
                field.data_type != self._zvec.DataType.INT64
                or field.nullable
                or field.index_param is None
                or field.index_param.type != self._zvec.IndexType.INVERT
                or not field.index_param.enable_range_optimization
            ):
                _schema_mismatch(f"{name} filter field differs")

        vectors = {vector.name: vector for vector in native_schema.vectors}
        if vectors.keys() != {_VECTOR_FIELD}:
            _schema_mismatch("vector fields differ")
        vector = vectors[_VECTOR_FIELD]
        index = vector.index_param
        expected_type = (
            self._zvec.IndexType.HNSW_RABITQ
            if self.quantization is IndexQuantization.RABITQ
            else self._zvec.IndexType.HNSW
        )
        expected_quantization = {
            IndexQuantization.NONE: self._zvec.QuantizeType.UNDEFINED,
            IndexQuantization.FP16: self._zvec.QuantizeType.FP16,
            IndexQuantization.INT8: self._zvec.QuantizeType.INT8,
            IndexQuantization.RABITQ: self._zvec.QuantizeType.RABITQ,
        }[self.quantization]
        if (
            vector.data_type != self._zvec.DataType.VECTOR_FP32
            or vector.dimension != self.dimension
            or index.type != expected_type
            or index.metric_type != self._zvec.MetricType.COSINE
            or index.m != _HNSW_M
            or index.ef_construction != _HNSW_EF_CONSTRUCTION
            or index.quantize_type != expected_quantization
            or (
                self.quantization is IndexQuantization.INT8
                and not index.quantizer_param.enable_rotate
            )
            or (
                self.quantization is IndexQuantization.RABITQ
                and (
                    index.total_bits != _RABITQ_TOTAL_BITS
                    or index.num_clusters != _RABITQ_NUM_CLUSTERS
                    or index.sample_count != 0
                )
            )
        ):
            _schema_mismatch("FP32 cosine vector index differs")

    def _vector_index_param(self) -> object:
        if self.quantization is IndexQuantization.RABITQ:
            return self._zvec.HnswRabitqIndexParam(
                metric_type=self._zvec.MetricType.COSINE,
                m=_HNSW_M,
                ef_construction=_HNSW_EF_CONSTRUCTION,
                total_bits=_RABITQ_TOTAL_BITS,
                num_clusters=_RABITQ_NUM_CLUSTERS,
            )
        quantize_type = {
            IndexQuantization.NONE: self._zvec.QuantizeType.UNDEFINED,
            IndexQuantization.FP16: self._zvec.QuantizeType.FP16,
            IndexQuantization.INT8: self._zvec.QuantizeType.INT8,
        }[self.quantization]
        return self._zvec.HnswIndexParam(
            metric_type=self._zvec.MetricType.COSINE,
            m=_HNSW_M,
            ef_construction=_HNSW_EF_CONSTRUCTION,
            quantize_type=quantize_type,
            quantizer_param=self._zvec.QuantizerParam(
                enable_rotate=self.quantization is IndexQuantization.INT8
            ),
        )

    def _query_param(self, *, ef: int, exact: bool) -> object:
        param_type = (
            self._zvec.HnswRabitqQueryParam
            if self.quantization is IndexQuantization.RABITQ
            else self._zvec.HnswQueryParam
        )
        return param_type(ef=ef, is_linear=exact)

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


def _require_file_descriptor_headroom(required: int = 0) -> None:
    usage = _file_descriptor_usage()
    if usage is None:
        return
    opened, soft_limit = usage
    if opened + required + _FILE_DESCRIPTOR_RESERVE <= soft_limit:
        return
    raise OSError(
        errno.EMFILE,
        "Zvec operation refused before file descriptor exhaustion: "
        f"{opened} descriptors are open, {required} persisted index files may be opened, "
        f"and the soft limit is {soft_limit}; rebuild the disposable index to compact segments",
    )


def _file_descriptor_usage() -> tuple[int, int] | None:
    descriptors = next(
        (path for path in (Path("/proc/self/fd"), Path("/dev/fd")) if path.is_dir()),
        None,
    )
    if os.name != "posix" or descriptors is None:
        return None
    try:
        resource = import_module("resource")
        soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if int(soft_limit) < 0:
            return None
        return len(os.listdir(descriptors)), int(soft_limit)
    except (AttributeError, ImportError, OSError):
        return None


def _persisted_index_file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        file.suffix in {".ipc", ".proxima", ".sst", ".wal"}
        for file in path.rglob("*")
        if file.is_file()
    )


def _persisted_segment_count(path: Path) -> int:
    if not path.exists():
        return 0
    return max(
        sum(1 for _file in path.rglob("*.ipc")),
        sum(1 for _file in path.rglob("*.proxima")),
        sum(1 for _file in (path / "idmap.0").glob("*.sst")),
    )


def _filter_expression(
    *,
    space_id: str | None = None,
    task: str | None = None,
    memory_type: str | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
) -> str | None:
    clauses = []
    if space_id is not None:
        clauses.append(f"{_SPACE_FIELD} = {_filter_literal(space_id, _SPACE_FIELD)}")
    if task is not None:
        clauses.append(f"{_TASK_FIELD} = {_filter_literal(task, _TASK_FIELD)}")
    if memory_type is not None:
        clauses.append(f"{_MEMORY_TYPE_FIELD} = {_filter_literal(memory_type, _MEMORY_TYPE_FIELD)}")
    if occurred_from is not None or occurred_until is not None:
        clauses.append(f"{_OCCURRED_AT_FIELD} > {_MISSING_OCCURRED_AT}")
    if occurred_from is not None:
        clauses.append(f"{_OCCURRED_END_FIELD} > {_timestamp(occurred_from)}")
    if occurred_until is not None:
        clauses.append(f"{_OCCURRED_AT_FIELD} < {_timestamp(occurred_until)}")
    if occurred_from is not None and occurred_until is not None and occurred_until <= occurred_from:
        raise ValueError("occurred_until must be later than occurred_from")
    return " AND ".join(clauses) or None


def _filter_literal(value: str, name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    if "'" in value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains characters unsupported by Zvec filters")
    return f"'{value}'"


def _timestamp(value: datetime | None) -> int:
    if value is None:
        return _MISSING_OCCURRED_AT
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at filters must include a timezone")
    delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _end_timestamp(start: datetime | None, end: datetime | None) -> int:
    if end is not None:
        return _timestamp(end)
    timestamp = _timestamp(start)
    return timestamp if timestamp == _MISSING_OCCURRED_AT else timestamp + 1


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


def _required_field(doc: object, name: str) -> str:
    value = cast(Any, doc).field(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Zvec query result {_required_id(doc)!r} has no {name}")
    return value


def _best_documents(documents: Iterable[object], limit: int) -> list[object]:
    return sorted(documents, key=lambda doc: (_required_score(doc), _required_id(doc)))[:limit]


def _indexed_document_count(stats: object) -> int:
    native_stats = cast(Any, stats)
    document_count = int(native_stats.doc_count)
    completeness = float(native_stats.index_completeness[_VECTOR_FIELD])
    if not math.isfinite(completeness) or not 0.0 <= completeness <= 1.0:
        raise RuntimeError("Zvec index completeness must be between zero and one")
    return round(document_count * completeness)


def _require_positive(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _schema_mismatch(detail: str) -> NoReturn:
    raise ValueError(f"Zvec schema mismatch: {detail}; rebuild the disposable index")
