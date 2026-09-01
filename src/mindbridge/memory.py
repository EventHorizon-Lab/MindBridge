"""Developer-facing local memory API."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from collections import Counter, deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import copy_context
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Condition, RLock
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    EMBEDDING_PARTS_ELIDED,
    IDENTITY_MATCHED,
    IDENTITY_OBSERVATIONS,
    MODEL_MODULE,
    MODEL_TTFT,
    SPAN_KIND,
    TRACER_NAME,
    current_model_request_count,
    mark_model_requests,
    model_span,
    operation_span,
    traced_span,
)
from mindbridge.configuration import MindBridgeConfig, resolve_memory_config
from mindbridge.exceptions import (
    IdentityNotFoundError,
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)
from mindbridge.infrastructure.local._lock import DataDirectoryInUseError
from mindbridge.infrastructure.local.assets import (
    AssetStore,
    AssetStoreError,
    AssetTooLargeError,
)
from mindbridge.infrastructure.local.store import (
    IndexDocument,
    LocalStore,
    SpeechRollback,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.zvec_index import (
    IndexHit,
    ZvecIndex,
    validate_index_configuration,
)
from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    FaceAnalysis,
    FaceBackend,
    GenerationBackend,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
    StreamingGenerationBackend,
    TranscriptionBackend,
    _modalities,
)
from mindbridge.plugins import MemoryConfig, MemoryPlugins
from mindbridge.types import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    Blob,
    ContentAtom,
    ContentInput,
    FaceObservation,
    IndexQuantization,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    PrefetchResult,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalTrace,
    SearchHit,
    SpeakerSegment,
    StreamInput,
    TracedSearchResult,
)

_DOCUMENT_TASK = EmbedTask.DOCUMENT.value
_INDEX_RECIPE_PREFIX = (
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:context-keys-v9"
)
_LEGACY_INDEX_RECIPES = frozenset(
    {
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:single-vector-v3",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:multi-vector-v4",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:multi-vector-v5",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:interval-filters:context-keys-v6",
        "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:grouped-range:context-keys-v7",
        *(
            "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:"
            f"context-keys-v8:quantization-{mode.value}"
            for mode in IndexQuantization
        ),
    }
)
_OUTBOX_BATCH_SIZE = 256
_REINDEX_PAGE_SIZE = 256
_REEMBED_PAGE_SIZE = 32
_RERANK_CANDIDATES = 100
_RANK_FLOOR = 0.3
_RANK_CEILING = 1.5
_DEFAULT_CONFIG = MemoryConfig()
# Confidence a full-text match earns on its own, which is what the `minimum_relevance` gate reads.
_LEXICAL_MATCH_CONFIDENCE = 0.6
_DECAY_REINFORCEMENT_LIMIT = 20
_CONFIRMATION_WEIGHT = 0.05
# What a full-text match contributes to the *ranking* score. It is deliberately far below the
# confidence above: `lexical_relevance_by_rank` is a reciprocal rank, not a similarity, so at 0.6
# the top full-text hit outranked every dense candidate under 0.6 cosine no matter how weak its
# match actually was. Measured on ranked candidates replayed from two benchmark corpora, gold
# recall at eight rose from 0.8239 to 0.8920 on Mem-Gallery and from 0.8194 to 0.9306 on
# ATM-Bench when the rank proxy stopped winning that comparison. The floor is still non-zero so a
# memory only the full-text route can reach stays orderable.
_LEXICAL_RANK_RELEVANCE = 0.24
# Covering every distinctive query term is evidence in a way that ranking first in the full-text
# index is not, so a complete match ranks as near-certain rather than through the demoted rank
# proxy, and a memory quoting the whole question cannot be buried by an unrelated dense neighbour.
# The threshold leaves slack for summation order; the coverage ratio is otherwise exactly one.
# Across both replayed corpora two candidates in more than twenty thousand reached full coverage,
# so this rescues the exact-phrase case without moving the measurement at all. The value sits
# between the two behaviours the search tests pin: after the coverage lift it must clear a
# mediocre dense neighbour and must still lose to strong semantic evidence, because a memory that
# merely echoes the question back is a complete term match and is not the answer.
_LEXICAL_FULL_COVERAGE = 0.999
_LEXICAL_FULL_COVERAGE_RELEVANCE = 0.75
# Weight on the IDF-weighted query-term coverage, which unlike the rank proxy is a normalised
# score, so it is the term that actually combines the two routes. It is applied as a lift toward
# one across the remaining headroom rather than as an addition, because a clamped sum turns every
# strong candidate into exactly 1.0 and loses the ordering among them.
_MAX_LEXICAL_RERANK_BONUS = 0.3
_NEGATION_WEIGHT = 6.0
_LEXICAL_NOISE_TERMS = frozenset(
    {
        "a",
        "an",
        "answer",
        "are",
        "based",
        "did",
        "do",
        "does",
        "how",
        "is",
        "memory",
        "memories",
        "only",
        "please",
        "question",
        "return",
        "the",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)
_NO_TRANSCRIPTION_SPACE = "none:asr-v1"
_NO_FACE_SPACE = "none:face-v1"
_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_LEXICAL_TERM = re.compile(r"\w+")
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_NEGATION_TERMS = frozenset(
    {"no", "not", "never", "neither", "nor", "without", "不", "没", "未", "无", "非", "别"}
)
_NEGATED_CONTRACTION = re.compile(r"n['\u2019]t\b", re.IGNORECASE)
_MAX_CONTENT_PARTS = 128
# Model span modules mapped onto the failure stage a caller sees.
_MODEL_STAGES = {
    "embedding": "embed",
    "face": "recognize",
    "generation": "generate",
    "transcription": "transcribe",
}
_MAX_TEXT_CHARACTERS = 65_536
_MAX_METADATA_BYTES = 262_144
_TEXT_KEY_CHARACTERS = 2_048
_TEXT_KEY_OVERLAP = 256
_TEXT_KEY_CONTEXT = 256
_MAX_RETRIEVAL_KEYS = 128
# Text-equivalent cost of one grounded media part, at the usual four characters per token. The
# modalities are an order of magnitude apart in what a model charges for them: an image part
# measured near five hundred tokens against this stack where a ten-second video part measured
# near three thousand, so one flat number would either starve text or overrun on video. These
# are coarse by design; the budget is the caller's knob, not these constants.
# The reason an embedding backend reports when media does not fit one inline request.
_PAYLOAD_TOO_LARGE = "payload_too_large"
_ASSET_EVIDENCE_CHARS: Mapping[Modality, int] = MappingProxyType(
    {
        Modality.IMAGE: 2_000,
        Modality.AUDIO: 4_000,
        Modality.VIDEO: 12_000,
    }
)
_DEFAULT_ASSET_EVIDENCE_CHARS = 4_000
_MAX_QUERY_RETRIEVAL_KEYS = 7
_MAX_INDEX_SEARCH_WORKERS = 4
_TODAY_ISO_DATE = re.compile(r"\btoday\s+is\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_TODAY_NAMED_DATE = re.compile(
    rf"\btoday\s+is\s+({_MONTH_NAME})"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?",
    re.IGNORECASE,
)
_NAMED_MONTH_YEAR = re.compile(rf"\b({_MONTH_NAME})\s+((?:19|20|21)\d{{2}})\b", re.IGNORECASE)
_CJK_YEAR_MONTH = re.compile(
    r"(?<![A-Za-z0-9_$-])((?:19|20|21)\d{2})年\s*(1[0-2]|0?[1-9])月"
    r"(?![A-Za-z0-9_$-])"
)
_CJK_CALENDAR_YEAR = re.compile(r"(?<![A-Za-z0-9_$-])((?:19|20|21)\d{2})年(?![A-Za-z0-9_$-])")
_CALENDAR_YEAR = re.compile(r"(?<![\w$-])((?:19|20|21)\d{2})(?![\w-])")
_MONTHS = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".avi": "video/x-msvideo",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".ogv": "video/ogg",
    ".opus": "audio/ogg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".wav": "audio/wav",
    ".weba": "audio/webm",
    ".webm": "video/webm",
    ".webp": "image/webp",
}
_T = TypeVar("_T")
_STORE_METADATA_KEYS = {
    "model": "embedding.model_id",
    "space": "embedding.space_id",
    "transcription": "transcription.space_id",
    "face": "face.space_id",
    "face_analysis": "face.analysis_space_id",
    "dimension": "embedding.dimension",
    "index": "index.recipe",
}


class _Index(Protocol):
    def upsert(self, documents: Sequence[IndexDocument]) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

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
    ) -> tuple[IndexHit, ...]: ...

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
    ) -> tuple[IndexHit, ...]: ...

    def flush(self) -> None: ...

    def optimize(self, *, concurrency: int = 0) -> None: ...

    def optimize_if_needed(self, *, minimum_unindexed: int = 100_000) -> bool: ...

    def rebuild(
        self,
        documents: Iterable[IndexDocument],
        *,
        batch_size: int = 1_024,
        optimize_concurrency: int = 0,
    ) -> int: ...

    def close(self) -> None: ...


class _Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _PreparedContent:
    text: str
    assets: tuple[StoredAsset, ...]
    modality: Modality
    canonical_parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PreparedMemory:
    memory_id: str
    content: _PreparedContent
    metadata_json: str
    occurred_at: datetime | None
    occurred_end: datetime | None
    memory_type: MemoryType


@dataclass(slots=True)
class _OperationAssets:
    leased: builtins.list[StoredAsset]
    cleanup: builtins.list[StoredAsset]
    persisted: set[str]
    transcripts: dict[str, str]
    transcript_updates: dict[str, str]
    speech_updates: dict[str, SpeechAnalysis]
    speech_segments: dict[str, tuple[SpeakerSegment, ...]]
    speech_rollbacks: builtins.list[SpeechRollback]
    face_observations: dict[str, tuple[FaceObservation, ...]]


@dataclass(frozen=True, slots=True)
class _IndexCandidates:
    dense: tuple[IndexHit, ...]
    lexical: tuple[IndexHit, ...]
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _SearchOutcome:
    hits: tuple[SearchHit, ...]
    trace: RetrievalTrace | None = None


class Memory:
    """Persist and retrieve native text, image, video, audio, and omni memories."""

    def __init__(
        self,
        data_dir: str | Path = ".mindbridge",
        *,
        embedder: EmbeddingBackend,
        answerer: GenerationBackend | None = None,
        transcriber: SpeechBackend | TranscriptionBackend | None = None,
        face_analyzer: FaceBackend | None = None,
        index_speech: bool = _DEFAULT_CONFIG.index_speech,
        index_quantization: IndexQuantization = _DEFAULT_CONFIG.index_quantization,
        minimum_relevance: float = _DEFAULT_CONFIG.minimum_relevance,
        ambiguity_margin: float = _DEFAULT_CONFIG.ambiguity_margin,
        evidence_budget_chars: int | None = _DEFAULT_CONFIG.evidence_budget_chars,
        decay_half_life_days: float | None = _DEFAULT_CONFIG.decay_half_life_days,
        speaker_similarity: float = _DEFAULT_CONFIG.speaker_similarity,
        speaker_margin: float = _DEFAULT_CONFIG.speaker_margin,
        face_similarity: float = _DEFAULT_CONFIG.face_similarity,
        face_margin: float = _DEFAULT_CONFIG.face_margin,
        tracer: Tracer | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._tracer = trace.get_tracer(TRACER_NAME) if tracer is None else tracer
        self._owner_pid = os.getpid()
        self._write_lock = RLock()
        self._lifecycle = Condition()
        self._active_operations = 0
        self._closing = False
        self._closed = True
        self._pending_asset_cleanup: dict[str, StoredAsset] = {}
        self._index_quantization = _index_quantization(index_quantization)
        self._index_recipe = _index_recipe(self._index_quantization)
        self._speaker_similarity = _unit_interval(speaker_similarity, "speaker_similarity")
        self._speaker_margin = _unit_interval(speaker_margin, "speaker_margin")
        self._face_similarity = _unit_interval(face_similarity, "face_similarity")
        self._face_margin = _unit_interval(face_margin, "face_margin")
        if not isinstance(index_speech, bool):
            raise ValidationError("index_speech must be a boolean")
        self._index_speech = index_speech
        self._minimum_relevance = _unit_interval(minimum_relevance, "minimum_relevance")
        self._ambiguity_margin = _unit_interval(ambiguity_margin, "ambiguity_margin")
        self._evidence_budget = _evidence_budget(evidence_budget_chars)
        self._decay_half_life = _decay_half_life(decay_half_life_days)

        self._store = _open_store(self.data_dir)
        self._embedder = embedder
        self._answerer = answerer
        self._transcriber = transcriber
        self._face_analyzer = face_analyzer

        try:
            (
                self._embedding_capabilities,
                self._embedding_model,
                self._space_id,
                self._embedding_dimension,
            ) = _embedding_contract(self._embedder)
            try:
                validate_index_configuration(
                    self._embedding_dimension,
                    self._index_quantization,
                )
            except ValueError as error:
                raise ValidationError(str(error)) from None
            self._generation_capabilities = _generation_contract(self._answerer)
            (
                self._transcription_capabilities,
                self._transcription_space,
            ) = _transcription_contract(self._transcriber)
            (
                self._face_capabilities,
                self._face_model,
                self._face_space,
                self._face_analysis_space,
            ) = _face_contract(self._face_analyzer)
            self._assets = AssetStore(self.data_dir)
            self._collect_orphan_assets(scan_physical=True)
            index_path = self.data_dir / "zvec"
            index_missing = not index_path.exists()
            index_rebuild, embedding_rebuild = self._ensure_store_metadata(index_path)
            if embedding_rebuild:
                self._reembed_memories()
                self._store.set_metadata(_STORE_METADATA_KEYS["space"], self._space_id)
                self._store.set_metadata(_STORE_METADATA_KEYS["index"], self._index_recipe)
            if index_missing or index_rebuild:
                with _translate_storage_errors("checkpoint a missing search index"):
                    self._store.queue_all_embeddings()
        except BaseException:
            self._close_models_and_store()
            raise

        try:
            self._index: _Index = ZvecIndex(
                index_path,
                dimension=self._embedding_dimension,
                quantization=self._index_quantization,
            )
        except Exception as error:
            self._close_models_and_store()
            raise IndexUnavailableError(
                "failed to open the local search index", stage="open"
            ) from error

        self._closed = False
        try:
            with self._write_lock:
                self._drain_outbox()
        except BaseException:
            self._closed = True
            self._close_resources()
            raise

    @classmethod
    def from_plugins(
        cls,
        data_dir: str | Path = ".mindbridge",
        *,
        plugins: MemoryPlugins,
        config: MemoryConfig | None = None,
        tracer: Tracer | None = None,
    ) -> Memory:
        """Open memory from an explicit capability bundle and local policy."""
        if not isinstance(plugins, MemoryPlugins):
            raise ValidationError("plugins must be a MemoryPlugins value")
        if config is None:
            config = MemoryConfig()
        elif not isinstance(config, MemoryConfig):
            raise ValidationError("config must be a MemoryConfig value")
        return cls(
            data_dir,
            embedder=plugins.embedder,
            answerer=plugins.answerer,
            transcriber=plugins.transcriber,
            face_analyzer=plugins.face_analyzer,
            index_speech=config.index_speech,
            index_quantization=config.index_quantization,
            minimum_relevance=config.minimum_relevance,
            ambiguity_margin=config.ambiguity_margin,
            evidence_budget_chars=config.evidence_budget_chars,
            decay_half_life_days=config.decay_half_life_days,
            speaker_similarity=config.speaker_similarity,
            speaker_margin=config.speaker_margin,
            face_similarity=config.face_similarity,
            face_margin=config.face_margin,
            tracer=tracer,
        )

    @classmethod
    def from_config(
        cls,
        config: MindBridgeConfig | Mapping[str, object],
        *,
        tracer: Tracer | None = None,
    ) -> Memory:
        """Open memory from validated declarative configuration."""
        resolved = resolve_memory_config(config)
        try:
            return cls.from_plugins(
                resolved.data_dir,
                plugins=resolved.plugins,
                config=resolved.settings,
                tracer=tracer,
            )
        except BaseException:
            resolved.close()
            raise

    def __enter__(self) -> Memory:
        self._require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def _trace(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> AbstractContextManager[Span]:
        values = dict(attributes or {})
        values[SPAN_KIND] = kind
        if kind == "operation":
            return operation_span(self._tracer, name, attributes=values)
        return _staged(
            traced_span(self._tracer, name, attributes=values),
            name.removeprefix("mindbridge."),
        )

    def _model_trace(
        self,
        module: str,
        operation: str,
        *,
        model: str | None,
        batch_size: int,
        modalities: Iterable[Modality],
    ) -> AbstractContextManager[Span]:
        attributes: dict[str, AttributeValue] = {
            MODEL_MODULE: module,
            "gen_ai.operation.name": operation,
            "mindbridge.model.batch_size": batch_size,
            "mindbridge.input.modalities": tuple(
                sorted({modality.value for modality in modalities})
            ),
        }
        if model is not None:
            attributes["gen_ai.request.model"] = model
        attributes[SPAN_KIND] = "model"
        return _staged(
            model_span(self._tracer, f"mindbridge.model.{module}", attributes=attributes),
            _MODEL_STAGES[module],
        )

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        """Add one native or mixed-modal memory and return its stable record."""
        with self._trace("mindbridge.add", kind="operation"), self._operation() as assets:
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared_content = self._prepare_content(content, assets)
            prepared = _prepare_memory(
                prepared_content,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
            )
            return self._add_prepared((prepared,), operation=assets)[0]

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]:
        """Add memories in one model batch and one SQLite transaction."""
        with self._trace("mindbridge.add_many", kind="operation"), self._operation() as assets:
            normalized_memory_type = _memory_type(memory_type)
            if isinstance(contents, (str, bytes, Path, Blob, AssetRef, Mapping)):
                raise ValidationError("contents must be a sequence of memory inputs")
            try:
                batch = tuple(contents)
            except TypeError:
                raise ValidationError("contents must be a sequence of memory inputs") from None
            occurrences = _batch_values(occurred_at, len(batch), "occurred_at")
            occurrence_ends = _batch_values(occurred_end, len(batch), "occurred_end")
            metadata_values = _batch_values(metadata, len(batch), "metadata")
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = tuple(
                    self._prepare_batch_item(
                        content,
                        index=index,
                        operation=assets,
                        occurred_at=event_time,
                        occurred_end=event_end,
                        metadata=item_metadata,
                        memory_type=normalized_memory_type,
                    )
                    for index, (content, event_time, event_end, item_metadata) in enumerate(
                        zip(
                            batch,
                            occurrences,
                            occurrence_ends,
                            metadata_values,
                            strict=True,
                        )
                    )
                )
            if not prepared:
                return ()
            return self._add_prepared(prepared, operation=assets)

    def add_stream(
        self,
        contents: Iterable[ContentInput | StreamInput],
    ) -> Iterator[MemoryRecord]:
        """Add a lazy omni stream one durable, searchable observation at a time."""
        if isinstance(contents, (str, bytes, Path, Blob, AssetRef, Mapping)):
            raise ValidationError("contents must be an iterable of memory inputs")
        try:
            iterator = iter(contents)
        except TypeError:
            raise ValidationError("contents must be an iterable of memory inputs") from None
        index = 0
        while True:
            try:
                content = next(iterator)
            except StopIteration:
                return
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            try:
                if isinstance(content, StreamInput):
                    record = self.add(
                        content.content,
                        occurred_at=content.occurred_at,
                        occurred_end=content.occurred_end,
                        metadata=content.metadata,
                        memory_type=content.memory_type,
                    )
                else:
                    record = self.add(content)
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            yield record
            index += 1

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return ranked memories for a native or mixed-modal query."""
        return self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            capture_trace=False,
        ).hits

    def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> TracedSearchResult:
        """Return ranked memories plus an opt-in candidate trace without evidence content."""
        outcome = self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            capture_trace=True,
        )
        assert outcome.trace is not None
        return TracedSearchResult(hits=outcome.hits, trace=outcome.trace)

    def _search(
        self,
        query: ContentInput,
        *,
        limit: int,
        memory_type: MemoryType | None,
        reference_at: datetime | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        capture_trace: bool,
    ) -> _SearchOutcome:
        with self._trace("mindbridge.search", kind="operation"), self._operation() as assets:
            _limit(limit, maximum=100)
            occurred_from, occurred_until = _search_occurrence_range(
                occurred_from,
                occurred_until,
            )
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = self._prepare_content(query, assets)
            explicit_reference = _reference_at(reference_at)
            reference, temporal_text = _temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            outcome = self._search_prepared(
                prepared,
                limit=limit,
                operation=assets,
                memory_type=_optional_memory_type(memory_type),
                reference_at=reference,
                temporal_range=_temporal_range(temporal_text, reference),
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                require_unambiguous=limit == 1,
                capture_trace=capture_trace,
            )
            self._persist_transcripts(assets)
            return outcome

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> AnswerResult:
        """Answer a native or mixed-modal question only from retrieved memories."""
        with self._trace("mindbridge.ask", kind="operation"), self._operation() as assets:
            _limit(limit, maximum=100)
            if self._answerer is None:
                raise ModelError(
                    "answer backend is not configured",
                    reason="backend_not_configured",
                    stage="generate",
                )
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = self._prepare_content(question, assets)
            explicit_reference = _reference_at(reference_at)
            reference, temporal_text = _temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            temporal_range = _temporal_range(temporal_text, reference)
            search = partial(
                self._search_prepared,
                prepared,
                # Without a budget the answer grounds on `limit` hits, so ranking three times
                # that is enough headroom for the modality round robin. With one, the budget is
                # what decides depth, so the whole rerank pool has to be ranked for it to spend.
                limit=(
                    _RERANK_CANDIDATES if self._evidence_budget is not None else min(100, limit * 3)
                ),
                operation=assets,
                memory_type=_optional_memory_type(memory_type),
                reference_at=reference,
                temporal_range=temporal_range,
                occurred_from=None,
                occurred_until=None,
                require_unambiguous=limit == 1,
                capture_trace=False,
            )
            speech_assets = self._answer_speech_assets(prepared.assets)
            if speech_assets and all(
                Modality(asset.modality) in self._embedding_capabilities for asset in speech_assets
            ):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    context = copy_context()
                    identities = executor.submit(
                        context.run,
                        self._recognize_speech,
                        speech_assets,
                        assets,
                    )
                    hits = search().hits
                    identities.result()
            else:
                if speech_assets:
                    self._recognize_speech(speech_assets, assets)
                hits = search().hits
            hits = _grounding_hits(hits, limit, budget_chars=self._evidence_budget)
            routed_question = self._route_generation(
                (
                    _with_reference_time(prepared, reference)
                    if explicit_reference is not None
                    or temporal_text != prepared.text
                    or temporal_range is not None
                    else prepared
                ),
                assets,
            )
            routed_hits = self._route_generation_hits(hits, assets) if hits else ()
            self._persist_transcripts(assets)
            result = self._answer(routed_question, routed_hits)
            used_ids = {hit.id for hit in result.hits}
            return AnswerResult(
                answer=result.answer,
                hits=tuple(hit for hit in hits if hit.id in used_ids),
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
            )

    def get(self, memory_id: str) -> MemoryRecord:
        """Return one memory or raise `MemoryNotFoundError`."""
        with (
            self._trace("mindbridge.get", kind="operation"),
            self._operation() as assets,
            self._write_lock,
        ):
            normalized_id = _identifier(memory_id, "memory_id")
            with _translate_storage_errors("read memory"):
                memory = self._store.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            self._lease_assets(memory.assets, assets.leased)
            return self._memory_record(memory)

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        """Transcribe speech and resolve stable local speaker identities."""
        with self._trace("mindbridge.speech", kind="operation"), self._operation() as operation:
            normalized_id = _identifier(memory_id, "memory_id")
            with self._write_lock, _translate_storage_errors("read speech memory"):
                memory = self._store.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            speech_assets = tuple(
                {
                    asset.asset_id: asset
                    for asset in memory.assets
                    if asset.modality in {"audio", "video"}
                }.values()
            )
            if not speech_assets:
                return ()
            if not isinstance(self._transcriber, SpeechBackend):
                raise ModelError(
                    "configured transcription backend does not provide speaker recognition"
                )
            self._lease_assets(speech_assets, operation.leased)
            operation.persisted.update(asset.asset_id for asset in speech_assets)
            self._recognize_speech(speech_assets, operation)
            return tuple(
                segment
                for asset in speech_assets
                for segment in operation.speech_segments[asset.asset_id]
            )

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        """Detect faces and resolve stable local face-and-voice identities."""
        with self._trace("mindbridge.faces", kind="operation"), self._operation() as operation:
            normalized_id = _identifier(memory_id, "memory_id")
            with self._write_lock, _translate_storage_errors("read face memory"):
                memory = self._store.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            visual_assets = tuple(
                {
                    asset.asset_id: asset
                    for asset in memory.assets
                    if asset.modality in {"image", "video"}
                }.values()
            )
            if not visual_assets:
                return ()
            if not isinstance(self._face_analyzer, FaceBackend):
                raise ModelError("no face backend is configured")
            unsupported = {
                Modality(asset.modality)
                for asset in visual_assets
                if Modality(asset.modality) not in self._face_capabilities
            }
            if unsupported:
                names = ", ".join(sorted(modality.value for modality in unsupported))
                raise ModelError(f"configured face backend does not support: {names}")
            self._lease_assets(visual_assets, operation.leased)
            operation.persisted.update(asset.asset_id for asset in visual_assets)
            self._recognize_faces(visual_assets, operation)
            return tuple(
                observation
                for asset in visual_assets
                for observation in operation.face_observations[asset.asset_id]
            )

    def register_speaker(self, speaker_id: str, name: str) -> None:
        """Assign or replace a human-readable name for one recognized speaker."""
        self._register_identity(speaker_id, name, speaker=True)

    def register_identity(self, identity_id: str, name: str) -> None:
        """Assign or replace a name shared by one face-and-voice identity."""
        self._register_identity(identity_id, name, speaker=False)

    def _register_identity(self, identity_id: str, name: str, *, speaker: bool) -> None:
        id_label = "speaker_id" if speaker else "identity_id"
        requested_id = _identifier(identity_id, id_label)
        normalized_name = _identity_name(name)
        operation_name = "register_speaker" if speaker else "register_identity"
        with (
            self._trace(f"mindbridge.{operation_name}", kind="operation"),
            self._operation() as operation,
            self._write_lock,
        ):
            with _translate_storage_errors("read identity memories"):
                normalized_id = self._store.resolve_identity_id(requested_id)
                if normalized_id is None:
                    identity_exists = False
                    memory_ids = None
                elif speaker:
                    memory_ids = self._store.speaker_memory_ids(normalized_id)
                    identity_exists = memory_ids is not None
                else:
                    identity_exists = self._store.identity_memory_ids(normalized_id) is not None
                    speaker_memory_ids = self._store.speaker_memory_ids(normalized_id)
                    memory_ids = () if speaker_memory_ids is None else speaker_memory_ids
            if not identity_exists:
                if speaker:
                    raise SpeakerNotFoundError(f"speaker does not exist: {requested_id}")
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            assert normalized_id is not None
            assert memory_ids is not None
            memories: tuple[StoredMemory, ...] = ()
            embeddings: tuple[StoredEmbedding, ...] = ()
            if memory_ids:
                with _translate_storage_errors("read speaker memories"):
                    stored = self._store.read_memories(memory_ids)
                indexed = tuple(
                    memory
                    for memory in stored
                    if any(
                        f"[speech identities:{asset.asset_id}]\n" in memory.content
                        for asset in memory.assets
                        if asset.modality in {"audio", "video"}
                    )
                )
                if indexed:
                    memories, embeddings = self._refresh_speaker_memories(
                        indexed,
                        speaker_id=normalized_id,
                        speaker_name=normalized_name,
                        operation=operation,
                    )
            with _translate_storage_errors("register identity"):
                registered = self._store.register_identity(
                    normalized_id,
                    normalized_name,
                    memories=memories,
                    embeddings=embeddings,
                )
            if not registered:
                if speaker:
                    raise SpeakerNotFoundError(f"speaker does not exist: {requested_id}")
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            self._drain_outbox()

    def reinforce(self, memory_ids: Sequence[str]) -> int:
        """Record explicit positive feedback for existing memories."""
        if isinstance(memory_ids, (str, bytes)):
            raise ValidationError("memory_ids must be a sequence of memory IDs")
        try:
            normalized = tuple(
                dict.fromkeys(_identifier(memory_id, "memory_id") for memory_id in memory_ids)
            )
        except TypeError:
            raise ValidationError("memory_ids must be a sequence of memory IDs") from None
        if not normalized:
            return 0
        with (
            self._trace("mindbridge.reinforce", kind="operation"),
            self._operation(),
            self._write_lock,
            _translate_storage_errors("reinforce memories"),
        ):
            return self._store.reinforce_memories(
                normalized,
                accessed_at=datetime.now(timezone.utc),
            )

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        """List newest memories with an opaque stable keyset cursor."""
        with (
            self._trace("mindbridge.list", kind="operation"),
            self._operation() as assets,
            self._write_lock,
        ):
            _limit(limit, maximum=100)
            after = None if cursor is None else _decode_cursor(cursor)
            with _translate_storage_errors("list memories"):
                memories = self._store.list_memories(limit=limit + 1, after=after)
            has_next = len(memories) > limit
            visible = memories[:limit]
            self._lease_assets(
                tuple(asset for memory in visible for asset in memory.assets),
                assets.leased,
            )
            next_cursor = _encode_cursor(visible[-1]) if has_next else None
            return Page(
                items=tuple(self._memory_record(memory) for memory in visible),
                next_cursor=next_cursor,
            )

    def delete(self, memory_id: str) -> bool:
        """Delete one memory and garbage-collect media no memory still references."""
        with (
            self._trace("mindbridge.delete", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            normalized_id = _identifier(memory_id, "memory_id")
            with _translate_storage_errors("delete memory"):
                deleted, orphaned = self._store.delete_memory_with_assets(normalized_id)
            self._queue_asset_cleanup(orphaned)
            self._drain_outbox()
            return deleted

    def reindex(self) -> int:
        """Rebuild the disposable Zvec collection from authoritative SQLite rows."""
        with (
            self._trace("mindbridge.reindex", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            self._drain_outbox()
            with _translate_storage_errors("checkpoint a search-index rebuild"):
                self._store.queue_all_embeddings()
            memory_count = 0

            def documents() -> Iterator[IndexDocument]:
                nonlocal memory_count
                for document in self._index_documents():
                    if document.embedding.object_part == 0:
                        memory_count += 1
                    yield document

            with _translate_index_errors("rebuild the search index"):
                self._index.rebuild(documents(), batch_size=_REINDEX_PAGE_SIZE)
            # Adds may commit SQLite while the rebuild owns the Zvec boundary. Replay instead of
            # blindly acknowledging so records committed after its SQLite scan cannot be lost.
            self._drain_outbox()
            return memory_count

    def optimize(self) -> None:
        """Merge staged Zvec vectors into the configured index."""
        with (
            self._trace("mindbridge.optimize", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            self._drain_outbox()
            with _translate_index_errors("optimize the search index"):
                self._index.optimize()
                self._index.flush()

    def close(self) -> None:
        """Close model, index, and SQLite resources; repeated calls are harmless."""
        self._require_owner_process()
        with self._lifecycle:
            while self._closing:
                self._lifecycle.wait()
            if self._closed:
                return
            self._closing = True
            while self._active_operations:
                self._lifecycle.wait()
        try:
            with self._write_lock:
                failures = []
                try:
                    self._cleanup_pending_assets()
                except Exception as error:
                    failures.append(error)
                failures.extend(self._close_resources())
        finally:
            with self._lifecycle:
                self._closed = True
                self._closing = False
                self._lifecycle.notify_all()
        if not failures:
            return
        first = failures[0]
        if isinstance(first, MindBridgeError):
            raise first
        raise StorageError(
            "failed to close local memory resources", reason="io_failed", stage="close"
        ) from first

    def _prepare_batch_item(
        self,
        content: ContentInput,
        *,
        index: int,
        operation: _OperationAssets,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
    ) -> _PreparedMemory:
        """Prepare one batch item, naming its position when it is the item that fails."""
        try:
            return _prepare_memory(
                self._prepare_content(content, operation),
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
            )
        except MindBridgeError as error:
            if error.subject is None:
                error.subject = f"contents[{index}]"
            raise

    def _prepare_content(
        self,
        content: ContentInput,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        atoms = _content_atoms(content)
        text_parts: builtins.list[str] = []
        assets: builtins.list[StoredAsset] = []
        canonical: builtins.list[tuple[str, str]] = []
        for atom in atoms:
            if isinstance(atom, str):
                text = _text(atom, "content")
                text_parts.append(text)
                canonical.append(("text", text))
                continue
            asset = self._materialize_atom(atom, operation)
            assets.append(asset)
            canonical.append(("asset", asset.sha256))
        text = "\n\n".join(text_parts)
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise ValidationError(f"content text must not exceed {_MAX_TEXT_CHARACTERS} characters")
        return _PreparedContent(
            text=text,
            assets=tuple(assets),
            modality=_memory_modality(assets),
            canonical_parts=tuple(canonical),
        )

    def _materialize_atom(
        self,
        atom: ContentAtom,
        operation: _OperationAssets,
    ) -> StoredAsset:
        try:
            if isinstance(atom, Path):
                modality, media_type = _media_hint(atom.name, None)
                candidate = self._assets.materialize_path(
                    atom,
                    modality=modality.value,
                    mime_type=media_type,
                    lease=True,
                )
            elif isinstance(atom, Blob):
                modality, media_type = _media_hint(atom.name, atom.media_type)
                candidate = self._assets.materialize_bytes(
                    atom.data,
                    modality=modality.value,
                    mime_type=media_type,
                    name=atom.name,
                    lease=True,
                )
            elif isinstance(atom, AssetRef):
                return self._resolve_asset_reference(atom, operation)
            else:
                raise ValidationError("content contains an unsupported input value")
        except MindBridgeError:
            raise
        except (AssetTooLargeError, OSError, ValueError):
            raise ValidationError("media input could not be safely materialized") from None
        except AssetStoreError as error:
            raise StorageError("failed to materialize media input") from error

        operation.leased.append(candidate)
        operation.cleanup.append(candidate)
        with _translate_storage_errors("resolve media metadata"):
            stored = self._store.read_asset(candidate.asset_id)
        if stored is not None:
            self._validate_asset_match(candidate, stored)
            operation.persisted.add(stored.asset_id)
            return stored
        return candidate

    def _resolve_asset_reference(
        self,
        reference: AssetRef,
        operation: _OperationAssets,
    ) -> StoredAsset:
        try:
            with self._write_lock, _translate_storage_errors("resolve media reference"):
                asset = self._store.read_asset(reference.id)
                if asset is not None:
                    self._lease_assets((asset,), operation.leased)
        except StorageError as error:
            if isinstance(error.__cause__, ValueError):
                raise ValidationError("asset id must be a SHA-256 identifier") from None
            raise
        if asset is None:
            raise ValidationError("asset reference does not exist in this data directory")
        if reference.modality is not None and reference.modality.value != asset.modality:
            raise ValidationError("asset reference modality does not match stored media")
        if reference.media_type is not None and reference.media_type != asset.mime_type:
            raise ValidationError("asset reference media_type does not match stored media")
        if reference.size_bytes is not None and reference.size_bytes != asset.size_bytes:
            raise ValidationError("asset reference size does not match stored media")
        if reference.sha256 is not None and reference.sha256 != asset.sha256:
            raise ValidationError("asset reference digest does not match stored media")
        operation.persisted.add(asset.asset_id)
        return asset

    @staticmethod
    def _validate_asset_match(candidate: StoredAsset, stored: StoredAsset) -> None:
        if any(
            getattr(candidate, name) != getattr(stored, name)
            for name in ("modality", "mime_type", "size_bytes", "sha256", "relative_path")
        ):
            raise StorageError("content-addressed media metadata is inconsistent")

    def _add_prepared(
        self,
        prepared: Sequence[_PreparedMemory],
        *,
        operation: _OperationAssets,
    ) -> tuple[MemoryRecord, ...]:
        unique = {memory.memory_id: memory for memory in prepared}
        ordered_ids = tuple(unique)
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            _translate_storage_errors("check existing memories"),
        ):
            existing_rows = self._store.read_memories(ordered_ids)
        existing_ids = {row.memory_id for row in existing_rows}
        missing = [unique[memory_id] for memory_id in ordered_ids if memory_id not in existing_ids]
        speech_indexed = self._index_speech and any(
            self._answer_speech_assets(memory.content.assets) for memory in missing
        )
        with self._speech_index_guard(operation, enabled=speech_indexed):
            stored_memories: tuple[StoredMemory, ...] = ()
            stored_embeddings: tuple[StoredEmbedding, ...] = ()
            if missing:
                if speech_indexed:
                    self._recognize_speech(
                        self._answer_speech_assets(
                            tuple(asset for memory in missing for asset in memory.content.assets)
                        ),
                        operation,
                        reversible=True,
                    )
                batched = tuple(asset for memory in missing for asset in memory.content.assets)
                fallback = Modality.AUDIO not in self._embedding_capabilities
                if fallback:
                    for memory in missing:
                        _fallback_unsupported(
                            memory.content,
                            self._embedding_capabilities,
                            "embedding",
                        )
                if fallback or self._derives_transcripts(batched):
                    self._cache_audio_transcripts(batched, operation)
                missing = [self._prepare_for_embedding(memory, operation) for memory in missing]
                embedding_parts = tuple(
                    (memory, object_part, model_input)
                    for memory in missing
                    for object_part, model_input in enumerate(
                        self._embedding_inputs(memory.content)
                    )
                )
                vectors, embedding_parts = self._embed_document_parts(embedding_parts)
                now = datetime.now(timezone.utc)
                stored_memories = tuple(
                    StoredMemory(
                        memory_id=memory.memory_id,
                        content=memory.content.text,
                        modality=memory.content.modality.value,
                        memory_type=memory.memory_type.value,
                        assets=memory.content.assets,
                        metadata_json=memory.metadata_json,
                        occurred_at=memory.occurred_at,
                        occurred_end=memory.occurred_end,
                        created_at=now,
                        updated_at=now,
                    )
                    for memory in missing
                )
                stored_embeddings = tuple(
                    StoredEmbedding(
                        embedding_id=_embedding_id(memory.memory_id, object_part),
                        memory_id=memory.memory_id,
                        values=vector,
                        model_id=self._embedding_model,
                        space_id=self._space_id,
                        task=_DOCUMENT_TASK,
                        created_at=now,
                        object_part=object_part,
                        normalized=True,
                    )
                    for (memory, object_part, _model_input), vector in zip(
                        embedding_parts, vectors, strict=True
                    )
                )
            if stored_memories:
                # SQLite is authoritative and uses one WAL connection per transaction. Commit
                # before taking the index lock so ordinary concurrent writers can share one flush.
                with (
                    self._trace("mindbridge.storage.write", kind="stage"),
                    _translate_storage_errors("write memories"),
                ):
                    self._store.write_memories(stored_memories, stored_embeddings)
            with self._write_lock:
                self._drain_outbox()
                with (
                    self._trace("mindbridge.storage.hydrate", kind="stage"),
                    _translate_storage_errors("hydrate written memories"),
                ):
                    authoritative = self._store.read_memories(ordered_ids)
                operation.persisted.update(
                    asset.asset_id for memory in authoritative for asset in memory.assets
                )
                if operation.speech_updates:
                    self._persist_transcripts(operation)
        rows_by_id = {memory.memory_id: memory for memory in authoritative}
        if rows_by_id.keys() != unique.keys():
            raise StorageError("written memories could not be read from SQLite")
        return tuple(self._memory_record(rows_by_id[memory.memory_id]) for memory in prepared)

    def _prepare_for_embedding(
        self,
        memory: _PreparedMemory,
        operation: _OperationAssets,
    ) -> _PreparedMemory:
        content = memory.content
        if self._index_speech and self._answer_speech_assets(content.assets):
            content = self._with_speech_identities(content, operation)
        return replace(
            memory,
            content=self._embedding_content(content, operation),
        )

    @contextmanager
    def _speech_index_guard(
        self,
        operation: _OperationAssets,
        *,
        enabled: bool,
    ) -> Iterator[None]:
        if not enabled:
            yield
            return
        # ponytail: serialize add-time identity matching through the memory commit; replace this
        # with a staged identity plan only if speech-indexed add throughput becomes material.
        with self._write_lock:
            try:
                yield
            except BaseException:
                with _translate_storage_errors("roll back speaker recognition"):
                    for rollback in reversed(operation.speech_rollbacks):
                        self._store.rollback_speech(rollback)
                raise
            finally:
                operation.speech_rollbacks.clear()

    def _embedding_inputs(
        self,
        prepared: _PreparedContent,
        *,
        maximum_keys: int = _MAX_RETRIEVAL_KEYS,
    ) -> tuple[ModelInput, ...]:
        prepared = _retrieval_content(prepared)
        aggregate = self._route_embedding(prepared)
        text_parts = tuple(value for kind, value in prepared.canonical_parts if kind == "text")
        if prepared.text and prepared.text != "\n\n".join(text_parts):
            text_parts = (*text_parts, prepared.text)
        text_keys = tuple(key for text in text_parts for key in _contextual_text_keys(text))
        atomic = [
            *(self._route_embedding(_text_content(text)) for text in text_keys),
            *(self._route_embedding(_asset_content(asset)) for asset in prepared.assets),
        ]
        unique = tuple(dict.fromkeys(value for value in atomic if value != aggregate))
        if len(unique) > maximum_keys:
            unique = tuple(
                unique[round(index * (len(unique) - 1) / (maximum_keys - 1))]
                for index in range(maximum_keys)
            )
        return (aggregate, *unique)

    def _refresh_speaker_memories(
        self,
        memories: Sequence[StoredMemory],
        *,
        speaker_id: str,
        speaker_name: str | None,
        operation: _OperationAssets,
        previous_speaker_id: str | None = None,
        update_operation: bool = True,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        prepared: list[tuple[StoredMemory, _PreparedContent]] = []
        for memory in memories:
            segments_by_asset: dict[str, tuple[SpeakerSegment, ...]] = {}
            speech_assets = tuple(
                asset for asset in memory.assets if asset.modality in {"audio", "video"}
            )
            with _translate_storage_errors("read cached speaker recognition"):
                for asset in speech_assets:
                    segments = self._store.read_speech(
                        asset.asset_id,
                        space_id=self._transcription_space,
                    )
                    if segments is None:
                        continue
                    refreshed = tuple(
                        replace(
                            segment,
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                        )
                        if segment.speaker_id in {speaker_id, previous_speaker_id}
                        else segment
                        for segment in segments
                    )
                    segments_by_asset[asset.asset_id] = refreshed
                    if update_operation:
                        operation.speech_segments[asset.asset_id] = refreshed
                        operation.transcripts[asset.asset_id] = "\n".join(
                            segment.text for segment in refreshed
                        )
            base = _without_speech_identities(memory.content, tuple(segments_by_asset))
            content = _PreparedContent(
                text=_speech_identity_text(
                    base,
                    tuple(asset for asset in speech_assets if asset.asset_id in segments_by_asset),
                    segments_by_asset,
                ),
                assets=memory.assets,
                modality=Modality(memory.modality),
                canonical_parts=(("text", base),) if base else (),
            )
            prepared.append((memory, self._embedding_content(content, operation)))

        parts = tuple(
            (memory, content, object_part, model_input)
            for memory, content in prepared
            for object_part, model_input in enumerate(self._embedding_inputs(content))
        )
        vectors = self._embed(
            tuple(model_input for _memory, _content, _part, model_input in parts),
            task=EmbedTask.DOCUMENT,
        )
        now = datetime.now(timezone.utc)
        updated = tuple(
            replace(memory, content=content.text, updated_at=now) for memory, content in prepared
        )
        embeddings = tuple(
            StoredEmbedding(
                embedding_id=_embedding_id(memory.memory_id, object_part),
                memory_id=memory.memory_id,
                values=vector,
                model_id=self._embedding_model,
                space_id=self._space_id,
                task=_DOCUMENT_TASK,
                created_at=now,
                object_part=object_part,
                normalized=True,
            )
            for (memory, _content, object_part, _model_input), vector in zip(
                parts,
                vectors,
                strict=True,
            )
        )
        return updated, embeddings

    def _search_prepared(
        self,
        prepared: _PreparedContent,
        *,
        limit: int,
        operation: _OperationAssets,
        memory_type: MemoryType | None,
        reference_at: datetime,
        temporal_range: tuple[datetime, datetime] | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        require_unambiguous: bool,
        capture_trace: bool,
    ) -> _SearchOutcome:
        trace_candidates: builtins.list[RetrievalCandidateTrace] | None = (
            [] if capture_trace else None
        )
        prepared = self._embedding_content(prepared, operation)
        aggregate = self._route_embedding(prepared)
        text_parts = tuple(value for kind, value in prepared.canonical_parts if kind == "text")
        focused_text = text_parts[0] if len(text_parts) > 1 else prepared.text
        focused = replace(
            prepared,
            text=focused_text,
            canonical_parts=(("text", focused_text),) if focused_text else (),
        )
        model_inputs = tuple(
            dict.fromkeys(
                (
                    aggregate,
                    *self._embedding_inputs(
                        focused,
                        maximum_keys=_MAX_QUERY_RETRIEVAL_KEYS,
                    ),
                )
            )
        )
        vectors = tuple(dict.fromkeys(self._embed(model_inputs, task=EmbedTask.QUERY)))
        lexical_query = focused_text
        if not (_lexical_terms(lexical_query) - _LEXICAL_NOISE_TERMS):
            lexical_query = ""
        candidate_limit = max(_RERANK_CANDIDATES, limit * 3)
        candidate_ceiling = max(candidate_limit, limit * (_MAX_RETRIEVAL_KEYS + 1))
        with self._write_lock:
            self._drain_outbox()
        while True:
            with _translate_index_errors("search memories"):
                if temporal_range is None:
                    candidates = self._index_candidates(
                        vectors,
                        lexical_query=lexical_query,
                        limit=candidate_limit,
                        memory_type=memory_type,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                    )
                else:
                    # Zvec range-filtered FTS is unstable after dense queries; the global
                    # fallback below still contributes the same lexical route.
                    preferred_range = _intersect_occurrence_range(
                        temporal_range,
                        occurred_from,
                        occurred_until,
                    )
                    preferred = (
                        _IndexCandidates(dense=(), lexical=(), exhausted=True)
                        if preferred_range is None
                        else self._index_candidates(
                            vectors,
                            lexical_query="",
                            limit=candidate_limit,
                            memory_type=memory_type,
                            occurred_from=preferred_range[0],
                            occurred_until=preferred_range[1],
                        )
                    )
                    fallback = self._index_candidates(
                        vectors,
                        lexical_query=lexical_query,
                        limit=candidate_limit,
                        memory_type=memory_type,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                    )
                    candidates = _IndexCandidates(
                        dense=_merge_index_hits(preferred.dense, fallback.dense),
                        lexical=_merge_index_hits(preferred.lexical, fallback.lexical),
                        exhausted=preferred.exhausted and fallback.exhausted,
                    )
            index_ids = tuple(
                dict.fromkeys(hit.id for hit in (*candidates.dense, *candidates.lexical))
            )
            if not index_ids:
                return _search_outcome(
                    (),
                    trace_candidates,
                    candidate_limit=candidate_limit,
                    exhaustive=candidates.exhausted,
                )
            with (
                self._trace("mindbridge.storage.hydrate", kind="stage"),
                _translate_storage_errors("hydrate search candidates"),
            ):
                hydrated_documents = self._store.read_index_documents(index_ids)
            with _translate_index_errors("search memories"):
                candidates, hydrated_documents = self._deepen_temporal_lexical_candidates(
                    candidates,
                    hydrated_documents,
                    lexical_query=lexical_query,
                    temporal_range=temporal_range,
                    memory_type=memory_type,
                    route_limit=candidate_limit,
                    result_limit=limit,
                )
            index_ids = tuple(
                dict.fromkeys(hit.id for hit in (*candidates.dense, *candidates.lexical))
            )
            documents = hydrated_documents
            if occurred_from is not None or occurred_until is not None:
                documents = tuple(
                    document
                    for document in documents
                    if _occurrence_overlaps(
                        document.occurred_at,
                        document.occurred_end,
                        occurred_from,
                        occurred_until,
                    )
                )
            parent_count = len({document.embedding.memory_id for document in documents})
            if (
                parent_count >= limit
                or candidates.exhausted
                or candidate_limit >= candidate_ceiling
            ):
                break
            candidate_limit = min(candidate_limit * 2, candidate_ceiling)
        index_ids_by_memory = (
            _parent_index_ids(hydrated_documents) if trace_candidates is not None else {}
        )
        _extend_hydration_traces(
            trace_candidates,
            candidates,
            index_ids,
            hydrated_documents,
            documents,
            index_ids_by_memory,
        )
        (
            dense_relevance,
            dense_confidence,
            lexical_relevance_by_rank,
            lexical_matches,
        ) = _parent_index_signals(candidates, documents)
        parent_ids = tuple(dict.fromkeys((*dense_relevance, *lexical_matches)))
        if not parent_ids:
            return _search_outcome(
                (),
                trace_candidates,
                candidate_limit=candidate_limit,
                exhaustive=candidates.exhausted,
            )
        with self._write_lock:
            with (
                self._trace("mindbridge.storage.hydrate", kind="stage"),
                _translate_storage_errors("hydrate search results"),
            ):
                memories = self._store.read_memories(parent_ids)
            with self._trace("mindbridge.retrieval.rank", kind="stage"):
                _extend_missing_memory_traces(
                    trace_candidates,
                    parent_ids,
                    memories,
                    index_ids_by_memory,
                    dense_relevance,
                    dense_confidence,
                    lexical_relevance_by_rank,
                    lexical_matches,
                )
                if memory_type is not None:
                    _extend_memory_type_traces(
                        trace_candidates,
                        memories,
                        memory_type,
                        index_ids_by_memory,
                        dense_relevance,
                        dense_confidence,
                        lexical_relevance_by_rank,
                        lexical_matches,
                    )
                    memories = tuple(
                        memory for memory in memories if memory.memory_type == memory_type.value
                    )
                lexical_relevance = _lexical_relevance(lexical_query, memories)
                ranked = []
                ranked_traces: dict[str, RetrievalCandidateTrace] | None = (
                    {} if trace_candidates is not None else None
                )
                for memory in memories:
                    memory_id = memory.memory_id
                    lexical_match = memory_id in lexical_matches
                    lexical_strength = (
                        lexical_relevance.get(memory_id, 0.0) if lexical_match else 0.0
                    )
                    lexical_floor = (
                        _LEXICAL_FULL_COVERAGE_RELEVANCE
                        if lexical_strength >= _LEXICAL_FULL_COVERAGE
                        else _LEXICAL_RANK_RELEVANCE
                    )
                    lexical_score = lexical_floor * lexical_relevance_by_rank.get(memory_id, 0.0)
                    base = max(dense_relevance.get(memory_id, 0.0), lexical_score)
                    relevance = _bounded_scale(
                        base,
                        1.0 + _MAX_LEXICAL_RERANK_BONUS * lexical_strength,
                    )
                    lexical_rerank_bonus = relevance - base
                    gate_confidence = max(
                        dense_confidence.get(memory_id, 0.0),
                        _LEXICAL_MATCH_CONFIDENCE if lexical_match else 0.0,
                    )
                    (
                        final_score,
                        reinforcement_factor,
                        temporal_factor,
                        retention_factor,
                    ) = _ranking_signals(
                        memory,
                        relevance,
                        reference_at=reference_at,
                        temporal_range=temporal_range,
                        decay_half_life=self._decay_half_life,
                    )
                    ranked.append(
                        (
                            memory,
                            final_score,
                            gate_confidence,
                            lexical_match,
                        )
                    )
                    _record_ranked_trace(
                        ranked_traces,
                        memory_id,
                        index_ids_by_memory,
                        dense_relevance,
                        dense_confidence,
                        lexical_relevance=lexical_score,
                        lexical_rerank_bonus=lexical_rerank_bonus,
                        lexical_match=lexical_match,
                        gate_confidence=gate_confidence,
                        base_relevance=relevance,
                        reinforcement_factor=reinforcement_factor,
                        temporal_factor=temporal_factor,
                        retention_factor=retention_factor,
                        final_score=final_score,
                    )
                ranked = _qualified_candidates(
                    ranked,
                    minimum_relevance=self._minimum_relevance,
                    trace_candidates=trace_candidates,
                    ranked_traces=ranked_traces,
                )
                ranked.sort(key=lambda item: (-item[1], item[0].memory_id))
                ambiguous = require_unambiguous and _retrieval_is_ambiguous(
                    ranked,
                    margin=self._ambiguity_margin,
                    temporal_range=temporal_range,
                )
                _extend_ranked_traces(
                    trace_candidates,
                    ranked_traces,
                    ranked,
                    limit=limit,
                    ambiguous=ambiguous,
                )
                visible = () if ambiguous else ranked[:limit]
                self._lease_assets(
                    tuple(
                        asset
                        for memory, _score, _confidence, _lexical in visible
                        for asset in memory.assets
                    ),
                    operation.leased,
                )
                operation.persisted.update(
                    asset.asset_id
                    for memory, _score, _confidence, _lexical in visible
                    for asset in memory.assets
                )
        hits = tuple(
            self._search_hit(memory, score) for memory, score, _confidence, _lexical in visible
        )
        return _search_outcome(
            hits,
            trace_candidates,
            candidate_limit=candidate_limit,
            exhaustive=candidates.exhausted,
            ambiguous=ambiguous,
        )

    def _deepen_temporal_lexical_candidates(
        self,
        candidates: _IndexCandidates,
        documents: tuple[IndexDocument, ...],
        *,
        lexical_query: str,
        temporal_range: tuple[datetime, datetime] | None,
        memory_type: MemoryType | None,
        route_limit: int,
        result_limit: int,
    ) -> tuple[_IndexCandidates, tuple[IndexDocument, ...]]:
        if temporal_range is None or not lexical_query or len(candidates.lexical) < route_limit:
            return candidates, documents
        lexical_by_id = {hit.id: hit for hit in candidates.lexical}
        qualified_parents = {
            document.embedding.memory_id
            for document in documents
            if (hit := lexical_by_id.get(document.embedding.embedding_id)) is not None
            and _LEXICAL_MATCH_CONFIDENCE * hit.relevance >= self._minimum_relevance
            and _overlaps_temporal_range(
                document.occurred_at,
                document.occurred_end,
                temporal_range,
            )
        }
        if len(qualified_parents) >= result_limit:
            return candidates, documents
        with (
            self._trace("mindbridge.storage.temporal_candidates", kind="stage"),
            _translate_storage_errors("read temporal search candidates"),
        ):
            in_range, total = self._store.embedding_ids_in_range(
                *temporal_range,
                space_id=self._space_id,
                task=_DOCUMENT_TASK,
                memory_type=None if memory_type is None else memory_type.value,
            )
        if not in_range:
            return candidates, documents
        required = min(result_limit, len(in_range))
        search_limit = route_limit
        lexical = candidates.lexical
        qualified = tuple(
            hit
            for hit in lexical
            if hit.id in in_range
            and _LEXICAL_MATCH_CONFIDENCE * hit.relevance >= self._minimum_relevance
        )
        while len(qualified) < required and len(lexical) >= search_limit and search_limit < total:
            search_limit = min(search_limit * 2, total)
            lexical = self._index_candidates(
                (),
                lexical_query=lexical_query,
                limit=search_limit,
                memory_type=memory_type,
            ).lexical
            qualified = tuple(
                hit
                for hit in lexical
                if hit.id in in_range
                and _LEXICAL_MATCH_CONFIDENCE * hit.relevance >= self._minimum_relevance
            )
        updated = _IndexCandidates(
            dense=candidates.dense,
            lexical=_merge_index_hits(candidates.lexical, qualified[:route_limit]),
            exhausted=candidates.exhausted,
        )
        added_ids = tuple(hit.id for hit in updated.lexical if hit.id not in lexical_by_id)
        if not added_ids:
            return updated, documents
        with (
            self._trace("mindbridge.storage.hydrate", kind="stage"),
            _translate_storage_errors("hydrate temporal lexical candidates"),
        ):
            added = self._store.read_index_documents(added_ids)
        by_id = {document.embedding.embedding_id: document for document in (*documents, *added)}
        return updated, tuple(by_id.values())

    def _index_candidates(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        lexical_query: str,
        limit: int,
        memory_type: MemoryType | None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> _IndexCandidates:
        memory_type_value = None if memory_type is None else memory_type.value
        dense_calls = tuple(
            partial(
                self._index.search,
                vector,
                limit=limit,
                space_id=self._space_id,
                task=_DOCUMENT_TASK,
                memory_type=memory_type_value,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
            )
            for vector in vectors
        )
        lexical_call = partial(
            self._index.lexical_search,
            lexical_query,
            limit=limit,
            space_id=self._space_id,
            task=_DOCUMENT_TASK,
            memory_type=memory_type_value,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
        )
        calls = (*dense_calls, lexical_call) if lexical_query else dense_calls
        routes: tuple[tuple[IndexHit, ...], ...]
        with self._trace("mindbridge.index.search", kind="stage") as span:
            span.set_attribute("mindbridge.index.route_count", len(calls))
            if len(calls) == 1:
                routes = (calls[0](),)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(_MAX_INDEX_SEARCH_WORKERS, len(calls))
                ) as executor:
                    futures = tuple(executor.submit(copy_context().run, call) for call in calls)
                    routes = tuple(future.result() for future in futures)
        dense_routes = routes[: len(dense_calls)]
        lexical = routes[-1] if lexical_query else ()
        return _IndexCandidates(
            dense=_merge_index_hits(*dense_routes),
            lexical=lexical,
            exhausted=all(len(route) < limit for route in routes),
        )

    def _route_embedding(self, prepared: _PreparedContent) -> ModelInput:
        value = self._resolved_model_input(prepared)
        missing = value.modalities - self._embedding_capabilities
        if not missing:
            return value
        if missing <= {Modality.AUDIO}:
            text = _derived_text(value.text, prepared.assets)
            assets = tuple(asset for asset in value.assets if asset.modality is not Modality.AUDIO)
            if text or assets:
                routed = ModelInput(text=text, assets=assets)
                missing = routed.modalities - self._embedding_capabilities
                if not missing:
                    return routed
        names = ", ".join(sorted(modality.value for modality in missing))
        raise ModelError(
            f"configured embedding model does not support: {names}",
            reason="unsupported_modality",
        )

    def _embedding_content(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        unsupported = _fallback_unsupported(
            prepared,
            self._embedding_capabilities,
            "embedding",
        )
        if Modality.AUDIO in unsupported:
            _require_audio_transcription(self._transcription_capabilities)
        elif not self._derives_transcripts(prepared.assets):
            return prepared
        return self._with_audio_transcripts(prepared, operation)

    def _route_generation(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> ModelInput:
        unsupported = _fallback_unsupported(
            prepared,
            self._generation_capabilities,
            "generation",
        )
        if self._answer_speech_assets(prepared.assets):
            prepared = self._with_speech_identities(prepared, operation)
        if Modality.AUDIO in unsupported:
            _require_audio_transcription(self._transcription_capabilities)
            prepared = self._with_audio_transcripts(prepared, operation)
        value = self._resolved_model_input(prepared)
        unsupported = value.modalities - self._generation_capabilities
        if not unsupported:
            return value
        text = _derived_text(value.text, prepared.assets)
        assets = tuple(asset for asset in value.assets if asset.modality is not Modality.AUDIO)
        if unsupported <= {Modality.AUDIO} and (text or assets):
            routed = ModelInput(
                text=text,
                assets=assets,
            )
            if not routed.modalities - self._generation_capabilities:
                return routed
        names = ", ".join(sorted(modality.value for modality in unsupported))
        raise ModelError(
            f"configured generation model does not support: {names}",
            reason="unsupported_modality",
        )

    def _route_generation_hits(
        self,
        hits: Sequence[SearchHit],
        operation: _OperationAssets,
    ) -> tuple[SearchHit, ...]:
        asset_ids = tuple(asset.id for hit in hits for asset in hit.assets)
        with (
            self._trace("mindbridge.storage.hydrate", kind="stage"),
            _translate_storage_errors("hydrate media for answer generation"),
        ):
            stored = self._store.read_assets(asset_ids)
        by_id = {asset.asset_id: asset for asset in stored}
        prepared_hits = []
        for hit in hits:
            try:
                assets = tuple(by_id[asset.id] for asset in hit.assets)
            except KeyError:
                raise StorageError("memory references missing media metadata") from None
            prepared_hits.append(
                _PreparedContent(
                    text=hit.content,
                    assets=assets,
                    modality=_memory_modality(assets),
                    canonical_parts=(),
                )
            )
        if Modality.AUDIO not in self._generation_capabilities:
            for prepared in prepared_hits:
                _fallback_unsupported(
                    prepared,
                    self._generation_capabilities,
                    "generation",
                )
        speech_assets = self._answer_speech_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if speech_assets:
            self._recognize_speech(speech_assets, operation)
        elif Modality.AUDIO not in self._generation_capabilities:
            self._cache_audio_transcripts(
                tuple(asset for prepared in prepared_hits for asset in prepared.assets),
                operation,
            )
        face_assets = self._answer_face_assets(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets)
        )
        if face_assets:
            self._recognize_faces(face_assets, operation)
        routed = []
        for hit, prepared in zip(hits, prepared_hits, strict=True):
            if self._answer_face_assets(prepared.assets):
                prepared = self._with_face_identities(prepared, operation)
            model_input = self._route_generation(prepared, operation)
            routed.append(
                replace(
                    hit,
                    content=model_input.text,
                    assets=model_input.assets,
                    modality=model_input.modality,
                )
            )
        return tuple(routed)

    def _answer_face_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if not isinstance(self._face_analyzer, FaceBackend):
            return ()
        supported = {modality.value for modality in self._face_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"} and asset.modality in supported
            }.values()
        )

    def _with_face_identities(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        face_assets = self._answer_face_assets(prepared.assets)
        self._recognize_faces(face_assets, operation)
        text = _face_identity_text(prepared.text, face_assets, operation.face_observations)
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise ModelError("face identity evidence exceeded the supported text length")
        return replace(prepared, text=text)

    def _recognize_faces(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
    ) -> None:
        if not isinstance(self._face_analyzer, FaceBackend):
            raise ModelError("no face backend is configured")
        face_assets = tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"image", "video"}
                and asset.asset_id not in operation.face_observations
            }.values()
        )
        speech_assets = self._answer_speech_assets(
            tuple(asset for asset in face_assets if asset.modality == "video")
        )
        if speech_assets:
            self._recognize_speech(speech_assets, operation)
        missing = []
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            _translate_storage_errors("read cached face recognition"),
        ):
            for asset in face_assets:
                observations = self._store.read_faces(
                    asset.asset_id,
                    space_id=self._face_analysis_space,
                )
                if observations is None:
                    missing.append(asset)
                else:
                    operation.face_observations[asset.asset_id] = observations
        if missing:
            analyses = self._analyze_faces(tuple(self._asset_ref(asset) for asset in missing))
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                _translate_storage_errors("persist face recognition"),
            ):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    speaker_ids = {
                        segment.speaker_id
                        for segment in operation.speech_segments.get(asset.asset_id, ())
                        if segment.speaker_id is not None
                    }
                    preferred = next(iter(speaker_ids)) if len(speaker_ids) == 1 else None
                    operation.face_observations[asset.asset_id] = self._store.write_faces(
                        asset.asset_id,
                        analysis,
                        model_id=self._face_model,
                        space_id=self._face_space,
                        analysis_space_id=self._face_analysis_space,
                        minimum_similarity=self._face_similarity,
                        minimum_margin=self._face_margin,
                        preferred_identity=preferred,
                    )
                written = tuple(
                    observation
                    for asset in missing
                    for observation in operation.face_observations[asset.asset_id]
                )
                _record_identity_matching(
                    len(written),
                    sum(1 for item in written if item.identity_score is not None),
                )
        with self._write_lock, _translate_storage_errors("link face and voice identities"):
            for asset in face_assets:
                self._link_asset_identity(asset.asset_id, operation)

    def _link_asset_identity(self, asset_id: str, operation: _OperationAssets) -> None:
        speaker_ids = {
            segment.speaker_id
            for segment in operation.speech_segments.get(asset_id, ())
            if segment.speaker_id is not None
        }
        face_ids = {
            observation.identity_id for observation in operation.face_observations.get(asset_id, ())
        }
        if len(speaker_ids) != 1 or len(face_ids) != 1:
            return
        speaker_id, face_id = next(iter(speaker_ids)), next(iter(face_ids))
        plan = self._store.identity_link_plan(speaker_id, face_id)
        if plan is None:
            return
        memories: tuple[StoredMemory, ...] = ()
        embeddings: tuple[StoredEmbedding, ...] = ()
        if self._index_speech and plan.source_id == speaker_id:
            memory_ids = self._store.speaker_memory_ids(plan.source_id)
            if memory_ids:
                stored = self._store.read_memories(memory_ids)
                indexed = tuple(
                    memory
                    for memory in stored
                    if any(
                        f"[speech identities:{asset.asset_id}]\n" in memory.content
                        for asset in memory.assets
                        if asset.modality in {"audio", "video"}
                    )
                )
                if indexed:
                    memories, embeddings = self._refresh_speaker_memories(
                        indexed,
                        speaker_id=plan.target_id,
                        speaker_name=plan.name,
                        previous_speaker_id=plan.source_id,
                        update_operation=False,
                        operation=operation,
                    )
        if (
            self._store.link_identities(
                speaker_id,
                face_id,
                expected=plan,
                memories=memories,
                embeddings=embeddings,
            )
            is None
        ):
            return
        if embeddings:
            self._drain_outbox()
        linked_ids = {plan.target_id, plan.source_id}
        for cached_asset, segments in tuple(operation.speech_segments.items()):
            operation.speech_segments[cached_asset] = tuple(
                replace(
                    segment,
                    speaker_id=plan.target_id,
                    speaker_name=plan.name,
                )
                if segment.speaker_id in linked_ids
                else segment
                for segment in segments
            )
        for cached_asset, observations in tuple(operation.face_observations.items()):
            operation.face_observations[cached_asset] = tuple(
                replace(
                    observation,
                    identity_id=plan.target_id,
                    identity_name=plan.name,
                )
                if observation.identity_id in linked_ids
                else observation
                for observation in observations
            )

    def _analyze_faces(
        self,
        assets: Sequence[AssetRef],
    ) -> tuple[FaceAnalysis, ...]:
        if not isinstance(self._face_analyzer, FaceBackend):
            raise ModelError("no face backend is configured")
        with self._model_trace(
            "face",
            "face_recognition",
            model=self._face_model,
            batch_size=len(assets),
            modalities=(cast(Modality, asset.modality) for asset in assets),
        ):
            try:
                analyses = self._face_analyzer.analyze(assets)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to analyze face input") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, FaceAnalysis) for analysis in analyses
            ):
                raise ModelError("face model returned invalid output")
            return tuple(analyses)

    def _transcribable_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        supported = {modality.value for modality in self._transcription_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"audio", "video"} and asset.modality in supported
            }.values()
        )

    def _answer_speech_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if not isinstance(self._transcriber, SpeechBackend):
            return ()
        return self._transcribable_assets(assets)

    def _derives_transcripts(self, assets: Sequence[StoredAsset]) -> bool:
        # A SpeechBackend indexes the same text through the explicit `index_speech` opt-in, so its
        # add-time analysis cost stays behind that flag instead of being taken twice.
        return not isinstance(self._transcriber, SpeechBackend) and bool(
            self._transcribable_assets(assets)
        )

    def _with_speech_identities(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        speech_assets = self._answer_speech_assets(prepared.assets)
        self._recognize_speech(speech_assets, operation, reversible=True)
        base = _without_speech_identities(
            prepared.text,
            tuple(asset.asset_id for asset in speech_assets),
        )
        text = _speech_identity_text(base, speech_assets, operation.speech_segments)
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise ModelError("speaker identity evidence exceeded the supported text length")
        return replace(prepared, text=text)

    def _recognize_speech(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
        *,
        reversible: bool = False,
    ) -> None:
        if not isinstance(self._transcriber, SpeechBackend):
            raise ModelError("configured transcription backend cannot analyze speakers")
        speech_assets = tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"audio", "video"}
                and asset.asset_id not in operation.speech_segments
            }.values()
        )
        missing = []
        with (
            self._trace("mindbridge.storage.lookup", kind="stage"),
            _translate_storage_errors("read cached speaker recognition"),
        ):
            for asset in speech_assets:
                segments = self._store.read_speech(
                    asset.asset_id,
                    space_id=self._transcription_space,
                )
                if segments is None:
                    missing.append(asset)
                else:
                    operation.speech_segments[asset.asset_id] = segments
        if missing:
            analyses = self._analyze_speech(tuple(self._asset_ref(asset) for asset in missing))
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                _translate_storage_errors("persist speaker recognition"),
            ):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    if reversible:
                        segments, rollback = self._store.write_speech_reversible(
                            asset.asset_id,
                            analysis,
                            model_id=self._transcriber.transcription_model,
                            space_id=self._transcription_space,
                            minimum_similarity=self._speaker_similarity,
                            minimum_margin=self._speaker_margin,
                        )
                        if rollback is not None:
                            operation.speech_rollbacks.append(rollback)
                    else:
                        segments = self._store.write_speech(
                            asset.asset_id,
                            analysis,
                            model_id=self._transcriber.transcription_model,
                            space_id=self._transcription_space,
                            minimum_similarity=self._speaker_similarity,
                            minimum_margin=self._speaker_margin,
                        )
                    operation.speech_segments[asset.asset_id] = segments
                written = tuple(
                    segment
                    for asset in missing
                    for segment in operation.speech_segments[asset.asset_id]
                )
                _record_identity_matching(
                    len(written),
                    sum(1 for item in written if item.identity_score is not None),
                )
        for asset in speech_assets:
            segments = operation.speech_segments[asset.asset_id]
            operation.transcripts[asset.asset_id] = "\n".join(segment.text for segment in segments)

    def _with_audio_transcripts(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        self._cache_audio_transcripts(prepared.assets, operation)
        assets = tuple(
            replace(asset, transcript=operation.transcripts[asset.asset_id])
            if asset.asset_id in operation.transcripts
            else asset
            for asset in prepared.assets
        )
        text = _derived_text(
            prepared.text,
            tuple(asset for asset in assets if asset.asset_id not in operation.speech_segments),
        )
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise ModelError("audio transcription exceeded the supported text length")
        return replace(prepared, text=text, assets=assets)

    def _cache_audio_transcripts(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
    ) -> None:
        speech = self._transcribable_assets(assets)
        for asset in speech:
            if asset.transcript is not None:
                operation.transcripts.setdefault(asset.asset_id, asset.transcript)
        missing = tuple(
            dict.fromkeys(
                asset.asset_id for asset in speech if asset.asset_id not in operation.transcripts
            )
        )
        if missing:
            by_id = {asset.asset_id: asset for asset in speech}
            refs = tuple(self._asset_ref(by_id[asset_id]) for asset_id in missing)
            try:
                transcribe = getattr(self._transcriber, "transcribe", None)
                if callable(transcribe):
                    model = getattr(self._transcriber, "transcription_model", None)
                    with self._model_trace(
                        "transcription",
                        "transcription",
                        model=model if isinstance(model, str) else None,
                        batch_size=len(refs),
                        modalities=(cast(Modality, asset.modality) for asset in refs),
                    ):
                        mark_model_requests(1)
                        generated = transcribe(refs)
                else:
                    analyses = self._analyze_speech(refs)
                    generated = tuple(
                        "\n".join(turn.text for turn in analysis.turns) for analysis in analyses
                    )
                    operation.speech_updates.update(zip(missing, analyses, strict=True))
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to transcribe audio input") from error
            if len(generated) != len(refs) or any(
                not isinstance(transcript, str) for transcript in generated
            ):
                raise ModelError("transcription model returned invalid output")
            cached = tuple(
                (asset_id, transcript.strip())
                for asset_id, transcript in zip(missing, generated, strict=True)
            )
            operation.transcripts.update(cached)
            operation.transcript_updates.update(cached)

    def _persist_transcripts(self, operation: _OperationAssets) -> None:
        speech_ids = operation.speech_updates.keys()
        updates = tuple(
            (asset_id, transcript)
            for asset_id, transcript in operation.transcript_updates.items()
            if asset_id in operation.persisted and asset_id not in speech_ids
        )
        speech = tuple(
            (asset_id, analysis)
            for asset_id, analysis in operation.speech_updates.items()
            if asset_id in operation.persisted
        )
        if updates or speech:
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                self._write_lock,
                _translate_storage_errors("cache audio transcripts"),
            ):
                if updates:
                    self._store.set_asset_transcripts(updates)
                if speech and isinstance(self._transcriber, SpeechBackend):
                    for asset_id, analysis in speech:
                        self._store.write_speech(
                            asset_id,
                            analysis,
                            model_id=self._transcriber.transcription_model,
                            space_id=self._transcription_space,
                            minimum_similarity=self._speaker_similarity,
                            minimum_margin=self._speaker_margin,
                        )

    def _analyze_speech(
        self,
        assets: Sequence[AssetRef],
    ) -> tuple[SpeechAnalysis, ...]:
        if not isinstance(self._transcriber, SpeechBackend):
            raise ModelError("configured transcription backend cannot analyze speakers")
        with self._model_trace(
            "transcription",
            "transcription",
            model=self._transcriber.transcription_model,
            batch_size=len(assets),
            modalities=(cast(Modality, asset.modality) for asset in assets),
        ):
            mark_model_requests(1)
            try:
                analyses = self._transcriber.analyze(assets)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to analyze speech input") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, SpeechAnalysis) for analysis in analyses
            ):
                raise ModelError("speech model returned invalid output")
            return tuple(analyses)

    def _resolved_model_input(self, prepared: _PreparedContent) -> ModelInput:
        return ModelInput(
            text=prepared.text,
            assets=tuple(self._asset_ref(asset) for asset in prepared.assets),
        )

    def _embed_document_parts(
        self,
        parts: Sequence[tuple[_PreparedMemory, int, ModelInput]],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[_PreparedMemory, int, ModelInput], ...]]:
        """Embed one write's retrieval keys, degrading a key the model cannot carry.

        A memory whose media exceeds what the embedding model accepts inline used to fail the
        whole write, which discards the memory rather than the route that could not take it.
        `_embedding_inputs` already produces one key per atomic part, so the oversized key can be
        dropped while the rest of the memory is stored and stays retrievable. A memory left with
        no key at all still fails, because then nothing would find it.
        """
        parts = tuple(parts)
        try:
            return self._embed(
                tuple(model_input for _memory, _part, model_input in parts),
                task=EmbedTask.DOCUMENT,
            ), parts
        except ModelError as error:
            if error.reason != _PAYLOAD_TOO_LARGE or len(parts) < 2:
                raise
        vectors: builtins.list[tuple[float, ...]] = []
        kept: builtins.list[tuple[_PreparedMemory, int, ModelInput]] = []
        for entry in parts:
            try:
                vectors.extend(self._embed((entry[2],), task=EmbedTask.DOCUMENT))
            except ModelError as error:
                if error.reason != _PAYLOAD_TOO_LARGE:
                    raise
                continue
            kept.append(entry)
        embedded = {entry[0].memory_id for entry in kept}
        unreachable = tuple(
            entry[0].memory_id for entry in parts if entry[0].memory_id not in embedded
        )
        if unreachable:
            raise ModelError(
                "every retrieval key for this memory exceeds what the embedding model accepts "
                "inline; supply smaller media or an embedding backend that uploads it",
                reason=_PAYLOAD_TOO_LARGE,
                subject=unreachable[0],
            )
        _record_elided_parts(len(parts) - len(kept))
        return tuple(vectors), tuple(kept)

    def _embed(
        self,
        inputs: Sequence[ModelInput],
        *,
        task: EmbedTask,
    ) -> tuple[tuple[float, ...], ...]:
        with self._model_trace(
            "embedding",
            "embeddings",
            model=self._embedding_model,
            batch_size=len(inputs),
            modalities=(modality for value in inputs for modality in value.modalities),
        ):
            mark_model_requests(1)
            try:
                vectors = self._embedder.embed(inputs, task=task)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to embed memory input") from error
            if len(vectors) != len(inputs):
                raise ModelError("embedding model returned the wrong number of vectors")
            return tuple(
                _normalized_vector(vector, self._embedding_dimension) for vector in vectors
            )

    def _answer(  # noqa: C901 - streaming and non-streaming validation share one model span
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> AnswerResult:
        if self._answerer is None:
            raise ModelError(
                "answer backend is not configured",
                reason="backend_not_configured",
                stage="generate",
            )
        model = getattr(self._answerer, "generation_model", None)
        with self._model_trace(
            "generation",
            "chat",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(
                modality
                for value in (question, *(ModelInput(hit.content, hit.assets) for hit in hits))
                for modality in value.modalities
            ),
        ):
            mark_model_requests(1)
            try:
                if isinstance(self._answerer, StreamingGenerationBackend):
                    started = perf_counter()
                    parts: builtins.list[str] = []
                    stream = iter(self._answerer.stream_answer(question, hits))
                    used_hits: object = None
                    while True:
                        try:
                            part = next(stream)
                        except StopIteration as completed:
                            used_hits = completed.value
                            break
                        if not isinstance(part, str):
                            raise ModelError("generation model returned an invalid answer chunk")
                        if part:
                            if not parts and current_model_request_count():
                                trace.get_current_span().set_attribute(
                                    MODEL_TTFT, perf_counter() - started
                                )
                            parts.append(part)
                    answer = "".join(parts)
                    if not answer.strip():
                        raise ModelError("generation model returned an invalid answer")
                    if isinstance(used_hits, AnswerResult):
                        if used_hits.answer != answer:
                            raise ModelError("generation model returned an invalid answer")
                        result = used_hits
                    elif used_hits is None:
                        grounded = tuple(hits)
                    elif isinstance(used_hits, tuple) and all(
                        isinstance(hit, SearchHit) for hit in used_hits
                    ):
                        grounded = used_hits
                        abstention_reason = getattr(used_hits, "abstention_reason", None)
                        if abstention_reason is not None and not isinstance(
                            abstention_reason, AbstentionReason
                        ):
                            raise ModelError("generation model returned invalid abstention status")
                    else:
                        raise ModelError("generation model returned invalid grounding hits")
                    if not isinstance(used_hits, AnswerResult):
                        reason = abstention_reason if isinstance(used_hits, tuple) else None
                        result = AnswerResult(
                            answer=answer,
                            hits=grounded,
                            abstained=reason is not None,
                            abstention_reason=reason,
                        )
                else:
                    result = self._answerer.answer(question, hits)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError("failed to generate a grounded answer") from error
            if not isinstance(result, AnswerResult):
                raise ModelError("generation model returned an invalid answer")
            return result

    def _memory_record(self, memory: StoredMemory) -> MemoryRecord:
        return MemoryRecord(
            id=memory.memory_id,
            content=memory.content,
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=_metadata_from_json(memory.metadata_json),
            assets=tuple(self._asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
        )

    def _search_hit(self, memory: StoredMemory, relevance: float) -> SearchHit:
        return SearchHit(
            id=memory.memory_id,
            content=memory.content,
            score=max(0.0, min(1.0, relevance)),
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=_metadata_from_json(memory.metadata_json),
            assets=tuple(self._asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
        )

    def _asset_ref(self, asset: StoredAsset) -> AssetRef:
        with _translate_storage_errors("resolve local media"):
            path = self._assets.resolve(asset)
        return AssetRef(
            id=asset.asset_id,
            modality=Modality(asset.modality),
            media_type=asset.mime_type,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            name=asset.name,
            path=path,
        )

    def _lease_assets(
        self,
        assets: Sequence[StoredAsset],
        leased: builtins.list[StoredAsset],
    ) -> None:
        unique = tuple({asset.asset_id: asset for asset in assets}.values())
        if not unique:
            return
        try:
            self._assets.acquire(unique)
        except AssetStoreError as error:
            raise StorageError("failed to lease local media") from error
        leased.extend(unique)

    def _queue_asset_cleanup(self, assets: Sequence[StoredAsset]) -> None:
        if not assets:
            return
        with self._lifecycle:
            for asset in assets:
                self._pending_asset_cleanup.setdefault(asset.asset_id, asset)

    def _cleanup_pending_assets(self) -> None:
        with self._lifecycle:
            assets = tuple(self._pending_asset_cleanup.values())
            for asset in assets:
                self._pending_asset_cleanup.pop(asset.asset_id, None)
        if not assets:
            return
        remaining = 0
        try:
            asset_ids = tuple(asset.asset_id for asset in assets)
            with _translate_storage_errors("check temporary media ownership"):
                persisted_assets = self._store.read_assets(asset_ids)
                unreferenced_assets = self._store.read_unreferenced_assets(asset_ids)
            persisted = {asset.asset_id for asset in persisted_assets}
            unreferenced = {asset.asset_id: asset for asset in unreferenced_assets}
            for index, asset in enumerate(assets):
                remaining = index
                asset_id = asset.asset_id
                if asset_id in persisted and asset_id not in unreferenced:
                    remaining = index + 1
                    continue
                deleted = self._assets.delete_if_unleased(unreferenced.get(asset_id, asset))
                if not deleted:
                    self._queue_asset_cleanup((asset,))
                    remaining = index + 1
                    continue
                if asset_id in unreferenced:
                    with _translate_storage_errors("delete orphaned media metadata"):
                        if not self._store.delete_asset_if_unreferenced(asset_id):
                            raise StorageError("orphaned media became referenced during cleanup")
                remaining = index + 1
        except AssetStoreError as error:
            self._queue_asset_cleanup(assets[remaining:])
            raise StorageError("failed to clean up orphaned media") from error
        except BaseException:
            self._queue_asset_cleanup(assets[remaining:])
            raise

    def _collect_orphan_assets(self, *, scan_physical: bool) -> None:
        while True:
            with _translate_storage_errors("list orphaned media"):
                orphaned = self._store.list_unreferenced_assets(limit=256)
            if not orphaned:
                break
            self._delete_orphan_assets(orphaned)
        if not scan_physical:
            return
        try:
            physical_ids = self._assets.list_ids()
        except AssetStoreError as error:
            raise StorageError("failed to scan local media") from error
        with _translate_storage_errors("reconcile local media"):
            tracked_ids = {asset.asset_id for asset in self._store.read_assets(physical_ids)}
        for asset_id in physical_ids:
            if asset_id in tracked_ids:
                continue
            try:
                self._assets.delete_id(asset_id)
            except AssetStoreError as error:
                raise StorageError("failed to delete untracked local media") from error

    def _delete_orphan_assets(self, orphaned: Sequence[StoredAsset]) -> None:
        for asset in orphaned:
            try:
                self._assets.delete(asset)
            except AssetStoreError as error:
                raise StorageError("failed to delete orphaned media") from error
            with _translate_storage_errors("delete orphaned media metadata"):
                self._store.delete_asset_if_unreferenced(asset.asset_id)

    def _drain_outbox(self) -> None:
        """Apply current SQLite truth, then acknowledge the exact durable operation batch."""
        with self._trace("mindbridge.index.sync", kind="stage"):
            self._apply_outbox()

    def _apply_outbox(self) -> None:
        while True:
            with _translate_storage_errors("read the search-index outbox"):
                operations = self._store.pending_index_operations(limit=_OUTBOX_BATCH_SIZE)
            if not operations:
                return
            last_by_embedding = {operation.embedding_id: operation for operation in operations}
            current = sorted(
                last_by_embedding.values(), key=lambda operation: operation.operation_id
            )
            with _translate_storage_errors("hydrate the search-index outbox"):
                hydrated = self._store.read_index_documents(
                    tuple(operation.embedding_id for operation in current)
                )
                identity_memory_ids = tuple(
                    dict.fromkeys(
                        document.embedding.memory_id
                        for document in hydrated
                        if "[speech identities:" in document.content
                    )
                )
                memories = self._store.read_memories(identity_memory_ids)
            by_id = {document.embedding.embedding_id: document for document in hydrated}
            asset_ids = {
                memory.memory_id: frozenset(asset.asset_id for asset in memory.assets)
                for memory in memories
            }
            documents = [
                _retrieval_document(
                    by_id[operation.embedding_id],
                    asset_ids.get(
                        by_id[operation.embedding_id].embedding.memory_id,
                        frozenset(),
                    ),
                )
                for operation in current
                if operation.embedding_id in by_id
            ]
            deleted_ids = [
                operation.embedding_id
                for operation in current
                if operation.embedding_id not in by_id
            ]
            with _translate_index_errors("update the search index"):
                if deleted_ids:
                    self._index.delete(deleted_ids)
                if documents:
                    self._index.upsert(documents)
                self._index.flush()
                self._index.optimize_if_needed()
            with _translate_storage_errors("acknowledge the search-index outbox"):
                acknowledged = self._store.acknowledge_index_operations(operations)
            if acknowledged != len(operations):
                raise StorageError("search-index outbox changed while it was being acknowledged")

    def _index_documents(self) -> Iterator[IndexDocument]:
        after: tuple[datetime, str] | None = None
        while True:
            with _translate_storage_errors("read memories for reindexing"):
                memories = self._store.list_memories(limit=_REINDEX_PAGE_SIZE, after=after)
            if not memories:
                return
            asset_ids = {
                memory.memory_id: frozenset(asset.asset_id for asset in memory.assets)
                for memory in memories
            }
            with _translate_storage_errors("hydrate memories for reindexing"):
                yield from (
                    _retrieval_document(
                        document,
                        asset_ids[document.embedding.memory_id],
                    )
                    for document in self._store.read_memory_index_documents(
                        tuple(memory.memory_id for memory in memories)
                    )
                )
            last = memories[-1]
            after = (last.created_at, last.memory_id)

    def _reembed_memories(self) -> None:
        after: tuple[datetime, str] | None = None
        while True:
            with _translate_storage_errors("read memories for embedding migration"):
                memories = self._store.list_memories(limit=_REEMBED_PAGE_SIZE, after=after)
            if not memories:
                return
            parts = tuple(
                (memory, object_part, model_input)
                for memory in memories
                for object_part, model_input in enumerate(
                    self._embedding_inputs(
                        _PreparedContent(
                            text=memory.content,
                            assets=memory.assets,
                            modality=Modality(memory.modality),
                            canonical_parts=((("text", memory.content),) if memory.content else ()),
                        )
                    )
                )
            )
            vectors = self._embed(
                tuple(model_input for _memory, _part, model_input in parts),
                task=EmbedTask.DOCUMENT,
            )
            now = datetime.now(timezone.utc)
            embeddings = tuple(
                StoredEmbedding(
                    embedding_id=_embedding_id(memory.memory_id, object_part),
                    memory_id=memory.memory_id,
                    values=vector,
                    model_id=self._embedding_model,
                    space_id=self._space_id,
                    task=_DOCUMENT_TASK,
                    created_at=now,
                    object_part=object_part,
                    normalized=True,
                )
                for (memory, object_part, _model_input), vector in zip(
                    parts,
                    vectors,
                    strict=True,
                )
            )
            with _translate_storage_errors("replace migrated embeddings"):
                self._store.replace_memory_embeddings(memories, embeddings)
            last = memories[-1]
            after = (last.created_at, last.memory_id)

    def _ensure_store_metadata(self, index_path: Path) -> tuple[bool, bool]:
        expected = {
            _STORE_METADATA_KEYS["model"]: self._embedding_model,
            _STORE_METADATA_KEYS["space"]: self._space_id,
            _STORE_METADATA_KEYS["transcription"]: self._transcription_space,
            _STORE_METADATA_KEYS["dimension"]: str(self._embedding_dimension),
            _STORE_METADATA_KEYS["index"]: self._index_recipe,
        }
        if self._face_analyzer is not None:
            expected[_STORE_METADATA_KEYS["face"]] = self._face_space
            expected[_STORE_METADATA_KEYS["face_analysis"]] = self._face_analysis_space
        rebuild_index = False
        rebuild_embeddings = False
        legacy_embedding_spaces = cast(
            frozenset[str],
            getattr(self._embedder, "_legacy_embedding_spaces", frozenset()),
        )
        with _translate_storage_errors("validate local store metadata"):
            for key, value in expected.items():
                stored = self._store.get_metadata(key)
                if stored is None:
                    if key == _STORE_METADATA_KEYS["index"] and index_path.exists():
                        rebuild_index = True
                        rebuild_embeddings = True
                    else:
                        self._store.set_metadata(key, value)
                elif stored == value:
                    continue
                elif (
                    requires_reembedding := _known_metadata_upgrade(
                        key,
                        stored,
                        legacy_embedding_spaces,
                    )
                ) is not None:
                    rebuild_index = True
                    rebuild_embeddings = rebuild_embeddings or requires_reembedding
                else:
                    raise StorageError(
                        f"local store metadata mismatch for {key}: expected {value!r}, "
                        f"found {stored!r}"
                    )
            if rebuild_index:
                if index_path.exists():
                    shutil.rmtree(index_path)
                if not rebuild_embeddings:
                    self._store.set_metadata(_STORE_METADATA_KEYS["index"], self._index_recipe)
        return rebuild_index, rebuild_embeddings

    def _close_models_and_store(self) -> None:
        resources = _present_resources(
            self._embedder,
            self._transcriber,
            self._face_analyzer,
            self._answerer,
            self._store,
        )
        for resource in _unique_resources(resources):
            with suppress(Exception):
                resource.close()

    def _close_resources(self) -> builtins.list[Exception]:
        failures: builtins.list[Exception] = []
        resources = _present_resources(
            self._embedder,
            self._transcriber,
            self._face_analyzer,
            self._answerer,
            self._index,
            self._store,
        )
        for resource in _unique_resources(resources):
            try:
                resource.close()
            except Exception as error:
                failures.append(error)
        return failures

    def _require_open(self) -> None:
        self._require_owner_process()
        if self._closed or self._closing:
            raise StorageError("Memory is closed")

    @contextmanager
    def _operation(self) -> Iterator[_OperationAssets]:
        self._require_owner_process()
        with self._lifecycle:
            if self._closed or self._closing:
                raise StorageError("Memory is closed")
            self._active_operations += 1
        failed = False
        assets = _OperationAssets(
            leased=[],
            cleanup=[],
            persisted=set(),
            transcripts={},
            transcript_updates={},
            speech_updates={},
            speech_segments={},
            speech_rollbacks=[],
            face_observations={},
        )
        try:
            yield assets
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_error: BaseException | None = None
            self._queue_asset_cleanup(assets.cleanup)
            try:
                self._assets.release(assets.leased)
                with self._lifecycle:
                    needs_cleanup = bool(self._pending_asset_cleanup)
                if needs_cleanup:
                    with self._write_lock:
                        self._cleanup_pending_assets()
            except BaseException as error:
                cleanup_error = error
            finally:
                with self._lifecycle:
                    self._active_operations -= 1
                    self._lifecycle.notify_all()
            if cleanup_error is not None and not failed:
                raise cleanup_error

    def _require_owner_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise StorageError(
                "Memory cannot be used after fork; create a new instance with a different data_dir"
            )


class AsyncMemory:
    """Async facade over the same synchronous local-memory core."""

    def __init__(
        self,
        data_dir: str | Path = ".mindbridge",
        *,
        embedder: EmbeddingBackend,
        answerer: GenerationBackend | None = None,
        transcriber: SpeechBackend | TranscriptionBackend | None = None,
        face_analyzer: FaceBackend | None = None,
        index_speech: bool = _DEFAULT_CONFIG.index_speech,
        index_quantization: IndexQuantization = _DEFAULT_CONFIG.index_quantization,
        minimum_relevance: float = _DEFAULT_CONFIG.minimum_relevance,
        ambiguity_margin: float = _DEFAULT_CONFIG.ambiguity_margin,
        evidence_budget_chars: int | None = _DEFAULT_CONFIG.evidence_budget_chars,
        decay_half_life_days: float | None = _DEFAULT_CONFIG.decay_half_life_days,
        speaker_similarity: float = _DEFAULT_CONFIG.speaker_similarity,
        speaker_margin: float = _DEFAULT_CONFIG.speaker_margin,
        face_similarity: float = _DEFAULT_CONFIG.face_similarity,
        face_margin: float = _DEFAULT_CONFIG.face_margin,
        tracer: Tracer | None = None,
    ) -> None:
        self._memory = Memory(
            data_dir=data_dir,
            embedder=embedder,
            answerer=answerer,
            transcriber=transcriber,
            face_analyzer=face_analyzer,
            index_speech=index_speech,
            index_quantization=index_quantization,
            minimum_relevance=minimum_relevance,
            ambiguity_margin=ambiguity_margin,
            evidence_budget_chars=evidence_budget_chars,
            decay_half_life_days=decay_half_life_days,
            speaker_similarity=speaker_similarity,
            speaker_margin=speaker_margin,
            face_similarity=face_similarity,
            face_margin=face_margin,
            tracer=tracer,
        )

    @classmethod
    def from_plugins(
        cls,
        data_dir: str | Path = ".mindbridge",
        *,
        plugins: MemoryPlugins,
        config: MemoryConfig | None = None,
        tracer: Tracer | None = None,
    ) -> AsyncMemory:
        """Open async memory from an explicit capability bundle and local policy."""
        if not isinstance(plugins, MemoryPlugins):
            raise ValidationError("plugins must be a MemoryPlugins value")
        if config is None:
            config = MemoryConfig()
        elif not isinstance(config, MemoryConfig):
            raise ValidationError("config must be a MemoryConfig value")
        return cls(
            data_dir,
            embedder=plugins.embedder,
            answerer=plugins.answerer,
            transcriber=plugins.transcriber,
            face_analyzer=plugins.face_analyzer,
            index_speech=config.index_speech,
            index_quantization=config.index_quantization,
            minimum_relevance=config.minimum_relevance,
            ambiguity_margin=config.ambiguity_margin,
            evidence_budget_chars=config.evidence_budget_chars,
            decay_half_life_days=config.decay_half_life_days,
            speaker_similarity=config.speaker_similarity,
            speaker_margin=config.speaker_margin,
            face_similarity=config.face_similarity,
            face_margin=config.face_margin,
            tracer=tracer,
        )

    @classmethod
    def from_config(
        cls,
        config: MindBridgeConfig | Mapping[str, object],
        *,
        tracer: Tracer | None = None,
    ) -> AsyncMemory:
        """Open async memory from validated declarative configuration."""
        resolved = resolve_memory_config(config)
        try:
            return cls.from_plugins(
                resolved.data_dir,
                plugins=resolved.plugins,
                config=resolved.settings,
                tracer=tracer,
            )
        except BaseException:
            resolved.close()
            raise

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()

    async def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.add,
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
        )

    async def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]:
        return await asyncio.to_thread(
            self._memory.add_many,
            contents,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
        )

    async def add_stream(
        self,
        contents: AsyncIterable[ContentInput | StreamInput],
    ) -> AsyncIterator[MemoryRecord]:
        """Add an async omni stream one durable, searchable observation at a time."""
        if not isinstance(contents, AsyncIterable):
            raise ValidationError("contents must be an async iterable of memory inputs")
        iterator = aiter(contents)
        index = 0
        while True:
            try:
                content = await anext(iterator)
            except StopAsyncIteration:
                return
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            try:
                if isinstance(content, StreamInput):
                    record = await self.add(
                        content.content,
                        occurred_at=content.occurred_at,
                        occurred_end=content.occurred_end,
                        metadata=content.metadata,
                        memory_type=content.memory_type,
                    )
                else:
                    record = await self.add(content)
            except MindBridgeError as error:
                if error.subject is None:
                    error.subject = f"contents[{index}]"
                raise
            yield record
            index += 1

    async def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> tuple[SearchHit, ...]:
        return await asyncio.to_thread(
            self._memory.search,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
        )

    async def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> TracedSearchResult:
        return await asyncio.to_thread(
            self._memory.search_with_trace,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
        )

    async def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> AnswerResult:
        return await asyncio.to_thread(
            self._memory.ask,
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
        )

    async def get(self, memory_id: str) -> MemoryRecord:
        return await asyncio.to_thread(self._memory.get, memory_id)

    async def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        return await asyncio.to_thread(self._memory.speech, memory_id)

    async def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        return await asyncio.to_thread(self._memory.faces, memory_id)

    async def register_speaker(self, speaker_id: str, name: str) -> None:
        await asyncio.to_thread(self._memory.register_speaker, speaker_id, name)

    async def register_identity(self, identity_id: str, name: str) -> None:
        await asyncio.to_thread(self._memory.register_identity, identity_id, name)

    async def reinforce(self, memory_ids: Sequence[str]) -> int:
        return await asyncio.to_thread(self._memory.reinforce, memory_ids)

    async def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        return await asyncio.to_thread(self._memory.list, limit=limit, cursor=cursor)

    async def delete(self, memory_id: str) -> bool:
        return await asyncio.to_thread(self._memory.delete, memory_id)

    async def reindex(self) -> int:
        return await asyncio.to_thread(self._memory.reindex)

    async def optimize(self) -> None:
        await asyncio.to_thread(self._memory.optimize)

    async def close(self) -> None:
        await asyncio.to_thread(self._memory.close)


class AsyncOmniPrefetch:
    """Coalesce evolving omni query snapshots into one in-flight search per turn."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> None:
        _limit(limit, maximum=100)
        occurred_from, occurred_until = _search_occurrence_range(
            occurred_from,
            occurred_until,
        )
        self._memory = memory
        self._limit = limit
        self._memory_type = _optional_memory_type(memory_type)
        self._reference_at = _reference_at(reference_at)
        self._occurred_from = occurred_from
        self._occurred_until = occurred_until
        self._revision = 0
        self._submitted: tuple[int, ContentInput] | None = None
        self._pending: tuple[int, ContentInput] | None = None
        self._latest: PrefetchResult | None = None
        self._failure: tuple[int, Exception] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def latest(self) -> PrefetchResult | None:
        """Return the newest completed result without waiting."""
        return self._latest

    def submit(self, query: ContentInput) -> int:
        """Queue a complete current snapshot, replacing any not-yet-started snapshot."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        loop = asyncio.get_running_loop()
        snapshot = _snapshot_content(query)
        if self._submitted is not None and self._submitted[1] == snapshot:
            failed = self._failure is not None and self._failure[0] == self._submitted[0]
            if not failed:
                return self._submitted[0]
        self._revision += 1
        revision = self._revision
        self._submitted = (revision, snapshot)
        self._pending = self._submitted
        self._failure = None
        if self._worker is None:
            self._worker = loop.create_task(self._run())
        return revision

    async def finalize(self, query: ContentInput | None = None) -> PrefetchResult:
        """Finish the turn and return a result for the exact final snapshot."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        if query is None:
            if self._submitted is None:
                raise ValidationError("prefetch has no submitted query")
            target = self._submitted[0]
        else:
            target = self.submit(query)
        self._closed = True
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)
        if self._latest is not None and self._latest.revision == target:
            return self._latest
        if self._failure is not None and self._failure[0] == target:
            raise self._failure[1]
        raise RuntimeError("prefetch completed without its final revision")

    async def close(self) -> None:
        """Discard queued snapshots and drain the one search that may already be running."""
        self._closed = True
        self._pending = None
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)

    async def _run(self) -> None:
        try:
            while self._pending is not None:
                revision, query = self._pending
                self._pending = None
                try:
                    hits = await self._memory.search(
                        query,
                        limit=self._limit,
                        memory_type=self._memory_type,
                        reference_at=self._reference_at,
                        occurred_from=self._occurred_from,
                        occurred_until=self._occurred_until,
                    )
                except Exception as error:
                    self._failure = (revision, error)
                    continue
                self._latest = PrefetchResult(revision=revision, hits=hits)
        finally:
            self._worker = None


@contextmanager
def _staged(span: AbstractContextManager[Span], stage: str) -> Iterator[Span]:
    """Name the failing stage on public errors so an agent never parses prose to find it.

    The innermost span wins: an adapter that already classified its own failure keeps its stage.
    """
    with span as opened:
        try:
            yield opened
        except MindBridgeError as error:
            if error.stage is None:
                error.stage = stage
            raise


def _content_atoms(content: ContentInput) -> tuple[ContentAtom, ...]:
    if isinstance(content, (str, Path, Blob, AssetRef)):
        return (content,)
    if isinstance(content, bytes) or not isinstance(content, Sequence):
        raise ValidationError("content must be text, media, or an ordered sequence of them")
    atoms = tuple(content)
    if not atoms:
        raise ValidationError("content must not be empty")
    if len(atoms) > _MAX_CONTENT_PARTS:
        raise ValidationError(f"content must not exceed {_MAX_CONTENT_PARTS} parts")
    if any(not isinstance(atom, (str, Path, Blob, AssetRef)) for atom in atoms):
        raise ValidationError("content contains an unsupported input value")
    return atoms


def _snapshot_content(content: ContentInput) -> ContentInput:
    atoms = _content_atoms(content)
    if any(isinstance(atom, Path) for atom in atoms):
        raise ValidationError("prefetch paths are mutable; use Blob or AssetRef")
    return atoms[0] if len(atoms) == 1 else atoms


def _prepare_memory(
    content: _PreparedContent,
    *,
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    metadata: Mapping[str, object] | None,
    memory_type: MemoryType,
) -> _PreparedMemory:
    normalized_occurred_at = _occurred_at(occurred_at)
    normalized_occurred_end = _occurred_end(normalized_occurred_at, occurred_end)
    normalized_memory_type = _memory_type(memory_type)
    metadata_json = _metadata_json(metadata)
    identity: dict[str, object] = {
        "parts": content.canonical_parts,
        "metadata": json.loads(metadata_json),
        "occurred_at": (
            None if normalized_occurred_at is None else _datetime_text(normalized_occurred_at)
        ),
    }
    if normalized_occurred_end is not None:
        identity["occurred_end"] = _datetime_text(normalized_occurred_end)
    if normalized_memory_type is not MemoryType.SEMANTIC:
        identity["memory_type"] = normalized_memory_type.value
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _PreparedMemory(
        memory_id=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        content=content,
        metadata_json=metadata_json,
        occurred_at=normalized_occurred_at,
        occurred_end=normalized_occurred_end,
        memory_type=normalized_memory_type,
    )


def _media_hint(name: str | None, media_type: str | None) -> tuple[Modality, str]:
    resolved = media_type
    if resolved is None and name:
        resolved = _MEDIA_TYPES.get(Path(name).suffix.casefold())
    if resolved is None:
        raise ValidationError(
            "media type could not be inferred; provide a known file suffix or media_type"
        )
    normalized = resolved.strip().lower()
    top_level = normalized.split("/", 1)[0]
    try:
        modality = Modality(top_level)
    except ValueError:
        raise ValidationError("media_type must be image, video, or audio") from None
    if modality not in {Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}:
        raise ValidationError("media_type must be image, video, or audio")
    return modality, normalized


def _memory_modality(assets: Sequence[StoredAsset]) -> Modality:
    kinds = {asset.modality for asset in assets}
    if not kinds:
        return Modality.TEXT
    if len(kinds) == 1:
        return Modality(next(iter(kinds)))
    return Modality.OMNI


def _text_content(text: str) -> _PreparedContent:
    return _PreparedContent(
        text=text,
        assets=(),
        modality=Modality.TEXT,
        canonical_parts=(("text", text),),
    )


def _contextual_text_keys(text: str) -> tuple[str, ...]:
    if len(text) <= _TEXT_KEY_CHARACTERS:
        return (text,)
    context = text.splitlines()[0][:_TEXT_KEY_CONTEXT].strip()
    step = _TEXT_KEY_CHARACTERS - _TEXT_KEY_OVERLAP
    keys = []
    for start in range(0, len(text), step):
        chunk = text[start : start + _TEXT_KEY_CHARACTERS].strip()
        if not chunk:
            continue
        if context and not chunk.startswith(context):
            chunk = f"{context}\n\n{chunk}"
        keys.append(chunk)
        if start + _TEXT_KEY_CHARACTERS >= len(text):
            break
    return tuple(keys)


def _asset_content(asset: StoredAsset) -> _PreparedContent:
    return _PreparedContent(
        text="",
        assets=(asset,),
        modality=Modality(asset.modality),
        canonical_parts=(("asset", asset.sha256),),
    )


def _prepared_modalities(prepared: _PreparedContent) -> frozenset[Modality]:
    values = {Modality(asset.modality) for asset in prepared.assets}
    if prepared.text:
        values.add(Modality.TEXT)
    return frozenset(values)


def _fallback_unsupported(
    prepared: _PreparedContent,
    supported: frozenset[Modality],
    operation: str,
) -> frozenset[Modality]:
    unsupported = _prepared_modalities(prepared) - supported
    fatal = set(unsupported - {Modality.AUDIO})
    if Modality.AUDIO in unsupported and Modality.TEXT not in supported:
        fatal.add(Modality.AUDIO)
    if fatal:
        names = ", ".join(sorted(modality.value for modality in fatal))
        raise ModelError(
            f"configured {operation} model does not support: {names}",
            reason="unsupported_modality",
        )
    return unsupported


def _require_audio_transcription(capabilities: frozenset[Modality]) -> None:
    if Modality.AUDIO not in capabilities:
        raise ModelError(
            "audio fallback requires a transcription model with audio capability",
            reason="unsupported_modality",
        )


def _derived_text(text: str, assets: Sequence[StoredAsset]) -> str:
    sections = [text] if text else []
    seen: set[str] = set()
    for asset in assets:
        # A transcript is only ever cached for an asset the transcriber declared, so its presence
        # is the routing decision; a second modality comparison here would discard video speech.
        transcript = asset.transcript
        if not transcript or asset.asset_id in seen:
            continue
        seen.add(asset.asset_id)
        # Modality-neutral on purpose. This text becomes `memory_records.content`, which is the
        # BM25 document, so every word in the marker is a term any query matches for free -- and a
        # lexical match alone clears `minimum_relevance`. Naming the modality here labelled video
        # speech "audio" and handed every video memory a free match on that word; deriving it from
        # `asset.modality` would only move the free match to the commoner word. The modality is
        # already published on the record's assets, so nothing is lost by leaving it out.
        marker = f"[transcript:{asset.asset_id}]"
        identity_marker = f"[speech identities:{asset.asset_id}]\n"
        if marker not in text and identity_marker not in text:
            sections.append(f"{marker}\n{transcript}")
    return "\n\n".join(sections)


def _speech_identity_text(
    text: str,
    assets: Sequence[StoredAsset],
    segments_by_asset: Mapping[str, tuple[SpeakerSegment, ...]],
) -> str:
    sections = [text] if text else []
    for asset in dict.fromkeys(asset.asset_id for asset in assets):
        segments = segments_by_asset[asset]
        if not segments:
            continue
        evidence = {
            "asset_id": asset,
            "segments": [
                {
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "text": segment.text,
                    "speaker_id": segment.speaker_id,
                    "speaker_name": segment.speaker_name,
                    "identity_score": segment.identity_score,
                }
                for segment in segments
            ],
        }
        sections.append(
            f"[speech identities:{asset}]\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(sections)


def _face_identity_text(
    text: str,
    assets: Sequence[StoredAsset],
    observations_by_asset: Mapping[str, tuple[FaceObservation, ...]],
) -> str:
    sections = [text] if text else []
    for asset_id in dict.fromkeys(asset.asset_id for asset in assets):
        grouped: dict[str, list[FaceObservation]] = {}
        for observation in observations_by_asset[asset_id]:
            grouped.setdefault(observation.identity_id, []).append(observation)
        if not grouped:
            continue
        identities = []
        for identity_id, observations in grouped.items():
            times = tuple(
                observation.observed_at_ms
                for observation in observations
                if observation.observed_at_ms is not None
            )
            scores = tuple(
                observation.identity_score
                for observation in observations
                if observation.identity_score is not None
            )
            identities.append(
                {
                    "identity_id": identity_id,
                    "identity_name": observations[0].identity_name,
                    "first_observed_at_ms": min(times) if times else None,
                    "last_observed_at_ms": max(times) if times else None,
                    "observation_count": len(observations),
                    "representative_box": observations[0].bounding_box,
                    "max_identity_score": max(scores) if scores else None,
                }
            )
        evidence = {"asset_id": asset_id, "identities": identities}
        sections.append(
            f"[face identities:{asset_id}]\n"
            + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(sections)


def _without_speech_identities(text: str, asset_ids: Sequence[str]) -> str:
    markers = tuple(f"[speech identities:{asset_id}]\n" for asset_id in asset_ids)
    return "\n\n".join(section for section in text.split("\n\n") if not section.startswith(markers))


def _retrieval_content(prepared: _PreparedContent) -> _PreparedContent:
    if "[speech identities:" not in prepared.text:
        return prepared
    asset_ids = frozenset(asset.asset_id for asset in prepared.assets)
    return replace(
        prepared,
        text=_retrieval_text(prepared.text, asset_ids),
        canonical_parts=tuple(
            (kind, _retrieval_text(value, asset_ids))
            if kind == "text" and "[speech identities:" in value
            else (kind, value)
            for kind, value in prepared.canonical_parts
        ),
    )


def _retrieval_document(
    document: IndexDocument,
    asset_ids: frozenset[str],
) -> IndexDocument:
    if "[speech identities:" not in document.content:
        return document
    return replace(document, content=_retrieval_text(document.content, asset_ids))


def _retrieval_text(text: str, asset_ids: frozenset[str]) -> str:
    sections = []
    for section in text.split("\n\n"):
        marker, separator, payload = section.partition("\n")
        if separator and marker.startswith("[speech identities:") and marker.endswith("]"):
            asset_id = marker.removeprefix("[speech identities:").removesuffix("]")
            if asset_id in asset_ids:
                projected = _speech_retrieval_text(payload, asset_id)
                if projected is not None:
                    sections.append(f"{marker}\n{projected}")
                    continue
        sections.append(section)
    return "\n\n".join(sections)


def _speech_retrieval_text(payload: str, asset_id: str) -> str | None:
    try:
        evidence = json.loads(payload)
        if not isinstance(evidence, dict) or evidence.get("asset_id") != asset_id:
            return None
        segments = evidence["segments"]
        if not isinstance(segments, list):
            return None
        aliases: dict[str, str] = {}
        normalized = []
        changed = False
        for segment in segments:
            if not isinstance(segment, dict) or "speaker_id" not in segment:
                return None
            speaker_id = segment.get("speaker_id")
            if isinstance(speaker_id, str) and speaker_id.startswith("identity_"):
                speaker_id = aliases.setdefault(speaker_id, f"speaker_{len(aliases) + 1}")
                changed = True
            normalized.append({**segment, "speaker_id": speaker_id})
        if not changed:
            return payload
        evidence = {**evidence, "segments": normalized}
        return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    except (KeyError, TypeError, ValueError):
        return None


def _open_store(data_dir: Path) -> LocalStore:
    try:
        return LocalStore(data_dir)
    except DataDirectoryInUseError as error:
        # The path is the caller's own configuration, but it is server state to every transport,
        # so it travels in `subject` instead of the message every surface forwards.
        raise StorageError(
            "the data directory is already in use by another live MindBridge instance",
            reason="data_dir_in_use",
            stage="open",
            subject=str(data_dir),
        ) from error
    except UnsupportedSchemaError as error:
        raise StorageError(str(error), reason="schema_unsupported", stage="open") from error
    except Exception as error:
        raise StorageError(
            "failed to open the local memory store", reason="io_failed", stage="open"
        ) from error


def _transcription_contract(
    transcriber: SpeechBackend | TranscriptionBackend | None,
) -> tuple[frozenset[Modality], str]:
    if transcriber is None:
        return frozenset(), _NO_TRANSCRIPTION_SPACE
    return _modalities(transcriber.transcription_capabilities, "transcription"), _model_text(
        transcriber.transcription_space,
        "transcription space",
    )


def _generation_contract(answerer: GenerationBackend | None) -> frozenset[Modality]:
    return (
        frozenset()
        if answerer is None
        else _modalities(answerer.generation_capabilities, "generation")
    )


def _face_contract(
    analyzer: FaceBackend | None,
) -> tuple[frozenset[Modality], str, str, str]:
    if analyzer is None:
        return frozenset(), "none", _NO_FACE_SPACE, _NO_FACE_SPACE
    capabilities = _modalities(analyzer.face_capabilities, "face")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("face capabilities must contain image or video")
    return (
        capabilities,
        _model_text(analyzer.face_model, "face model"),
        _embedding_space(analyzer.face_space),
        _embedding_space(analyzer.face_analysis_space),
    )


def _embedding_contract(
    embedder: EmbeddingBackend,
) -> tuple[frozenset[Modality], str, str, int]:
    return (
        _modalities(embedder.embedding_capabilities, "embedding"),
        _model_text(embedder.embedding_model, "embedding model"),
        _embedding_space(embedder.embedding_space),
        _positive_dimension(embedder.embedding_dimension, "embedder.embedding_dimension"),
    )


def _embedding_space(value: object) -> str:
    space = _model_text(value, "embedding space")
    if "'" in space or "\\" in space or any(ord(character) < 32 for character in space):
        raise ValidationError("embedding space contains characters unsupported by Zvec")
    return space


def _model_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()


def _positive_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def _unique_resources(resources: Sequence[_Closable]) -> tuple[_Closable, ...]:
    seen: set[int] = set()
    unique: builtins.list[_Closable] = []
    for resource in resources:
        identity = id(resource)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(resource)
    return tuple(unique)


def _present_resources(*resources: _Closable | None) -> tuple[_Closable, ...]:
    return tuple(resource for resource in resources if resource is not None)


def _metadata_json(metadata: Mapping[str, object] | None) -> str:
    if metadata is None:
        return "{}"
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be a mapping")
    copied = dict(metadata)
    if any(not isinstance(key, str) or not key.strip() for key in copied):
        raise ValidationError("metadata keys must be non-empty strings")
    try:
        serialized = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError):
        raise ValidationError("metadata must contain JSON-compatible values") from None
    if len(serialized.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValidationError(f"metadata must not exceed {_MAX_METADATA_BYTES} UTF-8 bytes")
    return serialized


def _metadata_from_json(value: str) -> Mapping[str, object]:
    try:
        decoded: object = json.loads(value)
    except ValueError as error:
        raise StorageError("stored memory metadata is invalid") from error
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise StorageError("stored memory metadata is not an object")
    return cast(dict[str, object], decoded)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if len(normalized) > _MAX_TEXT_CHARACTERS:
        raise ValidationError(f"{name} must not exceed {_MAX_TEXT_CHARACTERS} characters")
    return normalized


def _occurred_at(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("occurred_at must include a timezone")
    return value.astimezone(timezone.utc)


def _search_occurrence_range(
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    bounds = []
    for value, name in (
        (occurred_from, "occurred_from"),
        (occurred_until, "occurred_until"),
    ):
        if value is not None and (
            not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValidationError(f"{name} must include a timezone")
        bounds.append(None if value is None else value.astimezone(timezone.utc))
    start, end = bounds
    if start is not None and end is not None and end <= start:
        raise ValidationError("occurred_until must be later than occurred_from")
    return start, end


def _intersect_occurrence_range(
    temporal_range: tuple[datetime, datetime],
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> tuple[datetime, datetime] | None:
    start = (
        max(temporal_range[0], occurred_from) if occurred_from is not None else temporal_range[0]
    )
    end = (
        min(temporal_range[1], occurred_until) if occurred_until is not None else temporal_range[1]
    )
    return None if end <= start else (start, end)


def _occurrence_overlaps(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> bool:
    if occurred_from is None and occurred_until is None:
        return True
    if occurred_at is None:
        return False
    return (
        occurred_from is None
        or (
            occurred_end > occurred_from
            if occurred_end is not None
            else occurred_at >= occurred_from
        )
    ) and (occurred_until is None or occurred_at < occurred_until)


def _occurred_end(start: datetime | None, value: datetime | None) -> datetime | None:
    end = _occurred_at(value)
    if end is not None and (start is None or end <= start):
        raise ValidationError("occurred_end must be later than occurred_at")
    return end


def _reference_at(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("reference_at must include a timezone")
    return value


def _memory_type(value: object) -> MemoryType:
    if not isinstance(value, MemoryType):
        raise ValidationError("memory_type must be a MemoryType value")
    return value


def _optional_memory_type(value: object) -> MemoryType | None:
    return None if value is None else _memory_type(value)


def _index_quantization(value: object) -> IndexQuantization:
    if not isinstance(value, IndexQuantization):
        raise ValidationError("index_quantization must be an IndexQuantization value")
    return value


def _index_recipe(quantization: IndexQuantization) -> str:
    return f"{_INDEX_RECIPE_PREFIX}:quantization-{quantization.value}"


def _known_metadata_upgrade(
    key: str,
    stored: str,
    legacy_embedding_spaces: frozenset[str],
) -> bool | None:
    if key == _STORE_METADATA_KEYS["space"] and stored in legacy_embedding_spaces:
        return True
    if key == _STORE_METADATA_KEYS["index"] and stored in (
        _LEGACY_INDEX_RECIPES | {_index_recipe(mode) for mode in IndexQuantization}
    ):
        return stored in _LEGACY_INDEX_RECIPES
    return None


def _batch_values(
    values: Sequence[_T | None] | None,
    count: int,
    name: str,
) -> tuple[_T | None, ...]:
    if values is None:
        return (None,) * count
    if isinstance(values, (str, bytes, Mapping)):
        raise ValidationError(f"{name} must contain one value per content")
    try:
        batch = tuple(values)
    except TypeError:
        raise ValidationError(f"{name} must contain one value per content") from None
    if len(batch) != count:
        raise ValidationError(f"{name} must contain one value per content")
    return batch


def _with_reference_time(content: _PreparedContent, reference_at: datetime) -> _PreparedContent:
    note = f"Reference time for relative dates: {reference_at.isoformat(timespec='microseconds')}"
    return replace(content, text=f"{content.text}\n\n{note}" if content.text else note)


def _record_elided_parts(count: int) -> None:
    """Publish how many retrieval keys the embedding model could not carry, so no loss is silent."""
    if count <= 0:
        return
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(EMBEDDING_PARTS_ELIDED, count)


def _record_identity_matching(observed: int, matched: int) -> None:
    """Publish how many identity observations joined an identity that already existed.

    A recognizer whose similarities do not separate the people in front of it still
    returns an identity for every observation, so face and speaker analysis reports
    success while each memory quietly meets a stranger. Nothing else in the write path
    distinguishes that from recognition working, and the difference only shows up much
    later as an answer that cannot follow one person across two memories.

    Zero observations is recorded rather than skipped: a detector whose confidence threshold
    suits posed photographs finds no face at all in wide-angle or egocentric footage, and that
    silence is otherwise indistinguishable from having configured no analyzer.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(IDENTITY_OBSERVATIONS, observed)
        span.set_attribute(IDENTITY_MATCHED, matched)


def _grounding_hits(
    hits: Sequence[SearchHit],
    limit: int,
    *,
    budget_chars: int | None = None,
) -> tuple[SearchHit, ...]:
    queues: dict[Modality, deque[SearchHit]] = {}
    for hit in hits:
        queues.setdefault(hit.modality, deque()).append(hit)
    selected: list[SearchHit] = []
    while queues and len(selected) < limit:
        for modality in tuple(queues):
            selected.append(queues[modality].popleft())
            if not queues[modality]:
                del queues[modality]
            if len(selected) == limit:
                break
    if budget_chars is None:
        return tuple(selected)
    return (*selected, *_budgeted_hits(hits, selected, budget_chars))


def _budgeted_hits(
    hits: Sequence[SearchHit],
    selected: Sequence[SearchHit],
    budget_chars: int,
) -> tuple[SearchHit, ...]:
    """Extend a grounding set down the ranking while the evidence fits one budget.

    The guaranteed hits are never dropped, so this only ever widens what the answer sees. Media
    is charged per modality because an image or video part costs the model far more than its
    record's text; the provider adapter still enforces its own byte ceiling.
    """
    taken = {hit.id for hit in selected}
    used = sum(_evidence_cost(hit) for hit in selected)
    extra: list[SearchHit] = []
    for hit in hits:
        if hit.id in taken:
            continue
        cost = _evidence_cost(hit)
        if used + cost > budget_chars:
            break
        used += cost
        extra.append(hit)
    return tuple(extra)


def _evidence_cost(hit: SearchHit) -> int:
    # `AssetRef.modality` is optional on the public type; an unresolved asset is charged the
    # default rather than being treated as free.
    charges = (
        _DEFAULT_ASSET_EVIDENCE_CHARS
        if asset.modality is None
        else _ASSET_EVIDENCE_CHARS.get(asset.modality, _DEFAULT_ASSET_EVIDENCE_CHARS)
        for asset in hit.assets
    )
    return len(hit.content) + sum(charges)


def _merge_index_hits(*groups: Sequence[IndexHit]) -> tuple[IndexHit, ...]:
    merged: dict[str, IndexHit] = {}
    for group in groups:
        for hit in group:
            current = merged.get(hit.id)
            if current is None:
                merged[hit.id] = hit
            else:
                merged[hit.id] = IndexHit(
                    id=hit.id,
                    relevance=max(current.relevance, hit.relevance),
                    confidence=max(cast(float, current.confidence), cast(float, hit.confidence)),
                    lexical_match=current.lexical_match or hit.lexical_match,
                )
    return tuple(merged.values())


def _search_outcome(
    hits: Sequence[SearchHit],
    candidates: Sequence[RetrievalCandidateTrace] | None,
    *,
    candidate_limit: int,
    exhaustive: bool,
    ambiguous: bool = False,
) -> _SearchOutcome:
    trace = (
        None
        if candidates is None
        else RetrievalTrace(
            candidates=tuple(candidates),
            candidate_limit=candidate_limit,
            exhaustive=exhaustive,
            ambiguous=ambiguous,
        )
    )
    return _SearchOutcome(hits=tuple(hits), trace=trace)


def _parent_index_ids(documents: Sequence[IndexDocument]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, builtins.list[str]] = {}
    for document in documents:
        grouped.setdefault(document.embedding.memory_id, []).append(document.embedding.embedding_id)
    return {memory_id: tuple(index_ids) for memory_id, index_ids in grouped.items()}


def _early_candidate_trace(
    memory_id: str,
    index_ids: tuple[str, ...],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
    rejected_by: RetrievalRejection,
) -> RetrievalCandidateTrace:
    lexical_match = memory_id in lexical_matches
    return RetrievalCandidateTrace(
        memory_id=memory_id,
        index_ids=index_ids,
        dense_relevance=dense_relevance.get(memory_id, 0.0),
        dense_confidence=dense_confidence.get(memory_id, 0.0),
        lexical_relevance=(_LEXICAL_RANK_RELEVANCE * lexical_index_relevance.get(memory_id, 0.0)),
        lexical_match=lexical_match,
        gate_confidence=max(
            dense_confidence.get(memory_id, 0.0),
            _LEXICAL_MATCH_CONFIDENCE if lexical_match else 0.0,
        ),
        rejected_by=rejected_by,
    )


def _extend_hydration_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    candidates: _IndexCandidates,
    index_ids: Sequence[str],
    hydrated_documents: Sequence[IndexDocument],
    accepted_documents: Sequence[IndexDocument],
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
) -> None:
    if target is None:
        return
    (
        dense_relevance,
        dense_confidence,
        lexical_index_relevance,
        lexical_matches,
    ) = _parent_index_signals(candidates, hydrated_documents)
    dense_by_id = {hit.id: hit for hit in candidates.dense}
    lexical_by_id = {hit.id: hit for hit in candidates.lexical}
    hydrated_ids = {document.embedding.embedding_id for document in hydrated_documents}
    for index_id in index_ids:
        if index_id in hydrated_ids:
            continue
        dense_hit = dense_by_id.get(index_id)
        lexical_hit = lexical_by_id.get(index_id)
        dense_hit_confidence = None if dense_hit is None else cast(float, dense_hit.confidence)
        target.append(
            RetrievalCandidateTrace(
                memory_id=None,
                index_ids=(index_id,),
                dense_relevance=None if dense_hit is None else dense_hit.relevance,
                dense_confidence=dense_hit_confidence,
                lexical_relevance=(
                    None
                    if lexical_hit is None
                    else _LEXICAL_MATCH_CONFIDENCE * lexical_hit.relevance
                ),
                lexical_match=lexical_hit is not None,
                gate_confidence=max(
                    dense_hit_confidence or 0.0,
                    _LEXICAL_MATCH_CONFIDENCE if lexical_hit is not None else 0.0,
                ),
                rejected_by=RetrievalRejection.STALE_INDEX,
            )
        )
    accepted_parent_ids = {document.embedding.memory_id for document in accepted_documents}
    for memory_id, parent_index_ids in index_ids_by_memory.items():
        if memory_id in accepted_parent_ids:
            continue
        target.append(
            _early_candidate_trace(
                memory_id,
                parent_index_ids,
                dense_relevance,
                dense_confidence,
                lexical_index_relevance,
                lexical_matches,
                RetrievalRejection.OCCURRENCE_RANGE,
            )
        )


def _extend_missing_memory_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    parent_ids: Sequence[str],
    memories: Sequence[StoredMemory],
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
) -> None:
    if target is None:
        return
    hydrated_ids = {memory.memory_id for memory in memories}
    for memory_id in parent_ids:
        if memory_id not in hydrated_ids:
            target.append(
                _early_candidate_trace(
                    memory_id,
                    index_ids_by_memory[memory_id],
                    dense_relevance,
                    dense_confidence,
                    lexical_index_relevance,
                    lexical_matches,
                    RetrievalRejection.MISSING_MEMORY,
                )
            )


def _extend_memory_type_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    memories: Sequence[StoredMemory],
    memory_type: MemoryType,
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    lexical_index_relevance: Mapping[str, float],
    lexical_matches: set[str],
) -> None:
    if target is None:
        return
    for memory in memories:
        if memory.memory_type != memory_type.value:
            target.append(
                _early_candidate_trace(
                    memory.memory_id,
                    index_ids_by_memory[memory.memory_id],
                    dense_relevance,
                    dense_confidence,
                    lexical_index_relevance,
                    lexical_matches,
                    RetrievalRejection.MEMORY_TYPE,
                )
            )


def _record_ranked_trace(
    target: dict[str, RetrievalCandidateTrace] | None,
    memory_id: str,
    index_ids_by_memory: Mapping[str, tuple[str, ...]],
    dense_relevance: Mapping[str, float],
    dense_confidence: Mapping[str, float],
    *,
    lexical_relevance: float,
    lexical_rerank_bonus: float,
    lexical_match: bool,
    gate_confidence: float,
    base_relevance: float,
    reinforcement_factor: float,
    temporal_factor: float | None,
    retention_factor: float | None,
    final_score: float,
) -> None:
    if target is None:
        return
    target[memory_id] = RetrievalCandidateTrace(
        memory_id=memory_id,
        index_ids=index_ids_by_memory[memory_id],
        dense_relevance=dense_relevance.get(memory_id, 0.0),
        dense_confidence=dense_confidence.get(memory_id, 0.0),
        lexical_relevance=lexical_relevance,
        lexical_rerank_bonus=lexical_rerank_bonus,
        lexical_match=lexical_match,
        gate_confidence=gate_confidence,
        base_relevance=base_relevance,
        reinforcement_factor=reinforcement_factor,
        temporal_factor=temporal_factor,
        retention_factor=retention_factor,
        final_score=final_score,
    )


def _qualified_candidates(
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    minimum_relevance: float,
    trace_candidates: builtins.list[RetrievalCandidateTrace] | None,
    ranked_traces: Mapping[str, RetrievalCandidateTrace] | None,
) -> builtins.list[tuple[StoredMemory, float, float, bool]]:
    qualified = []
    for item in ranked:
        if item[2] >= minimum_relevance:
            qualified.append(item)
        elif trace_candidates is not None and ranked_traces is not None:
            trace_candidates.append(
                replace(
                    ranked_traces[item[0].memory_id],
                    rejected_by=RetrievalRejection.MINIMUM_RELEVANCE,
                )
            )
    return qualified


def _extend_ranked_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    ranked_traces: Mapping[str, RetrievalCandidateTrace] | None,
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    limit: int,
    ambiguous: bool,
) -> None:
    if target is None or ranked_traces is None:
        return
    for rank, item in enumerate(ranked, start=1):
        rejected_by = None
        if ambiguous:
            rejected_by = RetrievalRejection.AMBIGUITY if rank <= 2 else RetrievalRejection.LIMIT
        elif rank > limit:
            rejected_by = RetrievalRejection.LIMIT
        target.append(
            replace(
                ranked_traces[item[0].memory_id],
                rank=rank,
                rejected_by=rejected_by,
            )
        )


def _parent_index_signals(
    candidates: _IndexCandidates,
    documents: Sequence[IndexDocument],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], set[str]]:
    dense_by_id = {hit.id: hit for hit in candidates.dense}
    lexical_by_id = {hit.id: hit for hit in candidates.lexical}
    dense_relevance: dict[str, float] = {}
    dense_confidence: dict[str, float] = {}
    lexical_relevance: dict[str, float] = {}
    lexical_matches: set[str] = set()
    for document in documents:
        memory_id = document.embedding.memory_id
        embedding_id = document.embedding.embedding_id
        dense_hit = dense_by_id.get(embedding_id)
        if dense_hit is not None:
            dense_relevance[memory_id] = max(
                dense_relevance.get(memory_id, 0.0), dense_hit.relevance
            )
            dense_confidence[memory_id] = max(
                dense_confidence.get(memory_id, 0.0), cast(float, dense_hit.confidence)
            )
        lexical_hit = lexical_by_id.get(embedding_id)
        if lexical_hit is not None:
            lexical_matches.add(memory_id)
            lexical_relevance[memory_id] = max(
                lexical_relevance.get(memory_id, 0.0), lexical_hit.relevance
            )
    return dense_relevance, dense_confidence, lexical_relevance, lexical_matches


def _retrieval_is_ambiguous(
    ranked: Sequence[tuple[StoredMemory, float, float, bool]],
    *,
    margin: float,
    temporal_range: tuple[datetime, datetime] | None,
) -> bool:
    if margin == 0.0 or len(ranked) < 2:
        return False
    first, second = ranked[:2]
    if first[3]:
        return False
    difference = first[1] - second[1] if temporal_range is not None else first[2] - second[2]
    return difference < margin


def _ranked_relevance(
    memory: StoredMemory,
    relevance: float,
    *,
    reference_at: datetime,
    temporal_range: tuple[datetime, datetime] | None,
    decay_half_life: timedelta | None,
) -> float:
    return _ranking_signals(
        memory,
        relevance,
        reference_at=reference_at,
        temporal_range=temporal_range,
        decay_half_life=decay_half_life,
    )[0]


def _ranking_signals(
    memory: StoredMemory,
    relevance: float,
    *,
    reference_at: datetime,
    temporal_range: tuple[datetime, datetime] | None,
    decay_half_life: timedelta | None,
) -> tuple[float, float, float | None, float | None]:
    ranking_reference = temporal_range[1] if temporal_range is not None else reference_at
    confirmed_at = memory.last_accessed_at
    confirmations = (
        min(memory.access_count, _DECAY_REINFORCEMENT_LIMIT)
        if confirmed_at is not None and confirmed_at <= ranking_reference
        else 0
    )
    reinforcement_factor = 1.0 + _CONFIRMATION_WEIGHT * math.log2(1.0 + confirmations)
    score = _bounded_scale(
        relevance,
        reinforcement_factor,
    )
    temporal_factor = None
    if temporal_range is not None:
        temporal_factor = _temporal_factor(
            memory.occurred_at,
            memory.occurred_end,
            temporal_range,
        )
        score = _bounded_scale(
            score,
            temporal_factor,
        )
    retention_factor = None
    if decay_half_life is not None:
        decay_reference = ranking_reference
        accessed_at = memory.last_accessed_at
        anchor = memory.occurred_end or memory.occurred_at or memory.updated_at
        access_count = 0
        if accessed_at is not None and accessed_at <= decay_reference:
            anchor = accessed_at
            access_count = confirmations
        age = max(0.0, (decay_reference - anchor).total_seconds())
        strength = 1.0 + math.log2(1.0 + min(access_count, _DECAY_REINFORCEMENT_LIMIT))
        retention = 2.0 ** (-age / (decay_half_life.total_seconds() * strength))
        retention_factor = _RANK_FLOOR + (_RANK_CEILING - _RANK_FLOOR) * retention
        score = _bounded_scale(
            score,
            retention_factor,
        )
    return score, reinforcement_factor, temporal_factor, retention_factor


def _bounded_scale(score: float, factor: float) -> float:
    if factor <= 1.0:
        return score * factor
    return score + (1.0 - score) * (factor - 1.0)


def _lexical_relevance(
    query: str,
    memories: Sequence[StoredMemory],
) -> dict[str, float]:
    query_terms = _lexical_terms(query) - _LEXICAL_NOISE_TERMS
    if not query_terms or not memories:
        return {}
    documents = {memory.memory_id: _lexical_terms(memory.content) for memory in memories}
    frequencies = Counter(term for terms in documents.values() for term in query_terms & terms)
    count = len(documents)
    weights = {
        term: (math.log((count + 1.0) / (frequencies[term] + 1.0)) + 1.0)
        * (_NEGATION_WEIGHT if term in _NEGATION_TERMS else 1.0)
        for term in query_terms
    }
    total = sum(weights.values())
    return {
        memory_id: sum(weights[term] for term in query_terms & terms) / total
        for memory_id, terms in documents.items()
    }


def _lexical_terms(value: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    terms = {*_LEXICAL_TERM.findall(normalized), *_CJK_CHARACTER.findall(normalized)}
    if _NEGATED_CONTRACTION.search(normalized) is not None:
        terms.add("not")
    return frozenset(terms)


def _temporal_factor(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    temporal_range: tuple[datetime, datetime],
) -> float:
    if occurred_at is None:
        return _RANK_FLOOR
    start, until = temporal_range
    event_until = occurred_end or occurred_at + timedelta(microseconds=1)
    if _overlaps_temporal_range(occurred_at, occurred_end, temporal_range):
        return _RANK_CEILING
    distance = start - event_until if event_until <= start else occurred_at - until
    window = max((until - start).total_seconds(), timedelta(days=1).total_seconds())
    proximity = math.pow(2.0, -distance.total_seconds() / window)
    return _RANK_FLOOR + (_RANK_CEILING - _RANK_FLOOR) * proximity


def _overlaps_temporal_range(
    occurred_at: datetime | None,
    occurred_end: datetime | None,
    temporal_range: tuple[datetime, datetime],
) -> bool:
    if occurred_at is None:
        return False
    start, until = temporal_range
    event_until = occurred_end or occurred_at + timedelta(microseconds=1)
    return occurred_at < until and event_until > start


def _temporal_range(text: str, reference_at: datetime) -> tuple[datetime, datetime] | None:
    try:
        return _parse_temporal_range(text, reference_at)
    except (OverflowError, ValueError):
        raise ValidationError("temporal expression exceeds the supported date range") from None


def _temporal_context(
    text: str,
    reference_at: datetime,
    *,
    infer_reference: bool,
) -> tuple[datetime, str]:
    match = _TODAY_ISO_DATE.search(text)
    try:
        if match is not None:
            anchored = date.fromisoformat(match.group(1))
        else:
            match = _TODAY_NAMED_DATE.search(text)
            if match is None:
                return reference_at, text
            month = _MONTHS[match.group(1)[:3].casefold()]
            anchored = date(
                int(match.group(3)) if match.group(3) is not None else reference_at.year,
                month,
                int(match.group(2)),
            )
    except ValueError:
        raise ValidationError("today reference date is invalid") from None
    temporal_text = f"{text[: match.start()]} {text[match.end() :]}".strip()
    if infer_reference:
        reference_at = _day_start(anchored, reference_at)
    return reference_at, temporal_text


def _parse_temporal_range(  # noqa: C901 - ordered phrases are clearer as one parser
    text: str,
    reference_at: datetime,
) -> tuple[datetime, datetime] | None:
    if not text:
        return None
    dates = []
    for value in _ISO_DATE.findall(text):
        try:
            dates.append(date.fromisoformat(value))
        except ValueError:
            continue
    if dates:
        return _date_range(min(dates), max(dates), reference_at)

    normalized = text.casefold()
    months = [
        (int(match.group(2)), _MONTHS[match.group(1)[:3]])
        for match in _NAMED_MONTH_YEAR.finditer(normalized)
    ]
    months.extend(
        (int(match.group(1)), int(match.group(2))) for match in _CJK_YEAR_MONTH.finditer(normalized)
    )
    if months:
        first = min(date(year, month, 1) for year, month in months)
        last = max(date(year, month, 1) for year, month in months)
        start = _day_start(first, reference_at)
        return start, _shift_month(_day_start(last, reference_at), 1)

    relative_day = re.search(r"(?<!\d)(\d{1,5})\s*天前", normalized) or re.search(
        r"\b(\d{1,5})\s+days?\s+ago\b", normalized
    )
    if relative_day is not None:
        days = int(relative_day.group(1))
        if days <= 36_500:
            target = reference_at.date() - timedelta(days=days)
            return _date_range(target, target, reference_at)

    rolling = re.search(r"(?:过去|最近)\s*(\d{1,5})\s*天", normalized) or re.search(
        r"\b(?:past|last)\s+(\d{1,5})\s+days?\b", normalized
    )
    if rolling is not None:
        days = int(rolling.group(1))
        if 0 < days <= 36_500:
            return reference_at - timedelta(days=days), reference_at

    if _contains(normalized, "day before yesterday", "前天"):
        target = reference_at.date() - timedelta(days=2)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "yesterday", "昨天"):
        target = reference_at.date() - timedelta(days=1)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "day after tomorrow", "后天"):
        target = reference_at.date() + timedelta(days=2)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "tomorrow", "明天"):
        target = reference_at.date() + timedelta(days=1)
        return _date_range(target, target, reference_at)
    if _contains(normalized, "today", "今天"):
        target = reference_at.date()
        return _date_range(target, target, reference_at)

    day_start = _day_start(reference_at.date(), reference_at)
    week_start = day_start - timedelta(days=reference_at.weekday())
    if _contains(normalized, "last week", "上周", "上星期"):
        return week_start - timedelta(days=7), week_start
    if _contains(normalized, "this week", "本周", "这周", "本星期"):
        return week_start, week_start + timedelta(days=7)
    if _contains(normalized, "next week", "下周", "下星期"):
        return week_start + timedelta(days=7), week_start + timedelta(days=14)
    if _contains(normalized, "past week", "过去一周", "最近一周"):
        return reference_at - timedelta(days=7), reference_at

    month_start = day_start.replace(day=1)
    if _contains(normalized, "last month", "上个月", "上月"):
        return _shift_month(month_start, -1), month_start
    if _contains(normalized, "this month", "这个月", "本月"):
        return month_start, _shift_month(month_start, 1)
    if _contains(normalized, "next month", "下个月", "下月"):
        return _shift_month(month_start, 1), _shift_month(month_start, 2)

    year_start = day_start.replace(month=1, day=1)
    if _contains(normalized, "last year", "去年"):
        return year_start.replace(year=year_start.year - 1), year_start
    if _contains(normalized, "this year", "今年"):
        return year_start, year_start.replace(year=year_start.year + 1)
    if _contains(normalized, "next year", "明年"):
        return (
            year_start.replace(year=year_start.year + 1),
            year_start.replace(year=year_start.year + 2),
        )

    years = [int(value) for value in _CALENDAR_YEAR.findall(normalized)]
    years.extend(int(value) for value in _CJK_CALENDAR_YEAR.findall(normalized))
    if years:
        return (
            _day_start(date(min(years), 1, 1), reference_at),
            _day_start(date(max(years) + 1, 1, 1), reference_at),
        )
    return None


def _contains(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _day_start(value: date, reference_at: datetime) -> datetime:
    return datetime.combine(value, time.min, tzinfo=reference_at.tzinfo)


def _date_range(first: date, last: date, reference_at: datetime) -> tuple[datetime, datetime]:
    return _day_start(first, reference_at), _day_start(last + timedelta(days=1), reference_at)


def _shift_month(value: datetime, months: int) -> datetime:
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return value.replace(year=year, month=month + 1)


def _evidence_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("evidence_budget_chars must be a positive integer")
    return value


def _decay_half_life(days: float | None) -> timedelta | None:
    if days is None:
        return None
    if (
        isinstance(days, bool)
        or not isinstance(days, int | float)
        or not math.isfinite(days)
        or days <= 0
    ):
        raise ValidationError("decay_half_life_days must be a positive finite number")
    try:
        return timedelta(days=days)
    except OverflowError:
        raise ValidationError("decay_half_life_days is too large") from None


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{name} must be non-empty and trimmed")
    return value


def _embedding_id(memory_id: str, object_part: int) -> str:
    if object_part == 0:
        return memory_id
    return hashlib.sha256(f"mindbridge-embedding:{memory_id}:{object_part}".encode()).hexdigest()


def _identity_name(value: object) -> str:
    name = _text(value, "identity name")
    if len(name) > 255 or not name.isprintable():
        raise ValidationError("identity name must be at most 255 printable characters")
    return name


def _limit(value: object, *, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValidationError(f"limit must be between 1 and {maximum}")


def _normalized_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    if len(values) != dimension or any(
        isinstance(value, bool) or not isinstance(value, int | float) for value in values
    ):
        raise ModelError(f"embedding model must return {dimension} numeric values")
    vector = tuple(float(value) for value in values)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise ModelError("embedding model returned a non-finite or zero vector")
    if math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-12):
        return vector
    return tuple(value / norm for value in vector)


def _encode_cursor(memory: StoredMemory) -> str:
    payload = json.dumps(
        ["v1", _datetime_text(memory.created_at), memory.memory_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: object) -> tuple[datetime, str]:
    if not isinstance(cursor, str) or not cursor.strip() or cursor != cursor.strip():
        raise ValidationError("cursor is invalid")
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded: object = json.loads(payload)
        if (
            not isinstance(decoded, list)
            or len(decoded) != 3
            or decoded[0] != "v1"
            or not isinstance(decoded[1], str)
            or not isinstance(decoded[2], str)
        ):
            raise ValueError
        created_at = datetime.fromisoformat(decoded[1].replace("Z", "+00:00"))
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError
        memory_id = _identifier(decoded[2], "cursor memory_id")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise ValidationError("cursor is invalid") from None
    return created_at, memory_id


def _datetime_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@contextmanager
def _translate_storage_errors(action: str) -> Iterator[None]:
    try:
        yield
    except MindBridgeError:
        raise
    except Exception as error:
        raise StorageError(f"failed to {action}", reason="io_failed") from error


@contextmanager
def _translate_index_errors(action: str) -> Iterator[None]:
    try:
        yield
    except MindBridgeError:
        raise
    except Exception as error:
        raise IndexUnavailableError(f"failed to {action}") from error
