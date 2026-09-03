"""Developer-facing local memory API."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import io
import json
import math
import os
import re
import shutil
import unicodedata
import wave
from collections import Counter, deque
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import copy_context
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Condition, RLock
from time import perf_counter
from typing import Protocol, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue

from mindbridge._telemetry import (
    CAPTURE_FAILED,
    CAPTURE_SETTLED,
    CAPTURE_TIME_TO_SEARCHABLE,
    EMBEDDING_PARTS_ELIDED,
    IDENTITY_CACHED,
    IDENTITY_CREATED,
    IDENTITY_EVIDENCE_ASSETS,
    IDENTITY_EVIDENCE_REQUIRED,
    IDENTITY_IDENTITIES,
    IDENTITY_LINKED,
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
from mindbridge.context import compile_context, evidence_cost
from mindbridge.control import dump_operation, load_operation, operation_key
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
    IdentityLink,
    IndexCandidate,
    IndexDocument,
    LocalStore,
    SpeechRollback,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.zvec_index import (
    IndexHit,
    ZvecIndex,
    validate_index_configuration,
)
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    EmbedTask,
    FaceAnalysis,
    FaceBackend,
    FormationBackend,
    FormationInput,
    GenerationBackend,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
    StreamingGenerationBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
    _modalities,
)
from mindbridge.plugins import MemoryConfig, MemoryPlugins
from mindbridge.types import (
    AbstentionReason,
    AcousticBoundary,
    AnswerResult,
    ASRPartial,
    AssetRef,
    AudioBoundary,
    AudioStreamPacket,
    Blob,
    ConsolidationCandidate,
    ConsolidationReport,
    ContentAtom,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextUnknown,
    ContextUnknownKind,
    EvidenceBasis,
    FaceObservation,
    FormationProposal,
    IdentityErasure,
    IdentityProfile,
    IndexQuantization,
    MemoryCapabilities,
    MemoryContext,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    PCMChunk,
    PendingCapture,
    PrefetchResult,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalScope,
    RetrievalTrace,
    SceneBoundary,
    SearchHit,
    SpeakerSegment,
    StreamCommit,
    StreamEvent,
    StreamInput,
    StreamPhase,
    TracedSearchResult,
    VADPacket,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
    VisionStreamPacket,
)

_DOCUMENT_TASK = EmbedTask.DOCUMENT.value
_INDEX_RECIPE_PREFIX = (
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-stemmed-plus-bigram:grouped-range:context-keys-v10"
)
# Recipes whose stored embeddings are still correct, so the index is rebuilt from SQLite without
# paying to embed the content again. A full-text tokenizer change belongs here and not below: it
# rewrites the derived documents and leaves every vector untouched. Version 9 named its CJK field
# `fts-dual-language` because it ran a Chinese segmenter, which returned nothing for Japanese or
# Korean; version 10 tokenizes that field into script-agnostic character bigrams instead.
_REINDEXABLE_INDEX_RECIPES = frozenset(
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-dual-language:grouped-range:"
    f"context-keys-v9:quantization-{mode.value}"
    for mode in IndexQuantization
)
# Recipes that also invalidate the stored embeddings, so reopening pays for a full re-embed.
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
_DECAY_REINFORCEMENT_LIMIT = 20
_CONFIRMATION_WEIGHT = 0.05
# Covering every distinctive query term is evidence in a way that placing well in the full-text
# index is not, so a complete match ranks as near-certain and a memory quoting the whole question
# cannot be buried by an unrelated dense neighbour. Nothing short of complete coverage enters the
# ranking score through this branch. A general floor under every full-text hit used to, at 0.24,
# but that compared an index-side quantity against a cosine, and how often it decided anything
# turned entirely on where the configured embedder puts its cosines: replaying 841 dev queries,
# the share of dense candidates it could outrank was 0.92% on LoCoMo-Refined, 8.48% on
# Mem-Gallery and 50.26% on MemLens. Deleting it left R@1, R@5, R@10, R@20, R@100 and MRR
# unchanged to four decimals on all three and the top ten identical on 99.87% of queries.
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
# Runs `\w+` cannot split into words, because they are written without spaces and no segmenter
# is available here: Han ideographs (Chinese, and Japanese kanji) and Japanese kana. What follows
# is character and adjacent-character bigram splitting, which is script-agnostic, so no Japanese
# text is handed to a Chinese segmenter by being listed here.
#
# This is deliberately narrower than the identically named predicate in the Zvec index, which
# also lists Hangul. The two answer different questions. The index one asks which documents need
# the character-bigram full-text field instead of the English stemmer, and Korean needs it
# because its particles agglutinate onto the eojeol. This one asks which runs `\w+` fails to
# split at all, and Korean is space-delimited, so `\w+` already yields eojeol tokens here and
# per-syllable Hangul terms would only dilute them.
_UNSEGMENTED_RUN = re.compile(
    "["
    "\u3040-\u30ff"  # Hiragana and Katakana
    "\u31f0-\u31ff"  # Katakana Phonetic Extensions
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\uff66-\uff9f"  # Halfwidth Katakana
    "\U00020000-\U0003ffff"  # CJK Unified Ideographs Extension B and later
    "]+"
)
# The Chinese counterpart of `_LEXICAL_NOISE_TERMS`: particles, copulas, prepositions,
# conjunctions, pronouns, and interrogatives. Stripping them matters more than it does in
# English, because every one of them otherwise carries IDF weight into the coverage ratio that
# performs cross-route fusion, and a question is mostly these characters. Negation characters
# are excluded on purpose; `_NEGATION_TERMS` weights those up rather than away.
_CJK_NOISE_CHARACTERS = frozenset(
    "的了着地得之吗呢吧啊呀嘛么们"
    "是在就都也还把被给对从跟和与"
    "或及而但因所以什谁哪怎何我你"
    "他她它这那其请"
)
_NEGATION_TERMS = frozenset(
    {"no", "not", "never", "neither", "nor", "without", "不", "没", "未", "无", "非", "别"}
)
_NEGATED_CONTRACTION = re.compile(r"n['\u2019]t\b", re.IGNORECASE)
_MAX_CONTENT_PARTS = 128
# Model span modules mapped onto the failure stage a caller sees.
_MODEL_STAGES = {
    "consolidation": "consolidate",
    "embedding": "embed",
    "face": "recognize",
    "formation": "form",
    "generation": "generate",
    "transcription": "transcribe",
    "vision": "describe",
}
_MAX_TEXT_CHARACTERS = 65_536
_MAX_METADATA_BYTES = 262_144
_TEXT_KEY_CHARACTERS = 2_048
_TEXT_KEY_OVERLAP = 256
_TEXT_KEY_CONTEXT = 256
_MAX_RETRIEVAL_KEYS = 128
# The reason an embedding backend reports when media does not fit one inline request.
_PAYLOAD_TOO_LARGE = "payload_too_large"
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
    audio_transcript: bool = False
    visual_description: bool = False


@dataclass(slots=True)
class _AudioStreamState:
    pcm: bytearray
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bytes: int | None = None
    transcript: str = ""
    occurred_at: datetime | None = None


@dataclass(slots=True)
class _VisionStreamState:
    image: Blob | None = None
    description: str = ""
    occurred_at: datetime | None = None
    last_occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _PreparedMemory:
    memory_id: str
    content: _PreparedContent
    metadata_json: str
    occurred_at: datetime | None
    occurred_end: datetime | None
    memory_type: MemoryType
    context: ObservationContext | MemoryContext | None = None


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
        vision_describer: VisionDescriptionBackend | None = None,
        face_analyzer: FaceBackend | None = None,
        former: FormationBackend | None = None,
        consolidator: ConsolidationBackend | None = None,
        index_speech: bool = _DEFAULT_CONFIG.index_speech,
        index_quantization: IndexQuantization = _DEFAULT_CONFIG.index_quantization,
        minimum_relevance: float = _DEFAULT_CONFIG.minimum_relevance,
        ambiguity_margin: float = _DEFAULT_CONFIG.ambiguity_margin,
        evidence_budget_chars: int | None = _DEFAULT_CONFIG.evidence_budget_chars,
        decay_half_life_days: float | None = _DEFAULT_CONFIG.decay_half_life_days,
        reinforce_on_answer: bool = _DEFAULT_CONFIG.reinforce_on_answer,
        speaker_similarity: float = _DEFAULT_CONFIG.speaker_similarity,
        speaker_margin: float = _DEFAULT_CONFIG.speaker_margin,
        face_similarity: float = _DEFAULT_CONFIG.face_similarity,
        face_margin: float = _DEFAULT_CONFIG.face_margin,
        identity_link_min_assets: int = _DEFAULT_CONFIG.identity_link_min_assets,
        tracer: Tracer | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._tracer = trace.get_tracer(TRACER_NAME) if tracer is None else tracer
        self._owner_pid = os.getpid()
        self._write_lock = RLock()
        self._formation_lock = RLock()
        # Settlement is the one path that runs the expensive model stages over rows another
        # caller can already see queued, so two threads would otherwise embed the same record
        # twice and discover it only at the commit. Blocking, like the other two: a second
        # `settle()` waits and then finds the queue already drained.
        self._settle_lock = RLock()
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
        self._identity_link_min_assets = _positive_int(
            identity_link_min_assets, "identity_link_min_assets"
        )
        if not isinstance(index_speech, bool):
            raise ValidationError("index_speech must be a boolean")
        self._index_speech = index_speech
        self._minimum_relevance = _unit_interval(minimum_relevance, "minimum_relevance")
        self._ambiguity_margin = _unit_interval(ambiguity_margin, "ambiguity_margin")
        self._evidence_budget = _evidence_budget(evidence_budget_chars)
        self._decay_half_life = _decay_half_life(decay_half_life_days)
        if not isinstance(reinforce_on_answer, bool):
            raise ValidationError("reinforce_on_answer must be a boolean")
        self._reinforce_on_answer = reinforce_on_answer

        self._store = _open_store(self.data_dir)
        self._embedder = embedder
        self._answerer = answerer
        self._transcriber = transcriber
        self._vision_describer = vision_describer
        self._face_analyzer = face_analyzer
        self._former = former
        self._consolidator = consolidator

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
                self._vision_capabilities,
                self._vision_model,
            ) = _vision_contract(self._vision_describer)
            (
                self._face_capabilities,
                self._face_model,
                self._face_space,
                self._face_analysis_space,
            ) = _face_contract(self._face_analyzer)
            (
                self._formation_capabilities,
                self._formation_model,
                self._formation_space,
            ) = _formation_contract(self._former)
            (
                self._consolidation_model,
                self._consolidation_recipe,
            ) = _consolidation_contract(self._consolidator)
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
                "failed to open the local search index", stage="open", reason="index_missing"
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
            vision_describer=plugins.vision_describer,
            face_analyzer=plugins.face_analyzer,
            former=plugins.former,
            consolidator=plugins.consolidator,
            index_speech=config.index_speech,
            index_quantization=config.index_quantization,
            minimum_relevance=config.minimum_relevance,
            ambiguity_margin=config.ambiguity_margin,
            evidence_budget_chars=config.evidence_budget_chars,
            decay_half_life_days=config.decay_half_life_days,
            reinforce_on_answer=config.reinforce_on_answer,
            speaker_similarity=config.speaker_similarity,
            speaker_margin=config.speaker_margin,
            face_similarity=config.face_similarity,
            face_margin=config.face_margin,
            identity_link_min_assets=config.identity_link_min_assets,
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
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        """Add one native or mixed-modal memory and return its stable record."""
        return self._add_one(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def _add_one(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
        context: ObservationContext | None,
        transcript: str | None = None,
        description: str | None = None,
    ) -> MemoryRecord:
        with self._trace("mindbridge.add", kind="operation"), self._operation() as assets:
            if self._former is not None and context is None:
                context = ObservationContext()
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared_content = self._prepare_content(content, assets)
                if transcript is not None:
                    prepared_content = _with_stream_transcript(prepared_content, transcript)
                if description is not None:
                    prepared_content = _with_stream_description(prepared_content, description)
            prepared = _prepare_memory(
                prepared_content,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
                context=context,
            )
            record = self._add_prepared((prepared,), operation=assets)[0]
            self._form_sources((record,), operation=assets)
            self._complete_formation((record,))
            return record

    def _add_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._add_one(
            item.content,
            occurred_at=item.occurred_at,
            occurred_end=item.occurred_end,
            metadata=item.metadata,
            memory_type=item.memory_type,
            context=item.context,
            transcript=item.transcript,
            description=item.description,
        )

    def _stream_record(
        self,
        content: ContentInput | StreamInput,
        *,
        capture: bool,
    ) -> MemoryRecord:
        if isinstance(content, StreamInput):
            if capture:
                return self._capture_stream_input(content)
            return self._add_stream_input(content)
        if capture:
            return self.capture(content)
        return self.add(content)

    def _capture_stream_input(self, item: StreamInput) -> MemoryRecord:
        return self._capture_one(
            item.content,
            occurred_at=item.occurred_at,
            occurred_end=item.occurred_end,
            metadata=item.metadata,
            memory_type=item.memory_type,
            context=item.context,
            transcript=item.transcript,
            description=item.description,
        )

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
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
            context_values = _batch_values(context, len(batch), "context")
            if self._former is not None:
                context_values = tuple(
                    value if value is not None else ObservationContext() for value in context_values
                )
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
                        context=item_context,
                    )
                    for index, (
                        content,
                        event_time,
                        event_end,
                        item_metadata,
                        item_context,
                    ) in enumerate(
                        zip(
                            batch,
                            occurrences,
                            occurrence_ends,
                            metadata_values,
                            context_values,
                            strict=True,
                        )
                    )
                )
            if not prepared:
                return ()
            records = self._add_prepared(prepared, operation=assets)
            self._form_sources(records, operation=assets)
            self._complete_formation(records)
            return records

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        """Commit one memory without calling any model; `settle()` makes it searchable.

        The returned record is the content-addressed record `add()` returns for the same input, so
        capturing and then adding the same content is one memory. It is durable and readable
        through `get()` and `list()` immediately and invisible to `search()` until settled.

        `settle()` appends the text its models derive to `content`; it never rewrites what the
        caller supplied. See `MemoryRecord` for how derived sections are marked and where the raw
        evidence lives.
        """
        return self._capture_one(
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    def _capture_one(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None,
        occurred_end: datetime | None,
        metadata: Mapping[str, object] | None,
        memory_type: MemoryType,
        context: ObservationContext | None,
        transcript: str | None = None,
        description: str | None = None,
    ) -> MemoryRecord:
        with self._trace("mindbridge.capture", kind="operation"), self._operation() as assets:
            if self._former is not None and context is None:
                context = ObservationContext()
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared_content = self._prepare_content(content, assets)
                # Folded in here, exactly as `_add_one` folds them, so a streaming FINAL that
                # already carries ASR or caption text keeps the same content-addressed ID on
                # either commit path and `settle()` sees the marker instead of paying again.
                if transcript is not None:
                    prepared_content = _with_stream_transcript(prepared_content, transcript)
                if description is not None:
                    prepared_content = _with_stream_description(prepared_content, description)
            prepared = _prepare_memory(
                prepared_content,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
                context=context,
            )
            # Reject here what `add()` would reject, before the commit. Committing media no
            # configured model can ever take would make the record durable and then fail every
            # `settle()` forever, which is a queue entry no retry ceiling should have to absorb.
            # Unconditional: `add()` rejects twice, once early and once inside
            # `_embedding_content`, so guarding this on one capability would miss the second.
            # With an embedder that takes everything it is a no-op.
            _fallback_unsupported(
                prepared.content,
                self._embedding_capabilities,
                "embedding",
                rescuable=self._deferred_rescue(prepared.content.assets),
            )
            now = datetime.now(timezone.utc)
            # No index lock: a capture enqueues no vectors, so it has no outbox work to drain and
            # never has to wait behind a Zvec flush.
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                _translate_storage_errors("capture memory"),
            ):
                self._store.write_captures(
                    (
                        StoredMemory(
                            memory_id=prepared.memory_id,
                            content=prepared.content.text,
                            modality=prepared.content.modality.value,
                            memory_type=prepared.memory_type.value,
                            assets=prepared.content.assets,
                            metadata_json=prepared.metadata_json,
                            occurred_at=prepared.occurred_at,
                            occurred_end=prepared.occurred_end,
                            created_at=now,
                            updated_at=now,
                            context=_stored_memory_context(prepared, recorded_at=now),
                        ),
                    ),
                    enqueued_at=now,
                )
            with (
                self._trace("mindbridge.storage.hydrate", kind="stage"),
                _translate_storage_errors("hydrate captured memory"),
            ):
                authoritative = self._store.read_memories((prepared.memory_id,))
            if not authoritative:
                raise StorageError("captured memory could not be read from SQLite")
            assets.persisted.update(asset.asset_id for asset in authoritative[0].assets)
            return self._memory_record(authoritative[0])

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        """Enrich, embed, index, and form up to `limit` captured records in enqueue order.

        Every readable record is attempted: a failing one keeps its queue row, its attempt count,
        and its reason, and the records behind it still settle. The first failure is raised once
        the batch is done, with `subject` naming its record; the rest are readable through
        `pending_captures()`. A record that has already failed `max_attempts` times is skipped
        rather than retried, so one poisoned capture cannot block the queue forever. It stays
        queued and visible, and lowering the ceiling is what retries it.

        Pass `memory_ids` to settle only those records. Naming a record is the host asking for it
        by hand, so the retry ceiling does not apply and `max_attempts` is ignored: that is how a
        record parked at the ceiling is retried, quarantined by simply never being named, or
        settled ahead of the queue. IDs that are not queued are skipped.

        One settlement runs at a time per `Memory`. A concurrent call waits rather than paying
        for the same model work twice, and usually then finds the queue already drained.
        """
        with (
            self._trace("mindbridge.settle", kind="operation") as span,
            self._operation() as assets,
        ):
            _limit(limit, maximum=100)
            if (
                isinstance(max_attempts, bool)
                or not isinstance(max_attempts, int)
                or max_attempts < 1
            ):
                raise ValidationError("max_attempts must be a positive integer")
            with self._settle_lock:
                with _translate_storage_errors("read the capture queue"):
                    # Same filter `pending_captures()` publishes, so naming records reuses the
                    # queue read rather than adding a second way to select them.
                    queued = self._store.pending_captures(
                        limit=limit,
                        memory_ids=memory_ids,
                        max_attempts=None if memory_ids is not None else max_attempts,
                    )
                span.set_attribute(CAPTURE_SETTLED, 0)
                span.set_attribute(CAPTURE_FAILED, 0)
                if not queued:
                    return 0
                with _translate_storage_errors("hydrate captured memories"):
                    rows = self._store.read_memories(tuple(row.memory_id for row in queued))
                settled, failures = self._settle_stored(rows, operation=assets)
                span.set_attribute(CAPTURE_SETTLED, len(settled))
                span.set_attribute(CAPTURE_FAILED, len(failures))
            enqueued_at = {row.memory_id: row.enqueued_at for row in queued}
            now = datetime.now(timezone.utc)
            waits = tuple(
                (now - enqueued_at[memory_id]).total_seconds() * 1000.0 for memory_id in settled
            )
            if waits:
                span.set_attribute(CAPTURE_TIME_TO_SEARCHABLE, max(waits))
            if failures:
                raise failures[0]
            return len(settled)

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        """Return up to `limit` records whose deferred work is not finished, oldest first.

        With a formation backend, `add()` holds a row between its commit and formation, so a
        queued record may already be searchable and owe formation only. Pass `memory_ids` to ask
        whether specific records are still waiting: one that is absent from the result is not
        pending, which means it is settled or was never stored, and `get()` tells the two apart.
        """
        with (
            self._trace("mindbridge.pending_captures", kind="operation"),
            self._operation(),
        ):
            _limit(limit, maximum=100)
            with _translate_storage_errors("read the capture queue"):
                return self._store.pending_captures(limit=limit, memory_ids=memory_ids)

    def add_stream(
        self,
        contents: Iterable[ContentInput | StreamInput],
        *,
        capture: bool = False,
    ) -> Iterator[MemoryRecord]:
        """Add a lazy omni stream one durable, searchable observation at a time.

        With `capture=True` each item commits through `capture()` instead: every yielded record is
        durable and readable but has no vectors, so the stream keeps its acknowledgement off the
        model path and the host owes the matching `settle()` before anything is searchable.
        """
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
                record = self._stream_record(content, capture=capture)
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
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return ranked memories for a native or mixed-modal query."""
        return self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
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
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        """Return ranked memories plus an opt-in candidate trace without evidence content."""
        outcome = self._search(
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
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
        scope: RetrievalScope | None,
        capture_trace: bool,
    ) -> _SearchOutcome:
        with self._trace("mindbridge.search", kind="operation"), self._operation() as assets:
            _limit(limit, maximum=100)
            occurred_from, occurred_until = _search_occurrence_range(
                occurred_from,
                occurred_until,
            )
            scope = _retrieval_scope(scope)
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
                scope=scope,
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
        scope: RetrievalScope | None = None,
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
            scope = _retrieval_scope(scope)
            prepared_search = partial(
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
                scope=scope,
                require_unambiguous=limit == 1,
                capture_trace=False,
            )

            def search() -> _SearchOutcome:
                # `search()` opens a `mindbridge.search` operation span; `ask` reaches the same
                # retrieval plane directly, so without this the only retrieval span inside `ask`
                # is `mindbridge.index.search` -- the index lookup alone, excluding query
                # embedding and content preparation. Reporting that as "search latency" would name
                # a quantity nobody experiences, which is the failure the design doc's end-to-end
                # rule exists to prevent. One stage span per `ask`, covering the whole leg.
                with self._trace("mindbridge.retrieve", kind="stage"):
                    return prepared_search()

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
            grounding = tuple(hit for hit in hits if hit.id in used_ids)
            self._reinforce_answered(grounding)
            return AnswerResult(
                answer=result.answer,
                hits=grounding,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
            )

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> ContextBundle:
        """Compile one bounded, structured context bundle for a goal."""
        with self._trace("mindbridge.compile", kind="operation"), self._operation() as assets:
            # Taken before anything else, so `budget.max_latency_ms` bounds the compilation a
            # caller waits for rather than the selection pass alone.
            started_at = perf_counter()
            budget = ContextBudget() if budget is None else budget
            if not isinstance(budget, ContextBudget):
                raise ValidationError("budget must be a ContextBudget")
            scope = _retrieval_scope(scope)
            with self._trace("mindbridge.content.prepare", kind="stage"):
                prepared = self._prepare_content(goal, assets)
            explicit_reference = _reference_at(reference_at)
            reference, temporal_text = _temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            # `_limit` bounds what a caller may ask a public search to return, not how deep the
            # kernel ranks, so `_search_prepared` has no ceiling of its own and a bundle may rank
            # past one hundred. Three candidates per slot is the same headroom `ask()` gives its
            # modality round robin.
            candidate_limit = max(_RERANK_CANDIDATES, budget.max_items * 3)
            outcome = self._search_prepared(
                prepared,
                limit=candidate_limit,
                operation=assets,
                # The budget filters memory types itself, because it selects a set rather than
                # the one type the retrieval plane can push into the index.
                memory_type=None,
                reference_at=reference,
                temporal_range=_temporal_range(temporal_text, reference),
                occurred_from=None,
                occurred_until=None,
                scope=scope,
                require_unambiguous=False,
                capture_trace=False,
            )
            # The same transcript cache `search()` writes for a spoken query. It is a cache of
            # the query's own audio, not a memory: `compile()` stores nothing it retrieved.
            self._persist_transcripts(assets)
            return compile_context(
                prepared.text,
                outcome.hits,
                budget=budget,
                reference_at=reference,
                started_at=started_at,
                unknowns=self._request_unknowns(prepared, scope, outcome.hits),
                candidate_limit=candidate_limit,
            )

    def _request_unknowns(
        self,
        prepared: _PreparedContent,
        scope: RetrievalScope | None,
        hits: Sequence[SearchHit],
    ) -> tuple[ContextUnknown, ...]:
        """Name what the request asked for that the compiler cannot see in the hits alone."""
        unknowns: list[ContextUnknown] = []
        unsupported = sorted(
            {
                asset.modality
                for asset in prepared.assets
                if Modality(asset.modality) not in self._embedding_capabilities
            }
        )
        if unsupported:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.MODALITY_UNSUPPORTED,
                    detail=(
                        f"the goal's {', '.join(unsupported)} was not embedded natively;"
                        f" embedding accepts {_modality_names(self._embedding_capabilities)},"
                        " so only derived text could have matched"
                    ),
                )
            )
        if not hits and (described := _scope_description(scope)) is not None:
            unknowns.append(ContextUnknown(kind=ContextUnknownKind.SCOPE_EMPTY, detail=described))
        return tuple(unknowns)

    @property
    def capabilities(self) -> MemoryCapabilities:
        """What this composition supports, from the same declarations routing reads.

        Published so a caller does not have to construct a probe write to find out, and so a
        transport can report the composition instead of a bare liveness flag.
        """
        return declared_capabilities(
            embedder=self._embedder,
            answerer=self._answerer,
            transcriber=self._transcriber,
            vision_describer=self._vision_describer,
            face_analyzer=self._face_analyzer,
            former=self._former,
            consolidator=self._consolidator,
        )

    def _reinforce_answered(self, hits: Sequence[SearchHit]) -> None:
        """Count the evidence an answer cited, so `access_count` is not permanently zero.

        `_ranking_signals` scales a candidate by its access count and, when `decay_half_life_days`
        is set, ages it by the time since it was last read. Nothing but the explicit `reinforce`
        operation ever wrote either field, so for an agent driving MindBridge over MCP or REST the
        reinforcement factor was pinned at 1.0 for ever and enabling decay degraded memories
        purely by age, with no usage signal to hold up the ones that keep answering questions.
        Answering is that signal, and this set is already narrowed to what the model cited.

        `_write_lock` is reentrant and `ask` holds none of it at this point, so taking it here
        cannot deadlock against the searches above, which release it before returning. A failure
        is swallowed on purpose: this is bookkeeping, and losing it must not discard an answer
        that has already been generated and paid for.

        `reinforce_on_answer=False` turns it off, which measurement needs: reinforcing mid-run
        makes one question's retrieval depend on which earlier questions answered, and under
        concurrency on the order their updates committed.
        """
        if not hits or not self._reinforce_on_answer:
            return
        with suppress(Exception), self._write_lock:
            self._store.reinforce_memories(
                tuple(hit.id for hit in hits),
                accessed_at=datetime.now(timezone.utc),
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
                    "configured transcription backend does not provide speaker recognition",
                    reason="backend_not_configured",
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
                raise ModelError("no face backend is configured", reason="backend_not_configured")
            unsupported = {
                Modality(asset.modality)
                for asset in visual_assets
                if Modality(asset.modality) not in self._face_capabilities
            }
            if unsupported:
                names = ", ".join(sorted(modality.value for modality in unsupported))
                raise ModelError(
                    f"configured face backend does not support: {names}",
                    reason="unsupported_modality",
                )
            self._lease_assets(visual_assets, operation.leased)
            operation.persisted.update(asset.asset_id for asset in visual_assets)
            self._recognize_faces(visual_assets, operation)
            return tuple(
                observation
                for asset in visual_assets
                for observation in operation.face_observations[asset.asset_id]
            )

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        """Assign or replace a human-readable name for one recognized speaker."""
        self._register_identity(speaker_id, name, relationship=relationship, speaker=True)

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        """Assign or replace the name, and optionally the relationship, of one identity.

        Omitting ``relationship`` leaves any recorded relationship intact, so renaming a person
        never silently discards it. There is deliberately no way to clear one here.
        """
        self._register_identity(identity_id, name, relationship=relationship, speaker=False)

    def identity(self, identity_id: str) -> IdentityProfile | None:
        """Return one identity's recorded name and relationship, or None when it does not exist.

        Face and speech observations carry an ``identity_id``; this resolves that id, following
        a merge alias, to whatever a caller has registered for the person.
        """
        with (
            self._trace("mindbridge.identity", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            requested_id = _identifier(identity_id, "identity_id")
            with _translate_storage_errors("read identity profile"):
                return self._store.identity_profile(requested_id)

    def unlink_identity(self, alias_id: str) -> str | None:
        """Reverse one recorded face-and-voice merge, restoring ``alias_id`` as its own identity.

        Cross-modal binding infers that one voice and one face belong to the same person from
        their co-occurrence, which corroboration makes unlikely to be wrong but cannot make
        impossible. Returns the restored identity ID, or None when the merge is not reversible
        because no record names which modality was contributed. The restored identity keeps no
        name or relationship; the identity it was merged into keeps both.

        This resets the pair's accumulated evidence; it does not suppress the pair. If the same
        voice and face keep co-occurring, they will be corroborated and merged again.
        """
        with (
            self._trace("mindbridge.unlink_identity", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            requested_id = _identifier(alias_id, "alias_id")
            with _translate_storage_errors("unlink identity"):
                restored = self._store.unlink_identity(requested_id)
            self._drain_outbox()
            return restored

    def _register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
        speaker: bool,
    ) -> None:
        id_label = "speaker_id" if speaker else "identity_id"
        requested_id = _identifier(identity_id, id_label)
        normalized_name = _identity_name(name)
        normalized_relationship = (
            None if relationship is None else _identity_relationship(relationship)
        )
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
                    relationship=normalized_relationship,
                    memories=memories,
                    embeddings=embeddings,
                )
            if not registered:
                if speaker:
                    raise SpeakerNotFoundError(f"speaker does not exist: {requested_id}")
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            self._drain_outbox()

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        """Erase a person: their biometric template, their aliases, and their indexed name.

        `delete` removes a memory; this removes a *person* from every memory. Both are needed,
        because they answer different requests: "forget that evening" and "forget me". Erasing
        the identity rows alone would not honour the second one, since a registered name is
        written into the indexed document, so the search projection would still answer to it.

        Memories, their content and their media survive, and a transcript keeps its words with
        the speaker attribution dropped. Forgetting a person is not forgetting the evening.

        This does **not** stop a later encounter from minting a fresh identity for the same
        person: recognising someone as previously-forgotten would require keeping the template
        this destroys. A deployment that wants "never recognise this person again" needs a
        retained blocklist, which is the opposite of a deletion and must not be spelled like one.
        """
        requested_id = _identifier(identity_id, "identity_id")
        with (
            self._trace("mindbridge.forget_identity", kind="operation"),
            self._operation() as operation,
            self._write_lock,
        ):
            with _translate_storage_errors("read identity memories"):
                normalized_id = self._store.resolve_identity_id(requested_id)
                memory_ids = (
                    None
                    if normalized_id is None
                    else self._store.identity_memory_ids(normalized_id)
                )
            if normalized_id is None or memory_ids is None:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            memories: tuple[StoredMemory, ...] = ()
            embeddings: tuple[StoredEmbedding, ...] = ()
            if memory_ids:
                with _translate_storage_errors("read identity memories"):
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
                    # `speaker_name=None` is what drops the name from the projection. The rebuilt
                    # documents are handed to the store so the erasure and the reindex commit in
                    # one transaction: a crash between them would leave the name searchable for a
                    # person whose template was already gone.
                    memories, embeddings = self._refresh_speaker_memories(
                        indexed,
                        speaker_id=normalized_id,
                        speaker_name=None,
                        operation=operation,
                    )
            with _translate_storage_errors("forget identity"):
                erasure = self._store.forget_identity(
                    normalized_id,
                    memories=memories,
                    embeddings=embeddings,
                )
            if erasure is None:
                raise IdentityNotFoundError(f"identity does not exist: {requested_id}")
            self._drain_outbox()
            return erasure

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

    # -- Agentic memory control plane ----------------------------------------------------------
    # One bounded memory-management loop (gate 3 of docs/context-os.md). The backend sees a
    # bounded evidence set and only proposes; every field is validated here, each accepted
    # operation commits with its own append-only log row, and `rollback()` reverses it. Physical
    # deletion is not an intent: it stays on `delete()` under host authority. None of these
    # methods is exposed on REST or MCP.

    def consolidation_candidates(self, *, limit: int = 32) -> tuple[ConsolidationCandidate, ...]:
        """Ask what needs deliberation, at most `limit` rows, interleaved across triggers.

        This is the durable trigger the slow loop runs on: every row is derived from state
        already committed -- evidence links, lineage disagreement, recorded confirmations --
        rather than from a clock. Hand a row's `memory_ids` straight to
        `consolidate(evidence_ids=...)` with the row's `trigger`.
        """
        with (
            self._trace("mindbridge.consolidation_candidates", kind="operation"),
            self._operation(),
        ):
            _limit(limit, maximum=100)
            with _translate_storage_errors("list consolidation candidates"):
                rows = self._store.read_consolidation_candidates(limit=limit)
            return tuple(
                ConsolidationCandidate(
                    trigger=MemoryTrigger(row.trigger),
                    memory_ids=row.memory_ids,
                    evidence_count=row.evidence_count,
                )
                for row in rows
            )

    def consolidate(
        self,
        *,
        evidence_ids: Sequence[str] | None = None,
        query: ContentInput | None = None,
        limit: int = 32,
        trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    ) -> ConsolidationReport:
        """Deliberate over a bounded evidence set and apply the operations policy accepts."""
        with (
            self._trace("mindbridge.consolidate", kind="operation"),
            self._operation() as assets,
        ):
            _limit(limit, maximum=100)
            if self._consolidator is None:
                raise ModelError(
                    "consolidation backend is not configured",
                    reason="backend_not_configured",
                    stage="consolidate",
                )
            if not isinstance(trigger, MemoryTrigger):
                raise ValidationError("trigger must be a MemoryTrigger")
            shown = self._consolidation_evidence(
                evidence_ids,
                query,
                limit=limit,
                operation=assets,
            )
            if not shown:
                return ConsolidationReport()
            applied: builtins.list[MemoryOperationRecord] = []
            rejected: builtins.list[tuple[MemoryOperation, str]] = []
            # One pass must not contradict itself: an operation may not build on evidence an
            # earlier accepted operation retired, nor retire evidence an earlier one built on.
            # There is no way to submit a proposal built in some other pass, so this is the only
            # reachable form of the staleness the kernel is required to reject.
            consumed: set[str] = set()
            retired: set[str] = set()
            with self._formation_lock:
                for operation in self._propose_operations(
                    tuple(shown.values()),
                    trigger=trigger,
                ):
                    targets = _retiring_targets(operation)
                    if targets & consumed or set(operation.evidence_ids) & retired:
                        rejected.append((operation, "inconsistent_batch"))
                        continue
                    try:
                        applied.append(
                            self._apply_memory_operation(
                                operation,
                                trigger=trigger,
                                model_id=self._consolidation_model,
                                recipe=self._consolidation_recipe,
                                shown=shown,
                                assets=assets,
                            )
                        )
                    except _RejectedOperation as rejection:
                        rejected.append((operation, rejection.reason))
                    else:
                        consumed.update(set(operation.evidence_ids) - targets)
                        retired.update(targets)
            return ConsolidationReport(operations=tuple(applied), rejected=tuple(rejected))

    def forget(self, memory_ids: Sequence[str]) -> MemoryOperationRecord | None:
        """Cognitively forget memories: recall skips them, `get()` and `list()` keep them.

        This is not deletion. `MemoryRecord.forgotten_at` stays readable for audit and
        `rollback()` restores recall. Use `delete()` to remove a record and its media.
        """
        with (
            self._trace("mindbridge.forget", kind="operation"),
            self._operation() as assets,
        ):
            if isinstance(memory_ids, (str, bytes)):
                raise ValidationError("memory_ids must be a sequence of memory IDs")
            try:
                targets = tuple(_identifier(memory_id, "memory_id") for memory_id in memory_ids)
            except TypeError:
                raise ValidationError("memory_ids must be a sequence of memory IDs") from None
            if not targets:
                return None
            try:
                return self._apply_memory_operation(
                    MemoryOperation(intent=MemoryIntent.FORGET, target_ids=targets),
                    trigger=MemoryTrigger.MANUAL,
                    model_id=None,
                    recipe=None,
                    shown=None,
                    assets=assets,
                )
            except _RejectedOperation:
                return None

    def rollback(self, operation_id: int) -> bool:
        """Reverse one applied operation; an unknown or already-reversed one reports `False`."""
        with (
            self._trace("mindbridge.rollback", kind="operation"),
            self._operation(),
            self._write_lock,
        ):
            if (
                isinstance(operation_id, bool)
                or not isinstance(operation_id, int)
                or operation_id <= 0
            ):
                raise ValidationError("operation_id must be a positive integer")
            with _translate_storage_errors("read a memory operation"):
                logged = self._store.read_operations(operation_id=operation_id)
            if not logged or logged[0].rolled_back_at is not None:
                return False
            row = logged[0]
            operation = load_operation(row.operation_json)
            with _translate_storage_errors("roll back a memory operation"):
                # One transaction: the created records disappear in the same commit that marks
                # the operation rolled back, so a crash between the two cannot leave an active
                # operation whose recorded output is gone.
                reverted, orphaned = self._store.rollback_operation(
                    row.operation_id,
                    rolled_back_at=datetime.now(timezone.utc),
                    delete_memory_ids=(
                        row.created_ids if operation.intent is MemoryIntent.CONSOLIDATE else ()
                    ),
                    # `linked` is the evidence this operation actually inserted, so a link that
                    # predated it survives the reversal. Records deleted above take their own.
                    retire_evidence=row.linked,
                    restore_versions=(
                        row.changed_ids if operation.intent is MemoryIntent.CORRECT else ()
                    ),
                    # Recorded per operation, so cognitive forgetting and the consolidation
                    # forgetting a CONSOLIDATE carried both reverse through one field. A FORGET
                    # row logged before that field existed carries the same IDs in `changed_ids`,
                    # and must still un-forget rather than silently do nothing.
                    clear_forgotten=row.forgotten_ids
                    or (row.changed_ids if operation.intent is MemoryIntent.FORGET else ()),
                )
            self._queue_asset_cleanup(orphaned)
            self._drain_outbox()
            return reverted

    def operations(self, *, limit: int = 100) -> tuple[MemoryOperationRecord, ...]:
        """List logged control-plane operations, newest first."""
        with self._trace("mindbridge.operations", kind="operation"), self._operation():
            _limit(limit, maximum=100)
            with _translate_storage_errors("list memory operations"):
                logged = self._store.read_operations(limit=limit)
            return tuple(_operation_record(row) for row in logged)

    def _consolidation_evidence(
        self,
        evidence_ids: Sequence[str] | None,
        query: ContentInput | None,
        *,
        limit: int,
        operation: _OperationAssets,
    ) -> dict[str, MemoryRecord]:
        """Resolve the bounded, active evidence set the backend is allowed to cite."""
        if evidence_ids is not None:
            if isinstance(evidence_ids, (str, bytes)):
                raise ValidationError("evidence_ids must be a sequence of memory IDs")
            try:
                candidates = tuple(
                    dict.fromkeys(
                        _identifier(memory_id, "evidence_id") for memory_id in evidence_ids
                    )
                )
            except TypeError:
                raise ValidationError("evidence_ids must be a sequence of memory IDs") from None
        elif query is not None:
            prepared = self._prepare_content(query, operation)
            reference = datetime.now(timezone.utc)
            outcome = self._search_prepared(
                prepared,
                limit=limit,
                operation=operation,
                memory_type=None,
                reference_at=reference,
                temporal_range=None,
                occurred_from=None,
                occurred_until=None,
                scope=None,
                require_unambiguous=False,
                capture_trace=False,
            )
            candidates = tuple(hit.id for hit in outcome.hits)
        else:
            # ponytail: over-fetch and filter rather than teach the keyset query about
            # visibility. Raise the factor only if a store of mostly forgotten records
            # measurably under-fills the window.
            with _translate_storage_errors("list consolidation evidence"):
                newest = self._store.list_memories(limit=min(1_000, limit * 4))
            candidates = tuple(memory.memory_id for memory in newest)
        if not candidates:
            return {}
        with _translate_storage_errors("read consolidation evidence"):
            memories = self._store.read_memories(candidates, active_only=True)
        memories = memories[:limit]
        self._lease_assets(
            tuple(asset for memory in memories for asset in memory.assets),
            operation.leased,
        )
        return {memory.memory_id: self._memory_record(memory) for memory in memories}

    def _propose_operations(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        assert self._consolidator is not None
        try:
            with self._model_trace(
                "consolidation",
                "consolidate",
                model=self._consolidation_model,
                batch_size=len(evidence),
                modalities=(record.modality for record in evidence),
            ):
                proposals = self._consolidator.consolidate(evidence, trigger=trigger)
        except MindBridgeError:
            raise
        except Exception as error:
            raise ModelError(
                "memory consolidation failed",
                reason="model_failed",
                stage="consolidate",
            ) from error
        if not isinstance(proposals, tuple) or any(
            not isinstance(value, MemoryOperation) for value in proposals
        ):
            raise ModelError(
                "consolidation backend returned an invalid batch",
                reason="response_invalid",
                stage="consolidate",
            )
        return proposals

    def _apply_memory_operation(
        self,
        operation: MemoryOperation,
        *,
        trigger: MemoryTrigger,
        model_id: str | None,
        recipe: str | None,
        shown: Mapping[str, MemoryRecord] | None,
        assets: _OperationAssets,
    ) -> MemoryOperationRecord:
        key = operation_key(operation, recipe=recipe)
        with _translate_storage_errors("check a memory operation"):
            # Raise the rejection outside this block: `_RejectedOperation` is not a
            # `MindBridgeError`, so the translator would turn it into a StorageError.
            duplicate = bool(self._store.read_operations(operation_key=key))
        if duplicate:
            raise _RejectedOperation("duplicate")
        pending = StoredOperation(
            operation_key=key,
            intent=operation.intent.value,
            trigger=trigger.value,
            model_id=model_id,
            recipe=recipe,
            operation_json=dump_operation(operation),
            applied_at=datetime.now(timezone.utc),
        )
        if operation.intent is MemoryIntent.CONSOLIDATE:
            return _operation_record(
                self._apply_consolidation(operation, pending, shown=shown, assets=assets)
            )
        return _operation_record(self._apply_operation_effects(operation, pending, shown=shown))

    def _apply_operation_effects(
        self,
        operation: MemoryOperation,
        pending: StoredOperation,
        *,
        shown: Mapping[str, MemoryRecord] | None,
    ) -> StoredOperation:
        """Validate a REINFORCE, CORRECT, or FORGET proposal and commit its effect."""
        targets = self._control_records(operation.target_ids)
        if not targets:
            raise _RejectedOperation("unknown_target")
        reinforce: tuple[tuple[str, str], ...] = ()
        correct_ids: tuple[str, ...] = ()
        forget_ids: tuple[str, ...] = ()
        if operation.intent is MemoryIntent.REINFORCE:
            reinforce = self._reinforcement_pairs(operation, targets, shown=shown)
        elif operation.intent is MemoryIntent.CORRECT:
            correct_ids = tuple(
                memory_id for memory_id in operation.target_ids if _is_derived(targets, memory_id)
            )
            if not correct_ids:
                raise _RejectedOperation("not_derived")
        else:
            forget_ids = tuple(
                memory_id
                for memory_id, memory in (
                    (value, targets.get(value)) for value in operation.target_ids
                )
                if memory is not None and memory.forgotten_at is None
            )
            if not forget_ids:
                raise _RejectedOperation("already_forgotten")
        with self._write_lock:
            with _translate_storage_errors("apply a memory operation"):
                logged = self._store.apply_control_operation(
                    pending,
                    reinforce=reinforce,
                    correct_ids=correct_ids,
                    forget_ids=forget_ids,
                )
            if logged is None:
                raise _RejectedOperation("duplicate")
            self._drain_outbox()
        return logged

    def _reinforcement_pairs(
        self,
        operation: MemoryOperation,
        targets: Mapping[str, StoredMemory],
        *,
        shown: Mapping[str, MemoryRecord] | None,
    ) -> tuple[tuple[str, str], ...]:
        target = operation.target_ids[0]
        record = targets.get(target)
        if record is None:
            raise _RejectedOperation("unknown_target")
        if not _is_derived(targets, target):
            raise _RejectedOperation("not_derived")
        # Self-citation first: a record that is its own evidence is malformed whatever set it
        # came from, and a hidden target is never in the shown set anyway.
        if target in operation.evidence_ids:
            raise _RejectedOperation("target_is_evidence")
        if shown is not None and not set(operation.evidence_ids) <= set(shown):
            raise _RejectedOperation("evidence_not_shown")
        if set(self._control_records(operation.evidence_ids)) != set(operation.evidence_ids):
            raise _RejectedOperation("unknown_evidence")
        context = record.context
        linked = frozenset(() if context is None else context.evidence_ids)
        # Every cited source must be new. Reinforcement is a claim about independent support, so
        # a proposal that miscounts what already supports the record is refused, not trimmed.
        if any(source in linked for source in operation.evidence_ids):
            raise _RejectedOperation("already_linked")
        return tuple((target, source) for source in operation.evidence_ids)

    def _apply_consolidation(
        self,
        operation: MemoryOperation,
        pending: StoredOperation,
        *,
        shown: Mapping[str, MemoryRecord] | None,
        assets: _OperationAssets,
    ) -> StoredOperation:
        """Validate one consolidation proposal and commit it through the formation path."""
        if shown is None or not set(operation.evidence_ids) <= set(shown):
            raise _RejectedOperation("evidence_not_shown")
        # Consolidation forgetting retires sources *this* derived record replaces, so a target
        # outside its own evidence has no lineage relationship to it and is refused. Retiring a
        # record no derived memory now covers is a FORGET, not a consolidation.
        if not set(operation.target_ids) <= set(operation.evidence_ids):
            raise _RejectedOperation("target_not_evidence")
        proposal = operation.proposal
        assert proposal is not None
        sources = tuple(shown[memory_id] for memory_id in operation.evidence_ids)
        primary = _consolidation_primary(proposal, sources)
        if primary is None:
            raise _RejectedOperation("invalid_proposal")
        # One operation, one transaction time: the derived record, its evidence links, and any
        # consolidation forgetting all carry the timestamp the log row reports.
        now = pending.applied_at
        context = replace(
            _formation_context(
                primary,
                proposal,
                model_id=self._consolidation_model,
                recipe=self._consolidation_recipe,
                recorded_at=now,
            ),
            evidence_ids=tuple(sorted(source.id for source in sources)),
        )
        prepared = replace(
            _prepare_memory(
                self._prepare_content(proposal.content, assets),
                occurred_at=primary.occurred_at,
                occurred_end=primary.occurred_end,
                metadata=None,
                memory_type=_formation_memory_type(proposal.kind),
            ),
            memory_id=_formation_memory_id(
                "\n".join(sorted(source.id for source in sources)),
                proposal,
                recipe=self._consolidation_recipe,
                context=context,
            ),
            context=context,
        )
        logged = self._commit_formation(
            tuple((prepared, source.id, proposal.confidence) for source in sources),
            (),
            completed_at=now,
            recipe=self._consolidation_recipe,
            operation=pending,
            forget_ids=operation.target_ids,
        )
        if logged is None:
            raise _RejectedOperation("duplicate")
        return logged

    def _control_records(self, memory_ids: Sequence[str]) -> dict[str, StoredMemory]:
        if not memory_ids:
            return {}
        with _translate_storage_errors("read control-plane memories"):
            memories = self._store.read_memories(tuple(memory_ids))
        return {memory.memory_id: memory for memory in memories}

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
        context: ObservationContext | None,
    ) -> _PreparedMemory:
        """Prepare one batch item, naming its position when it is the item that fails."""
        try:
            return _prepare_memory(
                self._prepare_content(content, operation),
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                metadata=metadata,
                memory_type=memory_type,
                context=context,
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
            raise StorageError("failed to materialize media input", reason="io_failed") from error

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
            raise StorageError(
                "content-addressed media metadata is inconsistent", reason="asset_changed"
            )

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
        # Settled before this write takes its own speech-index guard: the two batches are
        # independent, and nesting the guards would share one rollback list between them.
        self._settle_queued(existing_rows, operation=operation)
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
                # Derived visual text can rescue media this embedder cannot take, so it has to
                # exist before the fallback guard below decides the write is impossible.
                described = self._pending_visual_descriptions(
                    tuple(memory.content for memory in missing)
                )
                missing = [
                    replace(
                        memory,
                        content=self._with_visual_descriptions(memory.content, described),
                    )
                    for memory in missing
                ]
                fallback = Modality.AUDIO not in self._embedding_capabilities
                if fallback:
                    for memory in missing:
                        _fallback_unsupported(
                            memory.content,
                            self._embedding_capabilities,
                            "embedding",
                            rescuable=self._transcript_fallback(memory.content.assets),
                        )
                if fallback:
                    self._cache_audio_transcripts(
                        tuple(
                            asset
                            for memory in missing
                            if not memory.content.audio_transcript
                            for asset in memory.content.assets
                        ),
                        operation,
                    )
                elif self._derives_transcripts(batched):
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
                        # Only an `ObservationContext` carries a place: it is what the caller
                        # supplies at capture time. `MemoryContext` is the formed-semantics shape
                        # and deliberately has none.
                        place_id=(
                            memory.context.place_id
                            if isinstance(memory.context, ObservationContext)
                            else None
                        ),
                        context=_stored_memory_context(memory, recorded_at=now),
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
                    # With a former configured, the same transaction enqueues the formation this
                    # call still owes. `add()` deletes the row once it returns, so the row only
                    # outlives the commit when the process did not: the next `settle()` then
                    # finds an embedded record that owes formation and finishes it.
                    self._store.write_memories(
                        stored_memories,
                        stored_embeddings,
                        formation_pending_at=(
                            stored_memories[0].created_at if self._former is not None else None
                        ),
                    )
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
            raise StorageError("written memories could not be read from SQLite", reason="io_failed")
        return tuple(self._memory_record(rows_by_id[memory.memory_id]) for memory in prepared)

    def _complete_formation(self, records: Sequence[MemoryRecord]) -> None:
        """Clear the queue rows `_add_prepared` wrote once formation has actually run.

        A no-op without a former, and harmless for a record that was never enqueued or whose row
        `_settle_queued` already removed: the delete simply matches nothing.
        """
        if self._former is None or not records:
            return
        with _translate_storage_errors("complete formation"):
            self._store.complete_captures(tuple(record.id for record in records))

    def _settle_queued(
        self,
        existing: Sequence[StoredMemory],
        *,
        operation: _OperationAssets,
    ) -> None:
        """Settle rows `add()` found already stored but still queued from `capture()`.

        Such a row is durable and has no vectors, so `add()` owes it the enrichment `capture()`
        deferred instead of taking the "already exists" shortcut past every model.
        """
        if not existing:
            return
        with _translate_storage_errors("check the capture queue"):
            queued = frozenset(
                pending.memory_id
                for pending in self._store.pending_captures(
                    limit=len(existing),
                    memory_ids=tuple(row.memory_id for row in existing),
                )
            )
        # No attempt ceiling here: `add()` promises a searchable return, so a record it is being
        # asked for has to be settled however many times it has already failed.
        _settled, failures = self._settle_stored(
            tuple(row for row in existing if row.memory_id in queued),
            operation=operation,
        )
        if failures:
            raise failures[0]

    def _settle_stored(
        self,
        rows: Sequence[StoredMemory],
        *,
        operation: _OperationAssets,
    ) -> tuple[tuple[str, ...], tuple[MindBridgeError, ...]]:
        """Run the stages `capture()` deferred over already-committed rows, and report both sides.

        `settle()` and the `add()` path share this routine, so a captured record reaches exactly
        the state a blocking `add()` would have left it in. Each row commits on its own and a
        failing one is collected rather than raised, so it keeps its queue row, its attempt count,
        and its reason while every other record in the batch still settles. Returns the IDs that
        settled and the failures, and the caller decides which of them to raise.

        Held under `_settle_lock`, which is the whole cross-thread guard: a `settle()` and an
        `add()` of the same captured content would otherwise both find the row unembedded and
        run every model stage over it. Serialized, the second caller reaches `_settle_row`'s
        vector check after the first committed and owes formation only.
        """
        settled: list[str] = []
        failures: list[MindBridgeError] = []
        for row in rows:
            try:
                with self._settle_lock:
                    self._settle_row(row, operation=operation)
            except MindBridgeError as error:
                with _translate_storage_errors("record a failed settlement"):
                    self._store.record_capture_failure(row.memory_id, str(error))
                if error.subject is None:
                    error.subject = row.memory_id
                failures.append(error)
                continue
            settled.append(row.memory_id)
        return tuple(settled), tuple(failures)

    def _settle_row(self, row: StoredMemory, *, operation: _OperationAssets) -> None:
        # An `add()` that crashed between its commit and formation leaves a queue row over a
        # record that is already enriched, embedded, and indexed. Its vectors are final, so
        # settling it owes formation only; re-running the model stages would buy the same
        # vectors twice.
        with _translate_storage_errors("check a captured memory's vectors"):
            embedded = self._store.read_embedding(_embedding_id(row.memory_id, 0)) is not None
        settled = row
        if not embedded:
            enriched = self._enrich_row(row, operation=operation)
            if enriched is None:
                return
            settled = enriched
        self._form_sources((self._memory_record(settled),), operation=operation)
        with _translate_storage_errors("complete captured memory"):
            self._store.complete_captures((row.memory_id,))

    def _enrich_row(
        self,
        row: StoredMemory,
        *,
        operation: _OperationAssets,
    ) -> StoredMemory | None:
        """Derive, embed, and commit one captured row, or return `None` if it lost the race."""
        memory = _prepared_from_stored(row)
        speech_assets = self._answer_speech_assets(memory.content.assets)
        speech_indexed = self._index_speech and bool(speech_assets)
        with self._speech_index_guard(operation, enabled=speech_indexed):
            if speech_indexed:
                self._recognize_speech(speech_assets, operation, reversible=True)
            # Same order as `_add_prepared`: derived visual text has to exist before the fallback
            # guard decides whether this embedder can take the media at all.
            described = self._pending_visual_descriptions((memory.content,))
            memory = replace(
                memory,
                content=self._with_visual_descriptions(memory.content, described),
            )
            if Modality.AUDIO not in self._embedding_capabilities:
                # Same rescue set `_add_prepared` allows, so a composition `add()` accepts is not
                # one `settle()` refuses after the record is already durable.
                _fallback_unsupported(
                    memory.content,
                    self._embedding_capabilities,
                    "embedding",
                    rescuable=self._transcript_fallback(memory.content.assets),
                )
                if not memory.content.audio_transcript:
                    self._cache_audio_transcripts(memory.content.assets, operation)
            elif self._derives_transcripts(memory.content.assets):
                self._cache_audio_transcripts(memory.content.assets, operation)
            memory = self._prepare_for_embedding(memory, operation)
            vectors, parts = self._embed_document_parts(
                tuple(
                    (memory, object_part, model_input)
                    for object_part, model_input in enumerate(
                        self._embedding_inputs(memory.content)
                    )
                )
            )
            now = datetime.now(timezone.utc)
            enriched = replace(row, content=memory.content.text, updated_at=now)
            with (
                self._trace("mindbridge.storage.write", kind="stage"),
                _translate_storage_errors("settle captured memory"),
            ):
                # The captured row already carries its assets and its observation context, so
                # this commit replaces only the derived text and adds the queued vectors. The
                # queue row survives it: formation is still owed, and a former failure below must
                # leave the record retryable rather than searchable but silently unformed.
                committed = self._store.settle_capture(
                    replace(enriched, context=None),
                    tuple(
                        StoredEmbedding(
                            embedding_id=_embedding_id(row.memory_id, object_part),
                            memory_id=row.memory_id,
                            values=vector,
                            model_id=self._embedding_model,
                            space_id=self._space_id,
                            task=_DOCUMENT_TASK,
                            created_at=now,
                            object_part=object_part,
                            normalized=True,
                        )
                        for (_memory, object_part, _model_input), vector in zip(
                            parts, vectors, strict=True
                        )
                    ),
                )
            with self._write_lock:
                self._drain_outbox()
                operation.persisted.update(asset.asset_id for asset in row.assets)
                if operation.speech_updates:
                    self._persist_transcripts(operation)
        return enriched if committed else None

    def _form_sources(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: _OperationAssets,
    ) -> None:
        if self._former is None or not sources:
            return
        with self._formation_lock:
            self._form_sources_locked(sources, operation=operation)

    def _form_sources_locked(
        self,
        sources: Sequence[MemoryRecord],
        *,
        operation: _OperationAssets,
    ) -> None:
        assert self._former is not None
        pending = tuple({source.id: source for source in sources}.values())
        pending = tuple(
            source
            for source in pending
            if not self._store.formation_completed(source.id, self._formation_space)
        )
        if not pending:
            return
        all_inputs = tuple(
            FormationInput(
                memory_id=source.id,
                content=ModelInput(text=source.content, assets=source.assets),
                context=_observation_from_record(source),
            )
            for source in pending
        )
        inputs = tuple(
            value
            for value in all_inputs
            if value.content.modalities <= self._formation_capabilities
        )
        if not inputs:
            return
        try:
            with self._model_trace(
                "formation",
                "form",
                model=self._formation_model,
                batch_size=len(inputs),
                modalities=(modality for value in inputs for modality in value.content.modalities),
            ):
                proposals_by_source = self._former.form(inputs)
        except MindBridgeError:
            raise
        except Exception as error:
            raise ModelError(
                "automatic memory formation failed",
                reason="model_failed",
                stage="form",
            ) from error
        if (
            not isinstance(proposals_by_source, tuple)
            or len(proposals_by_source) != len(inputs)
            or any(
                not isinstance(values, tuple)
                or any(not isinstance(value, FormationProposal) for value in values)
                for values in proposals_by_source
            )
        ):
            raise ModelError(
                "formation backend returned an invalid batch",
                reason="response_invalid",
                stage="form",
            )
        proposals_by_id: dict[str, tuple[FormationProposal, ...]] = {
            value.memory_id: proposals
            for value, proposals in zip(inputs, proposals_by_source, strict=True)
        }

        now = datetime.now(timezone.utc)
        pairs: builtins.list[tuple[_PreparedMemory, str, float]] = []
        inputs_by_id = {value.memory_id: value for value in inputs}
        formed_sources = tuple(source for source in pending if source.id in inputs_by_id)
        for source in formed_sources:
            proposals = proposals_by_id.get(source.id, ())
            for proposal in proposals:
                _validate_formation_proposal(proposal, inputs_by_id[source.id])
                prepared = _prepare_memory(
                    self._prepare_content(proposal.content, operation),
                    occurred_at=source.occurred_at,
                    occurred_end=source.occurred_end,
                    metadata=None,
                    memory_type=_formation_memory_type(proposal.kind),
                )
                context = _formation_context(
                    source,
                    proposal,
                    model_id=self._formation_model,
                    recipe=self._formation_space,
                    recorded_at=now,
                )
                pairs.append(
                    (
                        replace(
                            prepared,
                            memory_id=_formation_memory_id(
                                source.id,
                                proposal,
                                recipe=self._formation_space,
                                context=context,
                            ),
                            context=context,
                        ),
                        source.id,
                        proposal.confidence,
                    )
                )
        _validate_formation_pairs(pairs)
        self._commit_formation(pairs, formed_sources, completed_at=now)

    def _commit_formation(
        self,
        pairs: Sequence[tuple[_PreparedMemory, str, float]],
        sources: Sequence[MemoryRecord],
        *,
        completed_at: datetime,
        recipe: str | None = None,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
    ) -> StoredOperation | None:
        """Embed and commit derived records; return the log row when one was requested."""
        recipe = self._formation_space if recipe is None else recipe
        ordered_pairs = tuple(sorted(pairs, key=lambda value: (value[0].memory_id, value[1])))
        unique = tuple(
            {
                prepared.memory_id: prepared
                for prepared, _source, _score in reversed(ordered_pairs)
            }.values()
        )
        unique = tuple(sorted(unique, key=lambda value: value.memory_id))
        with _translate_storage_errors("check formed memories"):
            existing = self._store.read_memories(tuple(value.memory_id for value in unique))
        existing_ids = {value.memory_id for value in existing}
        missing = tuple(value for value in unique if value.memory_id not in existing_ids)
        parts = tuple(
            (memory, object_part, model_input)
            for memory in missing
            for object_part, model_input in enumerate(self._embedding_inputs(memory.content))
        )
        vectors = self._embed(
            tuple(model_input for _memory, _part, model_input in parts),
            task=EmbedTask.DOCUMENT,
        )
        stored = tuple(
            StoredMemory(
                memory_id=memory.memory_id,
                content=memory.content.text,
                modality=memory.content.modality.value,
                memory_type=memory.memory_type.value,
                assets=(),
                metadata_json=memory.metadata_json,
                occurred_at=memory.occurred_at,
                occurred_end=memory.occurred_end,
                created_at=completed_at,
                updated_at=completed_at,
                context=cast(MemoryContext, memory.context),
            )
            for memory in missing
        )
        embeddings = tuple(
            StoredEmbedding(
                embedding_id=_embedding_id(memory.memory_id, object_part),
                memory_id=memory.memory_id,
                values=vector,
                model_id=self._embedding_model,
                space_id=self._space_id,
                task=_DOCUMENT_TASK,
                created_at=completed_at,
                object_part=object_part,
                normalized=True,
            )
            for (memory, object_part, _model_input), vector in zip(parts, vectors, strict=True)
        )
        if operation is not None:
            created = tuple(value.memory_id for value in missing)
            operation = replace(
                operation,
                created_ids=created,
                changed_ids=tuple(
                    memory_id
                    for memory_id in dict.fromkeys(
                        prepared.memory_id for prepared, _source, _score in ordered_pairs
                    )
                    if memory_id not in set(created)
                ),
            )
        with self._write_lock:
            with _translate_storage_errors("commit automatic memory formation"):
                applied = self._store.apply_formation(
                    stored,
                    embeddings,
                    evidence=tuple(
                        (prepared.memory_id, source_id, confidence)
                        for prepared, source_id, confidence in ordered_pairs
                    ),
                    source_memory_ids=tuple(source.id for source in sources),
                    recipe=recipe,
                    completed_at=completed_at,
                    operation=operation,
                    forget_ids=forget_ids,
                )
            self._drain_outbox()
        if operation is None:
            return None
        if not applied:
            # A concurrent duplicate won the key inside the transaction. Its row is in the log
            # under the same key, but it is not this call's operation, so report the duplicate.
            return None
        with _translate_storage_errors("read a memory operation"):
            logged = self._store.read_operations(operation_key=operation.operation_key)
        return logged[0] if logged else None

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

    def _pending_visual_descriptions(self, contents: Sequence[_PreparedContent]) -> dict[str, str]:
        """Describe every yet-undescribed visual asset in one write, in one model call.

        Deriving the text is a paid call, so it follows from `vision_describer` being configured
        and from nothing else. Whether it is *indexed* is not a capability question -- see
        `_with_visual_descriptions`. Only the write path calls this: `_embedding_content` is
        shared with the query path, where describing an image query would buy a call per search.
        """
        if self._vision_describer is None:
            return {}
        assets = tuple(
            {
                asset.asset_id: asset
                for content in contents
                for asset in content.assets
                if Modality(asset.modality) in self._vision_capabilities
                and f"[visual description:{asset.asset_id}]\n" not in content.text
            }.values()
        )
        if not assets:
            return {}
        descriptions = self._vision_descriptions(
            tuple(self._resolved_model_input(_asset_content(asset)) for asset in assets)
        )
        return dict(zip((asset.asset_id for asset in assets), descriptions, strict=True))

    def _with_visual_descriptions(
        self,
        prepared: _PreparedContent,
        descriptions: Mapping[str, str],
    ) -> _PreparedContent:
        """Union derived visual text into the indexed document, whatever the embedder can take.

        Routing media to the embedder is a capability decision; what a memory's full-text document
        contains is not. Describing an image only when the embedder could not take one made the
        recommended omni composition the one that stored the empty string as its whole BM25
        document, so a stronger embedder deleted the lexical half of the dense+lexical union.
        Union does not lose here; replacement does.
        """
        sections = tuple(
            f"[visual description:{asset.asset_id}]\n{descriptions[asset.asset_id]}"
            for asset in prepared.assets
            if asset.asset_id in descriptions
            and f"[visual description:{asset.asset_id}]\n" not in prepared.text
        )
        if not sections:
            return prepared
        # A description is derived convenience, not the caller's content, so one that does not fit
        # is omitted rather than allowed to fail a write whose own text is inside the limit. The
        # asset is still stored and still embedded; each description that did land carries its
        # `[visual description:<asset_id>]` marker, so which ones are present is inspectable
        # through `get`. Failing here instead would make a long note plus an image unstorable for
        # no reason the caller could act on.
        kept: list[str] = []
        text = prepared.text
        for section in sections:
            candidate = "\n\n".join(value for value in (text, section) if value)
            if len(candidate) > _MAX_TEXT_CHARACTERS:
                continue
            text = candidate
            kept.append(section)
        if not kept:
            return prepared
        sections = tuple(kept)
        return replace(
            prepared,
            text=text,
            canonical_parts=(
                *prepared.canonical_parts,
                *(("visual_description", section) for section in sections),
            ),
            visual_description=True,
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
        text_parts = tuple(
            value
            for kind, value in prepared.canonical_parts
            if kind in {"text", "audio_transcript", "visual_description"}
        )
        if prepared.text and prepared.text != "\n\n".join(text_parts):
            text_parts = (*text_parts, prepared.text)
        if (
            prepared.audio_transcript or prepared.visual_description
        ) and Modality.TEXT not in self._embedding_capabilities:
            text_parts = ()
        text_keys = tuple(key for text in text_parts for key in _contextual_text_keys(text))
        atomic = [
            *(self._route_embedding(_text_content(text)) for text in text_keys),
            *(
                self._route_embedding(_asset_content(asset))
                for asset in prepared.assets
                if not (
                    (
                        prepared.audio_transcript
                        and asset.modality == Modality.AUDIO.value
                        and Modality.AUDIO not in self._embedding_capabilities
                        and asset.transcript is None
                    )
                    or (
                        prepared.visual_description
                        and asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
                        and Modality(asset.modality) not in self._embedding_capabilities
                    )
                )
            ),
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
                audio_transcript=_has_stream_transcript(base, memory.assets),
                visual_description=_has_stream_description(base, memory.assets),
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

    def _search_prepared(  # noqa: C901 - one shared retrieval plane
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
        scope: RetrievalScope | None,
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
        if not _lexical_query_terms(lexical_query):
            lexical_query = ""
        candidate_limit = max(_RERANK_CANDIDATES, limit * 3)
        candidate_ceiling = max(candidate_limit, limit * (_MAX_RETRIEVAL_KEYS + 1))
        seen_index_ids: set[str] = set()
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
                hydrated_documents = self._store.read_index_candidates(index_ids)
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
            candidate_parent_ids = tuple(
                dict.fromkeys(document.memory_id for document in documents)
            )
            with _translate_storage_errors("apply search scope"):
                active_count = len(
                    self._store.read_memories(
                        candidate_parent_ids,
                        valid_at=None if scope is None else scope.valid_at,
                        known_at=None if scope is None else scope.known_at,
                        near=None if scope is None else scope.near,
                        radius_m=None if scope is None else scope.radius_m,
                        # Passed here for consistency with every other scope axis, which all
                        # reach both reads. This one is the survivor count that drives candidate
                        # widening; no constructed corpus (30 or 120 memories) could make its
                        # absence change a result, so it is unproven rather than proven needed.
                        # Kept because omitting one axis at one of two sites is the anomaly a
                        # reader would have to explain, and because narrowing a count can only
                        # widen the search. The hydration site below is mutation-covered.
                        place_id=None if scope is None else scope.place_id,
                        active_only=True,
                    )
                )
            if (
                active_count >= limit
                or candidates.exhausted
                or candidate_limit >= candidate_ceiling
            ):
                break
            current_index_ids = set(index_ids)
            if current_index_ids <= seen_index_ids:
                break
            seen_index_ids.update(current_index_ids)
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
                memories = self._store.read_memories(
                    parent_ids,
                    valid_at=None if scope is None else scope.valid_at,
                    known_at=None if scope is None else scope.known_at,
                    near=None if scope is None else scope.near,
                    radius_m=None if scope is None else scope.radius_m,
                    place_id=None if scope is None else scope.place_id,
                    active_only=True,
                )
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
                    lexical_score = (
                        _LEXICAL_FULL_COVERAGE_RELEVANCE
                        * lexical_relevance_by_rank.get(memory_id, 0.0)
                        if lexical_strength >= _LEXICAL_FULL_COVERAGE
                        else 0.0
                    )
                    base = max(dense_relevance.get(memory_id, 0.0), lexical_score)
                    relevance = _bounded_scale(
                        base,
                        1.0 + _MAX_LEXICAL_RERANK_BONUS * lexical_strength,
                    )
                    lexical_rerank_bonus = relevance - base
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
                    # `minimum_relevance` gates evidence quality on the relevance scale: how well
                    # this memory matches, times how sure the observation itself was. It used to
                    # gate dense *confidence*, `(1 + cosine) / 2`, which made the 0.55 default
                    # admit cosine 0.15 and reject 0.05 while the caller read cosine back; and any
                    # full-text hit was handed a flat 0.6 there whatever its match strength, so one
                    # shared rare term carried a document at cosine -1.0 past the default floor and
                    # no floor above 0.6 could keep the lexical route at all. Both routes now
                    # contribute on this one scale and scale with match strength.
                    #
                    # The gate takes the signals the *query* asked about and leaves out the ones it
                    # did not. Temporal proximity is in: a caller who asks "in 2024" made the year
                    # part of the question, so overlapping it is evidence and missing it is not,
                    # and the boost is what lets an in-window full-text hit clear a floor its bare
                    # rank could not. Reinforcement and `decay_half_life_days` retention are out:
                    # neither is anything the query mentioned. They are bounded below by
                    # `_RANK_FLOOR`, so with retention inside the gate a *perfectly* relevant
                    # memory decayed to 0.30 and to 0.09 once a window also missed it, under the
                    # 0.10 default — turning "prefer recent" into "hide old" for exactly the
                    # caller who enabled decay and then asked about last year. A long-lived
                    # personal memory may rank an old event last; it may not stop returning it.
                    #
                    # The cost is that `SearchHit.score` carries every factor and so can sit below
                    # the floor the caller set. `search_with_trace` reports the gated quantity as
                    # `gate_relevance`, beside the `retention_factor` that moved the score off it.
                    gate_relevance = relevance
                    if temporal_factor is not None:
                        gate_relevance = _bounded_scale(gate_relevance, temporal_factor)
                    if memory.context is not None:
                        final_score *= memory.context.confidence
                        gate_relevance *= memory.context.confidence
                    ranked.append(
                        (
                            memory,
                            final_score,
                            gate_relevance,
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
                        gate_relevance=gate_relevance,
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
        documents: tuple[IndexCandidate, ...],
        *,
        lexical_query: str,
        temporal_range: tuple[datetime, datetime] | None,
        memory_type: MemoryType | None,
        route_limit: int,
        result_limit: int,
    ) -> tuple[_IndexCandidates, tuple[IndexCandidate, ...]]:
        if temporal_range is None or not lexical_query or len(candidates.lexical) < route_limit:
            return candidates, documents
        lexical_by_id = {hit.id: hit for hit in candidates.lexical}
        # A heuristic proxy for "worth deepening for", NOT a bound on the gate. It cannot be one:
        # the gate scores a full-text candidate from `lexical_relevance_by_rank`, a reciprocal
        # rank over the final candidate set, while `hit.relevance` here is the index's own
        # similarity -- and the rank a deepened candidate ends up with depends on the deepening
        # this decision is choosing whether to do. Tightening the proxy toward the gate's real
        # ceiling was tried and made it too permissive: the loop then stopped at a nearer weak
        # candidate and never reached a stronger in-range one outside the first window, which
        # `test_temporal_search_reads_lexical_evidence_from_authoritative_time_range` catches.
        # Treat this threshold as tuned, and re-run that test before changing it.
        qualified_parents = {
            document.memory_id
            for document in documents
            if (hit := lexical_by_id.get(document.embedding_id)) is not None
            and _LEXICAL_FULL_COVERAGE_RELEVANCE * hit.relevance >= self._minimum_relevance
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
            and _LEXICAL_FULL_COVERAGE_RELEVANCE * hit.relevance >= self._minimum_relevance
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
                and _LEXICAL_FULL_COVERAGE_RELEVANCE * hit.relevance >= self._minimum_relevance
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
            added = self._store.read_index_candidates(added_ids)
        by_id = {document.embedding_id: document for document in (*documents, *added)}
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
        if (
            missing == {Modality.TEXT}
            and (prepared.audio_transcript or prepared.visual_description)
            and value.modalities - {Modality.TEXT} <= self._embedding_capabilities
        ):
            return ModelInput(assets=value.assets)
        fallback = {Modality.AUDIO}
        if prepared.visual_description:
            fallback.update((Modality.IMAGE, Modality.VIDEO))
        if any(asset.transcript for asset in prepared.assets):
            # By here the transcript is attached to the asset, so the speech can stand in for a
            # video the embedder refuses. `_fallback_unsupported` already agreed a route exists on
            # the strength of it being derivable; this is the same decision once the text is in
            # hand, and without it the write raises after paying for the transcription. Read from
            # the assets rather than `prepared.audio_transcript`, which the transcript-derivation
            # path does not set -- the audio route never needed it, since AUDIO is unconditionally
            # in the fallback set above, and `_derived_text` reads the assets too.
            fallback.add(Modality.VIDEO)
        if missing <= fallback:
            text = _derived_text(value.text, prepared.assets)
            assets = tuple(asset for asset in value.assets if asset.modality not in missing)
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
        media = _prepared_modalities(prepared) - {Modality.TEXT}
        if (
            (prepared.audio_transcript or prepared.visual_description)
            and Modality.TEXT not in self._embedding_capabilities
            and media
            and media <= self._embedding_capabilities
        ):
            return prepared
        rescues = self._transcript_fallback(prepared.assets)
        unsupported = _fallback_unsupported(
            prepared,
            self._embedding_capabilities,
            "embedding",
            rescuable=rescues,
        )
        if Modality.AUDIO in unsupported:
            if prepared.audio_transcript:
                return prepared
            _require_audio_transcription(self._transcription_capabilities)
        elif Modality.VIDEO in unsupported and Modality.VIDEO in rescues:
            # The embedder cannot take the video but the transcriber can read it, so the speech
            # becomes the embedding key. The frames are not embedded in this composition -- the
            # honest cost of the route -- while the asset stays stored and on the record.
            if prepared.audio_transcript:
                return prepared
        elif unsupported & {Modality.IMAGE, Modality.VIDEO} or not self._derives_transcripts(
            prepared.assets
        ):
            return prepared
        return self._with_audio_transcripts(prepared, operation)

    def _route_generation(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> ModelInput:
        value = self._resolved_model_input(prepared)
        missing = value.modalities - self._generation_capabilities
        if (
            missing == {Modality.TEXT}
            and (prepared.audio_transcript or prepared.visual_description)
            and value.modalities - {Modality.TEXT} <= self._generation_capabilities
        ):
            return ModelInput(assets=value.assets)
        unsupported = _fallback_unsupported(
            prepared,
            self._generation_capabilities,
            "generation",
        )
        if self._answer_speech_assets(prepared.assets):
            prepared = self._with_speech_identities(prepared, operation)
        if Modality.AUDIO in unsupported and not prepared.audio_transcript:
            _require_audio_transcription(self._transcription_capabilities)
            prepared = self._with_audio_transcripts(prepared, operation)
        value = self._resolved_model_input(prepared)
        unsupported = value.modalities - self._generation_capabilities
        if not unsupported:
            return value
        text = _derived_text(value.text, prepared.assets)
        fallback = {Modality.AUDIO}
        if prepared.visual_description:
            fallback.update((Modality.IMAGE, Modality.VIDEO))
        assets = tuple(asset for asset in value.assets if asset.modality not in unsupported)
        if unsupported <= fallback and (text or assets):
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
                raise StorageError(
                    "memory references missing media metadata", reason="asset_unavailable"
                ) from None
            prepared_hits.append(
                _PreparedContent(
                    text=hit.content,
                    assets=assets,
                    modality=_memory_modality(assets),
                    canonical_parts=(),
                    audio_transcript=_has_stream_transcript(hit.content, assets),
                    visual_description=_has_stream_description(hit.content, assets),
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
                tuple(
                    asset
                    for prepared in prepared_hits
                    if not prepared.audio_transcript
                    for asset in prepared.assets
                ),
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
            raise ModelError(
                "face identity evidence exceeded the supported text length",
                reason="payload_too_large",
            )
        return replace(prepared, text=text)

    def _recognize_faces(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
    ) -> None:
        if not isinstance(self._face_analyzer, FaceBackend):
            raise ModelError("no face backend is configured", reason="backend_not_configured")
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
                    # Deliberately no preferred_identity. That shortcut adopted the asset's
                    # lone voice identity for its lone face with no corroboration at all, a
                    # second cross-modal door that bypassed the evidence gate in
                    # _link_asset_identity. Cross-modal binding now has exactly one entrance.
                    operation.face_observations[asset.asset_id] = self._store.write_faces(
                        asset.asset_id,
                        analysis,
                        model_id=self._face_model,
                        space_id=self._face_space,
                        analysis_space_id=self._face_analysis_space,
                        minimum_similarity=self._face_similarity,
                        minimum_margin=self._face_margin,
                    )
        analyzed = {asset.asset_id for asset in missing}
        with self._write_lock, _translate_storage_errors("link face and voice identities"):
            for asset in face_assets:
                self._link_asset_identity(asset.asset_id, operation)
        # Linking re-points these observations to the surviving identity, but every counted
        # value here is merge-invariant: one face identity maps to one surviving identity, and
        # `identity_score` is carried through unchanged, so the counts do not depend on whether
        # this runs before or after the link. The link decision has its own span.
        for asset in face_assets:
            self._trace_identity_yield(
                "mindbridge.identity.faces",
                tuple(
                    (observation.identity_id, observation.identity_score)
                    for observation in operation.face_observations[asset.asset_id]
                ),
                cached=asset.asset_id not in analyzed,
            )

    def _identity_link_is_corroborated(
        self,
        speaker_id: str,
        face_id: str,
        asset_id: str,
    ) -> bool:
        """Record this asset's voice-and-face co-occurrence and report whether it is enough.

        One asset's co-occurrence is not evidence that one person produced both. Egocentric
        capture is the adversarial case: the wearer speaks while a different person's face
        fills the frame, so a single clip would bind the listener's face to the wearer's voice
        and nothing downstream could tell.

        Counting assets raises the price of that mistake but does not prevent it. A wearer
        talks to the same person across many clips, so the wrong pair accumulates as fast as a
        genuine speaker's, and measurement on synthetic egocentric traffic confirms it: at the
        default of two assets the wearer still binds to an interlocutor's face under every
        ingestion order tried. What the count does buy is that a face seen once is never
        bound, and `_link_asset_identity` keeps the damage to that one bind by refusing to let
        an identity holding both modalities absorb anything further.
        """
        observed = self._store.record_identity_link_evidence(speaker_id, face_id, asset_id)
        corroborated = observed >= self._identity_link_min_assets
        with self._trace("mindbridge.identity.link", kind="stage") as span:
            span.set_attribute(IDENTITY_EVIDENCE_ASSETS, observed)
            span.set_attribute(IDENTITY_EVIDENCE_REQUIRED, self._identity_link_min_assets)
            span.set_attribute(IDENTITY_LINKED, corroborated)
        return corroborated

    def _relabelled_speaker_index(
        self,
        plan: IdentityLink,
        speaker_id: str,
        operation: _OperationAssets,
    ) -> tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]]:
        """Re-embed the merged speaker's indexed memories so the merge stays atomic."""
        if not (self._index_speech and plan.source_id == speaker_id):
            return (), ()
        memory_ids = self._store.speaker_memory_ids(plan.source_id)
        if not memory_ids:
            return (), ()
        indexed = tuple(
            memory
            for memory in self._store.read_memories(memory_ids)
            if any(
                f"[speech identities:{asset.asset_id}]\n" in memory.content
                for asset in memory.assets
                if asset.modality in {"audio", "video"}
            )
        )
        if not indexed:
            return (), ()
        return self._refresh_speaker_memories(
            indexed,
            speaker_id=plan.target_id,
            speaker_name=plan.name,
            previous_speaker_id=plan.source_id,
            update_operation=False,
            operation=operation,
        )

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
        if speaker_id == face_id:
            # Already one identity. Recording this would store an identity co-occurring with
            # itself, once per asset forever, and no such pair can ever yield a plan.
            return
        if not self._identity_link_is_corroborated(speaker_id, face_id, asset_id):
            return
        # Only a voice-only and a face-only identity may fuse here. Letting a fragment rejoin
        # an identity that already holds its modality is what turns one wrong cross-modal bind
        # into a cascade: once a wearer's voice owns one interlocutor's face, that identity
        # holds both modalities, and every later fragment (the interlocutor's own voice, then
        # the next interlocutor's face) is a fragment rejoining it. Measured on synthetic
        # egocentric traffic (one off-camera wearer, three interlocutors, random ingestion
        # order), permitting it collapsed all four people into one identity every time;
        # refusing it capped the damage at the single unavoidable first bind and raised
        # correct merges from 0/3 to 2/3. `LocalStore` still offers the wider merge to a
        # caller that has established the claim some other way.
        plan = self._store.identity_link_plan(speaker_id, face_id)
        if plan is None:
            return
        memories, embeddings = self._relabelled_speaker_index(plan, speaker_id, operation)
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
            raise ModelError("no face backend is configured", reason="backend_not_configured")
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
                raise ModelError("failed to analyze face input", reason="model_failed") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, FaceAnalysis) for analysis in analyses
            ):
                raise ModelError("face model returned invalid output", reason="response_invalid")
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

    def _transcript_fallback(self, assets: Sequence[StoredAsset]) -> frozenset[Modality]:
        """Modalities a derived transcript can rescue for an embedder that cannot take them.

        Empty when a `SpeechBackend` is configured: that composition indexes the same text through
        the `index_speech` opt-in instead, so claiming the rescue here would let a write past the
        guard and then fail later with no transcript in hand.
        """
        if not self._derives_transcripts(assets):
            return frozenset()
        return self._transcription_capabilities & {Modality.AUDIO, Modality.VIDEO}

    def _deferred_rescue(self, assets: Sequence[StoredAsset]) -> frozenset[Modality]:
        """Modalities `settle()` can still rescue for an embedder that cannot take them.

        `capture()` commits before any model runs, so it cannot see the transcript or the visual
        description that will exist by the time the record is embedded -- only which of them the
        configured composition will derive.
        """
        rescued = self._transcript_fallback(assets)
        if self._vision_describer is not None:
            rescued |= self._vision_capabilities
        return rescued

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
            raise ModelError(
                "speaker identity evidence exceeded the supported text length",
                reason="payload_too_large",
            )
        return replace(prepared, text=text)

    def _recognize_speech(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
        *,
        reversible: bool = False,
    ) -> None:
        if not isinstance(self._transcriber, SpeechBackend):
            raise ModelError(
                "configured transcription backend cannot analyze speakers",
                reason="backend_not_configured",
            )
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
        analyzed = {asset.asset_id for asset in missing}
        for asset in speech_assets:
            segments = operation.speech_segments[asset.asset_id]
            operation.transcripts[asset.asset_id] = "\n".join(segment.text for segment in segments)
            self._trace_identity_yield(
                "mindbridge.identity.speakers",
                tuple((segment.speaker_id, segment.identity_score) for segment in segments),
                cached=asset.asset_id not in analyzed,
            )

    def _trace_identity_yield(
        self,
        name: str,
        resolved: Sequence[tuple[str | None, float | None]],
        *,
        cached: bool,
    ) -> None:
        """Record one asset's recognizer yield under stable ``mindbridge.identity.*`` names.

        ``resolved`` pairs every observation's identity with its match score. All five attributes
        are emitted even for an empty asset: a recognizer whose detection threshold suits another
        domain runs, costs time, and yields nothing, which is otherwise indistinguishable from
        having configured no recognizer at all. ``cached`` separates re-read observations from a
        fresh analysis so that cheap-because-cached never reads as cheap-because-empty.

        ``observations`` and ``matched_existing`` count observations; ``identities`` and ``created``
        count distinct identities. The store reports ``identity_score`` only where an observation
        matched an existing identity by similarity, so its absence is the only available signal for
        a new identity. A cross-modal adoption, where one asset's lone face takes the identity of
        its lone voice, also arrives without a score and therefore counts as created here.
        """
        identities = {identity for identity, _ in resolved if identity is not None}
        created = {
            identity for identity, score in resolved if identity is not None and score is None
        }
        with self._trace(name, kind="stage") as span:
            span.set_attribute(IDENTITY_OBSERVATIONS, len(resolved))
            span.set_attribute(IDENTITY_IDENTITIES, len(identities))
            span.set_attribute(
                IDENTITY_MATCHED,
                sum(1 for _, score in resolved if score is not None),
            )
            span.set_attribute(IDENTITY_CREATED, len(created))
            span.set_attribute(IDENTITY_CACHED, cached)

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
            raise ModelError(
                "audio transcription exceeded the supported text length", reason="payload_too_large"
            )
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
                raise ModelError(
                    "failed to transcribe audio input", reason="model_failed"
                ) from error
            if len(generated) != len(refs) or any(
                not isinstance(transcript, str) for transcript in generated
            ):
                raise ModelError(
                    "transcription model returned invalid output", reason="response_invalid"
                )
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
            raise ModelError(
                "configured transcription backend cannot analyze speakers",
                reason="backend_not_configured",
            )
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
                raise ModelError("failed to analyze speech input", reason="model_failed") from error
            if len(analyses) != len(assets) or any(
                not isinstance(analysis, SpeechAnalysis) for analysis in analyses
            ):
                raise ModelError("speech model returned invalid output", reason="response_invalid")
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
        # The index carries a memory's full-text document on its `object_part == 0` row alone,
        # and part 0 is the aggregate key -- the one holding every asset, so the one an oversized
        # asset elides. Leaving the gap stores a memory with no lexical document at all, which is
        # a silent retrieval hole rather than the degradation this promises. Renumbering restores
        # the invariant; the part index orders a memory's keys and is not a handle on any input.
        renumbered: builtins.list[tuple[_PreparedMemory, int, ModelInput]] = []
        next_part: dict[str, int] = {}
        for memory, _dropped_part, model_input in kept:
            position = next_part.get(memory.memory_id, 0)
            next_part[memory.memory_id] = position + 1
            renumbered.append((memory, position, model_input))
        kept = renumbered
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

    def _describe_vision(self, images: Sequence[Blob]) -> tuple[str, ...]:
        if self._vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        if not images:
            raise ValidationError("vision description requires at least one image")
        with self._operation() as operation:
            prepared = tuple(self._prepare_content(image, operation) for image in images)
            return self._vision_descriptions(
                tuple(self._resolved_model_input(value) for value in prepared)
            )

    def _vision_descriptions(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        if self._vision_describer is None:
            raise ModelError(
                "vision description backend is not configured",
                reason="backend_not_configured",
            )
        inputs = tuple(inputs)
        unsupported = frozenset(
            modality
            for value in inputs
            for modality in value.modalities - self._vision_capabilities
        )
        if unsupported:
            names = ", ".join(sorted(modality.value for modality in unsupported))
            raise ModelError(
                f"configured vision model does not support: {names}",
                reason="unsupported_modality",
            )
        with self._model_trace(
            "vision",
            "vision.description",
            model=self._vision_model,
            batch_size=len(inputs),
            modalities=(modality for value in inputs for modality in value.modalities),
        ):
            mark_model_requests(1)
            try:
                descriptions = self._vision_describer.describe(inputs)
            except MindBridgeError:
                raise
            except Exception as error:
                raise ModelError(
                    "failed to describe vision input", reason="model_failed"
                ) from error
        if len(descriptions) != len(inputs) or any(
            not isinstance(description, str) or not description.strip()
            for description in descriptions
        ):
            raise ModelError("vision model returned invalid output", reason="response_invalid")
        normalized = tuple(description.strip() for description in descriptions)
        if any(len(description) > _MAX_TEXT_CHARACTERS for description in normalized):
            raise ModelError(
                "vision description exceeded the supported text length", reason="payload_too_large"
            )
        return normalized

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
                raise ModelError("failed to embed memory input", reason="model_failed") from error
            if len(vectors) != len(inputs):
                raise ModelError(
                    "embedding model returned the wrong number of vectors",
                    reason="response_invalid",
                )
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
                            raise ModelError(
                                "generation model returned an invalid answer chunk",
                                reason="response_invalid",
                            )
                        if part:
                            if not parts and current_model_request_count():
                                trace.get_current_span().set_attribute(
                                    MODEL_TTFT, perf_counter() - started
                                )
                            parts.append(part)
                    answer = "".join(parts)
                    if not answer.strip():
                        raise ModelError(
                            "generation model returned an invalid answer", reason="response_invalid"
                        )
                    if isinstance(used_hits, AnswerResult):
                        if used_hits.answer != answer:
                            raise ModelError(
                                "generation model returned an invalid answer",
                                reason="response_invalid",
                            )
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
                            raise ModelError(
                                "generation model returned invalid abstention status",
                                reason="response_invalid",
                            )
                    else:
                        raise ModelError(
                            "generation model returned invalid grounding hits",
                            reason="response_invalid",
                        )
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
                raise ModelError(
                    "failed to generate a grounded answer", reason="model_failed"
                ) from error
            if not isinstance(result, AnswerResult):
                raise ModelError(
                    "generation model returned an invalid answer", reason="response_invalid"
                )
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
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
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
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
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
            raise StorageError("failed to lease local media", reason="io_failed") from error
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
                            raise StorageError(
                                "orphaned media became referenced during cleanup",
                                reason="io_failed",
                            )
                remaining = index + 1
        except AssetStoreError as error:
            self._queue_asset_cleanup(assets[remaining:])
            raise StorageError("failed to clean up orphaned media", reason="io_failed") from error
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
            raise StorageError("failed to scan local media", reason="io_failed") from error
        with _translate_storage_errors("reconcile local media"):
            tracked_ids = {asset.asset_id for asset in self._store.read_assets(physical_ids)}
        for asset_id in physical_ids:
            if asset_id in tracked_ids:
                continue
            try:
                self._assets.delete_id(asset_id)
            except AssetStoreError as error:
                raise StorageError(
                    "failed to delete untracked local media", reason="io_failed"
                ) from error

    def _delete_orphan_assets(self, orphaned: Sequence[StoredAsset]) -> None:
        for asset in orphaned:
            try:
                self._assets.delete(asset)
            except AssetStoreError as error:
                raise StorageError("failed to delete orphaned media", reason="io_failed") from error
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
                raise StorageError(
                    "search-index outbox changed while it was being acknowledged",
                    reason="flush_failed",
                )

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
                            audio_transcript=_has_stream_transcript(
                                memory.content,
                                memory.assets,
                            ),
                            visual_description=_has_stream_description(
                                memory.content,
                                memory.assets,
                            ),
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
            self._vision_describer,
            self._face_analyzer,
            self._former,
            self._consolidator,
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
            self._vision_describer,
            self._face_analyzer,
            self._former,
            self._consolidator,
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
            raise StorageError("Memory is closed", reason="instance_unusable")

    @contextmanager
    def _operation(self) -> Iterator[_OperationAssets]:
        self._require_owner_process()
        with self._lifecycle:
            if self._closed or self._closing:
                raise StorageError("Memory is closed", reason="instance_unusable")
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
                "Memory cannot be used after fork; create a new instance with a different data_dir",
                reason="instance_unusable",
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
        vision_describer: VisionDescriptionBackend | None = None,
        face_analyzer: FaceBackend | None = None,
        former: FormationBackend | None = None,
        consolidator: ConsolidationBackend | None = None,
        index_speech: bool = _DEFAULT_CONFIG.index_speech,
        index_quantization: IndexQuantization = _DEFAULT_CONFIG.index_quantization,
        minimum_relevance: float = _DEFAULT_CONFIG.minimum_relevance,
        ambiguity_margin: float = _DEFAULT_CONFIG.ambiguity_margin,
        evidence_budget_chars: int | None = _DEFAULT_CONFIG.evidence_budget_chars,
        decay_half_life_days: float | None = _DEFAULT_CONFIG.decay_half_life_days,
        reinforce_on_answer: bool = _DEFAULT_CONFIG.reinforce_on_answer,
        speaker_similarity: float = _DEFAULT_CONFIG.speaker_similarity,
        speaker_margin: float = _DEFAULT_CONFIG.speaker_margin,
        face_similarity: float = _DEFAULT_CONFIG.face_similarity,
        face_margin: float = _DEFAULT_CONFIG.face_margin,
        identity_link_min_assets: int = _DEFAULT_CONFIG.identity_link_min_assets,
        tracer: Tracer | None = None,
    ) -> None:
        self._memory = Memory(
            data_dir=data_dir,
            embedder=embedder,
            answerer=answerer,
            transcriber=transcriber,
            vision_describer=vision_describer,
            face_analyzer=face_analyzer,
            former=former,
            consolidator=consolidator,
            index_speech=index_speech,
            index_quantization=index_quantization,
            minimum_relevance=minimum_relevance,
            ambiguity_margin=ambiguity_margin,
            evidence_budget_chars=evidence_budget_chars,
            decay_half_life_days=decay_half_life_days,
            reinforce_on_answer=reinforce_on_answer,
            speaker_similarity=speaker_similarity,
            speaker_margin=speaker_margin,
            face_similarity=face_similarity,
            face_margin=face_margin,
            identity_link_min_assets=identity_link_min_assets,
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
            vision_describer=plugins.vision_describer,
            face_analyzer=plugins.face_analyzer,
            former=plugins.former,
            consolidator=plugins.consolidator,
            index_speech=config.index_speech,
            index_quantization=config.index_quantization,
            minimum_relevance=config.minimum_relevance,
            ambiguity_margin=config.ambiguity_margin,
            evidence_budget_chars=config.evidence_budget_chars,
            decay_half_life_days=config.decay_half_life_days,
            reinforce_on_answer=config.reinforce_on_answer,
            speaker_similarity=config.speaker_similarity,
            speaker_margin=config.speaker_margin,
            face_similarity=config.face_similarity,
            face_margin=config.face_margin,
            identity_link_min_assets=config.identity_link_min_assets,
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
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.add,
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def _add_stream_input(self, item: StreamInput) -> MemoryRecord:
        return await asyncio.to_thread(self._memory._add_stream_input, item)

    async def _capture_stream_input(self, item: StreamInput) -> MemoryRecord:
        return await asyncio.to_thread(self._memory._capture_stream_input, item)

    async def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]:
        return await asyncio.to_thread(
            self._memory.add_many,
            contents,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.capture,
            content,
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )

    async def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self._memory.settle,
            limit=limit,
            max_attempts=max_attempts,
            memory_ids=memory_ids,
        )

    async def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        return await asyncio.to_thread(
            self._memory.pending_captures, limit=limit, memory_ids=memory_ids
        )

    async def add_stream(
        self,
        contents: AsyncIterable[ContentInput | StreamInput],
        *,
        capture: bool = False,
    ) -> AsyncIterator[MemoryRecord]:
        """Add an async omni stream one durable, searchable observation at a time.

        With `capture=True` each item commits through `capture()` and owes a later `settle()`.
        """
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
                    record = (
                        await self._capture_stream_input(content)
                        if capture
                        else await self._add_stream_input(content)
                    )
                elif capture:
                    record = await self.capture(content)
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
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        return await asyncio.to_thread(
            self._memory.search,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
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
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        return await asyncio.to_thread(
            self._memory.search_with_trace,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )

    async def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> AnswerResult:
        return await asyncio.to_thread(
            self._memory.ask,
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
        )

    async def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> ContextBundle:
        return await asyncio.to_thread(
            self._memory.compile,
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
        )

    @property
    def capabilities(self) -> MemoryCapabilities:
        """The composition's declared capabilities; read from memory, so no thread is needed."""
        return self._memory.capabilities

    async def get(self, memory_id: str) -> MemoryRecord:
        return await asyncio.to_thread(self._memory.get, memory_id)

    async def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        return await asyncio.to_thread(self._memory.speech, memory_id)

    async def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        return await asyncio.to_thread(self._memory.faces, memory_id)

    async def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            partial(
                self._memory.register_speaker,
                speaker_id,
                name,
                relationship=relationship,
            )
        )

    async def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            partial(
                self._memory.register_identity,
                identity_id,
                name,
                relationship=relationship,
            )
        )

    async def identity(self, identity_id: str) -> IdentityProfile | None:
        return await asyncio.to_thread(self._memory.identity, identity_id)

    async def unlink_identity(self, alias_id: str) -> str | None:
        return await asyncio.to_thread(self._memory.unlink_identity, alias_id)

    async def reinforce(self, memory_ids: Sequence[str]) -> int:
        return await asyncio.to_thread(self._memory.reinforce, memory_ids)

    async def consolidation_candidates(
        self,
        *,
        limit: int = 32,
    ) -> tuple[ConsolidationCandidate, ...]:
        return await asyncio.to_thread(partial(self._memory.consolidation_candidates, limit=limit))

    async def consolidate(
        self,
        *,
        evidence_ids: Sequence[str] | None = None,
        query: ContentInput | None = None,
        limit: int = 32,
        trigger: MemoryTrigger = MemoryTrigger.MANUAL,
    ) -> ConsolidationReport:
        return await asyncio.to_thread(
            partial(
                self._memory.consolidate,
                evidence_ids=evidence_ids,
                query=query,
                limit=limit,
                trigger=trigger,
            )
        )

    async def forget(self, memory_ids: Sequence[str]) -> MemoryOperationRecord | None:
        return await asyncio.to_thread(self._memory.forget, memory_ids)

    async def rollback(self, operation_id: int) -> bool:
        return await asyncio.to_thread(self._memory.rollback, operation_id)

    async def operations(self, *, limit: int = 100) -> tuple[MemoryOperationRecord, ...]:
        return await asyncio.to_thread(partial(self._memory.operations, limit=limit))

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


class AsyncCaptureStream:
    """Reduce associated capture streams into retrieval and durable memories.

    `capture=True` commits each `FINAL` through `Memory.capture()` instead of `Memory.add()`, so
    the acknowledgement leaves the model path and every `StreamCommit` reports
    `pending_settlement`. The host then owes `settle()`; the default stays the strong `add()`.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        _limit(limit, maximum=100)
        if not isinstance(capture, bool):
            raise ValidationError("capture must be a boolean")
        self._memory = memory
        self._limit = limit
        self._memory_type = _optional_memory_type(memory_type)
        self._reference_at = _reference_at(reference_at)
        self._max_streams = _positive_dimension(max_streams, "max_streams")
        self._capture = capture

    async def consume(  # noqa: C901 - the three-state reducer is intentionally inline
        self,
        events: AsyncIterable[StreamEvent],
    ) -> AsyncIterator[StreamCommit]:
        """Yield only final observations; partials, cancellation, and EOF never write."""
        if not isinstance(events, AsyncIterable):
            raise ValidationError("events must be an async iterable of StreamEvent values")
        prefetches: dict[str, AsyncOmniPrefetch] = {}
        try:
            async for event in events:
                if not isinstance(event, StreamEvent):
                    raise ValidationError("events must contain StreamEvent values")
                stream_id = event.stream_id
                if event.phase is StreamPhase.UPDATE:
                    prefetch = self._prefetch_for(prefetches, stream_id)
                    assert event.item is not None and not isinstance(event.item, StreamInput)
                    prefetch.submit(event.item)
                    continue
                if event.phase is StreamPhase.CANCEL:
                    cancelled_prefetch = (
                        prefetches.pop(stream_id) if stream_id in prefetches else None
                    )
                    if cancelled_prefetch is not None:
                        await cancelled_prefetch.close()
                    continue
                assert event.item is not None
                prefetch = self._prefetch_for(prefetches, stream_id)
                item = event.item
                final_content = self._final_query(item)
                retrieval_error: Exception | None = None
                retrieval: PrefetchResult | None = None
                try:
                    retrieval = await prefetch.finalize(final_content)
                except Exception as error:
                    retrieval_error = error
                    # finalize() drains its own worker only once it has started closing, so an
                    # early rejection would otherwise abandon a search already in flight.
                    await prefetch.close()
                prefetches.pop(stream_id, None)
                # ponytail: FINAL is the commit point; a cancellable storage transaction would
                # be needed to revoke it safely once the worker thread has started.
                commit = asyncio.create_task(self._add_final(item))
                cancelled = False
                while not commit.done():
                    try:
                        await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        if cancelled:
                            raise asyncio.CancelledError from None
                        raise
                try:
                    record = commit.result()
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
                if cancelled:
                    raise asyncio.CancelledError
                if retrieval_error is not None:
                    yield StreamCommit(
                        record=record,
                        prefetch=None,
                        retrieval_error=(
                            retrieval_error.code
                            if isinstance(retrieval_error, MindBridgeError)
                            else "retrieval_failed"
                        ),
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
                else:
                    assert retrieval is not None
                    yield StreamCommit(
                        record=record,
                        prefetch=retrieval,
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
        finally:
            for prefetch in tuple(prefetches.values()):
                await prefetch.close()

    def _prefetch_for(
        self,
        prefetches: dict[str, AsyncOmniPrefetch],
        stream_id: str,
    ) -> AsyncOmniPrefetch:
        prefetch = prefetches.get(stream_id)
        if prefetch is not None:
            return prefetch
        if len(prefetches) >= self._max_streams:
            raise ValidationError("capture stream exceeds max_streams")
        prefetch = self._new_prefetch()
        prefetches[stream_id] = prefetch
        return prefetch

    def _final_query(self, item: ContentInput | StreamInput) -> ContentInput:
        if not isinstance(item, StreamInput) or (
            item.transcript is None and item.description is None
        ):
            return item.content if isinstance(item, StreamInput) else item
        capabilities = self._memory.capabilities.embedding
        if Modality.TEXT not in capabilities:
            return item.content
        routed: builtins.list[ContentAtom] = [
            value for value in (item.transcript, item.description) if value is not None
        ]
        for atom in _content_atoms(item.content):
            modality = _declared_atom_modality(atom)
            if (
                modality is Modality.AUDIO
                and item.transcript is not None
                and modality not in capabilities
            ):
                continue
            if (
                modality in {Modality.IMAGE, Modality.VIDEO}
                and item.description is not None
                and modality not in capabilities
            ):
                continue
            routed.append(atom)
        return routed[0] if len(routed) == 1 else tuple(routed)

    def _new_prefetch(self) -> AsyncOmniPrefetch:
        return AsyncOmniPrefetch(
            self._memory,
            limit=self._limit,
            memory_type=self._memory_type,
            reference_at=self._reference_at,
        )

    async def _add_final(self, item: ContentInput | StreamInput) -> MemoryRecord:
        if isinstance(item, StreamInput):
            if self._capture:
                return await self._memory._capture_stream_input(item)
            return await self._memory._add_stream_input(item)
        if self._capture:
            return await self._memory.capture(item)
        return await self._memory.add(item)


StreamContext = ObservationContext | Callable[[], ObservationContext | None] | None
"""Provenance for streamed observations: one fixed value, or a sampler read at each boundary."""


def _stream_context(context: StreamContext) -> Callable[[], ObservationContext | None]:
    """Normalize a fixed context or a per-observation sampler into one sampler.

    A capture stream outlives the observations it commits, so a robot's pose is not a property
    of the stream. A callable is read once per closed observation, which is the only moment at
    which the adapter knows which interval the pose belongs to. A fixed `ObservationContext`
    stays accepted because a static camera really does have one.
    """
    if context is None:
        return lambda: None
    if isinstance(context, ObservationContext):
        return lambda: context
    if not callable(context):
        raise ValidationError("context must be an ObservationContext or a callable returning one")
    return context


class AsyncAudioStream:
    """Normalize PCM, VAD, ASR, and acoustic boundaries into associated capture events."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._memory = memory
        self._max_streams = _positive_dimension(max_streams, "max_streams")
        self._written_type = _optional_memory_type(memory_type) or MemoryType.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory._memory._embedding_capabilities
        self._native_audio = Modality.AUDIO in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._max_pcm_bytes = memory._memory._assets.max_bytes - 44
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    async def consume(
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each VAD or acoustic end boundary."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of audio stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _AudioStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (PCMChunk, VADPacket, ASRPartial, AcousticBoundary)):
                raise ValidationError("packets contain an invalid audio stream value")
            stream_id = packet.stream_id
            if isinstance(packet, AcousticBoundary):
                if packet.boundary is AudioBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is AudioBoundary.END:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.pcm or current.transcript):
                    raise ValidationError("audio stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                continue
            if isinstance(packet, VADPacket):
                if not packet.active:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                else:
                    self._state_for(states, stream_id, packet.occurred_at)
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, ASRPartial):
                state.transcript = packet.text
            else:
                self._append_pcm(state, packet)
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _AudioStreamState:
        state = states.get(stream_id)
        if state is not None:
            if state.occurred_at is None:
                state.occurred_at = occurred_at
            return state
        if len(states) >= self._max_streams:
            raise ValidationError("audio stream exceeds max_streams")
        state = _AudioStreamState(bytearray(), occurred_at=occurred_at)
        states[stream_id] = state
        return state

    def _append_pcm(self, state: _AudioStreamState, packet: PCMChunk) -> None:
        sample_format = (
            packet.sample_rate_hz,
            packet.channels,
            packet.sample_width_bytes,
        )
        current_format = (
            state.sample_rate_hz,
            state.channels,
            state.sample_width_bytes,
        )
        if state.sample_rate_hz is None:
            state.sample_rate_hz, state.channels, state.sample_width_bytes = sample_format
        elif current_format != sample_format:
            raise ValidationError("PCM format changed before an audio boundary")
        if len(state.pcm) + len(packet.data) > self._max_pcm_bytes:
            raise ValidationError("PCM stream exceeds the local asset size limit")
        state.pcm.extend(packet.data)

    def _query(self, state: _AudioStreamState) -> ContentInput | None:
        if self._native_audio and state.pcm:
            audio = _pcm_blob(state)
            if self._native_text and state.transcript:
                return (state.transcript, audio)
            return audio
        return state.transcript or None

    def _finish(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_at, occurred_end = _audio_interval(state, occurred_end)
        # Sampled here, at the boundary, so the pose belongs to the interval being committed.
        context = self._context()
        if state.pcm:
            item = StreamInput(
                _pcm_blob(state),
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                transcript=state.transcript or None,
            )
        elif state.transcript:
            item = StreamInput(
                state.transcript,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


class AsyncVisionStream:
    """Normalize image frames, visual descriptions, and scene boundaries."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._memory = memory
        self._max_streams = _positive_dimension(max_streams, "max_streams")
        self._written_type = _optional_memory_type(memory_type) or MemoryType.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory._memory._embedding_capabilities
        self._native_image = Modality.IMAGE in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    async def consume(
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each completed visual scene."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of vision stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _VisionStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (VisionFrame, VisionPartial, SceneBoundary)):
                raise ValidationError("packets contain an invalid vision stream value")
            stream_id = packet.stream_id
            if isinstance(packet, SceneBoundary):
                if packet.boundary is VisionBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is VisionBoundary.END:
                    final = await self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.image is not None or current.description):
                    raise ValidationError("vision stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                    current.last_occurred_at = packet.occurred_at
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, VisionPartial):
                state.description = packet.text
            else:
                # ponytail: retain one keyframe per scene; add a bounded sampler only when
                # multi-frame retrieval quality demonstrates that the latest frame is insufficient.
                state.image = packet.image
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _VisionStreamState:
        state = states.get(stream_id)
        if state is None:
            if len(states) >= self._max_streams:
                raise ValidationError("vision stream exceeds max_streams")
            state = _VisionStreamState(occurred_at=occurred_at)
            states[stream_id] = state
        elif state.occurred_at is None:
            state.occurred_at = occurred_at
        if occurred_at is not None:
            state.last_occurred_at = occurred_at
        return state

    def _query(self, state: _VisionStreamState) -> ContentInput | None:
        if self._native_image and state.image is not None:
            if self._native_text and state.description:
                return (state.description, state.image)
            return state.image
        return state.description or None

    async def _finish(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_end = occurred_end or state.last_occurred_at
        if (
            state.image is not None
            and not state.description
            and not self._native_image
            and self._native_text
            and self._memory._memory._vision_describer is not None
        ):
            state.description = (
                await asyncio.to_thread(self._memory._memory._describe_vision, (state.image,))
            )[0]
        # Sampled here, at the boundary, so the pose belongs to the scene being committed.
        context = self._context()
        if state.image is not None:
            item = StreamInput(
                state.image,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                description=state.description or None,
            )
        elif state.description:
            item = StreamInput(
                state.description,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


def _pcm_blob(state: _AudioStreamState) -> Blob:
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(state.channels)
        audio.setsampwidth(state.sample_width_bytes)
        audio.setframerate(state.sample_rate_hz)
        audio.writeframes(state.pcm)
    return Blob(output.getvalue(), "audio/wav", "capture.wav")


def _audio_interval(
    state: _AudioStreamState,
    occurred_end: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    occurred_at = state.occurred_at
    if not state.pcm:
        return occurred_at, occurred_end
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    frames = len(state.pcm) // (state.channels * state.sample_width_bytes)
    duration = timedelta(seconds=frames / state.sample_rate_hz)
    if occurred_at is None and occurred_end is not None:
        occurred_at = occurred_end - duration
    elif occurred_at is not None and occurred_end is None:
        occurred_end = occurred_at + duration
    return occurred_at, occurred_end


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


def _declared_atom_modality(atom: ContentAtom) -> Modality | None:
    if isinstance(atom, str):
        return Modality.TEXT
    if isinstance(atom, Blob):
        return _media_hint(atom.name, atom.media_type)[0]
    if isinstance(atom, Path):
        return _media_hint(atom.name, None)[0]
    if atom.modality is not None:
        return atom.modality
    if atom.media_type is not None:
        return _media_hint(atom.name, atom.media_type)[0]
    return None


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
    context: ObservationContext | None = None,
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
    if context is not None:
        if not isinstance(context, ObservationContext):
            raise ValidationError("context must be an ObservationContext")
        identity["context"] = _observation_context_identity(context)
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
        context=context,
    )


def _prepared_from_stored(memory: StoredMemory) -> _PreparedMemory:
    """Rebuild the prepared form of a captured row so one path enriches add and settle alike."""
    return _PreparedMemory(
        memory_id=memory.memory_id,
        content=_PreparedContent(
            text=memory.content,
            assets=memory.assets,
            modality=Modality(memory.modality),
            canonical_parts=(("text", memory.content),) if memory.content else (),
            audio_transcript=_has_stream_transcript(memory.content, memory.assets),
            visual_description=_has_stream_description(memory.content, memory.assets),
        ),
        metadata_json=memory.metadata_json,
        occurred_at=memory.occurred_at,
        occurred_end=memory.occurred_end,
        memory_type=MemoryType(memory.memory_type),
    )


def _observation_context_identity(context: ObservationContext) -> dict[str, object]:
    spatial = context.spatial
    value: dict[str, object] = {
        "basis": context.basis.value,
        "source_id": context.source_id,
        "confidence": context.confidence,
        "valid_from": (None if context.valid_from is None else _datetime_text(context.valid_from)),
        "valid_until": (
            None if context.valid_until is None else _datetime_text(context.valid_until)
        ),
    }
    if spatial is not None:
        value["spatial"] = {
            "frame_id": spatial.frame_id,
            "anchor": spatial.anchor.value,
            "position_m": (spatial.x, spatial.y, spatial.z),
            "orientation_xyzw": spatial.orientation_xyzw,
            "position_uncertainty_m": spatial.position_uncertainty_m,
        }
    return value


def _stored_memory_context(
    memory: _PreparedMemory,
    *,
    recorded_at: datetime,
) -> MemoryContext | None:
    context = memory.context
    if context is None or isinstance(context, MemoryContext):
        return context
    return MemoryContext(
        kind=MemoryKind.OBSERVATION,
        basis=context.basis,
        confidence=context.confidence,
        valid_from=context.valid_from,
        valid_until=context.valid_until,
        recorded_at=recorded_at,
        lineage_id=memory.memory_id,
        source_id=context.source_id,
        spatial=context.spatial,
    )


def _observation_from_record(record: MemoryRecord) -> ObservationContext:
    context = record.context
    if context is None:
        return ObservationContext()
    return ObservationContext(
        basis=context.basis,
        source_id=context.source_id,
        confidence=context.confidence,
        valid_from=context.valid_from,
        valid_until=context.valid_until,
        spatial=context.spatial,
    )


def _formation_memory_type(kind: MemoryKind) -> MemoryType:
    if kind in {MemoryKind.EVENT, MemoryKind.AFFECT}:
        return MemoryType.EPISODIC
    if kind is MemoryKind.RESPONSE_POLICY:
        return MemoryType.PROCEDURAL
    return MemoryType.SEMANTIC


def _validate_formation_proposal(
    proposal: FormationProposal,
    source: FormationInput,
) -> None:
    if proposal.kind is MemoryKind.AFFECT and (
        proposal.cue_modality is None or proposal.cue_modality not in source.content.modalities
    ):
        raise ModelError(
            "affect formation must name a modality present in its source",
            reason="response_invalid",
            stage="form",
        )
    if proposal.spatial is not None:
        observed = source.context.spatial
        if (
            observed is None
            or proposal.spatial.frame_id != observed.frame_id
            or proposal.spatial.anchor is not observed.anchor
        ):
            raise ModelError(
                "spatial formation must use the source observation frame and anchor",
                reason="response_invalid",
                stage="form",
            )


class _RejectedOperation(Exception):
    """Internal signal that kernel policy refused one proposed operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_derived(memories: Mapping[str, StoredMemory], memory_id: str) -> bool:
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return context is not None and context.kind is not MemoryKind.OBSERVATION


def _consolidation_primary(
    proposal: FormationProposal,
    sources: Sequence[MemoryRecord],
) -> MemoryRecord | None:
    """Return the newest cited source the proposal is grounded in, or `None`.

    Formation binds one proposal to one source. A consolidation cites several, so the kernel
    requires the AFFECT cue modality and the spatial frame to match at least one of them and
    inherits validity from the newest such source, which keeps the derived ID stable when the
    same evidence set is proposed again.
    """
    ranked = sorted(
        sources,
        key=lambda source: (
            source.occurred_end or source.occurred_at or source.created_at,
            source.id,
        ),
    )
    for source in reversed(ranked):
        try:
            _validate_formation_proposal(
                proposal,
                FormationInput(
                    memory_id=source.id,
                    content=ModelInput(text=source.content, assets=source.assets),
                    context=_observation_from_record(source),
                ),
            )
        except ModelError:
            continue
        return source
    return None


def _retiring_targets(operation: MemoryOperation) -> set[str]:
    """IDs this operation would take out of ordinary recall or retire the current version of.

    A `REINFORCE` target is strengthened rather than retired, so it is not one of these; a
    `CONSOLIDATE` target is consolidation forgetting and is.
    """
    if operation.intent is MemoryIntent.REINFORCE:
        return set()
    return set(operation.target_ids)


def _operation_record(logged: StoredOperation) -> MemoryOperationRecord:
    try:
        trigger = MemoryTrigger(logged.trigger)
    except ValueError:
        raise StorageError("a logged memory operation has an unknown trigger") from None
    return MemoryOperationRecord(
        operation_id=logged.operation_id,
        operation=load_operation(logged.operation_json),
        trigger=trigger,
        applied_at=logged.applied_at,
        model_id=logged.model_id,
        recipe=logged.recipe,
        created_ids=logged.created_ids,
        changed_ids=logged.changed_ids,
        forgotten_ids=logged.forgotten_ids,
        rolled_back_at=logged.rolled_back_at,
    )


def _validate_formation_pairs(
    pairs: Sequence[tuple[_PreparedMemory, str, float]],
) -> None:
    seen: dict[tuple[str, str], _PreparedMemory] = {}
    states: dict[tuple[str, str], builtins.list[MemoryContext]] = {}
    for prepared, source_id, _confidence in pairs:
        key = (prepared.memory_id, source_id)
        prior = seen.setdefault(key, prepared)
        if prior != prepared:
            raise ModelError(
                "formation returned conflicting duplicates for one source",
                reason="response_invalid",
                stage="form",
            )
        context = prepared.context
        if isinstance(context, MemoryContext) and context.kind is MemoryKind.STATE:
            states.setdefault((source_id, context.lineage_id or prepared.memory_id), []).append(
                context
            )
    for values in states.values():
        for index, left in enumerate(values):
            if any(
                left.value != right.value
                and _valid_intervals_overlap(
                    left.valid_from,
                    left.valid_until,
                    right.valid_from,
                    right.valid_until,
                )
                for right in values[index + 1 :]
            ):
                raise ModelError(
                    "formation returned conflicting states for one source",
                    reason="response_invalid",
                    stage="form",
                )


def _valid_intervals_overlap(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> bool:
    return not (
        (left_until is not None and right_from is not None and left_until <= right_from)
        or (right_until is not None and left_from is not None and right_until <= left_from)
    )


def _formation_context(
    source: MemoryRecord,
    proposal: FormationProposal,
    *,
    model_id: str,
    recipe: str,
    recorded_at: datetime,
) -> MemoryContext:
    source_context = _observation_from_record(source)
    valid_from = proposal.valid_from or source_context.valid_from or source.occurred_at
    if valid_from is None and proposal.kind in {MemoryKind.STATE, MemoryKind.TRAIT}:
        valid_from = recorded_at
    valid_until = proposal.valid_until or source_context.valid_until
    spatial = proposal.spatial or source_context.spatial
    return MemoryContext(
        kind=proposal.kind,
        basis=proposal.basis,
        confidence=proposal.confidence,
        valid_from=valid_from,
        valid_until=valid_until,
        recorded_at=recorded_at,
        lineage_id=_formation_lineage_id(proposal, spatial=spatial),
        source_id=source_context.source_id,
        subject=proposal.subject,
        predicate=proposal.predicate,
        value=proposal.value,
        evidence_ids=(source.id,),
        model_id=model_id,
        recipe=recipe,
        spatial=spatial,
        cue_modality=proposal.cue_modality,
        valence=proposal.valence,
        arousal=proposal.arousal,
    )


def _formation_lineage_id(
    proposal: FormationProposal,
    *,
    spatial: object = None,
) -> str:
    frame_id = getattr(spatial, "frame_id", None)
    anchor = getattr(getattr(spatial, "anchor", None), "value", None)
    payload = json.dumps(
        {
            "kind": proposal.kind.value,
            "subject": _semantic_text(proposal.subject),
            "predicate": _semantic_text(proposal.predicate),
            "frame_id": frame_id,
            "anchor": anchor,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"mindbridge-lineage-v1:{payload}".encode()).hexdigest()


def _formation_memory_id(
    source_id: str,
    proposal: FormationProposal,
    *,
    recipe: str,
    context: MemoryContext,
) -> str:
    episode_source = (
        source_id
        if (
            proposal.kind in {MemoryKind.EVENT, MemoryKind.AFFECT, MemoryKind.STATE}
            or (
                proposal.kind is MemoryKind.TRAIT and proposal.basis is EvidenceBasis.USER_STATEMENT
            )
        )
        else None
    )
    spatial = context.spatial
    payload = json.dumps(
        {
            "recipe": recipe,
            "kind": proposal.kind.value,
            "subject": _semantic_text(proposal.subject),
            "predicate": _semantic_text(proposal.predicate),
            "value": _semantic_text(proposal.value),
            "assertion_basis": (
                proposal.basis.value
                if proposal.basis is not EvidenceBasis.MODEL_INFERENCE
                else None
            ),
            "cue_modality": (
                None if proposal.cue_modality is None else proposal.cue_modality.value
            ),
            "episode_source": episode_source,
            "content": proposal.content if proposal.kind is MemoryKind.EVENT else None,
            "valid_from": (
                _datetime_text(context.valid_from)
                if episode_source is not None and context.valid_from is not None
                else None
            ),
            "valid_until": (
                _datetime_text(context.valid_until)
                if episode_source is not None and context.valid_until is not None
                else None
            ),
            "spatial": (
                None
                if spatial is None
                else {
                    "frame_id": spatial.frame_id,
                    "anchor": spatial.anchor.value,
                    "position_m": (spatial.x, spatial.y, spatial.z),
                    "orientation_xyzw": spatial.orientation_xyzw,
                    "position_uncertainty_m": spatial.position_uncertainty_m,
                }
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"mindbridge-formation-v1:{payload}".encode()).hexdigest()


def _semantic_text(value: str | None) -> str | None:
    return None if value is None else unicodedata.normalize("NFKC", value).casefold().strip()


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
    *,
    rescuable: frozenset[Modality] = frozenset(),
) -> frozenset[Modality]:
    unsupported = _prepared_modalities(prepared) - supported
    fallback = {Modality.AUDIO}
    if prepared.visual_description:
        fallback.update((Modality.IMAGE, Modality.VIDEO))
    # This runs before any derived text exists, so it cannot be seen yet -- only its possibility.
    # `rescuable` is the modality set a transcript or a visual description could still rescue,
    # which is why a video reaching a video-less embedder is no longer fatal when the transcriber
    # can read it. Empty by default, so a caller that does not pass it keeps the strict behaviour.
    fallback.update(rescuable)
    fatal = set(unsupported - fallback)
    if Modality.TEXT not in supported:
        fatal.update(unsupported & fallback)
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


def _with_stream_transcript(
    prepared: _PreparedContent,
    transcript: str,
) -> _PreparedContent:
    audio = tuple(asset for asset in prepared.assets if asset.modality == Modality.AUDIO.value)
    if len(audio) != 1:
        raise ValidationError("stream transcript requires exactly one audio asset")
    text = _text(transcript, "stream transcript")
    section = f"[transcript:{audio[0].asset_id}]\n{text}"
    content = (
        prepared.text
        if section in prepared.text
        else "\n\n".join(value for value in (prepared.text, section) if value)
    )
    if len(content) > _MAX_TEXT_CHARACTERS:
        raise ValidationError(f"content text must not exceed {_MAX_TEXT_CHARACTERS} characters")
    return replace(
        prepared,
        text=content,
        canonical_parts=(*prepared.canonical_parts, ("audio_transcript", section)),
        audio_transcript=True,
    )


def _with_stream_description(
    prepared: _PreparedContent,
    description: str,
) -> _PreparedContent:
    visual = tuple(
        asset
        for asset in prepared.assets
        if asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
    )
    if len(visual) != 1:
        raise ValidationError("stream description requires exactly one visual asset")
    text = _text(description, "stream description")
    section = f"[visual description:{visual[0].asset_id}]\n{text}"
    content = (
        prepared.text
        if section in prepared.text
        else "\n\n".join(value for value in (prepared.text, section) if value)
    )
    if len(content) > _MAX_TEXT_CHARACTERS:
        raise ValidationError(f"content text must not exceed {_MAX_TEXT_CHARACTERS} characters")
    return replace(
        prepared,
        text=content,
        canonical_parts=(*prepared.canonical_parts, ("visual_description", section)),
        visual_description=True,
    )


def _has_stream_transcript(text: str, assets: Sequence[StoredAsset]) -> bool:
    return any(
        asset.modality == Modality.AUDIO.value and f"[transcript:{asset.asset_id}]\n" in text
        for asset in assets
    )


def _has_stream_description(text: str, assets: Sequence[StoredAsset]) -> bool:
    return any(
        asset.modality in {Modality.IMAGE.value, Modality.VIDEO.value}
        and f"[visual description:{asset.asset_id}]\n" in text
        for asset in assets
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
    """Project stored speech evidence into the prose the derived index and embedding should carry.

    Stored content keeps the JSON, because the answering model reads it as structured evidence and
    needs the timings and scores. The derived projections do not: as JSON, `start_ms`, `end_ms`,
    `speaker_id` and `identity_score` all became BM25 tokens surrounding the words someone
    actually said, and the embedder encoded the schema along with the speech. Index content is the
    only lever that moves the R@20 ceiling, so the projection carries the utterance and who said
    it and nothing else.

    An unstable per-run `identity_*` id is still replaced by a stable `speaker_N` alias, which is
    what this function existed for: without it a recognizer that re-minted a person rewrote every
    document that mentioned them. A named speaker is projected under their name, which is also the
    token a caller would search for.
    """
    try:
        evidence = json.loads(payload)
        if not isinstance(evidence, dict) or evidence.get("asset_id") != asset_id:
            return None
        segments = evidence["segments"]
        if not isinstance(segments, list):
            return None
        aliases: dict[str, str] = {}
        lines = []
        for segment in segments:
            if not isinstance(segment, dict) or "speaker_id" not in segment:
                return None
            speaker_id = segment.get("speaker_id")
            if isinstance(speaker_id, str) and speaker_id.startswith("identity_"):
                speaker_id = aliases.setdefault(speaker_id, f"speaker_{len(aliases) + 1}")
            name = segment.get("speaker_name")
            speaker = name if isinstance(name, str) and name.strip() else speaker_id
            spoken = segment.get("text")
            said = spoken.strip() if isinstance(spoken, str) else ""
            if not isinstance(speaker, str) or not speaker.strip():
                if said:
                    lines.append(said)
                continue
            lines.append(f"{speaker.strip()}: {said}" if said else speaker.strip())
        if not lines:
            # Nothing was said, so there is no prose to carry. Leaving the payload alone keeps an
            # empty analysis byte-identical to what it was before this projection existed.
            return payload
        return "\n".join(lines)
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


def _vision_contract(
    describer: VisionDescriptionBackend | None,
) -> tuple[frozenset[Modality], str]:
    if describer is None:
        return frozenset(), "none"
    capabilities = _modalities(describer.vision_capabilities, "vision")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("vision capabilities must contain image or video")
    return (
        capabilities,
        _model_text(describer.vision_model, "vision model"),
    )


def _formation_contract(
    former: FormationBackend | None,
) -> tuple[frozenset[Modality], str, str]:
    if former is None:
        return frozenset(), "none", "none"
    capabilities = _modalities(former.formation_capabilities, "formation")
    if not capabilities:
        raise ValidationError("formation capabilities must not be empty")
    return (
        capabilities,
        _model_text(former.formation_model, "formation model"),
        _model_text(former.formation_space, "formation space"),
    )


def _consolidation_contract(
    consolidator: ConsolidationBackend | None,
) -> tuple[str, str]:
    if consolidator is None:
        return "none", "none"
    return (
        _model_text(consolidator.consolidation_model, "consolidation model"),
        _model_text(consolidator.consolidation_recipe, "consolidation recipe"),
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


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
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
        raise StorageError("stored memory metadata is invalid", reason="unexpected") from error
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise StorageError("stored memory metadata is not an object", reason="unexpected")
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


def _modality_names(modalities: Iterable[Modality]) -> str:
    return ", ".join(sorted(modality.value for modality in modalities)) or "nothing"


def _scope_description(scope: RetrievalScope | None) -> str | None:
    """Describe a requested scope that matched nothing, or `None` when none was requested."""
    if scope is None:
        return None
    bounds = []
    if scope.place_id is not None:
        bounds.append(f"place {scope.place_id}")
    if scope.near is not None:
        radius = "" if scope.radius_m is None else f" within {scope.radius_m} m"
        bounds.append(f"frame {scope.near.frame_id}{radius}")
    if scope.valid_at is not None:
        bounds.append(f"valid at {scope.valid_at.isoformat()}")
    if scope.known_at is not None:
        bounds.append(f"known at {scope.known_at.isoformat()}")
    if not bounds:
        return None
    return f"no memory matched the requested scope: {', '.join(bounds)}"


def declared_capabilities(
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    consolidator: ConsolidationBackend | None = None,
) -> MemoryCapabilities:
    """Declare what one set of backends can do, without opening a store.

    `Memory.capabilities` reads it, and so does `mindbridge doctor`, which probes recipes without
    owning a `data_dir`. Both therefore publish one document rather than two descriptions.
    """
    embedding, embedding_model, space_id, dimension = _embedding_contract(embedder)
    transcription, transcription_space = _transcription_contract(transcriber)
    vision, vision_model = _vision_contract(vision_describer)
    face, face_model, _face_space, _face_analysis_space = _face_contract(face_analyzer)
    formation, formation_model, _formation_space = _formation_contract(former)
    consolidation_model, _consolidation_recipe = _consolidation_contract(consolidator)
    return MemoryCapabilities(
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_space=space_id,
        embedding_dimension=dimension,
        generation=_generation_contract(answerer),
        transcription=transcription,
        vision=vision,
        face=face,
        formation=formation,
        # The contracts substitute a `"none"` sentinel for an absent backend because the space
        # and recipe digests hash these strings. That sentinel is an implementation detail, so
        # the published value is absent when the backend is, keyed on the backend itself rather
        # than on the string -- a real model could be named "none".
        generation_model=getattr(answerer, "generation_model", None),
        transcription_space=None if transcriber is None else transcription_space,
        vision_model=None if vision_describer is None else vision_model,
        face_model=None if face_analyzer is None else face_model,
        formation_model=None if former is None else formation_model,
        consolidation_model=None if consolidator is None else consolidation_model,
        speaker_recognition=isinstance(transcriber, SpeechBackend),
        streaming_generation=isinstance(answerer, StreamingGenerationBackend),
    )


def _retrieval_scope(value: RetrievalScope | None) -> RetrievalScope | None:
    if value is not None and not isinstance(value, RetrievalScope):
        raise ValidationError("scope must be a RetrievalScope")
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
        _LEGACY_INDEX_RECIPES
        | _REINDEXABLE_INDEX_RECIPES
        | {_index_recipe(mode) for mode in IndexQuantization}
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
    used = sum(evidence_cost(hit) for hit in selected)
    # No near-duplicate suppression here, and that is a measured decision rather than an
    # omission. Character-trigram Jaccard -- the measure VoiceMem uses for this at 0.30 -- cannot
    # separate a restatement from a log entry, because in this domain the bands overlap outright:
    # rephrasings of one fact score 0.635-0.836, while "dad took his medication at 8am" against
    # "...at 9am" scores 0.806, "the bicycle was in the shed on monday" against "...tuesday"
    # 0.737, and a temperature reading against the next day's 0.860. Five such pairs were tested
    # and all five fell inside the restatement band, so no threshold exists that keeps them.
    # Collapsing a medication time or a temperature series is information loss a companion memory
    # cannot afford, and it is strictly worse than the wasted budget it would save. A working
    # version needs a measure that reads the *differing* span rather than the shared template.
    extra: list[SearchHit] = []
    for hit in hits:
        if hit.id in taken:
            continue
        cost = evidence_cost(hit)
        if used + cost > budget_chars:
            break
        used += cost
        extra.append(hit)
    return tuple(extra)


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


def _parent_index_ids(documents: Sequence[IndexCandidate]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, builtins.list[str]] = {}
    for document in documents:
        grouped.setdefault(document.memory_id, []).append(document.embedding_id)
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
        # `lexical_relevance` is the ranking score contribution, which needs the term coverage
        # computed from content this candidate never had hydrated. Reporting the index-side
        # strength in its place would make the two figures incomparable across dispositions --
        # a rejected candidate would appear to carry more lexical evidence than a ranked one --
        # so the unknown component stays `None` and `lexical_match` carries what is known.
        lexical_match=lexical_match,
        # An upper bound on what the gate would have scored, not a value the gate produced: this
        # candidate was rejected before ranking, so the full-coverage test that decides whether
        # the lexical half counts at all, the temporal factor, and the observation's own
        # confidence were never applied. All three can only lower it.
        gate_relevance=max(
            dense_relevance.get(memory_id, 0.0),
            _LEXICAL_FULL_COVERAGE_RELEVANCE * lexical_index_relevance.get(memory_id, 0.0),
        ),
        rejected_by=rejected_by,
    )


def _extend_hydration_traces(
    target: builtins.list[RetrievalCandidateTrace] | None,
    candidates: _IndexCandidates,
    index_ids: Sequence[str],
    hydrated_documents: Sequence[IndexCandidate],
    accepted_documents: Sequence[IndexCandidate],
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
    hydrated_ids = {document.embedding_id for document in hydrated_documents}
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
                    else _LEXICAL_FULL_COVERAGE_RELEVANCE * lexical_hit.relevance
                ),
                lexical_match=lexical_hit is not None,
                gate_relevance=max(
                    0.0 if dense_hit is None else dense_hit.relevance,
                    0.0
                    if lexical_hit is None
                    else _LEXICAL_FULL_COVERAGE_RELEVANCE * lexical_hit.relevance,
                ),
                rejected_by=RetrievalRejection.STALE_INDEX,
            )
        )
    accepted_parent_ids = {document.memory_id for document in accepted_documents}
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
    gate_relevance: float,
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
        gate_relevance=gate_relevance,
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
    documents: Sequence[IndexCandidate],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], set[str]]:
    dense_by_id = {hit.id: hit for hit in candidates.dense}
    lexical_by_id = {hit.id: hit for hit in candidates.lexical}
    dense_relevance: dict[str, float] = {}
    dense_confidence: dict[str, float] = {}
    lexical_relevance: dict[str, float] = {}
    lexical_matches: set[str] = set()
    for document in documents:
        memory_id = document.memory_id
        embedding_id = document.embedding_id
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
    query_terms = _lexical_query_terms(query)
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
    """Split text into words, plus characters and adjacent-character bigrams for unspaced runs.

    `\\w+` matches an entire CJK run as one token. That token is by construction the rarest term
    in any corpus, so it took the largest IDF weight into the coverage ratio and could never
    match anything, which put `_LEXICAL_FULL_COVERAGE` — the one term that actually fuses the
    dense and full-text routes — permanently out of reach for every multi-character Chinese
    query. Runs are therefore removed before word splitting rather than left alongside their
    parts, and re-emitted as characters and bigrams: the dependency-free stand-in for a
    segmenter, and the reason a query needs an adjacent pair to match rather than one character
    that happens to appear somewhere.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    terms = set(_LEXICAL_TERM.findall(_UNSEGMENTED_RUN.sub(" ", normalized)))
    for run in _UNSEGMENTED_RUN.findall(normalized):
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    if _NEGATED_CONTRACTION.search(normalized) is not None:
        terms.add("not")
    return frozenset(terms)


def _lexical_query_terms(value: str) -> frozenset[str]:
    """Query terms with the scaffolding dropped, so coverage measures the distinctive ones.

    A bigram is scaffolding only when both of its characters are, so `什么` goes and `的饮`
    stays; otherwise a question's function words would keep full coverage unreachable through
    the bigrams they appear in.
    """
    return frozenset(
        term
        for term in _lexical_terms(value)
        if term not in _LEXICAL_NOISE_TERMS
        and not all(character in _CJK_NOISE_CHARACTERS for character in term)
    )


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


def _identity_relationship(value: object) -> str:
    relationship = _text(value, "identity relationship")
    if len(relationship) > 255 or not relationship.isprintable():
        raise ValidationError("identity relationship must be at most 255 printable characters")
    return relationship


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
        raise ModelError(
            "embedding model returned a non-finite or zero vector", reason="response_invalid"
        )
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
        # `io_failed` is this wrapper's pre-existing contract and a test pins it. It is coarse by
        # design -- the wrapper catches whatever any storage action raised -- and deliberately not
        # in `RETRYABLE_REASONS`, so a caller does not retry a failure it cannot know is transient.
        raise StorageError(f"failed to {action}", reason="io_failed") from error


@contextmanager
def _translate_index_errors(action: str) -> Iterator[None]:
    try:
        yield
    except MindBridgeError:
        raise
    except Exception as error:
        raise IndexUnavailableError(f"failed to {action}") from error
