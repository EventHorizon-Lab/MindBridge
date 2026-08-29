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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Condition, RLock
from typing import Protocol, cast

from mindbridge.exceptions import (
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
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    UnsupportedSchemaError,
)
from mindbridge.infrastructure.local.zvec_index import IndexHit, ZvecIndex
from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    GenerationBackend,
    ModelInput,
    SpeechAnalysis,
    SpeechBackend,
    TranscriptionBackend,
    _modalities,
)
from mindbridge.types import (
    AnswerResult,
    AssetRef,
    Blob,
    ContentAtom,
    ContentInput,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    SearchHit,
    SpeakerSegment,
)

_DOCUMENT_TASK = EmbedTask.DOCUMENT.value
_INDEX_RECIPE = (
    "zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:type-time-filters:single-vector-v3"
)
_LEGACY_INDEX_RECIPES = frozenset(
    {"zvec-0.7:hnsw-cosine-m50-efc500:fts-standard-lowercase:single-vector-v2"}
)
_OUTBOX_BATCH_SIZE = 256
_REINDEX_PAGE_SIZE = 256
_RERANK_CANDIDATES = 50
_RANK_FLOOR = 0.3
_RANK_CEILING = 1.5
_DECAY_REINFORCEMENT_LIMIT = 20
_NO_TRANSCRIPTION_SPACE = "none:asr-v1"
_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_MAX_CONTENT_PARTS = 128
_MAX_TEXT_CHARACTERS = 65_536
_MAX_METADATA_BYTES = 262_144
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
_STORE_METADATA_KEYS = {
    "model": "embedding.model_id",
    "space": "embedding.space_id",
    "transcription": "transcription.space_id",
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
    ) -> tuple[IndexHit, ...]: ...

    def flush(self) -> None: ...

    def optimize(self, *, concurrency: int = 0) -> None: ...

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


class Memory:
    """Persist and retrieve native text, image, video, audio, and omni memories."""

    def __init__(
        self,
        data_dir: str | Path = ".mindbridge",
        *,
        embedder: EmbeddingBackend,
        answerer: GenerationBackend | None = None,
        transcriber: SpeechBackend | TranscriptionBackend | None = None,
        decay_half_life_days: float | None = None,
        speaker_similarity: float = 0.78,
        speaker_margin: float = 0.05,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._owner_pid = os.getpid()
        self._write_lock = RLock()
        self._lifecycle = Condition()
        self._active_operations = 0
        self._closing = False
        self._closed = True
        self._pending_asset_cleanup: dict[str, StoredAsset] = {}
        self._speaker_similarity = _unit_interval(speaker_similarity, "speaker_similarity")
        self._speaker_margin = _unit_interval(speaker_margin, "speaker_margin")
        self._decay_half_life = _decay_half_life(decay_half_life_days)

        self._store = _open_store(self.data_dir)
        self._embedder = embedder
        self._answerer = answerer
        self._transcriber = transcriber

        try:
            (
                self._embedding_capabilities,
                self._embedding_model,
                self._space_id,
                self._embedding_dimension,
            ) = _embedding_contract(self._embedder)
            self._generation_capabilities = _generation_contract(self._answerer)
            (
                self._transcription_capabilities,
                self._transcription_space,
            ) = _transcription_contract(self._transcriber)
            self._assets = AssetStore(self.data_dir)
            self._collect_orphan_assets(scan_physical=True)
            index_path = self.data_dir / "zvec"
            index_missing = not index_path.exists()
            index_rebuild = self._ensure_store_metadata(index_path)
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
            )
        except Exception as error:
            self._close_models_and_store()
            raise IndexUnavailableError("failed to open the local search index") from error

        self._closed = False
        try:
            with self._write_lock:
                self._drain_outbox()
        except BaseException:
            self._closed = True
            self._close_resources()
            raise

    def __enter__(self) -> Memory:
        self._require_open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        """Add one native or mixed-modal memory and return its stable record."""
        with self._operation() as assets:
            prepared = _prepare_memory(
                self._prepare_content(content, assets),
                occurred_at=occurred_at,
                metadata=metadata,
                memory_type=memory_type,
            )
            return self._add_prepared((prepared,), operation=assets)[0]

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]:
        """Add memories in one model batch and one SQLite transaction."""
        with self._operation() as assets:
            normalized_memory_type = _memory_type(memory_type)
            if isinstance(contents, (str, bytes, Path, Blob, AssetRef)):
                raise ValidationError("contents must be a sequence of memory inputs")
            prepared = tuple(
                _prepare_memory(
                    self._prepare_content(content, assets),
                    occurred_at=None,
                    metadata=None,
                    memory_type=normalized_memory_type,
                )
                for content in contents
            )
            if not prepared:
                return ()
            return self._add_prepared(prepared, operation=assets)

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> tuple[SearchHit, ...]:
        """Return ranked memories for a native or mixed-modal query."""
        with self._operation() as assets:
            _limit(limit, maximum=100)
            prepared = self._prepare_content(query, assets)
            reference = _reference_at(reference_at) or datetime.now(timezone.utc)
            hits = self._search_prepared(
                prepared,
                limit=limit,
                operation=assets,
                memory_type=_optional_memory_type(memory_type),
                reference_at=reference,
                temporal_range=_temporal_range(prepared.text, reference),
            )
            self._persist_transcripts(assets)
            return hits

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> AnswerResult:
        """Answer a native or mixed-modal question only from retrieved memories."""
        with self._operation() as assets:
            _limit(limit, maximum=100)
            if self._answerer is None:
                raise ModelError("answer backend is not configured")
            prepared = self._prepare_content(question, assets)
            reference = _reference_at(reference_at) or datetime.now(timezone.utc)
            temporal_range = _temporal_range(prepared.text, reference)
            search = partial(
                self._search_prepared,
                prepared,
                limit=limit,
                operation=assets,
                memory_type=_optional_memory_type(memory_type),
                reference_at=reference,
                temporal_range=temporal_range,
            )
            speech_assets = self._answer_speech_assets(prepared.assets)
            if speech_assets and all(
                Modality(asset.modality) in self._embedding_capabilities for asset in speech_assets
            ):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    identities = executor.submit(
                        self._recognize_speech,
                        speech_assets,
                        assets,
                    )
                    hits = search()
                    identities.result()
            else:
                if speech_assets:
                    self._recognize_speech(speech_assets, assets)
                hits = search()
            routed_question = self._route_generation(
                (
                    _with_reference_time(prepared, reference)
                    if reference_at is not None or temporal_range is not None
                    else prepared
                ),
                assets,
            )
            routed_hits = self._route_generation_hits(hits, assets) if hits else ()
            self._persist_transcripts(assets)
            result = self._answer(routed_question, routed_hits)
            return AnswerResult(answer=result.answer, hits=hits)

    def get(self, memory_id: str) -> MemoryRecord:
        """Return one memory or raise `MemoryNotFoundError`."""
        with self._operation() as assets, self._write_lock:
            normalized_id = _identifier(memory_id, "memory_id")
            with _translate_storage_errors("read memory"):
                memory = self._store.read_memory(normalized_id)
            if memory is None:
                raise MemoryNotFoundError(f"memory does not exist: {normalized_id}")
            self._lease_assets(memory.assets, assets.leased)
            return self._memory_record(memory)

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        """Transcribe speech and resolve stable local speaker identities."""
        with self._operation() as operation:
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

    def register_speaker(self, speaker_id: str, name: str) -> None:
        """Assign or replace a human-readable name for one recognized speaker."""
        normalized_id = _identifier(speaker_id, "speaker_id")
        normalized_name = _speaker_name(name)
        with self._operation(), self._write_lock:
            with _translate_storage_errors("register speaker"):
                registered = self._store.register_speaker(normalized_id, normalized_name)
            if not registered:
                raise SpeakerNotFoundError(f"speaker does not exist: {normalized_id}")

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        """List newest memories with an opaque stable keyset cursor."""
        with self._operation() as assets, self._write_lock:
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
        with self._operation(), self._write_lock:
            normalized_id = _identifier(memory_id, "memory_id")
            with _translate_storage_errors("delete memory"):
                deleted, orphaned = self._store.delete_memory_with_assets(normalized_id)
            self._queue_asset_cleanup(orphaned)
            self._drain_outbox()
            return deleted

    def reindex(self) -> int:
        """Rebuild the disposable Zvec collection from authoritative SQLite rows."""
        with self._operation(), self._write_lock:
            self._drain_outbox()
            with _translate_storage_errors("checkpoint a search-index rebuild"):
                self._store.queue_all_embeddings()
            with _translate_index_errors("rebuild the search index"):
                count = self._index.rebuild(self._index_documents(), batch_size=_REINDEX_PAGE_SIZE)
            # Adds may commit SQLite while the rebuild owns the Zvec boundary. Replay instead of
            # blindly acknowledging so records committed after its SQLite scan cannot be lost.
            self._drain_outbox()
            return count

    def optimize(self) -> None:
        """Merge staged Zvec vectors into the configured index."""
        with self._operation(), self._write_lock:
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
        raise StorageError("failed to close local memory resources") from first

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
        with _translate_storage_errors("check existing memories"):
            existing_rows = self._store.read_memories(ordered_ids)
        existing_ids = {row.memory_id for row in existing_rows}
        missing = [unique[memory_id] for memory_id in ordered_ids if memory_id not in existing_ids]
        stored_memories: tuple[StoredMemory, ...] = ()
        stored_embeddings: tuple[StoredEmbedding, ...] = ()
        if missing:
            if Modality.AUDIO not in self._embedding_capabilities:
                for memory in missing:
                    _fallback_unsupported(
                        memory.content,
                        self._embedding_capabilities,
                        "embedding",
                    )
                self._cache_audio_transcripts(
                    tuple(asset for memory in missing for asset in memory.content.assets),
                    operation,
                )
            missing = [self._prepare_for_embedding(memory, operation) for memory in missing]
            inputs = tuple(self._route_embedding(memory.content) for memory in missing)
            vectors = self._embed(inputs, task=EmbedTask.DOCUMENT)
            now = datetime.now(timezone.utc)
            # ponytail: one aggregate vector per memory; add segmentation only when retrieval
            # benchmarks justify its storage and write-amplification cost.
            stored_memories = tuple(
                StoredMemory(
                    memory_id=memory.memory_id,
                    content=memory.content.text,
                    modality=memory.content.modality.value,
                    memory_type=memory.memory_type.value,
                    assets=memory.content.assets,
                    metadata_json=memory.metadata_json,
                    occurred_at=memory.occurred_at,
                    created_at=now,
                    updated_at=now,
                )
                for memory in missing
            )
            stored_embeddings = tuple(
                StoredEmbedding(
                    embedding_id=memory.memory_id,
                    memory_id=memory.memory_id,
                    values=vector,
                    model_id=self._embedding_model,
                    space_id=self._space_id,
                    task=_DOCUMENT_TASK,
                    created_at=now,
                    normalized=True,
                )
                for memory, vector in zip(missing, vectors, strict=True)
            )
        if stored_memories:
            # SQLite is authoritative and uses one WAL connection per transaction. Commit before
            # taking the index lock so concurrent writers can accumulate one durable outbox batch
            # and share the expensive Zvec flush.
            with _translate_storage_errors("write memories"):
                self._store.write_memories(stored_memories, stored_embeddings)
        with self._write_lock:
            self._drain_outbox()
            with _translate_storage_errors("hydrate written memories"):
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
        return replace(
            memory,
            content=self._embedding_content(memory.content, operation),
        )

    def _search_prepared(
        self,
        prepared: _PreparedContent,
        *,
        limit: int,
        operation: _OperationAssets,
        memory_type: MemoryType | None,
        reference_at: datetime,
        temporal_range: tuple[datetime, datetime] | None,
    ) -> tuple[SearchHit, ...]:
        prepared = self._embedding_content(prepared, operation)
        model_input = self._route_embedding(prepared)
        vector = self._embed((model_input,), task=EmbedTask.QUERY)[0]
        rerank = temporal_range is not None or self._decay_half_life is not None
        candidate_limit = max(_RERANK_CANDIDATES, limit * 3) if rerank else limit
        with self._write_lock:
            self._drain_outbox()
            with _translate_index_errors("search memories"):
                preferred = self._index_candidates(
                    model_input,
                    vector,
                    limit=candidate_limit,
                    memory_type=memory_type,
                    occurred_from=None if temporal_range is None else temporal_range[0],
                    occurred_until=None if temporal_range is None else temporal_range[1],
                )
                index_hits = preferred
                if temporal_range is not None and len(preferred) < candidate_limit:
                    index_hits = _merge_index_hits(
                        preferred,
                        self._index_candidates(
                            model_input,
                            vector,
                            limit=candidate_limit,
                            memory_type=memory_type,
                        ),
                    )
            if not index_hits:
                return ()
            with _translate_storage_errors("hydrate search results"):
                memories = self._store.read_memories(tuple(hit.id for hit in index_hits))
            if memory_type is not None:
                memories = tuple(
                    memory for memory in memories if memory.memory_type == memory_type.value
                )
            relevance = {hit.id: hit.relevance for hit in index_hits}
            ranked = [
                (
                    memory,
                    _ranked_relevance(
                        memory,
                        relevance[memory.memory_id],
                        reference_at=reference_at,
                        temporal_range=temporal_range,
                        decay_half_life=self._decay_half_life,
                    ),
                    _in_time_range(memory.occurred_at, temporal_range),
                )
                for memory in memories
            ]
            ranked.sort(key=lambda item: (item[2], item[1]), reverse=True)
            visible = ranked[:limit]
            self._lease_assets(
                tuple(asset for memory, _score, _matched in visible for asset in memory.assets),
                operation.leased,
            )
            operation.persisted.update(
                asset.asset_id for memory, _score, _matched in visible for asset in memory.assets
            )
            if self._decay_half_life is not None and visible:
                with _translate_storage_errors("reinforce retrieved memories"):
                    self._store.reinforce_memories(
                        tuple(memory.memory_id for memory, _score, _matched in visible),
                        accessed_at=datetime.now(timezone.utc),
                    )
        return tuple(self._search_hit(memory, score) for memory, score, _matched in visible)

    def _index_candidates(
        self,
        model_input: ModelInput,
        vector: Sequence[float],
        *,
        limit: int,
        memory_type: MemoryType | None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
    ) -> tuple[IndexHit, ...]:
        memory_type_value = None if memory_type is None else memory_type.value
        if model_input.text:
            return self._index.hybrid_search(
                model_input.text,
                vector,
                limit=limit,
                space_id=self._space_id,
                task=_DOCUMENT_TASK,
                memory_type=memory_type_value,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
            )
        return self._index.search(
            vector,
            limit=limit,
            space_id=self._space_id,
            task=_DOCUMENT_TASK,
            memory_type=memory_type_value,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
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
        raise ModelError(f"configured embedding model does not support: {names}")

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
        return (
            self._with_audio_transcripts(prepared, operation)
            if Modality.AUDIO in unsupported
            else prepared
        )

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
        raise ModelError(f"configured generation model does not support: {names}")

    def _route_generation_hits(
        self,
        hits: Sequence[SearchHit],
        operation: _OperationAssets,
    ) -> tuple[SearchHit, ...]:
        asset_ids = tuple(asset.id for hit in hits for asset in hit.assets)
        with _translate_storage_errors("hydrate media for answer generation"):
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
        routed = []
        for hit, prepared in zip(hits, prepared_hits, strict=True):
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

    def _answer_speech_assets(
        self,
        assets: Sequence[StoredAsset],
    ) -> tuple[StoredAsset, ...]:
        if not isinstance(self._transcriber, SpeechBackend):
            return ()
        supported = {modality.value for modality in self._transcription_capabilities}
        return tuple(
            {
                asset.asset_id: asset
                for asset in assets
                if asset.modality in {"audio", "video"} and asset.modality in supported
            }.values()
        )

    def _with_speech_identities(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        speech_assets = self._answer_speech_assets(prepared.assets)
        self._recognize_speech(speech_assets, operation)
        text = _speech_identity_text(prepared.text, speech_assets, operation.speech_segments)
        if len(text) > _MAX_TEXT_CHARACTERS:
            raise ModelError("speaker identity evidence exceeded the supported text length")
        return replace(prepared, text=text)

    def _recognize_speech(
        self,
        assets: Sequence[StoredAsset],
        operation: _OperationAssets,
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
        with _translate_storage_errors("read cached speaker recognition"):
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
            with self._write_lock, _translate_storage_errors("persist speaker recognition"):
                for asset, analysis in zip(missing, analyses, strict=True):
                    self._store.write_asset(asset)
                    operation.persisted.add(asset.asset_id)
                    operation.speech_segments[asset.asset_id] = self._store.write_speech(
                        asset.asset_id,
                        analysis,
                        model_id=self._transcriber.transcription_model,
                        space_id=self._transcription_space,
                        minimum_similarity=self._speaker_similarity,
                        minimum_margin=self._speaker_margin,
                    )
        for asset in speech_assets:
            segments = operation.speech_segments[asset.asset_id]
            operation.transcripts[asset.asset_id] = "\n".join(segment.text for segment in segments)

    def _with_audio_transcripts(
        self,
        prepared: _PreparedContent,
        operation: _OperationAssets,
    ) -> _PreparedContent:
        audio = tuple(asset for asset in prepared.assets if asset.modality == "audio")
        self._cache_audio_transcripts(audio, operation)
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
        audio = tuple(asset for asset in assets if asset.modality == "audio")
        for asset in audio:
            if asset.transcript is not None:
                operation.transcripts.setdefault(asset.asset_id, asset.transcript)
        missing = tuple(
            dict.fromkeys(
                asset.asset_id for asset in audio if asset.asset_id not in operation.transcripts
            )
        )
        if missing:
            if Modality.AUDIO not in self._transcription_capabilities:
                raise ModelError(
                    "audio fallback requires a transcription model with audio capability"
                )
            by_id = {asset.asset_id: asset for asset in audio}
            refs = tuple(self._asset_ref(by_id[asset_id]) for asset_id in missing)
            try:
                transcribe = getattr(self._transcriber, "transcribe", None)
                if callable(transcribe):
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
            with self._write_lock, _translate_storage_errors("cache audio transcripts"):
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

    def _embed(
        self,
        inputs: Sequence[ModelInput],
        *,
        task: EmbedTask,
    ) -> tuple[tuple[float, ...], ...]:
        try:
            vectors = self._embedder.embed(inputs, task=task)
        except MindBridgeError:
            raise
        except Exception as error:
            raise ModelError("failed to embed memory input") from error
        if len(vectors) != len(inputs):
            raise ModelError("embedding model returned the wrong number of vectors")
        return tuple(_normalized_vector(vector, self._embedding_dimension) for vector in vectors)

    def _answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> AnswerResult:
        if self._answerer is None:
            raise ModelError("answer backend is not configured")
        try:
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
            by_id = {document.embedding.embedding_id: document for document in hydrated}
            documents = [
                by_id[operation.embedding_id]
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
            with _translate_storage_errors("hydrate memories for reindexing"):
                yield from self._store.read_index_documents(
                    tuple(memory.memory_id for memory in memories)
                )
            last = memories[-1]
            after = (last.created_at, last.memory_id)

    def _ensure_store_metadata(self, index_path: Path) -> bool:
        expected = {
            _STORE_METADATA_KEYS["model"]: self._embedding_model,
            _STORE_METADATA_KEYS["space"]: self._space_id,
            _STORE_METADATA_KEYS["transcription"]: self._transcription_space,
            _STORE_METADATA_KEYS["dimension"]: str(self._embedding_dimension),
            _STORE_METADATA_KEYS["index"]: _INDEX_RECIPE,
        }
        rebuild_index = False
        with _translate_storage_errors("validate local store metadata"):
            for key, value in expected.items():
                stored = self._store.get_metadata(key)
                if stored is None:
                    if key == _STORE_METADATA_KEYS["index"] and index_path.exists():
                        rebuild_index = True
                    else:
                        self._store.set_metadata(key, value)
                elif stored == value:
                    continue
                elif key == _STORE_METADATA_KEYS["index"] and stored in _LEGACY_INDEX_RECIPES:
                    rebuild_index = True
                else:
                    raise StorageError(
                        f"local store metadata mismatch for {key}: expected {value!r}, "
                        f"found {stored!r}"
                    )
            if rebuild_index:
                if index_path.exists():
                    shutil.rmtree(index_path)
                self._store.set_metadata(_STORE_METADATA_KEYS["index"], _INDEX_RECIPE)
        return rebuild_index

    def _close_models_and_store(self) -> None:
        resources = _present_resources(
            self._embedder,
            self._transcriber,
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
        decay_half_life_days: float | None = None,
        speaker_similarity: float = 0.78,
        speaker_margin: float = 0.05,
    ) -> None:
        self._memory = Memory(
            data_dir=data_dir,
            embedder=embedder,
            answerer=answerer,
            transcriber=transcriber,
            decay_half_life_days=decay_half_life_days,
            speaker_similarity=speaker_similarity,
            speaker_margin=speaker_margin,
        )

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()

    async def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        return await asyncio.to_thread(
            self._memory.add,
            content,
            occurred_at=occurred_at,
            metadata=metadata,
            memory_type=memory_type,
        )

    async def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]:
        return await asyncio.to_thread(
            self._memory.add_many,
            contents,
            memory_type=memory_type,
        )

    async def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> tuple[SearchHit, ...]:
        return await asyncio.to_thread(
            self._memory.search,
            query,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
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

    async def register_speaker(self, speaker_id: str, name: str) -> None:
        await asyncio.to_thread(self._memory.register_speaker, speaker_id, name)

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


def _prepare_memory(
    content: _PreparedContent,
    *,
    occurred_at: datetime | None,
    metadata: Mapping[str, object] | None,
    memory_type: MemoryType,
) -> _PreparedMemory:
    normalized_occurred_at = _occurred_at(occurred_at)
    normalized_memory_type = _memory_type(memory_type)
    metadata_json = _metadata_json(metadata)
    identity: dict[str, object] = {
        "parts": content.canonical_parts,
        "metadata": json.loads(metadata_json),
        "occurred_at": (
            None if normalized_occurred_at is None else _datetime_text(normalized_occurred_at)
        ),
    }
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
        raise ModelError(f"configured {operation} model does not support: {names}")
    return unsupported


def _derived_text(text: str, assets: Sequence[StoredAsset]) -> str:
    sections = [text] if text else []
    seen: set[str] = set()
    for asset in assets:
        transcript = asset.transcript
        if asset.modality != "audio" or not transcript or asset.asset_id in seen:
            continue
        seen.add(asset.asset_id)
        marker = f"[audio transcript:{asset.asset_id}]"
        if marker not in text:
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


def _open_store(data_dir: Path) -> LocalStore:
    try:
        return LocalStore(data_dir)
    except (DataDirectoryInUseError, UnsupportedSchemaError) as error:
        raise StorageError(str(error)) from error
    except Exception as error:
        raise StorageError("failed to open the local memory store") from error


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


def _with_reference_time(content: _PreparedContent, reference_at: datetime) -> _PreparedContent:
    note = f"Reference time for relative dates: {reference_at.isoformat(timespec='microseconds')}"
    return replace(content, text=f"{content.text}\n\n{note}" if content.text else note)


def _merge_index_hits(
    preferred: Sequence[IndexHit],
    fallback: Sequence[IndexHit],
) -> tuple[IndexHit, ...]:
    merged = {hit.id: hit for hit in preferred}
    for hit in fallback:
        merged.setdefault(hit.id, hit)
    return tuple(merged.values())


def _in_time_range(
    occurred_at: datetime | None,
    temporal_range: tuple[datetime, datetime] | None,
) -> bool:
    return (
        occurred_at is not None
        and temporal_range is not None
        and temporal_range[0] <= occurred_at < temporal_range[1]
    )


def _ranked_relevance(
    memory: StoredMemory,
    relevance: float,
    *,
    reference_at: datetime,
    temporal_range: tuple[datetime, datetime] | None,
    decay_half_life: timedelta | None,
) -> float:
    score = relevance
    if temporal_range is not None:
        score *= (
            _RANK_CEILING if _in_time_range(memory.occurred_at, temporal_range) else _RANK_FLOOR
        )
    if decay_half_life is not None:
        accessed_at = memory.last_accessed_at
        anchor = memory.occurred_at or memory.updated_at
        access_count = 0
        if accessed_at is not None and accessed_at <= reference_at:
            anchor = accessed_at
            access_count = memory.access_count
        age = max(0.0, (reference_at - anchor).total_seconds())
        strength = 1.0 + math.log2(1.0 + min(access_count, _DECAY_REINFORCEMENT_LIMIT))
        retention = 2.0 ** (-age / (decay_half_life.total_seconds() * strength))
        score *= _RANK_FLOOR + (_RANK_CEILING - _RANK_FLOOR) * retention
    return score


def _temporal_range(text: str, reference_at: datetime) -> tuple[datetime, datetime] | None:
    try:
        return _parse_temporal_range(text, reference_at)
    except (OverflowError, ValueError):
        raise ValidationError("temporal expression exceeds the supported date range") from None


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


def _decay_half_life(days: float | None) -> timedelta | None:
    if days is None:
        return None
    try:
        return timedelta(days=days)
    except OverflowError:
        raise ValidationError("decay_half_life_days is too large") from None


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValidationError(f"{name} must be non-empty and trimmed")
    return value


def _speaker_name(value: object) -> str:
    name = _text(value, "speaker name")
    if len(name) > 255 or not name.isprintable():
        raise ValidationError("speaker name must be at most 255 printable characters")
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
        raise StorageError(f"failed to {action}") from error


@contextmanager
def _translate_index_errors(action: str) -> Iterator[None]:
    try:
        yield
    except MindBridgeError:
        raise
    except Exception as error:
        raise IndexUnavailableError(f"failed to {action}") from error
